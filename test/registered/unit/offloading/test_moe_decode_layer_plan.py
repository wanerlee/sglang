"""CPU contracts for explicit layer preparation; no NPU replay certification."""

import random
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from sglang.srt.offloading.moe.config import (
    ExpertCatalog,
    ExpertLayerSpec,
    Phase,
    fixed_waves,
)
from sglang.srt.offloading.moe.execution import (
    ExecutionRequest,
    ExpertWeightAccess,
    LocalExecution,
)
from sglang.srt.offloading.moe.npu.transfer import MemFabricDeviceTransferOps
from sglang.srt.offloading.moe.programs import (
    DecodeLocalExpertProgram,
    DispatchOnceProgram,
)
from sglang.srt.offloading.moe.residency import BoundExecutionWave, TensorResidencyOps
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.transfer import Completion, CopySourceKind, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Fence(Completion):
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def wait_on(self, stream):
        self.calls.append(("wait", self.name, stream))

    def synchronize(self):
        raise AssertionError("No host synchronization in layer planning")

    def done(self):
        raise AssertionError("No completion polling in layer planning")


class Context:
    phase = Phase.DECODE
    capturing = True
    compute_stream = "compute"
    copy_stream = "copy"

    def __init__(self):
        self.invocation_id = object()
        self.calls, self.retained = [], []

    def retain(self, value):
        self.retained.append(value)

    retain_completion = retain

    def record_compute_completion(self):
        name = "compute-" + str(len(self.calls))
        self.calls.append(("record", name))
        return TransferTicket(Fence(self.calls, name))

    def copied(self):
        return TransferTicket(Fence(self.calls, "copy-done"))

    def wait_for(self, ticket, stream):
        ticket.completion.wait_on(stream)

    def temporary_scope(self):
        return nullcontext()


def fixture(experts=7, width=3, hot=3):
    catalog = ExpertCatalog((ExpertLayerSpec(3, experts, 1, 0),))
    state = ResidencyState.allocate(catalog.num_experts, hot, width, "cpu")
    error = torch.zeros((), dtype=torch.int32)
    binding = SimpleNamespace(error_status=error, hot_generations=state.slot_generation)
    waves = fixed_waves(experts, width)
    ops = TensorResidencyOps(
        state,
        catalog,
        execution_waves=[BoundExecutionWave(3, wave, 0, binding) for wave in waves],
        update_bindings=binding,
        error_status=error,
    )
    return ops, state, waves, Context()


def load_hot(state, keys):
    state.key_to_slot.fill_(-1)
    state.slot_to_key.copy_(torch.tensor(keys, dtype=torch.int64))
    state.slot_valid.copy_(state.slot_to_key >= 0)
    for slot, key in enumerate(keys):
        if key >= 0:
            state.key_to_slot[key] = slot
    state.slot_generation.add_(1)


def retire(ops, reservation, ctx):
    ops.publish_execution(reservation, ctx.copied(), ctx)
    ops.finish_execution(reservation, ctx.copied(), ctx)


class TensorTrace(TorchDispatchMode):
    """Replay recorded CPU tensor operations without rerunning planner Python.

    This checks dynamic data dependencies, not an NPU graph/event implementation.
    """

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        self.calls.append((func, args, kwargs, output))
        return output

    def replay(self):
        values = {}

        def remap(value):
            return (
                values.get(id(value), value)
                if isinstance(value, torch.Tensor)
                else value
            )

        for func, args, kwargs, original in self.calls:
            result = func(*tree_map(remap, args), **tree_map(remap, kwargs))

            def save(old, new):
                if isinstance(old, torch.Tensor):
                    values[id(old)] = new

            tree_map(save, original, result)
        return values


class TestDecodeLayerPlan(unittest.TestCase):
    def test_tensor_trace_replay_reads_new_demand_and_hot_maps(self):
        ops, state, waves, ctx = fixture()
        demand = torch.ones(7, dtype=torch.int64)
        load_hot(state, [0, 2, 4])
        snapshots = []
        trace = TensorTrace()
        with trace:
            ops.begin_step(ctx)
            prepared = ops.prepare_decode_layer(3, demand, ctx)
            for wave in waves:
                reservation = ops.resolve_execution(
                    ExecutionRequest(3, demand, wave, Phase.DECODE, prepared), ctx
                )
                snapshots.append(
                    tuple(
                        getattr(reservation.copy_plan, name).clone()
                        for name in (
                            "source_kind",
                            "source_index",
                            "expected_generation",
                            "valid",
                        )
                    )
                )
                retire(ops, reservation, ctx)
            ops.finish_decode_layer(prepared, ctx)
        for counts, keys in (
            ([0, 2, 0, 3, 0, 1, 0], [1, 3, 5]),
            ([0] * 7, [0, 1, 6]),
            ([1] * 7, [-1, -1, -1]),
        ):
            demand.copy_(torch.tensor(counts))
            load_hot(state, keys)
            with patch.object(
                ops,
                "prepare_decode_layer",
                side_effect=AssertionError("planner Python reran"),
            ):
                result = trace.replay()
            self.assertEqual(state.step_access_count.tolist(), counts)
            for wave, saved in zip(waves, snapshots):
                kind, index, generation, valid = [result[id(value)] for value in saved]
                for lane, key in enumerate(range(wave.local_start, wave.local_end)):
                    self.assertEqual(valid[lane].item(), bool(counts[key]))
                    hit = key in keys
                    self.assertEqual(
                        index[lane].item(), keys.index(key) if hit else key
                    )
                    self.assertEqual(
                        generation[lane].item(),
                        state.slot_generation[keys.index(key)].item() if hit else -1,
                    )
                    self.assertEqual(
                        kind[lane].item(),
                        int(
                            (CopySourceKind.HOT_SLOT if hit else CopySourceKind.HOST)
                            if counts[key]
                            else CopySourceKind.NONE
                        ),
                    )

    def test_prepared_matches_standalone_randomized_and_looks_up_once(self):
        rng = random.Random(19)
        for hot in (0, 3):
            for _ in range(8):
                ops, state, waves, ctx = fixture(hot=hot)
                ref, ref_state, _, ref_ctx = fixture(hot=hot)
                keys = rng.sample(range(7), hot)
                load_hot(state, keys)
                load_hot(ref_state, keys)
                demand = torch.tensor([rng.randrange(5) for _ in range(7)])
                stats = Mock()
                ops.cache_stats = stats
                with patch.object(
                    torch, "index_select", wraps=torch.index_select
                ) as lookup:
                    prepared = ops.prepare_decode_layer(3, demand, ctx)
                    for wave in waves:
                        before = len(ctx.calls)
                        actual = ops.resolve_execution(
                            ExecutionRequest(3, demand, wave, Phase.DECODE, prepared),
                            ctx,
                        )
                        # New wave copy frontier still includes its prior reader.
                        if wave.index:
                            self.assertEqual(
                                ctx.calls[before], ("wait", "copy-done", "compute")
                            )
                        self.assertEqual(ctx.calls[-1][0], "record")
                        with patch.object(
                            torch, "index_select", new=lookup._mock_wraps
                        ):
                            expected = ref.resolve_execution(
                                ExecutionRequest(3, demand, wave, Phase.DECODE), ref_ctx
                            )
                        for name in (
                            "source_kind",
                            "source_index",
                            "destination_slot",
                            "valid",
                            "expected_generation",
                            "count",
                        ):
                            self.assertTrue(
                                torch.equal(
                                    getattr(actual.copy_plan, name),
                                    getattr(expected.copy_plan, name),
                                ),
                                name,
                            )
                        retire(ops, actual, ctx)
                        retire(ref, expected, ref_ctx)
                    self.assertEqual(lookup.call_count, 4 if hot else 1)
                ops.finish_decode_layer(prepared, ctx)
                self.assertEqual(stats.record_counts.call_count, 1)
                self.assertTrue(
                    torch.equal(state.step_access_count, ref_state.step_access_count)
                )

    def test_repeated_layer_same_context_refreshes_counts_and_maps(self):
        ops, state, waves, ctx = fixture()
        pointers = [(a.storage_key, a.nbytes) for a in ops.allocations()]
        demand = torch.zeros(7, dtype=torch.int64)
        expected = torch.zeros_like(demand)
        addresses = []
        old = None
        for keys, counts in (
            ([0, 2, 6], [1, 0, 2, 0, 0, 0, 3]),
            ([1, 3, 5], [0, 4, 0, 2, 0, 1, 0]),
        ):
            load_hot(state, keys)
            demand.copy_(torch.tensor(counts))
            prepared = ops.prepare_decode_layer(3, demand, ctx)
            if old is not None:
                with self.assertRaisesRegex(ValueError, "stale"):
                    ops.resolve_execution(
                        ExecutionRequest(3, demand, waves[0], Phase.DECODE, old), ctx
                    )
            this_addresses = []
            for wave in waves:
                reservation = ops.resolve_execution(
                    ExecutionRequest(3, demand, wave, Phase.DECODE, prepared), ctx
                )
                plan = reservation.copy_plan
                this_addresses.append(
                    (plan.source_index.data_ptr(), plan.expected_generation.data_ptr())
                )
                for lane, key in enumerate(range(wave.local_start, wave.local_end)):
                    if counts[key]:
                        self.assertEqual(
                            plan.source_kind[lane].item(),
                            int(
                                CopySourceKind.HOT_SLOT
                                if key in keys
                                else CopySourceKind.HOST
                            ),
                        )
                retire(ops, reservation, ctx)
            addresses.append(this_addresses)
            ops.finish_decode_layer(prepared, ctx)
            expected.add_(demand)
            old = prepared
        self.assertEqual(addresses[0], addresses[1])
        self.assertTrue(torch.equal(state.step_access_count, expected))
        self.assertEqual(
            pointers, [(a.storage_key, a.nbytes) for a in ops.allocations()]
        )
        # Persistent views share owned layer storage; the ledger reports owners only.
        source = ops._layer_sources[3]["plan"].source_index
        for wave in waves:
            self.assertEqual(
                ops._waves[(3, wave)]["prepared_plan"]
                .source_index.untyped_storage()
                .data_ptr(),
                source.untyped_storage().data_ptr(),
            )

    def test_late_error_gates_next_wave_and_publication(self):
        ops, state, waves, ctx = fixture()
        demand = torch.ones(7, dtype=torch.int64)
        prepared = ops.prepare_decode_layer(3, demand, ctx)
        first = ops.resolve_execution(
            ExecutionRequest(3, demand, waves[0], Phase.DECODE, prepared), ctx
        )
        retire(ops, first, ctx)
        ops.error_status.fill_(128)
        second = ops.resolve_execution(
            ExecutionRequest(3, demand, waves[1], Phase.DECODE, prepared), ctx
        )
        self.assertFalse(second.copy_plan.valid.any())
        self.assertEqual(second.copy_plan.count.item(), 0)
        self.assertTrue(
            (second.copy_plan.source_kind == int(CopySourceKind.NONE)).all()
        )
        retire(ops, second, ctx)
        self.assertFalse(state.execution_valid.any())
        ops.finish_decode_layer(prepared, ctx)

    def test_generation_snapshot_is_still_checked_by_transfer(self):
        ops, state, waves, ctx = fixture()
        load_hot(state, [0, 1, 2])
        demand = torch.ones(7, dtype=torch.int64)
        prepared = ops.prepare_decode_layer(3, demand, ctx)
        state.slot_generation.add_(1)
        reservation = ops.resolve_execution(
            ExecutionRequest(3, demand, waves[0], Phase.DECODE, prepared), ctx
        )
        binding = SimpleNamespace(
            host_count=7,
            hot_count=3,
            destination_count=3,
            hot_generations=state.slot_generation,
            hot_targets_alias=False,
            destination_writes=torch.zeros(4, dtype=torch.int64),
            error_status=ops.error_status,
            mask=torch.zeros(3, dtype=torch.bool),
        )
        MemFabricDeviceTransferOps._mask(reservation.copy_plan, binding)
        self.assertNotEqual(ops.error_status.item(), 0)
        self.assertFalse(binding.mask.any())
        retire(ops, reservation, ctx)
        self.assertFalse(state.execution_valid.any())
        ops.finish_decode_layer(prepared, ctx)

    def test_explicit_scope_rejects_stale_foreign_and_duplicate_requests(self):
        ops, state, waves, ctx = fixture()
        demand = torch.ones(7, dtype=torch.int64)
        prepared = ops.prepare_decode_layer(3, demand, ctx)
        with self.assertRaises(RuntimeError):
            ops.prepare_decode_layer(3, demand, ctx)
        with self.assertRaises(RuntimeError):
            ops.prepare_update(state.slot_to_key, ctx)
        with self.assertRaises(RuntimeError):
            ops.resolve_execution(
                ExecutionRequest(3, demand, waves[0], Phase.DECODE), ctx
            )
        for wrong_ctx, wrong_demand in ((Context(), demand), (ctx, demand.clone())):
            with self.assertRaises(ValueError):
                ops.resolve_execution(
                    ExecutionRequest(3, wrong_demand, waves[0], Phase.DECODE, prepared),
                    wrong_ctx,
                )
        first = ops.resolve_execution(
            ExecutionRequest(3, demand, waves[0], Phase.DECODE, prepared), ctx
        )
        retire(ops, first, ctx)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            ops.resolve_execution(
                ExecutionRequest(3, demand, waves[0], Phase.DECODE, prepared), ctx
            )
        ops.finish_decode_layer(prepared, ctx)
        with self.assertRaises(ValueError):
            ops.resolve_execution(
                ExecutionRequest(3, demand, waves[1], Phase.DECODE, prepared), ctx
            )

    def test_empty_zero_hot_and_invalid_demands(self):
        ops, state, waves, ctx = fixture(hot=0)
        for demand in (torch.ones(6, dtype=torch.int64), torch.ones(7)):
            with self.assertRaises(ValueError):
                ops.prepare_decode_layer(3, demand, ctx)
            self.assertIsNone(ops._active_layer_plan)
        demand = torch.zeros(7, dtype=torch.int32)
        prepared = ops.prepare_decode_layer(3, demand, ctx)
        for wave in waves:
            plan = ops.resolve_execution(
                ExecutionRequest(3, demand, wave, Phase.DECODE, prepared), ctx
            )
            self.assertEqual(plan.copy_plan.count.item(), 0)
            self.assertTrue((plan.copy_plan.expected_generation == -1).all())
            retire(ops, plan, ctx)
        ops.finish_decode_layer(prepared, ctx)
        self.assertFalse(state.step_access_count.any())

    def test_dispatch_once_prepares_after_bind_and_closes_even_on_failure(self):
        ops, state, waves, ctx = fixture()
        events = []
        in_compute_scope = []
        demand = torch.ones(7, dtype=torch.int64)

        @contextmanager
        def compute_scope(context):
            in_compute_scope.append(True)
            try:
                yield
            finally:
                in_compute_scope.pop()

        class Access(ExpertWeightAccess):
            def _materialize(self, reservation, context):
                if not in_compute_scope:
                    raise AssertionError("Wave preparation left the compute stream")
                events.append("copy")
                for dependency in reservation.copy_dependencies:
                    dependency.wait_on(context.copy_stream)
                return context.copied()

        class Layout:
            fail = False

            def bind(self, dispatched, context):
                events.append("bind")
                return SimpleNamespace(counts=demand)

            def select_wave(self, layout, wave, context):
                events.append("select")
                return SimpleNamespace(local_demand=layout.counts)

            def write_wave(self, layout, wave, output, context):
                events.append("write")
                if self.fail:
                    raise RuntimeError("write failed")

            def combine_input(self, layout, context):
                return layout

        layout = Layout()
        dispatcher = SimpleNamespace(
            dispatch=lambda *args: events.append("dispatch"),
            combine=lambda result, context: result,
            finish=lambda result, context: result,
        )

        def get_demand(layout):
            self.assertTrue(in_compute_scope)
            events.append("prepare")
            return layout.counts

        local = DecodeLocalExpertProgram(
            Access(ops),
            lambda layer, selected, wave, context: ExecutionRequest(
                layer, selected.local_demand, wave, Phase.DECODE
            ),
            lambda *args: LocalExecution(None, ctx.copied()),
            layer_demand=get_demand,
            compute_scope=compute_scope,
        )
        program = DispatchOnceProgram(waves, dispatcher, local, layout, local_experts=7)
        with patch.object(torch, "index_select", wraps=torch.index_select) as lookup:
            program.run(3, None, None, ctx)
            self.assertEqual(lookup.call_count, 4)
        self.assertEqual(events[:4], ["dispatch", "bind", "prepare", "select"])
        self.assertIsNone(ops._active_layer_plan)
        layout.fail = True
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            program.run(3, None, None, ctx)
        self.assertIsNone(ops._active_layer_plan)
        self.assertEqual(state.step_access_count.tolist(), [2] * 7)
        self.assertFalse(in_compute_scope)


if __name__ == "__main__":
    unittest.main()
