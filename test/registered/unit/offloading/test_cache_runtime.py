"""Cache caller integration using real CPU weights and a delayed test transport.

The test backend is deliberately confined to this file. It does not implement
NPU H2D, native EP, graph replay, or a serving fallback.
"""

import ast
import contextlib
import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch

from sglang.srt.offloading.moe import (
    CacheExecutionContext,
    CacheRuntime,
    CopySource,
    DecodeLocalExpertProgram,
    ExpertLayerSpec,
    LocalExecution,
    LocalRoutes,
    MoEOffloadConfig,
    MoEOffloaderV2,
    PeriodicLFUConfig,
    Phase,
)
from sglang.srt.offloading.moe.config import fixed_waves
from sglang.srt.offloading.moe.interfaces import (
    MoEBackend,
    RuntimeCapabilities,
    WaveDispatcher,
)
from sglang.srt.offloading.moe.owner import OwnerState
from sglang.srt.offloading.moe.programs import ComposedDFCProgram
from sglang.srt.offloading.moe.resources import Allocation, ResourceKind
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.transfer import (
    Completion,
    ImmediateCompletion,
    TransferTicket,
    WeightTransferBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDevice:
    def __init__(self):
        self.current = "compute"
        self.log = []
        self.sequence = 0

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def current_stream(self):
        return self.current

    def Event(self):
        self.sequence += 1
        device, number = self, self.sequence

        class Event:
            def record(self, stream):
                device.log.append(("record", number, stream))

            def wait(self, stream):
                device.log.append(("event_wait", number, stream))

            def query(self):
                return True

            def synchronize(self):
                device.log.append(("event_sync", number))

        return Event()

    def synchronize(self):
        self.log.append(("synchronize",))


class TestStream:
    def __init__(self, name):
        self.name = name

    def wait_event(self, event):
        event.wait(self.name)


class DeferredCopy(Completion):
    def __init__(self, callback, log):
        self.callback, self.log = callback, log
        self.ready = False

    def wait_on(self, stream):
        self.log.append(("weights_wait", stream.name))
        if not self.ready:
            self.callback()
            self.ready = True

    def synchronize(self):
        self.wait_on(TestStream("host"))

    def done(self):
        return self.ready


class DeferredTransport(WeightTransferBackend):
    """A complete bundle has two distinguishable values, both must be copied."""

    supports_device_plans = True  # Test-only implementation, not capture proof.

    def __init__(self, device):
        self.device = device
        self.calls = []
        self.fail_next = False

    def materialize_into(self, plan, bindings, stream):
        self.device.log.append(("submit", bindings.pool))
        old_map = bindings.state.key_to_slot.clone()
        entry = dict(pool=bindings.pool, entries=[], completed=False)
        self.calls.append(entry)

        def copy():
            # The caller may invalidate a victim but must not publish admission.
            torch.testing.assert_close(bindings.state.key_to_slot, old_map)
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("test copy failed before completion")
            self.assert_plan(plan)
            for i in range(plan.capacity):
                if not plan.valid[i]:
                    continue
                source = int(plan.source_index[i])
                destination = int(plan.destination_slot[i])
                kind = int(plan.source_kind[i])
                if bindings.pool == "hot":
                    assert not bindings.state.slot_valid[destination]
                if kind == CopySource.HOST:
                    bundle = bindings.host[source]
                else:
                    assert kind == CopySource.HOT_SLOT
                    assert bindings.state.slot_valid[source]
                    assert (
                        bindings.state.slot_generation[source]
                        == plan.expected_generation[i]
                    )
                    bundle = bindings.hot[source]
                bindings.destination[destination].copy_(bundle)
                entry["entries"].append((kind, source, destination))
            entry["completed"] = True
            self.device.log.append(("copied", bindings.pool))

        return TransferTicket(DeferredCopy(copy, self.device.log), (plan, bindings))

    @staticmethod
    def assert_plan(plan):
        assert int(plan.count) == int(plan.valid.sum())
        assert plan.valid.shape == (plan.capacity,)


class Dispatcher(WaveDispatcher):
    def mask_routes(self, routing, wave, context):
        return LocalRoutes(
            routing.expert_ids,
            routing.valid
            & (routing.expert_ids >= wave.local_start)
            & (routing.expert_ids < wave.local_end),
        )

    def dispatch(self, hidden_states, routing, context):
        return routing

    def combine(self, local_output, context):
        return local_output

    def accumulator(self, hidden_states, context):
        return torch.zeros_like(hidden_states)

    def accumulate(self, accumulator, partial, context):
        accumulator.add_(partial)

    def finish(self, accumulator, context):
        return accumulator


class Backend(MoEBackend):
    def prepare_layer(self, module, spec):
        pass

    def finalize(self, layers, catalog, config, policy):
        self.device = TestDevice()
        self.compute, self.copy = TestStream("compute"), TestStream("copy")
        self.device.current = self.compute
        self.transfer = DeferredTransport(self.device)
        state = ResidencyState.allocate(
            catalog.num_experts,
            config.hot_slots(catalog),
            config.execution_slots,
            "cpu",
        )
        self.host = (
            torch.arange(catalog.num_experts * 2, dtype=torch.float32).reshape(-1, 2)
            + 1
        )
        self.hot = torch.zeros(config.hot_slots(catalog), 2)
        self.execution = torch.zeros(config.execution_slots, 2)
        hot_bindings = SimpleNamespace(
            pool="hot", host=self.host, hot=self.hot, destination=self.hot, state=state
        )
        execution_bindings = SimpleNamespace(
            pool="execution",
            host=self.host,
            hot=self.hot,
            destination=self.execution,
            state=state,
        )
        resources = tuple(
            Allocation(
                kind.value,
                kind,
                tensor.numel() * tensor.element_size(),
                tensor.data_ptr(),
                tensor,
            )
            for kind, tensor in (
                (ResourceKind.HOST_WEIGHTS, self.host),
                (ResourceKind.HOT_WEIGHTS, self.hot),
                (ResourceKind.EXECUTION_WEIGHTS, self.execution),
            )
        )

        self.runtime = CacheRuntime(
            catalog=catalog,
            residency=state,
            policy=policy,
            transfer=self.transfer,
            context=CacheExecutionContext(self.device, self.compute, self.copy),
            hot_bindings=hot_bindings,
            execution_bindings=execution_bindings,
            bank_slots=2,
            capabilities=RuntimeCapabilities(True, True, True),
            expert_nbytes=8,
            resources=resources,
            model_binding=SimpleNamespace(
                install=lambda runtime: None, close=lambda: None
            ),
            borrow_hot_arena=lambda: self.hot.fill_(-999),
        )
        self.runtime.prefill_program = object()

        def native(layer, routes, bindings, context):
            # Compute from execution weights, never from the host oracle.
            active = routes.valid
            ids = routes.expert_ids[active]
            keys = ids + catalog.key(layer.layer_id, 0)
            lanes = (
                (keys[:, None] == bindings.state.execution_key)
                .to(torch.int64)
                .argmax(dim=1)
            )
            assert self.runtime.residency.execution_valid[lanes].all()
            result = self.execution[lanes].sum()
            self.device.log.append(("native", layer.layer_id))
            return LocalExecution(result, context.record_compute_completion())

        self.runtime.decode_program = ComposedDFCProgram(
            fixed_waves(4, 2),
            Dispatcher(),
            DecodeLocalExpertProgram(
                self.runtime.access,
                lambda layer, output, wave, context: (
                    self.runtime.make_execution_request(
                        layer.layer_id, output, wave, context
                    )
                ),
                native,
            ),
        )
        self.runtime.initialize("toy")
        return self.runtime

    def close_unbound(self):
        pass


class Mode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "DECODE"

    def is_idle(self):
        return self.name == "IDLE"

    def is_cpu_graph(self):
        return self.is_decode()

    def is_split_prefill(self):
        return False

    def is_extend(self, **kwargs):
        return self.name == "EXTEND"


class IdentityAdapter:
    def describe(self, layer):
        return layer


class TestCacheRuntime(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        profile = Path(self.temp.name) / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "model_id": "toy",
                    "layers": {
                        "3": {"4": 100, "6": 90},
                        "4": {"4": 80, "7": 70},
                    },
                }
            ),
            encoding="utf-8",
        )
        config = MoEOffloadConfig(0.5, 2, 2, 1, metadata_limit_bytes=100000)
        self.backend = Backend()
        self.offloader = MoEOffloaderV2(
            config,
            self.backend,
            adapter=IdentityAdapter(),
            periodic_lfu_config=PeriodicLFUConfig(str(profile)),
        )
        self.layers = tuple(ExpertLayerSpec(i, 8, 2, 1, i < 3) for i in range(5))
        self.offloader.wrap_modules(iter(self.layers))
        self.offloader.post_init()
        self.runtime = self.backend.runtime
        self.runner = SimpleNamespace(
            model=SimpleNamespace(), capture_head_hooks=[], capture_tail_hooks=[]
        )
        self.offloader.bind_model_runner(self.runner)
        # This fixture drives Programs directly; production model installation
        # is exercised separately in test_cache_model_binding.py.
        self.runner.model.decode = self.runtime.decode_program
        self.addCleanup(self.offloader.close)

    def batch(self, name="DECODE", global_mode=None, counts=None, extend=False):
        return SimpleNamespace(
            forward_mode=Mode(name),
            global_forward_mode=Mode(global_mode) if global_mode else None,
            original_global_num_tokens_cpu=counts,
            global_num_tokens_cpu=None,
            is_extend_in_batch=extend,
            input_ids=torch.empty(0, dtype=torch.int64)
            if name == "IDLE"
            else torch.tensor([1]),
        )

    def model(self, routes):
        for layer_id, ids in routes.items():
            ids = torch.tensor(ids, dtype=torch.int64)
            routing = LocalRoutes(ids, ids >= 0)
            result = self.runner.model.decode.run(
                self.layers[layer_id], torch.tensor(0.0), routing, self.runtime.context
            )
            offset = self.runtime.cache.catalog.key(layer_id, 0)
            oracle = self.backend.host[ids[ids >= 0] + offset].sum()
            torch.testing.assert_close(result, oracle)

    def run_step(self, routes, batch=None):
        batch = batch or self.batch()
        # Actual offloader facade -> concrete runtime -> D/F/C -> transfer.
        with self.offloader.forward_scope(batch):
            with self.offloader.eager_step_scope(batch):
                self.model(routes)

    def test_initial_load_waits_and_installs_complete_bundles(self):
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [0, 2, 4, 7])
        torch.testing.assert_close(self.backend.hot, self.backend.host[[0, 2, 4, 7]])
        self.assertEqual(self.runtime.cache.policy_state.usage.sum(), 0)
        self.assertTrue(self.backend.transfer.calls[0]["completed"])
        self.assertEqual(len(self.runner.capture_head_hooks), 1)

    def test_four_forwards_replace_then_next_forward_hits(self):
        l3 = ([1, 1, 0], [1, 2, 2], [1, 1, 3, 2], [1, 1, 1, 0, 2, 2, 3, 3])
        l4 = ([1, 1, 3], [1, 3, 2], [1, 1, 3], [1, 1, 0, 2, 3])
        for step in range(4):
            self.run_step({3: l3[step], 4: l4[step]})
            if step < 3:
                self.assertEqual(
                    self.runtime.residency.slot_to_key.tolist(), [0, 2, 4, 7]
                )
        state = self.runtime.residency
        self.assertEqual(state.slot_to_key.tolist(), [1, 2, 5, 7])
        self.assertEqual(state.key_to_slot.tolist(), [-1, 0, 1, -1, -1, 2, -1, 3])
        self.assertEqual(state.slot_generation.tolist(), [2, 1, 2, 1])
        torch.testing.assert_close(self.backend.hot, self.backend.host[[1, 2, 5, 7]])
        torch.testing.assert_close(
            self.runtime.cache.policy_state.usage,
            torch.tensor([2, 8, 5, 3, 1, 7, 2, 4]) * 0.7,
        )
        self.run_step({3: [1], 4: [1]})
        self.assertEqual(int(self.runtime.cache.stats.hits), 2)
        self.assertEqual(int(self.runtime.cache.stats.misses), 0)

    def test_batch_counting_deduplicates_loads_not_assignments(self):
        before = len(self.backend.transfer.calls)
        self.run_step({3: [[0, 1], [1, 2], [1, -1]]})
        self.assertEqual(
            self.runtime.residency.step_access_count.tolist(), [1, 3, 1, 0, 0, 0, 0, 0]
        )
        self.assertEqual(int(self.runtime.cache.stats.hits), 2)
        self.assertEqual(int(self.runtime.cache.stats.misses), 3)
        loads = [
            c for c in self.backend.transfer.calls[before:] if c["pool"] == "execution"
        ]
        self.assertEqual([len(c["entries"]) for c in loads], [2, 1])
        self.assertEqual(loads[0]["entries"], [(2, 0, 0), (1, 1, 1)])
        self.assertEqual(loads[1]["entries"], [(2, 1, 0)])

    def test_empty_wave_does_not_load_all_wave_experts(self):
        before = len(self.backend.transfer.calls)
        self.run_step({3: [1, 1]})
        loads = [
            c for c in self.backend.transfer.calls[before:] if c["pool"] == "execution"
        ]
        self.assertEqual([len(c["entries"]) for c in loads], [1, 0])
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [0, 2, 4, 7])

    def test_local_idle_with_global_work_advances_but_global_idle_does_not(self):
        self.run_step({3: []}, self.batch("IDLE", global_mode="DECODE", counts=[3, 0]))
        self.assertEqual(int(self.runtime.cache.policy_state.step_mod), 1)
        self.run_step({3: []}, self.batch("IDLE", counts=[0, 0]))
        self.assertEqual(int(self.runtime.cache.policy_state.step_mod), 1)

    def test_idle_without_tbo_counts_post_dispatch_routes(self):
        self.run_step({3: [1, 1]}, self.batch("IDLE", counts=[3, 0]))
        self.assertEqual(int(self.runtime.cache.policy_state.step_mod), 1)
        self.assertEqual(float(self.runtime.cache.policy_state.usage[1]), 2)
        self.run_step({3: []}, self.batch("IDLE", counts=[3, 0]))
        self.assertEqual(int(self.runtime.cache.policy_state.step_mod), 2)
        self.assertEqual(self.runtime._step_mod, 2)

    def test_global_prefill_overrides_idle_and_local_decode_without_tbo(self):
        self.run_step({3: [1, 1]})
        usage = self.runtime.cache.policy_state.usage.clone()
        for local in ("IDLE", "DECODE"):
            batch = self.batch(local, counts=[3, 0], extend=True)
            with self.offloader.forward_scope(batch):
                with self.offloader.eager_step_scope(batch):
                    self.assertIs(self.runtime.context.phase, Phase.PREFILL)
                    self.assertTrue(self.runtime.cache._has_snapshot)
                    self.assertTrue((self.backend.hot == -999).all())
        torch.testing.assert_close(self.runtime.cache.policy_state.usage, usage)
        self.assertEqual(self.runtime._step_mod, 1)
        self.run_step({3: [1]})
        self.assertFalse(self.runtime.cache._has_snapshot)
        self.assertEqual(self.runtime._step_mod, 2)
        self.assertEqual(int(self.runtime.cache.policy_state.step_mod), 2)
        torch.testing.assert_close(self.backend.hot, self.backend.host[[0, 2, 4, 7]])

    def test_global_idle_uses_unpadded_counts_and_preserves_prefill(self):
        batch = self.batch("IDLE", counts=[0, 0])
        batch.global_num_tokens_cpu = [4, 4]  # Graph padding is not real work.
        self.run_step({}, batch)
        self.assertEqual(self.runtime._step_mod, 0)
        with self.offloader.forward_scope(self.batch("EXTEND")):
            pass
        before = len(self.backend.transfer.calls)
        self.run_step({}, batch)
        self.assertIs(self.runtime.context.phase, Phase.PREFILL)
        self.assertEqual(len(self.backend.transfer.calls), before)
        self.assertTrue(self.runtime.cache._has_snapshot)
        self.assertEqual(self.runtime._step_mod, 0)

    def test_idle_falls_back_to_global_counts_when_original_absent(self):
        batch = self.batch("IDLE")
        batch.global_num_tokens_cpu = [2, 0]
        self.run_step({3: [1]}, batch)
        self.assertEqual(self.runtime._step_mod, 1)
        self.assertEqual(float(self.runtime.cache.policy_state.usage[1]), 1)
        self.run_step({}, self.batch("IDLE"))
        self.assertEqual(self.runtime._step_mod, 1)

    def test_eager_skips_sorting_planning_and_hot_transfer_between_rounds(self):
        policy, ops = self.runtime.cache.policy, self.runtime.residency_ops
        for step in range(1, 9):
            before = len(self.backend.transfer.calls)
            with (
                patch.object(torch, "argsort", wraps=torch.argsort) as argsort,
                patch.object(torch, "sort", wraps=torch.sort) as sort,
                patch.object(
                    ops, "prepare_update", wraps=ops.prepare_update
                ) as prepare,
                patch.object(ops, "commit_update", wraps=ops.commit_update) as commit,
            ):
                self.run_step({3: [1, 1]})
            due = step % policy.config.update_interval == 0
            self.assertEqual(argsort.call_count, 6 if due else 0)
            self.assertEqual(sort.call_count, 2 if due else 0)
            self.assertEqual(prepare.call_count, int(due))
            self.assertEqual(commit.call_count, int(due))
            hot = [
                c for c in self.backend.transfer.calls[before:] if c["pool"] == "hot"
            ]
            self.assertEqual(len(hot), int(due))
            self.assertEqual(self.runtime._step_mod, step % 4)
            self.assertEqual(int(self.runtime.cache.policy_state.step_mod), step % 4)
        torch.testing.assert_close(
            self.runtime.cache.policy_state.usage[1], torch.tensor((8 * 0.7 + 8) * 0.7)
        )

    def test_eager_period_survives_prefill_and_graph_execution(self):
        self.run_step({3: [1, 1]})
        # Execute the same tensor body a replay would submit; this is not a
        # hardware graph test. Its host scope must advance the schedule once.
        with self.offloader.forward_scope(self.batch()):
            self.runtime.emit_step_begin(self.runtime.context)
            self.model({3: [1, 1]})
            self.runtime.emit_step_end(self.runtime.context)
        with self.offloader.forward_scope(self.batch("EXTEND")):
            with self.offloader.eager_step_scope(self.batch("EXTEND")):
                pass
        for step in (3, 4):
            with patch.object(
                self.runtime.cache.policy,
                "select_hot_set",
                wraps=self.runtime.cache.policy.select_hot_set,
            ) as select:
                self.run_step({3: [1, 1]})
            self.assertEqual(select.call_count, int(step == 4))
            self.assertEqual(self.runtime._step_mod, step % 4)
            self.assertEqual(int(self.runtime.cache.policy_state.step_mod), step % 4)
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [1, 2, 4, 7])
        self.assertAlmostEqual(
            float(self.runtime.cache.policy_state.usage[1]), 5.6, places=5
        )

    def test_eager_respects_configured_period(self):
        policy = self.runtime.cache.policy
        for period in (1, 2, 5):
            policy.config = replace(policy.config, update_interval=period)
            for step in range(1, period + 1):
                with patch.object(
                    policy, "select_hot_set", wraps=policy.select_hot_set
                ) as select:
                    self.run_step({3: [1]})
                self.assertEqual(select.call_count, int(step == period))
                self.assertEqual(self.runtime._step_mod, step % period)
                self.assertEqual(
                    int(self.runtime.cache.policy_state.step_mod), step % period
                )

    def test_eager_schedule_does_not_read_device_state_on_host(self):
        self.runtime.access.transfer_backend = SimpleNamespace(
            materialize_into=lambda *args: TransferTicket(ImmediateCompletion())
        )
        batch = self.batch()
        routes = LocalRoutes(torch.tensor([1, 1]), torch.tensor([True, True]))
        with ExitStack() as stack:
            for method in ("item", "tolist", "cpu", "__bool__"):
                stack.enter_context(
                    patch.object(
                        torch.Tensor, method, side_effect=AssertionError(method)
                    )
                )
            for _ in range(4):
                with self.offloader.forward_scope(batch):
                    with self.offloader.eager_step_scope(batch):
                        for wave in fixed_waves(4, 2):
                            lease = self.runtime.access.stage(
                                self.runtime.make_execution_request(
                                    3, routes, wave, self.runtime.context
                                ),
                                self.runtime.context,
                            )
                            self.runtime.access.wait_ready(
                                lease, self.runtime.context.compute_stream
                            )
                            self.runtime.access.release_after(
                                lease,
                                self.runtime.context.record_compute_completion(),
                                self.runtime.context,
                            )
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [1, 2, 4, 7])

    def test_prefill_borrow_refill_preserves_slots_usage_and_cycle(self):
        self.run_step({3: [1, 1]})
        state = self.runtime.residency
        original = {
            name: getattr(state, name).clone()
            for name in ("key_to_slot", "slot_to_key", "slot_valid", "slot_generation")
        }
        usage = self.runtime.cache.policy_state.usage.clone()
        pointers = [getattr(state, name).data_ptr() for name in original]
        with self.offloader.forward_scope(self.batch("EXTEND")):
            with self.offloader.eager_step_scope(self.batch("EXTEND")):
                self.assertTrue((self.backend.hot == -999).all())
        torch.testing.assert_close(self.runtime.cache.policy_state.usage, usage)
        # Transition back without a model invocation checks restore alone.
        with self.offloader.forward_scope(self.batch()):
            torch.testing.assert_close(
                self.backend.hot, self.backend.host[[0, 2, 4, 7]]
            )
            for name, value in original.items():
                torch.testing.assert_close(getattr(state, name), value)
            self.assertFalse(state.execution_valid.any())
            self.assertEqual(int(self.runtime.cache.policy_state.step_mod), 1)
        self.assertEqual(
            pointers, [getattr(state, name).data_ptr() for name in original]
        )

    def test_capture_hooks_do_not_count_warmup_and_host_scope_does_not_double_step(
        self,
    ):
        state = self.runtime.cache.policy_state
        for _ in range(5):
            self.runner.capture_head_hooks[0](self.runner, self.batch(), 1)
            self.model({3: [1, 1]})
            self.runner.capture_tail_hooks[0](self.runner, None, self.batch(), 1)
        self.assertEqual(int(state.step_mod), 0)
        self.assertEqual(float(state.usage.sum()), 0)
        self.assertEqual(self.runtime._step_mod, 0)
        self.run_step({3: [1, 1]})
        self.assertEqual(int(state.step_mod), 1)
        # A replay's host scope must not emit an additional Python step.
        with self.offloader.forward_scope(self.batch()):
            # Stand-in for the recorded device operations, without eager scope.
            self.runtime.emit_step_begin(self.runtime.context)
            self.model({3: [1, 1]})
            self.runtime.emit_step_end(self.runtime.context)
        self.assertEqual(int(state.step_mod), 2)
        self.assertEqual(self.runtime._step_mod, 2)
        self.assertFalse(bool(self.runtime._enabled))
        self.assertEqual(self.runtime.context.retained(), ())

    def test_failed_copy_never_publishes_new_expert(self):
        for _ in range(3):
            self.run_step({3: [1, 1, 1]})
        before = self.runtime.residency.key_to_slot.clone()
        # Fail only the hot admission, after successful execution-bank loads.
        materialize = self.backend.transfer.materialize_into

        def fail_hot(plan, bindings, stream):
            if bindings.pool == "hot" and plan.valid.any():
                self.backend.transfer.fail_next = True
            return materialize(plan, bindings, stream)

        self.backend.transfer.materialize_into = fail_hot
        with self.assertRaisesRegex(RuntimeError, "test copy failed"):
            self.run_step({3: [1, 1, 1]})
        torch.testing.assert_close(self.runtime.residency.key_to_slot, before)
        self.assertEqual(self.offloader.owner.state, OwnerState.FAILED)
        self.assertFalse(self.runtime.residency.slot_valid[0])
        self.assertEqual(self.runtime._step_mod, 3)  # Failed forward is not committed.

    def test_verify_remains_unsupported(self):
        with self.assertRaises(NotImplementedError):
            self.run_step({}, self.batch("TARGET_VERIFY"))

    def test_actual_model_runner_eager_branch_and_graph_branch(self):
        # Execute the actual method body without importing GPU model modules.
        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/model_executor/model_runner.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_forward_raw"
        )
        namespace = dict(
            Optional=Optional,
            ForwardBatch=object,
            PPProxyTensors=object,
            ModelRunnerOutput=SimpleNamespace,
            contextlib=contextlib,
            has_forward_context=lambda: True,
            get_global_dwdp_manager=lambda: None,
        )
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        runner = self.runner
        runner.device = "cpu"
        runner.is_draft_worker = False
        runner.offloader = self.offloader
        runner.hisparse_coordinator = None
        runner._prepare_eager_forward_batch = lambda batch: None
        runner._maybe_execute_deferred_mamba_cow_and_clear = lambda batch: None
        runner.decode_cuda_graph_runner = None
        runner.eager_runner = SimpleNamespace(
            execute=lambda *args, **kwargs: self.model({3: [1, 1]})
        )
        batch = self.batch()
        with self.offloader.forward_scope(batch):
            output = namespace["_forward_raw"](runner, batch, None)
        self.assertFalse(output.can_run_graph)
        self.assertEqual(int(self.runtime.cache.policy_state.step_mod), 1)

        # Stand-in graph execution: the real branch must bypass eager hooks,
        # while the recorded tensor operations advance the device cycle once.
        def replay(*args, **kwargs):
            self.runtime.emit_step_begin(self.runtime.context)
            self.model({3: [1, 1]})
            self.runtime.emit_step_end(self.runtime.context)
            return "graph-result"

        runner.decode_cuda_graph_runner = SimpleNamespace(
            can_run_graph=lambda batch: True,
            execute=replay,
        )
        with self.offloader.forward_scope(batch):
            output = namespace["_forward_raw"](runner, batch, None)
        self.assertTrue(output.can_run_graph)
        self.assertEqual(output.logits_output, "graph-result")
        self.assertEqual(int(self.runtime.cache.policy_state.step_mod), 2)
        self.assertEqual(self.runtime._step_mod, 2)

    def test_copy_failure_during_restore_does_not_restore_metadata(self):
        with self.offloader.forward_scope(self.batch("EXTEND")):
            pass
        before = self.runtime.residency.slot_generation.clone()
        self.backend.transfer.fail_next = True
        with self.assertRaisesRegex(RuntimeError, "test copy failed"):
            with self.offloader.forward_scope(self.batch()):
                self.fail("Failed restore must not enter model execution")
        self.assertTrue(self.runtime.cache._has_snapshot)
        self.assertFalse(self.runtime.residency.slot_valid.any())
        torch.testing.assert_close(self.runtime.residency.slot_generation, before)
        self.assertEqual(self.offloader.owner.state, OwnerState.FAILED)

    def test_zero_and_full_hot_capacity_use_same_call_chain(self):
        for ratio in (0.0, 1.0):
            with self.subTest(ratio=ratio):
                backend = Backend()
                runtime = backend.finalize(
                    (),
                    self.runtime.cache.catalog,
                    MoEOffloadConfig(ratio, 2, 2, 1),
                    self.runtime.cache.policy,
                )
                runtime._enabled.fill_(True)
                for _ in range(4):
                    runtime.emit_step_begin(runtime.context)
                    routes = LocalRoutes(
                        torch.tensor([0, 1, 1, 3]), torch.tensor([True] * 4)
                    )
                    output = runtime.decode_program.run(
                        self.layers[3], torch.tensor(0.0), routes, runtime.context
                    )
                    torch.testing.assert_close(output, backend.host[[0, 1, 1, 3]].sum())
                    runtime.emit_step_end(runtime.context)
                self.assertEqual(int(runtime.cache.stats.hits), 4 if ratio else 0)
                self.assertEqual(
                    int(runtime.residency.slot_valid.sum()), int(8 * ratio)
                )
                runtime.synchronize()
                runtime.close()

    def test_two_banks_use_distinct_physical_destinations(self):
        backend = Backend()
        runtime = backend.finalize(
            (),
            self.runtime.cache.catalog,
            MoEOffloadConfig(0.5, 4, 2, 1),
            self.runtime.cache.policy,
        )
        runtime.decode_program.waves = fixed_waves(4, 2, bank_count=2)
        runtime._enabled.fill_(True)
        runtime.emit_step_begin(runtime.context)
        routes = LocalRoutes(torch.tensor([0, 1, 2, 3]), torch.tensor([True] * 4))
        output = runtime.decode_program.run(
            self.layers[3], torch.tensor(0.0), routes, runtime.context
        )
        torch.testing.assert_close(output, backend.host[:4].sum())
        loads = [c for c in backend.transfer.calls if c["pool"] == "execution"]
        self.assertEqual(
            [[e[2] for e in c["entries"]] for c in loads], [[0, 1], [2, 3]]
        )
        runtime.emit_step_end(runtime.context)
        runtime.synchronize()
        runtime.close()

    def test_cache_caller_does_not_read_device_routes_on_host(self):
        # No-op transport isolates production tensor planning from the test
        # transport's Python copy loop. It is never used to validate weights.
        self.runtime.access.transfer_backend = SimpleNamespace(
            materialize_into=lambda *args: TransferTicket(ImmediateCompletion())
        )
        self.runtime._enabled.fill_(True)
        routes = LocalRoutes(
            torch.tensor([[1, 1], [1, -1]]), torch.tensor([[True, True], [True, False]])
        )
        with ExitStack() as stack:
            for method in ("item", "tolist", "cpu", "__bool__"):
                stack.enter_context(
                    patch.object(
                        torch.Tensor, method, side_effect=AssertionError(method)
                    )
                )
            for _ in range(4):
                self.runtime.emit_step_begin(self.runtime.context)
                for wave in fixed_waves(4, 2):
                    lease = self.runtime.access.stage(
                        self.runtime.make_execution_request(
                            3, routes, wave, self.runtime.context
                        ),
                        self.runtime.context,
                    )
                    self.runtime.access.wait_ready(
                        lease, self.runtime.context.compute_stream
                    )
                    self.runtime.access.release_after(
                        lease,
                        self.runtime.context.record_compute_completion(),
                        self.runtime.context,
                    )
                self.runtime.emit_step_end(self.runtime.context)
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [1, 2, 4, 7])


if __name__ == "__main__":
    unittest.main()
