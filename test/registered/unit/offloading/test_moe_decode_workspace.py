"""CPU tests of MoE compute lifetimes/layout; native EP/GMM are simulated."""

import unittest
import weakref
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.offloading.moe.config import ExpertWave, Phase, fixed_waves
from sglang.srt.offloading.moe.npu import experts as native
from sglang.srt.offloading.moe.runtime import BoundMoERuntime
from sglang.srt.offloading.moe.programs import (
    DecodeLocalExpertProgram,
    DispatchOnceProgram,
    PrefillDispatchOnceProgram,
    PrefillLocalExpertProgram,
)
from sglang.srt.offloading.moe.runtime import ExecutionContext
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

Routes = namedtuple("Routes", "topk_ids topk_weights")


@dataclass
class QuantInfo:
    w13_weight: object
    w2_weight: object


class TestDecodeWorkspace(unittest.TestCase):
    def test_bank_rejects_mismatched_payload_before_native_execution(self):
        valid = torch.ones(2, dtype=torch.bool)
        error = torch.zeros((), dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "exact FFN bank"):
            native.NPUExpertBank(
                QuantInfo(torch.ones(2, 1, 1), torch.ones(3, 1, 1)), valid, error
            )
        bank = native.NPUExpertBank(
            QuantInfo(torch.ones(2, 1, 1), torch.ones(2, 1, 1)), valid, error
        )
        runner = native.NPUExpertRunner(None, _core=object(), _input_type=object)
        with self.assertRaisesRegex(ValueError, "bound expert view"):
            runner(
                3,
                SimpleNamespace(expert_tokens=torch.zeros(1, dtype=torch.int64)),
                SimpleNamespace(expert_bank=bank, error_status=error),
                None,
            )

    def fixture(self, ids=(0, 2), experts=4, slots=2, *, phase=Phase.DECODE):
        ids = torch.tensor(ids, dtype=torch.int64).unsqueeze(1)
        rows, hidden = max(8, ids.numel()), 4
        raw_refs, output_refs, waits = [], [], []
        error = torch.zeros((), dtype=torch.int32)
        layout = native.NPUDispatchLayout(experts, hidden, rows, "cpu", error)
        quant = QuantInfo(torch.ones(slots, 1, 1), torch.ones(slots, 1, 1))
        bank = native.NPUExpertBank(quant, torch.ones(slots, dtype=torch.bool), error)
        test = self

        class Dispatch:
            handle = None

            def dispatch(self, hidden_states, topk_output):
                expert_ids = topk_output.topk_ids[:, 0]
                selected = (expert_ids >= 0).nonzero().flatten()
                order = selected[expert_ids[selected].argsort()]
                receive = torch.full((rows, hidden), float("nan"), dtype=torch.bfloat16)
                receive[: order.numel()].copy_(hidden_states[order])
                raw_refs.append(weakref.ref(receive))
                self.handle = order
                self.token_rows = hidden_states.shape[0]
                return SimpleNamespace(
                    format=SimpleNamespace(
                        value="deepep_normal" if phase is Phase.PREFILL else "deepep_ll"
                    ),
                    hidden_states=receive,
                    hidden_states_scale=None,
                    masked_m=torch.bincount(expert_ids[order], minlength=experts),
                    num_recv_tokens_per_expert=torch.bincount(
                        expert_ids[order], minlength=experts
                    ).tolist(),
                    topk_ids=topk_output.topk_ids,
                    topk_weights=topk_output.topk_weights,
                )

            def combine(self, combine_input):
                result = torch.zeros(self.token_rows, hidden, dtype=torch.bfloat16)
                result.index_copy_(
                    0, self.handle, combine_input.hidden_states[: self.handle.numel()]
                )
                self.handle = None
                return result

        class Core:
            def run(self, runner_input, quant_info, running_state):
                # Previous waves/layers must not remain alive during a new F.
                test.assertTrue(all(ref() is None for ref in output_refs))
                output = torch.full_like(runner_input.hidden_states, float("nan"))
                offset = 0
                for lane, count in enumerate(runner_input.expert_tokens.tolist()):
                    output[offset : offset + count].copy_(
                        runner_input.hidden_states[offset : offset + count]
                        + quant_info.w13_weight[lane, 0, 0]
                    )
                    offset += count
                output_refs.append(weakref.ref(output))
                return SimpleNamespace(hidden_states=output)

        class Access:
            def stage(self, request, context):
                wave = request.wave
                size = wave.local_end - wave.local_start
                quant.w13_weight[:size, 0, 0].copy_(
                    torch.arange(wave.local_start + 1, wave.local_end + 1)
                )
                return SimpleNamespace(
                    expert_bank=native.NPUExpertBank(
                        QuantInfo(quant.w13_weight[:size], quant.w2_weight[:size]),
                        bank.execution_valid[:size],
                        error,
                    ),
                    error_status=error,
                )

            def wait_ready(self, lease, stream, *, context=None):
                waits.append("ready")
                return lease

            def release_after(self, lease, ticket, context):
                waits.append("read_done")

        dispatcher = native.NPUOffloadDispatcher(
            SimpleNamespace(dispatcher=Dispatch(), reduce_results=False),
            error_status=error,
        )
        # Communication results are simulated; the packing/runner code is real.
        dispatcher._combine_input = lambda raw, hidden_states: SimpleNamespace(
            hidden_states=hidden_states
        )
        runner = native.NPUExpertRunner(None, _core=Core(), _input_type=SimpleNamespace)
        waves = fixed_waves(experts, slots)
        access = Access()
        if phase is Phase.PREFILL:

            class Pipeline:
                def __init__(self):
                    self.access = access
                    self.active_layer = None

                def start(self, layer, context):
                    test.assertIsNone(self.active_layer)
                    self.active_layer = layer

                def validate_session(self, layer, context):
                    test.assertEqual(self.active_layer, layer)

                def seal(self, layer, ready, context):
                    self.validate_session(layer, context)
                    return waves

                def acquire(self, layer, wave, context):
                    self.validate_session(layer, context)
                    return self.access.stage(SimpleNamespace(wave=wave), context)

                def finish(self, layer, context):
                    self.validate_session(layer, context)
                    self.active_layer = None
                    return TransferTicket(ImmediateCompletion())

                def abort(self, context):
                    self.active_layer = None
                    return TransferTicket(ImmediateCompletion())

            pipeline = Pipeline()
            program = PrefillDispatchOnceProgram(
                dispatcher,
                PrefillLocalExpertProgram(pipeline, runner),
                layout,
                local_experts=experts,
                pipeline=pipeline,
            )
        else:
            local = DecodeLocalExpertProgram(
                access,
                lambda layer, wave_input, wave, context: SimpleNamespace(wave=wave),
                runner,
            )
            program = DispatchOnceProgram(
                waves, dispatcher, local, layout, local_experts=experts
            )
        device = SimpleNamespace(
            copy_stream="copy",
            current_stream=lambda: "compute",
            record=lambda stream: TransferTicket(ImmediateCompletion()),
            synchronize=lambda: waits.append("drain"),
        )
        runtime = BoundMoERuntime(
            device_runtime=device,
            residency=SimpleNamespace(update_enabled=torch.tensor(True)),
            residency_ops=SimpleNamespace(begin_step=lambda context: None),
            cache_updates=SimpleNamespace(policy=object()),
            snapshot=None,
            reload_weights=None,
            layer_programs={
                i: SimpleNamespace(decode=program, prefill=program)
                for i in range(3, 43)
            },
            resources=(),
            expert_nbytes=1,
            storage=(),
        )
        context = runtime._new_context(phase, capturing=phase is Phase.DECODE)
        context.forward_batch = SimpleNamespace(
            moe_num_token_non_padded=lambda: torch.tensor(ids.shape[0])
        )
        runtime._context, runtime._capture = context, []
        if phase is Phase.PREFILL:
            runtime.pipelines = (pipeline,)
        return SimpleNamespace(
            runtime=runtime,
            context=context,
            layout=layout,
            routes=Routes(ids, torch.ones_like(ids, dtype=torch.float32)),
            input=torch.ones(ids.shape[0], hidden, dtype=torch.bfloat16),
            raw_refs=raw_refs,
            output_refs=output_refs,
            bank=bank,
            waits=waits,
        )

    def test_forty_layers_reuse_workspace_without_pinning_receive_or_ffn_outputs(self):
        with (
            patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}),
            patch.object(native, "_compute_scope", lambda *args: nullcontext()),
        ):
            for phase in Phase:
                with self.subTest(phase=phase):
                    fixture = self.fixture(phase=phase)
                    pointers = {
                        k: v.data_ptr() for k, v in fixture.layout._owned.items()
                    }
                    value = fixture.input
                    for step, layer in enumerate(range(3, 43), 1):
                        if phase is Phase.PREFILL:
                            fixture.runtime.layer_programs[3].prefill.layer_start(
                                layer, fixture.context
                            )
                        routes = Routes(
                            fixture.routes.topk_ids.clone(),
                            fixture.routes.topk_weights.clone(),
                        )
                        route_refs = [weakref.ref(tensor) for tensor in routes]
                        value = fixture.runtime.run_layer(layer, value, routes)
                        del routes
                        self.assertTrue(all(ref() is None for ref in route_refs))
                        self.assertTrue(all(ref() is None for ref in fixture.raw_refs))
                        self.assertTrue(
                            all(ref() is None for ref in fixture.output_refs)
                        )
                        torch.testing.assert_close(
                            value,
                            fixture.input + step * (fixture.routes.topk_ids + 1),
                        )
                    self.assertEqual(
                        pointers,
                        {k: v.data_ptr() for k, v in fixture.layout._owned.items()},
                    )
                    self.assertEqual(fixture.waits, ["ready", "read_done"] * 80)
                    # Neither prefill nor decode adds a host synchronization.
                    self.assertNotIn("drain", fixture.waits)
                    completion = fixture.context.finish()
                    self.assertTrue(completion.resources_idle.completion.done())

    def test_padding_is_sanitized_in_place_including_empty_and_partial_waves(self):
        cases = [((0, 2), 4, 2), ((2, 2), 4, 2), ((-1, -1), 4, 2), ((4, 0), 5, 2)]
        with (
            patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}),
            patch.object(native, "_compute_scope", lambda *args: nullcontext()),
        ):
            for ids, experts, slots in cases:
                with self.subTest(ids=ids, experts=experts):
                    fixture = self.fixture(ids, experts, slots)
                    write = fixture.layout.write_wave

                    def checked_write(layout, wave, output, context):
                        # Padding cleanup must reuse the native GMM allocation.
                        self.assertIs(output, fixture.output_refs[-1]())
                        self.assertTrue(bool(torch.isfinite(output).all()))
                        count = int(layout.selected.row_count)
                        self.assertTrue(bool((output[count:] == 0).all()))
                        write(layout, wave, output, context)

                    fixture.layout.write_wave = checked_write
                    result = fixture.runtime.run_layer(3, fixture.input, fixture.routes)
                    torch.testing.assert_close(
                        result,
                        torch.where(
                            fixture.routes.topk_ids >= 0,
                            fixture.input + fixture.routes.topk_ids + 1,
                            0,
                        ),
                    )

    def test_scopes_keep_transfer_storage_and_events_until_completion(self):
        class Completion(ImmediateCompletion):
            ready = False

            def done(self):
                return self.ready

            def synchronize(self):
                self.ready = True

        for phase in Phase:
            with self.subTest(phase=phase):
                completion = Completion()
                context = ExecutionContext(
                    phase,
                    compute_stream="compute",
                    copy_stream="copy",
                    record_completion=lambda stream: TransferTicket(
                        ImmediateCompletion()
                    ),
                    capturing=phase is Phase.DECODE,
                )
                with context.temporary_scope():
                    temporary = torch.ones(4)
                    protected = torch.ones(4)
                    temporary_ref = weakref.ref(temporary)
                    protected_ref = weakref.ref(protected)
                    context.retain(temporary)
                    ticket = TransferTicket(completion, (protected,))
                    context.retain(ticket)
                    context.register_completion(ticket)
                    del temporary, protected
                self.assertIsNone(temporary_ref())
                self.assertIsNotNone(protected_ref())
                self.assertFalse(completion.done())
                finished = context.finish()
                self.assertFalse(finished.resources_idle.completion.done())
                finished.resources_idle.completion.synchronize()
                self.assertTrue(completion.done())
                # Event/transfer storage survives normal compute-scope release.
                self.assertIsNotNone(protected_ref())

    def test_prefill_keeps_invocation_lifetimes_for_worker_submissions(self):
        context = ExecutionContext(
            Phase.PREFILL,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=lambda stream: TransferTicket(ImmediateCompletion()),
        )

        def submit():
            tensor = torch.ones(4)
            reference = weakref.ref(tensor)
            context.retain(tensor)
            return reference

        with ThreadPoolExecutor(max_workers=1) as workers:
            with context.temporary_scope():
                tensor = torch.ones(4)
                compute_reference = weakref.ref(tensor)
                context.retain(tensor)
                reference = workers.submit(submit).result()
                del tensor
            self.assertIsNone(compute_reference())
            self.assertIsNotNone(reference())
        context.finish()
        context.dispose_completed()
        self.assertIsNone(reference())

    def test_nested_prefill_scopes_release_only_their_own_compute_temporaries(self):
        context = ExecutionContext(
            Phase.PREFILL,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=lambda stream: TransferTicket(ImmediateCompletion()),
        )
        with context.temporary_scope():
            outer = torch.ones(4)
            outer_ref = weakref.ref(outer)
            context.retain(outer)
            del outer
            with context.temporary_scope():
                tensor = torch.ones(4)
                reference = weakref.ref(tensor)
                context.retain(tensor)
                del tensor
            self.assertIsNone(reference())
            self.assertIsNotNone(outer_ref())
        self.assertIsNone(outer_ref())
        context.finish()

    def test_worker_cannot_open_submission_thread_temporary_scopes(self):
        context = ExecutionContext(
            Phase.PREFILL,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=lambda stream: TransferTicket(ImmediateCompletion()),
        )

        def submit():
            with context.temporary_scope():
                self.fail("Worker must keep invocation lifetimes")

        with ThreadPoolExecutor(max_workers=1) as workers:
            with self.assertRaisesRegex(RuntimeError, "submission thread"):
                workers.submit(submit).result()
        self.assertFalse(context._temporary_resources)
        context.finish()

    def test_failed_prefill_wave_keeps_tensors_through_failed_drain(self):
        with (
            patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}),
            patch.object(native, "_compute_scope", lambda *args: nullcontext()),
        ):
            fixture = self.fixture(phase=Phase.PREFILL)
            fixture.runtime.layer_programs[3].prefill.layer_start(3, fixture.context)

            def fail(*args):
                raise RuntimeError("write failed after F")

            fixture.layout.write_wave = fail
            with self.assertRaisesRegex(RuntimeError, "write failed after F"):
                fixture.runtime.run_layer(3, fixture.input, fixture.routes)
            self.assertIsNotNone(fixture.raw_refs[-1]())
            self.assertIsNotNone(fixture.output_refs[-1]())
            self.assertFalse(fixture.context._temporary_resources)
            with patch.object(
                fixture.runtime.device_runtime,
                "synchronize",
                side_effect=RuntimeError("drain failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "drain failed"):
                    fixture.runtime.synchronize()
            self.assertIsNotNone(fixture.raw_refs[-1]())
            self.assertIsNotNone(fixture.output_refs[-1]())
            fixture.runtime.synchronize()
            fixture.context.dispose_after_drain()
            self.assertIsNone(fixture.raw_refs[-1]())
            self.assertIsNone(fixture.output_refs[-1]())

    def test_failed_wave_keeps_tensors_until_failure_drain(self):
        with (
            patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}),
            patch.object(native, "_compute_scope", lambda *args: nullcontext()),
        ):
            fixture = self.fixture()

            def fail(*args):
                raise RuntimeError("write failed after F")

            fixture.layout.write_wave = fail
            with self.assertRaisesRegex(RuntimeError, "write failed after F"):
                fixture.runtime.run_layer(3, fixture.input, fixture.routes)
            self.assertIsNotNone(fixture.raw_refs[-1]())
            self.assertIsNotNone(fixture.output_refs[-1]())
            self.assertFalse(fixture.context._temporary_resources)
            fixture.runtime.synchronize()
            with self.assertRaisesRegex(RuntimeError, "Destroy the captured graph"):
                fixture.context.dispose_after_drain()
            self.assertIsNotNone(fixture.raw_refs[-1]())
            fixture.context.dispose_after_drain(graph_destroyed=True)
            self.assertIn("drain", fixture.waits)
            self.assertIsNone(fixture.raw_refs[-1]())
            self.assertIsNone(fixture.output_refs[-1]())

    def test_eager_idle_prefill_serves_remote_experts_and_joins_empty_waves(self):
        with (
            patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}),
            patch.object(native, "_compute_scope", lambda *args: nullcontext()),
        ):
            for width in (8, 16):
                for received in (0, 2):
                    with self.subTest(width=width, received=received):
                        fixture = self.fixture(ids=(), phase=Phase.PREFILL)
                        runtime = fixture.runtime
                        runtime._context, runtime._capture = None, None
                        runtime.device_runtime.stream_scope = (
                            lambda stream: nullcontext()
                        )
                        owner = SimpleNamespace(
                            phase=Phase.DECODE,
                            begin_forward=lambda phase, stream: SimpleNamespace(
                                sequence=1
                            ),
                            finish_forward=lambda token, completion: None,
                            fail_forward=lambda token: None,
                        )
                        batch = SimpleNamespace(
                            forward_mode=SimpleNamespace(
                                name="IDLE",
                                is_decode_or_idle=lambda: True,
                                is_extend_without_speculative=lambda: False,
                            ),
                            is_extend_in_batch=True,
                            original_global_num_tokens_cpu=[1] + [0] * (width - 1),
                            input_ids=torch.empty(0, dtype=torch.int64),
                            moe_num_token_non_padded=lambda: torch.tensor(0),
                        )
                        program = runtime.layer_programs[3].prefill
                        dispatcher = program.dispatcher.dispatcher
                        original_dispatch = dispatcher.dispatch
                        calls = []

                        def receive_remote(hidden_states, topk_output):
                            self.assertEqual(hidden_states.shape[0], 0)
                            raw = original_dispatch(hidden_states, topk_output)
                            raw.hidden_states[:received].fill_(2)
                            raw.num_recv_tokens_per_expert = [0, received, 0, 0]
                            raw.masked_m = torch.tensor([0, received, 0, 0])
                            calls.append("dispatch")
                            return raw

                        dispatcher.dispatch = receive_remote
                        original_write = fixture.layout.write_wave

                        def checked_write(layout, wave, output, context):
                            count = int(layout.selected.row_count)
                            calls.append((wave.index, count))
                            if count:
                                torch.testing.assert_close(
                                    output[:count], torch.full_like(output[:count], 4)
                                )
                            self.assertTrue(bool((output[count:] == 0).all()))
                            original_write(layout, wave, output, context)

                        fixture.layout.write_wave = checked_write
                        with runtime.forward_scope(owner, batch):
                            self.assertIs(runtime.context.phase, Phase.PREFILL)
                            runtime._enter_layer(3)
                            output = runtime.run_layer(3, fixture.input, fixture.routes)
                            self.assertEqual(output.shape, (0, 4))
                        self.assertEqual(calls, ["dispatch", (0, received), (1, 0)])
                        self.assertIsNone(dispatcher.handle)
                        self.assertIsNone(program.pipeline.active_layer)


class TestPrefillCompactLayout(unittest.TestCase):
    def setUp(self):
        self.scope = patch.object(native, "_compute_scope", lambda *args: nullcontext())
        self.scope.start()
        self.addCleanup(self.scope.stop)
        self.events = []

        def record(stream):
            self.events.append(stream)
            return TransferTicket(ImmediateCompletion())

        self.context = ExecutionContext(
            Phase.PREFILL,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=record,
        )
        self.error = torch.zeros((), dtype=torch.int32)

    def bind(self, counts, x, scale=None):
        adapter = native.NPUDispatchLayout(
            len(counts), x.shape[1], max(1, x.shape[0]), "cpu", self.error
        )
        raw = SimpleNamespace(
            format=SimpleNamespace(value="deepep_normal"),
            hidden_states=x,
            hidden_states_scale=scale,
            num_recv_tokens_per_expert=counts,
        )
        dispatched = native.NPUDispatch(
            raw, TransferTicket(ImmediateCompletion()), x.shape[1], object()
        )
        return adapter, adapter.bind(dispatched, self.context)

    def bindings(self, wave):
        width = wave.local_end - wave.local_start
        weights = torch.arange(wave.local_start + 1, wave.local_end + 1).reshape(
            width, 1, 1
        )
        bank = native.NPUExpertBank(
            QuantInfo(weights, weights), torch.ones(width, dtype=torch.bool), self.error
        )
        return SimpleNamespace(expert_bank=bank, error_status=self.error)

    def test_prefill_slices_share_activation_and_scale_storage_without_device_reads(
        self,
    ):
        payload = torch.arange(8 * 16).to(torch.uint8).reshape(8, 16)
        scales = torch.arange(8 * 4).to(torch.uint8).reshape(8, 4)
        for dtype in (torch.bfloat16, torch.float16, torch.int8, torch.float8_e4m3fn):
            for scale in (None, scales.view(torch.float32).flatten(), scales):
                with self.subTest(
                    dtype=dtype, scale=None if scale is None else scale.dtype
                ):
                    x = payload.view(dtype)
                    with (
                        patch.object(
                            torch.Tensor,
                            "item",
                            side_effect=AssertionError("device read"),
                        ),
                        patch.object(
                            torch.Tensor,
                            "tolist",
                            side_effect=AssertionError("device read"),
                        ),
                        patch.object(
                            torch,
                            "index_select",
                            side_effect=AssertionError("full gather"),
                        ),
                    ):
                        adapter, layout = self.bind([1, 0, 2, 1], x, scale)
                        wave = ExpertWave(0, 1, 3, 0)
                        selected = adapter.select_wave(layout, wave, self.context)
                        self.assertEqual(selected.hidden_states.shape, (2, x.shape[1]))
                        self.assertEqual(
                            selected.hidden_states.data_ptr(), x[1:].data_ptr()
                        )
                        self.assertTrue(
                            torch.equal(
                                selected.hidden_states.view(torch.uint8), payload[1:3]
                            )
                        )
                        if scale is not None:
                            self.assertEqual(
                                selected.hidden_states_scale.data_ptr(),
                                scale[1:].data_ptr(),
                            )
                            self.assertTrue(
                                torch.equal(
                                    selected.hidden_states_scale.view(torch.uint8),
                                    scales[1:3],
                                )
                            )
                        output = torch.ones(2, x.shape[1], dtype=torch.bfloat16)
                        adapter.write_wave(layout, wave, output, self.context)
                    self.assertTrue(torch.equal(layout.output[1:3], output))
                    self.assertEqual(
                        layout.output[[0, 3, 4, 5, 6, 7]].count_nonzero(), 0
                    )
                    self.assertTrue(torch.equal(x.view(torch.uint8), payload))

    def test_prefill_matches_reference_for_variable_groups_empty_waves_and_padding(
        self,
    ):
        cases = ([0, 2, 0, 3, 1], [3, 0, 0, 0, 2], [0] * 5)
        schedules = (
            fixed_waves(5, 2),
            (ExpertWave(0, 0, 1, 0), ExpertWave(1, 1, 4, 1), ExpertWave(2, 4, 5, 0)),
        )
        for counts in cases:
            for waves in schedules:
                for dtype in (torch.bfloat16, torch.float16):
                    with self.subTest(counts=counts, waves=waves, dtype=dtype):
                        rows = sum(counts)
                        x = torch.arange((rows + 3) * 4).reshape(rows + 3, 4).to(dtype)
                        x[rows:].fill_(float("nan"))
                        adapter, layout = self.bind(counts, x)
                        seen_rows = []

                        class Core:
                            def run(self, inputs, quant, state):
                                seen_rows.append(inputs.hidden_states.shape[0])
                                output = torch.full_like(
                                    inputs.hidden_states, float("nan")
                                )
                                offset = 0
                                for lane, count in enumerate(
                                    inputs.expert_tokens.tolist()
                                ):
                                    output[offset : offset + count] = (
                                        inputs.hidden_states[offset : offset + count]
                                        * quant.w13_weight[lane, 0, 0]
                                    )
                                    offset += count
                                return SimpleNamespace(hidden_states=output)

                        runner = native.NPUExpertRunner(
                            None, _core=Core(), _input_type=SimpleNamespace
                        )
                        events_before = len(self.events)
                        for wave in waves:
                            selected = adapter.select_wave(layout, wave, self.context)
                            count = sum(counts[wave.local_start : wave.local_end])
                            self.assertEqual(selected.hidden_states.shape[0], count)
                            local = runner(
                                3, selected, self.bindings(wave), self.context
                            )
                            self.assertEqual(local.output.dtype, dtype)
                            self.assertTrue(local.read_done.completion.done())
                            adapter.write_wave(layout, wave, local.output, self.context)
                        # Empty groups still record readers, but never enter the core.
                        self.assertEqual(len(self.events) - events_before, len(waves))
                        self.assertEqual(
                            seen_rows,
                            [
                                sum(counts[w.local_start : w.local_end])
                                for w in waves
                                if sum(counts[w.local_start : w.local_end])
                            ],
                        )
                        self.assertEqual(sum(seen_rows), rows)
                        expected = torch.zeros_like(x)
                        row_experts = torch.repeat_interleave(
                            torch.arange(1, 6), torch.tensor(counts)
                        )
                        expected[:rows] = x[:rows] * row_experts.unsqueeze(1)
                        torch.testing.assert_close(
                            adapter.combine_input(layout, self.context).output, expected
                        )

    def test_empty_receive_skips_quantized_core_and_preserves_output_dtype(self):
        for dtype in (torch.int8, torch.bfloat16, torch.float16):
            adapter, layout = self.bind(
                [0, 0], torch.empty(0, 4, dtype=dtype), torch.empty(0)
            )
            wave = ExpertWave(0, 0, 2, 0)
            selected = adapter.select_wave(layout, wave, self.context)
            runner = native.NPUExpertRunner(None, _core=object(), _input_type=object)
            local = runner(3, selected, self.bindings(wave), self.context)
            expected_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16
            self.assertEqual(local.output.dtype, expected_dtype)
            self.assertEqual(local.output.shape, (0, 4))
            adapter.write_wave(layout, wave, local.output, self.context)
            self.assertEqual(
                adapter.combine_input(layout, self.context).output.shape, (0, 4)
            )

    def test_late_weight_fault_masks_earlier_prefill_results_and_nan_outputs(self):
        for fault_before_bind in (False, True):
            self.error.fill_(128 if fault_before_bind else 0)
            adapter, layout = self.bind([2, 1], torch.ones(5, 4, dtype=torch.bfloat16))

            class Core:
                def run(self, inputs, quant, state):
                    output = torch.full_like(inputs.hidden_states, float("nan"))
                    count = int(inputs.expert_tokens.sum())
                    output[:count].fill_(7)
                    return SimpleNamespace(hidden_states=output)

            runner = native.NPUExpertRunner(
                None, _core=Core(), _input_type=SimpleNamespace
            )
            for wave in fixed_waves(2, 1):
                selected = adapter.select_wave(layout, wave, self.context)
                bindings = self.bindings(wave)
                if wave.index == 1:
                    bindings.expert_bank.execution_valid.zero_()
                local = runner(3, selected, bindings, self.context)
                self.assertTrue(bool(torch.isfinite(local.output).all()))
                adapter.write_wave(layout, wave, local.output, self.context)
            self.assertNotEqual(int(self.error), 0)
            self.assertEqual(
                adapter.combine_input(layout, self.context).output.count_nonzero(), 0
            )

    def test_normal_counts_must_be_valid_host_metadata(self):
        for counts in ([1, -1], [1, 1.5], [True, 1], [5, 0], torch.tensor([1, 1])):
            with self.subTest(counts=counts), self.assertRaisesRegex(
                ValueError, "Prefill dispatch"
            ):
                self.bind(counts, torch.ones(4, 4, dtype=torch.bfloat16))

    def test_prefill_rejects_full_receive_output_for_a_compact_wave(self):
        adapter, layout = self.bind([1, 2], torch.ones(4, 4, dtype=torch.bfloat16))
        wave = ExpertWave(0, 0, 1, 0)
        adapter.select_wave(layout, wave, self.context)
        with self.assertRaisesRegex(ValueError, "selected receive shape"):
            adapter.write_wave(layout, wave, torch.zeros(4, 4), self.context)


if __name__ == "__main__":
    unittest.main()
