"""CPU tests of MoE compute lifetimes/layout; native EP/GMM are simulated."""

import unittest
import weakref
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.layer_execution import LayerExecutionRequest
from sglang.srt.offloading.moe.layer_prefetch import LayerPrefetchPipeline
from sglang.srt.offloading.moe.layer_programs import LayerExpertProgram
from sglang.srt.offloading.moe.npu import experts as native
from sglang.srt.offloading.moe.runtime import BoundMoERuntime, ExecutionContext
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

Routes = namedtuple("Routes", "topk_ids topk_weights")


@dataclass
class QuantInfo:
    w13_weight: object
    w2_weight: object


class TestLayerWorkspace(unittest.TestCase):
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

    def fixture(
        self, ids=(0, 2), experts=4, *, phase=Phase.DECODE, layers=tuple(range(3, 43))
    ):
        slots = experts
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
                # Previous layers must not remain alive during a new F.
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
                quant.w13_weight[:, 0, 0].copy_(torch.arange(1, experts + 1))
                binding = SimpleNamespace(
                    expert_bank=bank, error_status=error, bank_id=request.bank_id
                )
                return SimpleNamespace(
                    bindings=binding,
                    read_done=None,
                    reservations=(SimpleNamespace(bindings=binding),),
                    ticket=TransferTicket(ImmediateCompletion()),
                )

            def wait_ready(self, lease, stream, *, context=None):
                waits.append("ready")
                return lease.bindings

            def release_after(self, lease, ticket, context):
                waits.append("read_done")
                lease.read_done = ticket

        dispatcher = native.NPUOffloadDispatcher(
            SimpleNamespace(dispatcher=Dispatch(), reduce_results=False),
            error_status=error,
        )
        # Communication results are simulated; the packing/runner code is real.
        dispatcher._combine_input = lambda raw, hidden_states: SimpleNamespace(
            hidden_states=hidden_states
        )
        runner = native.NPUExpertRunner(None, _core=Core(), _input_type=SimpleNamespace)
        access = Access()
        pipeline = None
        if phase is Phase.PREFILL:
            pipeline = LayerPrefetchPipeline(
                access,
                layers,
                make_request=lambda layer, bank, ctx: LayerExecutionRequest(
                    layer, None, Phase.PREFILL, bank
                ),
                record_copy_completion=lambda ctx: TransferTicket(
                    ImmediateCompletion()
                ),
            )
        program = LayerExpertProgram(
            dispatcher,
            layout,
            runner,
            access,
            compute_scope=lambda ctx: nullcontext(),
            pipeline=pipeline,
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
                i: SimpleNamespace(decode=program, prefill=program) for i in layers
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
                    self.assertEqual(fixture.waits, ["ready", "read_done"] * 40)
                    # Neither prefill nor decode adds a host synchronization.
                    self.assertNotIn("drain", fixture.waits)
                    completion = fixture.context.finish()
                    self.assertTrue(completion.resources_idle.completion.done())

    def test_padding_is_sanitized_in_place_including_empty_and_partial_layers(self):
        cases = [((0, 2), 4), ((2, 2), 4), ((-1, -1), 4), ((4, 0), 5)]
        with (
            patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}),
            patch.object(native, "_compute_scope", lambda *args: nullcontext()),
        ):
            for ids, experts in cases:
                with self.subTest(ids=ids, experts=experts):
                    fixture = self.fixture(ids, experts)
                    write = fixture.layout.write_layer

                    def checked_write(layout, output, context):
                        # Padding cleanup must reuse the native GMM allocation.
                        self.assertIs(output, fixture.output_refs[-1]())
                        self.assertTrue(bool(torch.isfinite(output).all()))
                        count = int(layout.selected.row_count)
                        self.assertTrue(bool((output[count:] == 0).all()))
                        write(layout, output, context)

                    fixture.layout.write_layer = checked_write
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

    def test_failed_prefill_layer_keeps_tensors_through_failed_drain(self):
        with (
            patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}),
            patch.object(native, "_compute_scope", lambda *args: nullcontext()),
        ):
            fixture = self.fixture(phase=Phase.PREFILL)
            fixture.runtime.layer_programs[3].prefill.layer_start(3, fixture.context)

            def fail(*args):
                raise RuntimeError("write failed after F")

            fixture.layout.write_layer = fail
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

    def test_failed_layer_keeps_tensors_until_failure_drain(self):
        with (
            patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}),
            patch.object(native, "_compute_scope", lambda *args: nullcontext()),
        ):
            fixture = self.fixture()

            def fail(*args):
                raise RuntimeError("write failed after F")

            fixture.layout.write_layer = fail
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

    def test_eager_idle_prefill_serves_remote_experts_and_handles_empty_receive(self):
        with (
            patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}),
            patch.object(native, "_compute_scope", lambda *args: nullcontext()),
        ):
            for width in (8, 16):
                for received in (0, 2):
                    with self.subTest(width=width, received=received):
                        fixture = self.fixture(ids=(), phase=Phase.PREFILL, layers=(3,))
                        runtime = fixture.runtime
                        runtime._context, runtime._capture = None, None
                        runtime.device_runtime.stream_scope = lambda stream: (
                            nullcontext()
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
                        original_write = fixture.layout.write_layer

                        def checked_write(layout, output, context):
                            count = int(layout.selected.row_count)
                            calls.append(("layer", count))
                            if count:
                                torch.testing.assert_close(
                                    output[:count], torch.full_like(output[:count], 4)
                                )
                            self.assertTrue(bool((output[count:] == 0).all()))
                            original_write(layout, output, context)

                        fixture.layout.write_layer = checked_write
                        with runtime.forward_scope(owner, batch):
                            self.assertIs(runtime.context.phase, Phase.PREFILL)
                            runtime._enter_layer(3)
                            output = runtime.run_layer(3, fixture.input, fixture.routes)
                            self.assertEqual(output.shape, (0, 4))
                        self.assertEqual(calls, ["dispatch", ("layer", received)])
                        self.assertIsNone(dispatcher.handle)
                        self.assertIsNone(program.pipeline._session)


if __name__ == "__main__":
    unittest.main()
