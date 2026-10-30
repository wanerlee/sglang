"""CPU contracts for whole-layer plans, native assembly and prefetch ordering.

No native NPU graph/tiling/performance certification is implied by these tests.
"""

import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from moe_test_utils import Context, Fence, TensorTrace, load_hot
from test_npu_backend import Assembly

from sglang.srt.offloading.moe.config import ExpertCatalog, ExpertLayerSpec, Phase
from sglang.srt.offloading.moe.layer_execution import (
    BoundExecutionLayer,
    LayerExecutionRequest,
    whole_layer_bank_layout,
)
from sglang.srt.offloading.moe.layer_prefetch import LayerPrefetchPipeline
from sglang.srt.offloading.moe.layer_programs import LayerExpertProgram
from sglang.srt.offloading.moe.npu import experts
from sglang.srt.offloading.moe.npu.config import NpuOffloadSettings
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.residency import TensorResidencyOps
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.transfer import TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestLayerPlanner(unittest.TestCase):
    def test_bindings_reject_overlapping_banks_offsets_and_duplicate_layers(self):
        catalog = ExpertCatalog(
            (ExpertLayerSpec(3, 4, 1, 0), ExpertLayerSpec(4, 4, 1, 0))
        )
        state = ResidencyState.allocate(8, 2, 6, "cpu")
        for records in (
            (BoundExecutionLayer(3, 0, 0, None), BoundExecutionLayer(4, 1, 2, None)),
            (BoundExecutionLayer(3, 0, 0, None), BoundExecutionLayer(4, 0, 2, None)),
            (BoundExecutionLayer(3, 0, 0, None), BoundExecutionLayer(3, 0, 0, None)),
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                TensorResidencyOps(
                    state, catalog, execution_layers=records, update_bindings=None
                )

    def test_recorded_layer_plan_reads_changed_demand_and_hot_mapping(self):
        catalog = ExpertCatalog((ExpertLayerSpec(3, 4, 1, 0),))
        state = ResidencyState.allocate(4, 2, 4, "cpu")
        binding = SimpleNamespace(
            error_status=torch.zeros((), dtype=torch.int32),
            hot_generations=state.slot_generation,
        )
        ops = TensorResidencyOps(
            state,
            catalog,
            update_bindings=binding,
            execution_layers=(BoundExecutionLayer(3, 0, 0, binding),),
            error_status=binding.error_status,
        )
        load_hot(state, [0, 1])
        demand = torch.tensor([1, 0, 2, 0])
        ctx, trace = Context(), TensorTrace()
        with trace:
            ops.begin_step(ctx)
            reservation = ops.resolve_execution(
                LayerExecutionRequest(3, demand, Phase.DECODE), ctx
            )
        pointers = tuple(
            t.data_ptr()
            for t in (
                reservation.copy_plan.source_kind,
                reservation.copy_plan.source_index,
                reservation.copy_plan.valid,
                reservation.copy_plan.destination_slot,
            )
        )
        for keys, counts in (([2, 3], [0, 0, 1, 2]), ([1, 2], [2, 1, 0, 1])):
            load_hot(state, keys)
            demand.copy_(torch.tensor(counts))
            trace.replay()
            p = reservation.copy_plan
            self.assertEqual(p.valid.tolist(), [n > 0 for n in counts])
            for i, count in enumerate(counts):
                if count:
                    self.assertEqual(int(p.source_kind[i]), 2 if i in keys else 1)
                    self.assertEqual(
                        int(p.source_index[i]), keys.index(i) if i in keys else i
                    )
            self.assertEqual(
                pointers,
                tuple(
                    t.data_ptr()
                    for t in (
                        p.source_kind,
                        p.source_index,
                        p.valid,
                        p.destination_slot,
                    )
                ),
            )
        self.assertFalse(any("_local_scalar_dense" in str(f) for f, *_ in trace.calls))

    def test_shipped_configs_use_layers_and_removed_wave_knobs_fail(self):
        root = Path(__file__).resolve().parents[4]
        for relative in (
            "scripts/moe_offload_a3.json",
            "python/sglang/srt/offloading/moe/npu/examples/v4.json",
        ):
            settings = NpuOffloadSettings.from_file(root / relative)
            self.assertEqual(
                (settings.prefill_mode, settings.decode_mode), ("layer", "layer")
            )
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            data = json.loads((root / "scripts/moe_offload_a3.json").read_text())
            data["prefill_group_size"] = 8
            path = Path(folder) / "old.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(
                ValueError, "Unknown MoE offload settings.*prefill_group_size"
            ):
                NpuOffloadSettings.from_file(path)
            del data["prefill_group_size"]
            data["decode_mode"] = "wave"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Wave execution was removed"):
                NpuOffloadSettings.from_file(path)

    def test_stale_cache_mapping_suppresses_whole_layer(self):
        a = Assembly(local_experts=4, slots=4, layer_count=2)
        a.backend.settings.decode_mode = "layer"
        self.addCleanup(a.close)
        runtime = a.finalize()
        runtime.initialize_hot_cache([0, 1, 4, 5])
        runtime.residency.slot_to_key[0] = 2
        context = runtime._new_context(Phase.DECODE, capturing=True)
        runtime._context = context
        runtime.ops.begin_step(context)
        lease = runtime.cache_updates.access.stage(
            LayerExecutionRequest(3, torch.ones(4, dtype=torch.int64), Phase.DECODE),
            context,
        )
        self.assertNotEqual(int(runtime.ops.error_status), 0)
        self.assertFalse(runtime.residency.execution_valid.any())
        runtime.cache_updates.access.release_after(
            lease, context.record_compute_completion(), context
        )

    def test_explicit_memory_and_settings(self):
        self.assertEqual(
            whole_layer_bank_layout(16, 64, 16), (("execution", 0, 16), ("hot", 0, 16))
        )
        self.assertEqual(
            whole_layer_bank_layout(16, 0, 32),
            (("execution", 0, 16), ("execution", 16, 16)),
        )
        for args in ((16, 64, 8), (16, 8, 16)):
            with self.assertRaises(ValueError):
                whole_layer_bank_layout(*args)
        base = dict(
            cache_ratio=0.5,
            execution_slots=4,
            host_pool_gib=1,
            workspace_limit_bytes=1024,
            metadata_limit_bytes=1024,
            max_dispatch_rows=16,
            prefill_mode="layer",
            decode_mode="layer",
        )
        self.assertEqual(NpuOffloadSettings(**base).prefill_mode, "layer")
        with self.assertRaises(ValueError):
            NpuOffloadSettings(**dict(base, decode_mode="invalid"))


class TestLayerPrefetch(unittest.TestCase):
    def make_pipeline(self, layers=(3, 5, 8, 9)):
        ctx = Context()
        ctx.phase, ctx.capturing = Phase.PREFILL, False
        access = SimpleNamespace()

        def stage(request, context):
            context.calls.append(("load", request.layer_id, request.bank_id))
            return SimpleNamespace(
                layer=request.layer_id,
                read_done=None,
                ticket=TransferTicket(
                    Fence(context.calls, f"loaded-{request.layer_id}")
                ),
            )

        def release(lease, done, context):
            if lease.read_done is not None:
                raise RuntimeError("Duplicate retirement")
            lease.read_done = done

        access.stage, access.release_after = stage, release
        p = LayerPrefetchPipeline(
            access,
            layers,
            make_request=lambda layer, bank, _: LayerExecutionRequest(
                layer, None, Phase.PREFILL, bank
            ),
            record_copy_completion=lambda c: TransferTicket(Fence(c.calls, "abort")),
        )
        return p, ctx

    def test_lookahead_and_per_bank_dependencies_without_poll_or_host_wait(self):
        p, ctx = self.make_pipeline()
        p.start(3, ctx)
        self.assertEqual(
            [x for x in ctx.calls if x[0] == "load"], [("load", 3, 0), ("load", 5, 1)]
        )
        for layer in p.layer_ids:
            if layer != 3:
                p.start(layer, ctx)
            lease = p.acquire(layer, ctx)
            done = TransferTicket(Fence(ctx.calls, f"reader-{layer}"))
            p.retire(layer, lease, done, ctx)
            p.finish_layer(layer, ctx)
        p.finish_forward(ctx)
        calls = ctx.calls
        self.assertLess(
            calls.index(("wait", "reader-3", "copy")), calls.index(("load", 8, 0))
        )
        self.assertLess(
            calls.index(("wait", "reader-5", "copy")), calls.index(("load", 9, 1))
        )
        self.assertFalse(any(x[0] == "wait" and x[-1] == "compute" for x in calls))
        # A new forward starts a fresh load sequence; no stale ready events.
        next_ctx = Context()
        next_ctx.phase, next_ctx.capturing = Phase.PREFILL, False
        p.start(3, next_ctx)
        self.assertEqual(len([x for x in next_ctx.calls if x[0] == "load"]), 2)
        p.abort(next_ctx)
        p.close()

    def test_wrong_order_incomplete_forward_and_partial_failure(self):
        p, ctx = self.make_pipeline()
        with self.assertRaises(RuntimeError):
            p.start(5, ctx)
        p.start(3, ctx)
        with self.assertRaises(RuntimeError):
            p.start(3, ctx)
        with self.assertRaises(RuntimeError):
            p.finish_forward(ctx)
        p.abort(ctx)
        with self.assertRaises(RuntimeError):
            p.start(3, ctx)
        p.close()
        p, ctx = self.make_pipeline()
        original = p.access.stage

        def fail(request, context):
            if request.layer_id == 5:
                raise RuntimeError("partial copy failure")
            return original(request, context)

        p.access.stage = fail
        with self.assertRaisesRegex(RuntimeError, "partial copy"):
            p.start(3, ctx)
        self.assertEqual(p.abort(ctx).completion.name, "abort")
        p.close()


class TestLayerLayout(unittest.TestCase):
    def test_layer_byte_copy_preserves_quantized_payloads_without_uint8_mask(self):
        layout = experts.NPUDispatchLayout(
            4, 256, 8, "cpu", torch.zeros((), dtype=torch.int32)
        )
        fill = torch.Tensor.masked_fill_

        def npu_fill(tensor, mask, value):
            if tensor.dtype == torch.uint8:
                raise RuntimeError("CANN masked-fill does not support uint8")
            return fill(tensor, mask, value)

        dtypes = [
            torch.int8,
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
        if hasattr(torch, "float8_e8m0fnu"):
            dtypes.append(torch.float8_e8m0fnu)
        for name, width in (("input_bytes", 256), ("scale_bytes", 32)):
            payload = (torch.arange(8 * width).reshape(8, width) * 17).to(torch.uint8)
            for dtype in dtypes:
                source = payload.view(dtype)
                for count in (0, 3, 8):
                    with patch.object(torch.Tensor, "masked_fill_", npu_fill):
                        result = layout._copy_layer_bytes(
                            source, torch.arange(8) >= count, name
                        )
                    expected = payload.clone()
                    expected[count:] = 0
                    self.assertTrue(torch.equal(result.view(torch.uint8), expected))
                    self.assertTrue(torch.equal(source.view(torch.uint8), payload))
                    self.assertEqual(result.data_ptr(), layout._owned[name].data_ptr())

    def test_whole_layer_prefix_empty_and_nan_padding(self):
        for phase in (Phase.PREFILL, Phase.DECODE):
            for counts in ([0] * 4, [0, 2, 1, 0], [2] * 4):
                layout = experts.NPUDispatchLayout(
                    4, 256, 16, "cpu", torch.zeros((), dtype=torch.int32)
                )
                used = sum(counts)
                x = torch.randn(12, 256).to(torch.bfloat16)
                x[used:] = float("nan")
                raw = SimpleNamespace(
                    hidden_states=x,
                    hidden_states_scale=None,
                    format="deepep_normal" if phase is Phase.PREFILL else "deepep_ll",
                    num_recv_tokens_per_expert=counts,
                    masked_m=torch.tensor(counts),
                )
                ctx = Context()
                ctx.phase, ctx.capturing = phase, phase is Phase.DECODE
                d = experts.NPUDispatch(raw, 256, object())
                with patch.object(
                    experts, "_compute_scope", return_value=nullcontext()
                ):
                    bound = layout.bind(d, ctx)
                    inputs = layout.select_layer(bound, ctx)
                    self.assertFalse(hasattr(inputs, "wave"))
                    self.assertEqual(
                        inputs.hidden_states.shape[0],
                        used if phase is Phase.PREFILL else 12,
                    )
                    self.assertFalse(inputs.hidden_states.isnan().any())
                    result = inputs.hidden_states * 2
                    layout.write_layer(bound, result, ctx)
                    out = layout.combine_input(bound, ctx).output
                    torch.testing.assert_close(out[:used], x[:used] * 2)
                    self.assertTrue((out[used:] == 0).all())


class TestLayerAssembly(unittest.TestCase):
    def test_dispatch_and_combine_once_with_real_materialization_and_runner_guards(
        self,
    ):
        # Only transport/device and FFN math are CPU doubles. Exercise the real
        # program, layout, cache access, runner masks and prefetch coordinator.
        a = Assembly(local_experts=4, slots=4, layer_count=4)
        a.backend.settings.prefill_mode = a.backend.settings.decode_mode = "layer"
        self.addCleanup(a.close)
        runtime = a.finalize()
        runtime.initialize_hot_cache([0, 1, 4, 5, 8, 9, 12, 13])
        owner = MoEOffloadOwner(a.config, a.catalog, runtime)
        counts = [0, 2, 1, 0]
        used, rows, width = sum(counts), 8, 256
        x = torch.ones(rows, width, dtype=torch.bfloat16)
        x[used:] = float("nan")
        calls = []

        class Core:
            def run(self, data, quant, _):
                calls.append("ffn")
                out = torch.full_like(data.hidden_states, float("nan"))
                start = 0
                for expert, count in enumerate(data.expert_tokens.tolist()):
                    out[start : start + count] = (
                        data.hidden_states[start : start + count]
                        * quant.w13_weight[expert, 0]
                    )
                    start += count
                return SimpleNamespace(hidden_states=out)

        class Dispatcher:
            def dispatch(self, hidden_states, routing, context):
                calls.append("dispatch")
                raw = SimpleNamespace(
                    hidden_states=hidden_states,
                    hidden_states_scale=None,
                    format="deepep_normal"
                    if context.phase is Phase.PREFILL
                    else "deepep_ll",
                    num_recv_tokens_per_expert=counts,
                    masked_m=torch.tensor(counts),
                )
                return experts.NPUDispatch(raw, width, object())

            def combine(self, layout, context):
                calls.append("combine")
                return layout.output

            def finish(self, value, context):
                return value

        layout = experts.NPUDispatchLayout(
            4, width, rows, "cpu", runtime.ops.error_status
        )
        runner = experts.NPUExpertRunner(
            SimpleNamespace(), _core=Core(), _input_type=SimpleNamespace
        )
        pointers = []
        for phase in (Phase.PREFILL, Phase.DECODE, Phase.PREFILL, Phase.DECODE):
            owner.switch_phase(phase)
            ctx = runtime._new_context(phase, capturing=phase is Phase.DECODE)
            runtime._context = ctx
            if phase is Phase.DECODE:
                runtime.ops.begin_step(ctx)
            pipeline = runtime.pipelines[0] if phase is Phase.PREFILL else None
            program = LayerExpertProgram(
                Dispatcher(),
                layout,
                runner,
                runtime.cache_updates.access,
                compute_scope=lambda _: nullcontext(),
                pipeline=pipeline,
            )
            with patch.object(experts, "_compute_scope", return_value=nullcontext()):
                for spec in a.catalog.offloaded_layers:
                    layer = spec.layer_id
                    if pipeline is not None:
                        program.layer_start(layer, ctx)
                    output = program.run(layer, x, None, ctx)
                    expected = torch.zeros_like(output)
                    expected[:2] = layer * 10 + 1
                    expected[2:3] = layer * 10 + 2
                    torch.testing.assert_close(output, expected)
                    pointers.append(layout._owned["input_bytes"].data_ptr())
            if pipeline is not None:
                pipeline.finish_forward(ctx)
            ctx.finish().resources_idle.completion.synchronize()
            ctx.dispose_completed(graph_destroyed=True)
            runtime._context = None
        self.assertEqual(calls, ["dispatch", "ffn", "combine"] * 16)
        self.assertEqual(len(set(pointers)), 1)
        owner.close()

    def test_prefill_real_weight_bundles_and_exact_cache_restore(self):
        a = Assembly(local_experts=4, slots=4, layer_count=4)
        a.backend.settings.prefill_mode = a.backend.settings.decode_mode = "layer"
        self.addCleanup(a.close)
        runtime = a.finalize()
        runtime.initialize_hot_cache([0, 1, 4, 5, 8, 9, 12, 13])
        hot = a.backend.format.banks[0]
        saved = {n: t.clone() for n, t in hot.tensors.items()}
        mapping = runtime.residency.key_to_slot.clone()
        runtime._decode_step_mod = 2
        owner = MoEOffloadOwner(a.config, a.catalog, runtime)
        owner.switch_phase(Phase.PREFILL)
        p = runtime.pipelines[0]
        for _ in range(2):
            ctx = runtime._new_context(Phase.PREFILL)
            runtime._context = ctx
            for layer in p.layer_ids:
                p.start(layer, ctx)
                lease = p.acquire(layer, ctx)
                b = p.access.wait_ready(lease, ctx.compute_stream, context=ctx)
                for tensor in b.bank.tensors.values():
                    self.assertEqual(
                        tensor[:, 0].tolist(), [layer * 10 + i for i in range(4)]
                    )
                p.retire(layer, lease, ctx.record_compute_completion(), ctx)
                p.finish_layer(layer, ctx)
            p.finish_forward(ctx)
            ctx.finish().resources_idle.completion.synchronize()
            ctx.dispose_completed()
            runtime._context = None
        owner.switch_phase(Phase.DECODE)
        for n, t in hot.tensors.items():
            torch.testing.assert_close(t, saved[n])
        torch.testing.assert_close(runtime.residency.key_to_slot, mapping)
        self.assertEqual(runtime._decode_step_mod, 2)
        self.assertEqual(len(a.backend.format.banks), 2)  # No extra weight allocations.
        owner.close()


if __name__ == "__main__":
    unittest.main()
