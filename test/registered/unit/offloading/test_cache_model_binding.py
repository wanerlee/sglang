"""Exercise production model binding/Ascend adapters with CPU kernel/EP doubles."""

import ast
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import test_cache_runtime as fixtures
import torch

from sglang.srt.offloading.moe import (
    CacheBackend,
    CacheBackendBinding,
    ExecutionReservation,
    ExpertWeightAccess,
    MoEOffloadConfig,
    MoEOffloaderV2,
    PeriodicLFUConfig,
    Phase,
)
from sglang.srt.offloading.moe.ascend_adapter import (
    AscendDispatchLayout,
    ExpertBankBindings,
)
from sglang.srt.offloading.moe.interfaces import RuntimeCapabilities
from sglang.srt.offloading.moe.resources import Allocation, ResourceKind
from sglang.srt.offloading.transfer import (
    HostCopyPlan,
    HostCopyRegion,
    ImmediateCompletion,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Format:
    def __init__(self, normal=False):
        self.normal = normal

    def is_deepep_ll(self):
        return not self.normal

    def is_deepep_normal(self):
        return self.normal


@dataclass
class Combine:
    hidden_states: object
    topk_ids: object
    topk_weights: object


def native_types():
    """Use AscendRunnerCore.run from source; replace only hardware primitives.

    Importing the full module on CPU requires unrelated accelerator packages.
    Keeping the actual run body tests that our bindings reach both projections.
    """
    root = Path(__file__).resolve().parents[4]
    source = root / "python/sglang/srt/layers/moe/moe_runner/ascend.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"AscendRunnerInput", "AscendRunnerOutput", "AscendQuantInfo"}
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    core = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AscendRunnerCore"
    )
    run = next(
        n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    module = ModuleType("sglang.srt.layers.moe.moe_runner.ascend")
    module.__dict__.update(
        torch=torch,
        dataclass=dataclass,
        Optional=__import__("typing").Optional,
        Any=object,
        MoeQuantInfo=object,
        RunnerInput=object,
        RunnerOutput=object,
        MoeRunnerBackend=SimpleNamespace(ASCEND="ascend"),
        _uses_fused_gmm1=lambda *args: False,
    )
    for name in (
        "NPUSwigluDeepEPKernel",
        "NPUSitu",
        "NPUSituMXFP8Quant",
        "NPUSwigluMxfp8Quant",
    ):
        module.__dict__[name] = type(name, (), {})
    exec(
        compile(ast.Module(body=classes + [run], type_ignores=[]), str(source), "exec"),
        module.__dict__,
    )

    class Core:
        def __init__(self, config):
            self.config = config
            self.activation = SimpleNamespace(
                _apply_activation=lambda x: (x.relu(), None)
            )

        run = module.run

    module.AscendRunnerCore = Core
    return module


class Kernel:
    def apply(self, quant, x, counts, *, weight_prefix, **kwargs):
        # Test double for NPU grouped matmul, preserving native quant-info ABI.
        weights = getattr(quant, weight_prefix + "_weight")
        rows = torch.repeat_interleave(torch.arange(len(counts)), counts)
        return torch.bmm(x[:, None, :], weights[rows]).squeeze(1)


class Dispatcher:
    def __init__(self):
        self.calls = []

    def dispatch(self, hidden_states, topk_output):
        x, routing = hidden_states, topk_output
        self.calls.append("D")
        self.rows = routing.rows
        self.weights = routing.weights
        return SimpleNamespace(
            hidden_states=x[routing.rows],
            hidden_states_scale=None,
            topk_ids=routing.rows,
            topk_weights=routing.weights,
            masked_m=routing.counts,
            num_recv_tokens_per_expert=routing.counts.tolist(),
            format=Format(routing.normal),
        )

    def combine(self, combine_input):
        received = combine_input
        self.calls.append("C")
        result = received.hidden_states.new_zeros((3, 2))
        # Weight exactly once, in native combine, including repeated assignments.
        result.index_add_(0, self.rows, received.hidden_states * self.weights[:, None])
        return result


def wrapped_dispatcher(inner):
    """Use the actual keyword-only MaybeTbo dispatch/combine forwarding methods."""
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/batch_overlap/two_batch_overlap.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MaybeTboDeepEPDispatcher"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_execute", "dispatch", "combine")
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future] + methods, type_ignores=[])
            ),
            str(source),
            "exec",
        ),
        namespace,
    )
    wrapper = type(
        "TestMaybeTboDispatcher",
        (),
        {name: namespace[name] for name in ("_execute", "dispatch", "combine")},
    )()
    wrapper._inners = [inner]
    wrapper.calls = inner.calls
    return wrapper


class Experts(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.moe_tp_size, self.moe_ep_size, self.moe_ep_rank = 1, 2, 1
        self._num_global_routed, self._num_local_routed = 8, 4
        self.reduce_results = False
        self.dispatcher = wrapped_dispatcher(Dispatcher())
        self.w13_kernel = self.w2_kernel = Kernel()
        self.moe_runner_config = SimpleNamespace(layer=self, hidden_size=2)

    def forward(self, hidden_states, topk_output, pre_quant_input=None):
        return hidden_states + 100


class Decoder(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.mlp = torch.nn.Module()
        self.mlp.is_deepseek_v4 = True
        self.mlp.layer_id = layer_id
        self.mlp.experts = Experts(layer_id)

    def forward(self, hidden, routing):
        # Enclosing model's scaling/shared branch must remain outside replacement.
        return self.mlp.experts(hidden, routing) * 2 + hidden

    def forward_hc_pre_from_prev(self, hidden, routing):
        return self.mlp.experts(hidden, routing) * 2 + hidden


class Transport:
    supports_device_plans = True  # CPU test double, not a hardware capability claim.

    def __init__(self, host, hot):
        self.host, self.hot = host, hot

    def materialize_into(self, plan, bindings, stream):
        def copy():
            for i in range(plan.capacity):
                if plan.valid[i]:
                    source = self.host if plan.source_kind[i] == 1 else self.hot
                    for name, destination in bindings.components.items():
                        destination[plan.destination_slot[i]].copy_(
                            source[name][plan.source_index[i]]
                        )

        return TransferTicket(fixtures.DeferredCopy(copy, []), (plan, bindings))

    def materialize_host_into(self, plan, bindings, stream):
        def copy():
            for region in plan.regions:
                region.destination.copy_(region.source)

        return TransferTicket(fixtures.DeferredCopy(copy, []), (plan, bindings))


class HostResidency:
    """Physical host-plan provider double; native bank ABI is production code."""

    def __init__(self, host, hot):
        self.host, self.hot = host, hot
        self.reader = ImmediateCompletion()

    def resolve_execution(self, request, context):
        wave = request.wave
        regions = []
        for name, tensor in self.host.items():
            src = tensor[wave.local_start : wave.local_end]
            dst = self.hot[name][: wave.local_end - wave.local_start]
            size = src.numel() * src.element_size()
            regions.append(HostCopyRegion(src, dst, 0, 0, size, size, size))
        return ExecutionReservation(
            HostCopyPlan(tuple(regions)),
            ExpertBankBindings(self.hot, (0,)),
            copy_dependencies=(self.reader,),
        )

    def publish_execution(self, reservation, copied, context):
        copied.completion.wait_on(context.compute_stream)
        return context.record_compute_completion()

    def finish_execution(self, reservation, read_done, context):
        self.reader = read_done.completion


class TestCacheModelBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        profile = Path(self.temp.name) / "profile.json"
        profile.write_text(
            json.dumps({"model_id": "toy", "layers": {"3": {"4": 10, "6": 9}}})
        )
        self.device = fixtures.TestDevice()
        self.compute, self.copy = (
            fixtures.TestStream("compute"),
            fixtures.TestStream("copy"),
        )
        self.host = {
            name: torch.stack([torch.eye(2) * (i + 1) for i in range(4)])
            for name in ("w13_weight", "w2_weight")
        }
        self.hot = {name: torch.zeros(2, 2, 2) for name in self.host}
        self.bank = {name: torch.zeros(2, 2, 2) for name in self.host}
        from test_cache_integration import SequentialPipeline

        transport = Transport(self.host, self.hot)
        self.pipeline = SequentialPipeline(
            ExpertWeightAccess(HostResidency(self.host, self.hot), transport)
        )
        self.prepared, self.closed = [], []
        resources = []
        for pool, kind in (
            (self.host, ResourceKind.HOST_WEIGHTS),
            (self.hot, ResourceKind.HOT_WEIGHTS),
            (self.bank, ResourceKind.EXECUTION_WEIGHTS),
        ):
            for name, tensor in pool.items():
                resources.append(
                    Allocation(
                        f"{kind.value}.{name}",
                        kind,
                        tensor.untyped_storage().nbytes(),
                        (str(tensor.device), tensor.data_ptr()),
                        tensor,
                    )
                )
        binding = CacheBackendBinding(
            self.device,
            "cpu",
            self.compute,
            self.copy,
            transport,
            ExpertBankBindings(self.hot, (0,)),
            ExpertBankBindings(self.bank, (0,)),
            self.pipeline,
            lambda: [t.fill_(-999) for t in self.hot.values()],
            tuple(resources),
            32,
            2,
            8,
            2,
            torch.float32,
            RuntimeCapabilities(True, True, True),
        )
        self.backend = CacheBackend(
            model_id="toy",
            prepare_weights=lambda m, s: self.prepared.append(s.layer_id),
            bind_weights=lambda *args: binding,
            close_weights=lambda: self.closed.append(True),
        )
        self.offloader = MoEOffloaderV2(
            MoEOffloadConfig(
                0.5, 2, 2, 1, workspace_limit_bytes=4096, metadata_limit_bytes=1_000_000
            ),
            self.backend,
            periodic_lfu_config=PeriodicLFUConfig(str(profile)),
        )
        self.resident, self.decoder = Decoder(0), Decoder(3)
        self.offloader.wrap_modules(iter((self.resident, self.decoder)))
        native = native_types()
        deepep = ModuleType("sglang.srt.layers.moe.token_dispatcher.deepep")
        deepep.DeepEPLLCombineInput = deepep.DeepEPNormalCombineInput = Combine
        deepep.DeepEPDispatcher = Dispatcher
        for name, module in ((native.__name__, native), (deepep.__name__, deepep)):
            previous = sys.modules.get(name)
            sys.modules[name] = module

            def restore(name=name, previous=previous):
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

            self.addCleanup(restore)
        self.offloader.post_init()
        self.runner = SimpleNamespace(
            model=SimpleNamespace(), capture_head_hooks=[], capture_tail_hooks=[]
        )
        self.offloader.bind_model_runner(self.runner)
        self.addCleanup(self.offloader.close)
        self.runtime = self.offloader.owner.runtime

    def routing(self, counts=(0, 3, 1, 0), normal=False):
        return SimpleNamespace(
            counts=torch.tensor(counts),
            rows=torch.tensor([0, 0, 1, 2]),
            weights=torch.tensor([0.2, 0.3, 0.4, 0.5]),
            normal=normal,
        )

    def test_real_module_call_dispatch_once_lfu_and_restore(self):
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        routing = self.routing()
        batch = SimpleNamespace(
            forward_mode=fixtures.Mode("DECODE"), input_ids=torch.arange(3)
        )
        for _ in range(4):
            with (
                self.offloader.forward_scope(batch),
                self.offloader.eager_step_scope(batch),
            ):
                actual = self.decoder(hidden, routing)
            # E1 three assignments (two for token 0), E2 one assignment.
            expected = hidden * torch.tensor([2.0, 1.6, 4.5])[:, None] * 2 + hidden
            torch.testing.assert_close(actual, expected)
        self.assertEqual(self.prepared, [3])
        self.assertEqual(self.decoder.mlp.experts.dispatcher.calls, ["D", "C"] * 4)
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [1, 2])
        torch.testing.assert_close(
            self.runtime.cache.policy_state.usage,
            torch.tensor([0.0, 12.0, 4.0, 0.0]) * 0.7,
        )
        # Resident layers were never redirected; close restores original forward.
        torch.testing.assert_close(
            self.resident(hidden, routing), (hidden + 100) * 2 + hidden
        )
        self.offloader.close()
        torch.testing.assert_close(
            self.decoder(hidden, routing), (hidden + 100) * 2 + hidden
        )
        torch.testing.assert_close(
            self.decoder.forward_hc_pre_from_prev(hidden, routing),
            (hidden + 100) * 2 + hidden,
        )
        self.assertEqual(self.closed, [True])
        self.assertEqual(self.runner.capture_head_hooks, [])

    def test_prefill_layer_start_precedes_router_and_uses_prefill_program(self):
        calls = []
        self.runtime.context.phase = Phase.PREFILL
        self.runtime.prefill_program = SimpleNamespace(
            layer_start=lambda layer_id, ctx: calls.append("start"),
            run=lambda *args: calls.append("prefill") or args[1],
        )
        x = torch.ones(3, 2)
        torch.testing.assert_close(self.decoder(x, self.routing()), 3 * x)
        self.assertEqual(calls, ["start", "prefill"])

    def test_padding_empty_wave_and_scale_rows(self):
        layout_adapter = AscendDispatchLayout(4, 8, 2, torch.float32, "cpu")
        dispatched = SimpleNamespace(
            hidden_states=torch.arange(12.0).reshape(6, 2),
            hidden_states_scale=torch.arange(6.0)[:, None],
            masked_m=torch.tensor([0, 2, 1, 0]),
            format=Format(),
        )
        layout = layout_adapter.bind(dispatched, self.runtime.context)
        from sglang.srt.offloading.moe.config import fixed_waves

        for wave in fixed_waves(4, 1):
            selected = layout_adapter.select_wave(layout, wave, self.runtime.context)
            self.assertEqual(int(selected.expert_tokens.sum()), 6)
            torch.testing.assert_close(
                selected.hidden_states_scale[selected.valid],
                dispatched.hidden_states_scale[selected.row_order[selected.valid]],
            )
            layout_adapter.write_wave(
                layout, wave, selected.hidden_states + 7, self.runtime.context
            )
        torch.testing.assert_close(layout.output[:3], dispatched.hidden_states[:3] + 7)
        self.assertEqual(layout.output[3:].count_nonzero(), 0)

    def test_normal_counts_are_not_accepted_in_decode(self):
        layout = AscendDispatchLayout(4, 8, 2, torch.float32, "cpu")
        received = self.decoder.mlp.experts.dispatcher.dispatch(
            hidden_states=torch.ones(3, 2), topk_output=self.routing(normal=True)
        )
        with self.assertRaisesRegex(ValueError, "prefill-only"):
            layout.bind(received, self.runtime.context)
        self.runtime.context.phase = Phase.PREFILL
        bound = layout.bind(received, self.runtime.context)
        self.assertEqual(bound.routes.expert_ids.tolist(), [1, 1, 1, 2])

    def test_decode_prefill_decode_through_installed_programs(self):
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected = hidden * torch.tensor([2.0, 1.6, 4.5])[:, None] * 2 + hidden
        for mode in ("DECODE", "EXTEND", "DECODE"):
            batch = SimpleNamespace(
                forward_mode=fixtures.Mode(mode), input_ids=torch.arange(3)
            )
            with (
                self.offloader.forward_scope(batch),
                self.offloader.eager_step_scope(batch),
            ):
                result = self.decoder.forward_hc_pre_from_prev(
                    hidden=hidden, routing=self.routing(normal=mode == "EXTEND")
                )
            torch.testing.assert_close(result, expected)
        self.assertEqual(
            self.pipeline.calls, ["start", "seal", "acquire", "acquire", "finish"]
        )
        self.assertEqual(self.decoder.mlp.experts.dispatcher.calls, ["D", "C"] * 3)
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [0, 2])
        self.assertEqual(int(self.runtime.cache.policy_state.step_mod), 2)
        torch.testing.assert_close(
            self.runtime.cache.policy_state.usage, torch.tensor([0.0, 6.0, 2.0, 0.0])
        )
        for name in self.host:
            torch.testing.assert_close(self.hot[name], self.host[name][[0, 2]])

    def test_bank_views_use_physical_bank_offset(self):
        from sglang.srt.offloading.moe.config import ExpertWave

        bank = ExpertBankBindings(self.host, (0, 2))
        views = bank.wave_components(ExpertWave(1, 2, 4, 1))
        self.assertEqual(
            views["w13_weight"].data_ptr(), self.host["w13_weight"][2:].data_ptr()
        )

    def test_changing_receive_counts_and_padding_without_host_decisions(self):
        from unittest.mock import patch

        from sglang.srt.offloading.moe.config import fixed_waves

        adapter = AscendDispatchLayout(4, 8, 2, torch.float32, "cpu")
        address = adapter.output.data_ptr()
        for counts in ([2, 0, 0, 1], [0, 0, 0, 0], [0, 1, 3, 1]):
            received = SimpleNamespace(
                hidden_states=torch.ones(8, 2),
                hidden_states_scale=None,
                masked_m=torch.tensor(counts),
                format=Format(),
            )
            with (
                patch.object(
                    torch.Tensor, "item", side_effect=AssertionError("host read")
                ),
                patch.object(
                    torch.Tensor, "tolist", side_effect=AssertionError("host read")
                ),
                patch.object(
                    torch.Tensor, "cpu", side_effect=AssertionError("host read")
                ),
            ):
                layout = adapter.bind(received, self.runtime.context)
                for wave in fixed_waves(4, 2):
                    selected = adapter.select_wave(layout, wave, self.runtime.context)
                    adapter.write_wave(
                        layout, wave, selected.hidden_states * 3, self.runtime.context
                    )
            torch.testing.assert_close(
                layout.output[: sum(counts)], torch.full((sum(counts), 2), 3.0)
            )
            self.assertEqual(int(layout.output[sum(counts) :].count_nonzero()), 0)
            self.assertEqual(adapter.output.data_ptr(), address)

    def test_native_reader_wait_precedes_the_next_wave_copy(self):
        batch = SimpleNamespace(
            forward_mode=fixtures.Mode("DECODE"), input_ids=torch.arange(3)
        )
        calls = []
        native = self.runtime.decode_program.programs[3].local_program.run_experts
        from sglang.srt.offloading.moe import LocalExecution

        def delayed_reader(*args):
            result = native(*args)
            return LocalExecution(
                result.output,
                TransferTicket(
                    fixtures.DeferredCopy(lambda: calls.append("read_done"), [])
                ),
            )

        self.runtime.decode_program.programs[
            3
        ].local_program.run_experts = delayed_reader
        transfer = self.runtime.transfer
        original = transfer.materialize_into

        def checked_copy(*args):
            calls.append("copy")
            return original(*args)

        transfer.materialize_into = checked_copy
        with (
            self.offloader.forward_scope(batch),
            self.offloader.eager_step_scope(batch),
        ):
            self.decoder(torch.ones(3, 2), self.routing())
        self.assertEqual(calls, ["copy", "read_done", "copy", "read_done"])

    def test_unused_bank_lanes_do_not_pollute_output_or_lfu(self):
        # Empty waves may read padding lanes that have never held an expert.
        # NaNs make accidental accumulation or demand for those lanes visible.
        for tensor in self.bank.values():
            tensor.fill_(float("nan"))
        batch = SimpleNamespace(
            forward_mode=fixtures.Mode("DECODE"), input_ids=torch.arange(3)
        )
        hidden = torch.ones(3, 2)
        with (
            self.offloader.forward_scope(batch),
            self.offloader.eager_step_scope(batch),
        ):
            output = self.decoder(hidden, self.routing(counts=(4, 0, 0, 0)))
        expected = hidden * torch.tensor([0.5, 0.4, 0.5])[:, None] * 2 + hidden
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(
            self.runtime.cache.policy_state.usage, torch.tensor([4.0, 0, 0, 0])
        )
        for tensor in self.bank.values():
            self.assertTrue(torch.isnan(tensor[1]).all())

    def test_zero_receive_rows_preserve_native_output_geometry(self):
        from sglang.srt.offloading.moe.config import fixed_waves

        # Packed input width/dtype need not equal the native F output geometry.
        adapter = AscendDispatchLayout(4, 8, 2, torch.bfloat16, "cpu")
        received = SimpleNamespace(
            hidden_states=torch.empty((0, 1), dtype=torch.uint8),
            hidden_states_scale=None,
            masked_m=torch.zeros(4, dtype=torch.int64),
            format=Format(),
        )
        layout = adapter.bind(received, self.runtime.context)
        native = self.runtime.decode_program.programs[3].local_program.run_experts
        for wave in fixed_waves(4, 2):
            selected = adapter.select_wave(layout, wave, self.runtime.context)
            result = native(
                self.decoder.mlp.experts,
                selected,
                ExpertBankBindings(self.bank, (0,)),
                self.runtime.context,
            )
            self.assertEqual(result.output.shape, (0, 2))
            self.assertEqual(result.output.dtype, torch.bfloat16)
            adapter.write_wave(layout, wave, result.output, self.runtime.context)
        self.assertEqual(layout.output.shape, (0, 2))

    def test_partial_model_install_can_be_closed_and_reinstalled(self):
        from unittest.mock import patch

        from sglang.srt.offloading.moe.model_binding import DeepSeekV4ProgramBinding

        decoder = Decoder(3)
        spec = self.runtime.cache.catalog.offloaded_layers[0]
        binding = DeepSeekV4ProgramBinding(((decoder, spec),))
        self.addCleanup(binding.close)
        hidden = torch.ones(3, 2)
        original = decoder(hidden, self.routing())
        with patch.object(
            decoder,
            "register_forward_pre_hook",
            side_effect=RuntimeError("hook failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "hook failed"):
                binding.install(self.runtime)
        binding.close()
        torch.testing.assert_close(decoder(hidden, self.routing()), original)
        binding.install(self.runtime)
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            binding.install(self.runtime)
        binding.close()
        torch.testing.assert_close(
            decoder.forward_hc_pre_from_prev(hidden, self.routing()), original
        )


if __name__ == "__main__":
    unittest.main()
