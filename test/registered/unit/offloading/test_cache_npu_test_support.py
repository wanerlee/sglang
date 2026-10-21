"""CPU checks of the NPU test's oracle and toy compute, NOT NPU validation."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe import (
    CacheLocalExpertProgram,
    ExpertCatalog,
    ExpertLayerSpec,
    LocalRoutes,
)
from sglang.srt.offloading.moe.config import fixed_waves
from sglang.srt.offloading.moe.programs import ComposedDFCProgram
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def load_manual_test():
    path = (
        Path(__file__).resolve().parents[4]
        / "test/manual/test_moe_cache_runtime_npu.py"
    )
    name = "_moe_npu_test_support"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manual = load_manual_test()


class TestNPUTestSupport(unittest.TestCase):
    def test_reference_counts_duplicate_choices_and_ignores_padding(self):
        weights = manual.make_weights()
        hidden = torch.arange(2 * manual.DIM).reshape(2, manual.DIM).float() / 32
        routes = manual.padded_routes([1, 1, 2], 2)
        actual = manual.cpu_reference(hidden, routes, weights, 4)
        expected = torch.stack(
            (
                torch.relu(hidden[0] @ weights[5, 0]) @ weights[5, 1],
                torch.relu(hidden[1] @ weights[6, 0]) @ weights[6, 1] * 0.5,
            )
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(routes.expert_ids.tolist(), [[1, 1], [2, -1]])
        self.assertFalse(manual.EagerNPUTransfer(None).supports_device_plans)

    def test_toy_bank_compute_matches_oracle_including_empty_waves(self):
        # Exercise exactly the manual test's native_experts function on CPU.
        # Only loading is replaced; no hardware or asynchronous claims are made.
        fixture = manual.TestNPUCacheRuntime("test_real_h2d_d2d_lfu_and_next_step_hits")
        fixture.catalog = ExpertCatalog(
            tuple(ExpertLayerSpec(i, 8, 2, 1) for i in (3, 4))
        )
        fixture.execution = torch.full((2, 2, manual.DIM, manual.DIM), torch.nan)
        fixture.compute_audits = []
        state = ResidencyState.allocate(8, 4, 2, "cpu")
        weights = manual.make_weights()

        def prepare(layer_id, routes, wave):
            state.execution_key.fill_(-1)
            state.execution_valid.zero_()
            offset = fixture.catalog.key(layer_id, 0)
            for local in routes.expert_ids[routes.valid].unique().tolist():
                lane = local - wave.local_start
                fixture.execution[lane].copy_(weights[offset + local])
                state.execution_key[lane] = offset + local
                state.execution_valid[lane] = True

        fixture.runtime = SimpleNamespace(
            residency=state, prepare_local_experts=prepare
        )
        program = ComposedDFCProgram(
            fixed_waves(4, 2),
            manual.ToyDispatcher(),
            CacheLocalExpertProgram(
                fixture.runtime,
                lambda layer, output, wave: output.routes,
                fixture.native_experts,
            ),
        )
        for assignments in ([0, 1, 1, 3], [1, 1], []):
            hidden = torch.arange(2 * manual.DIM).reshape(2, manual.DIM).float() / 32
            routes = manual.padded_routes(assignments, 2)
            for layer in fixture.catalog.layers:
                with self.subTest(assignments=assignments, layer=layer.layer_id):
                    fixture.execution.fill_(torch.nan)
                    actual = program.run(
                        layer,
                        hidden,
                        LocalRoutes(routes.expert_ids, routes.valid),
                        None,
                    )
                    expected = manual.cpu_reference(
                        hidden, routes, weights, fixture.catalog.key(layer.layer_id, 0)
                    )
                    torch.testing.assert_close(actual, expected)
                    self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
