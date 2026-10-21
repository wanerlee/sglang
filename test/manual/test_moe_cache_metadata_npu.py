"""Standalone NPU metadata/capture check; no H2D expert pipeline is needed.

Run on an Ascend host with SGLang and torch_npu installed:
    python test/manual/test_moe_cache_metadata_npu.py

This validates bookkeeping and LFU only, NOT weight readiness or model outputs.
"""

import importlib.util
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe import (
    CacheMetadataAdapter,
    CacheUpdateCoordinator,
    ExpertCatalog,
    ExpertLayerSpec,
    PeriodicLFUConfig,
    PeriodicLFUPolicy,
    Phase,
)
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket


def make_cache(device):
    cat = ExpertCatalog(tuple(ExpertLayerSpec(i, 8, 2, 1) for i in (3, 4)))
    state = ResidencyState.allocate(8, 4, 2, device)
    state.key_to_slot.copy_(torch.tensor([0, -1, 1, -1, 2, -1, -1, 3], device=device))
    state.slot_to_key.copy_(torch.tensor([0, 2, 4, 7], device=device))
    state.slot_valid.fill_(True)
    policy = PeriodicLFUPolicy(PeriodicLFUConfig("unused.json"), cat)
    return CacheMetadataAdapter(cat, state, policy)


def run_step(cache, routes, valid, enabled):
    cache.begin_step(enabled)
    cache.record_local_routes(3, routes[0], valid[0])
    cache.record_local_routes(4, routes[1], valid[1])
    CacheUpdateCoordinator(
        cache.policy,
        cache.policy_state,
        cache.residency,
        cache.target_keys,
        SimpleNamespace(
            apply_update=lambda *args: TransferTicket(ImmediateCompletion())
        ),
    ).emit_step_end(SimpleNamespace(phase=Phase.DECODE, retain=lambda resource: None))


class TestNPUCacheMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch_npu") is None:
            raise unittest.SkipTest("torch_npu is not installed")
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise unittest.SkipTest("No Ascend device is available")
        torch.npu.set_device(0)

    def test_capture_replay_matches_cpu_with_changing_inputs(self):
        cpu, npu = make_cache("cpu"), make_cache("npu:0")
        routes = torch.tensor([[1, 1, 1], [1, 1, -1]], device="npu:0")
        valid = torch.ones_like(routes, dtype=torch.bool)
        enabled = torch.tensor(False, device="npu:0")
        addresses = [
            npu.policy_state.usage.data_ptr(),
            npu.residency.step_access_count.data_ptr(),
            npu.target_keys.data_ptr(),
        ]
        stream = torch.npu.Stream()
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            for _ in range(2):
                run_step(npu, routes, valid, enabled)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            run_step(npu, routes, valid, enabled)
        torch.npu.synchronize()
        self.assertEqual(npu.policy_state.step_mod.item(), 0)
        self.assertEqual(npu.policy_state.usage.sum().item(), 0)

        for step in range(10):
            host_routes = torch.tensor(
                [[1, 1, 1], [1, 1, -1]] if step < 5 else [[3, 3, 0], [2, 2, 0]]
            )
            host_valid = torch.ones_like(host_routes, dtype=torch.bool)
            host_valid[0, 2] = step % 2 == 0
            host_enabled = torch.tensor(step != 4)
            routes.copy_(host_routes)
            valid.copy_(host_valid)
            enabled.copy_(host_enabled)
            graph.replay()
            torch.npu.synchronize()
            run_step(cpu, host_routes, host_valid, host_enabled)
            for name in ("usage", "step_mod", "update_due"):
                torch.testing.assert_close(
                    getattr(npu.policy_state, name).cpu(),
                    getattr(cpu.policy_state, name),
                )
            torch.testing.assert_close(npu.target_keys.cpu(), cpu.target_keys)
            for name in ("accesses", "hits", "misses"):
                torch.testing.assert_close(
                    getattr(npu.stats, name).cpu(), getattr(cpu.stats, name)
                )
        self.assertEqual(
            addresses,
            [
                npu.policy_state.usage.data_ptr(),
                npu.residency.step_access_count.data_ptr(),
                npu.target_keys.data_ptr(),
            ],
        )


if __name__ == "__main__":
    unittest.main()
