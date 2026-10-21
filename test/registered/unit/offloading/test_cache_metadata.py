"""Cache bookkeeping tests with real tensors and no transfer/backend mocks."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

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
from sglang.srt.offloading.moe.resources import AllocationLedger, ResourceKind
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def finish_cache(cache, *, adjust=True):
    # Exercise the public coordinator, isolating metadata from physical copies.
    access = SimpleNamespace(
        apply_update=lambda *args: TransferTicket(ImmediateCompletion())
    )
    context = SimpleNamespace(phase=Phase.DECODE, retain=lambda resource: None)
    return CacheUpdateCoordinator(
        cache.policy, cache.policy_state, cache.residency, cache.target_keys, access
    ).emit_step_end(context, adjust=adjust)


def adapter(capacity=4):
    cat = ExpertCatalog(tuple(ExpertLayerSpec(i, 8, 2, 1, i < 3) for i in range(5)))
    state = ResidencyState.allocate(8, capacity, 2, "cpu")
    if capacity:
        assert capacity == 4
        keys = torch.tensor([0, 2, 4, 7])
        state.slot_to_key.copy_(keys)
        state.key_to_slot[keys] = torch.arange(4)
        state.slot_valid.fill_(True)
    return CacheMetadataAdapter(
        cat, state, PeriodicLFUPolicy(PeriodicLFUConfig("unused.json"), cat)
    )


class TestRouteAccounting(unittest.TestCase):
    def test_disabled_statistics_still_returns_actual_weight_demand(self):
        cache = adapter()
        cache.begin_step(torch.tensor(False))
        counts = cache.record_local_routes(
            3,
            torch.tensor([0, 1, 1, 3, -1, 8]),
            torch.tensor([True] * 6),
            local_start=0,
            local_end=2,
        )
        self.assertEqual(counts.tolist(), [1, 2, 0, 0])
        self.assertEqual(cache.residency.step_access_count.tolist(), [0] * 8)
        self.assertEqual(int(cache.stats.accesses), 0)

    def test_batch_counts_hits_misses_and_padding(self):
        cache = adapter()
        cache.begin_step(torch.tensor(True))
        # E5 appears three times, but only two assignments are valid.
        cache.record_local_routes(
            3,
            torch.tensor([[0, 1], [1, 2], [1, -1], [4, 3]]),
            torch.tensor([[1, 1], [1, 1], [0, 1], [1, 1]], dtype=torch.bool),
        )
        self.assertEqual(
            cache.residency.step_access_count.tolist(), [1, 2, 1, 1, 0, 0, 0, 0]
        )
        self.assertEqual(
            (
                cache.stats.accesses.item(),
                cache.stats.hits.item(),
                cache.stats.misses.item(),
            ),
            (5, 2, 3),
        )

    def test_local_ids_are_counted_separately_per_layer(self):
        cache = adapter()
        cache.begin_step(torch.tensor(True))
        ids = torch.tensor([-4, -1, 0, 1, 3, 4, -5])
        valid = torch.ones_like(ids, dtype=torch.bool)
        cache.record_local_routes(3, ids, valid)
        cache.record_local_routes(4, ids, valid)
        self.assertEqual(
            cache.residency.step_access_count.tolist(), [1, 1, 0, 1, 1, 1, 0, 1]
        )

    def test_disjoint_wave_ranges_equal_single_pass(self):
        whole, waves = adapter(), adapter()
        ids = torch.tensor([3, 0, 1, 1, 2, -1, 4])
        valid = torch.ones_like(ids, dtype=torch.bool)
        whole.begin_step(torch.tensor(True))
        waves.begin_step(torch.tensor(True))
        whole.record_local_routes(3, ids, valid)
        for start, end in ((0, 2), (2, 4)):
            waves.record_local_routes(3, ids, valid, local_start=start, local_end=end)
        torch.testing.assert_close(
            whole.residency.step_access_count, waves.residency.step_access_count
        )
        for name in ("accesses", "hits", "misses"):
            torch.testing.assert_close(
                getattr(whole.stats, name), getattr(waves.stats, name)
            )
        self.assertEqual(waves.policy_state.step_mod.item(), 0)

    def test_invalid_slot_and_zero_cache_are_misses(self):
        for cache in (adapter(), adapter(capacity=0)):
            if cache.residency.slot_valid.numel():
                cache.residency.slot_valid[0] = False
                cache.residency.slot_to_key[1] = 7  # Mismatched reverse map.
            cache.begin_step(torch.tensor(True))
            cache.record_local_routes(
                3, torch.tensor([0, 2]), torch.tensor([True, True])
            )
            self.assertEqual(cache.stats.hits.item(), 0)
            self.assertEqual(cache.stats.misses.item(), 2)

    def test_disabled_and_empty_routes(self):
        cache = adapter()
        cache.begin_step(torch.tensor(False))
        cache.record_local_routes(3, torch.tensor([1, 1]), torch.tensor([True, True]))
        finish_cache(cache)
        self.assertEqual(cache.policy_state.usage.sum().item(), 0)
        self.assertEqual(cache.stats.accesses.item(), 0)
        self.assertEqual(cache.policy_state.step_mod.item(), 0)
        self.assertEqual(cache.target_keys.tolist(), [0, 2, 4, 7])
        # Empty local demand is still a valid global decode step.
        for _ in range(4):
            cache.begin_step(torch.tensor(True))
            cache.record_local_routes(
                3, torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.bool)
            )
            finish_cache(cache)
        self.assertEqual(cache.policy_state.step_mod.item(), 0)
        self.assertEqual(cache.stats.accesses.item(), 0)

    def test_static_input_contracts(self):
        cache = adapter()
        ids, valid = torch.tensor([0, 1]), torch.tensor([True, True])
        with self.assertRaises(ValueError):
            cache.record_local_routes(3, ids, valid, local_end=5)
        with self.assertRaises(ValueError):
            cache.record_local_routes(3, ids, valid[:1])
        with self.assertRaises(KeyError):
            cache.record_local_routes(0, ids, valid)
        with self.assertRaises(TypeError):
            cache.record_local_routes(3, ids.float(), valid)
        with self.assertRaises(TypeError):
            cache.record_local_routes(3, ids, valid.long())


class TestCacheTargets(unittest.TestCase):
    def test_skipped_selection_only_accumulates_statistics(self):
        cache = adapter()
        cache.target_keys.fill_(-1)
        with patch.object(
            cache.policy, "select_hot_set", wraps=cache.policy.select_hot_set
        ) as select:
            for _ in range(3):
                cache.begin_step(torch.tensor(True))
                cache.record_local_routes(
                    3, torch.tensor([1, 1]), torch.tensor([True, True])
                )
                finish_cache(cache, adjust=False)
            select.assert_not_called()
        self.assertEqual(cache.target_keys.tolist(), [-1] * 4)
        self.assertEqual(cache.policy_state.step_mod.item(), 3)
        self.assertEqual(cache.policy_state.usage[1].item(), 6)
        cache.begin_step(torch.tensor(True))
        finish_cache(cache)
        self.assertEqual(cache.target_keys.tolist(), [1, 2, 4, 7])
        self.assertAlmostEqual(cache.policy_state.usage[1].item(), 4.2, places=5)

    def test_complete_step_sequence_matches_document(self):
        cache = adapter()
        # Four forwards accumulate the document's example, including all misses.
        ids_l3 = ([1, 1, 0], [1, 2, 2], [1, 1, 3, 2], [1, 1, 1, 0, 2, 2, 3, 3])
        ids_l4 = ([1, 1, 3], [1, 3, 2], [1, 1, 3], [1, 1, 0, 2, 3])
        before = cache.residency.key_to_slot.clone()
        for i in range(4):
            cache.begin_step(torch.tensor(True))
            for layer, routes in ((3, ids_l3[i]), (4, ids_l4[i])):
                cache.record_local_routes(
                    layer,
                    torch.tensor(routes),
                    torch.ones(len(routes), dtype=torch.bool),
                )
            finish_cache(cache)
            if i < 3:
                self.assertEqual(cache.target_keys.tolist(), [0, 2, 4, 7])
            torch.testing.assert_close(cache.residency.key_to_slot, before)
        self.assertEqual(cache.target_keys.tolist(), [1, 2, 5, 7])
        torch.testing.assert_close(
            cache.policy_state.usage, torch.tensor([2, 8, 5, 3, 1, 7, 2, 4]) * 0.7
        )


class TestMetadataSnapshot(unittest.TestCase):
    def test_restore_is_exact_in_place_and_invalidates_execution(self):
        cache = adapter()
        state = cache.residency
        state.slot_generation.copy_(torch.tensor([3, 5, 7, 9]))
        cache.policy_state.usage.copy_(torch.arange(8).float() * 0.7)
        cache.policy_state.step_mod.fill_(2)
        original = {
            name: getattr(state, name).clone()
            for name in ("key_to_slot", "slot_to_key", "slot_valid", "slot_generation")
        }
        pointers = [getattr(state, name).data_ptr() for name in original]
        usage = cache.policy_state.usage.clone()
        cache.save_decode_metadata()
        for name in original:
            getattr(state, name).zero_()
        cache.policy_state.usage.zero_()
        cache.policy_state.step_mod.zero_()
        state.execution_key.copy_(torch.tensor([6, 7]))
        state.execution_valid.fill_(True)
        # Simulates the lifecycle point AFTER the backend has restored weights.
        cache.restore_decode_metadata_after_refill()
        for name, value in original.items():
            torch.testing.assert_close(getattr(state, name), value)
        self.assertEqual(
            pointers, [getattr(state, name).data_ptr() for name in original]
        )
        torch.testing.assert_close(cache.policy_state.usage, usage)
        self.assertEqual(cache.policy_state.step_mod.item(), 2)
        self.assertEqual(state.execution_key.tolist(), [-1, -1])
        self.assertFalse(state.execution_valid.any().item())
        cache.begin_step(torch.tensor(True))
        finish_cache(cache)
        self.assertEqual(cache.policy_state.step_mod.item(), 3)
        with self.assertRaises(RuntimeError):
            cache.restore_decode_metadata_after_refill()

    def test_owned_metadata_is_accounted_once(self):
        cache = adapter()
        allocations = cache.residency.allocations() + cache.allocations()
        total = sum(item.nbytes for item in allocations)
        ledger = AllocationLedger({ResourceKind.METADATA: total}, allocations)
        self.assertEqual(ledger.usage[ResourceKind.METADATA], total)
        self.assertTrue(all(item.kind == ResourceKind.METADATA for item in allocations))
        self.assertFalse(any("delta" in item.name for item in allocations))


class TestCacheTensorDataFlow(unittest.TestCase):
    def test_no_host_reads_during_step(self):
        cache = adapter()
        ids = torch.tensor([0, 1, 1])
        valid, enabled = torch.tensor([True] * 3), torch.tensor(True)
        with ExitStack() as stack:
            for method in ("item", "tolist", "cpu", "__bool__"):
                stack.enter_context(
                    patch.object(
                        torch.Tensor, method, side_effect=AssertionError(method)
                    )
                )
            for _ in range(4):
                cache.begin_step(enabled)
                cache.record_local_routes(3, ids, valid)
                finish_cache(cache)
        self.assertEqual(cache.target_keys.tolist(), [0, 1, 4, 7])

    def test_trace_uses_live_routes_and_gate(self):
        cache = adapter()

        class Step(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Register captured state as mutable buffers, not trace constants.
                for prefix, group in (
                    ("residency", cache.residency),
                    ("policy", cache.policy_state),
                    ("stats", cache.stats),
                ):
                    for name, tensor in vars(group).items():
                        self.register_buffer(f"{prefix}_{name}", tensor)
                self.register_buffer("target_keys", cache.target_keys)

            def forward(self, ids, valid, enabled):
                cache.begin_step(enabled)
                cache.record_local_routes(3, ids, valid)
                finish_cache(cache)
                return cache.target_keys, cache.policy_state.usage

        ids, valid = torch.tensor([1, 1, 1]), torch.tensor([True] * 3)
        traced = torch.jit.trace(
            Step(), (ids, valid, torch.tensor(False)), check_trace=False
        )
        for _ in range(4):
            output = traced(ids, valid, torch.tensor(True))
        self.assertEqual(output[0].tolist(), [1, 2, 4, 7])
        self.assertAlmostEqual(output[1][1].item(), 8.4, places=5)
        before = cache.policy_state.usage.clone()
        traced(ids, valid, torch.tensor(False))
        torch.testing.assert_close(cache.policy_state.usage, before)
        self.assertEqual(cache.target_keys.tolist(), [0, 2, 4, 7])
        ids.fill_(3)
        for _ in range(4):
            output = traced(ids, valid, torch.tensor(True))
        self.assertEqual(output[0].tolist(), [1, 3, 4, 7])


if __name__ == "__main__":
    unittest.main()
