"""CPU policy tests; these do not certify accelerator graph/transfer support."""

import json
import random
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.offloading.moe import (
    ExpertCatalog,
    ExpertLayerSpec,
    PeriodicLFUConfig,
    PeriodicLFUPolicy,
    PeriodicLFUState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def catalog():
    return ExpertCatalog(tuple(ExpertLayerSpec(i, 8, 2, 1, i < 3) for i in range(5)))


def mapping(keys, count=8):
    result = torch.full((count,), -1, dtype=torch.int64)
    for slot, key in enumerate(keys):
        result[key] = slot
    return result


def reference(usage, resident, cat, cfg):
    """Independent host oracle, never used by the production policy."""
    pairs = []
    for layer in cat.offloaded_layers:
        keys = [cat.key(layer.layer_id, i) for i in range(layer.local_experts)]
        hot = sorted((e for e in keys if e in resident), key=lambda e: (usage[e], e))
        cold = sorted(
            (e for e in keys if e not in resident), key=lambda e: (-usage[e], e)
        )
        for candidate, victim in zip(cold, hot):
            gap = usage[candidate] - usage[victim]
            if usage[candidate] >= cfg.min_usage and gap >= cfg.replacement_min_gain:
                pairs.append((gap, layer.layer_id, candidate, victim))
    target = set(resident)
    for _, _, candidate, victim in sorted(
        pairs, key=lambda p: (-p[0], p[1], p[2], p[3])
    )[: cfg.max_swaps]:
        target.remove(victim)
        target.add(candidate)
    return sorted(target)


class TestPeriodicLFU(unittest.TestCase):
    def make_policy(self, cat=None, **kwargs):
        return PeriodicLFUPolicy(
            PeriodicLFUConfig(offline_stats_path="unused.json", **kwargs),
            cat or catalog(),
        )

    def select(self, policy, usage, resident, due=True):
        state = policy.allocate_state(len(usage), "cpu")
        state.usage.copy_(torch.tensor(usage))
        state.update_due.fill_(due)
        slots = mapping(resident, len(usage))
        before = slots.clone()
        target = torch.full((len(resident),), -1, dtype=torch.int64)
        policy.select_hot_set(state, slots, target)
        torch.testing.assert_close(slots, before)
        torch.testing.assert_close(
            state.usage, torch.tensor(usage, dtype=torch.float32)
        )
        return target.tolist()

    def test_document_example_and_tie_at_global_limit(self):
        usage = [2, 8, 5, 3, 1, 7, 2, 4]
        resident = [0, 2, 4, 7]
        self.assertEqual(self.select(self.make_policy(), usage, resident), [1, 2, 5, 7])
        self.assertEqual(
            self.select(self.make_policy(max_swaps=1), usage, resident), [1, 2, 4, 7]
        )

    def test_layer_quotas_do_not_move_to_a_hotter_layer(self):
        # Global pairing would evict keys 0 and 2; layer-local pairing cannot.
        usage = [2, 1, 3, 1, 10, 20, 19, 11]
        self.assertEqual(
            self.select(self.make_policy(), usage, [0, 2, 4, 7]), [0, 2, 5, 6]
        )
        # A layer without residents cannot admit even extremely hot experts.
        self.assertEqual(
            self.select(self.make_policy(), [100] * 4 + [0] * 4, [4, 5, 6, 7]),
            [4, 5, 6, 7],
        )

    def test_inclusive_additive_threshold_and_minimum_usage(self):
        cat = ExpertCatalog((ExpertLayerSpec(3, 2, 1, 0),))
        policy = self.make_policy(cat)
        for cold, hot, expected in (
            (3.5, 2, [1]),
            (3, 2, [0]),
            (12, 10, [1]),
            (2, 0.5, [1]),
            (1.75, 0, [0]),
        ):
            with self.subTest(cold=cold, hot=hot):
                self.assertEqual(self.select(policy, [hot, cold], [0]), expected)

    def test_batch_counts_period_decay_and_next_cycle(self):
        policy = self.make_policy()
        state = policy.allocate_state(8, "cpu")
        slots = mapping([0, 2, 4, 7])
        target = torch.empty(4, dtype=torch.int64)
        counts = torch.tensor([0, 3, 0, 0, 0, 2, 0, 0])
        pointers = [tensor.data_ptr() for tensor in vars(state).values()]
        for step in range(1, 5):
            policy.update_statistics(state, counts, torch.tensor(True))
            self.assertEqual(state.step_mod.item(), step % 4)
            self.assertEqual(state.update_due.item(), step == 4)
            policy.select_hot_set(state, slots, target)
            self.assertEqual(
                target.tolist(), [1, 2, 5, 7] if step == 4 else [0, 2, 4, 7]
            )
            policy.finish_step(state)
        torch.testing.assert_close(state.usage, counts.float() * 4 * 0.7)
        policy.finish_step(state)  # Does not decay twice.
        torch.testing.assert_close(state.usage, counts.float() * 4 * 0.7)
        policy.update_statistics(state, counts, torch.tensor(True))
        torch.testing.assert_close(state.usage, counts.float() * (4 * 0.7 + 1))
        self.assertEqual(state.step_mod.item(), 1)
        self.assertEqual(
            pointers, [tensor.data_ptr() for tensor in vars(state).values()]
        )

    def test_disabled_updates_and_valid_rank_with_zero_local_counts(self):
        policy = self.make_policy()
        state = policy.allocate_state(8, "cpu")
        for _ in range(6):
            policy.update_statistics(
                state, torch.ones(8, dtype=torch.int64), torch.tensor(False)
            )
            self.assertFalse(state.update_due.item())
            policy.finish_step(state)
        self.assertEqual(state.step_mod.item(), 0)
        self.assertEqual(state.usage.sum().item(), 0)
        state.usage.fill_(10)
        for _ in range(4):
            policy.update_statistics(
                state, torch.zeros(8, dtype=torch.int64), torch.tensor(True)
            )
        self.assertTrue(state.update_due.item())
        policy.finish_step(state)
        torch.testing.assert_close(state.usage, torch.full((8,), 7.0))
        # A disabled call must also clear a stale due flag.
        state.update_due.fill_(True)
        policy.update_statistics(
            state, torch.ones(8, dtype=torch.int64), torch.tensor(False)
        )
        self.assertFalse(state.update_due.item())

    def test_snapshot_restores_all_fields_without_rebinding(self):
        policy = self.make_policy()
        state = policy.allocate_state(8, "cpu")
        saved = policy.allocate_state(8, "cpu")
        state.usage.copy_(torch.arange(8).float() * 0.7)
        state.step_mod.fill_(2)
        state.update_due.fill_(True)
        policy.copy_state(state, saved)
        pointers = [tensor.data_ptr() for tensor in vars(state).values()]
        state.usage.zero_()
        state.step_mod.zero_()
        state.update_due.zero_()
        policy.copy_state(saved, state)
        for name in vars(state):
            torch.testing.assert_close(getattr(state, name), getattr(saved, name))
        self.assertEqual(
            pointers, [tensor.data_ptr() for tensor in vars(state).values()]
        )

    def test_global_limit_96_prefers_gain_over_layer_order(self):
        cat = ExpertCatalog(tuple(ExpertLayerSpec(i, 120, 1, 0) for i in (3, 4)))
        resident = list(range(60)) + list(range(120, 180))
        usage = [0] * 60 + [10] * 60 + [0] * 60 + [20] * 60
        result = self.select(self.make_policy(cat), usage, resident)
        self.assertEqual(len(set(result) - set(resident)), 96)
        self.assertTrue(set(range(180, 240)).issubset(result))
        self.assertEqual(sum(e < 120 for e in result), 60)

    def test_zero_full_empty_and_no_swap_rounds(self):
        for count, resident in ((8, []), (8, list(range(8))), (0, [])):
            cat = catalog() if count else ExpertCatalog(())
            policy = self.make_policy(cat, update_interval=1)
            state = policy.allocate_state(count, "cpu")
            state.usage.fill_(10)
            target = torch.empty(len(resident), dtype=torch.int64)
            policy.update_statistics(
                state, torch.zeros(count, dtype=torch.int64), torch.tensor(True)
            )
            policy.select_hot_set(state, mapping(resident, count), target)
            policy.finish_step(state)
            self.assertEqual(target.tolist(), resident)
            torch.testing.assert_close(state.usage, torch.full((count,), 7.0))
        self.assertEqual(
            self.select(
                self.make_policy(max_swaps=0),
                [0, 100, 0, 100, 0, 100, 100, 0],
                [0, 2, 4, 7],
            ),
            [0, 2, 4, 7],
        )

    def test_randomized_against_host_reference(self):
        rng = random.Random(17)
        for _ in range(100):
            cat = ExpertCatalog(
                tuple(ExpertLayerSpec(i, n, 1, 0) for i, n in enumerate((5, 8, 3)))
            )
            policy = self.make_policy(cat, max_swaps=rng.randrange(7))
            resident = rng.sample(range(16), rng.randrange(17))
            usage = [rng.randrange(40) * 0.5 for _ in range(16)]
            self.assertEqual(
                self.select(policy, usage, resident),
                reference(usage, set(resident), cat, policy.config),
            )

    def test_online_operations_do_not_read_device_values_on_host(self):
        policy = self.make_policy()
        state = policy.allocate_state(8, "cpu")
        slots, target = mapping([0, 2, 4, 7]), torch.empty(4, dtype=torch.int64)
        counts, enabled = torch.ones(8, dtype=torch.int64), torch.tensor(True)
        with ExitStack() as stack:
            for method in ("item", "tolist", "cpu", "__bool__"):
                stack.enter_context(
                    patch.object(
                        torch.Tensor, method, side_effect=AssertionError(method)
                    )
                )
            for _ in range(4):
                policy.update_statistics(state, counts, enabled)
                policy.select_hot_set(state, slots, target)
                policy.finish_step(state)

    def test_invalid_configuration_and_capacity(self):
        for overrides in (
            {"update_interval": 0},
            {"update_interval": 1.5},
            {"max_swaps": -1},
            {"max_swaps": 1.5},
            {"min_usage": -1},
            {"min_usage": float("nan")},
            {"replacement_min_gain": float("inf")},
            {"usage_decay": 1.1},
            {"usage_decay": float("nan")},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.make_policy(**overrides)
        policy = self.make_policy()
        with self.assertRaises(ValueError):
            policy.allocate_state(7, "cpu")
        with self.assertRaises(ValueError):
            policy.select_hot_set(
                policy.allocate_state(8, "cpu"),
                mapping([]),
                torch.empty(9, dtype=torch.int64),
            )


class TestOfflineInitialization(unittest.TestCase):
    def test_rank_filter_global_topk_missing_and_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile = {
                "model_id": "test-model",
                "layers": {
                    "0": {"4": 9999},  # Resident layer.
                    "3": {"0": 9999, "4": 120, "6": 110},  # Foreign expert 0.
                    "4": {"4": 100, "7": 90},
                },
            }
            path.write_text(json.dumps(profile), encoding="utf-8")
            policy = PeriodicLFUPolicy(PeriodicLFUConfig(str(path)), catalog())
            self.assertEqual(policy.initial_keys(4, "test-model"), (0, 2, 4, 7))
            self.assertEqual(
                policy.initial_keys(8, "test-model"), (0, 2, 4, 7, 1, 3, 5, 6)
            )
            self.assertEqual(policy.initial_keys(0, "test-model"), ())
            self.assertEqual(policy.allocate_state(8, "cpu").usage.sum().item(), 0)
            with self.assertRaises(ValueError):
                policy.initial_keys(4, "other-model")
            with self.assertRaises(ValueError):
                policy.initial_keys(9, "test-model")
            profile["layers"]["3"]["4"] = -1
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaises(ValueError):
                policy.initial_keys(4, "test-model")


class TestTensorReplay(unittest.TestCase):
    def test_traced_step_responds_to_enabled_counts_and_residency(self):
        # CPU tracing checks tensor data flow, not CUDA/NPU capture compatibility.
        policy = PeriodicLFUPolicy(PeriodicLFUConfig("unused.json"), catalog())

        class Step(torch.nn.Module):
            def forward(self, usage, step, due, counts, enabled, slots, target):
                state = PeriodicLFUState(usage, step, due)
                policy.update_statistics(state, counts, enabled)
                policy.select_hot_set(state, slots, target)
                policy.finish_step(state)
                return usage, step, due, target

        state = policy.allocate_state(8, "cpu")
        counts = torch.tensor([0, 3, 0, 0, 0, 2, 0, 0])
        args = (
            state.usage,
            state.step_mod,
            state.update_due,
            counts,
            torch.tensor(False),
            mapping([0, 2, 4, 7]),
            torch.empty(4, dtype=torch.int64),
        )
        traced = torch.jit.trace(Step(), args, check_trace=False)
        enabled = torch.tensor(True)
        for i in range(4):
            result = traced(*args[:4], enabled, *args[5:])
            self.assertEqual(
                result[3].tolist(), [1, 2, 5, 7] if i == 3 else [0, 2, 4, 7]
            )
        torch.testing.assert_close(state.usage, counts.float() * 4 * 0.7)
        # Subsequent replay sees changed resident keys and changed route counts.
        args[5].copy_(mapping([1, 2, 5, 7]))
        counts.copy_(torch.tensor([20, 0, 0, 0, 20, 0, 0, 0]))
        for _ in range(4):
            result = traced(*args[:4], enabled, *args[5:])
        self.assertEqual(result[3].tolist(), [0, 1, 4, 5])


if __name__ == "__main__":
    unittest.main()
