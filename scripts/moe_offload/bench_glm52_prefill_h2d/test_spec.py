"""CPU checks for byte budgets, selection rules, and bandwidth accounting."""

import json
import tempfile
import unittest
from pathlib import Path

from bench_spec import GIB, build_spec, summarize_samples


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.json"
        self.cfg = {
            "model_type": "glm_moe_dsa",
            "architectures": ["GlmMoeDsaForCausalLM"],
            "hidden_size": 6144,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 256,
            "num_hidden_layers": 78,
            "first_k_dense_replace": 3,
            "moe_layer_freq": 1,
        }
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.cfg))

    def spec(self, **kwargs):
        return build_spec(self.path, "18,20,24,16", kwargs.pop("pp_rank", 1), 4, **kwargs)

    def test_full_stage_budget_includes_metadata(self):
        spec = self.spec()
        self.assertEqual(spec["local_experts"], 64)
        self.assertEqual(spec["expected_bytes_per_layer"], 2419851264)
        self.assertEqual(spec["host_bytes_per_layer"], 2421161984)
        self.assertEqual(spec["expected_stage_bytes"], 2419851264 * 20)
        self.assertEqual(spec["two_bank_bytes"], 2419851264 * 2)
        self.assertEqual(spec["source_layer_count"], 20)
        self.assertFalse(spec["source_reused"])

    def test_stage_selection_excludes_dense_not_first_moe(self):
        routed = [self.spec(pp_rank=rank)["offloaded_layer_ids"] for rank in range(4)]
        self.assertEqual([len(ids) for ids in routed], [15, 20, 24, 16])
        self.assertEqual([layer for ids in routed for layer in ids], list(range(3, 78)))

    def test_scale_host_stride_preserves_checkpoint_padding(self):
        components = {row["name"]: row for row in self.spec()["components"]}
        self.assertEqual(components["w13_weight_scale"]["nbytes"], 8192)
        self.assertEqual(components["w13_weight_scale"]["host_nbytes"], 16384)
        self.assertEqual(components["w2_weight_scale"]["nbytes"], 12288)
        self.assertEqual(components["w2_weight_scale"]["host_nbytes"], 24576)
        self.assertEqual(components["w13_weight"]["runtime_shape"], [1, 6144, 4096])

    def test_source_reuse_reduces_memory_not_transferred_bytes(self):
        spec = self.spec(source_layers=2, host_pool_gib=5)
        self.assertTrue(spec["source_reused"])
        self.assertEqual(spec["host_slab_bytes"], 2 * 2421161984)
        self.assertEqual(spec["expected_stage_bytes"], 20 * 2419851264)
        self.assertEqual(spec["host_pool_bytes"], 5 * GIB)
        with self.assertRaisesRegex(ValueError, "Host sources need"):
            self.spec(source_layers=2, host_pool_gib=4)

    def test_invalid_geometry_and_config_rejected(self):
        for field, value in (("hidden_size", 6145), ("n_routed_experts", 255), ("moe_layer_freq", 0), ("first_k_dense_replace", -1), ("hidden_size", True), ("model_type", "other")):
            previous = self.cfg[field]
            self.cfg[field] = value
            self.save()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.spec()
            self.cfg[field] = previous
        self.save()
        for source in (0, -1, 21, True):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.spec(source_layers=source)
        with self.assertRaises(ValueError):
            build_spec(self.path, "18,20,24,15", 1, 4)

    def test_nested_metadata_and_frequency_match_model(self):
        self.cfg["moe_layer_freq"] = 3
        self.path.write_text(json.dumps({"text_config": self.cfg}))
        spec = build_spec(self.path.parent, "18,20,24,16", 0, 4)
        self.assertEqual(spec["offloaded_layer_ids"], [3, 6, 9, 12, 15])


def sample(device=0, iteration=0, scenario="all", start=0, end=1000000000, nbytes=10**9, event_ms=1000):
    return {
        "scenario": scenario,
        "iteration": iteration,
        "device": device,
        "bytes": nbytes,
        "host_start_ns": start,
        "host_submit_end_ns": start + 1000000,
        "host_end_ns": end,
        "event_ms": event_ms,
    }


class SummaryTests(unittest.TestCase):
    def test_common_window_is_not_sum_of_own_rates(self):
        result = summarize_samples([sample(), sample(device=1, start=1000000000, end=2000000000)])
        self.assertEqual(result["summary"][0]["median_total_gbps"], 1)
        self.assertEqual(result["summary"][0]["median_window_ms"], 2000)
        self.assertEqual(result["summary"][0]["max_start_skew_ms"], 1000)
        self.assertEqual(sum(row["median_host_gbps"] for row in result["ranks"]), 2)
        self.assertEqual(result["summary"][0]["median_event_estimate_gbps"], 2)

    def test_missing_duplicate_and_invalid_samples_rejected(self):
        for rows in (
            [],
            [sample(), sample()],
            [sample(), sample(device=1), sample(iteration=1)],
            [sample(event_ms=float("nan"))],
            [sample(end=0)],
            [sample(nbytes=0)],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_samples(rows)

    def test_medians_and_submit_wait_split(self):
        rows = [sample(iteration=i, end=ms * 10**6, event_ms=ms) for i, ms in enumerate([1000, 1000, 9000])]
        result = summarize_samples(rows)
        summary, rank = result["summary"][0], result["ranks"][0]
        self.assertEqual(summary["median_window_ms"], 1000)
        self.assertEqual(summary["median_total_gbps"], 1)
        self.assertEqual(rank["median_submit_ms"], 1)
        self.assertEqual(rank["median_wait_ms"], 999)
        self.assertGreater(summary["window_cv_pct"], 0)

    def test_solo_and_all_have_independent_membership(self):
        result = summarize_samples([sample(scenario="solo:0"), sample(), sample(device=1)])
        summaries = {row["scenario"]: row for row in result["summary"]}
        self.assertEqual(summaries["all"]["devices"], [0, 1])
        self.assertEqual(summaries["all"]["median_total_gbps"], 2)
        self.assertEqual(summaries["solo:0"]["median_total_gbps"], 1)


if __name__ == "__main__":
    unittest.main()
