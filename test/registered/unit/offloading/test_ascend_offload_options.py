"""Host-only config/catalog contracts; runnable without Torch or an NPU."""

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace

# Import only these host-side files, avoiding sglang's public runtime imports.
_MOE = Path(__file__).resolve().parents[4] / "python/sglang/srt/offloading/moe"
_PACKAGE = "_ascend_offload_options_contract"
for _name in (_PACKAGE, f"{_PACKAGE}.npu"):
    _package = ModuleType(_name)
    _package.__path__ = []
    sys.modules[_name] = _package


def _load(name, path):
    spec = importlib.util.spec_from_file_location(f"{_PACKAGE}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load("config", _MOE / "config.py")
AscendOffloadOptions = _load(
    "npu.options", _MOE / "npu/options.py"
).AscendOffloadOptions
build_expert_catalog = _load(
    "npu.model_adapter", _MOE / "npu/model_adapter.py"
).build_expert_catalog


class TestAscendOffloadOptions(unittest.TestCase):
    def _read(self, values, *, model_id="GLM-5.2", role="null"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offload.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            return AscendOffloadOptions.from_path(
                path, model_id=model_id, disaggregation_mode=role
            )

    def test_existing_mixed_defaults_and_profile_remain_unchanged(self):
        values = {
            "host_pool_gib": 64,
            "offline_stats_path": "initial_profile.json",
            "native_extension_directory": "/tmp/sglang-moe-native",
        }
        for model_id, ratio in (("GLM-5.2", 0.25), ("DeepSeek-V4-Flash", 0.28125)):
            with self.subTest(model_id=model_id):
                options = self._read(values, model_id=model_id)
                self.assertEqual(options.mode, "mixed")
                self.assertFalse(options.prefill_only)
                self.assertEqual(options.cache_ratio, ratio)
                self.assertEqual(options.values, values)
                with self.assertRaises(FrozenInstanceError):
                    options.mode = "prefill_only"

    def test_explicit_mixed_options_keep_policy_values(self):
        values = {
            "mode": "mixed",
            "cache_ratio": 1,
            "execution_slots": 32,
            "decode_mask_capacity": 128,
            "offline_model_id": "custom-profile-id",
            "update_interval": 4,
            "max_swaps": 0,
            "min_usage": 0,
            "replacement_min_gain": 1.5,
            "usage_decay": 0.7,
        }
        options = self._read(values)
        self.assertEqual(options.values, values)
        self.assertEqual(options.cache_ratio, 1.0)

    def test_prefill_only_needs_only_a_mode_switch(self):
        options = self._read({"mode": "prefill_only"}, role="prefill")
        self.assertTrue(options.prefill_only)
        self.assertEqual(options.cache_ratio, 0.0)
        self.assertEqual(options.values, {"mode": "prefill_only"})

    def test_prefill_only_accepts_explicit_host_and_bank_budgets(self):
        options = self._read(
            {
                "mode": "prefill_only",
                "cache_ratio": 0,
                "host_pool_gib": 48,
                "execution_slots": 32,
                "native_extension_directory": "/tmp/native-acl",
            },
            role="prefill",
        )
        self.assertEqual(options.values["execution_slots"], 32)
        self.assertEqual(options.values["host_pool_gib"], 48)

    def test_prefill_resident_ratio_is_fractional_and_exclusive_to_prefill(self):
        option = "prefill_resident_ratio"
        options = self._read({"mode": "prefill_only", option: 0.25}, role="prefill")
        self.assertEqual(options.prefill_resident_ratio, 0.25)
        with self.assertRaisesRegex(ValueError, "requires mode=prefill_only"):
            self._read({option: 0.25})
        for value in (-0.01, 1, float("nan"), True, "0.25"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, option):
                self._read({"mode": "prefill_only", option: value}, role="prefill")

    def test_prefill_only_rejects_decode_and_mixed_worker_roles(self):
        for role in (None, "null", "decode", "mixed", "PREFILL"):
            with (
                self.subTest(role=role),
                self.assertRaisesRegex(ValueError, "disaggregation-mode prefill"),
            ):
                self._read({"mode": "prefill_only"}, role=role)

    def test_prefill_only_rejects_every_cache_and_decode_option(self):
        for name in (
            "offline_stats_path",
            "offline_model_id",
            "decode_mask_capacity",
            "update_interval",
            "min_usage",
            "replacement_min_gain",
            "max_swaps",
            "usage_decay",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self._read({"mode": "prefill_only", name: None}, role="prefill")

    def test_zero_cache_is_exclusive_to_prefill_only(self):
        with self.assertRaisesRegex(ValueError, "mixed mode"):
            self._read({"cache_ratio": 0})
        for value in (0.01, 1, -1):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "cache_ratio=0"),
            ):
                self._read(
                    {"mode": "prefill_only", "cache_ratio": value}, role="prefill"
                )

    def test_cache_ratio_rejects_bool_strings_and_nonfinite_values(self):
        for mode in ("mixed", "prefill_only"):
            for value in (True, False, "0", None, float("nan"), float("inf"), 10**400):
                with (
                    self.subTest(mode=mode, value=value),
                    self.assertRaisesRegex(ValueError, "cache_ratio"),
                ):
                    self._read({"mode": mode, "cache_ratio": value}, role="prefill")
        for value in (-0.1, 1.01):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "cache_ratio"),
            ):
                self._read({"cache_ratio": value})

    def test_memory_and_execution_budgets_require_positive_integers(self):
        for name in ("host_pool_gib", "execution_slots"):
            for value in (True, False, 0, -1, 1.5, "32", None):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    self._read({"mode": "prefill_only", name: value}, role="prefill")

    def test_policy_counts_reject_invalid_types_and_ranges(self):
        for name in ("decode_mask_capacity", "update_interval", "max_swaps"):
            for value in (True, False, -1, 1.5, "4", None):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    self._read({name: value})
        for name in ("decode_mask_capacity", "update_interval"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self._read({name: 0})

    def test_policy_weights_reject_invalid_types_and_ranges(self):
        for name in ("min_usage", "replacement_min_gain", "usage_decay"):
            for value in (True, False, -1, "2", None, float("nan"), float("inf")):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    self._read({name: value})
        with self.assertRaisesRegex(ValueError, "usage_decay"):
            self._read({"usage_decay": 1.01})

    def test_path_and_profile_options_require_strings(self):
        for name in (
            "offline_stats_path",
            "offline_model_id",
            "native_extension_directory",
        ):
            for value in (True, 64, [], ""):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    self._read({name: value})
        self._read({"offline_stats_path": None, "native_extension_directory": None})

    def test_unknown_options_and_invalid_modes_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "resident_routed_layers"):
            self._read({"resident_routed_layers": 1})
        for mode in (None, True, 1, [], "prefill", "decode_only"):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, "mode must"),
            ):
                self._read({"mode": mode})
        for value in ([], None, 1, "prefill_only"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "JSON object"),
            ):
                self._read(value)


class TestPrefillOnlyExpertCatalog(unittest.TestCase):
    @staticmethod
    def _glm(**changes):
        values = dict(
            model_type="glm_moe_dsa",
            architectures=["GlmMoeDsaForCausalLM"],
            num_hidden_layers=78,
            first_k_dense_replace=3,
            moe_layer_freq=1,
            n_routed_experts=256,
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def test_glm_prefill_offloads_all_75_routed_layers(self):
        catalog = build_expert_catalog(self._glm(), 16, 9, offload_all_routed=True)
        self.assertEqual(len(catalog.layers), 75)
        self.assertEqual(catalog.offloaded_layers, catalog.layers)
        self.assertEqual(catalog.num_experts, 75 * 16)
        self.assertEqual(catalog.layer(3).global_start, 9 * 16)
        self.assertEqual(catalog.key(3, 0), 0)
        self.assertEqual(catalog.key(77, 15), 75 * 16 - 1)
        for layer in (0, 1, 2, 78):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                catalog.layer(layer)

    def test_mixed_glm_keeps_first_routed_layer_resident(self):
        catalog = build_expert_catalog(self._glm(), 16, 0)
        self.assertTrue(catalog.layer(3).resident)
        self.assertEqual(len(catalog.offloaded_layers), 74)
        self.assertEqual(catalog.num_experts, 74 * 16)

    def test_sparse_glm_layers_keep_their_original_identity(self):
        config = self._glm(num_hidden_layers=12, moe_layer_freq=2)
        catalog = build_expert_catalog(config, 16, 0, offload_all_routed=True)
        self.assertEqual(
            tuple(layer.layer_id for layer in catalog.layers), (4, 6, 8, 10)
        )
        self.assertEqual(catalog.offloaded_layers, catalog.layers)
        self.assertEqual(catalog.key(4, 0), 0)
        self.assertEqual(catalog.key(10, 0), 48)

    def test_v4_prefill_includes_the_mixed_resident_hash_prefix(self):
        for hash_layers in (0, 5):
            with self.subTest(hash_layers=hash_layers):
                config = SimpleNamespace(
                    model_type="deepseek_v4",
                    num_hidden_layers=8,
                    n_routed_experts=256,
                    num_hash_layers=hash_layers,
                )
                mixed = build_expert_catalog(config, 16, 0)
                prefill = build_expert_catalog(config, 16, 0, offload_all_routed=True)
                self.assertEqual(
                    mixed.offloaded_layers[0].layer_id, max(3, hash_layers)
                )
                self.assertEqual(prefill.offloaded_layers, prefill.layers)
                self.assertEqual(prefill.num_experts, 8 * 16)

    def test_single_routed_layer_can_be_offloaded_by_prefill(self):
        config = self._glm(num_hidden_layers=4)
        with self.assertRaisesRegex(ValueError, "nonresident"):
            build_expert_catalog(config, 16, 0)
        catalog = build_expert_catalog(config, 16, 0, offload_all_routed=True)
        self.assertEqual(len(catalog.offloaded_layers), 1)

    def test_empty_or_invalid_catalog_still_fails(self):
        for changes in (
            {"first_k_dense_replace": 78},
            {"moe_layer_freq": 0},
            {"num_hidden_layers": 0},
            {"n_routed_experts": 255},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_expert_catalog(
                    self._glm(**changes), 16, 0, offload_all_routed=True
                )


if __name__ == "__main__":
    unittest.main()
