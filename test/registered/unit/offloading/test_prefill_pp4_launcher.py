"""CPU-only launch preflight; no SGLang, NPU or model weights are imported."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh"
)
BASH = shutil.which("bash")


class TestPrefillPP4Launcher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # Exclude ip / ifconfig / hostname and accelerator executables.
        for name in ("dirname", "date", "cat"):
            (self.bin / name).symlink_to(shutil.which(name))
        self.model = self.root / "checkpoint with spaces"
        self.model.mkdir()
        self.config = {
            "model_type": "glm_moe_dsa",
            "architectures": ["GlmMoeDsaForCausalLM"],
            "num_hidden_layers": 78,
            "first_k_dense_replace": 3,
            "moe_layer_freq": 1,
            "n_routed_experts": 256,
            "hidden_size": 6144,
            "moe_intermediate_size": 2048,
        }
        self._write_config()
        self.env = dict(os.environ)
        self.env.update(
            PATH=str(self.bin),
            MODEL_PATH=str(self.model),
            PYTHON_BIN=sys.executable,
            RUNTIME_DIR=str(self.root / "runtime"),
            LOG_DIR=str(self.root / "logs"),
            HOST_POOL_GIB="64",
            P_MOE_A2A_BACKEND="ascend_ops",
            EXECUTION_SLOTS="",
            PREFILL_RESIDENT_RATIO="0",
            SGLANG_PP_LAYER_PARTITION="18,20,24,16",
            ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, range(16))),
            CONTEXT_LENGTH="16384",
            CHUNKED_PREFILL_SIZE="8192",
            MAX_PREFILL_TOKENS="8192",
            MAX_RUNNING_REQUESTS="8",
            MEM_FRACTION_STATIC="0.8",
            ENABLE_MTP="0",
            P_IP="127.0.0.1",
            COMM_IFACE="lo",
            SERVER_PORT="8000",
            DIST_PORT="5000",
            BOOTSTRAP_PORT="8998",
            MF_STORE_PORT="24707",
            CANN_ENV="/missing/cann.sh",
        )

    def _write_config(self):
        (self.model / "config.json").write_text(json.dumps(self.config))

    def _run(self, **settings):
        result = subprocess.run(
            [BASH, str(SCRIPT), "--dry-run"],
            env={**self.env, **settings},
            text=True,
            capture_output=True,
            timeout=20,
        )
        # Both successful and rejected preflight must stay read-only.
        self.assertFalse((self.root / "runtime").exists())
        self.assertFalse((self.root / "logs").exists())
        return result

    def test_dry_run_has_stage_ep_topology_and_no_decode_resources(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        config, _ = json.JSONDecoder().raw_decode(result.stdout)
        self.assertEqual(config["mode"], "prefill_only")
        self.assertEqual(config["cache_ratio"], 0)
        self.assertEqual(config["execution_slots"], 128)
        self.assertEqual(config["host_pool_gib"], 64)
        self.assertNotIn("decode_mask_capacity", config)
        self.assertIn("[15, 20, 24, 16]", result.stdout)
        self.assertIn("1024 GiB/node", result.stdout)
        command = shlex.split(result.stdout.split("Command:", 1)[1])
        expected = {
            "--model-path": str(self.model),
            "--pp-size": "4",
            "--tp-size": "4",
            "--dp-size": "1",
            "--attn-dp-size": "1",
            "--attn-cp-size": "4",
            "--ep-size": "4",
            "--moe-a2a-backend": "ascend_ops",
            "--moe-dp-size": "1",
            "--cp-strategy": "zigzag",
            "--disaggregation-mode": "prefill",
            "--disaggregation-transfer-backend": "ascend",
            "--cuda-graph-backend-prefill": "disabled",
            "--cuda-graph-backend-decode": "disabled",
        }
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--enable-prefill-cp", command)
        self.assertIn("--disable-overlap-schedule", command)
        self.assertIn("--disable-shared-experts-fusion", command)
        self.assertNotIn("--speculative-algorithm", command)
        self.assertNotIn("--enable-dp-lm-head", command)
        self.assertNotIn("sglang_router.launch_router", command)

    def test_deepep_prefill_selects_normal_with_int8_activations(self):
        result = self._run(P_MOE_A2A_BACKEND="deepep", CHUNKED_PREFILL_SIZE="16384")
        self.assertEqual(result.returncode, 0, result.stderr)
        config, _ = json.JSONDecoder().raw_decode(result.stdout)
        self.assertEqual(config["mode"], "prefill_only")
        self.assertEqual(config["execution_slots"], 128)
        command = shlex.split(result.stdout.split("Command:", 1)[1])
        for flag, value in {
            "--moe-a2a-backend": "deepep",
            "--deepep-mode": "normal",
            "--deepep-dispatcher-output-dtype": "int8",
            "--chunked-prefill-size": "16384",
        }.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--moe-ascend-ops-config", command)

    def test_partial_residency_reports_stage_sizes_and_respects_budget(self):
        result = self._run(PREFILL_RESIDENT_RATIO="0.25", P_MOE_A2A_BACKEND="deepep")
        self.assertEqual(result.returncode, 0, result.stderr)
        config, _ = json.JSONDecoder().raw_decode(result.stdout)
        self.assertEqual(config["prefill_resident_ratio"], 0.25)
        self.assertNotIn("execution_slots", config)
        self.assertIn(
            "16/64; HBM slots/rank by stage: [336, 416, 480, 352]", result.stdout
        )

        result = self._run(PREFILL_RESIDENT_RATIO="0.25", EXECUTION_SLOTS="479")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at least 480", result.stderr)

    def test_rejects_insufficient_slots_and_host_before_startup(self):
        for settings, message in (
            ({"EXECUTION_SLOTS": "127"}, "at least 128"),
            ({"HOST_POOL_GIB": "48"}, "HOST_POOL_GIB leaves no room"),
        ):
            with self.subTest(settings=settings):
                result = self._run(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_rejects_bad_partitions_devices_and_mtp(self):
        for settings, message in (
            ({"SGLANG_PP_LAYER_PARTITION": "18,20,24,15"}, "summing to 78"),
            ({"SGLANG_PP_LAYER_PARTITION": "3,20,24,31"}, "at least one routed"),
            ({"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"}, "16 distinct A3"),
            ({"ENABLE_MTP": "1"}, "requires ENABLE_MTP=0"),
            ({"P_MOE_A2A_BACKEND": "invalid"}, "P_MOE_A2A_BACKEND"),
        ):
            with self.subTest(settings=settings):
                result = self._run(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_rejects_wrong_model(self):
        self.config["model_type"] = "deepseek_v4"
        self._write_config()
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GLM-5.2", result.stderr)


if __name__ == "__main__":
    unittest.main()
