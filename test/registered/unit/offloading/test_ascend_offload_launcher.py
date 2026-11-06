"""Exercise the launcher without loading weights or touching NPU devices."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAscendOffloadLauncher(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[4]
        self.script = self.root / "scripts/launch_moe_offload_npu.sh"
        self.bash = shutil.which("bash")
        if os.name == "nt":
            self.bash = str(Path(os.environ["ProgramFiles"]) / "Git/bin/bash.exe")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        model = tmp / "model"
        model.mkdir()
        (model / "quant_model_description.json").write_text(
            json.dumps({"model_quant_type": "W8A8_DYNAMIC"})
        )
        (tmp / "profile.json").write_text(
            json.dumps({"model_id": "DeepSeek-V4-Flash", "layers": {}})
        )
        config = json.loads((self.root / "scripts/moe_offload/config.json").read_text())
        config.update(
            offline_stats_path="profile.json",
            native_extension_directory=(tmp / "native").as_posix(),
        )
        (tmp / "config.json").write_text(json.dumps(config))
        self.env = dict(os.environ)
        for name in (
            "WORLD_SIZE",
            "ATTN_DP_SIZE",
            "ASCEND_RT_VISIBLE_DEVICES",
            "MOE_EXPERT_FORMAT",
            "SGLANG_LOGGING_CONFIG_PATH",
            "PROFILE_CAPTURE",
            "ENABLE_TRACE",
            "BUILD_NATIVE",
            "MAX_RUNNING_REQUESTS",
        ):
            self.env.pop(name, None)
        self.env.update(
            MODEL_PATH=model.as_posix(),
            SGLANG_ROOT=self.root.as_posix(),
            PYTHON_BIN=Path(sys.executable).as_posix(),
            MOE_OFFLOAD_CONFIG=(tmp / "config.json").as_posix(),
            LOG_DIR=(tmp / "logs").as_posix(),
        )

    def run_dry(self, **overrides):
        return subprocess.run(
            [self.bash, str(self.script), "--dry-run"],
            cwd=self.root,
            env=self.env | {key: str(value) for key, value in overrides.items()},
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_default_dp2_and_graph_alignment(self):
        for world, requests, expected_bucket in (
            (4, 8, 8),
            (8, 1, 4),
            (16, 1, 8),
            (16, 9, 16),
        ):
            with self.subTest(world=world, requests=requests):
                result = self.run_dry(WORLD_SIZE=world, MAX_RUNNING_REQUESTS=requests)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = shlex.split(
                    next(
                        line
                        for line in result.stdout.splitlines()
                        if "-m sglang.launch_server" in line
                    )
                )
                for flag, value in (
                    ("--tp-size", world),
                    ("--ep-size", world),
                    ("--dp-size", 1),
                    ("--attn-dp-size", 2),
                    ("--moe-dp-size", 1),
                    ("--cuda-graph-max-bs-decode", expected_bucket),
                ):
                    self.assertEqual(args[args.index(flag) + 1], str(value))
                self.assertIn("--enable-dp-lm-head", args)
                self.assertEqual(
                    args[args.index("--moe-a2a-backend") + 1], "ascend_ops"
                )
                self.assertFalse(Path(self.env["LOG_DIR"]).exists())

    def test_dp_override_and_invalid_divisor(self):
        for dp in (1, 8):
            result = self.run_dry(WORLD_SIZE=8, ATTN_DP_SIZE=dp)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            args = shlex.split(
                next(
                    line
                    for line in result.stdout.splitlines()
                    if "-m sglang.launch_server" in line
                )
            )
            self.assertEqual(args[args.index("--attn-dp-size") + 1], str(dp))
            self.assertEqual("--enable-dp-lm-head" in args, dp > 1)
        result = self.run_dry(WORLD_SIZE=8, ATTN_DP_SIZE=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must divide WORLD_SIZE", result.stderr)


class TestAscendOffloadFormatDetection(unittest.TestCase):
    """Run the launcher's format probe directly, without Bash or device access."""

    def setUp(self):
        script = (
            Path(__file__).resolve().parents[4] / "scripts/launch_moe_offload_npu.sh"
        ).read_text()
        self.probe = script.split("<<'PYFORMAT'\n", 1)[1].split("\nPYFORMAT\n", 1)[0]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name)
        self.description = self.model / "quant_model_description.json"

    def run_probe(self, requested="auto"):
        return subprocess.run(
            [sys.executable, "-c", self.probe, str(self.model), requested],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_w4_and_w8_format_detection(self):
        for requested in ("auto", "w4a8"):
            self.description.write_text(
                json.dumps({"model_quant_type": "W4A8_DYNAMIC"})
            )
            result = self.run_probe(requested)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), "w4a8")
        self.description.write_text(json.dumps({"model_quant_type": "W8A8_DYNAMIC"}))
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "w8a8")

    def test_w4_expert_entries_override_model_wide_label(self):
        self.description.with_name("config.json").write_text(
            json.dumps(
                {"num_hidden_layers": 43, "dspark_full_dequantized_to_bf16": True}
            )
        )
        for prefix, projections in (
            ("model.layers.3.mlp", ("gate_proj", "up_proj", "down_proj")),
            ("layers.3.ffn", ("w1", "w3", "w2")),
        ):
            quant = {"model_quant_type": "W8A8_DYNAMIC", "group_size": 0}
            quant["model.layers.3.self_attn.wq_a.weight"] = "W8A8_DYNAMIC"
            quant["mtp.0.ffn.experts.0.w1.weight"] = "FLOAT"
            quant["layers.43.ffn.experts.0.w1.weight"] = "FLOAT"
            quant.update(
                {
                    f"{prefix}.experts.0.{projection}.weight": "W4A8_DYNAMIC"
                    for projection in projections
                }
            )
            self.description.write_text(json.dumps(quant))
            result = self.run_probe()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), "w4a8")
        result = self.run_probe("w8a8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match checkpoint experts", result.stderr)

    def test_unsupported_and_mixed_expert_formats_fail_before_launch(self):
        for quant in (
            {"model_quant_type": "W4A8_MXFP"},
            {
                "model_quant_type": "W8A8_DYNAMIC",
                "model.layers.3.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC",
                "model.layers.3.mlp.experts.0.down_proj.weight": "W8A8_DYNAMIC",
            },
        ):
            self.description.write_text(json.dumps(quant))
            result = self.run_probe()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("uniform ModelSlim", result.stderr)


if __name__ == "__main__":
    unittest.main()
