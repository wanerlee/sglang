"""Launch the real Bash script with a fake server; no NPU or weights are loaded."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMoEOffloadProfileConfig(unittest.TestCase):
    """Run the launcher's actual Python preflight without requiring Bash/NPU."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        script = (
            Path(__file__).resolve().parents[4] / "scripts/launch_moe_offload_npu.sh"
        ).read_text()
        self.code = script.split(" <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        self.config = self.root / "offload.json"
        self.profile = self.root / "profile.json"

    def run_config(self, **overrides):
        config = {
            "cache_ratio": 0.25,
            "native_extension_directory": str(self.root / "native"),
            **overrides,
        }
        self.config.write_text(json.dumps(config))
        return subprocess.run(
            [
                sys.executable,
                "-",
                str(self.config),
                "0",
                str(self.root),
                "info",
                "info",
                "0",
            ],
            input=self.code,
            text=True,
            capture_output=True,
        )

    def test_nonzero_cache_accepts_absent_or_null_profile(self):
        for ratio in (0, 0.25, 1):
            for profile in ({}, {"offline_stats_path": None, "offline_model_id": None}):
                with self.subTest(ratio=ratio, profile=profile):
                    result = self.run_config(cache_ratio=ratio, **profile)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(
                        (self.root / "offload-config.snapshot.json").is_file()
                    )

    def test_profile_needs_only_layers_and_resolves_relative_path(self):
        self.profile.write_text(json.dumps({"layers": {"3": {"0": 10}}}))
        result = self.run_config(offline_stats_path="profile.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = json.loads((self.root / "offload-config.snapshot.json").read_text())
        self.assertEqual(snapshot["offline_stats_path"], str(self.profile.resolve()))
        self.assertNotIn("offline_model_id", snapshot)

    def test_explicit_missing_or_invalid_profile_fails(self):
        result = self.run_config(offline_stats_path="profile.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing offline expert profile", result.stderr)
        self.profile.write_text(json.dumps({"layers": []}))
        result = self.run_config(offline_stats_path="profile.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("layers object", result.stderr)

    def test_explicit_model_label_still_checks_for_mismatch(self):
        self.profile.write_text(json.dumps({"model_id": "toy", "layers": {}}))
        for label, success in (("toy", True), ("other", False)):
            result = self.run_config(
                offline_stats_path="profile.json", offline_model_id=label
            )
            self.assertEqual(result.returncode == 0, success, result.stderr)


class TestMoEOffloadLaunchLogging(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = Path(
            os.environ.get("MOE_LAUNCH_TEST_ROOT", Path(__file__).resolve().parents[4])
        )
        self.script = Path(
            os.environ.get(
                "MOE_LAUNCH_TEST_SCRIPT",
                self.source / "scripts/launch_moe_offload_npu.sh",
            )
        )
        self.model = self.root / "model with spaces"
        self.model.mkdir()
        (self.model / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4", "n_routed_experts": 8})
        )
        (self.model / "test.safetensors").touch()
        self.profile = self.root / "profile.json"
        self.profile.write_text(
            json.dumps({"model_id": "toy", "layers": {"3": {"0": 10}}})
        )
        self.config = self.root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "cache_ratio": 0.25,
                    "offline_stats_path": str(self.profile),
                    "offline_model_id": "toy",
                    "native_extension_directory": str(self.root / "native build"),
                    "batch_log_interval": 1,
                    "batch_log_capacity": 64,
                }
            )
        )
        self.log_dir = self.root / "run logs"
        self.server_output = self.root / "server.json"
        self.build_marker = self.root / "build.marker"
        self.python = self.root / "fake-python"
        self.python.write_text(
            "#!/usr/bin/env python3\n"
            + """
import json
import os
import sys
from pathlib import Path

if sys.argv[1:3] == ['-m', 'sglang.launch_server']:
    keys = ['SGLANG_LOGGING_CONFIG_PATH', 'PYTHONUNBUFFERED', 'TASK_QUEUE_ENABLE',
            'ASCEND_LAUNCH_BLOCKING', 'SGLANG_TRACE_ASYNC', 'SGLANG_TORCH_PROFILER_DIR']
    Path(os.environ['FAKE_SERVER_OUTPUT']).write_text(json.dumps({
        'args': sys.argv[3:], 'pid': os.getpid(),
        'env': {key: os.environ.get(key) for key in keys},
    }))
    print('fake server stdout', flush=True)
    print('fake server stderr', file=sys.stderr, flush=True)
    sys.exit(int(os.environ.get('FAKE_SERVER_EXIT', '0')))
if sys.argv[1:3] == ['-m', 'sglang.srt.offloading.moe.npu.native_acl']:
    Path(os.environ['FAKE_BUILD_MARKER']).touch()
    sys.exit(0)
if len(sys.argv) > 2 and sys.argv[1] == '-c' and 'opentelemetry' in sys.argv[2]:
    sys.exit(int(os.environ.get('FAKE_TRACE_DEPS_EXIT', '0')))
os.execv(REAL_PYTHON, [REAL_PYTHON] + sys.argv[1:])
""".replace("REAL_PYTHON", repr(sys.executable))
        )
        self.python.chmod(0o755)
        self.env = dict(os.environ)
        for key in (
            "SGLANG_LOGGING_CONFIG_PATH",
            "SGLANG_TRACE_ASYNC",
            "SGLANG_TORCH_PROFILER_DIR",
            "ATTN_DP_SIZE",
            "LOG_LEVEL",
            "MOE_LOG_LEVEL",
            "LOG_LEVEL_HTTP",
            "DECODE_LOG_INTERVAL",
            "LOG_REQUESTS",
            "LOG_REQUESTS_LEVEL",
            "LOG_REQUESTS_FORMAT",
            "ENABLE_TRACE",
            "OTLP_TRACES_ENDPOINT",
            "OTLP_SERVICE_NAME",
            "PROFILE_CAPTURE",
        ):
            self.env.pop(key, None)
        self.env.update(
            SGLANG_ROOT=str(self.source),
            PYTHON_BIN=str(self.python),
            MODEL_PATH=str(self.model),
            MOE_OFFLOAD_CONFIG=str(self.config),
            LOG_DIR=str(self.log_dir),
            WORLD_SIZE="4",
            ASCEND_RT_VISIBLE_DEVICES="0,1,2,3",
            BUILD_NATIVE="1",
            MAX_RUNNING_REQUESTS="8",
            CHUNKED_PREFILL_SIZE="1024",
            CONTEXT_LENGTH="8192",
            FAKE_SERVER_OUTPUT=str(self.server_output),
            FAKE_BUILD_MARKER=str(self.build_marker),
            FAKE_SERVER_EXIT="0",
            FAKE_TRACE_DEPS_EXIT="0",
            ASCEND_LAUNCH_BLOCKING="1",
        )

    def run_script(self, *args, **overrides):
        env = dict(self.env, **overrides)
        process = subprocess.Popen(
            ["/bin/bash", str(self.script), *args],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            output, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            self.fail("Launch script did not exit")
        self.launch_pid = process.pid
        return process.returncode, output

    def server_result(self):
        return json.loads(self.server_output.read_text())

    def test_dry_run_does_not_build_launch_or_create_logs(self):
        code, output = self.run_script("--dry-run", MOE_LOG_LEVEL="debug")
        self.assertEqual(code, 0, output)
        self.assertIn("MoE=debug", output)
        self.assertIn("--log-requests-level 0", output)
        self.assertIn("batch", output)
        self.assertFalse(self.log_dir.exists())
        self.assertFalse(self.build_marker.exists())
        self.assertFalse(self.server_output.exists())

    def test_live_launch_preserves_exec_exit_status_and_inference_flags(self):
        code, output = self.run_script(FAKE_SERVER_EXIT="17")
        self.assertEqual(code, 17, output)
        result = self.server_result()
        self.assertEqual(result["pid"], self.launch_pid)
        self.assertEqual(result["env"]["TASK_QUEUE_ENABLE"], "1")
        self.assertIsNone(result["env"]["ASCEND_LAUNCH_BLOCKING"])
        self.assertEqual(result["env"]["PYTHONUNBUFFERED"], "1")
        arguments = result["args"]
        for flag, value in (
            ("--tp-size", "4"),
            ("--ep-size", "4"),
            ("--dp-size", "1"),
            ("--attn-dp-size", "1"),
            ("--moe-dp-size", "1"),
            ("--cuda-graph-backend-decode", "full"),
            ("--cuda-graph-backend-prefill", "disabled"),
            ("--decode-log-interval", "40"),
        ):
            self.assertEqual(arguments[arguments.index(flag) + 1], value)
        self.assertNotIn("--enable-trace", arguments)
        self.assertNotIn("--enable-profile-cuda-graph", arguments)
        self.assertTrue(self.build_marker.exists())
        log = (self.log_dir / "server.log").read_text()
        self.assertIn("fake server stdout", log)
        self.assertIn("fake server stderr", log)
        self.assertTrue((self.log_dir / "offload-config.snapshot.json").is_file())

    def test_w8a8_metadata_selects_modelslim_and_disables_fp4(self):
        (self.model / "quant_model_description.json").write_text(
            json.dumps({"model_quant_type": "W8A8_DYNAMIC"})
        )
        code, output = self.run_script()
        self.assertEqual(code, 0, output)
        arguments = self.server_result()["args"]
        self.assertEqual(arguments[arguments.index("--quantization") + 1], "modelslim")
        self.assertNotIn("--speculative-algorithm", arguments)
        self.assertIn("Expert format: w8a8", output)

    def test_explicit_fp4_rejects_w8a8_metadata_before_build(self):
        (self.model / "quant_model_description.json").write_text(
            json.dumps({"model_quant_type": "W8A8_DYNAMIC"})
        )
        code, output = self.run_script("--dry-run", MOE_EXPERT_FORMAT="fp4")
        self.assertNotEqual(code, 0)
        self.assertIn("disagrees", output)
        self.assertFalse(self.build_marker.exists())

    def test_moe_debug_does_not_enable_other_debug_loggers(self):
        code, output = self.run_script(MOE_LOG_LEVEL="debug")
        self.assertEqual(code, 0, output)
        config = self.server_result()["env"]["SGLANG_LOGGING_CONFIG_PATH"]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, logging, logging.config, sys; logging.config.dictConfig(json.load(open(sys.argv[1]))); logging.getLogger('sglang.srt.offloading.moe.programs').debug('moe debug visible'); logging.getLogger('unrelated').debug('unrelated debug hidden')",
                config,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("moe debug visible", result.stderr)
        self.assertNotIn("unrelated debug hidden", result.stderr)
        self.assertIn("pid=", result.stderr)

    def test_custom_logging_config_is_preserved(self):
        custom = self.root / "custom.json"
        original = '{"version": 1, "root": {"level": "ERROR"}}\n'
        custom.write_text(original)
        code, output = self.run_script(
            SGLANG_LOGGING_CONFIG_PATH=str(custom), MOE_LOG_LEVEL="debug"
        )
        self.assertEqual(code, 0, output)
        self.assertEqual(custom.read_text(), original)
        self.assertEqual(
            self.server_result()["env"]["SGLANG_LOGGING_CONFIG_PATH"], str(custom)
        )
        self.assertFalse((self.log_dir / "logging.json").exists())
        self.assertIn("controls logger levels", output)

    def test_trace_and_capture_profile_are_explicit_opt_ins(self):
        code, output = self.run_script(
            ENABLE_TRACE="1",
            PROFILE_CAPTURE="1",
            OTLP_TRACES_ENDPOINT="collector:4317",
            LOG_REQUESTS="0",
        )
        self.assertEqual(code, 0, output)
        result = self.server_result()
        self.assertIn("--enable-trace", result["args"])
        self.assertIn("collector:4317", result["args"])
        self.assertIn("--enable-profile-cuda-graph", result["args"])
        self.assertNotIn("--log-requests", result["args"])
        self.assertEqual(result["env"]["SGLANG_TRACE_ASYNC"], "1")
        self.assertEqual(
            result["env"]["SGLANG_TORCH_PROFILER_DIR"], str(self.log_dir / "profiler")
        )

    def test_missing_trace_dependencies_fail_before_native_build(self):
        code, output = self.run_script(ENABLE_TRACE="1", FAKE_TRACE_DEPS_EXIT="1")
        self.assertNotEqual(code, 0)
        self.assertIn("OpenTelemetry SDK", output)
        self.assertFalse(self.build_marker.exists())
        self.assertFalse(self.server_output.exists())

    def test_missing_profile_fails_before_native_build(self):
        self.profile.unlink()
        code, output = self.run_script()
        self.assertNotEqual(code, 0)
        self.assertIn("Missing offline expert profile", output)
        self.assertFalse(self.build_marker.exists())
        self.assertFalse(self.server_output.exists())

    def test_profile_model_mismatch_fails_before_native_build(self):
        self.profile.write_text(json.dumps({"model_id": "wrong", "layers": {}}))
        code, output = self.run_script()
        self.assertNotEqual(code, 0)
        self.assertIn("model_id must match", output)
        self.assertFalse(self.build_marker.exists())

    def test_relative_profile_snapshot_preserves_runtime_path_semantics(self):
        config = json.loads(self.config.read_text())
        config["offline_stats_path"] = "profile.json"
        self.config.write_text(json.dumps(config))
        code, output = self.run_script()
        self.assertEqual(code, 0, output)
        snapshot = json.loads(
            (self.log_dir / "offload-config.snapshot.json").read_text()
        )
        self.assertEqual(snapshot["offline_stats_path"], str(self.profile.resolve()))
        arguments = self.server_result()["args"]
        self.assertEqual(
            arguments[arguments.index("--moe-offload-config") + 1], str(self.config)
        )
        self.assertEqual(
            json.loads(self.config.read_text())["offline_stats_path"], "profile.json"
        )

    def test_zero_cache_requires_no_offline_profile(self):
        config = json.loads(self.config.read_text())
        config["cache_ratio"] = 0
        del config["offline_stats_path"]
        del config["offline_model_id"]
        self.config.write_text(json.dumps(config))
        code, output = self.run_script()
        self.assertEqual(code, 0, output)
        self.assertTrue(self.server_output.exists())

    def test_invalid_logging_options_are_rejected_without_starting(self):
        for overrides in (
            {"MOE_LOG_LEVEL": "verbose"},
            {"ENABLE_TRACE": "yes"},
            {"DECODE_LOG_INTERVAL": "0"},
            {"LOG_REQUESTS_LEVEL": "4"},
        ):
            with self.subTest(overrides=overrides):
                code, _ = self.run_script("--dry-run", **overrides)
                self.assertNotEqual(code, 0)
        self.assertFalse(self.server_output.exists())

    def test_attention_dp8_and_dp16_keep_ep_width_and_single_model_replica(self):
        for width in (8, 16):
            with self.subTest(width=width):
                (self.model / "config.json").write_text(
                    json.dumps({"model_type": "deepseek_v4", "n_routed_experts": width})
                )
                code, output = self.run_script(
                    WORLD_SIZE=str(width), ATTN_DP_SIZE=str(width),
                    ASCEND_RT_VISIBLE_DEVICES="",
                )
                self.assertEqual(code, 0, output)
                arguments = self.server_result()["args"]
                for flag, value in (
                    ("--tp-size", str(width)), ("--ep-size", str(width)),
                    ("--attn-dp-size", str(width)), ("--dp-size", "1"),
                    ("--moe-dp-size", "1"),
                ):
                    self.assertEqual(arguments[arguments.index(flag) + 1], value)
                start = arguments.index("--cuda-graph-bs-decode") + 1
                end = arguments.index("--cuda-graph-max-bs-decode")
                self.assertEqual(arguments[start:end], ["1", "2", "4", "8"])
                self.assertIn(f"ATTN_DP={width} ATTN_TP=1", output)
                self.assertNotIn("--enable-dp-attention", arguments)

    def test_partial_attention_dp_dry_run_does_not_create_logs(self):
        code, output = self.run_script(
            "--dry-run", WORLD_SIZE="16", ATTN_DP_SIZE="4",
            ASCEND_RT_VISIBLE_DEVICES="",
        )
        self.assertEqual(code, 0, output)
        self.assertIn("ATTN_DP=4 ATTN_TP=4", output)
        self.assertIn("--attn-dp-size 4", output)
        self.assertFalse(self.log_dir.exists())
        self.assertFalse(self.server_output.exists())

    def test_graph_buckets_cover_attention_tp_when_concurrency_is_smaller(self):
        code, output = self.run_script(
            "--dry-run", WORLD_SIZE="16", ATTN_DP_SIZE="1",
            ASCEND_RT_VISIBLE_DEVICES="", MAX_RUNNING_REQUESTS="8",
        )
        self.assertEqual(code, 0, output)
        self.assertIn("--cuda-graph-bs-decode 1 2 4 8 16", output)
        self.assertIn("--cuda-graph-max-bs-decode 16", output)
        self.assertIn("--max-running-requests 8", output)
        self.assertFalse(self.log_dir.exists())

    def test_invalid_attention_dp_layouts_fail_before_build_or_launch(self):
        for width, dp in (("16", "3"), ("8", "16"), ("8", "0"), ("8", "-1")):
            with self.subTest(width=width, dp=dp):
                code, output = self.run_script(
                    "--dry-run", WORLD_SIZE=width, ATTN_DP_SIZE=dp,
                    ASCEND_RT_VISIBLE_DEVICES="",
                )
                self.assertNotEqual(code, 0)
                self.assertIn("ATTN_DP_SIZE", output)
        self.assertFalse(self.build_marker.exists())
        self.assertFalse(self.server_output.exists())

if __name__ == "__main__":
    unittest.main()
