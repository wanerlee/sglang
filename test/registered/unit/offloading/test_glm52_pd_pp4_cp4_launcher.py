"""Contract tests for the 1P-offload / 2D launcher without accelerator dependencies."""

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
    / "scripts/moe_offload/serve_glm52_pd_pp4_cp4.sh"
)
BASH = shutil.which("bash")
IPS = ("10.20.30.10", "10.20.30.11", "10.20.30.12")


class TestGlm52PdPP4Launcher(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="glm52-pd-pp4-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # No ip / ifconfig / hostname or accelerator programs in PATH.
        for name in ("dirname", "date", "cat", "mkdir"):
            (self.bin / name).symlink_to(shutil.which(name))
        tee = self.bin / "tee"
        tee.write_text(
            "#!/bin/bash\n"
            "if [[ ${1-} == /sys/devices/system/cpu/* ]]; then\n"
            "  cat >/dev/null\n"
            "  printf 'governor performance\\n' >> \"$STUB_TUNING_LOG\"\n"
            "else\n"
            f"  exec {shlex.quote(shutil.which('tee'))} \"$@\"\n"
            "fi\n"
        )
        tee.chmod(0o755)
        sysctl = self.bin / "sysctl"
        sysctl.write_text(
            "#!/bin/bash\n"
            "printf '%s\\n' \"$*\" >> \"$STUB_TUNING_LOG\"\n"
        )
        sysctl.chmod(0o755)
        python = self.bin / "python3"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "if len(sys.argv) > 1 and sys.argv[1].endswith('/pd_node.py'):\n"
            "    sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent))\n"
            "    import pd_node\n"
            "    from unittest.mock import patch\n"
            "    local = {ip: {'bond0'} for ip in os.environ.get('STUB_LOCAL_IPS', '').split(',') if ip}\n"
            "    with patch.object(pd_node, 'local_ipv4_interfaces', return_value=local):\n"
            "        pd_node.main(sys.argv[2:])\n"
            "    sys.exit(0)\n"
            "if sys.argv[1:4] in (['-u', '-m', 'sglang.launch_server'], ['-u', '-m', 'sglang_router.launch_router']):\n"
            "    keys = ['SGLANG_HOST_IP', 'SGLANG_PP_LAYER_PARTITION', 'HCCL_SOCKET_IFNAME',\n"
            "            'ASCEND_MF_STORE_URL', 'SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE', 'PYTHONPATH',\n"
            "            'HCCL_BUFFSIZE', 'TASK_QUEUE_ENABLE', 'SGLANG_ENABLE_OVERLAP_PLAN_STREAM',\n"
            "            'SGLANG_NPU_USE_MULTI_STREAM', 'SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK',\n"
            "            'DEEPEP_HCCL_BUFFSIZE', 'ASCEND_MF_LOG_LEVEL', 'LD_LIBRARY_PATH',\n"
            "            'DEEPEP_NORMAL_LONG_SEQ_ROUND', 'DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS',\n"
            "            'DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ', 'DEEP_NORMAL_MODE_USE_INT8_QUANT',\n"
            "            'MOE_EXPERT_TOKEN_NUMS_TYPE',\n"
            "            'SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT', 'SGLANG_DISAGGREGATION_WAITING_TIMEOUT',\n"
            "            'STUB_P_CUSTOMIZE_SOURCED', 'STUB_D_CUSTOMIZE_SOURCED']\n"
            "    data = {'args': sys.argv[1:], 'env': {k: os.environ.get(k) for k in keys}}\n"
            "    pathlib.Path(os.environ['STUB_RECORD']).write_text(json.dumps(data))\n"
            "    sys.exit(0)\n"
            "if sys.argv[1:3] == ['-m', 'sglang.srt.offloading.moe.npu.native_acl']:\n"
            "    sys.exit(0)\n"
            "if len(sys.argv) > 2 and sys.argv[1] == '-' and sys.argv[2] == os.environ['RUNTIME_DIR'] + '/python':\n"
            "    pathlib.Path(sys.argv[2], 'memfabric_hybrid', 'lib').mkdir(parents=True, exist_ok=True)\n"
            "    sys.exit(0)\n"
            "if len(sys.argv) > 2 and sys.argv[1] == '-' and sys.argv[2].endswith('/memfabric_hybrid/lib'):\n"
            "    sys.exit(0)\n"
            f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        self.model = self.root / "model with spaces"
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
        (self.model / "config.json").write_text(json.dumps(self.config))
        env_file = self.root / "ascend-env.sh"
        env_file.write_text("# mock Ascend environment\n")
        p_custom_env = self.root / "p-custom.sh"
        p_custom_env.write_text("export STUB_P_CUSTOMIZE_SOURCED=1\n")
        d_custom_env = self.root / "d-custom.sh"
        d_custom_env.write_text("export STUB_D_CUSTOMIZE_SOURCED=1\n")
        self.env = {
            "PATH": str(self.bin),
            "LC_ALL": "C",
            "MODEL_PATH": str(self.model),
            "P_IP": IPS[0],
            "D0_IP": IPS[1],
            "D1_IP": IPS[2],
            "STUB_LOCAL_IPS": IPS[0],
            "STUB_RECORD": str(self.root / "launched.json"),
            "RUNTIME_DIR": str(self.root / "runtime"),
            "LOG_DIR": str(self.root / "logs"),
            "CANN_ENV": str(env_file),
            "ATB_ENV": str(env_file),
            "CUSTOMIZE_ENV": str(p_custom_env),
            "CUSTOM_TRANSFORMER_ENV": str(env_file),
            "D_CUSTOMIZE_ENV": str(d_custom_env),
            "D_CUSTOMIZE_OP_API_LIB": str(self.root / "decode custom ops"),
            "STUB_TUNING_LOG": str(self.root / "host-tuning.log"),
            # Test that these pre-existing values do not leak to the wrong role.
            "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
            "SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE": "1",
        }

    def run_launcher(self, *args, dry=True, overrides=None):
        command = [BASH, str(SCRIPT), *args]
        if dry:
            command.append("--dry-run")
        result = subprocess.run(
            command,
            env={**self.env, **(overrides or {})},
            capture_output=True,
            text=True,
            timeout=20,
        )
        if dry:
            self.assertFalse((self.root / "runtime").exists())
            self.assertFalse((self.root / "logs").exists())
            self.assertFalse((self.root / "launched.json").exists())
            self.assertFalse((self.root / "host-tuning.log").exists())
        return result

    def command(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(
            line for line in result.stdout.splitlines() if line.startswith("Command:")
        )
        return shlex.split(line[len("Command:") :])

    def option(self, command, name, value):
        self.assertEqual(command[command.index(name) + 1], str(value))

    def test_prefill_auto_uses_stage_ep_cp_and_offload(self):
        result = self.run_launcher()
        command = self.command(result)
        for key, value in {
            "--disaggregation-mode": "prefill",
            "--disaggregation-transfer-backend": "ascend",
            "--dist-init-addr": f"{IPS[0]}:5000",
            "--nnodes": 1,
            "--node-rank": 0,
            "--pp-size": 4,
            "--tp-size": 4,
            "--dp-size": 1,
            "--attn-dp-size": 1,
            "--attn-cp-size": 4,
            "--cp-strategy": "zigzag",
            "--ep-size": 4,
            "--moe-a2a-backend": "ascend_ops",
            "--served-model-name": "glm-5",
            "--chunked-prefill-size": 8192,
            "--max-prefill-tokens": 8192,
            "--max-running-requests": 128,
            "--mem-fraction-static": "0.6",
            "--load-format": "safetensors",
            "--host": "0.0.0.0",
            "--base-gpu-id": 0,
            "--page-size": 128,
            "--kv-cache-dtype": "bfloat16",
            "--port": 8000,
            "--disaggregation-bootstrap-port": 8998,
        }.items():
            self.option(command, key, value)
        self.assertIn("--enable-prefill-cp", command)
        self.assertIn("--disable-overlap-schedule", command)
        self.assertNotIn("--speculative-algorithm", command)
        self.assertIn("interface=bond0", result.stdout)
        config = json.JSONDecoder().raw_decode(
            result.stdout[result.stdout.index('{\n  "mode"') :]
        )[0]
        self.assertEqual(config["mode"], "prefill_only")
        self.assertEqual(config["execution_slots"], 128)
        self.assertEqual(config["cache_ratio"], 0)
        self.assertEqual(config["host_pool_gib"], 56)
        self.assertIn("896 GiB/node", result.stdout)

    def test_partial_residency_preflight_is_prefill_only(self):
        overrides = {
            "PREFILL_RESIDENT_RATIO": "0.25",
            "EXECUTION_SLOTS": "480",
        }
        prefill = self.run_launcher(overrides={**overrides, "STUB_LOCAL_IPS": IPS[0]})
        self.assertEqual(prefill.returncode, 0, prefill.stderr)
        self.assertIn('"prefill_resident_ratio": 0.25', prefill.stdout)
        self.assertIn('"execution_slots": 480', prefill.stdout)
        self.assertIn("HBM slots/rank by stage: [336, 416, 480, 352]", prefill.stdout)
        decode = self.run_launcher(overrides={**overrides, "STUB_LOCAL_IPS": IPS[1]})
        self.assertEqual(decode.returncode, 0, decode.stderr)
        self.assertNotIn("prefill_resident_ratio", decode.stdout)
        self.assertNotIn("--moe-offload-config", decode.stdout)

    def test_decode_uses_two_node_baseline_without_offload_or_cp(self):
        for rank, address in enumerate(IPS[1:]):
            with self.subTest(rank=rank):
                result = self.run_launcher(overrides={"STUB_LOCAL_IPS": address})
                command = self.command(result)
                for key, value in {
                    "--disaggregation-mode": "decode",
                    "--nnodes": 2,
                    "--node-rank": rank,
                    "--dist-init-addr": f"{IPS[1]}:5000",
                    "--tp-size": 32,
                    "--attn-dp-size": 32,
                    "--ep-size": 32,
                    "--moe-a2a-backend": "deepep",
                    "--deepep-mode": "low_latency",
                    "--cuda-graph-backend-decode": "full",
                    "--max-running-requests": 32,
                    "--mem-fraction-static": "0.97",
                    "--host": address,
                    "--disaggregation-transfer-backend": "ascend",
                    "--page-size": 128,
                    "--kv-cache-dtype": "bfloat16",
                    "--served-model-name": "glm-5",
                    "--port": 8003,
                }.items():
                    self.option(command, key, value)
                for absent in (
                    "--moe-offload-config",
                    "--pp-size",
                    "--attn-cp-size",
                    "--speculative-algorithm",
                    "--load-format",
                    "--cuda-graph-max-bs-decode",
                ):
                    self.assertNotIn(absent, command)
                bs_index = command.index("--cuda-graph-bs-decode")
                self.assertEqual(command[bs_index + 1 : bs_index + 4], ["1", "2", "3"])
                self.assertIn(
                    f"IP={address} role=decode node_rank={rank}", result.stdout
                )

    def test_decode_graph_overrides_preserve_existing_max_bs_option(self):
        for overrides, option, values in (
            ({"D_CUDA_GRAPH_BS": "1 3 5"}, "--cuda-graph-bs-decode", ["1", "3", "5"]),
            ({"D_CUDA_GRAPH_MAX_BS": "4"}, "--cuda-graph-max-bs-decode", ["4"]),
        ):
            with self.subTest(overrides=overrides):
                command = self.command(
                    self.run_launcher(overrides={"STUB_LOCAL_IPS": IPS[1], **overrides})
                )
                start = command.index(option) + 1
                self.assertEqual(command[start : start + len(values)], values)

    def test_router_needs_no_model_or_node_enumeration(self):
        result = self.run_launcher(
            "router",
            overrides={
                "MODEL_PATH": "/missing",
                "STUB_LOCAL_IPS": "",
                "D1_IP": "unused",
            },
        )
        command = self.command(result)
        self.assertIn("sglang_router.launch_router", command)
        self.option(command, "--prefill", f"http://{IPS[0]}:8000")
        self.option(command, "--decode", f"http://{IPS[1]}:8003")
        self.option(command, "--port", 6688)
        self.assertEqual(command.count("--prefill"), 1)
        self.assertEqual(command.count("--decode"), 1)

    def test_invalid_settings_fail_before_start(self):
        cases = (
            ({"D0_IP": IPS[0]}, "distinct node addresses"),
            ({"STUB_LOCAL_IPS": ""}, "exactly one local node IP match"),
            ({"NODE_RANK": "1"}, "conflicts with detected P_IP"),
            ({"ENABLE_MTP": "1"}, "requires ENABLE_MTP=0"),
            ({"HOST_POOL_GIB": "48"}, "HOST_POOL_GIB leaves no room"),
            ({"EXECUTION_SLOTS": "127"}, "at least 128"),
            ({"SGLANG_PP_LAYER_PARTITION": "18,20,24,15"}, "summing to 78"),
            ({"CHUNKED_PREFILL_SIZE": "8193"}, "divisible by page size 128"),
            ({"ASCEND_RT_VISIBLE_DEVICES": "0,1"}, "16 distinct"),
            ({"P_PORT": "0"}, "Invalid TCP port"),
            ({"MEM_FRACTION_STATIC": "nan"}, "MEM_FRACTION_STATIC"),
            ({"STUB_LOCAL_IPS": IPS[1], "D_CUDA_GRAPH_BS": "1 0 3"}, "D_CUDA_GRAPH_BS"),
            ({"STUB_LOCAL_IPS": IPS[1], "D_CUDA_GRAPH_BS": "1\n2 3"}, "D_CUDA_GRAPH_BS"),
            ({"STUB_LOCAL_IPS": IPS[1], "D_CUDA_GRAPH_MAX_BS": "0"}, "D_CUDA_GRAPH_MAX_BS"),
            ({"STUB_LOCAL_IPS": IPS[1], "D_APPLY_HOST_TUNING": "2"}, "D_APPLY_HOST_TUNING"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                result = self.run_launcher(overrides=overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn("Command:", result.stdout)

    def test_actual_launch_generates_only_prefill_config_and_correct_environment(self):
        for index, address in enumerate(IPS):
            with self.subTest(index=index):
                logs = self.root / f"logs-{index}"
                record = self.root / f"launched-{index}.json"
                result = self.run_launcher(
                    dry=False,
                    overrides={
                        "STUB_LOCAL_IPS": address,
                        "STUB_RECORD": str(record),
                        "LOG_DIR": str(logs),
                    },
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                data = json.loads(record.read_text())
                command, env = data["args"], data["env"]
                self.assertEqual(env["SGLANG_HOST_IP"], address)
                self.assertEqual(env["HCCL_SOCKET_IFNAME"], "bond0")
                self.assertEqual(env["ASCEND_MF_STORE_URL"], f"tcp://{IPS[0]}:24707")
                self.assertIsNone(env["SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE"])
                if index == 0:
                    self.assertEqual(env["HCCL_BUFFSIZE"], "1024")
                    self.assertEqual(env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"], "600")
                    self.assertIsNone(env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"])
                    self.assertEqual(env["STUB_P_CUSTOMIZE_SOURCED"], "1")
                    self.assertIsNone(env["STUB_D_CUSTOMIZE_SOURCED"])
                    self.assertFalse((self.root / "host-tuning.log").exists())
                    self.assertEqual(env["SGLANG_PP_LAYER_PARTITION"], "18,20,24,16")
                    self.assertEqual(env["TASK_QUEUE_ENABLE"], "1")
                    self.assertEqual(env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"], "0")
                    config = json.loads((logs / "offload.json").read_text())
                    self.assertEqual(config["mode"], "prefill_only")
                    self.option(command, "--moe-offload-config", logs / "offload.json")
                    self.assertIn(str(self.root / "runtime/python"), env["PYTHONPATH"])
                else:
                    self.assertIsNone(env["HCCL_BUFFSIZE"])
                    self.assertEqual(env["DEEPEP_HCCL_BUFFSIZE"], "200")
                    self.assertEqual(env["ASCEND_MF_LOG_LEVEL"], "2")
                    self.assertEqual(env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"], "1200")
                    self.assertEqual(env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"], "1200")
                    self.assertIsNone(env["STUB_P_CUSTOMIZE_SOURCED"])
                    self.assertEqual(env["STUB_D_CUSTOMIZE_SOURCED"], "1")
                    self.assertEqual(env["LD_LIBRARY_PATH"], str(self.root / "decode custom ops"))
                    self.assertIsNone(env["SGLANG_PP_LAYER_PARTITION"])
                    self.assertEqual(env["TASK_QUEUE_ENABLE"], "0")
                    self.assertEqual(env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"], "1")
                    self.assertEqual(env["SGLANG_NPU_USE_MULTI_STREAM"], "1")
                    self.assertEqual(
                        env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"], "16"
                    )
                    self.assertFalse((logs / "offload.json").exists())
                    self.assertNotIn("--moe-offload-config", command)
                    self.assertNotIn(
                        str(self.root / "runtime/python"), env["PYTHONPATH"]
                    )
                    tuning = (self.root / "host-tuning.log").read_text().splitlines()
                    self.assertEqual(
                        tuning[-4:],
                        ["governor performance", "-w vm.swappiness=0", "-w kernel.numa_balancing=0", "-w kernel.sched_migration_cost_ns=50000"],
                    )

    def test_deepep_switch_passes_normal_settings_to_prefill_only(self):
        for index, address in enumerate(IPS[:2]):
            with self.subTest(index=index):
                record = self.root / f"deepep-{index}.json"
                result = self.run_launcher(
                    dry=False,
                    overrides={
                        "STUB_LOCAL_IPS": address,
                        "STUB_RECORD": str(record),
                        "LOG_DIR": str(self.root / f"logs-deepep-{index}"),
                        "P_MOE_A2A_BACKEND": "deepep",
                        "D_APPLY_HOST_TUNING": "0",
                    },
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                data = json.loads(record.read_text())
                command, env = data["args"], data["env"]
                self.option(command, "--moe-a2a-backend", "deepep")
                if index == 0:
                    self.option(command, "--deepep-mode", "normal")
                    self.option(command, "--deepep-dispatcher-output-dtype", "int8")
                    self.assertIn("--moe-offload-config", command)
                    self.assertNotIn("--moe-ascend-ops-config", command)
                    self.assertEqual(env["DEEPEP_HCCL_BUFFSIZE"], "1024")
                    self.assertEqual(env["DEEPEP_NORMAL_LONG_SEQ_ROUND"], "72")
                    self.assertEqual(env["DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS"], "1024")
                    self.assertEqual(env["DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ"], "1")
                    self.assertEqual(env["DEEP_NORMAL_MODE_USE_INT8_QUANT"], "1")
                    self.assertEqual(env["MOE_EXPERT_TOKEN_NUMS_TYPE"], "1")
                else:
                    self.option(command, "--deepep-mode", "low_latency")
                    self.assertNotIn("--moe-offload-config", command)
                    self.assertIsNone(env["DEEPEP_NORMAL_LONG_SEQ_ROUND"])
                    self.assertIsNone(env["DEEP_NORMAL_MODE_USE_INT8_QUANT"])

    def test_decode_can_skip_host_tuning(self):
        result = self.run_launcher(
            dry=False,
            overrides={"STUB_LOCAL_IPS": IPS[1], "D_APPLY_HOST_TUNING": "0"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.root / "host-tuning.log").exists())

    def test_missing_decode_environment_fails_before_host_tuning(self):
        result = self.run_launcher(
            dry=False,
            overrides={"STUB_LOCAL_IPS": IPS[1], "D_CUSTOMIZE_ENV": str(self.root / "missing")},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "host-tuning.log").exists())
        self.assertIn("Missing Ascend environment script", (self.root / "logs/server.log").read_text())


if __name__ == "__main__":
    unittest.main()
