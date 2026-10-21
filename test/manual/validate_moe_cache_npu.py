"""One-command cache validation; run from any directory with the NPU Python.

    python test/manual/validate_moe_cache_npu.py --device 0

No model download, package installation or repository mutation. Child processes
isolate CPU hardware doubles from real NPU tests. Exit 0 means all listed suites
ran without skips; 1 means failure/timeout; 2 means incomplete hardware validation.
Read report.json's coverage limits before interpreting a successful run.
"""

import argparse
import datetime
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGES = {
    "cpu": "All offloading CPU UT; communication/accelerator doubles where declared",
    "npu_runtime": "Real eager H2D/D2D, events, LFU replacement and exact refill",
    "npu_metadata_graph": "Real NPU metadata/LFU capture and changing-input replay",
    "npu_layout": "Production AscendDispatchLayout on NPU, eager and graph replay",
}
LIMITS = [
    "No real DeepEP/HCCL communication or multi-rank EP validation.",
    "No native Ascend MoE grouped-matmul/quantization kernel validation; layout tests use arithmetic F.",
    "No colleague production transfer/prefill pipeline or complete weight-transfer graph validation.",
    "No full DeepSeek V4 inference or performance validation.",
    "Model installer/CacheBackend and graph lease lifecycle are covered by CPU doubles, not an NPU serving model.",
]


def layout_suite(torch):
    """Exercise actual layout tensor ops, with scripted compact receive buffers."""
    from sglang.srt.offloading.moe.ascend_adapter import AscendDispatchLayout
    from sglang.srt.offloading.moe.cache_runtime import CacheExecutionContext
    from sglang.srt.offloading.moe.config import Phase, fixed_waves

    class TestAscendLayoutNPU(unittest.TestCase):
        def setUp(self):
            self.device = f"npu:{os.environ['MOE_TEST_NPU_DEVICE']}"
            self.adapter = AscendDispatchLayout(5, 8, 4, torch.float32, self.device)
            self.x = torch.zeros((8, 4), device=self.device)
            self.scales = torch.zeros((8, 1), device=self.device)
            self.counts = torch.zeros(5, dtype=torch.int64, device=self.device)
            self.context = CacheExecutionContext(
                torch.npu, torch.npu.current_stream(), torch.npu.current_stream()
            )
            self.received = types.SimpleNamespace(
                hidden_states=self.x,
                hidden_states_scale=self.scales,
                masked_m=self.counts,
                num_recv_tokens_per_expert=[1, 0, 2, 0, 1],
                format=types.SimpleNamespace(
                    is_deepep_ll=lambda: True, is_deepep_normal=lambda: False
                ),
            )

        def tearDown(self):
            torch.npu.synchronize()
            self.context.clear()

        def execute(self):
            layout = self.adapter.bind(self.received, self.context)
            # Five experts / two lanes includes a partial tail and empty waves.
            for wave in fixed_waves(5, 2):
                selected = self.adapter.select_wave(layout, wave, self.context)
                output = selected.hidden_states * selected.hidden_states_scale
                # Padding must be discarded even when F returns NaNs there.
                output = torch.where(selected.valid[:, None], output, float("nan"))
                self.adapter.write_wave(layout, wave, output, self.context)
            return layout.output

        def check_sequence(self, graph=None):
            address = self.adapter.output.data_ptr()
            for step, counts in enumerate(
                ([1, 0, 2, 0, 1], [0, 0, 0, 0, 0], [0, 3, 0, 1, 4])
            ):
                x = torch.arange(32, dtype=torch.float32).reshape(8, 4) + step
                scale = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1)
                self.x.copy_(x)
                self.scales.copy_(scale)
                self.counts.copy_(torch.tensor(counts, dtype=torch.int64))
                if graph is None:
                    actual = self.execute()
                else:
                    graph.replay()
                    actual = self.adapter.output
                torch.npu.synchronize()
                expected = x * scale
                expected[sum(counts) :] = 0
                torch.testing.assert_close(actual.cpu(), expected)
                self.assertEqual(self.adapter.output.data_ptr(), address)

        def test_eager_ll_and_normal_prefill(self):
            self.check_sequence()
            self.context.phase = Phase.PREFILL
            self.received.format = types.SimpleNamespace(
                is_deepep_ll=lambda: False, is_deepep_normal=lambda: True
            )
            actual = self.execute()
            torch.npu.synchronize()
            expected = self.x.cpu() * self.scales.cpu()
            expected[4:] = 0
            torch.testing.assert_close(actual.cpu(), expected)

        def test_layout_capture_replay_with_changing_counts(self):
            stream = torch.npu.Stream()
            torch.npu.synchronize()
            self.context.compute_stream = stream
            with torch.npu.stream(stream):
                for _ in range(2):
                    self.execute()
            torch.npu.synchronize()
            self.context.capturing = True
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                self.execute()
            torch.npu.synchronize()
            self.check_sequence(graph)

    return unittest.defaultTestLoader.loadTestsFromTestCase(TestAscendLayoutNPU)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_worker(stage, device, output_dir):
    report = {"stage": stage, "status": "failed", "coverage": STAGES[stage]}
    exit_code = 1
    try:
        # Load this checkout's real submodules without the unrelated serving API
        # initializer. This is package isolation, not a replacement of cache code.
        package = types.ModuleType("sglang")
        package.__path__ = [str(ROOT / "python/sglang")]
        sys.modules["sglang"] = package
        sys.path.insert(0, str(ROOT / "python"))
        os.environ["MOE_TEST_NPU_DEVICE"] = str(device)
        import torch

        report["torch"] = torch.__version__
        if stage != "cpu":
            try:
                import torch_npu
            except ImportError:
                report["status"] = "unavailable"
                raise
            report["torch_npu"] = torch_npu.__version__
            if not torch.npu.is_available():
                report["status"] = "unavailable"
                raise RuntimeError("No Ascend NPU available")
            torch.npu.set_device(device)
            report["device"] = f"npu:{device}"
            report["device_name"] = torch.npu.get_device_name(device)
            probe = torch.ones(1, device=f"npu:{device}") + 1
            torch.npu.synchronize()
            assert probe.cpu().item() == 2

        if stage == "cpu":
            suite = unittest.defaultTestLoader.discover(
                str(ROOT / "test/registered/unit/offloading"), pattern="test_*.py"
            )
        elif stage == "npu_layout":
            suite = layout_suite(torch)
        else:
            module_name = {
                "npu_runtime": "test_moe_cache_runtime_npu",
                "npu_metadata_graph": "test_moe_cache_metadata_npu",
            }[stage]
            suite = unittest.defaultTestLoader.loadTestsFromModule(
                importlib.import_module(module_name)
            )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report.update(
            tests_run=result.testsRun,
            failures=[(str(test), details) for test, details in result.failures],
            errors=[(str(test), details) for test, details in result.errors],
            skipped=[(str(test), reason) for test, reason in result.skipped],
            expected_failures=[str(test) for test, _ in result.expectedFailures],
            unexpected_successes=[str(test) for test in result.unexpectedSuccesses],
        )
        if not result.wasSuccessful():
            report["status"] = "failed"
        elif result.skipped or result.expectedFailures or result.testsRun == 0:
            report["status"] = "incomplete"
        else:
            report["status"] = "passed"
        exit_code = {"passed": 0, "failed": 1, "incomplete": 2}[report["status"]]
    except Exception:
        report["exception"] = traceback.format_exc()
        print(report["exception"], file=sys.stderr)
        exit_code = 2 if report["status"] == "unavailable" else 1
    report["exit_code"] = exit_code
    write_json(output_dir / f"{stage}.json", report)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Visible NPU index")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=900, help="Seconds per suite")
    parser.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0 or args.timeout <= 0:
        parser.error("device must be nonnegative and timeout must be positive")
    output_dir = args.output_dir or ROOT / (
        "moe-cache-validation-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage:
        return run_worker(args.stage, args.device, output_dir)
    # Reject accidental reuse, so an old success cannot masquerade as this run.
    if any(output_dir.iterdir()):
        parser.error("output-dir must be empty; use a new directory per run")

    report = {
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkout": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "device": args.device,
        "coverage_limits": LIMITS,
        "package_bootstrap": "Checkout submodules; serving API initializer bypassed",
        "stages": [],
    }
    for name, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("working_tree", ["git", "status", "--short", "--untracked-files=no"]),
    ):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        report[name] = result.stdout.strip() if result.returncode == 0 else "unknown"

    for stage in STAGES:
        print(f"[{stage}] {STAGES[stage]}", flush=True)
        started = time.monotonic()
        log_path = output_dir / f"{stage}.log"
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--stage",
            stage,
            "--device",
            str(args.device),
            "--output-dir",
            str(output_dir),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    env={**os.environ, "PYTHONUTF8": "1"},
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = -1
                log.write(f"\nTIMEOUT after {args.timeout} seconds\n")
        stage_path = output_dir / f"{stage}.json"
        detail = (
            json.loads(stage_path.read_text(encoding="utf-8"))
            if stage_path.exists()
            else {"stage": stage, "status": "failed", "reason": "See worker log"}
        )
        if returncode not in (0, 2) or (
            returncode != 0 and detail["status"] == "passed"
        ):
            detail["status"] = "failed"
        detail.update(
            process_exit_code=returncode,
            seconds=round(time.monotonic() - started, 2),
            log=str(log_path),
        )
        report["stages"].append(detail)
        print(f"  {detail['status']} -> {log_path}", flush=True)
        write_json(output_dir / "report.json", report)

    statuses = [entry["status"] for entry in report["stages"]]
    exit_code = 1 if "failed" in statuses else (0 if set(statuses) == {"passed"} else 2)
    report["exit_code"] = exit_code
    report["status"] = {0: "passed_within_scope", 1: "failed", 2: "incomplete"}[
        exit_code
    ]
    write_json(output_dir / "report.json", report)
    print(f"\n{report['status']}: {output_dir / 'report.json'}", flush=True)
    print("Coverage limits:\n- " + "\n- ".join(LIMITS), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
