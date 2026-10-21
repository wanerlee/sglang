"""Run NPU graph backend lifecycle methods without importing the NPU stack."""

import ast
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNPUMoEGraphLifetime(unittest.TestCase):
    def test_backend_without_moe_preserves_existing_constructor(self):
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        init = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        namespace = {
            "ThreadPoolExecutor": lambda **kwargs: None,
            "get_parallel": lambda: SimpleNamespace(tp_group=None),
            "TorchMemorySaverAdapter": SimpleNamespace(create=lambda **kwargs: None),
        }
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, init], type_ignores=[])
                ),
                str(source),
                "exec",
            ),
            namespace,
        )
        device = SimpleNamespace(
            current_device=lambda: 0, set_device=lambda device: None
        )
        for owner in (None, object()):
            runner = SimpleNamespace(device_module=device)
            if owner is not None:
                runner.model_runner = SimpleNamespace(
                    offloader=SimpleNamespace(owner=owner)
                )
            backend = SimpleNamespace()
            namespace["__init__"](backend, runner)
            self.assertIs(backend._moe_owner, owner)
            self.assertEqual(backend._moe_graph_leases, {})

    def _graph_backend(self):
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("capture_one", "can_run", "cleanup")
        ]
        log = []
        graphs = {}

        class Graph:
            def __del__(self):
                log.append("graph_destroy")

        class Lease:
            def close(self):
                self_test.assertFalse(graphs)
                self_test.assertIn("graph_destroy", log)
                log.append("lease_close")

        self_test = self
        npu = SimpleNamespace(
            NPUGraph=Graph, graph=lambda *args, **kwargs: nullcontext()
        )
        namespace = {"torch": SimpleNamespace(npu=npu), "empty_context": nullcontext}
        # Postponed annotations avoid importing graph runner/typing dependencies.
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future] + methods, type_ignores=[])
                ),
                str(source),
                "exec",
            ),
            namespace,
        )
        backend = SimpleNamespace(
            _graphs=graphs,
            _outputs={},
            _pool=None,
            _capture_stream=None,
            _device_module=SimpleNamespace(synchronize=lambda: log.append("sync")),
            _tp_group=SimpleNamespace(barrier=lambda: None),
            _enable_torch_compile=False,
            _memory_saver_adapter=None,
            _moe_owner=SimpleNamespace(
                acquire_graph=lambda key: log.append(("acquire", key)) or Lease()
            ),
            _moe_graph_leases={},
            _update_executor=SimpleNamespace(
                shutdown=lambda **kwargs: log.append("shutdown")
            ),
        )
        return namespace, backend, log

    def test_graph_lease_lives_until_cleanup_after_device_drain(self):
        namespace, backend, log = self._graph_backend()
        # Only this key is patched; avoid reverting unrelated torch module imports.
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            namespace["capture_one"](backend, "bucket", lambda: torch.ones(1))
        self.assertEqual(len(backend._moe_graph_leases), 1)
        self.assertEqual(log.count("lease_close"), 0)
        namespace["cleanup"](backend)
        self.assertLess(log.index("shutdown"), log.index("graph_destroy"))
        self.assertEqual(log[-3:], ["sync", "graph_destroy", "lease_close"])
        self.assertFalse(backend._moe_graph_leases)

    def test_failed_capture_is_retained_but_not_replayable_until_cleanup(self):
        namespace, backend, log = self._graph_backend()
        calls = []

        def forward():
            calls.append(True)
            if len(calls) == 3:  # Warmups succeed; fail inside actual capture.
                raise RuntimeError("capture failed")

        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                namespace["capture_one"](backend, "bucket", forward)
        self.assertIn("bucket", backend._graphs)
        self.assertFalse(namespace["can_run"](backend, None, "bucket"))
        self.assertNotIn("lease_close", log)
        namespace["cleanup"](backend)
        self.assertEqual(log[-3:], ["sync", "graph_destroy", "lease_close"])

    def test_duplicate_capture_is_rejected_before_warmup(self):
        namespace, backend, log = self._graph_backend()
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            namespace["capture_one"](backend, "bucket", lambda: None)
            original_graph = backend._graphs["bucket"]
            original_lease = backend._moe_graph_leases["bucket"]

            def reject(key):
                raise ValueError("Graph bucket already bound")

            backend._moe_owner.acquire_graph = reject
            with self.assertRaisesRegex(ValueError, "already bound"):
                namespace["capture_one"](
                    backend, "bucket", lambda: self.fail("Unexpected warmup")
                )
        self.assertIs(backend._graphs["bucket"], original_graph)
        self.assertIs(backend._moe_graph_leases["bucket"], original_lease)
        del original_graph
        namespace["cleanup"](backend)


if __name__ == "__main__":
    unittest.main()
