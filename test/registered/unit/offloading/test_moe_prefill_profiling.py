"""Prefill trace attribution; transfers/GMM are simulated, no NPU is required."""

import unittest
from contextlib import contextmanager, nullcontext
from threading import local
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.offloading.moe import prefill, programs
from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.execution import (
    ExecutionRequest,
    LocalExecution,
    SourcePolicy,
)
from sglang.srt.offloading.moe.runtime import ManagedExecutionContext
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillProfiling(unittest.TestCase):
    def run_pipeline(self, local_experts=6):
        actions, spans = [], []
        active = local()
        original_record = torch.profiler.record_function

        @contextmanager
        def record(name):
            active.name = name
            spans.append(name)
            try:
                with original_record(name):
                    yield
            finally:
                active.name = None

        def ticket():
            return TransferTicket(ImmediateCompletion())

        class Access:
            def stage(self, request, context):
                actions.append(("load", request.wave, active.name))
                return SimpleNamespace(
                    bindings=request.wave, ticket=ticket(), read_done=None
                )

            def combine_prefill(self, leases, bindings, context):
                return SimpleNamespace(
                    bindings=bindings, ticket=ticket(), read_done=None
                )

            def wait_ready(self, lease, stream):
                actions.append(("ready", lease.bindings, active.name))
                return lease.bindings

            def release_after(self, lease, read_done, context):
                lease.read_done = read_done

        context = ManagedExecutionContext(
            Phase.PREFILL,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=lambda stream: ticket(),
            failure_drain=lambda: self.fail("Profiling must not add a device drain"),
        )
        pipeline = prefill.BoundedPrefillWeightPipeline(
            Access(),
            prefill.PrefillPipelineConfig(
                local_experts=local_experts,
                initial_capacity=2,
                group_size=2,
                secondary_capacity=2,
                microbatch_size=1,
                max_inflight=2,
            ),
            make_request=lambda layer, wave, offset, ctx: ExecutionRequest(
                layer, None, wave, Phase.PREFILL, SourcePolicy.HOST_ONLY
            ),
            combine_bindings=lambda layer, wave, leases, ctx: wave,
            worker_scope=lambda ctx: nullcontext(),
            record_copy_completion=lambda ctx: ticket(),
        )

        def run_experts(layer, value, wave, ctx):
            actions.append(("compute", wave, active.name))
            # Real CPU operation checks that compute events nest in the span.
            return LocalExecution(value.clone(), ticket())

        program = programs.PrefillLocalExpertProgram(pipeline, run_experts)
        with (
            patch.object(prefill, "record_function", record),
            patch.object(programs, "record_function", record),
        ):
            pipeline.start(10, context)
            # Model a dispatch that lasts until the initial bank is full.
            # Otherwise immediate dispatch completion can seal at one expert.
            worker = pipeline._session(10, context).worker
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            waves = pipeline.seal(10, ImmediateCompletion(), context)
            value = torch.ones(2)
            for wave in waves:
                value = program.run(10, value, wave, context)
            pipeline.finish(10, context)
        torch.testing.assert_close(value, torch.ones(2))
        return context.invocation_id, actions, spans

    def test_initial_and_next_loads_are_attributed_to_consuming_wave(self):
        forward, actions, spans = self.run_pipeline()
        self.assertTrue(all(f"forward={forward} layer=10 " in s for s in spans))
        loads = [item for item in actions if item[0] == "load"]
        self.assertEqual(len(loads), 4)
        for microbatch, (_, wave, span) in enumerate(loads[:2]):
            self.assertIn(f"kind=initial wave=0 microbatch={microbatch}", span)
            self.assertIn(f"experts=[{microbatch},{microbatch + 1}) bank=0", span)
        for index, (_, wave, span) in enumerate(loads[2:], 1):
            self.assertIn(f"kind=next current_wave={index - 1} wave={index}", span)
            self.assertIn(
                f"experts=[{wave.local_start},{wave.local_end}) bank={wave.bank}", span
            )
            load_pos = actions.index(loads[index + 1])
            compute_pos = next(
                i
                for i, (op, w, _) in enumerate(actions)
                if op == "compute" and w.index == index - 1
            )
            self.assertLess(load_pos, compute_pos)
        for op, wave, span in actions:
            if op in ("ready", "compute"):
                self.assertTrue(span.startswith("moe.prefill.compute_submit "))
                self.assertIn(f"wave={wave.index} ", span)
                self.assertIn(f"experts=[{wave.local_start},{wave.local_end})", span)

    def test_single_wave_has_no_next_prefetch_and_forward_ids_do_not_repeat(self):
        first, _, spans = self.run_pipeline(local_experts=2)
        second, _, _ = self.run_pipeline(local_experts=2)
        self.assertNotEqual(first, second)
        self.assertFalse(any("kind=next" in s for s in spans))
        self.assertEqual(sum("compute_submit" in s for s in spans), 1)

    def test_cpu_profiler_records_next_load_and_compute_parentage(self):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            self.run_pipeline()
        events = prof.events()
        loads = [e for e in events if e.name.startswith("moe.prefill.load_submit ")]
        self.assertEqual(sum("kind=next" in e.name for e in loads), 2)
        computes = [
            e for e in events if e.name.startswith("moe.prefill.compute_submit ")
        ]
        self.assertEqual(len(computes), 3)
        for event in computes:
            self.assertTrue(
                any(child.name == "aten::clone" for child in event.cpu_children)
            )


if __name__ == "__main__":
    unittest.main()
