"""Merged cache/public-program contracts; CPU fixtures are not NPU backends."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import test_cache_runtime as fixtures
import torch

from sglang.srt.offloading.moe import (
    ExecutionRequest,
    ExecutionReservation,
    ExpertWeightAccess,
    LocalExecution,
    LocalRoutes,
    Phase,
    SourcePolicy,
)
from sglang.srt.offloading.moe.config import fixed_waves
from sglang.srt.offloading.moe.interfaces import DispatchLayoutAdapter, MoEDispatcher
from sglang.srt.offloading.moe.prefill import PrefillWeightPipeline
from sglang.srt.offloading.moe.programs import (
    DispatchOnceProgram,
    PrefillDispatchOnceProgram,
    PrefillLocalExpertProgram,
)
from sglang.srt.offloading.transfer import (
    HostCopyPlan,
    HostCopyRegion,
    ImmediateCompletion,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class OneDispatch(MoEDispatcher):
    def __init__(self):
        self.calls = []

    def dispatch(self, hidden, routing, context):
        self.calls.append("D")
        return routing

    def dispatch_completion(self, dispatched, context):
        self.calls.append("D_ready")
        return context.record_compute_completion().completion

    def combine(self, output, context):
        self.calls.append("C")
        return output

    def finish(self, output, context):
        return output


class ScalarLayout(DispatchLayoutAdapter):
    """Test F sums bundles; retain masks and accumulate its scalar wave outputs."""

    def bind(self, dispatched, context):
        return SimpleNamespace(routes=dispatched, output=torch.tensor(0.0))

    def select_wave(self, layout, wave, context):
        routes = layout.routes
        return LocalRoutes(
            routes.expert_ids,
            routes.valid
            & (routes.expert_ids >= wave.local_start)
            & (routes.expert_ids < wave.local_end),
        )

    def write_wave(self, layout, wave, output, context):
        layout.output.add_(output)

    def combine_input(self, layout, context):
        return layout.output


class BorrowedHostResidency:
    """Test-only prefill reservation provider over the runtime's borrowed hot pool.

    The production backend supplies its own host-region/address layout and A/B
    reservations. Cache residency remains responsible for decode maps/refill.
    """

    def __init__(self, backend, catalog):
        self.backend, self.catalog = backend, catalog
        self.reader = ImmediateCompletion()

    def resolve_execution(self, request, context):
        assert request.source_policy is SourcePolicy.HOST_ONLY
        wave = request.wave
        offset = self.catalog.key(request.layer_id, wave.local_start)
        width = wave.local_end - wave.local_start
        source = self.backend.host[offset : offset + width].view(torch.uint8).flatten()
        destination = self.backend.hot[:width].view(torch.uint8).flatten()
        plan = HostCopyPlan(
            (
                HostCopyRegion(
                    source,
                    destination,
                    0,
                    0,
                    source.numel(),
                    source.numel(),
                    destination.numel(),
                ),
            )
        )
        return ExecutionReservation(
            plan,
            SimpleNamespace(
                destination=self.backend.hot[:width], start=wave.local_start
            ),
            copy_dependencies=(self.reader,),
        )

    def publish_execution(self, reservation, copied, context):
        copied.completion.wait_on(context.compute_stream)
        return context.record_compute_completion()

    def finish_execution(self, reservation, read_done, context):
        self.reader = read_done.completion


class HostTransport:
    def __init__(self, log):
        self.log, self.copies = log, 0

    def materialize_host_into(self, plan, bindings, stream):
        self.copies += 1

        def copy():
            for region in plan.regions:
                region.destination.copy_(region.source)

        return TransferTicket(fixtures.DeferredCopy(copy, self.log), (plan, bindings))


class SequentialPipeline(PrefillWeightPipeline):
    """No optimization/worker: exercise public prefill hooks with real CPU copies."""

    def __init__(self, access):
        super().__init__(access)
        self.session = None
        self.calls = []

    def start(self, layer_id, context):
        self.session = (layer_id, context)
        self.calls.append("start")

    def validate_session(self, layer, context):
        if self.session != (layer.layer_id, context):
            raise RuntimeError("Missing prefill session")

    def seal(self, layer, dispatch_ready, context):
        dispatch_ready.wait_on(context.compute_stream)
        self.calls.append("seal")
        return fixed_waves(4, 2)

    def acquire(self, layer, wave, context):
        self.calls.append("acquire")
        return self.access.stage(
            ExecutionRequest(
                layer.layer_id, None, wave, Phase.PREFILL, SourcePolicy.HOST_ONLY
            ),
            context,
        )

    def finish(self, layer, context):
        self.calls.append("finish")
        self.session = None
        return context.record_compute_completion()

    def abort(self, context):
        self.calls.append("abort")
        self.session = None
        return context.record_compute_completion()


class TestMergedCacheIntegration(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TestCacheRuntime()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.runtime = self.fixture.runtime
        self.backend = self.fixture.backend

    def request(self, ids=(1, 1), wave=None):
        ids = torch.tensor(ids, dtype=torch.int64)
        return self.runtime.make_execution_request(
            3,
            LocalRoutes(ids, ids >= 0),
            wave or fixed_waves(4, 2)[0],
            self.runtime.context,
        )

    def test_coordinator_finishes_lfu_once_and_clears_due(self):
        policy = self.runtime.cache.policy
        with patch.object(policy, "finish_step", wraps=policy.finish_step) as finish:
            for _ in range(4):
                self.fixture.run_step({3: [1, 1]})
        self.assertEqual(finish.call_count, 4)
        state = self.runtime.cache.policy_state
        self.assertAlmostEqual(float(state.usage[1]), 5.6, places=5)
        self.assertFalse(state.update_due)
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [1, 2, 4, 7])

    def test_execution_publication_ticket_is_after_copy_and_mapping(self):
        ops, ctx = self.runtime.residency_ops, self.runtime.context
        reservation = ops.resolve_execution(self.request(), ctx)
        copied = self.backend.transfer.materialize_into(
            reservation.copy_plan, reservation.bindings, ctx.copy_stream
        )
        self.assertFalse(self.runtime.residency.execution_valid.any())
        self.assertFalse(copied.completion.done())
        published = ops.publish_execution(reservation, copied, ctx)
        self.assertIsInstance(published, TransferTicket)
        self.assertTrue(copied.completion.done())
        self.assertEqual(self.runtime.residency.execution_key.tolist(), [-1, 1])
        self.assertEqual(self.backend.device.log[-1][0], "record")
        ops.finish_execution(reservation, ctx.record_compute_completion(), ctx)

    def test_bank_and_plan_wait_for_auxiliary_reader_before_reuse(self):
        access, ctx = self.runtime.access, self.runtime.context
        lease = access.stage(self.request(), ctx)
        before_keys = self.runtime.residency.execution_key.clone()
        before_weights = self.backend.execution.clone()

        def read_old_weights():
            torch.testing.assert_close(
                self.runtime.residency.execution_key, before_keys
            )
            torch.testing.assert_close(self.backend.execution, before_weights)

        reader = fixtures.DeferredCopy(read_old_weights, self.backend.device.log)
        access.release_after(lease, TransferTicket(reader), ctx)
        self.assertFalse(reader.done())
        next_lease = access.stage(self.request((0,)), ctx)
        self.assertTrue(reader.done())
        self.assertEqual(self.runtime.residency.execution_key.tolist(), [0, -1])
        access.release_after(next_lease, ctx.record_compute_completion(), ctx)

    def test_unreleased_execution_cannot_be_overwritten_or_evicted(self):
        access, ops, ctx = (
            self.runtime.access,
            self.runtime.residency_ops,
            self.runtime.context,
        )
        lease = access.stage(self.request(), ctx)
        for action in (
            lambda: ops.resolve_execution(self.request(), ctx),
            lambda: ops.prepare_update(self.runtime.residency.slot_to_key, ctx),
        ):
            with self.assertRaisesRegex(RuntimeError, "Release"):
                action()
        access.release_after(lease, ctx.record_compute_completion(), ctx)

    def test_resources_idle_joins_worker_tickets_and_retains_objects(self):
        ctx = self.runtime.context
        completed = []
        tickets = [
            TransferTicket(
                fixtures.DeferredCopy(
                    lambda i=i: completed.append(i), self.backend.device.log
                )
            )
            for i in range(20)
        ]
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(ctx.retain, tickets))
        done = ctx.finish()
        ctx.clear()
        self.assertTrue(done.output_ready.completion.done())
        self.assertFalse(done.resources_idle.completion.done())
        done.resources_idle.completion.synchronize()
        self.assertEqual(sorted(completed), list(range(20)))
        self.assertTrue(done.resources_idle.completion.done())

    def test_forward_context_identity_changes_and_capture_stays_retained(self):
        before = self.runtime.context
        self.fixture.run_step({3: [1]})
        first = self.runtime.context
        self.fixture.run_step({3: [1]})
        self.assertIsNot(before, first)
        self.assertIsNot(first, self.runtime.context)
        self.runtime._capture_head(self.fixture.runner, self.fixture.batch(), 1)
        captured = self.runtime.context
        self.fixture.model({3: [1]})
        self.runtime._capture_tail(self.fixture.runner, None, self.fixture.batch(), 1)
        self.fixture.run_step({3: [1]})
        self.assertTrue(captured.capturing)
        self.assertTrue(captured.retained())
        self.assertIn(captured, self.runtime._capture_contexts)

    def test_dispatch_once_counts_assignments_once_and_deduplicates_copies(self):
        dispatcher = OneDispatch()
        self.fixture.runner.model.decode = DispatchOnceProgram(
            fixed_waves(4, 2),
            dispatcher,
            self.runtime.decode_program.local_program,
            ScalarLayout(),
            local_experts=4,
        )
        self.fixture.run_step({3: [1, 1, 2, -1]})
        self.assertEqual(dispatcher.calls, ["D", "C"])
        self.assertEqual(
            self.runtime.cache.policy_state.usage.tolist(), [0, 2, 1, 0, 0, 0, 0, 0]
        )
        copies = [c for c in self.backend.transfer.calls if c["pool"] == "execution"]
        self.assertEqual([len(c["entries"]) for c in copies], [1, 1])

    def make_prefill(self, fail_native=False):
        transport = HostTransport(self.backend.device.log)
        ops = BorrowedHostResidency(self.backend, self.runtime.cache.catalog)
        pipeline = SequentialPipeline(ExpertWeightAccess(ops, transport))
        self.runtime.prefill_pipeline = pipeline
        dispatcher = OneDispatch()

        def native(layer, routes, bindings, ctx):
            if fail_native:
                raise RuntimeError("prefill native failed")
            ids = routes.expert_ids[routes.valid] - bindings.start
            return LocalExecution(
                bindings.destination[ids].sum(), ctx.record_compute_completion()
            )

        self.runtime.prefill_program = PrefillDispatchOnceProgram(
            dispatcher,
            PrefillLocalExpertProgram(pipeline, native),
            ScalarLayout(),
            local_experts=4,
            pipeline=pipeline,
        )
        return pipeline, dispatcher, transport

    def test_prefill_hooks_host_copies_then_exact_refill_without_lfu(self):
        self.fixture.run_step({3: [1, 1]})
        saved = self.runtime.cache.policy_state.usage.clone()
        pipeline, dispatcher, transport = self.make_prefill()
        ids = torch.tensor([1, 1, 3, -1])
        with self.fixture.offloader.forward_scope(self.fixture.batch("EXTEND")):
            ctx = self.runtime.context
            self.runtime.emit_layer_start(3, ctx)
            output = self.runtime.prefill_program.run(
                self.fixture.layers[3], None, LocalRoutes(ids, ids >= 0), ctx
            )
            self.runtime.emit_step_end(ctx)
            torch.testing.assert_close(output, self.backend.host[[1, 1, 3]].sum())
        self.assertEqual(
            pipeline.calls, ["start", "seal", "acquire", "acquire", "finish"]
        )
        self.assertEqual(dispatcher.calls, ["D", "D_ready", "C"])
        self.assertEqual(transport.copies, 2)
        torch.testing.assert_close(self.runtime.cache.policy_state.usage, saved)
        with self.fixture.offloader.forward_scope(self.fixture.batch()):
            torch.testing.assert_close(
                self.backend.hot, self.backend.host[[0, 2, 4, 7]]
            )
            self.assertEqual(int(self.runtime.cache.policy_state.step_mod), 1)
        self.assertFalse(self.runtime.cache._has_snapshot)

    def test_prefill_failure_aborts_and_keeps_inflight_resources(self):
        pipeline, _, _ = self.make_prefill(fail_native=True)
        with self.assertRaisesRegex(RuntimeError, "prefill native failed"):
            with self.fixture.offloader.forward_scope(self.fixture.batch("EXTEND")):
                ctx = self.runtime.context
                self.runtime.emit_layer_start(3, ctx)
                ids = torch.tensor([1])
                self.runtime.prefill_program.run(
                    self.fixture.layers[3], None, LocalRoutes(ids, ids >= 0), ctx
                )
        self.assertIn("abort", pipeline.calls)
        self.assertIsNone(pipeline.session)
        self.assertTrue(self.runtime.context.retained())
        self.assertTrue(self.runtime.cache._has_snapshot)

    def test_decode_cleanup_does_not_abort_prefill_with_decode_context(self):
        pipeline, _, _ = self.make_prefill()
        with patch.object(
            pipeline, "abort", side_effect=AssertionError("decode abort")
        ):
            self.fixture.run_step({3: [1]})
            self.fixture.offloader.close()

    def test_prefill_cleanup_stops_submitter_before_device_drain(self):
        pipeline, _, _ = self.make_prefill()
        calls = []
        with self.fixture.offloader.forward_scope(self.fixture.batch("EXTEND")):
            self.runtime.emit_layer_start(3, self.runtime.context)
        with (
            patch.object(
                pipeline,
                "abort",
                side_effect=lambda ctx: (
                    calls.append(("abort", ctx.phase)),
                    ctx.record_compute_completion(),
                )[1],
            ),
            patch.object(
                self.backend.device,
                "synchronize",
                side_effect=lambda: calls.append("drain"),
            ),
        ):
            self.runtime.synchronize()
        self.assertEqual(calls, [("abort", Phase.PREFILL), "drain"])


if __name__ == "__main__":
    unittest.main()
