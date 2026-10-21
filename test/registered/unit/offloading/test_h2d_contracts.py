"""CPU protocol tests; no ACL implementation or accelerator graph certification."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe.config import ExpertWave, Phase
from sglang.srt.offloading.moe.execution import (
    CacheUpdateCoordinator,
    ExecutionRequest,
    ExecutionReservation,
    ExpertWeightAccess,
    LocalExecution,
    SourcePolicy,
    UpdatePlan,
)
from sglang.srt.offloading.moe.interfaces import (
    BoundMoERuntime,
    ExecutionContext,
    MoEDispatcher,
    ResidencyOps,
)
from sglang.srt.offloading.moe.policy import TensorLFUPolicy
from sglang.srt.offloading.moe.prefill import PrefillWeightPipeline
from sglang.srt.offloading.moe.programs import (
    DecodeLocalExpertProgram,
    PrefillDispatchOnceProgram,
    PrefillLocalExpertProgram,
)
from sglang.srt.offloading.transfer import (
    Completion,
    ComposedTransferBackend,
    DeviceCopyPlan,
    DeviceTransferOps,
    HostCopyPlan,
    HostCopyRegion,
    HostTransferOps,
    TorchTransferBackend,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Fence(Completion):
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def wait_on(self, stream):
        self.calls.append(("wait", self.name, stream))

    def synchronize(self):
        raise AssertionError("Execution composition must not synchronize the host")

    def done(self):
        raise AssertionError("Graph composition must not query device completion")


class Context(ExecutionContext):
    compute_stream = "compute"
    copy_stream = "copy"

    def __init__(self, phase=Phase.DECODE, capturing=False):
        self.phase, self.capturing = phase, capturing
        self.calls, self.retained = [], []

    def retain(self, resource):
        self.retained.append(resource)

    def record_compute_completion(self):
        name = "frontier" + str(len(self.calls))
        self.calls.append(("record", name))
        return self.ticket(name)

    def ticket(self, name):
        return TransferTicket(Fence(name, self.calls))


def device_plan():
    # Opaque fixed bindings; a CPU test must not pretend these are device kernels.
    return DeviceCopyPlan(*(object() for _ in range(6)), capacity=2)


def request(phase=Phase.DECODE, wave=None):
    return ExecutionRequest(
        3,
        object(),
        wave or ExpertWave(0, 0, 2, 0),
        phase,
        (
            SourcePolicy.HOST_ONLY
            if phase is Phase.PREFILL
            else SourcePolicy.PREFER_RESIDENT
        ),
    )


class HostOps(HostTransferOps):
    def __init__(self, context):
        self.context = context

    def materialize_into(self, plan, bindings, stream):
        self.context.calls.append(("host_copy", bindings))
        for r in plan.regions:
            r.destination[r.destination_offset : r.destination_offset + r.nbytes] = (
                r.source[r.source_offset : r.source_offset + r.nbytes]
            )
        return self.context.ticket("copied")


class DeviceOps(DeviceTransferOps):
    def __init__(self, context):
        self.context = context
        self.fail = False

    def materialize_into(self, plan, bindings, stream):
        self.context.calls.append(("device_copy", bindings))
        if self.fail:
            raise RuntimeError("partial submission")
        return self.context.ticket("copied")


class Residency(ResidencyOps):
    def __init__(self, context):
        self.context = context
        self.reservation = None
        self.update = None
        self.last_targets = None
        self.bad_plan = False

    def resolve_execution(self, req, context):
        context.calls.append(("resolve", req.wave.index))
        plan = HostCopyPlan(()) if req.phase is Phase.PREFILL else device_plan()
        if self.bad_plan:
            plan = HostCopyPlan(())
        self.reservation = ExecutionReservation(
            plan,
            req.wave.index,
            (Fence("plan_ready", context.calls), Fence("old_reader", context.calls)),
        )
        return self.reservation

    def publish_execution(self, reservation, weights_ready, context):
        weights_ready.completion.wait_on(context.compute_stream)
        context.calls.append(("publish", reservation.bindings))
        return context.ticket("published")

    def finish_execution(self, reservation, read_done, context):
        context.calls.append(("retire", reservation.bindings))

    def prepare_update(self, target_keys, context):
        self.last_targets = target_keys.clone()  # CPU reference only.
        context.calls.append(("prepare_update",))
        self.update = UpdatePlan(
            device_plan(), "hot", (Fence("victim_reader", context.calls),)
        )
        return self.update

    def commit_update(self, update_plan, weights_ready, context):
        weights_ready.completion.wait_on(context.compute_stream)
        context.calls.append(("commit",))
        return context.ticket("committed")


def access_for(context):
    residency, device = Residency(context), DeviceOps(context)
    backend = ComposedTransferBackend(host_ops=HostOps(context), device_ops=device)
    return ExpertWeightAccess(residency, backend), residency, device


class TestTransfersAndLeases(unittest.TestCase):
    def test_host_descriptors_copy_offsets_and_preserve_other_bytes(self):
        context = Context(Phase.PREFILL)
        source, destination = bytearray(b"abcdefgh"), bytearray(b"........")
        region = HostCopyRegion(source, destination, 2, 3, 3, 8, 8)
        plan = HostCopyPlan([region])
        backend = ComposedTransferBackend(host_ops=HostOps(context))
        ticket = backend.materialize_host_into(plan, destination, "copy")
        self.assertEqual(destination, bytearray(b"...cde.."))
        self.assertIn(plan, ticket.keepalive)
        self.assertFalse(backend.supports_device_plans)
        with self.assertRaises(NotImplementedError):
            backend.materialize_into(device_plan(), None, "copy")

    def test_invalid_ranges_and_plan_kind_are_rejected(self):
        for offsets in [(-1, 0, 1), (7, 0, 2), (0, 7, 2)]:
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                HostCopyRegion(object(), object(), *offsets, 8, 8)
        backend = ComposedTransferBackend(host_ops=HostOps(Context()))
        with self.assertRaises(TypeError):
            backend.materialize_into(HostCopyPlan(()), None, "copy")
        with self.assertRaises(NotImplementedError):
            TorchTransferBackend().materialize_host_into(HostCopyPlan(()), None, None)

    def test_execution_does_not_admit_hot_and_waits_for_publication(self):
        context = Context(capturing=True)
        access, residency, _ = access_for(context)
        lease = access.stage(request(), context)
        self.assertEqual(access.wait_ready(lease, "compute"), 0)
        copied = context.calls.index(("device_copy", 0))
        self.assertLess(context.calls.index(("wait", "old_reader", "copy")), copied)
        self.assertIn(("wait", "published", "compute"), context.calls)
        self.assertNotIn(("prepare_update",), context.calls)
        self.assertNotIn(("commit",), context.calls)
        self.assertTrue(any(x is residency.reservation for x in context.retained))
        reader = context.ticket("reader")
        access.release_after(lease, reader, context)
        self.assertIs(lease.read_done, reader)
        with self.assertRaises(RuntimeError):
            access.release_after(lease, reader, context)
        with self.assertRaises(RuntimeError):
            access.wait_ready(lease, "compute")

    def test_foreign_access_or_invocation_cannot_release(self):
        context = Context()
        access, _, _ = access_for(context)
        other, _, _ = access_for(context)
        lease = access.stage(request(), context)
        with self.assertRaises(ValueError):
            other.wait_ready(lease, "compute")
        with self.assertRaises(ValueError):
            access.release_after(lease, context.ticket("reader"), Context())

    def test_partial_copy_failure_retains_reservation_without_publication(self):
        context = Context()
        access, residency, device = access_for(context)
        device.fail = True
        with self.assertRaisesRegex(RuntimeError, "partial submission"):
            access.stage(request(), context)
        self.assertTrue(any(x is residency.reservation for x in context.retained))
        self.assertFalse(
            any(call[0] in ("publish", "retire") for call in context.calls)
        )

    def test_decode_rejects_host_plan_instead_of_falling_back(self):
        context = Context(capturing=True)
        access, residency, _ = access_for(context)
        residency.bad_plan = True
        with self.assertRaisesRegex(TypeError, "DeviceCopyPlan"):
            access.stage(request(), context)
        self.assertFalse(any("copy" in call[0] for call in context.calls))

    def test_publication_failure_keeps_completed_copy_resources_alive(self):
        context = Context()
        access, residency, _ = access_for(context)

        def fail(*args):
            raise RuntimeError("publication failed")

        residency.publish_execution = fail
        with self.assertRaisesRegex(RuntimeError, "publication failed"):
            access.stage(request(), context)
        self.assertIn(("device_copy", 0), context.calls)
        self.assertTrue(any(x is residency.reservation for x in context.retained))
        self.assertTrue(any(isinstance(x, TransferTicket) for x in context.retained))
        self.assertNotIn(("retire", 0), context.calls)

    def test_prefill_requires_host_only_and_cannot_capture(self):
        with self.assertRaises(ValueError):
            ExecutionRequest(3, None, ExpertWave(0, 0, 1, 0), Phase.PREFILL)
        context = Context(Phase.PREFILL, capturing=True)
        access, _, _ = access_for(context)
        with self.assertRaises(RuntimeError):
            access.stage(request(Phase.PREFILL), context)
        self.assertEqual(context.calls, [])

    def test_prefill_microbatches_join_without_copy_and_retire_all_readers(self):
        context = Context(Phase.PREFILL)
        access, _, _ = access_for(context)
        leases = [
            access.stage(request(Phase.PREFILL, ExpertWave(i, i, i + 1, 0)), context)
            for i in range(3)
        ]
        copies = sum(call[0] == "host_copy" for call in context.calls)
        group = access.combine_prefill(leases, "large_view", context)
        self.assertEqual(access.wait_ready(group, "compute"), "large_view")
        self.assertEqual(sum(c[0] == "host_copy" for c in context.calls), copies)
        with self.assertRaisesRegex(RuntimeError, "transferred"):
            access.wait_ready(leases[0], "compute")
        access.release_after(group, context.ticket("large_F_read_done"), context)
        self.assertEqual(
            [c for c in context.calls if c[0] == "retire"],
            [("retire", 0), ("retire", 1), ("retire", 2)],
        )

    def test_prefill_group_rejects_duplicate_or_foreign_invocation_leases(self):
        context = Context(Phase.PREFILL)
        access, _, _ = access_for(context)
        lease = access.stage(request(Phase.PREFILL), context)
        with self.assertRaises(ValueError):
            access.combine_prefill([lease, lease], None, context)
        with self.assertRaises(ValueError):
            access.combine_prefill([lease], None, Context(Phase.PREFILL))
        with self.assertRaises(RuntimeError):
            access.combine_prefill([lease], None, Context(Phase.DECODE))
        self.assertFalse(lease.transferred)


class TestDecodeTail(unittest.TestCase):
    def make_coordinator(self, context):
        access, residency, device = access_for(context)
        policy = TensorLFUPolicy()
        state = policy.allocate_state(4, "cpu")
        canonical = SimpleNamespace(
            step_access_count=torch.tensor([1, 0, 9, 0]),
            update_enabled=torch.tensor(True),
            key_to_slot=torch.tensor([0, 1, -1, -1]),
        )
        targets = torch.empty(2, dtype=torch.int64)
        coordinator = CacheUpdateCoordinator(policy, state, canonical, targets, access)
        return coordinator, residency, device

    def test_lfu_selects_a_miss_only_at_tail_and_publishes_after_copy(self):
        context = Context(capturing=True)
        coordinator, residency, _ = self.make_coordinator(context)
        ticket = coordinator.emit_step_end(context)
        self.assertEqual(residency.last_targets.tolist(), [2, 0])
        self.assertEqual(coordinator.policy_state.frequency.tolist(), [1, 0, 9, 0])
        self.assertLess(
            context.calls.index(("wait", "victim_reader", "copy")),
            context.calls.index(("device_copy", "hot")),
        )
        self.assertLess(
            context.calls.index(("device_copy", "hot")),
            context.calls.index(("commit",)),
        )
        context.calls.clear()
        ticket.completion.wait_on("next_forward")
        self.assertEqual(
            context.calls,
            [("wait", "copied", "next_forward"), ("wait", "committed", "next_forward")],
        )

    def test_prefill_tail_does_not_touch_policy_or_residency(self):
        context = Context(Phase.PREFILL)
        coordinator, residency, _ = self.make_coordinator(context)
        coordinator.emit_step_end(context)
        self.assertEqual(coordinator.policy_state.frequency.tolist(), [0] * 4)
        self.assertIsNone(residency.last_targets)
        self.assertEqual(context.calls, [])

    def test_failed_update_cannot_commit(self):
        context = Context()
        coordinator, residency, device = self.make_coordinator(context)
        device.fail = True
        with self.assertRaises(RuntimeError):
            coordinator.emit_step_end(context)
        self.assertNotIn(("commit",), context.calls)
        self.assertTrue(any(x is residency.update for x in context.retained))

    def test_decode_waves_do_not_copy_ahead_even_with_different_banks(self):
        context = Context(capturing=True)
        access, _, _ = access_for(context)

        def make_request(layer, dispatched, wave, ctx):
            return request(wave=wave)

        def run(layer, dispatched, bindings, ctx):
            ctx.calls.append(("F", bindings))
            return LocalExecution(bindings, ctx.ticket(f"read{bindings}"))

        local = DecodeLocalExpertProgram(access, make_request, run)
        for wave in (ExpertWave(0, 0, 2, 0), ExpertWave(1, 2, 4, 1)):
            self.assertEqual(local.run(None, None, wave, context), wave.index)
        first_read = context.calls.index(("wait", "read0", "compute"))
        second_copy = context.calls.index(("device_copy", 1))
        frontier = next(
            call[1] for call in context.calls[first_read + 1 :] if call[0] == "record"
        )
        wait_index = context.calls.index(("wait", frontier, "copy"))
        self.assertLess(first_read, wait_index)
        self.assertLess(wait_index, second_copy)


class Pipeline(PrefillWeightPipeline):
    """No worker: exercise only the eager session/program contract."""

    def __init__(self, access):
        super().__init__(access)
        self.started = None
        self.k = 3
        self.fail_start = False
        self.bad_schedule = False

    def start(self, layer_id, context):
        context.calls.append(("start", layer_id))
        self.started = context
        if self.fail_start:
            raise RuntimeError("start failed")

    def validate_session(self, layer, context):
        if self.started is not context:
            raise RuntimeError("No layer-start session")

    def seal(self, layer, dispatch_ready, context):
        self.dispatch_ready = dispatch_ready
        context.calls.append(("seal", self.k))
        if self.bad_schedule:
            return (ExpertWave(0, 1, 5, 0),)
        return (ExpertWave(0, 0, self.k, 0), ExpertWave(1, self.k, 5, 1))

    def acquire(self, layer, wave, context):
        return self.access.stage(request(Phase.PREFILL, wave), context)

    def finish(self, layer, context):
        context.calls.append(("finish_pipeline",))
        self.started = None
        return context.ticket("pipeline_idle")

    def abort(self, context):
        context.calls.append(("abort",))
        self.started = None
        return context.ticket("partial_work")


class Dispatcher(MoEDispatcher):
    def dispatch(self, hidden, routing, context):
        context.calls.append(("D",))
        return hidden

    def dispatch_completion(self, dispatched, context):
        self.ready = Fence("dispatch_ready", context.calls)
        return self.ready

    def combine(self, result, context):
        context.calls.append(("C",))
        return result

    def finish(self, result, context):
        return result


class Layout:
    def bind(self, dispatched, context):
        return {"source": dispatched, "out": [None] * len(dispatched)}

    def select_wave(self, layout, wave, context):
        return layout["source"][wave.local_start : wave.local_end]

    def write_wave(self, layout, wave, result, context):
        layout["out"][wave.local_start : wave.local_end] = result

    def combine_input(self, layout, context):
        return layout["out"]


class TestPrefillAndRuntimeHooks(unittest.TestCase):
    def make_program(self, context, fail_ffn=False):
        access, _, _ = access_for(context)
        pipeline = Pipeline(access)

        def run(layer, dispatched, bindings, ctx):
            ctx.calls.append(("F", bindings))
            if fail_ffn:
                raise RuntimeError("F failed after submission")
            return LocalExecution([x * 2 for x in dispatched], ctx.ticket("read"))

        local = PrefillLocalExpertProgram(pipeline, run)
        program = PrefillDispatchOnceProgram(
            Dispatcher(), local, Layout(), local_experts=5, pipeline=pipeline
        )
        return program, pipeline

    def test_per_invocation_k_preserves_single_dispatch_and_combine(self):
        context = Context(Phase.PREFILL)
        program, pipeline = self.make_program(context)
        for k in (1, 3, 4):
            context.calls.clear()
            pipeline.k = k
            program.layer_start(3, context)
            self.assertEqual(
                program.run(3, [1, 2, 3, 4, 5], None, context), [2, 4, 6, 8, 10]
            )
            self.assertEqual(context.calls.count(("D",)), 1)
            self.assertEqual(context.calls.count(("C",)), 1)
            self.assertIs(pipeline.dispatch_ready, program.dispatcher.ready)
            self.assertEqual(program.waves, ())
            self.assertIn(("finish_pipeline",), context.calls)

    def test_missing_start_rejected_before_collective(self):
        context = Context(Phase.PREFILL)
        program, _ = self.make_program(context)
        with self.assertRaisesRegex(RuntimeError, "layer-start"):
            program.run(3, [], None, context)
        self.assertNotIn(("D",), context.calls)

    def test_bad_partition_aborts_without_ffn_or_combine(self):
        context = Context(Phase.PREFILL)
        program, pipeline = self.make_program(context)
        pipeline.bad_schedule = True
        program.layer_start(3, context)
        with self.assertRaises(ValueError):
            program.run(3, [1] * 5, None, context)
        self.assertIn(("abort",), context.calls)
        self.assertFalse(any(call[0] in ("F", "C") for call in context.calls))

    def test_start_and_ffn_failures_preserve_partial_work_ticket(self):
        for fail_start in (False, True):
            context = Context(Phase.PREFILL)
            program, pipeline = self.make_program(context, fail_ffn=not fail_start)
            pipeline.fail_start = fail_start
            with self.assertRaises(RuntimeError):
                program.layer_start(3, context)
                program.run(3, [1] * 5, None, context)
            self.assertIn(("abort",), context.calls)
            self.assertNotIn(("C",), context.calls)
            self.assertTrue(
                any(
                    x.completion.name == "partial_work"
                    for x in context.retained
                    if isinstance(x, TransferTicket) and isinstance(x.completion, Fence)
                )
            )

    def test_prefill_program_rejects_decode_and_capture_before_start(self):
        for phase, capture in [(Phase.DECODE, False), (Phase.PREFILL, True)]:
            context = Context(phase, capture)
            program, _ = self.make_program(context)
            with self.assertRaises(RuntimeError):
                program.layer_start(3, context)
            with self.assertRaises(RuntimeError):
                program.run(3, [], None, context)
            self.assertEqual(context.calls, [])

    def test_runtime_layer_start_and_unbound_decode_tail(self):
        prefill = Context(Phase.PREFILL)
        program, _ = self.make_program(prefill)
        runtime = SimpleNamespace(prefill_program=program, cache_updates=None)
        BoundMoERuntime.emit_layer_start(runtime, 3, prefill)
        self.assertIn(("start", 3), prefill.calls)
        self.assertIsNone(BoundMoERuntime.emit_step_end(runtime, prefill))
        decode = Context()
        BoundMoERuntime.emit_layer_start(runtime, 3, decode)
        self.assertEqual(decode.calls, [])
        with self.assertRaises(NotImplementedError):
            BoundMoERuntime.emit_step_end(runtime, decode)


if __name__ == "__main__":
    unittest.main()
