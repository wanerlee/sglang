"""CPU contracts for logging: immutable snapshots and no added device barriers."""

import logging
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from test_npu_backend import Assembly

from sglang.srt.offloading.moe.cache_metadata import CacheRouteStats
from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.diagnostics import BatchDiagnostics
from sglang.srt.offloading.moe.execution import ExecutionRequest
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Completion:
    def __init__(self, ready=False):
        self.ready = threading.Event()
        self.queried = threading.Event()
        if ready:
            self.ready.set()

    def done(self):
        self.queried.set()
        return self.ready.is_set()

    def synchronize(self):
        raise AssertionError("Diagnostics must not synchronize")

    def wait_on(self, stream):
        raise AssertionError("Diagnostics must not enqueue a wait")


class LogCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class TestBatchDiagnostics(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("sglang.srt.offloading.moe.diagnostics")
        self.old_level = self.logger.level
        self.logger.setLevel(logging.INFO)
        self.logs = LogCollector()
        self.logger.addHandler(self.logs)
        self.addCleanup(self.logger.setLevel, self.old_level)
        self.addCleanup(self.logger.removeHandler, self.logs)
        self.stats = CacheRouteStats.allocate("cpu")
        self.error = torch.zeros((), dtype=torch.int32)
        self.batch = SimpleNamespace(
            forward_mode=SimpleNamespace(name="DECODE"),
            batch_size=2,
            input_ids=torch.tensor([1, 2]),
        )

    def sink(self, *, interval=1, capacity=4):
        sink = BatchDiagnostics(
            self.stats,
            self.error,
            rank=1,
            device="cpu",
            interval=interval,
            capacity=capacity,
            host_buffer=torch.empty((capacity, 4), dtype=torch.int64),
        )
        self.addCleanup(sink.close)
        return sink

    def wait_logs(self, count):
        deadline = time.monotonic() + 2
        while len(self.logs.messages) < count and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertGreaterEqual(len(self.logs.messages), count, self.logs.messages)

    def set_counts(self, accesses, hits):
        self.stats.accesses.fill_(accesses)
        self.stats.hits.fill_(hits)
        self.stats.misses.fill_(accesses - hits)

    def test_reads_only_after_existing_completion_without_waiting(self):
        sink = self.sink()
        self.set_counts(10, 7)
        done = Completion()
        sink.publish(sink.snapshot(Phase.DECODE, 1, self.batch, bucket=4), done)
        self.assertTrue(done.queried.wait(2))
        self.assertEqual(self.logs.messages, [])
        done.ready.set()
        self.wait_logs(1)
        self.assertIn("seq=1", self.logs.messages[0])
        self.assertIn("bucket=4", self.logs.messages[0])
        self.assertIn(
            "assignments=10 hits=7 misses=3 hit_rate=70.00%", self.logs.messages[0]
        )

    def test_next_batch_cannot_overwrite_pending_snapshot(self):
        sink = self.sink()
        self.set_counts(10, 7)
        first = sink.snapshot(Phase.DECODE, 1, self.batch)
        self.set_counts(5, 1)
        second = sink.snapshot(Phase.DECODE, 2, self.batch)
        before = [
            int(x) for x in (self.stats.accesses, self.stats.hits, self.stats.misses)
        ]
        done = Completion()
        sink.publish(first, done)
        sink.publish(second, Completion(ready=True))
        done.ready.set()
        self.wait_logs(2)
        self.assertIn("hit_rate=70.00%", self.logs.messages[0])
        self.assertIn("hit_rate=20.00%", self.logs.messages[1])
        self.assertEqual(
            before,
            [int(x) for x in (self.stats.accesses, self.stats.hits, self.stats.misses)],
        )

    def test_prefill_empty_demand_and_fault_do_not_report_misleading_rates(self):
        sink = self.sink()
        self.set_counts(10, 7)
        prefill = sink.snapshot(Phase.PREFILL, 1, self.batch)
        self.set_counts(0, 0)
        empty = sink.snapshot(Phase.DECODE, 2, self.batch)
        self.error.fill_(128)
        self.set_counts(10, 7)
        fault = sink.snapshot(Phase.DECODE, 3, self.batch)
        for record in (prefill, empty, fault):
            sink.publish(record, Completion(ready=True))
        self.wait_logs(3)
        self.assertIn("assignments=N/A", self.logs.messages[0])
        self.assertIn("assignments=0", self.logs.messages[1])
        self.assertIn("error=128", self.logs.messages[2])
        self.assertTrue(all("hit_rate=N/A" in line for line in self.logs.messages))

    def test_full_buffer_drops_logs_without_blocking_or_reusing_busy_slot(self):
        sink = self.sink(capacity=1)
        first = sink.snapshot(Phase.DECODE, 1, self.batch)
        done = Completion()
        sink.publish(first, done)
        self.assertIsNone(sink.snapshot(Phase.DECODE, 2, self.batch))
        self.assertEqual(sink._dropped, 1)
        self.wait_logs(1)
        self.assertIn("buffer full", self.logs.messages[0])
        done.ready.set()
        self.wait_logs(2)

    def test_interval_zero_and_log_level_skip_all_snapshot_copies(self):
        disabled = self.sink(interval=0)
        self.assertEqual(disabled.allocations(), ())
        self.assertIsNone(disabled.snapshot(Phase.DECODE, 1, self.batch))
        sink = self.sink(interval=2)
        with mock.patch.object(
            torch.Tensor, "copy_", side_effect=AssertionError("copy")
        ):
            self.assertIsNone(sink.snapshot(Phase.DECODE, 1, self.batch))
            self.logger.setLevel(logging.ERROR)
            self.assertIsNone(sink.snapshot(Phase.DECODE, 2, self.batch))

    def test_metadata_budget_charges_fixed_host_storage_once(self):
        sink = self.sink(capacity=3)
        allocations = sink.allocations()
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].nbytes, 3 * 4 * 8)

    def test_snapshot_uses_only_nonblocking_copies_with_matching_dtypes(self):
        sink = self.sink()
        copies = []
        original = torch.Tensor.copy_

        def copy(destination, source, *, non_blocking=False):
            copies.append((destination.dtype, source.dtype, non_blocking))
            return original(destination, source, non_blocking=non_blocking)

        with mock.patch.object(torch.Tensor, "copy_", new=copy):
            record = sink.snapshot(Phase.DECODE, 1, self.batch)
        self.assertEqual(len(copies), 4)
        self.assertTrue(
            all(
                destination == source and nonblocking
                for destination, source, nonblocking in copies
            )
        )
        sink.publish(record, Completion(ready=True))
        self.wait_logs(1)

    def test_query_failure_disables_logging_without_reusing_busy_slots(self):
        sink = self.sink(capacity=1)
        completion = Completion()
        completion.done = mock.Mock(side_effect=RuntimeError("query failed"))
        sink.publish(sink.snapshot(Phase.DECODE, 1, self.batch), completion)
        self.wait_logs(1)
        self.assertTrue(sink._disabled)
        self.assertIsNone(sink.snapshot(Phase.DECODE, 2, self.batch))
        self.assertIn("batch logging disabled", self.logs.messages[0])

    def test_partial_snapshot_failure_is_retained_without_poisoning_inference(self):
        sink = self.sink(capacity=1)
        with mock.patch.object(
            torch.Tensor, "copy_", side_effect=RuntimeError("copy failed")
        ):
            record = sink.snapshot(Phase.DECODE, 1, self.batch)
        self.assertIsNotNone(record)
        self.assertIsNone(sink.snapshot(Phase.DECODE, 2, self.batch))
        done = Completion()
        sink.publish(record, done)
        self.assertTrue(done.queried.wait(2))
        self.assertEqual(self.logs.messages, [])
        done.ready.set()
        self.wait_logs(1)
        self.assertIn("copy_error=copy failed", self.logs.messages[0])

    def test_runtime_logging_preserves_transfer_events_cache_and_policy(self):
        results = []
        for enabled in (False, True):
            assembly = Assembly(ratio=0.5, layer_count=1, local_experts=4)
            # Keep route counters in both capture programs; only the diagnostics
            # sink differs. Disabled batch logging omits these counters entirely.
            assembly.backend.settings.batch_log_interval = 1
            runtime = assembly.finalize()
            if enabled:
                runtime.diagnostics = BatchDiagnostics(
                    runtime.cache_stats,
                    runtime.ops.error_status,
                    rank=1,
                    device="cpu",
                    capacity=2,
                    host_buffer=torch.empty((2, 4), dtype=torch.int64),
                )
            owner = MoEOffloadOwner(assembly.config, assembly.catalog, runtime)
            self.batch.forward_mode.is_decode_or_idle = lambda: True
            with runtime.forward_scope(owner, self.batch):
                context = runtime.context
                # Stand in for the exact same captured nodes in both runs.
                runtime.ops.begin_step(context)
                spec = assembly.catalog.offloaded_layers[0]
                for wave in runtime.layer_programs[spec.layer_id].decode_waves:
                    access = runtime.cache_updates.access
                    lease = access.stage(
                        ExecutionRequest(
                            spec.layer_id,
                            torch.tensor([2, 3, 0, 0]),
                            wave,
                            Phase.DECODE,
                        ),
                        context,
                    )
                    access.wait_ready(lease, context.compute_stream, context=context)
                    access.release_after(
                        lease, context.record_compute_completion(), context
                    )
                runtime.cache_updates.emit_step_end(context)
            keys = runtime.residency.slot_to_key.clone()
            frequency = runtime.cache_updates.policy_state.frequency.clone()
            owner.close()
            results.append((assembly.events, keys, frequency))
        self.assertEqual(results[0][0], results[1][0])
        torch.testing.assert_close(results[0][1], results[1][1])
        torch.testing.assert_close(results[0][2], results[1][2])
        self.wait_logs(1)
        self.assertIn("assignments=5 hits=0 misses=5", self.logs.messages[0])


if __name__ == "__main__":
    unittest.main()
