"""Portable behavioral checks for asynchronous benchmark ordering and reads.

The fake stream intentionally completes work only at synchronize(), so a
premature release or read cannot pass merely because Python calls returned.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from bench import Runtime, byte_pattern, verification_ranges


class FakeStream:
    def __init__(self):
        self.pending = []
        self.clock_ms = 0
        self.sync_checks = []
        self.failed = False

    def enqueue(self, action):
        self.pending.append(action)

    def synchronize(self):
        for check in self.sync_checks:
            check()
        if self.failed:
            raise RuntimeError("simulated stream drain failure")
        while self.pending:
            self.pending.pop(0)()


class FakeEvent:
    def __init__(self, **kwargs):
        self.timestamp = None

    def record(self, stream):
        def complete():
            self.timestamp = stream.clock_ms
        stream.enqueue(complete)

    def elapsed_time(self, other):
        if self.timestamp is None or other.timestamp is None:
            raise AssertionError("Device event inspected before completion")
        return other.timestamp - self.timestamp


class RetainedTickets(list):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.clear_count = 0

    def clear(self):
        if self.stream.pending:
            raise AssertionError("Queued copy resources released before completion")
        self.clear_count += 1
        super().clear()


class FakeOps:
    def __init__(self, retain):
        self.retain = retain
        self.issued = []
        self.completed = []
        self.clear_cache = Mock()

    def materialize_into(self, plan, bank, stream):
        self.issued.append((plan.layer, plan.projection, bank.name))
        self.retain.append(object())

        def complete():
            if not self.retain:
                raise AssertionError("Copy ticket did not survive host submission")
            stream.clock_ms += 7
            bank.last_layer = plan.layer
            self.completed.append((plan.layer, plan.projection, bank.name))

        stream.enqueue(complete)


def measured_runtime(passes=2, projection_timing=False):
    options = {"passes": passes, "projection_timing": projection_timing, "verify": "sample"}
    runtime = Runtime(3, {"offloaded_layer_ids": [3, 4, 5]}, options, Mock())
    stream = FakeStream()
    runtime.stream = stream
    runtime.keepalive = RetainedTickets(stream)
    runtime.torch = SimpleNamespace(npu=SimpleNamespace(Event=FakeEvent, synchronize=lambda device: stream.synchronize()))
    runtime.ops = FakeOps(runtime.keepalive)
    runtime.sources = [object(), object(), object()]
    runtime.banks = [SimpleNamespace(name="bank0"), SimpleNamespace(name="bank1")]
    runtime.bank_owner = object()
    runtime.pool = SimpleNamespace(close=Mock())
    runtime.plans = [
        tuple(SimpleNamespace(layer=layer, projection=projection,
                              regions=[SimpleNamespace(nbytes=nbytes)])
              for projection, nbytes in (("w13", 128), ("w2", 64)))
        for layer in (3, 4, 5)
    ]
    runtime.sample_bytes = 192 * 3 * passes
    return runtime


class AsyncTests(unittest.TestCase):
    def test_stage_traversals_preserve_projection_order_and_reset_odd_bank_cycle(self):
        runtime = measured_runtime()
        runtime.stream.sync_checks.append(
            lambda: self.assertEqual(len(runtime.keepalive), 12)
        )
        row, details = runtime.measure()
        one_stage = [(layer, projection, f"bank{index % 2}")
                     for index, layer in enumerate((3, 4, 5))
                     for projection in ("w13", "w2")]
        self.assertEqual(runtime.ops.completed, one_stage + one_stage)
        self.assertEqual(runtime.ops.issued, runtime.ops.completed)
        self.assertEqual(row["bytes"], 1152)
        self.assertEqual(row["event_ms"], 84)
        self.assertLessEqual(row["host_start_ns"], row["host_submit_end_ns"])
        self.assertLessEqual(row["host_submit_end_ns"], row["host_end_ns"])
        self.assertEqual(details, [])
        self.assertFalse(runtime.keepalive)
        self.assertEqual(runtime.keepalive.clear_count, 1)

    def test_projection_timing_accounts_for_each_transfer_after_completion(self):
        runtime = measured_runtime(passes=1, projection_timing=True)
        row, details = runtime.measure()
        self.assertEqual(row["event_ms"], 42)
        self.assertEqual(len(details), 6)
        self.assertEqual(sum(detail["bytes"] for detail in details), row["bytes"])
        self.assertEqual([detail["projection"] for detail in details], ["w13", "w2"] * 3)
        self.assertTrue(all(detail["event_ms"] == 7 for detail in details))
        self.assertEqual({detail["layer"] for detail in details}, {3, 4, 5})

    def test_untimed_source_verification_waits_before_read_and_reuse(self):
        runtime = measured_runtime(passes=1)
        seen = []

        def verify(index):
            self.assertFalse(runtime.stream.pending)
            self.assertFalse(runtime.keepalive)
            self.assertEqual(runtime.banks[index % 2].last_layer, (3, 4, 5)[index])
            seen.append(index)
            return 192

        runtime.verify_bank = verify
        self.assertEqual(runtime.verify_sources(), 576)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(runtime.keepalive.clear_count, 3)

    def test_failed_drain_preserves_tickets_and_pool_for_safe_failure(self):
        runtime = measured_runtime(passes=1)
        runtime.stream.failed = True
        with self.assertRaisesRegex(RuntimeError, "drain failure"):
            runtime.measure()
        self.assertEqual(len(runtime.keepalive), 6)
        pool, banks, plans = runtime.pool, runtime.banks, runtime.plans
        with self.assertRaisesRegex(RuntimeError, "drain failure"):
            runtime.close()
        self.assertIs(runtime.pool, pool)
        self.assertIs(runtime.banks, banks)
        self.assertIs(runtime.plans, plans)
        self.assertEqual(runtime.keepalive.clear_count, 0)
        pool.close.assert_not_called()
        runtime.ops.clear_cache.assert_not_called()


class FakeSlice:
    def __init__(self, data):
        self.data = data

    def eq(self, expected):
        return SimpleNamespace(all=lambda: all(value == expected for value in self.data))


class Scratch:
    def __init__(self, size):
        self.data = bytearray(size)

    def data_ptr(self):
        return 0x7000000000

    def numel(self):
        return len(self.data)

    def __getitem__(self, key):
        return FakeSlice(self.data[key])


class D2HMemory:
    def __init__(self, regions):
        self.regions = regions
        self.scratch = None
        self.calls = []
        self.corrupt = False

    def empty(self, size, **kwargs):
        if kwargs != {"dtype": "uint8", "device": "cpu"}:
            raise AssertionError("Verification scratch must remain a byte-oriented CPU buffer")
        self.scratch = Scratch(size)
        return self.scratch

    def memcpy_sync(self, dst, capacity, src, size, direction, device):
        if direction != "device_to_host" or device != 3:
            raise AssertionError("Verification must read raw bytes from the selected device")
        if dst != self.scratch.data_ptr() or capacity != self.scratch.numel() or not 0 < size <= capacity:
            raise AssertionError("D2H exceeds scratch allocation")
        matching = [(base, extent, value) for base, extent, value in self.regions
                    if base <= src and src + size <= base + extent]
        if len(matching) != 1:
            raise AssertionError("D2H exceeds one component/expert physical extent")
        _, _, value = matching[0]
        self.scratch.data[:size] = bytes([value ^ int(self.corrupt)]) * size
        self.calls.append((src, size))


def verifying_runtime():
    runtime = measured_runtime(passes=1)
    layouts = {"w13_weight": SimpleNamespace(nbytes=191),
               "w13_weight_scale": SimpleNamespace(nbytes=257),
               "w2_weight": SimpleNamespace(nbytes=32)}
    regions = []
    banks = []
    for bank_index in range(2):
        component_regions = {}
        for ci, (name, layout) in enumerate(layouts.items()):
            entries = []
            for expert in range(2):
                base = 0x100000 + bank_index * 0x10000 + ci * 0x1000 + expert * 0x400
                # With three layer plans, final bank0 holds source layer2 and
                # final bank1 holds source layer1.
                source = 2 if bank_index == 0 else 1
                entries.append(SimpleNamespace(address=base, nbytes=layout.nbytes))
                regions.append((base, layout.nbytes, byte_pattern(source, expert, ci)))
            component_regions[name] = entries
        banks.append(SimpleNamespace(component_regions=component_regions))
    memory = D2HMemory(regions)
    runtime.native = memory
    runtime.torch = SimpleNamespace(empty=memory.empty, uint8="uint8")
    runtime.banks = banks
    runtime.fmt = SimpleNamespace(layouts=layouts)
    return runtime, memory


class VerificationTests(unittest.TestCase):
    def test_sample_reads_each_expert_component_within_actual_extent(self):
        runtime, memory = verifying_runtime()
        checked = runtime.verify_bank(2)
        # 191 and 257 byte components: three 64B samples; 32B component: one.
        self.assertEqual(checked, (192 + 192 + 32) * 2)
        self.assertEqual(len(memory.calls), 14)
        self.assertEqual(len({src for src, size in memory.calls}), 14)
        self.assertEqual(memory.scratch.numel(), 64)

    def test_bad_payload_is_not_reported_as_a_successful_sample(self):
        runtime, memory = verifying_runtime()
        memory.corrupt = True
        with self.assertRaisesRegex(RuntimeError, "Byte verification failed"):
            runtime.verify_bank(2)

    def test_final_validation_checks_both_last_resident_layers(self):
        runtime, memory = verifying_runtime()
        self.assertEqual(runtime.verify_final(), (192 + 192 + 32) * 2 * 2)
        self.assertEqual(len(memory.calls), 28)

    def test_full_validation_ranges_cover_tail_without_overlap_or_overread(self):
        nbytes = 17 * 1024**2 + 13
        ranges = verification_ranges(nbytes, full=True)
        self.assertEqual(sum(size for offset, size in ranges), nbytes)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][0] + ranges[-1][1], nbytes)
        self.assertTrue(all(a + size == b for (a, size), (b, _) in zip(ranges, ranges[1:])))
        self.assertTrue(all(0 < size <= 8 * 1024**2 for offset, size in ranges))

    def test_full_mode_uses_all_bytes_of_small_physical_regions(self):
        runtime, memory = verifying_runtime()
        runtime.options["verify"] = "full"
        self.assertEqual(runtime.verify_bank(2), (191 + 257 + 32) * 2)
        self.assertEqual(len(memory.calls), 6)


if __name__ == "__main__":
    unittest.main()
