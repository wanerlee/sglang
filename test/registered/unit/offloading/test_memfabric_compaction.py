"""CPU contracts for the real MemFabric descriptor lowering.

Only device allocation, stream events and the native byte-copy call are replaced.
These tests do not certify NPU kernel counts, timing or graph capture. The harness
accepts another transfer module so the same workload can exercise an old revision.
"""

import ctypes
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.offloading.moe.npu import transfer
from sglang.srt.offloading.moe.storage import HostMemoryCapability
from sglang.srt.offloading.transfer import (
    CopySourceKind,
    DeviceCopyPlan,
    ImmediateCompletion,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@dataclass(frozen=True)
class SimulatedNpuDevice:
    index: int = 0
    type: str = "npu"

    def __str__(self):
        return f"npu:{self.index}"


class ScatterCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.scatter_add_.default:
            self.calls += 1
        return func(*args, **(kwargs or {}))


class TransferHarness:
    """Real tables/plans/lowering plus a checked src[i] -> dst[j] byte copier."""

    component_sizes = dict(
        zip(
            (
                "w13_weight",
                "w13_weight_scale",
                "w13_weight_offset",
                "w2_weight",
                "w2_weight_scale",
                "w2_weight_offset",
            ),
            (8, 3, 5, 12, 7, 9),
        )
    )

    def __init__(self, capacity, *, module=transfer, max_batch_entries=256):
        self.capacity = capacity
        self.module = module
        self.max_batch_entries = max_batch_entries
        self.device = SimulatedNpuDevice()
        self.stack = ExitStack()
        self.retained, self.calls, self.events = [], [], []
        self.fail_on_call = None
        self.fail_record = False

    def __enter__(self):
        original_device, original_tensor = torch.device, torch.tensor

        def device(value):
            return (
                value
                if isinstance(value, SimulatedNpuDevice)
                else original_device(value)
            )

        def tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)

        self.stack.enter_context(mock.patch.object(torch, "device", side_effect=device))
        self.stack.enter_context(mock.patch.object(torch, "tensor", side_effect=tensor))
        self.stack.enter_context(
            mock.patch.object(torch.Tensor, "device", property(lambda _: self.device))
        )
        try:
            self.source_count = max(3, self.capacity + 1)
            self.destination_count = max(2, self.capacity + 2)
            self.host, self.hot, self.destination = {}, {}, {}
            host_regions, hot_regions, destination_regions = {}, {}, {}
            self.source_rows, self.destination_rows = {}, {}
            for component, (name, nbytes) in enumerate(self.component_sizes.items()):
                self.host[name] = self._rows(
                    self.source_count, nbytes, component * 31 + 1
                )
                self.hot[name] = self._rows(
                    self.source_count, nbytes, component * 31 + 117
                )
                self.destination[name] = torch.full(
                    (self.destination_count, nbytes), 231, dtype=torch.uint8
                )
                host_regions[name] = tuple(
                    HostMemoryCapability(
                        row.data_ptr(), nbytes, True, row.data_ptr(), self.host[name]
                    )
                    for row in self.host[name]
                )
                hot_regions[name] = self._regions(self.hot[name], nbytes)
                destination_regions[name] = self._regions(
                    self.destination[name], nbytes
                )
                for row in (*self.host[name], *self.hot[name]):
                    self.source_rows[row.data_ptr()] = row
                for row in self.destination[name]:
                    self.destination_rows[row.data_ptr()] = row
            self.catalog = self.module.MemFabricAddressCatalog(
                host_regions, hot_regions, self.component_sizes, device=self.device
            )
            self.destination_regions = destination_regions
            self.generations = torch.arange(self.source_count, dtype=torch.int64) + 19
            self.error = torch.tensor(0, dtype=torch.int32)
            self.binding = self.new_binding()
            self.stream = SimpleNamespace(device=self.device, stream_id=1)
            self.ops = self.module.MemFabricDeviceTransferOps(
                retain=self.retained.append,
                group_pack_copy=self.native_copy,
                record_completion=self.record,
                stream_context=lambda stream: nullcontext(),
            )
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    @staticmethod
    def _rows(count, nbytes, seed):
        return torch.tensor(
            [
                [(seed + row * 13 + col) % 251 for col in range(nbytes)]
                for row in range(count)
            ],
            dtype=torch.uint8,
        )

    def _regions(self, rows, nbytes):
        return tuple(
            self.module.DeviceMemoryRegion(row.data_ptr(), nbytes, self.device, rows)
            for row in rows
        )

    def new_binding(self):
        return self.module.MemFabricTransferBindings(
            self.catalog,
            self.destination_regions,
            capacity=self.capacity,
            hot_generations=self.generations,
            error_status=self.error,
            max_batch_entries=self.max_batch_entries,
        )

    def plan(self, valid=None):
        indices = torch.tensor(
            [(i * 3 + 1) % self.source_count for i in range(self.capacity)],
            dtype=torch.int64,
        )
        mask = torch.tensor(
            [True] * self.capacity if valid is None else valid, dtype=torch.bool
        )
        return DeviceCopyPlan(
            source_kind=torch.tensor(
                [
                    int(CopySourceKind.HOST if i % 2 == 0 else CopySourceKind.HOT_SLOT)
                    for i in range(self.capacity)
                ],
                dtype=torch.int64,
            ),
            source_index=indices,
            # Reverse order exercises non-monotonic destinations and real slot 0.
            destination_slot=torch.tensor(
                list(reversed(range(self.capacity))), dtype=torch.int64
            ),
            valid=mask,
            expected_generation=self.generations[indices].clone(),
            count=mask.sum(dtype=torch.int32),
            capacity=self.capacity,
        )

    def expected(self, plan, *, suppressed=False):
        expected = {name: rows.clone() for name, rows in self.destination.items()}
        if suppressed:
            return expected
        # Independent oracle: apply the logical expert mapping, without packing.
        for lane in range(self.capacity):
            if not bool(plan.valid[lane]):
                continue
            source = (
                self.host
                if int(plan.source_kind[lane]) == int(CopySourceKind.HOST)
                else self.hot
            )
            for name in expected:
                expected[name][int(plan.destination_slot[lane])].copy_(
                    source[name][int(plan.source_index[lane])]
                )
        return expected

    def native_copy(self, src, dst, lengths, scan, groups, packed, device):
        if device != self.device:
            raise AssertionError("Wrong copy device")
        if not bool(torch.all(packed == 0)):
            raise AssertionError("Native packed scratch was not cleared")
        self.calls.append((src.tolist(), dst.tolist(), groups.tolist()))
        self.events.append("copy")
        if len(self.calls) == self.fail_on_call:
            return 17
        selected = [i for i in range(int(scan)) if int(groups[i]) != 0]
        for j, i in enumerate(selected):
            source = self.source_rows.get(int(src[i]))
            target = self.destination_rows.get(int(dst[j]))
            if source is None or target is None:
                raise AssertionError("Invalid source or packed destination pointer")
            nbytes = int(lengths[i])
            if nbytes != source.numel() or nbytes != target.numel():
                raise AssertionError("Copy length does not match the component")
            ctypes.memmove(target.data_ptr(), source.data_ptr(), nbytes)
        # Model native-owned mutation; these arrays must not alias shared slots.
        packed.fill_(91)
        groups.fill_(73)
        return 0

    def record(self, stream):
        self.events.append("record")
        if self.fail_record:
            raise RuntimeError("Injected completion failure")
        return ImmediateCompletion()

    def run(self, plan):
        with ScatterCounter() as counter:
            ticket = self.ops.materialize_into(plan, self.binding, self.stream)
        return ticket, counter.calls


def profile_compaction(
    module=transfer, *, capacity=32, valid=None, max_batch_entries=256
):
    """Comparable CPU operator/descriptor counts, never an NPU performance claim."""
    with TransferHarness(
        capacity, module=module, max_batch_entries=max_batch_entries
    ) as h:
        plan = h.plan(valid)
        expected = h.expected(plan)
        _, scatters = h.run(plan)
        if any(
            not torch.equal(h.destination[name], rows)
            for name, rows in expected.items()
        ):
            raise AssertionError("Copied component bytes differ from the logical plan")
        return {
            "scatter_add_calls": scatters,
            "copy_calls": len(h.calls),
            "capacity": capacity,
        }


class TestMemFabricCompaction(unittest.TestCase):
    def assert_destinations(self, harness, expected):
        for name, rows in expected.items():
            self.assertTrue(torch.equal(harness.destination[name], rows), name)

    def test_real_lowering_counts_and_bytes_at_batch_boundaries(self):
        for capacity in (0, 1, 255, 256, 257):
            with self.subTest(capacity=capacity):
                result = profile_compaction(capacity=capacity)
                batches = (capacity + 255) // 256
                self.assertEqual(result["scatter_add_calls"], 1 + batches)
                self.assertEqual(result["copy_calls"], 6 * batches)

    def test_sparse_full_empty_replays_preserve_unselected_destinations(self):
        with TransferHarness(7) as h:
            plan = h.plan()
            storage = [a.storage_key for a in h.binding.allocations()]
            for mask in (
                [False, True, False, True, False, False, True],
                [True] * 7,
                [False] * 7,
                [True, False, False, False, False, False, True],
            ):
                with self.subTest(mask=mask):
                    plan.valid.copy_(torch.tensor(mask, dtype=torch.bool))
                    plan.count.copy_(plan.valid.sum(dtype=torch.int32))
                    expected = h.expected(plan)
                    start = len(h.calls)
                    _, scatters = h.run(plan)
                    self.assert_destinations(h, expected)
                    self.assertEqual(scatters, 2)
                    self.assertEqual(len(h.calls) - start, 6)
                    self.assertEqual(int(h.error), 0)
                    # Gather must overwrite the full packed prefix and zero tail.
                    for _, dst, _ in h.calls[start:]:
                        self.assertEqual(dst[sum(mask) :], [0] * (7 - sum(mask)))
                    self.assertEqual(
                        storage, [a.storage_key for a in h.binding.allocations()]
                    )

    def test_each_batch_has_independent_compaction_rank(self):
        result = profile_compaction(
            capacity=8,
            valid=[False, True, False, True, False, True, False, True],
            max_batch_entries=3,
        )
        self.assertEqual(result["scatter_add_calls"], 4)
        self.assertEqual(result["copy_calls"], 18)

    def test_inactive_invalid_entries_never_become_pointers(self):
        with TransferHarness(5) as h:
            plan = h.plan([True, False, True, False, True])
            plan.source_index[1] = -99
            plan.destination_slot[1] = -99
            plan.source_kind[3] = 99
            plan.destination_slot[3] = 999
            expected = h.expected(plan)
            h.run(plan)
            self.assert_destinations(h, expected)
            self.assertEqual(int(h.error), 0)

    def test_dynamic_validation_suppresses_the_whole_plan(self):
        for fault in (
            "generation",
            "count",
            "source",
            "destination",
            "kind",
            "duplicate",
            "cross_batch_duplicate",
            "latched",
        ):
            capacity = 257 if fault == "cross_batch_duplicate" else 5
            with self.subTest(fault=fault), TransferHarness(capacity) as h:
                plan = h.plan()
                if fault == "generation":
                    plan.expected_generation[1] += 1
                elif fault == "count":
                    plan.count.add_(1)
                elif fault == "source":
                    plan.source_index[0] = h.source_count
                elif fault == "destination":
                    plan.destination_slot[0] = h.destination_count
                elif fault == "kind":
                    plan.source_kind[0] = 99
                elif fault in ("duplicate", "cross_batch_duplicate"):
                    plan.destination_slot[-1] = plan.destination_slot[0]
                else:
                    h.error.fill_(8)
                expected = h.expected(plan, suppressed=True)
                h.run(plan)
                self.assert_destinations(h, expected)
                self.assertNotEqual(int(h.error), 0)
                self.assertFalse(bool(h.binding.mask.any()))
                for src, dst, groups in h.calls:
                    self.assertTrue(all(value == 0 for value in (*src, *dst, *groups)))

    def test_partial_submission_failure_retains_completion_and_resources(self):
        with TransferHarness(4) as h:
            plan = h.plan()
            expected = h.expected(plan)
            original = h.expected(plan, suppressed=True)
            h.fail_on_call = 3
            with self.assertRaises(h.module.TransferSubmissionError) as raised:
                h.run(plan)
            ticket = raised.exception.completion_ticket
            self.assertIsNotNone(ticket)
            self.assertEqual(h.events, ["copy", "copy", "copy", "record"])
            self.assertIs(ticket.keepalive[0], plan)
            self.assertIs(ticket.keepalive[1], h.binding)
            self.assertIs(ticket.keepalive[2], h.ops)
            self.assertTrue(any(item is ticket for item in h.retained))
            for index, name in enumerate(h.component_sizes):
                rows = expected[name] if index < 2 else original[name]
                self.assertTrue(torch.equal(h.destination[name], rows), name)

    def test_failed_completion_still_retains_pre_submit_resources(self):
        with TransferHarness(4) as h:
            plan = h.plan()
            h.fail_on_call, h.fail_record = 2, True
            with self.assertRaises(h.module.TransferSubmissionError) as raised:
                h.run(plan)
            self.assertIsNone(raised.exception.completion_ticket)
            self.assertTrue(
                any(
                    isinstance(item, tuple) and item[0] is plan and item[1] is h.binding
                    for item in h.retained
                )
            )

    def test_shared_slot_scratch_is_owned_accounted_and_separate_from_native(self):
        with TransferHarness(257) as h:
            first, second = h.binding, h.new_binding()
            self.assertEqual(
                [t.numel() for t in first.packed_destination_slots], [256, 1]
            )
            ledger = {a.storage_key: a for a in first.allocations()}
            slot_keys = set()
            for slots in first.packed_destination_slots:
                self.assertEqual(slots.dtype, torch.int64)
                key = (str(slots.device), slots.untyped_storage().data_ptr())
                slot_keys.add(key)
                self.assertIn(key, ledger)
                self.assertEqual(ledger[key].nbytes, slots.numel() * 8)
            self.assertEqual(len(slot_keys), 2)
            native_keys = set()
            for batches in first.batches:
                for batch in batches:
                    for tensor in batch[6:8]:
                        native_keys.add(
                            (str(tensor.device), tensor.untyped_storage().data_ptr())
                        )
            self.assertEqual(len(native_keys), 6 * 2 * 2)
            self.assertTrue(slot_keys.isdisjoint(native_keys))
            self.assertTrue(
                set(ledger).isdisjoint(a.storage_key for a in second.allocations())
            )


if __name__ == "__main__":
    unittest.main()
