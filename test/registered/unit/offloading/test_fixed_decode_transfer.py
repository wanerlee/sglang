"""Fixed decode destination contracts using the real MemFabric lowering on CPU.

These tests check descriptor contents and byte copies, not NPU timing or capture.
"""

import unittest
from dataclasses import replace

import torch
from test_memfabric_compaction import TransferHarness
from test_npu_backend import Assembly

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestFixedDecodeTransfer(unittest.TestCase):
    def assert_destinations(self, harness, expected):
        for name, rows in expected.items():
            self.assertTrue(torch.equal(harness.destination[name], rows), name)

    def test_binding_rejects_unproved_target_layouts(self):
        with TransferHarness(4) as h:
            for slots in (
                (0, 1, 2),
                (0, 1, 2, 2),
                (-1, 1, 2, 3),
                (0, 1, 2, h.destination_count),
                (0, 1, 2, 3.0),
                (False, 1, 2, 3),
                [0, 1, 2, 3],
            ):
                with self.subTest(slots=slots), self.assertRaises(ValueError):
                    h.new_binding(fixed_destination_slots=slots)

    def test_only_shared_compaction_scatters_remain_at_batch_boundaries(self):
        for capacity, batch_size in (
            (0, 256),
            (1, 256),
            (255, 256),
            (256, 256),
            (257, 256),
            (8, 3),
        ):
            with self.subTest(capacity=capacity, batch_size=batch_size):
                with TransferHarness(capacity, max_batch_entries=batch_size) as h:
                    slots = tuple(range(capacity))
                    h.binding = h.new_binding(fixed_destination_slots=slots)
                    plan = h.plan()
                    expected = h.expected(plan, destination_slots=slots)
                    _, scatters = h.run(plan)
                    batches = (capacity + batch_size - 1) // batch_size
                    self.assertEqual(scatters, batches)
                    self.assertEqual(len(h.calls), 6 * batches)
                    self.assert_destinations(h, expected)
                    self.assertIsNone(h.binding.destination_writes)
                    self.assertEqual(int(h.error), 0)

    def test_plan_target_mutation_cannot_redirect_fixed_copies(self):
        with TransferHarness(7, max_batch_entries=3) as h:
            slots = (4, 2, 0, 6, 1, 5, 3)
            h.binding = h.new_binding(fixed_destination_slots=slots)
            plan = h.plan()
            bound_indices = h.binding._fixed_destination_indices
            saved_indices = bound_indices.clone()
            self.assertNotEqual(
                bound_indices.data_ptr(), plan.destination_slot.data_ptr()
            )
            for mask, bogus_target in (
                ([True] * 7, 0),
                ([False, True, False, True, False, False, True], -99),
                ([False] * 7, 999),
                ([True, False, False, False, False, False, True], 999),
            ):
                with self.subTest(mask=mask, bogus_target=bogus_target):
                    plan.destination_slot.fill_(bogus_target)
                    plan.valid.copy_(torch.tensor(mask, dtype=torch.bool))
                    plan.count.copy_(plan.valid.sum(dtype=torch.int32))
                    expected = h.expected(plan, destination_slots=slots)
                    h.run(plan)
                    self.assert_destinations(h, expected)
                    self.assertTrue(torch.equal(bound_indices, saved_indices))
                    self.assertEqual(int(h.error), 0)

    def test_dynamic_source_count_and_fault_guards_remain_active(self):
        for fault in (
            "generation",
            "count",
            "host_source",
            "hot_source",
            "kind",
            "latched",
        ):
            with self.subTest(fault=fault), TransferHarness(5) as h:
                h.binding = h.new_binding(fixed_destination_slots=tuple(range(5)))
                plan = h.plan()
                if fault == "generation":
                    plan.expected_generation[1] += 1
                elif fault == "count":
                    plan.count.add_(1)
                elif fault == "host_source":
                    plan.source_index[0] = h.source_count
                elif fault == "hot_source":
                    plan.source_index[1] = h.source_count
                elif fault == "kind":
                    plan.source_kind[0] = 99
                else:
                    h.error.fill_(8)
                expected = h.expected(plan, suppressed=True)
                h.run(plan)
                self.assert_destinations(h, expected)
                self.assertNotEqual(int(h.error), 0)
                self.assertFalse(bool(h.binding.mask.any()))
                for src, dst, groups in h.calls:
                    self.assertTrue(all(value == 0 for value in (*src, *dst, *groups)))

    def test_plan_target_shape_and_dtype_still_obey_the_public_contract(self):
        with TransferHarness(3) as h:
            h.binding = h.new_binding(fixed_destination_slots=(0, 1, 2))
            plan = h.plan()
            for targets in (plan.destination_slot[:1], plan.destination_slot.float()):
                with self.subTest(shape=targets.shape, dtype=targets.dtype):
                    with self.assertRaises(ValueError):
                        h.run(replace(plan, destination_slot=targets))
            self.assertEqual(h.calls, [])

    def test_bound_indices_are_accounted_and_private_to_each_binding(self):
        with TransferHarness(4) as h:
            first = h.new_binding(fixed_destination_slots=(0, 1, 2, 3))
            second = h.new_binding(fixed_destination_slots=(0, 1, 2, 3))
            indices = first._fixed_destination_indices
            key = (str(indices.device), indices.untyped_storage().data_ptr())
            ledger = {
                allocation.storage_key: allocation for allocation in first.allocations()
            }
            self.assertIn(key, ledger)
            self.assertEqual(ledger[key].nbytes, 4 * 8)
            self.assertNotEqual(
                indices.data_ptr(), second._fixed_destination_indices.data_ptr()
            )
            for batches in first.batches:
                for batch in batches:
                    for native_tensor in batch[2:]:
                        self.assertNotEqual(
                            indices.data_ptr(), native_tensor.data_ptr()
                        )


class TestFixedDecodeBackendBinding(unittest.TestCase):
    def test_only_decode_uses_fixed_slots_matching_the_native_bank_order(self):
        assembly = Assembly(slots=4, local_experts=5)
        try:
            assembly.finalize()
            decode = [
                binding
                for binding in assembly.bindings
                if binding.name.startswith("decode.")
            ]
            self.assertTrue(decode)
            for binding in decode:
                self.assertEqual(
                    binding.fixed_destination_slots, tuple(range(binding.capacity))
                )
                self.assertEqual(binding.destination_count, binding.capacity)
                self.assertEqual(
                    binding.expert_bank.execution_valid.numel(), binding.capacity
                )
            update = next(
                binding for binding in assembly.bindings if binding.name == "hot.update"
            )
            self.assertIsNone(update.fixed_destination_slots)
        finally:
            assembly.close()


if __name__ == "__main__":
    unittest.main()
