"""CPU differential checks for the graph-oriented fixed tensor residency planner."""

import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe.config import (
    ExpertCatalog,
    ExpertLayerSpec,
    Phase,
)
from sglang.srt.offloading.moe.execution import (
    MaterializationPlan,
)
from sglang.srt.offloading.moe.layer_execution import (
    BoundExecutionLayer,
    LayerExecutionRequest,
)
from sglang.srt.offloading.moe.residency import TensorResidencyOps
from sglang.srt.offloading.moe.resources import (
    AllocationLedger,
    ResourceKind,
    tensor_allocations,
)
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.transfer import (
    Completion,
    CopySourceKind,
    HostCopyPlan,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Fence(Completion):
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def wait_on(self, stream):
        self.calls.append(("wait", self.name, stream))

    def synchronize(self):
        raise AssertionError("Tensor planning must not synchronize the host")

    def done(self):
        raise AssertionError("Tensor planning must not query device completion")


class Context:
    phase = Phase.DECODE
    capturing = True
    compute_stream = "compute"
    copy_stream = "copy"

    def __init__(self):
        self.calls, self.retained = [], []
        self.invocation_id = object()

    def retain(self, resource):
        self.retained.append(resource)

    def record_compute_completion(self):
        name = "event" + str(len(self.calls))
        self.calls.append(("record", name))
        return TransferTicket(Fence(self.calls, name))

    def copied(self):
        return TransferTicket(Fence(self.calls, "bundle"))

    def wait_for(self, ticket, stream):
        ticket.completion.wait_on(stream)


def setup(hot=3, experts=7):
    cat = ExpertCatalog(
        (ExpertLayerSpec(3, experts, 1, 0), ExpertLayerSpec(4, experts, 1, 0))
    )
    state = ResidencyState.allocate(cat.num_experts, hot, experts, "cpu")
    error = torch.zeros((), dtype=torch.int32)
    binding = SimpleNamespace(error_status=error, hot_generations=state.slot_generation)
    records = [
        BoundExecutionLayer(layer.layer_id, 0, 0, binding) for layer in cat.layers
    ]
    ops = TensorResidencyOps(
        state,
        cat,
        execution_layers=records,
        update_bindings=binding,
        error_status=error,
    )
    return ops, state, records, Context()


def load_hot(state, keys):
    state.key_to_slot.fill_(-1)
    state.slot_to_key.copy_(torch.tensor(keys))
    state.slot_valid.copy_(state.slot_to_key >= 0)
    for slot, key in enumerate(keys):
        if key >= 0:
            state.key_to_slot[key] = slot
    state.slot_generation.copy_(torch.arange(len(keys)) + 5)


class TestTensorResidency(unittest.TestCase):
    def test_allocation_reports_physical_storage_without_hiding_aliases(self):
        storage = torch.empty(8, dtype=torch.int64)
        view = storage[2:4]
        records = tensor_allocations(
            "scratch.",
            (("view", view), ("empty", torch.empty(0))),
            ResourceKind.WORKSPACE,
        )
        self.assertEqual(len(records), 1)
        allocation = records[0]
        self.assertEqual(allocation.name, "scratch.view")
        self.assertEqual(allocation.nbytes, storage.untyped_storage().nbytes())
        self.assertEqual(allocation.storage_key, ("cpu", storage.data_ptr()))
        self.assertIs(allocation.keepalive, view)
        empty_view = tensor_allocations("view.", (("empty", storage[:0]),))
        self.assertEqual(empty_view[0].nbytes, allocation.nbytes)
        ledger = AllocationLedger({ResourceKind.WORKSPACE: 64}, records)
        self.assertEqual(ledger.usage[ResourceKind.WORKSPACE], 64)
        aliases = tensor_allocations("alias.", (("base", storage), ("view", view)))
        with self.assertRaisesRegex(ValueError, "physical allocations once"):
            AllocationLedger({ResourceKind.METADATA: 128}, aliases)

    def test_policy_gate_is_latched_before_policy_reuses_due(self):
        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        due = torch.tensor(True)
        update = ops.prepare_update(torch.tensor([1, 2, 9]), ctx, update_enabled=due)
        due.zero_()
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertEqual(state.slot_to_key.tolist(), [1, 2, 9])
        self.assertEqual(state.slot_generation.tolist(), [5, 7, 7])

    def test_false_policy_gate_ignores_stale_targets_and_preserves_generation(self):
        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        before = {name: value.clone() for name, value in vars(state).items()}
        update = ops.prepare_update(
            torch.tensor([99, 99, -2]), ctx, update_enabled=torch.tensor(False)
        )
        self.assertEqual(int(update.copy_plan.count), 0)
        self.assertFalse(bool(update.copy_plan.valid.any()))
        ops.commit_update(update, ctx.copied(), ctx)
        for name, value in before.items():
            torch.testing.assert_close(getattr(state, name), value)
        self.assertEqual(int(ops.error_status), 0)
        self.assertNotIn("tail.target", ops._owned)

    def test_live_execution_bank_cannot_be_replanned_or_evicted(self):
        # Migrated from the retired cache integration stack: native production
        # residency must enforce exactly the same reader ownership contract.
        ops, state, _, ctx = setup()
        request = LayerExecutionRequest(
            3, torch.ones(7, dtype=torch.int64), Phase.DECODE
        )
        reservation = ops.resolve_execution(request, ctx)
        with self.assertRaisesRegex(RuntimeError, "Retire"):
            ops.resolve_execution(request, ctx)
        with self.assertRaisesRegex(RuntimeError, "Retire"):
            ops.prepare_update(state.slot_to_key, ctx)
        with self.assertRaisesRegex(RuntimeError, "Retire"):
            ops.begin_step(ctx)
        ops.publish_execution(reservation, ctx.copied(), ctx)
        ops.finish_execution(reservation, ctx.copied(), ctx)
        ops.begin_step(ctx)

    def test_hot_update_joins_auxiliary_readers_before_invalidating_sources(self):
        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        reservation = ops.resolve_execution(
            LayerExecutionRequest(3, torch.ones(7, dtype=torch.int64), Phase.DECODE),
            ctx,
        )
        ops.publish_execution(reservation, ctx.copied(), ctx)
        ops.finish_execution(
            reservation, TransferTicket(Fence(ctx.calls, "aux-reader")), ctx
        )
        before = len(ctx.calls)
        update = ops.prepare_update(torch.tensor([1, 2, 9]), ctx)
        self.assertEqual(ctx.calls[before], ("wait", "aux-reader", "compute"))
        self.assertEqual(len(update.copy_dependencies), 1)
        self.assertEqual(ctx.calls[-1], ("record", update.copy_dependencies[0].name))
        self.assertEqual(len(reservation.copy_dependencies), 1)
        ops.commit_update(update, ctx.copied(), ctx)

    def test_mixed_sources_and_counts_come_from_post_dispatch(self):
        ops, state, _, ctx = setup()
        load_hot(state, [1, 5, 9])
        counts = torch.tensor([2, 4, 0, 0, 1, 3, 8])
        ops.begin_step(ctx)
        expected = [0] * 14
        request = LayerExecutionRequest(3, counts, Phase.DECODE)
        reservation = ops.resolve_execution(request, ctx)
        plan = reservation.copy_plan
        for i, key in enumerate(range(7)):
            expected[key] = counts[key].item()
            self.assertEqual(plan.valid[i].item(), bool(counts[key]))
            if counts[key]:
                self.assertEqual(
                    plan.source_kind[i].item(),
                    int(
                        CopySourceKind.HOT_SLOT
                        if key in (1, 5)
                        else CopySourceKind.HOST
                    ),
                )
                self.assertEqual(
                    plan.source_index[i].item(),
                    [1, 5, 9].index(key) if key in (1, 5) else key,
                )
        ops.publish_execution(reservation, ctx.copied(), ctx)
        self.assertEqual(
            state.execution_valid[: plan.capacity].tolist(), plan.valid.tolist()
        )
        ops.finish_execution(reservation, ctx.copied(), ctx)
        self.assertEqual(state.step_access_count.tolist(), expected)
        self.assertEqual(state.slot_to_key.tolist(), [1, 5, 9])
        self.assertEqual(state.slot_generation.tolist(), [5, 6, 7])

    def test_empty_layer_demand_and_zero_hot(self):
        ops, state, _, ctx = setup(hot=0)
        counts = torch.zeros(7, dtype=torch.int32)
        reservation = ops.resolve_execution(
            LayerExecutionRequest(3, counts, Phase.DECODE), ctx
        )
        self.assertEqual(reservation.copy_plan.count.item(), 0)
        ops.publish_execution(reservation, ctx.copied(), ctx)
        self.assertFalse(state.execution_valid[: reservation.copy_plan.capacity].any())
        ops.finish_execution(reservation, ctx.copied(), ctx)
        update = ops.prepare_update(torch.empty(0, dtype=torch.int64), ctx)
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertEqual(ops.error_status.item(), 0)

    def test_tail_preserves_slots_invalidates_victims_before_commit(self):
        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        update = ops.prepare_update(torch.tensor([9, 2, 1]), ctx)
        self.assertEqual(state.slot_to_key.tolist(), [1, 5, 9])
        self.assertEqual(state.slot_valid.tolist(), [True, False, True])
        self.assertEqual(state.key_to_slot[5].item(), -1)
        self.assertEqual(state.key_to_slot[2].item(), -1)
        self.assertEqual(update.copy_plan.valid.tolist(), [False, True, False])
        self.assertEqual(update.copy_plan.source_index.tolist(), [1, 2, 9])
        self.assertEqual(update.copy_plan.count.item(), 1)
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertEqual(state.slot_to_key.tolist(), [1, 2, 9])
        self.assertEqual(state.slot_generation.tolist(), [5, 7, 7])
        self.assertEqual(state.key_to_slot[2].item(), 1)
        self.assertIn(("wait", "bundle", "compute"), ctx.calls)
        self.assertEqual(ctx.calls[-1][0], "record")

    def test_disabled_updates_preserve_every_metadata_tensor(self):
        ops, state, _, ctx = setup()
        load_hot(state, [1, 5, -1])
        old = {name: tensor.clone() for name, tensor in vars(state).items()}
        update = ops.prepare_update(torch.tensor([3, 3, 888]), ctx)
        self.assertEqual(update.copy_plan.count.item(), 0)
        ops.commit_update(update, ctx.copied(), ctx)
        for name, tensor in old.items():
            self.assertTrue(torch.equal(getattr(state, name), tensor), name)
        self.assertEqual(ops.error_status.item(), 0)

    def test_randomized_tail_matches_reference_and_stable_allocations(self):
        rng = random.Random(1021)
        for hot in (1, 3, 7, 14):
            ops, state, _, ctx = setup(hot=hot)
            state.update_enabled.fill_(True)
            keys = [-1] * hot
            generation = [0] * hot
            pointers = [a.storage_key for a in ops.allocations()]
            for _ in range(80):
                chosen = rng.sample(range(14), rng.randrange(hot + 1))
                targets = chosen + [-1] * (hot - len(chosen))
                retained = set(keys) & set(chosen)
                newcomers = iter(key for key in chosen if key not in retained)
                proposed = [
                    key if key in retained else next(newcomers, -1) for key in keys
                ]
                generation = [
                    g + int(a != b) for a, b, g in zip(keys, proposed, generation)
                ]
                update = ops.prepare_update(
                    torch.tensor(targets, dtype=torch.int64), ctx
                )
                self.assertEqual(
                    update.copy_plan.count.item(),
                    sum(a != b and b >= 0 for a, b in zip(keys, proposed)),
                )
                ops.commit_update(update, ctx.copied(), ctx)
                self.assertEqual(state.slot_to_key.tolist(), proposed)
                self.assertEqual(state.slot_generation.tolist(), generation)
                self.assertEqual(
                    state.slot_valid.tolist(), [key >= 0 for key in proposed]
                )
                inverse = [-1] * 14
                for slot, key in enumerate(proposed):
                    if key >= 0:
                        inverse[key] = slot
                self.assertEqual(state.key_to_slot.tolist(), inverse)
                self.assertEqual(ops.error_status.item(), 0)
                keys = proposed
            self.assertEqual([a.storage_key for a in ops.allocations()], pointers)
            self.assertEqual(len(set(pointers)), len(pointers))
            self.assertGreater(sum(item.nbytes for item in ops.allocations()), 0)

    def test_copy_error_suppresses_execution_and_hot_publication(self):
        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        reservation = ops.resolve_execution(
            LayerExecutionRequest(3, torch.ones(7, dtype=torch.int64), Phase.DECODE),
            ctx,
        )
        ops.error_status.fill_(1)
        ops.publish_execution(reservation, ctx.copied(), ctx)
        self.assertFalse(state.execution_valid.any())
        self.assertFalse(bool(state.execution_valid.any()))
        ops.finish_execution(reservation, ctx.copied(), ctx)
        ops.error_status.zero_()
        update = ops.prepare_update(torch.tensor([1, 2, 9]), ctx)
        ops.error_status.fill_(1)
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertEqual(state.key_to_slot[2].item(), -1)
        self.assertFalse(state.slot_valid[1])

    def test_generation_change_during_copy_poisoned(self):
        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        update = ops.prepare_update(torch.tensor([1, 2, 9]), ctx)
        state.slot_generation[1] += 1
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertNotEqual(ops.error_status.item(), 0)
        self.assertFalse(state.slot_valid[1])
        self.assertEqual(state.key_to_slot[2].item(), -1)

    def test_bad_targets_latch_error_and_never_copy_or_publish(self):
        for target in ([1, 1, -1], [-2, -1, -1], [14, -1, -1]):
            ops, state, _, ctx = setup()
            load_hot(state, [1, 5, 9])
            state.update_enabled.fill_(True)
            old = state.slot_to_key.clone()
            update = ops.prepare_update(torch.tensor(target), ctx)
            self.assertEqual(update.copy_plan.count.item(), 0)
            ops.commit_update(update, ctx.copied(), ctx)
            self.assertNotEqual(ops.error_status.item(), 0)
            self.assertTrue(torch.equal(old, state.slot_to_key))

    def test_rejects_foreign_double_and_unbound_operations(self):
        ops, _, _, ctx = setup()
        req = LayerExecutionRequest(3, torch.ones(7, dtype=torch.int64), Phase.DECODE)
        reservation = ops.resolve_execution(req, ctx)
        with self.assertRaises(ValueError):
            ops.publish_execution(reservation, ctx.copied(), Context())
        ops.publish_execution(reservation, ctx.copied(), ctx)
        with self.assertRaises(RuntimeError):
            ops.publish_execution(reservation, ctx.copied(), ctx)
        ops.finish_execution(reservation, ctx.copied(), ctx)
        with self.assertRaises(RuntimeError):
            ops.finish_execution(reservation, ctx.copied(), ctx)
        with self.assertRaises(ValueError):
            ops.resolve_execution(
                LayerExecutionRequest(5, req.local_demand, Phase.DECODE), ctx
            )
        update = ops.prepare_update(torch.tensor([-1, -1, -1]), ctx)
        with self.assertRaises(RuntimeError):
            ops.prepare_update(torch.tensor([-1, -1, -1]), ctx)
        ops.commit_update(update, ctx.copied(), ctx)
        with self.assertRaises(RuntimeError):
            ops.commit_update(update, ctx.copied(), ctx)

    def test_prefill_injected_host_plan_does_not_mutate_decode_state(self):
        ops, state, _, ctx = setup()
        ctx.phase, ctx.capturing = Phase.PREFILL, False
        old = {name: tensor.clone() for name, tensor in vars(state).items()}
        ops.prepare_prefill = lambda request, context: MaterializationPlan(
            HostCopyPlan(()), "borrowed"
        )
        request = LayerExecutionRequest(3, None, Phase.PREFILL)
        reservation = ops.resolve_execution(request, ctx)
        ops.publish_execution(reservation, ctx.copied(), ctx)
        ops.finish_execution(reservation, ctx.copied(), ctx)
        for name, tensor in old.items():
            self.assertTrue(torch.equal(getattr(state, name), tensor), name)

    def test_planning_does_not_extract_tensor_values(self):
        from unittest.mock import patch

        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        demand, targets = torch.ones(7, dtype=torch.int64), torch.tensor([1, 3, 5])
        with (
            patch.object(torch.Tensor, "item", side_effect=AssertionError("host item")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("host list")
            ),
        ):
            ops.begin_step(ctx)
            reservation = ops.resolve_execution(
                LayerExecutionRequest(3, demand, Phase.DECODE), ctx
            )
            ops.publish_execution(reservation, ctx.copied(), ctx)
            ops.finish_execution(reservation, ctx.copied(), ctx)
            update = ops.prepare_update(targets, ctx)
            ops.commit_update(update, ctx.copied(), ctx)

    def test_invalid_slot_with_same_old_key_is_reloaded(self):
        ops, state, _, ctx = setup()
        load_hot(state, [1, 5, 9])
        state.update_enabled.fill_(True)
        state.slot_valid[1] = False
        state.key_to_slot[5] = -1
        update = ops.prepare_update(torch.tensor([1, 5, 9]), ctx)
        self.assertEqual(update.copy_plan.valid.tolist(), [False, True, False])
        self.assertEqual(update.copy_plan.count.item(), 1)
        self.assertEqual(state.slot_generation.tolist(), [5, 7, 7])
        self.assertFalse(state.slot_valid[1])
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertTrue(state.slot_valid[1])
        self.assertEqual(state.key_to_slot[5].item(), 1)

    def test_corrupt_inverse_is_not_silently_treated_as_host_miss(self):
        for kind in ("invalid", "wrong_key", "too_large", "negative"):
            ops, state, _, ctx = setup()
            load_hot(state, [1, 5, 9])
            if kind == "invalid":
                state.slot_valid[0] = False
            elif kind == "wrong_key":
                state.slot_to_key[0] = 7
            elif kind == "too_large":
                state.key_to_slot[1] = 100
            else:
                state.key_to_slot[1] = -2
            request = LayerExecutionRequest(
                3, torch.ones(7, dtype=torch.int64), Phase.DECODE
            )
            reservation = ops.resolve_execution(request, ctx)
            self.assertNotEqual(ops.error_status.item(), 0, kind)
            self.assertEqual(reservation.copy_plan.count.item(), 0, kind)
            ops.publish_execution(reservation, ctx.copied(), ctx)
            self.assertFalse(state.execution_valid.any())

    def test_tail_rejects_corrupt_canonical_maps_without_out_of_bounds(self):
        for kind in (
            "missing_inverse",
            "duplicate",
            "key_out_of_bounds",
            "duplicate_inverse_sum",
        ):
            ops, state, _, ctx = setup()
            load_hot(state, [1, 5, 9])
            state.update_enabled.fill_(True)
            if kind == "missing_inverse":
                state.key_to_slot[5] = -1
            elif kind == "duplicate":
                state.slot_to_key[1] = 1
            elif kind == "duplicate_inverse_sum":
                state.slot_to_key.copy_(torch.tensor([1, 1, -1]))
                state.slot_valid.copy_(torch.tensor([True, True, False]))
                state.key_to_slot.fill_(-1)
                state.key_to_slot[1] = 2  # Sum of slots+1, minus 1, mimics an inverse.
            else:
                state.slot_to_key[1] = 100
            update = ops.prepare_update(torch.tensor([1, 5, 9]), ctx)
            self.assertEqual(update.copy_plan.count.item(), 0, kind)
            ops.commit_update(update, ctx.copied(), ctx)
            self.assertNotEqual(ops.error_status.item(), 0, kind)

    def test_retired_bank_reader_waits_before_next_overwrite(self):
        ops, _, _, ctx = setup()
        demand = torch.ones(7, dtype=torch.int64)
        first = ops.resolve_execution(
            LayerExecutionRequest(3, demand, Phase.DECODE), ctx
        )
        ops.publish_execution(first, ctx.copied(), ctx)
        reader = TransferTicket(Fence(ctx.calls, "reader"))
        ops.finish_execution(first, reader, ctx)
        before = len(ctx.calls)
        ops.resolve_execution(LayerExecutionRequest(3, demand, Phase.DECODE), ctx)
        self.assertEqual(ctx.calls[before], ("wait", "reader", "compute"))

    def test_prefill_publication_defers_compute_wait_until_consumer(self):
        ops, _, _, ctx = setup()
        ctx.phase, ctx.capturing = Phase.PREFILL, False
        ops.prepare_prefill = lambda request, context: MaterializationPlan(
            HostCopyPlan(()), "borrowed"
        )
        request = LayerExecutionRequest(3, None, Phase.PREFILL)
        reservation = ops.resolve_execution(request, ctx)
        ready = ops.publish_execution(reservation, ctx.copied(), ctx)
        self.assertEqual(ctx.calls, [])
        ready.completion.wait_on(ctx.compute_stream)
        self.assertEqual(ctx.calls, [("wait", "bundle", "compute")])
