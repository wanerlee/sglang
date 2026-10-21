"""Tensor-only cache plans; physical bindings and copies are backend supplied.

Source encoding used by this caller: NONE=0, HOST=1, HOT_SLOT=2. HOST indexes
are catalog keys; HOT_SLOT indexes are hot-pool slots. Destinations index the
pool passed as bindings, never an implicit allocation made by the transport.
"""

from dataclasses import dataclass, fields
from enum import IntEnum

from ..transfer import DeviceCopyPlan, ImmediateCompletion
from .config import Phase
from .execution import ExecutionReservation, SourcePolicy, UpdatePlan
from .interfaces import ResidencyOps
from .resources import Allocation, ResourceKind


class CopySource(IntEnum):
    NONE = 0
    HOST = 1
    HOT_SLOT = 2


@dataclass
class HotPublication:
    slot_keys: object
    publish_enabled: object


def allocate_copy_plan(capacity, device):
    import torch

    return DeviceCopyPlan(
        source_kind=torch.zeros(capacity, dtype=torch.int64, device=device),
        source_index=torch.zeros(capacity, dtype=torch.int64, device=device),
        destination_slot=torch.arange(capacity, dtype=torch.int64, device=device),
        valid=torch.zeros(capacity, dtype=torch.bool, device=device),
        expected_generation=torch.zeros(capacity, dtype=torch.int64, device=device),
        count=torch.zeros((), dtype=torch.int64, device=device),
        capacity=capacity,
    )


class TensorResidencyOps(ResidencyOps):
    """One serialized rank. Plans are allocated once, reused only after readers.

    The runtime establishes compute -> copy -> compute dependencies. Preparing
    an update invalidates victims; commit publishes on compute after the copy
    ticket. No device-dependent Python branches or host route reads are used.
    """

    def __init__(self, state, hot_bindings, execution_bindings):
        import torch

        self.state = state
        device = state.key_to_slot.device
        self.update = UpdatePlan(
            allocate_copy_plan(state.slot_to_key.numel(), device),
            hot_bindings,
            state=HotPublication(
                torch.full_like(state.slot_to_key, -1),
                torch.zeros((), dtype=torch.bool, device=device),
            ),
        )
        self.execution = allocate_copy_plan(state.execution_key.numel(), device)
        self.execution_keys = torch.full_like(state.execution_key, -1)
        self.execution_bindings = execution_bindings
        self._read_done = ImmediateCompletion()
        self._active = None

    def _wait_readers(self, context):
        # Order plan writes as well as weight writes after the last bank reader.
        if self._active is not None:
            raise RuntimeError("Release the previous execution reservation first")
        self._read_done.wait_on(context.compute_stream)

    def _update_reservation(self, context):
        from dataclasses import replace

        ready = context.record_compute_completion()
        context.retain(ready)
        return replace(
            self.update,
            copy_dependencies=(ready.completion, self._read_done),
        )

    def prepare_update(self, target_keys, context):
        import torch

        self._wait_readers(context)
        state, plan = self.state, self.update.state
        plan.publish_enabled.copy_(state.update_enabled)
        count, capacity = state.key_to_slot.numel(), state.slot_to_key.numel()
        if not capacity:
            return self._update_reservation(context)
        # Target keys are a set, not physical positions. Preserve all survivors.
        membership = torch.zeros(
            count + 1, dtype=torch.int64, device=target_keys.device
        )
        membership.scatter_add_(
            0,
            torch.where(target_keys >= 0, target_keys, count),
            torch.ones_like(target_keys),
        )
        wanted = membership[:count] > 0
        old = state.slot_to_key
        keep = (old >= 0) & state.slot_valid & wanted[old.clamp(min=0)]
        keys = torch.arange(count, device=target_keys.device)
        incoming = torch.sort(
            torch.where(wanted & (state.key_to_slot < 0), keys, count)
        ).values[:capacity]
        free_order = torch.argsort(keep.to(torch.int64), stable=True)
        replacements = torch.empty_like(incoming)
        replacements.scatter_(0, free_order, incoming)
        proposed = torch.where(
            keep, old, torch.where(replacements < count, replacements, -1)
        )
        plan.slot_keys.copy_(torch.where(state.update_enabled, proposed, old))
        self._prepare_hot_copies((plan.slot_keys >= 0) & ~keep & state.update_enabled)
        return self._update_reservation(context)

    def prepare_refill(self, slot_keys, context):
        """Host lifecycle: exact positions for initial load or snapshot refill."""
        self._wait_readers(context)
        self.update.state.slot_keys.copy_(slot_keys)
        self.update.state.publish_enabled.fill_(True)
        self._prepare_hot_copies(slot_keys >= 0)
        return self._update_reservation(context)

    def _prepare_hot_copies(self, valid):
        import torch

        plan = self.update.copy_plan
        plan.valid.copy_(valid)
        plan.source_kind.copy_(torch.where(valid, int(CopySource.HOST), 0))
        plan.source_index.copy_(self.update.state.slot_keys.clamp(min=0))
        plan.expected_generation.zero_()
        plan.count.copy_(valid.to(torch.int64).sum())
        self.state.slot_valid.logical_and_(~valid)

    def commit_update(self, update_plan, weights_ready, context):
        import torch

        # Device wait, followed by publications on that same compute stream.
        weights_ready.completion.wait_on(context.compute_stream)
        state = self.state
        with context.device_module.stream(context.compute_stream):
            keys = update_plan.state.slot_keys
            count = state.key_to_slot.numel()
            reverse = torch.full(
                (count + 1,), -1, dtype=torch.int64, device=keys.device
            )
            reverse.scatter_(
                0,
                torch.where(keys >= 0, keys, count),
                update_plan.copy_plan.destination_slot,
            )
            enabled = update_plan.state.publish_enabled
            state.key_to_slot.copy_(
                torch.where(enabled, reverse[:count], state.key_to_slot)
            )
            state.slot_to_key.copy_(torch.where(enabled, keys, state.slot_to_key))
            state.slot_valid.copy_(torch.where(enabled, keys >= 0, state.slot_valid))
            state.slot_generation.add_(
                (update_plan.copy_plan.valid & enabled).to(torch.int64)
            )
            return context.record_compute_completion()

    def resolve_execution(self, request, context):
        """local_demand is a full-bank key vector, -1 for unneeded lanes.

        The caller derives it from actual dispatch assignments, not wave size.
        Wave-local expert ID determines its native F lane; other banks are -1.
        """
        import torch

        if request.phase is not Phase.DECODE or context.phase is not Phase.DECODE:
            raise ValueError("TensorResidencyOps resolves decode device plans only")
        self._wait_readers(context)
        local_demand = request.local_demand
        state, plan = self.state, self.execution
        self.execution_keys.copy_(local_demand)
        valid = local_demand >= 0
        keys = local_demand.clamp(min=0)
        slots = state.key_to_slot[keys]
        if (
            state.slot_to_key.numel()
            and request.source_policy is SourcePolicy.PREFER_RESIDENT
        ):
            safe = slots.clamp(0, state.slot_to_key.numel() - 1)
            hit = (
                valid
                & (slots >= 0)
                & (slots < state.slot_to_key.numel())
                & state.slot_valid[safe]
                & (state.slot_to_key[safe] == keys)
            )
            plan.expected_generation.copy_(
                torch.where(hit, state.slot_generation[safe], 0)
            )
        else:
            hit = torch.zeros_like(valid)
            plan.expected_generation.zero_()
        plan.valid.copy_(valid)
        plan.source_kind.copy_(torch.where(valid, torch.where(hit, 2, 1), 0))
        plan.source_index.copy_(torch.where(hit, slots, keys))
        plan.count.copy_(valid.to(torch.int64).sum())
        state.execution_valid.zero_()
        ready = context.record_compute_completion()
        context.retain(ready)
        self._active = ExecutionReservation(
            plan,
            self.execution_bindings,
            copy_dependencies=(ready.completion, self._read_done),
            state=self.execution_keys,
        )
        return self._active

    def publish_execution(self, reservation, ticket, context):
        ticket.completion.wait_on(context.compute_stream)
        with context.device_module.stream(context.compute_stream):
            self.state.execution_key.copy_(reservation.state)
            self.state.execution_valid.copy_(reservation.copy_plan.valid)
            return context.record_compute_completion()

    def finish_execution(self, reservation, read_done, context):
        if reservation is not self._active:
            raise ValueError("Foreign execution reservation")
        self._read_done = read_done.completion
        self._active = None

    def allocations(self):
        tensors = [
            ("slot_keys", self.update.state.slot_keys),
            ("publish_enabled", self.update.state.publish_enabled),
            ("execution_keys", self.execution_keys),
        ]
        for prefix, plan in (
            ("hot", self.update.copy_plan),
            ("execution", self.execution),
        ):
            tensors.extend(
                (f"{prefix}.{field.name}", getattr(plan, field.name))
                for field in fields(plan)
                if field.name != "capacity"
            )
        return tuple(
            Allocation(
                f"cache_plan.{name}",
                ResourceKind.METADATA,
                tensor.untyped_storage().nbytes(),
                (str(tensor.device), tensor.untyped_storage().data_ptr()),
                tensor,
            )
            for name, tensor in tensors
            if tensor.numel()
        )
