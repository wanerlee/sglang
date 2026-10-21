"""Tensor-only cache plans; physical bindings and copies are backend supplied.

Source encoding used by this caller: NONE=0, HOST=1, HOT_SLOT=2. HOST indexes
are catalog keys; HOT_SLOT indexes are hot-pool slots. Destinations index the
pool passed as bindings, never an implicit allocation made by the transport.
"""

from dataclasses import dataclass, fields
from enum import IntEnum

from ..transfer import DeviceCopyPlan
from .interfaces import ResidencyOps
from .resources import Allocation, ResourceKind


class CopySource(IntEnum):
    NONE = 0
    HOST = 1
    HOT_SLOT = 2


@dataclass
class CacheUpdatePlan:
    copies: DeviceCopyPlan
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

    def __init__(self, state):
        import torch

        self.state = state
        device = state.key_to_slot.device
        self.update = CacheUpdatePlan(
            allocate_copy_plan(state.slot_to_key.numel(), device),
            torch.full_like(state.slot_to_key, -1),
            torch.zeros((), dtype=torch.bool, device=device),
        )
        self.execution = allocate_copy_plan(state.execution_key.numel(), device)
        self.execution_keys = torch.full_like(state.execution_key, -1)

    def prepare_update(self, target_keys, context):
        import torch

        state, plan = self.state, self.update
        plan.publish_enabled.copy_(state.update_enabled)
        count, capacity = state.key_to_slot.numel(), state.slot_to_key.numel()
        if not capacity:
            return plan
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
        return plan

    def prepare_refill(self, slot_keys):
        """Host lifecycle: exact positions for initial load or snapshot refill."""
        self.update.slot_keys.copy_(slot_keys)
        self.update.publish_enabled.fill_(True)
        self._prepare_hot_copies(slot_keys >= 0)
        return self.update

    def _prepare_hot_copies(self, valid):
        import torch

        plan = self.update
        plan.copies.valid.copy_(valid)
        plan.copies.source_kind.copy_(torch.where(valid, int(CopySource.HOST), 0))
        plan.copies.source_index.copy_(plan.slot_keys.clamp(min=0))
        plan.copies.expected_generation.zero_()
        plan.copies.count.copy_(valid.to(torch.int64).sum())
        self.state.slot_valid.logical_and_(~valid)

    def commit_update(self, update_plan, weights_ready, context):
        import torch

        # Device wait, followed by publications on that same compute stream.
        weights_ready.completion.wait_on(context.compute_stream)
        state = self.state
        with context.device_module.stream(context.compute_stream):
            keys = update_plan.slot_keys
            count = state.key_to_slot.numel()
            reverse = torch.full(
                (count + 1,), -1, dtype=torch.int64, device=keys.device
            )
            reverse.scatter_(
                0,
                torch.where(keys >= 0, keys, count),
                update_plan.copies.destination_slot,
            )
            enabled = update_plan.publish_enabled
            state.key_to_slot.copy_(
                torch.where(enabled, reverse[:count], state.key_to_slot)
            )
            state.slot_to_key.copy_(torch.where(enabled, keys, state.slot_to_key))
            state.slot_valid.copy_(torch.where(enabled, keys >= 0, state.slot_valid))
            state.slot_generation.add_(
                (update_plan.copies.valid & enabled).to(torch.int64)
            )

    def resolve_execution(self, local_demand, context):
        """local_demand is a full-bank key vector, -1 for unneeded lanes.

        The caller derives it from actual dispatch assignments, not wave size.
        Wave-local expert ID determines its native F lane; other banks are -1.
        """
        import torch

        state, plan = self.state, self.execution
        self.execution_keys.copy_(local_demand)
        valid = local_demand >= 0
        keys = local_demand.clamp(min=0)
        slots = state.key_to_slot[keys]
        if state.slot_to_key.numel():
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
        return plan

    def publish_execution(self, ticket, context):
        ticket.completion.wait_on(context.compute_stream)
        with context.device_module.stream(context.compute_stream):
            self.state.execution_key.copy_(self.execution_keys)
            self.state.execution_valid.copy_(self.execution.valid)

    def allocations(self):
        tensors = [
            ("slot_keys", self.update.slot_keys),
            ("publish_enabled", self.update.publish_enabled),
            ("execution_keys", self.execution_keys),
        ]
        for prefix, plan in (
            ("hot", self.update.copies),
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
