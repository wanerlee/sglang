"""Canonical tensor state ABI shared by composed and fused programs."""

from dataclasses import dataclass, fields

from .resources import Allocation, ResourceKind


@dataclass
class ResidencyState:
    key_to_slot: object
    slot_to_key: object
    slot_valid: object
    slot_generation: object
    execution_key: object
    execution_valid: object
    step_access_count: object
    update_enabled: object

    @classmethod
    def allocate(cls, num_experts, hot_slots, execution_slots, device):
        """Allocate before capture. CachePolicy owns separate algorithm state."""
        import torch

        if min(num_experts, hot_slots, execution_slots) < 0 or hot_slots > num_experts:
            raise ValueError("Invalid residency capacities")
        return cls(
            torch.full((num_experts,), -1, dtype=torch.int64, device=device),
            torch.full((hot_slots,), -1, dtype=torch.int64, device=device),
            torch.zeros(hot_slots, dtype=torch.bool, device=device),
            torch.zeros(hot_slots, dtype=torch.int64, device=device),
            torch.full((execution_slots,), -1, dtype=torch.int64, device=device),
            torch.zeros(execution_slots, dtype=torch.bool, device=device),
            torch.zeros(num_experts, dtype=torch.int64, device=device),
            torch.zeros((), dtype=torch.bool, device=device),
        )

    def allocations(self):
        """Report residency tensors for the binder's allocation ledger.

        The binder also reports policy state, snapshots and plans.
        """
        allocations = []
        for field in fields(self):
            tensor = getattr(self, field.name)
            storage = tensor.untyped_storage()
            if storage.nbytes() == 0:
                continue
            allocations.append(
                Allocation(
                    f"residency.{field.name}",
                    ResourceKind.METADATA,
                    storage.nbytes(),
                    (str(tensor.device), storage.data_ptr()),
                    tensor,
                )
            )
        return tuple(allocations)
