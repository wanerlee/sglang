"""Cache bookkeeping independent of transfer plans, slot placement and streams.

Route adapters supply valid rank-local assignments after EP dispatch. All online
decisions stay on the tensor device. These helpers do NOT implement ResidencyOps
or claim accelerator capture support; they own metadata, never expert weights.
"""

from dataclasses import dataclass, fields

from .config import ExpertCatalog
from .periodic_lfu import PeriodicLFUPolicy
from .resources import Allocation, ResourceKind
from .state import ResidencyState


@dataclass
class CacheRouteStats:
    """Valid token-expert assignments, not unique experts or transferred bytes."""

    accesses: object
    hits: object
    misses: object

    @classmethod
    def allocate(cls, device):
        import torch

        return cls(
            *(torch.zeros((), dtype=torch.int64, device=device) for _ in range(3))
        )

    def clear(self):
        for field in fields(self):
            getattr(self, field.name).zero_()


@dataclass
class HotMetadataSnapshot:
    """Only hot residency metadata. Execution banks contain disposable weights."""

    key_to_slot: object
    slot_to_key: object
    slot_valid: object
    slot_generation: object

    @classmethod
    def allocate(cls, state):
        import torch

        return cls(
            *(torch.empty_like(getattr(state, field.name)) for field in fields(cls))
        )

    def save(self, state):
        for field in fields(self):
            getattr(self, field.name).copy_(getattr(state, field.name))

    def restore(self, state):
        for field in fields(self):
            getattr(state, field.name).copy_(getattr(self, field.name))


class CacheMetadataAdapter:
    """Metadata-only step lifecycle for a bound PeriodicLFU policy.

    Allocate before capture. The runtime calls begin_step once, records each
    actual assignment once across disjoint waves; the coordinator finishes LFU.
    It supplies synchronization for phase transitions and handles H2D/publication
    separately. No routine here modifies live hot residency during a decode step.
    """

    def __init__(
        self,
        catalog: ExpertCatalog,
        residency: ResidencyState,
        policy: PeriodicLFUPolicy,
    ):
        import torch

        if policy.catalog != catalog:
            raise ValueError("Policy and metadata adapter must share the catalog")
        if residency.key_to_slot.numel() != catalog.num_experts:
            raise ValueError("Residency must match the expert catalog")
        self.catalog = catalog
        self.residency = residency
        self.policy = policy
        self.layer_layout = {
            layer.layer_id: (
                catalog.key(layer.layer_id, 0),
                layer.local_experts,
            )
            for layer in catalog.offloaded_layers
        }
        device = residency.key_to_slot.device
        self.policy_state = policy.allocate_state(catalog.num_experts, device)
        self.saved_policy_state = policy.allocate_state(catalog.num_experts, device)
        self.target_keys = torch.full_like(residency.slot_to_key, -1)
        self.stats = CacheRouteStats.allocate(device)
        self.snapshot = HotMetadataSnapshot.allocate(residency)
        self._has_snapshot = False

    def begin_step(self, update_enabled):
        """update_enabled is a fixed-address device bool, mutable before replay."""
        self.residency.step_access_count.zero_()
        self.residency.update_enabled.copy_(update_enabled)
        self.stats.clear()

    def record_local_routes(
        self, layer_id, local_expert_ids, valid, *, local_start=0, local_end=None
    ):
        """Count normalized post-dispatch token-expert assignments.

        ids and bool valid have the same fixed shape (any padding is masked).
        IDs are local to this EP rank and layer, not global IDs, keys, or slots.
        Invalid/out-of-range IDs are ignored. To reuse a full routing tensor
        across waves, pass disjoint [local_start, local_end) ranges. Otherwise
        supply each wave's distinct dispatched rows; never count both forms.
        The range is static scheduling metadata, not a device routing value.
        Return per-expert demand counts, independent of the LFU update gate so
        warmup/capture can still prepare the weights needed by native F.
        """
        import torch

        offset, size = self.layer_layout[layer_id]
        end = size if local_end is None else local_end
        if not 0 <= local_start <= end <= size:
            raise ValueError("Route range is outside the layer's local experts")
        if local_expert_ids.shape != valid.shape:
            raise ValueError("Route IDs and validity must have matching shapes")
        if local_expert_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("Route IDs must be int32 or int64")
        if valid.dtype != torch.bool:
            raise TypeError("Route validity must be bool")
        ids = local_expert_ids.reshape(-1)
        active = valid.reshape(-1) & (ids >= local_start) & (ids < end)
        # One histogram serves both LFU accounting and current-wave demand.
        # The sentinel absorbs invalid/padded IDs without a host read.
        counts = torch.zeros(size + 1, dtype=torch.int64, device=ids.device)
        counts.scatter_add_(
            0, torch.where(active, ids.to(torch.int64), size), active.to(torch.int64)
        )
        counts = counts[:size]
        self.residency.step_access_count[offset : offset + size].add_(
            torch.where(self.residency.update_enabled, counts, 0)
        )
        active = active & self.residency.update_enabled
        keys = ids.clamp(0, size - 1).to(torch.int64) + offset
        increments = active.to(torch.int64)

        slots = self.residency.key_to_slot[keys]
        capacity = self.residency.slot_to_key.numel()
        if capacity:  # Static shape, independent of route/cache contents.
            safe_slots = slots.clamp(0, capacity - 1)
            hit = (
                active
                & (slots >= 0)
                & (slots < capacity)
                & self.residency.slot_valid[safe_slots]
                & (self.residency.slot_to_key[safe_slots] == keys)
            )
        else:
            hit = torch.zeros_like(active)
        accesses = increments.sum()
        hits = hit.to(torch.int64).sum()
        self.stats.accesses.add_(accesses)
        self.stats.hits.add_(hits)
        self.stats.misses.add_(accesses - hits)
        return counts

    def save_decode_metadata(self):
        """Host phase transition, after prior streams have drained; no weights."""
        self.snapshot.save(self.residency)
        self.policy.copy_state(self.policy_state, self.saved_policy_state)
        self._has_snapshot = True

    def restore_decode_metadata_after_refill(self):
        """Host transition AFTER original hot weights are back in original slots.

        This method does not wait for a transfer and must never be used as proof
        that weights are ready. The runtime supplies that precondition. Execution
        bank contents are not snapshotted/restored and are invalidated here.
        """
        if not self._has_snapshot:
            raise RuntimeError("Save decode metadata before restoring it")
        self.snapshot.restore(self.residency)
        self.policy.copy_state(self.saved_policy_state, self.policy_state)
        self.residency.execution_key.fill_(-1)
        self.residency.execution_valid.zero_()
        self.residency.step_access_count.zero_()
        self.residency.update_enabled.zero_()
        self.stats.clear()
        self.target_keys.fill_(-1)
        self._has_snapshot = False

    def allocations(self):
        """Owned persistent tensors only; binder separately budgets op workspace.

        ResidencyState.allocations() is reported separately; no weights or
        residency tensor aliases are charged a second time here.
        """
        groups = {
            "policy": self.policy_state,
            "saved_policy": self.saved_policy_state,
            "stats": self.stats,
            "snapshot": self.snapshot,
        }
        tensors = [("target_keys", self.target_keys)]
        for prefix, group in groups.items():
            tensors.extend(
                (f"{prefix}.{field.name}", getattr(group, field.name))
                for field in fields(group)
            )
        result = []
        for name, tensor in tensors:
            storage = tensor.untyped_storage()
            if storage.nbytes():
                result.append(
                    Allocation(
                        f"cache_metadata.{name}",
                        ResourceKind.METADATA,
                        storage.nbytes(),
                        (str(tensor.device), storage.data_ptr()),
                        tensor,
                    )
                )
        return tuple(result)
