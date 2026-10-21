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


def cache_update_enabled(
    forward_mode,
    *,
    has_global_tokens: bool,
    warmup: bool = False,
    capture: bool = False,
    verify_enabled: bool = False,
) -> bool:
    """Host-side gate computed before eager execution or graph replay.

    forward_mode is the group's logical ForwardMode, not a rank's local IDLE
    substitution. A valid DECODE with zero local routes still advances the LFU.
    The caller writes this result to a fixed-address device bool replay input;
    do not embed the capture-time False as a constant inside the graph.
    verify_enabled is opt-in for future integration, not permission to bypass
    ModelRunner's current speculative-decoding restriction.
    """
    if warmup or capture or not has_global_tokens:
        return False
    return forward_mode.is_decode() or (
        verify_enabled and forward_mode.is_target_verify()
    )


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
    actual assignment once across disjoint waves, then calls finish_step once.
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
        self._layers = {
            layer.layer_id: (
                catalog.key(layer.layer_id, 0),
                layer.local_experts,
                layer.global_start,
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
        """
        import torch

        offset, size, _ = self._layers[layer_id]
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
        active = (
            valid.reshape(-1)
            & (ids >= local_start)
            & (ids < end)
            & self.residency.update_enabled
        )
        # Invalid IDs use a safe index and a zero increment.
        keys = ids.clamp(0, size - 1).to(torch.int64) + offset
        increments = active.to(torch.int64)
        self.residency.step_access_count.scatter_add_(0, keys, increments)

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

    def record_global_routes(self, layer_id, global_expert_ids, valid, **kwargs):
        """Normalize contiguous EP ownership; foreign experts are not counted."""
        _, _, global_start = self._layers[layer_id]
        self.record_local_routes(
            layer_id, global_expert_ids - global_start, valid, **kwargs
        )

    def finish_step(self, *, select_hot_set=True):
        """Accumulate once; optionally select a target set, without publishing.

        Eager callers can skip selection using the host forward-cycle counter.
        Captured calls keep the default tensor-gated path. target_keys is only
        an output when selection runs; skipped steps must not consume it.
        """
        self.policy.update_statistics(
            self.policy_state,
            self.residency.step_access_count,
            self.residency.update_enabled,
        )
        if select_hot_set:
            self.policy.select_hot_set(
                self.policy_state, self.residency.key_to_slot, self.target_keys
            )
        self.policy.finish_step(self.policy_state)

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
