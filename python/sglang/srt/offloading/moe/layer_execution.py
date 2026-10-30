"""Whole-layer execution contracts; no expert partition or wave identity."""

from dataclasses import dataclass

from .config import Phase


@dataclass(frozen=True)
class LayerExecutionRequest:
    layer_id: int
    local_demand: object
    phase: Phase
    bank_id: int = 0

    def __post_init__(self):
        if self.layer_id < 0 or self.bank_id < 0 or not isinstance(self.phase, Phase):
            raise ValueError("Layer execution requires a layer, bank and phase")


@dataclass(frozen=True)
class BoundExecutionLayer:
    layer_id: int
    bank_id: int
    execution_start: int
    bindings: object


def whole_layer_bank_layout(local_experts, hot_slots, execution_slots):
    """Return two disjoint (pool, offset, count) views, without allocating.

    Execution must hold one full layer for decode. Prefer borrowing Hot for B;
    with a small/empty Hot pool require an explicitly budgeted second layer in
    Execution. Never silently shrink Hot or fall back to partitioned compute.
    """
    if local_experts <= 0 or execution_slots < local_experts:
        raise ValueError(
            f"Layer execution requires execution_slots >= {local_experts}; "
            f"got {execution_slots}. Hot cache capacity is not reduced automatically."
        )
    a = ("execution", 0, local_experts)
    if hot_slots >= local_experts:
        return a, ("hot", 0, local_experts)
    if execution_slots >= 2 * local_experts:
        return a, ("execution", local_experts, local_experts)
    raise ValueError(
        f"Layer prefetch needs a second bank of {local_experts} experts: "
        f"hot_slots={hot_slots}, execution_slots={execution_slots}. "
        "Provide sufficient capacity explicitly."
    )
