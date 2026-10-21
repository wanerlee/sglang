"""Logical EP identity is independent of physical cache and execution slots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Phase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class MoEOffloadConfig:
    cache_ratio: float
    execution_slots: int
    ep_size: int
    ep_rank: int
    tp_size: int = 1
    moe_tp_size: int = 1
    workspace_limit_bytes: int = 0
    metadata_limit_bytes: int = 0

    def __post_init__(self):
        if not math.isfinite(self.cache_ratio) or not 0 <= self.cache_ratio <= 1:
            raise ValueError("cache_ratio must be finite and in [0, 1]")
        if self.execution_slots <= 0:
            raise ValueError("execution_slots must be positive and explicitly budgeted")
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("Invalid EP topology")
        if self.tp_size <= 0 or (self.tp_size > 1 and self.ep_size == 1):
            raise ValueError("Multi-rank TP requires EP for MoE offload")
        if self.moe_tp_size != 1:
            raise ValueError("MoE TP is not supported")
        if min(self.workspace_limit_bytes, self.metadata_limit_bytes) < 0:
            raise ValueError("Memory limits must be nonnegative")

    def hot_slots(self, catalog: ExpertCatalog) -> int:
        return math.floor(self.cache_ratio * catalog.num_experts)


@dataclass(frozen=True)
class ExpertLayerSpec:
    layer_id: int
    global_experts: int
    ep_size: int
    ep_rank: int
    resident: bool = False

    def __post_init__(self):
        if self.layer_id < 0 or self.ep_size <= 0:
            raise ValueError("Invalid layer or EP size")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("Invalid EP rank")
        if self.global_experts <= 0 or self.global_experts % self.ep_size:
            raise ValueError("Routed experts must divide evenly across EP ranks")

    @property
    def local_experts(self):
        return self.global_experts // self.ep_size

    @property
    def global_start(self):
        return self.ep_rank * self.local_experts

    def local_id(self, global_id):
        local_id = global_id - self.global_start
        if not 0 <= local_id < self.local_experts:
            raise ValueError("Expert is not owned by this EP rank")
        return local_id


@dataclass(frozen=True)
class ExpertCatalog:
    layers: Tuple[ExpertLayerSpec, ...]

    def __post_init__(self):
        if len({layer.layer_id for layer in self.layers}) != len(self.layers):
            raise ValueError("Duplicate expert layer")
        if len({(layer.ep_size, layer.ep_rank) for layer in self.layers}) > 1:
            raise ValueError("A catalog belongs to exactly one EP rank")
        object.__setattr__(
            self, "layers", tuple(sorted(self.layers, key=lambda x: x.layer_id))
        )

    @property
    def offloaded_layers(self):
        return tuple(layer for layer in self.layers if not layer.resident)

    @property
    def num_experts(self):
        return sum(layer.local_experts for layer in self.offloaded_layers)

    def key(self, layer_id, local_id):
        offset = 0
        for layer in self.offloaded_layers:
            if layer.layer_id == layer_id:
                if not 0 <= local_id < layer.local_experts:
                    raise ValueError("Invalid rank-local expert ID")
                return offset + local_id
            offset += layer.local_experts
        raise ValueError("Layer is resident or unregistered")


@dataclass(frozen=True)
class ExpertWave:
    index: int
    local_start: int
    local_end: int
    bank: int


def fixed_waves(local_experts, bank_slots, bank_count=1):
    """Static partition of all local experts, including empty execution waves.

    Communication keeps original global expert IDs. Bank lanes are a local F
    concern. Never renormalize top-k weights when masking routes for a wave.
    Repeated DFC uses the same collective count/order on every EP rank; a
    dispatch-once program uses the waves only for local F between its D and C.
    """
    if min(local_experts, bank_slots, bank_count) <= 0:
        raise ValueError("Expert and bank capacities must be positive")
    return tuple(
        ExpertWave(i, start, min(start + bank_slots, local_experts), i % bank_count)
        for i, start in enumerate(range(0, local_experts, bank_slots))
    )
