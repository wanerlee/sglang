"""Logical EP identity is independent of physical cache and execution slots."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Optional, Tuple


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
    # External allocators may reserve page-rounded storage. It is still charged.
    host_allocation_limit_bytes: Optional[int] = None

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
        if (
            self.host_allocation_limit_bytes is not None
            and self.host_allocation_limit_bytes < 0
        ):
            raise ValueError("Host allocation limit must be nonnegative")

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
    _offloaded: tuple = field(init=False, repr=False, compare=False)
    _by_layer: object = field(init=False, repr=False, compare=False)
    _offsets: object = field(init=False, repr=False, compare=False)
    _num_experts: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if len({layer.layer_id for layer in self.layers}) != len(self.layers):
            raise ValueError("Duplicate expert layer")
        if len({(layer.ep_size, layer.ep_rank) for layer in self.layers}) > 1:
            raise ValueError("A catalog belongs to exactly one EP rank")
        object.__setattr__(
            self, "layers", tuple(sorted(self.layers, key=lambda x: x.layer_id))
        )
        offloaded = tuple(layer for layer in self.layers if not layer.resident)
        offsets, count = {}, 0
        for layer in offloaded:
            offsets[layer.layer_id] = count
            count += layer.local_experts
        object.__setattr__(self, "_offloaded", offloaded)
        object.__setattr__(
            self, "_by_layer", MappingProxyType({s.layer_id: s for s in self.layers})
        )
        object.__setattr__(self, "_offsets", MappingProxyType(offsets))
        object.__setattr__(self, "_num_experts", count)

    @property
    def offloaded_layers(self):
        return self._offloaded

    @property
    def num_experts(self):
        return self._num_experts

    def layer(self, layer_id):
        try:
            return self._by_layer[layer_id]
        except KeyError:
            raise ValueError("Unregistered expert layer") from None

    def key(self, layer_id, local_id):
        if layer_id not in self._offsets:
            raise ValueError("Layer is resident or unregistered")
        if not 0 <= local_id < self._by_layer[layer_id].local_experts:
            raise ValueError("Invalid rank-local expert ID")
        return self._offsets[layer_id] + local_id
