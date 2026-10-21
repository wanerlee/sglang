"""Physical allocation accounting and host-side reservations.

Slot leases reserve a bank or region for a bound program, outside capture. Device
ResidencyOps must enforce the equivalent ready/read_done/generation protocol for
dynamic entries inside replay; these Python objects are not a graph cache policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple


class ResourceKind(Enum):
    HOST_WEIGHTS = "host_weights"
    HOT_WEIGHTS = "hot_weights"
    EXECUTION_WEIGHTS = "execution_weights"
    WORKSPACE = "workspace"
    METADATA = "metadata"


@dataclass(frozen=True)
class Allocation:
    name: str
    kind: ResourceKind
    nbytes: int
    storage_key: Any
    keepalive: Any


class AllocationLedger:
    """Account unique physical allocations and enforce byte budgets."""

    def __init__(self, limits, allocations):
        self.limits = dict(limits)
        self.allocations = tuple(allocations)
        self.usage = {kind: 0 for kind in ResourceKind}
        names, storages = set(), set()
        for item in self.allocations:
            if item.name in names or item.storage_key in storages:
                raise ValueError("Report physical allocations once; aliases are views")
            if item.nbytes < 0:
                raise ValueError("Allocation size cannot be negative")
            names.add(item.name)
            storages.add(item.storage_key)
            self.usage[item.kind] += item.nbytes
        for kind, used in self.usage.items():
            if used > self.limits.get(kind, 0):
                raise ValueError(f"{kind.value} budget exceeded: {used} bytes")


@dataclass(frozen=True)
class SlotLease:
    slots: Tuple[int, ...]
    generation: int
    manager_token: Any


class SlotLeaseManager:
    """Reserve logical slot regions until their device consumers finish.

    Physical tensors belong to the runtime; this manager tracks usage rights.
    """

    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("Slot capacity cannot be negative")
        self.capacity = capacity
        self._token = object()
        self._generation = 0
        self._live = {}
        self._retiring = {}

    def collect(self):
        for generation, ticket in tuple(self._retiring.items()):
            if ticket.completion.done():
                del self._retiring[generation]
                del self._live[generation]

    def acquire(self, slots):
        self.collect()
        slots = tuple(slots)
        if not slots or len(set(slots)) != len(slots):
            raise ValueError("A lease needs distinct slots")
        if any(slot < 0 or slot >= self.capacity for slot in slots):
            raise ValueError("Slot outside lease manager capacity")
        occupied = {slot for lease in self._live.values() for slot in lease.slots}
        if occupied.intersection(slots):
            raise RuntimeError("Slots are still reserved or have unfinished readers")
        self._generation += 1
        lease = SlotLease(slots, self._generation, self._token)
        self._live[lease.generation] = lease
        return lease

    def retire(self, lease, resources_idle):
        if (
            lease.manager_token is not self._token
            or self._live.get(lease.generation) is not lease
            or lease.generation in self._retiring
        ):
            raise ValueError("Foreign, stale or already retired lease")
        self._retiring[lease.generation] = resources_idle
        self.collect()

    def require_empty(self):
        self.collect()
        if self._live:
            raise RuntimeError("Slot lease manager still has live program leases")
