"""Periodic LFU over one rank's shared, cross-layer hot pool.

Online expert decisions stay on the tensor device; no routing/popularity tensor
is read by the host. A host batch clock selects full decode graph variants only.
Accelerator bindings must validate capture support and budget the sorting
temporaries/workspace.
"""

import json
import math
from dataclasses import dataclass
from itertools import islice, zip_longest
from typing import Optional

from .config import ExpertCatalog
from .interfaces import CachePolicy


@dataclass(frozen=True)
class PeriodicLFUConfig:
    offline_stats_path: Optional[str] = None
    update_interval: int = 4
    min_usage: float = 2.0
    replacement_min_gain: float = 1.5
    max_swaps: int = 96
    usage_decay: float = 0.7

    def __post_init__(self):
        if self.offline_stats_path is not None and (
            not isinstance(self.offline_stats_path, str) or not self.offline_stats_path
        ):
            raise ValueError("offline_stats_path must be a nonempty path")
        if type(self.update_interval) is not int or self.update_interval <= 0:
            raise ValueError("update_interval must be a positive integer")
        if type(self.max_swaps) is not int or self.max_swaps < 0:
            raise ValueError("max_swaps must be a nonnegative integer")
        for name in ("min_usage", "replacement_min_gain"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.usage_decay) or not 0 <= self.usage_decay <= 1:
            raise ValueError("usage_decay must be finite and in [0, 1]")


@dataclass
class PeriodicLFUState:
    usage: object
    step_mod: object
    update_due: object


class PeriodicLFUPolicy(CachePolicy):
    def __init__(self, config: PeriodicLFUConfig, catalog: ExpertCatalog):
        self.config = config
        self.catalog = catalog

    @property
    def graph_maintenance_interval(self):
        return self.config.update_interval

    def initial_keys(self, capacity: int, model_id=None) -> tuple[int, ...]:
        """Initialize from offline Top-K, or spread slots across layers without it.

        Frequencies use global expert IDs. Resident layers and foreign EP experts
        are excluded by the catalog. Online usage is not seeded with this data.
        """
        if type(capacity) is not int or not 0 <= capacity <= self.catalog.num_experts:
            raise ValueError("Hot capacity must be within the eligible expert count")
        if capacity == 0:
            return ()
        if self.config.offline_stats_path is None:
            ranges, offset = [], 0
            for layer in self.catalog.offloaded_layers:
                ranges.append(range(offset, offset + layer.local_experts))
                offset += layer.local_experts
            keys = (
                key for row in zip_longest(*ranges) for key in row if key is not None
            )
            return tuple(sorted(islice(keys, capacity)))
        with open(self.config.offline_stats_path, encoding="utf-8") as stream:
            profile = json.load(stream)
        if not isinstance(profile, dict):
            raise ValueError("Offline profile must contain a layers object")
        if model_id is not None and profile.get("model_id") != model_id:
            raise ValueError("Offline profile model_id does not match the model")
        layers = profile.get("layers")
        if not isinstance(layers, dict):
            raise ValueError("Offline profile must contain a layers object")
        counts = []
        for layer in self.catalog.offloaded_layers:
            frequencies = layers.get(str(layer.layer_id), {})
            if not isinstance(frequencies, dict):
                raise ValueError("Layer frequencies must be an object")
            for local_id in range(layer.local_experts):
                frequency = frequencies.get(str(layer.global_start + local_id), 0)
                if (
                    type(frequency) not in (int, float)
                    or not math.isfinite(frequency)
                    or frequency < 0
                ):
                    raise ValueError(
                        "Offline frequencies must be finite and nonnegative"
                    )
                counts.append(frequency)
        return tuple(
            sorted(range(len(counts)), key=lambda key: (-counts[key], key))[:capacity]
        )

    def allocate_state(self, num_experts, device):
        import torch

        if num_experts != self.catalog.num_experts:
            raise ValueError("State size must match the expert catalog")
        return PeriodicLFUState(
            usage=torch.zeros(num_experts, dtype=torch.float32, device=device),
            step_mod=torch.zeros((), dtype=torch.int64, device=device),
            update_due=torch.zeros((), dtype=torch.bool, device=device),
        )

    def observe(self, state, step_counts, update_enabled, due_out):
        import torch

        next_step = (state.step_mod + 1).remainder(self.config.update_interval)
        state.usage.add_(torch.where(update_enabled, step_counts, 0))
        state.step_mod.copy_(torch.where(update_enabled, next_step, state.step_mod))
        state.update_due.copy_(update_enabled & (next_step == 0))
        # The public transaction gate must survive finish_step clearing the
        # policy's private flag; the caller owns this separate fixed address.
        due_out.copy_(state.update_due)

    def select_hot_set(self, state, key_to_slot, target_keys):
        import torch

        count = self.catalog.num_experts
        capacity = target_keys.numel()
        if capacity > count:
            raise ValueError("Hot capacity exceeds the eligible expert count")
        if state.usage.numel() != count or key_to_slot.numel() != count:
            raise ValueError("Policy tensors must match the expert catalog")
        # These branches depend only on static capacity, never on device values.
        if capacity == 0:
            return
        resident = key_to_slot >= 0
        keys = torch.arange(count, device=state.usage.device, dtype=torch.int64)
        usage = state.usage
        # Canonical keys span every offloaded layer owned by this EP rank.
        # A hot candidate may replace a colder expert from ANY of those layers.
        # Stable sorting breaks equal-usage ties by (layer_id, local_expert_id).
        candidate = torch.argsort(
            torch.where(~resident, usage, -torch.inf), descending=True, stable=True
        )
        victim = torch.argsort(torch.where(resident, usage, torch.inf), stable=True)
        # Descending cold usage minus ascending hot usage is nonincreasing;
        # valid pairs form a prefix, so no second gain-ranking sort is needed.
        candidate = candidate[: self.config.max_swaps]
        victim = victim[: self.config.max_swaps]
        gain = usage[candidate] - usage[victim]
        valid = (
            ~resident[candidate]
            & resident[victim]
            & (usage[candidate] >= self.config.min_usage)
            & (gain >= self.config.replacement_min_gain)
        )
        swaps = (valid & state.update_due).to(torch.int64)
        membership = resident.to(torch.int64)
        membership.scatter_add_(0, victim, -swaps)
        membership.scatter_add_(0, candidate, swaps)

        # Canonical set representation; output position is NOT a physical slot.
        selected = torch.sort(torch.where(membership > 0, keys, count)).values[
            :capacity
        ]
        target_keys.copy_(torch.where(selected < count, selected, -1))

    def finish_step(self, state):
        import torch

        # Separate from pure selection. Call once after select_hot_set, even on
        # empty-update rounds. Clearing due makes repeated finish calls harmless.
        state.usage.copy_(
            torch.where(
                state.update_due, state.usage * self.config.usage_decay, state.usage
            )
        )
        state.update_due.zero_()

    def copy_state(self, source, destination):
        destination.usage.copy_(source.usage)
        destination.step_mod.copy_(source.step_mod)
        destination.update_due.copy_(source.update_due)
