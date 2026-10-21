"""Periodic, layer-local LFU with rank-wide admission limits.

Online decisions stay on the tensor device. Python loops only over static layer
ranges; no routing/popularity tensor is read by the host. Accelerator bindings
must validate capture support and budget the sorting temporaries/workspace.
"""

import json
import math
from dataclasses import dataclass

from .config import ExpertCatalog
from .interfaces import CachePolicy


@dataclass(frozen=True)
class PeriodicLFUConfig:
    offline_stats_path: str
    update_interval: int = 4
    min_usage: float = 2.0
    replacement_min_gain: float = 1.5
    max_swaps: int = 96
    usage_decay: float = 0.7

    def __post_init__(self):
        if not self.offline_stats_path:
            raise ValueError("offline_stats_path is required")
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
        offset = 0
        ranges = []
        for layer in catalog.offloaded_layers:
            ranges.append((offset, offset + layer.local_experts))
            offset += layer.local_experts
        # Catalog order is layer_id order, also the global pair tie-break order.
        self.layer_ranges = tuple(ranges)

    def initial_keys(self, capacity: int, model_id: str) -> tuple[int, ...]:
        """Host-only offline Top-K; backend loads/publishes these before serving.

        Frequencies use global expert IDs. Resident layers and foreign EP experts
        are excluded by the catalog. Online usage is not seeded with this data.
        """
        if type(capacity) is not int or not 0 <= capacity <= self.catalog.num_experts:
            raise ValueError("Hot capacity must be within the eligible expert count")
        with open(self.config.offline_stats_path, encoding="utf-8") as stream:
            profile = json.load(stream)
        if not isinstance(profile, dict) or profile.get("model_id") != model_id:
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

    def update_statistics(self, state, step_counts, update_enabled):
        import torch

        next_step = (state.step_mod + 1).remainder(self.config.update_interval)
        state.usage.add_(torch.where(update_enabled, step_counts, 0))
        state.step_mod.copy_(torch.where(update_enabled, next_step, state.step_mod))
        state.update_due.copy_(update_enabled & (next_step == 0))

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
        candidates, victims, gains, valid_pairs = [], [], [], []
        for start, end in self.layer_ranges:
            usage = state.usage[start:end]
            hot = resident[start:end]
            # Fixed-size permutations: invalid entries sort to the end. Their
            # pairs are masked out instead of constructing variable-size arrays.
            cold_order = torch.argsort(
                torch.where(~hot, usage, -torch.inf), descending=True, stable=True
            )
            hot_order = torch.argsort(torch.where(hot, usage, torch.inf), stable=True)
            gain = usage[cold_order] - usage[hot_order]
            valid = (
                ~hot[cold_order]
                & hot[hot_order]
                & (usage[cold_order] >= self.config.min_usage)
                & (gain >= self.config.replacement_min_gain)
            )
            candidates.append(cold_order + start)
            victims.append(hot_order + start)
            gains.append(gain)
            valid_pairs.append(valid)

        candidate = torch.cat(candidates)
        victim = torch.cat(victims)
        gain = torch.cat(gains)
        valid = torch.cat(valid_pairs)
        # Stable sorting preserves layer_id, candidate key, victim key on ties:
        # within each layer equal gains imply equal candidate/victim usages.
        order = torch.argsort(
            torch.where(valid, gain, -torch.inf), descending=True, stable=True
        )
        chosen = order[: self.config.max_swaps]
        swaps = (valid[chosen] & state.update_due).to(torch.int64)
        membership = resident.to(torch.int64)
        membership.scatter_add_(0, victim[chosen], -swaps)
        membership.scatter_add_(0, candidate[chosen], swaps)

        # Canonical set representation; output position is NOT a physical slot.
        keys = torch.arange(count, device=state.usage.device, dtype=torch.int64)
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
