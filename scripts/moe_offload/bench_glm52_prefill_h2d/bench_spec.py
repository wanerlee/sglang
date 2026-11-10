"""Metadata-only GLM prefill copy budgets and common-clock result aggregation.

This module deliberately does not import torch, SGLang, or an NPU runtime.
Component sizes are ND/NZ-aligned estimates; the worker must use the production
format probe to obtain and verify actual device allocation extents.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

GIB = 1024**3


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _align(value, alignment=64):
    return (value + alignment - 1) // alignment * alignment


def build_spec(
    model_config: Path | str,
    pp_partition: str,
    pp_rank: int,
    ep_size: int,
    source_layers: int | None = None,
    host_pool_gib: int = 56,
):
    """Read only config.json and describe one PP stage's routed weight copies.

    Source layers default to all routed layers in the selected stage. A smaller
    value explicitly models repeated reads from a reduced set of host masters.
    ``host_slab_bytes`` excludes OS-page registration rounding, which the actual
    production MemFabricHostPool applies on the machine executing the test.
    """
    path = Path(model_config).expanduser()
    if path.is_dir():
        path /= "config.json"
    path = path.resolve()
    cfg = json.loads(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Model config must be a JSON object")
    if "text_config" in cfg:
        text_config = cfg["text_config"]
        if not isinstance(text_config, dict):
            raise ValueError("text_config must be an object")
        cfg = {**cfg, **text_config}
    if cfg.get("model_type") != "glm_moe_dsa" or (
        "GlmMoeDsaForCausalLM" not in (cfg.get("architectures") or [])
    ):
        raise ValueError("Expected GLM-5.2 glm_moe_dsa / GlmMoeDsaForCausalLM config")
    hidden = _positive_int(cfg.get("hidden_size"), "hidden_size")
    intermediate = _positive_int(
        cfg.get("moe_intermediate_size"), "moe_intermediate_size"
    )
    if hidden % 64 or intermediate % 64:
        raise ValueError("W8A8 hidden/intermediate dimensions must be multiples of 64")
    experts = _positive_int(cfg.get("n_routed_experts"), "n_routed_experts")
    layers = _positive_int(cfg.get("num_hidden_layers"), "num_hidden_layers")
    first = _nonnegative_int(cfg.get("first_k_dense_replace"), "first_k_dense_replace")
    frequency = _positive_int(cfg.get("moe_layer_freq"), "moe_layer_freq")
    _positive_int(ep_size, "ep_size")
    _positive_int(host_pool_gib, "host_pool_gib")
    if experts % ep_size:
        raise ValueError("n_routed_experts must divide evenly across EP ranks")
    if not isinstance(pp_partition, str):
        raise ValueError("pp_partition must be a comma-separated list")
    try:
        parts = [int(part.strip()) for part in pp_partition.split(",")]
    except ValueError as exc:
        raise ValueError("pp_partition must contain positive integer lengths") from exc
    if not parts or any(part <= 0 for part in parts) or sum(parts) != layers:
        raise ValueError(f"PP partition must have positive lengths summing to {layers}")
    _nonnegative_int(pp_rank, "pp_rank")
    if pp_rank >= len(parts):
        raise ValueError("pp_rank is outside the PP partition")
    start = sum(parts[:pp_rank])
    stage_ids = list(range(start, start + parts[pp_rank]))
    offloaded = [layer for layer in stage_ids if layer >= first and layer % frequency == 0]
    if not offloaded:
        raise ValueError("Selected PP stage contains no routed MoE layers")
    if source_layers is None:
        source_layers = len(offloaded)
    _positive_int(source_layers, "source_layers")
    if source_layers > len(offloaded):
        raise ValueError("source_layers cannot exceed this stage's routed layer count")

    components = []
    for prefix, n, k in (("w13", 2 * intermediate, hidden), ("w2", hidden, intermediate)):
        for suffix, runtime_shape, dtype, nbytes, host_nbytes in (
            ("weight", [1, k, n], "int8", n * k, n * k),
            ("weight_scale", [1, n], "bfloat16", n * 2, n * 4),
            ("weight_offset", [1, n], "float32", n * 4, n * 4),
        ):
            components.append(
                {
                    "name": f"{prefix}_{suffix}",
                    "runtime_shape": runtime_shape,
                    "dtype": dtype,
                    "nbytes": nbytes,
                    "host_nbytes": _align(max(nbytes, host_nbytes)),
                }
            )
    local = experts // ep_size
    copy_bytes = sum(component["nbytes"] for component in components) * local
    host_bytes = sum(component["host_nbytes"] for component in components) * local
    slab_bytes = host_bytes * source_layers
    pool_bytes = host_pool_gib * GIB
    if slab_bytes > pool_bytes:
        required = math.ceil(slab_bytes / GIB)
        raise ValueError(
            f"Host sources need {slab_bytes / GIB:.6f} GiB/rank; "
            f"host_pool_gib must be at least {required} (or reduce source_layers)"
        )
    return {
        "model_config": str(path),
        "model_type": "glm_moe_dsa",
        "format_kind": "w8a8",
        "layout_is_estimate": True,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "total_experts": experts,
        "local_experts": local,
        "num_layers": layers,
        "first_k_dense_replace": first,
        "moe_layer_freq": frequency,
        "pp_partition": parts,
        "pp_rank": pp_rank,
        "ep_size": ep_size,
        "stage_layer_ids": stage_ids,
        "offloaded_layer_ids": offloaded,
        "source_layer_ids": offloaded[:source_layers],
        "source_layer_count": source_layers,
        "source_reused": source_layers < len(offloaded),
        "components": components,
        "expected_bytes_per_layer": copy_bytes,
        "host_bytes_per_layer": host_bytes,
        "expected_stage_bytes": copy_bytes * len(offloaded),
        "host_slab_bytes": slab_bytes,
        "host_pool_bytes": pool_bytes,
        "two_bank_bytes": copy_bytes * 2,
    }


def _finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def summarize_samples(samples):
    """Aggregate rank samples using the common monotonic host clock.

    One row describes one device and iteration, with scenario, iteration,
    device, bytes, host_start_ns, host_submit_end_ns, host_end_ns, event_ms.
    Different scenarios may use different device sets. Within each scenario,
    every iteration must contain an identical nonempty set without duplicates.
    Device event durations only provide a secondary estimate; they cannot form
    a synchronized cross-device time window.
    """
    by_scenario = defaultdict(lambda: defaultdict(dict))
    for original in samples:
        row = dict(original)
        scenario = row.get("scenario")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a nonempty string")
        iteration = _nonnegative_int(row.get("iteration"), "iteration")
        device = _nonnegative_int(row.get("device"), "device")
        _positive_int(row.get("bytes"), "bytes")
        for field in ("host_start_ns", "host_submit_end_ns", "host_end_ns"):
            _nonnegative_int(row.get(field), field)
        if not row["host_start_ns"] <= row["host_submit_end_ns"] <= row["host_end_ns"]:
            raise ValueError("Host timestamps must be ordered start <= submit <= end")
        if row["host_start_ns"] == row["host_end_ns"]:
            raise ValueError("Host timing window must be positive")
        _finite_positive(row.get("event_ms"), "event_ms")
        bucket = by_scenario[scenario][iteration]
        if device in bucket:
            raise ValueError(f"Duplicate sample for {scenario}/{iteration}/device {device}")
        bucket[device] = row
    if not by_scenario:
        raise ValueError("At least one complete sample is required")

    summaries, ranks, iterations = [], [], []
    for scenario, batches in sorted(by_scenario.items()):
        expected_devices = set(next(iter(batches.values())))
        if any(set(batch) != expected_devices for batch in batches.values()):
            raise ValueError(f"Incomplete or inconsistent device sets in {scenario}")
        scenario_iterations = []
        for iteration, batch in sorted(batches.items()):
            rows = list(batch.values())
            starts = [row["host_start_ns"] for row in rows]
            ends = [row["host_end_ns"] for row in rows]
            total = sum(row["bytes"] for row in rows)
            window_ms = (max(ends) - min(starts)) / 1e6
            record = {
                "scenario": scenario,
                "iteration": iteration,
                "devices": sorted(expected_devices),
                "total_bytes": total,
                "window_ms": window_ms,
                "total_gbps": total / (window_ms * 1e6),
                "start_skew_ms": (max(starts) - min(starts)) / 1e6,
                "event_estimate_gbps": total / (max(row["event_ms"] for row in rows) * 1e6),
            }
            iterations.append(record)
            scenario_iterations.append(record)
        windows = [row["window_ms"] for row in scenario_iterations]
        totals = {row["total_bytes"] for row in scenario_iterations}
        summaries.append(
            {
                "scenario": scenario,
                "devices": sorted(expected_devices),
                "num_samples": len(scenario_iterations),
                "total_bytes_per_iteration": next(iter(totals)) if len(totals) == 1 else None,
                "median_window_ms": median(windows),
                "median_total_gbps": median(row["total_gbps"] for row in scenario_iterations),
                "window_cv_pct": pstdev(windows) / mean(windows) * 100,
                "max_start_skew_ms": max(row["start_skew_ms"] for row in scenario_iterations),
                "median_event_estimate_gbps": median(row["event_estimate_gbps"] for row in scenario_iterations),
            }
        )
        for device in sorted(expected_devices):
            rows = [batch[device] for batch in batches.values()]
            host_ms = [(row["host_end_ns"] - row["host_start_ns"]) / 1e6 for row in rows]
            ranks.append(
                {
                    "scenario": scenario,
                    "device": device,
                    "num_samples": len(rows),
                    "median_host_ms": median(host_ms),
                    "median_submit_ms": median((row["host_submit_end_ns"] - row["host_start_ns"]) / 1e6 for row in rows),
                    "median_wait_ms": median((row["host_end_ns"] - row["host_submit_end_ns"]) / 1e6 for row in rows),
                    "median_event_ms": median(row["event_ms"] for row in rows),
                    "median_host_gbps": median(row["bytes"] / (duration * 1e6) for row, duration in zip(rows, host_ms)),
                    "median_event_gbps": median(row["bytes"] / (row["event_ms"] * 1e6) for row in rows),
                }
            )
    return {"summary": summaries, "ranks": ranks, "iterations": iterations}
