"""Validate and time decode wave writeback on one Ascend NPU.

Run from the repository root, in the same environment as the NPU server:
  PYTHONPATH=python python test/manual/validate_moe_decode_writeback.py \
      --rows 256 --capacity 4096 --hidden 7168 \
      --profile-dir /tmp/moe-writeback-profile

Use the actual receive rows for --rows and the layout's max_dispatch_rows for
--capacity. They can differ substantially; IndexCopy sees the whole [2M,H]
destination even when it writes only R rows. --capacity defaults to --rows.

The new path calls the production NPUDispatchLayout. The baseline reproduces
the previous per-wave output MaskedFill -> IndexSelect -> Where sequence.
Correctness covers NaN padding, empty/full waves, changing device offsets/counts,
and errors, in eager execution and after graph capture. Timing includes only
per-wave output cleanup/writeback: no FFN, EP, setup/copies, or final layer-wide
error masking. It is not a decode TPOT benchmark. No CPU fallback is provided.
"""

import argparse
import json
import statistics
from types import SimpleNamespace

import torch


class WritebackFixture:
    def __init__(self, rows, hidden, dtype, device, capacity=None):
        from sglang.srt.offloading.moe.config import Phase
        from sglang.srt.offloading.moe.npu.experts import (
            NPUDispatchLayout,
            NPUWaveInput,
        )

        if "defer_output_mask" not in NPUWaveInput.__dataclass_fields__:
            raise RuntimeError(
                "Update the checkout to the optimized writeback implementation"
            )

        self.rows, self.hidden, self.dtype = rows, hidden, dtype
        self.capacity = rows if capacity is None else capacity
        if self.capacity < rows:
            raise ValueError("Layout capacity must be at least the receive row count")
        self.error = torch.zeros((), dtype=torch.int32, device=device)
        # The old runner reuses healthy from its pre-FFN counts gate. Model that
        # shared prefix outside this output-only measurement region.
        self.baseline_healthy = torch.ones((), dtype=torch.bool, device=device)
        self.offset = torch.zeros((), dtype=torch.int64, device=device)
        self.count = torch.zeros((), dtype=torch.int64, device=device)
        self.row_ids = torch.arange(rows, dtype=torch.int64, device=device)
        self.context = SimpleNamespace(
            compute_stream=torch.npu.current_stream(),
            phase=Phase.DECODE,
            capturing=False,
            retain=lambda value: None,
        )
        self.adapter = NPUDispatchLayout(1, hidden, self.capacity, device, self.error)
        raw = SimpleNamespace(
            format="deepep_ll",
            hidden_states=torch.zeros((rows, hidden), dtype=dtype, device=device),
            hidden_states_scale=None,
            masked_m=torch.tensor([rows], dtype=torch.int64, device=device),
        )
        self.layout = self.adapter.bind(SimpleNamespace(raw=raw), self.context)
        self.selection = SimpleNamespace(
            wave=0, row_count=self.count, offset=self.offset
        )
        self.source = torch.empty_like(raw.hidden_states)
        self.baseline_source = torch.empty_like(raw.hidden_states)
        self.baseline_output = torch.empty_like(raw.hidden_states)
        # Exactly representable small values make equality checks independent of
        # rounding. The original source is regenerated outside measured regions.
        self.cpu_values = (
            torch.arange(rows * hidden).reshape(rows, hidden).remainder(97).to(dtype)
        )

    def setup(self, count, offset, error=0, *, reset_output=True):
        if not 0 <= count <= self.rows or not 0 <= offset <= self.rows - count:
            raise ValueError("The requested wave does not fit the receive buffer")
        self.count.fill_(count)
        self.offset.fill_(offset)
        self.error.fill_(error)
        self.baseline_healthy.fill_(error == 0)
        source = self.cpu_values.clone()
        source[count:].fill_(float("nan"))
        self.source.copy_(source)
        self.baseline_source.copy_(source)
        if reset_output:
            self.layout.output.fill_(-17)
            self.baseline_output.fill_(-17)

    def baseline(self):
        # Previous NPUExpertRunner output cleanup. Match its per-wave arange.
        rows = torch.arange(self.rows, device=self.source.device)
        good_rows = (rows < self.count) & self.baseline_healthy
        self.baseline_source.masked_fill_(~good_rows.unsqueeze(1), 0)
        # Previous NPUDispatchLayout.write_wave.
        inverse = (self.row_ids - self.offset).clamp(0, self.rows - 1)
        values = torch.index_select(self.baseline_source, 0, inverse)
        belongs = (self.row_ids >= self.offset) & (
            self.row_ids < self.offset + self.count
        )
        torch.where(
            belongs.unsqueeze(1), values, self.baseline_output, out=self.baseline_output
        )

    def optimized(self):
        self.layout.selected = self.selection
        self.adapter.write_wave(self.layout, 0, self.source, self.context)

    def check(self, count, offset, error):
        # On failure the two intermediate outputs may differ: the old path
        # writes zeros to the current wave, while the new one skips that range.
        # Both must suppress *all* waves at the final layer error gate.
        self.baseline_output.masked_fill_(self.error != 0, 0)
        self.adapter.combine_input(self.layout, self.context)
        torch.npu.synchronize()
        expected = torch.full_like(self.cpu_values, -17)
        if error:
            expected.zero_()
        else:
            expected[offset : offset + count].copy_(self.cpu_values[:count])
        actual = self.layout.output.cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, self.baseline_output.cpu(), rtol=0, atol=0)
        if not bool(torch.isfinite(actual).all()):
            raise AssertionError("NaN padding escaped into visible output")


def capture(fn, context):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    context.compute_stream = stream
    with torch.npu.stream(stream):
        for _ in range(3):
            fn()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    context.capturing = True
    try:
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            fn()
    finally:
        context.capturing = False
        context.compute_stream = torch.npu.current_stream()
    return graph


def cases(rows):
    partial = max(1, rows // 3)
    offset = rows - partial
    return [
        ("full", rows, 0, 0),
        ("partial_at_end", partial, offset, 0),
        ("empty_at_end", 0, rows, 0),
        ("empty_at_start", 0, 0, 0),
        ("one_row", 1, rows - 1, 0),
        ("latched_error", partial, offset, 128),
        ("recover_with_refreshed_state", rows, 0, 0),
    ]


def validate(fixture, baseline, optimized, mode):
    for name, count, offset, error in cases(fixture.rows):
        fixture.setup(count, offset, error)
        baseline()
        optimized()
        fixture.check(count, offset, error)
        print(json.dumps({"validation": mode, "case": name, "passed": True}))
    # A later failing wave must suppress an earlier successfully written wave.
    first = fixture.rows // 2
    fixture.setup(first, 0)
    baseline()
    optimized()
    fixture.setup(fixture.rows - first, first, 128, reset_output=False)
    baseline()
    optimized()
    fixture.check(fixture.rows - first, first, 128)
    print(json.dumps({"validation": mode, "case": "late_error", "passed": True}))


def time_graph(graph, iterations):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Layout max_dispatch_rows (M >= rows); defaults to rows.",
    )
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--profile-dir", default=None)
    args = parser.parse_args()
    if min(args.rows, args.hidden, args.iters, args.samples) <= 0 or args.warmup < 0:
        parser.error("rows, hidden, iters and samples must be positive; warmup >= 0")
    if args.capacity is None:
        args.capacity = args.rows
    if args.capacity < args.rows:
        parser.error("capacity must be at least rows")
    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit(
            "This validation requires torch_npu and a real Ascend NPU"
        ) from exc
    if not torch.npu.is_available():
        raise SystemExit("No Ascend NPU is available; no CPU timing is substituted")
    torch.npu.set_device(args.device)
    fixture = WritebackFixture(
        args.rows,
        args.hidden,
        getattr(torch, args.dtype),
        args.device,
        capacity=args.capacity,
    )
    validate(fixture, fixture.baseline, fixture.optimized, "eager")
    fixture.setup(args.rows, 0)
    graphs = {
        "baseline_mask_gather_where": capture(fixture.baseline, fixture.context),
        "optimized_production_writeback": capture(fixture.optimized, fixture.context),
    }
    validate(
        fixture,
        graphs["baseline_mask_gather_where"].replay,
        graphs["optimized_production_writeback"].replay,
        "graph_replay_dynamic_metadata",
    )
    results = []
    for name, count, offset, error in cases(args.rows):
        if name not in ("full", "partial_at_end", "empty_at_end"):
            continue
        fixture.setup(count, offset, error)
        for graph in graphs.values():
            for _ in range(args.warmup):
                graph.replay()
        torch.npu.synchronize()
        timings = {variant: [] for variant in graphs}
        # Alternate measurement order to reduce first/last-run bias.
        for sample in range(args.samples):
            names = list(graphs)
            if sample % 2:
                names.reverse()
            for variant in names:
                timings[variant].append(time_graph(graphs[variant], args.iters))
        medians = {key: statistics.median(value) for key, value in timings.items()}
        result = {
            "case": name,
            "rows": args.rows,
            "capacity": args.capacity,
            "hidden": args.hidden,
            "dtype": args.dtype,
            "count": count,
            "offset": offset,
            "median_us_per_wave_writeback": medians,
            "samples_us_per_wave_writeback": timings,
            "speedup": medians["baseline_mask_gather_where"]
            / medians["optimized_production_writeback"],
        }
        results.append(result)
        print(json.dumps(result), flush=True)

        if args.profile_dir:
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=True,
                with_stack=False,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    f"{args.profile_dir}/{name}"
                ),
            ) as profiler:
                for variant, graph in graphs.items():
                    with torch.autograd.profiler.record_function(variant):
                        graph.replay()
                        torch.npu.synchronize()
                    profiler.step()
    print(
        json.dumps(
            {
                "passed": True,
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "device": torch.npu.get_device_name(),
                "scope": "captured per-wave output cleanup/writeback only; not TPOT",
                "results": results,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
