"""CPU contracts for decode writeback; NPU kernel timing needs hardware."""

import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.offloading.moe.config import ExpertWave, Phase
from sglang.srt.offloading.moe.npu import experts as native
from sglang.srt.offloading.moe.resources import AllocationLedger, ResourceKind
from sglang.srt.offloading.moe.runtime import ExecutionContext
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclass
class _QuantInfo:
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor


class _WritebackTrace(TorchDispatchMode):
    """Observe dispatched operations without substituting their implementation."""

    def __init__(self):
        super().__init__()
        self.matrix_ops = []
        self.index_copies = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        if name in {
            "aten::masked_fill_",
            "aten::index_select",
            "aten::where",
            "aten::index_copy_",
        } and any(isinstance(arg, torch.Tensor) and arg.ndim == 2 for arg in args):
            self.matrix_ops.append(name)
        if name == "aten::index_copy_":
            target, _, indices, source = args
            self.index_copies.append(
                (
                    indices.clone(),
                    target.shape[0],
                    target.untyped_storage().data_ptr()
                    == source.untyped_storage().data_ptr(),
                )
            )
        return func(*args, **(kwargs or {}))


class TestDecodeWriteback(unittest.TestCase):
    def setUp(self):
        scope = patch.object(native, "_compute_scope", lambda *args: nullcontext())
        scope.start()
        self.addCleanup(scope.stop)
        self.context = ExecutionContext(
            Phase.DECODE,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=lambda stream: TransferTicket(ImmediateCompletion()),
            capturing=True,
        )
        self.error = torch.zeros((), dtype=torch.int32)

    def bind(self, counts, source, *, capacity=8, adapter=None):
        if adapter is None:
            adapter = native.NPUDispatchLayout(
                len(counts), source.shape[1], capacity, "cpu", self.error
            )
        raw = SimpleNamespace(
            format=SimpleNamespace(value="deepep_ll"),
            hidden_states=source,
            hidden_states_scale=None,
            masked_m=torch.tensor(counts, dtype=torch.int64),
        )
        dispatched = native.NPUDispatch(
            raw, TransferTicket(ImmediateCompletion()), source.shape[1], object()
        )
        return adapter, adapter.bind(dispatched, self.context)

    def execute(self, selected, *, rows=(), valid=True, alias=None):
        """Use the real runner; the fake GMM only defines valid row results."""

        class Core:
            def run(self, inputs, quant, state):
                if alias is not None:
                    return SimpleNamespace(hidden_states=alias)
                output = torch.full_like(inputs.hidden_states, float("nan"))
                count = int(inputs.expert_tokens.sum())
                if count:
                    output[:count].copy_(torch.tensor(rows, dtype=output.dtype))
                return SimpleNamespace(hidden_states=output)

        width = selected.expert_tokens.numel()
        bank = native.NPUExpertBank(
            _QuantInfo(torch.ones(width, 1, 1), torch.ones(width, 1, 1)),
            torch.full((width,), valid, dtype=torch.bool),
            self.error,
        )
        runner = native.NPUExpertRunner(None, _core=Core(), _input_type=SimpleNamespace)
        result = runner(
            3,
            selected,
            SimpleNamespace(expert_bank=bank, error_status=self.error),
            self.context,
        )
        self.context.wait_for(result.read_done, self.context.compute_stream)
        return result.output

    def test_valid_rows_empty_wave_and_nonzero_offset_preserve_nan_isolation(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                adapter, layout = self.bind([2, 0, 1], torch.ones(5, 4, dtype=dtype))
                results = (
                    [[11, 12, 13, 14], [21, 22, 23, 24]],
                    [],
                    [[31, 32, 33, 34]],
                )
                trace = _WritebackTrace()
                for expert, rows in enumerate(results):
                    wave = ExpertWave(expert, expert, expert + 1, 0)
                    selected = adapter.select_wave(layout, wave, self.context)
                    self.assertTrue(selected.defer_output_mask)
                    output = self.execute(selected, rows=rows)
                    self.assertTrue(bool(torch.isnan(output[len(rows) :]).all()))
                    with trace:
                        adapter.write_wave(layout, wave, output, self.context)
                actual = adapter.combine_input(layout, self.context).output
                expected = torch.tensor(
                    [
                        [11, 12, 13, 14],
                        [21, 22, 23, 24],
                        [31, 32, 33, 34],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=dtype,
                )
                torch.testing.assert_close(actual, expected)
                self.assertEqual(trace.matrix_ops, ["aten::index_copy_"] * 3)
                for indices, target_rows, aliases in trace.index_copies:
                    self.assertEqual(indices.unique().numel(), 5)
                    self.assertTrue(
                        bool(((indices >= 0) & (indices < target_rows)).all())
                    )
                    self.assertFalse(aliases)
                # An empty wave must leave visible rows alone, even with all NaNs.
                self.assertTrue(bool((trace.index_copies[1][0] >= 8).all()))

    def test_one_wave_keeps_input_pack_but_removes_output_matrix_cleanup(self):
        adapter, layout = self.bind([1], torch.ones(4, 4, dtype=torch.bfloat16))
        wave = ExpertWave(0, 0, 1, 0)
        trace = _WritebackTrace()
        with trace:
            selected = adapter.select_wave(layout, wave, self.context)
            output = self.execute(selected, rows=[[2, 3, 4, 5]])
            adapter.write_wave(layout, wave, output, self.context)
        self.assertEqual(trace.matrix_ops.count("aten::index_select"), 1)
        self.assertEqual(trace.matrix_ops.count("aten::masked_fill_"), 1)
        self.assertNotIn("aten::where", trace.matrix_ops)
        self.assertEqual(trace.matrix_ops.count("aten::index_copy_"), 1)

    def test_late_bank_fault_suppresses_previous_valid_wave(self):
        adapter, layout = self.bind([1, 1], torch.ones(4, 4, dtype=torch.bfloat16))
        first, second = ExpertWave(0, 0, 1, 0), ExpertWave(1, 1, 2, 0)
        selected = adapter.select_wave(layout, first, self.context)
        output = self.execute(selected, rows=[[7, 8, 9, 10]])
        adapter.write_wave(layout, first, output, self.context)
        torch.testing.assert_close(
            layout.output[0], torch.tensor([7, 8, 9, 10], dtype=output.dtype)
        )
        selected = adapter.select_wave(layout, second, self.context)
        output = self.execute(selected, valid=False)
        self.assertTrue(bool(torch.isnan(output).all()))
        trace = _WritebackTrace()
        with trace:
            adapter.write_wave(layout, second, output, self.context)
        self.assertTrue(bool((trace.index_copies[0][0] >= 8).all()))
        self.assertEqual(int(self.error), 128)
        actual = adapter.combine_input(layout, self.context).output
        torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_invalid_dispatch_counts_cannot_publish_rows(self):
        for counts in ([-1, 1], [3, 3]):
            with self.subTest(counts=counts):
                self.error.zero_()
                adapter, layout = self.bind(
                    counts, torch.ones(4, 4, dtype=torch.bfloat16)
                )
                for i in range(2):
                    wave = ExpertWave(i, i, i + 1, 0)
                    selected = adapter.select_wave(layout, wave, self.context)
                    output = self.execute(selected)
                    adapter.write_wave(layout, wave, output, self.context)
                self.assertEqual(int(self.error), 64)
                actual = adapter.combine_input(layout, self.context).output
                torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_input_alias_is_copied_before_sink_overwrites_input(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                source = torch.tensor(
                    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [99] * 4],
                    dtype=dtype,
                )
                adapter, layout = self.bind([2, 1], source)
                trace = _WritebackTrace()
                for i in range(2):
                    wave = ExpertWave(i, i, i + 1, 0)
                    selected = adapter.select_wave(layout, wave, self.context)
                    output = self.execute(selected, alias=selected.hidden_states)
                    with trace:
                        adapter.write_wave(layout, wave, output, self.context)
                actual = adapter.combine_input(layout, self.context).output
                expected = torch.tensor(
                    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0] * 4],
                    dtype=dtype,
                )
                torch.testing.assert_close(actual, expected)
                self.assertTrue(
                    all(not aliases for _, _, aliases in trace.index_copies)
                )

    def test_visible_output_alias_preserves_nonzero_offset_source(self):
        adapter, layout = self.bind([2, 2], torch.ones(5, 4, dtype=torch.bfloat16))
        first, second = ExpertWave(0, 0, 1, 0), ExpertWave(1, 1, 2, 0)
        selected = adapter.select_wave(layout, first, self.context)
        output = self.execute(selected, rows=[[11] * 4, [22] * 4])
        adapter.write_wave(layout, first, output, self.context)
        selected = adapter.select_wave(layout, second, self.context)
        output = self.execute(selected, alias=layout.output)
        trace = _WritebackTrace()
        with trace:
            adapter.write_wave(layout, second, output, self.context)
        self.assertFalse(trace.index_copies[0][2])
        expected = torch.tensor(
            [[11] * 4, [22] * 4, [11] * 4, [22] * 4, [0] * 4], dtype=output.dtype
        )
        torch.testing.assert_close(
            adapter.combine_input(layout, self.context).output, expected
        )

    def test_next_input_pack_replaces_sink_nans_and_clears_padding(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                source = torch.tensor(
                    [[1] * 4, [2] * 4, [3] * 4, [float("nan")] * 4, [float("nan")] * 4],
                    dtype=dtype,
                )
                adapter, layout = self.bind([2, 1], source)
                first, second = ExpertWave(0, 0, 1, 0), ExpertWave(1, 1, 2, 0)
                selected = adapter.select_wave(layout, first, self.context)
                pointer = selected.hidden_states.data_ptr()
                output = self.execute(selected, rows=[[11] * 4, [12] * 4])
                adapter.write_wave(layout, first, output, self.context)
                selected = adapter.select_wave(layout, second, self.context)
                self.assertEqual(selected.hidden_states.data_ptr(), pointer)
                expected = torch.tensor(
                    [[3] * 4, [0] * 4, [0] * 4, [0] * 4, [0] * 4], dtype=dtype
                )
                torch.testing.assert_close(selected.hidden_states, expected)

    def test_zero_receive_shape_submits_no_writeback_kernel(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                adapter, layout = self.bind([0], torch.empty(0, 4, dtype=dtype))
                wave = ExpertWave(0, 0, 1, 0)
                selected = adapter.select_wave(layout, wave, self.context)
                output = self.execute(selected)
                trace = _WritebackTrace()
                with trace:
                    adapter.write_wave(layout, wave, output, self.context)
                self.assertEqual(trace.index_copies, [])
                self.assertIsNone(layout.selected)
                actual = adapter.combine_input(layout, self.context).output
                self.assertEqual(actual.shape, (0, 4))
                self.assertEqual(actual.dtype, dtype)

    def test_physical_allocation_budget_unchanged_across_rebinds(self):
        experts, hidden, capacity = 3, 65, 8
        adapter, layout = self.bind(
            [1, 0, 0], torch.ones(3, hidden, dtype=torch.bfloat16), capacity=capacity
        )
        # Previous workspace: input(2MH), output(2MH), scales, counts, offsets, rows.
        scale_width = max(4, ((hidden + 31) // 32) * 4)
        original_budget = (
            4 * capacity * hidden
            + capacity * scale_width
            + 8 * (experts + experts + 1 + capacity)
        )
        allocations = adapter.allocations()
        ledger = AllocationLedger(
            {ResourceKind.WORKSPACE: original_budget}, allocations
        )
        self.assertEqual(ledger.usage[ResourceKind.WORKSPACE], original_budget)
        keys = [allocation.storage_key for allocation in allocations]
        self.assertEqual(len(keys), len(set(keys)))
        for rows, dtype in ((7, torch.float16), (2, torch.bfloat16)):
            _, rebound = self.bind(
                [1, 0, 0], torch.ones(rows, hidden, dtype=dtype), adapter=adapter
            )
            self.assertEqual([a.storage_key for a in adapter.allocations()], keys)
            self.assertEqual(rebound.output.dtype, dtype)
            torch.testing.assert_close(rebound.output, torch.zeros_like(rebound.output))

    def test_wrong_output_dtype_is_rejected_before_reinterpreting_storage(self):
        adapter, layout = self.bind([1], torch.ones(3, 4, dtype=torch.bfloat16))
        wave = ExpertWave(0, 0, 1, 0)
        adapter.select_wave(layout, wave, self.context)
        with self.assertRaisesRegex(ValueError, "combine dtype"):
            adapter.write_wave(
                layout, wave, torch.ones(3, 4, dtype=torch.float16), self.context
            )


if __name__ == "__main__":
    unittest.main()
