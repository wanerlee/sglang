"""CPU reference contracts, not a DeepEP implementation or graph validation."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe.config import ExpertWave, fixed_waves
from sglang.srt.offloading.moe.interfaces import (
    DispatchLayoutAdapter,
    LocalExpertProgram,
    MoEDispatcher,
)
from sglang.srt.offloading.moe.programs import DispatchOnceProgram
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class ReferenceDispatcher(MoEDispatcher):
    """Synthetic receive layout with optional padding and activation scales."""

    def __init__(self, calls, padding=0):
        self.calls = calls
        self.padding = padding
        self.global_start = 8
        self.local_experts = 5

    def dispatch(self, hidden_states, routing, context):
        self.calls.append("D")
        rows = []
        for expert in range(self.local_experts):
            for token, choices in enumerate(routing.ids.tolist()):
                for choice, global_id in enumerate(choices):
                    if global_id == expert + self.global_start:
                        rows.append((expert, token, choice))
            rows.extend([(-1, -1, -1)] * self.padding)
        x = hidden_states.new_full((len(rows), hidden_states.shape[1]), float("nan"))
        scales = hidden_states.new_ones((len(rows), 1))
        for row, (expert, token, choice) in enumerate(rows):
            if expert >= 0:
                scales[row] = token + 2
                x[row] = hidden_states[token] / scales[row]
        self.dispatched = SimpleNamespace(
            x=x,
            scales=scales,
            experts=torch.tensor([row[0] for row in rows], dtype=torch.int64),
            rows=rows,
            routing=routing,
            handle=object(),
            original=hidden_states,
        )
        return self.dispatched

    def combine(self, local_output, context):
        self.calls.append("C")
        assert local_output.dispatch is self.dispatched
        assert local_output.handle is self.dispatched.handle
        output = torch.zeros_like(self.dispatched.original)
        for row, (expert, token, choice) in enumerate(self.dispatched.rows):
            if expert < 0:
                assert torch.equal(
                    local_output.x[row], torch.zeros_like(local_output.x[row])
                )
            else:
                weight = self.dispatched.routing.weights[token, choice]
                output[token] += local_output.x[row] * weight
        return output

    def finish(self, output, context):
        self.calls.append("tail")
        return output + 0.125


class ReferenceLayout(DispatchLayoutAdapter):
    """CPU-only packing; production adapters must use budgeted device buffers."""

    def bind(self, dispatch_output, context):
        return SimpleNamespace(
            dispatch=dispatch_output, output=torch.zeros_like(dispatch_output.x)
        )

    def select_wave(self, layout, wave, context):
        dispatched = layout.dispatch
        selected = (dispatched.experts >= wave.local_start) & (
            dispatched.experts < wave.local_end
        )
        # Deliberately change row order: concatenating wave outputs is incorrect.
        layout.rows = torch.nonzero(selected).flatten().flip(0)
        lanes = dispatched.experts[layout.rows] - wave.local_start
        return SimpleNamespace(
            x=dispatched.x[layout.rows],
            scales=dispatched.scales[layout.rows],
            lanes=lanes,
            counts=torch.bincount(lanes, minlength=wave.local_end - wave.local_start),
        )

    def write_wave(self, layout, wave, local_output, context):
        layout.output[layout.rows] = local_output

    def combine_input(self, layout, context):
        return SimpleNamespace(
            x=layout.output, dispatch=layout.dispatch, handle=layout.dispatch.handle
        )


class ReferenceFFN(LocalExpertProgram):
    def __init__(self, calls, w1, w2):
        self.calls, self.w1, self.w2 = calls, w1, w2
        # Reused scratch forces each write_wave to preserve the preceding result.
        self.scratch = torch.empty(32, w2.shape[-1], dtype=w2.dtype)
        self.fail_wave = None

    def run(self, layer, dispatch_output, wave, context):
        self.calls.append(("F", wave.index))
        if wave.index == self.fail_wave:
            raise RuntimeError("FFN failed")
        out = self.scratch[: dispatch_output.x.shape[0]]
        for lane in range(wave.local_end - wave.local_start):
            mask = dispatch_output.lanes == lane
            assert mask.sum() == dispatch_output.counts[lane]
            expert = wave.local_start + lane
            x = dispatch_output.x[mask] * dispatch_output.scales[mask]
            out[mask] = torch.relu(x @ self.w1[expert]) @ self.w2[expert]
        return out


class TestDispatchOnce(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(23)
        self.hidden = torch.randn(3, 4, generator=generator, dtype=torch.float64)
        self.w1 = torch.randn(5, 4, 7, generator=generator, dtype=torch.float64)
        self.w2 = torch.randn(5, 7, 4, generator=generator, dtype=torch.float64)
        self.routing = SimpleNamespace(
            # Experts 2/3 are idle; global 1 is foreign and -1 is padding.
            ids=torch.tensor([[8, 12, 1], [9, -1, 12], [8, 9, 12]]),
            weights=torch.tensor(
                [[0.2, 0.3, 0.5], [0.7, 0, 0.3], [0.1, 0.6, 0.3]],
                dtype=torch.float64,
            ),
        )

    def make_program(self, bank_slots=2, padding=0):
        calls = []
        dispatcher = ReferenceDispatcher(calls, padding)
        local = ReferenceFFN(calls, self.w1, self.w2)
        program = DispatchOnceProgram(
            fixed_waves(5, bank_slots, bank_count=2),
            dispatcher,
            local,
            ReferenceLayout(),
            local_experts=5,
        )
        return program, calls

    def reference(self, hidden, routing):
        output = torch.zeros_like(hidden)
        for token, choices in enumerate(routing.ids.tolist()):
            for choice, global_id in enumerate(choices):
                expert = global_id - 8
                if 0 <= expert < 5:
                    result = (
                        torch.relu(hidden[token] @ self.w1[expert]) @ self.w2[expert]
                    )
                    output[token] += result * routing.weights[token, choice]
        return output + 0.125

    def check_run(self, program, calls, hidden, routing):
        calls.clear()
        saved_ids, saved_weights = routing.ids.clone(), routing.weights.clone()
        output = program.run(None, hidden, routing, None)
        torch.testing.assert_close(output, self.reference(hidden, routing))
        torch.testing.assert_close(routing.ids, saved_ids)
        torch.testing.assert_close(routing.weights, saved_weights)
        self.assertIs(program.dispatcher.dispatched.routing, routing)
        self.assertEqual(
            calls,
            ["D"] + [("F", w.index) for w in program.waves] + ["C", "tail"],
        )

    def test_matches_unsliced_ffn_with_tail_empty_waves_and_padding(self):
        for bank_slots in (1, 2, 5, 8):
            for padding in (0, 1):
                with self.subTest(bank_slots=bank_slots, padding=padding):
                    program, calls = self.make_program(bank_slots, padding)
                    self.check_run(program, calls, self.hidden, self.routing)

    def test_empty_rank_still_dispatches_computes_empty_waves_and_combines(self):
        routing = SimpleNamespace(
            ids=torch.zeros_like(self.routing.ids), weights=self.routing.weights
        )
        program, calls = self.make_program(padding=1)
        self.check_run(program, calls, self.hidden, routing)

    def test_zero_tokens(self):
        routing = SimpleNamespace(
            ids=self.routing.ids[:0], weights=self.routing.weights[:0]
        )
        program, calls = self.make_program()
        self.check_run(program, calls, self.hidden[:0], routing)

    def test_repeated_invocations_use_fresh_routes_and_handle(self):
        program, calls = self.make_program(padding=1)
        self.check_run(program, calls, self.hidden, self.routing)
        handle = program.dispatcher.dispatched.handle
        routing = SimpleNamespace(
            ids=torch.tensor([[10, 11, 12], [10, 8, 9], [12, 11, -1]]),
            weights=self.routing.weights.flip(1),
        )
        self.check_run(program, calls, self.hidden * 2, routing)
        self.assertIsNot(program.dispatcher.dispatched.handle, handle)

    def test_invalid_partitions_rejected_before_any_collective(self):
        wave = ExpertWave
        invalid = (
            (),
            (wave(0, 0, 4, 0),),  # incomplete
            (wave(0, 0, 6, 0),),  # overrun
            (wave(0, 1, 5, 0),),  # missing prefix
            (wave(0, 0, 2, 0), wave(1, 3, 5, 1)),  # gap
            (wave(0, 0, 3, 0), wave(1, 2, 5, 1)),  # overlap
            (wave(0, 0, 0, 0), wave(1, 0, 5, 1)),  # empty interval
            (wave(1, 0, 5, 0),),  # invalid index
            (wave(0, 0, 5, -1),),  # invalid bank
        )
        for waves in invalid:
            with self.subTest(waves=waves), self.assertRaises(ValueError):
                DispatchOnceProgram(waves, None, None, None, local_experts=5)
        with self.assertRaises(ValueError):
            DispatchOnceProgram(fixed_waves(5, 2), None, None, None, local_experts=0)

    def test_failure_does_not_combine_incomplete_output(self):
        program, calls = self.make_program()
        program.local_program.fail_wave = 1
        with self.assertRaisesRegex(RuntimeError, "FFN failed"):
            program.run(None, self.hidden, self.routing, None)
        self.assertEqual(calls, ["D", ("F", 0), ("F", 1)])


if __name__ == "__main__":
    unittest.main()
