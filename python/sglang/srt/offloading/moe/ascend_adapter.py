"""Ascend compact DeepEP receive rows -> wave-local native grouped matmul.

No communication, weight transfer or quantization kernel is reimplemented here.
The existing NPU DeepEP dispatcher joins receive/combine work to the calling
stream; AscendRunnerCore consumes the same weights/scales as the resident path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .config import ExpertWave, Phase
from .execution import LocalExecution
from .interfaces import DispatchLayoutAdapter, LocalRoutes, MoEDispatcher
from .resources import Allocation, ResourceKind

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
    )


@dataclass(frozen=True)
class ExpertBankBindings:
    """Native component tensors indexed by physical slot, plus bank offsets.

    Names match AscendQuantInfo (w13_weight, w2_weight and optional scales,
    offsets/biases). No repacking/conversion occurs here. Transfer implementations
    may subclass this record to retain their host addresses/registration handles.
    Prefill reservations can return different bank views with their own offsets.
    """

    components: dict[str, torch.Tensor]
    bank_offsets: tuple[int, ...]

    def wave_components(self, wave: ExpertWave) -> dict[str, torch.Tensor]:
        start = self.bank_offsets[wave.bank]
        end = start + wave.local_end - wave.local_start
        if any(end > tensor.shape[0] for tensor in self.components.values()):
            raise ValueError("Wave exceeds native weight bank")
        return {name: tensor[start:end] for name, tensor in self.components.items()}


class AscendDeepEPDispatcher(MoEDispatcher):
    def __init__(self, dispatcher):
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher

        # FusedMoE creates MaybeTboDeepEPDispatcher even when TBO is disabled.
        # Keep its public entry points/hooks, but require exactly one DeepEP
        # inner dispatcher (other implementations have different row layouts).
        inners = getattr(dispatcher, "_inners", (dispatcher,))
        if len(inners) != 1 or not isinstance(inners[0], DeepEPDispatcher):
            raise ValueError("Ascend offload currently binds the DeepEP dispatcher")
        self.dispatcher = dispatcher

    def dispatch(self, hidden_states, routing, context):
        result = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=routing
        )
        context.retain(result)
        return result

    def dispatch_completion(self, dispatched, context):
        # DeepEP dispatch_b calls current_stream_wait / its receive hook before
        # returning. This event therefore covers actual receives, not just the
        # Python dispatch call. Keep this adapter specific to that dispatcher.
        ticket = context.record_compute_completion()
        context.retain(ticket)
        return ticket.completion

    def combine(self, combine_input, context):
        result = self.dispatcher.combine(combine_input=combine_input)
        context.retain(context.record_compute_completion())
        return result

    def finish(self, output, context):
        return output


@dataclass
class AscendWaveInput:
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor | None
    expert_tokens: torch.Tensor
    routes: LocalRoutes
    wave: ExpertWave
    row_order: torch.Tensor
    valid: torch.Tensor


@dataclass
class _Layout:
    dispatched: DeepEPLLDispatchOutput | DeepEPNormalDispatchOutput
    counts: torch.Tensor
    routes: LocalRoutes
    output: torch.Tensor
    current: AscendWaveInput | None = None


class AscendDispatchLayout(DispatchLayoutAdapter):
    """Compact, expert-major NPU rows; fixed-capacity packing for each wave.

    LL counts stay on device; NORMAL host counts are allowed only in prefill.
    CUDA's [expert, capacity, hidden] DeepEP layout is deliberately not accepted.
    Scratch capacity is explicit and allocated before capture, shared serially
    across layers. Outputs retain original receive order for one native combine.
    """

    def __init__(self, local_experts, max_rows, hidden_size, dtype, device):
        self.local_experts = local_experts
        self.output = torch.empty((max_rows, hidden_size), dtype=dtype, device=device)

    def allocations(self):
        storage = self.output.untyped_storage()
        return (
            Allocation(
                "ascend.receive_output",
                ResourceKind.WORKSPACE,
                storage.nbytes(),
                (str(self.output.device), storage.data_ptr()),
                self.output,
            ),
        )

    def bind(self, dispatched, context):
        x = dispatched.hidden_states
        if x.ndim != 2 or x.shape[0] > self.output.shape[0]:
            raise ValueError("Expected bounded compact Ascend receive rows")
        if dispatched.format.is_deepep_ll():
            counts = dispatched.masked_m.to(torch.int64)
        elif dispatched.format.is_deepep_normal():
            if context.phase is not Phase.PREFILL or context.capturing:
                raise ValueError("DeepEP NORMAL host counts are prefill-only")
            counts = torch.as_tensor(
                dispatched.num_recv_tokens_per_expert,
                dtype=torch.int64,
                device=x.device,
            )
        else:
            raise ValueError("Expected Ascend DeepEP dispatch output")
        if counts.shape != (self.local_experts,):
            raise ValueError("Receive counts must cover this rank's local experts")
        rows = torch.arange(x.shape[0], device=x.device)
        ids = torch.searchsorted(counts.cumsum(0), rows, right=True)
        routes = LocalRoutes(ids, ids < self.local_experts)
        output = self.output[: x.shape[0]]
        output.zero_()
        layout = _Layout(dispatched, counts, routes, output)
        context.retain(layout)
        return layout

    def select_wave(self, layout, wave, context):
        ids = layout.routes.expert_ids
        valid = layout.routes.valid & (ids >= wave.local_start) & (ids < wave.local_end)
        # Sorting the mask retains expert-major order among selected rows. Shapes
        # depend only on dispatch capacity, never on a device count read by host.
        order = torch.argsort((~valid).to(torch.int32), stable=True)
        packed_valid = valid[order]

        def pack(tensor):
            if tensor is None:
                return None
            if tensor.shape[0] != ids.numel():
                raise ValueError(
                    "Activation scales must have the receive row axis first"
                )
            mask = packed_valid.reshape((-1,) + (1,) * (tensor.ndim - 1))
            return torch.where(mask, tensor[order], 0)

        counts = layout.counts[wave.local_start : wave.local_end].clone()
        # GMM receives fixed-capacity rows. Zero padding belongs to the last lane
        # solely for F; it is excluded from LFU and removed before combine.
        counts[-1] += ids.numel() - counts.sum()
        selected = AscendWaveInput(
            pack(layout.dispatched.hidden_states),
            pack(layout.dispatched.hidden_states_scale),
            counts,
            LocalRoutes(ids, valid),
            wave,
            order,
            packed_valid,
        )
        layout.current = selected
        return selected

    def write_wave(self, layout, wave, local_output, context):
        selected = layout.current
        masked = torch.where(selected.valid[:, None], local_output, 0)
        layout.output.index_add_(0, selected.row_order, masked)

    def combine_input(self, layout, context):
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLCombineInput,
            DeepEPNormalCombineInput,
        )

        dispatched = layout.dispatched
        cls = (
            DeepEPLLCombineInput
            if dispatched.format.is_deepep_ll()
            else DeepEPNormalCombineInput
        )
        return cls(layout.output, dispatched.topk_ids, dispatched.topk_weights)


class AscendExpertRunner:
    def __init__(self, experts):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore

        self.core = AscendRunnerCore(experts.moe_runner_config)

    def __call__(self, layer, wave_input, bindings, context):
        from sglang.srt.layers.moe.moe_runner.ascend import (
            AscendQuantInfo,
            AscendRunnerInput,
        )

        quant_info = AscendQuantInfo(**bindings.wave_components(wave_input.wave))
        if wave_input.hidden_states.shape[0] == 0:
            # Packed activation width/dtype can differ from F's output, even
            # when there are zero rows (e.g. FP4 activations).
            output = torch.empty(
                (0, self.core.config.hidden_size),
                dtype=(
                    torch.float16
                    if wave_input.hidden_states.dtype == torch.float16
                    else torch.bfloat16
                ),
                device=wave_input.hidden_states.device,
            )
        else:
            output = self.core.run(
                AscendRunnerInput(
                    wave_input.hidden_states,
                    wave_input.hidden_states_scale,
                    wave_input.expert_tokens,
                    1,
                ),
                quant_info,
                {},
            ).hidden_states
        context.retain((wave_input, bindings, quant_info, output))
        return LocalExecution(output, context.record_compute_completion())
