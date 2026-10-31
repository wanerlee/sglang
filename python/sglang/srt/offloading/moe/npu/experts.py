"""Native Ascend dispatch-once execution over fixed offload banks.

The NPU DeepEP receive buffer is a *flat*, expert-grouped allocation. Its device
counts delimit the meaningful prefix; CUDA's [expert, capacity, hidden] layout
is deliberately not accepted. Decode keeps fixed activation capacity; eager
NORMAL prefill uses the dispatcher's host counts to slice the full layer's actual
rows. Native GMM/activation temporaries still belong to the workspace budget.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, fields

import torch

from ...transfer import EventCompletion, JoinedCompletion, TransferTicket
from ..config import Phase
from ..execution import LocalExecution
from ..interfaces import DispatchLayoutAdapter, MoEDispatcher
from ..resources import ResourceKind, tensor_allocations
from .config import validate_native_environment


def _compute_scope(context, device):
    if torch.device(device).type != "npu":
        raise ValueError("Native expert execution requires an NPU device")
    scope = ExitStack()
    scope.enter_context(torch.npu.device(device))
    scope.enter_context(torch.npu.stream(context.compute_stream))
    return scope


def _format_name(raw):
    value = raw.format
    return getattr(value, "value", value)


@dataclass(frozen=True)
class NPUExpertBank:
    """An exact native weight/validity view, validated once when bound.

    Decode binds views before capture; eager prefill binds its actual group width.
    Weight tensors and Parameters are never rebound during native execution.
    """

    quant_info: object
    execution_valid: torch.Tensor
    error_status: torch.Tensor

    def __post_init__(self):
        valid, error = self.execution_valid, self.error_status
        if (
            valid.ndim != 1
            or valid.dtype != torch.bool
            or valid.numel() == 0
            or error.shape != torch.Size([])
            or error.dtype != torch.int32
            or error.device != valid.device
        ):
            raise ValueError("FFN bank requires boolean validity and int32 error")
        for name in ("w13_weight", "w2_weight"):
            weight = getattr(self.quant_info, name, None)
            if not isinstance(weight, torch.Tensor) or weight.ndim < 2:
                raise ValueError("FFN bank weights must be native expert-major tensors")
        for field in fields(self.quant_info):
            tensor = getattr(self.quant_info, field.name)
            if tensor is not None and (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim == 0
                or tensor.shape[0] != valid.numel()
                or tensor.device != valid.device
            ):
                raise ValueError(
                    "Quantization components must match the exact FFN bank"
                )


@dataclass(frozen=True)
class NPUDispatch:
    raw: object
    original_hidden_width: int
    handle: object
    valid_token_count: object = None


@dataclass
class _Layout:
    dispatched: NPUDispatch
    counts: torch.Tensor
    offsets: torch.Tensor | None
    output: torch.Tensor
    row_ids: torch.Tensor
    selected: object = None
    prefill_rows: int | None = None


@dataclass(frozen=True)
class NPULayerInput:
    hidden_states: torch.Tensor
    hidden_states_scale: object
    expert_tokens: torch.Tensor
    row_count: torch.Tensor | int


@dataclass(frozen=True)
class _Combined:
    hidden_states: torch.Tensor
    original_hidden_width: int
    valid_token_count: object = None


class NPUOffloadDispatcher(MoEDispatcher):
    """Preserve native EP IDs, weights, receive waits and combine handles."""

    def __init__(self, experts, *, error_status):
        self.experts = experts
        if (
            not isinstance(error_status, torch.Tensor)
            or error_status.shape != torch.Size([])
            or error_status.dtype != torch.int32
        ):
            raise ValueError("Dispatcher requires the canonical int32 error scalar")
        self.error_status = error_status
        self.dispatcher = experts.dispatcher
        inners = getattr(self.dispatcher, "_inners", None)
        if inners is not None and len(inners) != 1:
            raise ValueError("MoE offload does not support two-batch overlap")
        self.inner = inners[0] if inners is not None else self.dispatcher
        validate_native_environment()

    def _mask_routing(self, hidden_states, routing, context):
        count = getattr(context, "num_token_non_padded", None)
        if count is None:
            if context.phase is Phase.DECODE:
                raise ValueError("Decode dispatch requires device num_token_non_padded")
            return routing, None
        if (
            not isinstance(count, torch.Tensor)
            or count.numel() != 1
            or count.ndim > 1
            or count.dtype not in (torch.int32, torch.int64)
            or count.device != hidden_states.device
            or self.error_status.device != hidden_states.device
        ):
            raise ValueError("Non-padding token count must be a device integer scalar")
        if not hasattr(routing, "_replace") or not hasattr(routing, "topk_ids"):
            raise ValueError("NPU offload requires standard, unpacked TopKOutput")
        ids, weights = routing.topk_ids, routing.topk_weights
        if (
            ids.ndim != 2
            or ids.shape[0] != hidden_states.shape[0]
            or weights.shape != ids.shape
            or ids.device != hidden_states.device
            or weights.device != hidden_states.device
        ):
            raise ValueError("Routing tensors must match the dispatched token rows")
        count = count.reshape(())
        invalid = (count < 0) | (count > hidden_states.shape[0])
        self.error_status.bitwise_or_(invalid.to(torch.int32) * 256)
        active = (torch.arange(ids.shape[0], device=ids.device) < count) & (
            self.error_status == 0
        )
        result = routing._replace(
            topk_ids=torch.where(active.unsqueeze(1), ids, -1),
            topk_weights=torch.where(active.unsqueeze(1), weights, 0),
        )
        context.retain((count, result))
        return result, count

    def dispatch(self, hidden_states, routing, context):
        context.retain((self, hidden_states, routing))
        with _compute_scope(context, hidden_states.device):
            routing, count = self._mask_routing(hidden_states, routing, context)
            # Native dispatch_b joins its communication event/receive hook onto
            # this stream. Recording after dispatch therefore covers real D.
            raw = self.dispatcher.dispatch(
                hidden_states=hidden_states, topk_output=routing
            )
            context.retain(raw)
            implementation = (
                self.inner._get_impl()
                if hasattr(self.inner, "_get_impl")
                else self.inner
            )
            handle = getattr(implementation, "handle", raw)
            # combine clears implementation.handle; retain the *value* now.
            context.retain(handle)
            result = NPUDispatch(raw, hidden_states.shape[-1], handle, count)
            context.retain(result)
            return result

    @staticmethod
    def _combine_input(raw, hidden_states):
        name = _format_name(raw)
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLCombineInput,
            DeepEPNormalCombineInput,
        )

        constructors = {
            "deepep_ll": DeepEPLLCombineInput,
            "deepep_normal": DeepEPNormalCombineInput,
        }
        if name not in constructors:
            raise ValueError("Unsupported NPU offload dispatcher: " + str(name))
        return constructors[name](hidden_states, raw.topk_ids, raw.topk_weights)

    def combine(self, local_output, context):
        layout = local_output
        context.retain(layout)
        with _compute_scope(context, layout.output.device):
            combine_input = self._combine_input(layout.dispatched.raw, layout.output)
            context.retain(combine_input)
            # Ascend uses HCCL; CUDA symmetric-memory allocation is inapplicable.
            output = self.dispatcher.combine(combine_input=combine_input)
            result = _Combined(
                output,
                layout.dispatched.original_hidden_width,
                layout.dispatched.valid_token_count,
            )
            context.retain(result)
            return result

    def finish(self, accumulator, context):
        with _compute_scope(context, accumulator.hidden_states.device):
            output = accumulator.hidden_states[
                ..., : accumulator.original_hidden_width
            ].contiguous()
            count = accumulator.valid_token_count
            if count is not None:
                good = (torch.arange(output.shape[0], device=output.device) < count) & (
                    self.error_status == 0
                )
                output = torch.where(good.unsqueeze(1), output, 0)
            if self.experts.reduce_results and (
                self.experts.moe_tp_size > 1 or self.experts.moe_ep_size > 1
            ):
                from sglang.srt.distributed import tensor_model_parallel_all_reduce

                output = tensor_model_parallel_all_reduce(output)
            context.retain(output)
            return output


class NPUDispatchLayout(DispatchLayoutAdapter):
    """Bounded decode packing and compact eager prefill over one receive buffer.

    Byte views preserve MXFP8 activation/scale bits without relying on float8
    index_select support. Receive padding is zeroed and never counted as expert
    demand. NORMAL prefill slices contiguous expert rows
    using existing host counts, without reading device counts back to the CPU.
    Buffers are reused only on the compute stream after the local program has
    joined every registered FFN reader.
    """

    def __init__(
        self, local_experts, hidden_size, max_dispatch_rows, device, error_status
    ):
        if min(local_experts, hidden_size, max_dispatch_rows) <= 0:
            raise ValueError("NPU dispatch workspace capacities must be positive")
        self.local_experts = local_experts
        self.hidden_size = hidden_size
        self.max_rows = max_dispatch_rows
        self.device = torch.device(device)
        if (
            error_status.shape != torch.Size([])
            or error_status.dtype != torch.int32
            or error_status.device != self.device
        ):
            raise ValueError("Layout requires the canonical device error scalar")
        self.error_status = error_status
        self._owned = {}
        self._allocate(
            "input_bytes", (max_dispatch_rows * hidden_size * 2,), torch.uint8
        )
        # Covers E8M0 MXFP8 scales and FP32 per-token/group scales.
        scale_bytes_per_row = max(4, (hidden_size + 31) // 32 * 4)
        self._allocate(
            "scale_bytes", (max_dispatch_rows * scale_bytes_per_row,), torch.uint8
        )
        self._allocate("output", (max_dispatch_rows, hidden_size), torch.bfloat16)
        self._allocate("counts", (local_experts,), torch.int64)
        self._allocate("offsets", (local_experts + 1,), torch.int64)
        rows = self._allocate("rows", (max_dispatch_rows,), torch.int64)
        torch.arange(max_dispatch_rows, out=rows)

    def _allocate(self, name, shape, dtype):
        value = torch.empty(shape, dtype=dtype, device=self.device)
        self._owned[name] = value
        return value

    def allocations(self, prefix="npu_dispatch"):
        return tensor_allocations(
            prefix + ".", self._owned.items(), ResourceKind.WORKSPACE
        )

    def bind(self, dispatched, context):
        raw = dispatched.raw
        x = raw.hidden_states
        if (
            x.ndim != 2
            or x.shape[1] != self.hidden_size
            or x.shape[0] > self.max_rows
            or x.device != self.device
            or not x.is_contiguous()
            or x.element_size() > 2
        ):
            raise ValueError("Unsupported NPU compact dispatch shape or capacity")
        name = _format_name(raw)
        prefill_rows = None
        if name == "deepep_normal":
            if context.capturing or context.phase is Phase.DECODE:
                raise ValueError("Decode requires device-count DeepEP low latency")
            host_counts = raw.num_recv_tokens_per_expert
            if (
                not isinstance(host_counts, (list, tuple))
                or len(host_counts) != self.local_experts
                or any(type(count) is not int or count < 0 for count in host_counts)
            ):
                raise ValueError(
                    "Prefill dispatch requires nonnegative host counts for every expert"
                )
            prefill_rows = sum(host_counts)
            if prefill_rows > x.shape[0]:
                raise ValueError("Prefill dispatch counts exceed receive capacity")
            source = torch.as_tensor(host_counts, dtype=torch.int64, device=self.device)
        elif name == "deepep_ll":
            source = raw.masked_m
        else:
            raise ValueError("Unsupported NPU dispatch layout or count convention")
        if (
            source.shape != (self.local_experts,)
            or source.dtype not in (torch.int32, torch.int64)
            or source.device != self.device
        ):
            raise ValueError("Dispatch counts must cover every local expert")
        context.retain((self, dispatched, source))
        with _compute_scope(context, self.device):
            if prefill_rows is not None:
                # NORMAL already supplies checked host counts. Whole-layer FFN
                # needs their sum, not per-expert offsets or a second copy.
                counts, offsets = source, None
            else:
                counts = self._owned["counts"]
                counts.copy_(source)
                invalid = (counts < 0).any() | (counts.sum() > x.shape[0])
                self.error_status.bitwise_or_(invalid.to(torch.int32) * 64)
                counts.masked_fill_(self.error_status != 0, 0)
                offsets = self._owned["offsets"]
                offsets[0].zero_()
                torch.cumsum(counts, 0, out=offsets[1:])
            output = self._owned["output"][: x.shape[0]]
            if x.dtype == torch.float16:
                output = output.view(torch.float16)
            if prefill_rows is None:
                output.zero_()
            result = _Layout(
                dispatched,
                counts,
                offsets,
                output,
                self._owned["rows"][: x.shape[0]],
                prefill_rows=prefill_rows,
            )
            context.retain(result)
            return result

    def select_layer(self, layout, context):
        """Use the already expert-grouped prefix; no offset gather or partition."""
        if layout.selected is not None:
            raise RuntimeError("Write the previous layer before reusing its input")
        raw = layout.dispatched.raw
        rows = raw.hidden_states.shape[0]
        count = (
            layout.prefill_rows
            if layout.prefill_rows is not None
            else layout.offsets[-1]
        )
        with _compute_scope(context, self.device):
            if layout.prefill_rows is not None:
                used = layout.prefill_rows
                x = raw.hidden_states[:used]
            else:
                used = rows
                x = self._copy_layer_bytes(
                    raw.hidden_states, layout.row_ids >= count, "input_bytes"
                )
            scale = raw.hidden_states_scale
            if scale is not None:
                if rows == 0:
                    if scale.numel():
                        raise ValueError("Empty input cannot have nonempty scales")
                else:
                    if not scale.is_contiguous() or scale.numel() % rows:
                        raise ValueError(
                            "Layer scales require a fixed contiguous row shape"
                        )
                    scale = scale.view(rows, -1)[:used]
                    if layout.prefill_rows is None:
                        scale = self._copy_layer_bytes(
                            scale, layout.row_ids >= count, "scale_bytes"
                        )
            result = NPULayerInput(x, scale, layout.counts, count)
            layout.selected = result
            context.retain(result)
            return result

    def _copy_layer_bytes(self, source, padding, name):
        # Reuse the already budgeted workspace, including float8/E8M0 bits.
        # No per-layer clone or graph-dependent receive allocation is needed.
        nbytes = source.numel() * source.element_size()
        storage = self._owned[name]
        if nbytes > storage.numel():
            raise ValueError("Layer input exceeds its reserved byte workspace")
        target = storage[:nbytes].view(source.dtype).view(source.shape)
        target.view(torch.int8).copy_(source.view(torch.int8))
        # CANN masked-fill does not support uint8 on the supported builds.
        target.view(torch.int8).masked_fill_(padding.unsqueeze(1), 0)
        return target

    def write_layer(self, layout, output, context):
        if not isinstance(layout.selected, NPULayerInput):
            raise RuntimeError("Layer write requires its selected full-layer input")
        rows = layout.selected.hidden_states.shape[0]
        target = layout.output[:rows]
        if output.shape != target.shape:
            raise ValueError("Layer FFN must preserve the selected receive shape")
        with _compute_scope(context, self.device):
            if layout.prefill_rows is not None and rows == layout.output.shape[0]:
                # Compact NORMAL output already has the native combine layout.
                layout.output = output
            else:
                if layout.prefill_rows is not None:
                    # Retain the receive capacity when it includes padding.
                    # Only the unwritten tail needs initialization.
                    layout.output[rows:].zero_()
                target.copy_(output)
            layout.selected = None

    def combine_input(self, layout, context):
        if layout.selected is not None:
            raise RuntimeError("Cannot combine an unfinished FFN layer")
        # A latched failure suppresses the complete output too; never publish a partial
        # routed result when bank validation fails.
        with _compute_scope(context, self.device):
            layout.output.masked_fill_(self.error_status != 0, 0)
        return layout


class NPUExpertRunner:
    """Run existing native GMM + activation with an offload bank QuantInfo.

    AscendRunnerCore submits both GMMs and activation on the current stream. It
    receives no overlap hooks. Additional weight-reading streams must be declared
    here and registered with the runtime's failure drain before use.
    """

    def __init__(self, experts, *, reader_streams=(), _core=None, _input_type=None):
        if _core is None:
            from sglang.srt.layers.moe.moe_runner.ascend import (
                AscendRunnerCore,
                AscendRunnerInput,
            )

            _core = AscendRunnerCore(experts.moe_runner_config)
            _input_type = AscendRunnerInput
        self.core = _core
        self.input_type = _input_type
        self.reader_streams = tuple(reader_streams)

    def _reader_ticket(self, context):
        compute = context.record_compute_completion()
        tickets = [compute]
        for stream in self.reader_streams:
            event = torch.npu.Event()
            event.record(stream)
            tickets.append(TransferTicket(EventCompletion(event), (stream, event)))
        ticket = TransferTicket(
            JoinedCompletion(t.completion for t in tickets), (self, *tickets)
        )
        context.retain(ticket)
        return ticket

    def __call__(self, layer, dispatched, bindings, context):
        bank = getattr(bindings, "expert_bank", None)
        if not isinstance(bank, NPUExpertBank):
            raise TypeError("Native FFN requires bindings.expert_bank")
        if (
            getattr(bindings, "error_status", bank.error_status)
            is not bank.error_status
        ):
            raise ValueError("FFN and transfer must share canonical error_status")
        width = dispatched.expert_tokens.numel()
        if width != bank.execution_valid.numel():
            raise ValueError("FFN layer must match its bound expert view")
        quant = bank.quant_info
        # Retain *before* submission, including partial native failures.
        context.retain((self, bank, quant, dispatched, bindings, self.reader_streams))
        with _compute_scope(context, dispatched.hidden_states.device):
            if (
                context.phase is Phase.PREFILL
                and not context.capturing
                and dispatched.hidden_states.shape[0] == 0
            ):
                # Empty eager layers still retire their weight lease and readers,
                # but do not call native quantization/GMM with a zero-row input.
                dtype = (
                    torch.float16
                    if dispatched.hidden_states.dtype == torch.float16
                    else torch.bfloat16
                )
                output = torch.empty_like(dispatched.hidden_states, dtype=dtype)
                context.retain(output)
                return LocalExecution(output, self._reader_ticket(context))
            compact_prefill = context.phase is Phase.PREFILL and isinstance(
                dispatched.row_count, int
            )
            if compact_prefill:
                # The complete bank is ready after wait_ready; no dynamic slot
                # validity to query. Check the shared fault latch after the wait.
                counts = dispatched.expert_tokens
                counts.masked_fill_(bank.error_status != 0, 0)
            else:
                valid = bank.execution_valid
                bad = ((dispatched.expert_tokens > 0) & ~valid).any()
                bank.error_status.bitwise_or_(bad.to(torch.int32) * 128)
                healthy = bank.error_status == 0
                counts = torch.where(healthy, dispatched.expert_tokens, 0)
            runner_input = self.input_type(
                hidden_states=dispatched.hidden_states,
                hidden_states_scale=dispatched.hidden_states_scale,
                expert_tokens=counts,
                group_list_type=1,
            )
            context.retain(runner_input)
            output = self.core.run(runner_input, quant, {}).hidden_states
            context.retain(output)
            if not compact_prefill:
                # Unwritten GMM padding may contain NaN; multiply-by-zero is unsafe.
                row_ids = torch.arange(output.shape[0], device=output.device)
                good_rows = (row_ids < dispatched.row_count) & healthy
                output.masked_fill_(~good_rows.unsqueeze(1), 0)
            # Compact prefill has no padding. combine_input applies the shared
            # fault latch once, including any unwritten output on a failed step.
            return LocalExecution(output, self._reader_ticket(context))


@dataclass(frozen=True)
class NPULayerPrograms:
    prefill: object
    decode: object
    layout: NPUDispatchLayout

    def allocations(self, prefix=None):
        if prefix is None:
            prefix = f"npu_dispatch.{self.prefill.dispatcher.experts.layer_id}"
        return self.layout.allocations(prefix)


def build_npu_layer_programs(
    experts,
    *,
    access,
    residency,
    pipeline,
    max_dispatch_rows,
    device=None,
    shared_layout=None,
    profile=None,
):
    """Bind a real FusedMoE's native dispatcher and Ascend FFN to offload access.

    The caller binds banks on every lease, budgets allocations(), and invokes
    prefill.layer_start at the decoder-layer entry. Forward results already have
    FusedMoE's crop/reduce tail; shared experts/scaling remain in the outer MLP.
    """
    device = residency.error_status.device if device is None else device
    dispatcher = NPUOffloadDispatcher(experts, error_status=residency.error_status)
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher

    if not isinstance(dispatcher.inner, DeepEPDispatcher):
        raise ValueError("Native offload execution requires the DeepEP dispatcher")
    if shared_layout is None:
        layout = NPUDispatchLayout(
            experts.num_local_experts,
            experts.moe_runner_config.hidden_size,
            max_dispatch_rows,
            device,
            residency.error_status,
        )
    else:
        layout = shared_layout
        if (
            not isinstance(layout, NPUDispatchLayout)
            or layout.local_experts != experts.num_local_experts
            or layout.hidden_size != experts.moe_runner_config.hidden_size
            or layout.max_rows < max_dispatch_rows
            or layout.device != torch.device(device)
            or layout.error_status is not residency.error_status
        ):
            raise ValueError("Shared dispatch workspace has incompatible bindings")
    runner = NPUExpertRunner(experts)

    from ..layer_programs import LayerExpertProgram

    prefill = LayerExpertProgram(
        dispatcher,
        layout,
        runner,
        access,
        compute_scope=lambda ctx: _compute_scope(ctx, device),
        pipeline=pipeline,
        profile=profile,
    )
    decode = LayerExpertProgram(
        dispatcher,
        layout,
        runner,
        access,
        compute_scope=lambda ctx: _compute_scope(ctx, device),
        profile=profile,
    )
    return NPULayerPrograms(prefill, decode, layout)
