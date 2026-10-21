"""DFC schedules independent of weight transport and accelerator layouts."""

from ..transfer import Completion, TransferTicket
from .config import Phase
from .execution import LocalExecution, require_prefill
from .interfaces import LocalExpertProgram, WholeMoEProgram


class ComposedDFCProgram(WholeMoEProgram):
    def __init__(self, waves, dispatcher, local_program):
        self.waves = tuple(waves)
        if not self.waves:
            raise ValueError("A DFC program requires a static nonempty wave schedule")
        self.dispatcher = dispatcher
        self.local_program = local_program

    def run(self, layer, hidden_states, routing, context):
        # This static Python loop is unrolled at capture. Its length never depends
        # on misses or device route contents. A dispatcher handle lives for D/F/C.
        result = self.dispatcher.accumulator(hidden_states, context)
        for wave in self.waves:
            masked = self.dispatcher.mask_routes(routing, wave, context)
            dispatched = self.dispatcher.dispatch(hidden_states, masked, context)
            local = self.local_program.run(layer, dispatched, wave, context)
            partial = self.dispatcher.combine(local, context)
            self.dispatcher.accumulate(result, partial, context)
        return self.dispatcher.finish(result, context)


class DispatchOnceProgram(WholeMoEProgram):
    """One full dispatch, fixed local F waves, then one full combine.

    Bind one instance to a local expert count. The layout adapter supplies views
    or explicitly budgeted packing buffers; the local program owns weight
    readiness and reader protection. Neither transport nor cache policy is
    selected here. This sequential schedule does not itself add H2D overlap.
    """

    def __init__(
        self, waves, dispatcher, local_program, layout_adapter, *, local_experts
    ):
        self._bind(dispatcher, local_program, layout_adapter, local_experts)
        self.waves = tuple(waves)
        self._validate_waves(self.waves)

    def _bind(self, dispatcher, local_program, layout_adapter, local_experts):
        if local_experts <= 0:
            raise ValueError("Dispatch-once requires local experts")
        self.local_experts = local_experts
        self.dispatcher = dispatcher
        self.local_program = local_program
        self.layout_adapter = layout_adapter

    def _validate_waves(self, waves):
        if not waves:
            raise ValueError("Dispatch-once requires a nonempty local expert schedule")
        end = 0
        for index, wave in enumerate(waves):
            if (
                wave.index != index
                or wave.local_start != end
                or not end < wave.local_end <= self.local_experts
                or wave.bank < 0
            ):
                raise ValueError("Waves must partition local experts in order")
            end = wave.local_end
        if end != self.local_experts:
            raise ValueError("Waves must cover every local expert before dispatch")
        # Capacity/format compatibility of each bank is validated by the runtime.

    def _wave_schedule(self, layer, dispatched, layout, context):
        return self.waves

    def run(self, layer, hidden_states, routing, context):
        # Preserve original global IDs/weights and keep this D handle through C.
        dispatched = self.dispatcher.dispatch(hidden_states, routing, context)
        layout = self.layout_adapter.bind(dispatched, context)
        # Unrolled at capture, including empty waves. Dynamic rows stay on device.
        for wave in self._wave_schedule(layer, dispatched, layout, context):
            wave_input = self.layout_adapter.select_wave(layout, wave, context)
            local = self.local_program.run(layer, wave_input, wave, context)
            self.layout_adapter.write_wave(layout, wave, local, context)
        combined = self.dispatcher.combine(
            self.layout_adapter.combine_input(layout, context), context
        )
        return self.dispatcher.finish(combined, context)


class PrefillDispatchOnceProgram(DispatchOnceProgram):
    """Reuse D/F/C with per-invocation eager waves sealed by dispatch feedback."""

    def __init__(
        self, dispatcher, local_program, layout_adapter, *, local_experts, pipeline
    ):
        if local_program.pipeline is not pipeline:
            raise ValueError("Whole/local prefill programs must share one pipeline")
        self._bind(dispatcher, local_program, layout_adapter, local_experts)
        self.pipeline = pipeline
        self.waves = ()  # Actual waves belong to the invocation, not this program.

    def layer_start(self, layer_id, context):
        require_prefill(context)
        context.retain(self.pipeline)
        try:
            self.pipeline.start(layer_id, context)
        except BaseException:
            self._retain_completion(self.pipeline.abort(context), context)
            raise

    def _wave_schedule(self, layer, dispatched, layout, context):
        ready = self.dispatcher.dispatch_completion(dispatched, context)
        if not isinstance(ready, Completion):
            raise TypeError("Dispatcher must return a recorded Completion")
        context.retain(ready)
        waves = tuple(self.pipeline.seal(layer, ready, context))
        self._validate_waves(waves)
        return waves

    def run(self, layer, hidden_states, routing, context):
        require_prefill(context)
        context.retain(self.pipeline)
        try:
            self.pipeline.validate_session(layer, context)
            output = super().run(layer, hidden_states, routing, context)
            self._retain_completion(self.pipeline.finish(layer, context), context)
            return output
        except BaseException:
            self._retain_completion(self.pipeline.abort(context), context)
            raise

    @staticmethod
    def _retain_completion(ticket, context):
        if not isinstance(ticket, TransferTicket):
            raise TypeError("Pipeline completion must return TransferTicket")
        context.retain(ticket)


def _run_local(run_experts, access, layer, dispatch_output, lease, context):
    bindings = access.wait_ready(lease, context.compute_stream)
    result = run_experts(layer, dispatch_output, bindings, context)
    if not isinstance(result, LocalExecution):
        raise TypeError("Expert runner must return LocalExecution")
    context.retain(result)
    if not isinstance(result.read_done, TransferTicket):
        raise TypeError("Expert runner must return a reader TransferTicket")
    # Join auxiliary FFN streams before write_wave or the next decode load.
    result.read_done.completion.wait_on(context.compute_stream)
    access.release_after(lease, result.read_done, context)
    return result.output


class PrefillLocalExpertProgram(LocalExpertProgram):
    def __init__(self, pipeline, run_experts):
        self.pipeline = pipeline
        self.run_experts = run_experts

    def run(self, layer, dispatch_output, wave, context):
        require_prefill(context)
        lease = self.pipeline.acquire(layer, wave, context)
        return _run_local(
            self.run_experts,
            self.pipeline.access,
            layer,
            dispatch_output,
            lease,
            context,
        )


class DecodeLocalExpertProgram(LocalExpertProgram):
    """Demand -> current-wave copy -> readiness -> F -> reader, without prefetch.

    make_request and run_experts are backend adapters. They emit device operations
    with fixed bindings; no replay-time Python routing/shape decisions are allowed.
    The runner retains resources before any submission, including on exceptions.
    """

    def __init__(self, access, make_request, run_experts):
        self.access = access
        self.make_request = make_request
        self.run_experts = run_experts

    def run(self, layer, dispatch_output, wave, context):
        if context.phase is not Phase.DECODE:
            raise RuntimeError("Decode local program requires decode mode")
        request = self.make_request(layer, dispatch_output, wave, context)
        if request.wave != wave:
            raise ValueError("Decode must load only the current wave")
        lease = self.access.stage(request, context)
        return _run_local(
            self.run_experts, self.access, layer, dispatch_output, lease, context
        )
