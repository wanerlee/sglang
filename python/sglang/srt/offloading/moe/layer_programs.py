"""Dispatch/FFN/combine once per layer, retaining the native cache policy."""

from contextlib import nullcontext

from .config import Phase
from .execution import LocalExecution, require_prefill
from .interfaces import WholeMoEProgram
from .layer_execution import LayerExecutionRequest


class LayerExpertProgram(WholeMoEProgram):
    def __init__(
        self,
        dispatcher,
        layout,
        runner,
        access,
        *,
        compute_scope,
        pipeline=None,
        profile=None,
    ):
        self.dispatcher, self.layout, self.runner = dispatcher, layout, runner
        self.access, self.compute_scope = access, compute_scope
        self.pipeline, self.profile = pipeline, profile

    def _scope(self, name, layer, bank, context):
        return (
            self.profile(name, layer, bank, context)
            if self.profile is not None
            else nullcontext()
        )

    def layer_start(self, layer, context):
        require_prefill(context)
        if self.pipeline is None:
            raise RuntimeError("Decode has no eager layer prefetch entry")
        self.pipeline.start(layer, context)

    def run(self, layer, hidden_states, routing, context):
        prefill = context.phase is Phase.PREFILL
        if prefill != (self.pipeline is not None):
            raise RuntimeError("Wrong phase for whole-layer program")
        if prefill:
            require_prefill(context)
        with context.temporary_scope(), self.compute_scope(context):
            dispatched = self.dispatcher.dispatch(hidden_states, routing, context)
            layout = self.layout.bind(dispatched, context)
            inputs = self.layout.select_layer(layout, context)
            if prefill:
                lease = self.pipeline.acquire(layer, context)
                bank = lease.reservations[0].bindings.bank_id
            else:
                bank = 0
                with self._scope("layer_load_submit", layer, bank, context):
                    lease = self.access.stage(
                        LayerExecutionRequest(layer, layout.counts, Phase.DECODE),
                        context,
                    )
            with self._scope("layer_wait_ready", layer, bank, context):
                bindings = self.access.wait_ready(
                    lease, context.compute_stream, context=context
                )
            with self._scope("layer_compute_submit", layer, bank, context):
                result = self.runner(layer, inputs, bindings, context)
            if not isinstance(result, LocalExecution):
                raise TypeError("Layer FFN must return LocalExecution")
            context.retain(result)
            context.wait_for(result.read_done, context.compute_stream)
            if prefill:
                self.pipeline.retire(layer, lease, result.read_done, context)
            else:
                self.access.release_after(lease, result.read_done, context)
            self.layout.write_layer(layout, result.output, context)
            combined = self.dispatcher.combine(
                self.layout.combine_input(layout, context), context
            )
            output = self.dispatcher.finish(combined, context)
            if prefill:
                self.pipeline.finish_layer(layer, context)
            return output
