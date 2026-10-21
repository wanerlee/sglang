"""Install whole-MoE programs on the existing DeepSeek V4 routed path."""

from functools import wraps

from .config import Phase
from .interfaces import WholeMoEProgram


class LayerPrograms(WholeMoEProgram):
    """One runtime entry point, with the actual dispatcher bound per layer."""

    def __init__(self, programs):
        self.programs = dict(programs)

    def layer_start(self, layer_id, context):
        self.programs[layer_id].layer_start(layer_id, context)

    def run(self, layer, hidden_states, routing, context):
        return self.programs[layer.layer_id].run(layer, hidden_states, routing, context)


class DeepSeekV4ProgramBinding:
    """Leave attention, router, shared experts and routed scaling in the model.

    Only the routed experts' forward is replaced. This also covers DeepEPMoE's
    forward override without redirecting its native quantization method or
    changing logical expert counts. All handles are removed on runtime close.
    """

    def __init__(self, layers):
        self.layers = tuple(layers)
        self.runtime = None
        self._handles = []
        self._forwards = []

    def install(self, runtime):
        if self.runtime is not None:
            raise RuntimeError("Model programs are already installed")
        self.runtime = runtime
        for decoder, spec in self.layers:
            experts = decoder.mlp.experts
            self._forwards.append((experts, "forward", experts.__dict__.get("forward")))
            experts.forward = self._routed_forward(experts)
            self._handles.append(decoder.register_forward_pre_hook(self._layer_start))
            # The actual non-CUDA V4 model calls this method directly, bypassing
            # nn.Module.__call__ and hence forward_pre_hooks.
            name = "forward_hc_pre_from_prev"
            if hasattr(decoder, name):
                self._forwards.append((decoder, name, decoder.__dict__.get(name)))
                setattr(
                    decoder,
                    name,
                    self._platform_entry(getattr(decoder, name), spec.layer_id),
                )

    def _platform_entry(self, original, layer_id):
        @wraps(original)
        def forward(*args, **kwargs):
            self.runtime.emit_layer_start(layer_id, self.runtime.context)
            return original(*args, **kwargs)

        return forward

    def _layer_start(self, decoder, args):
        self.runtime.emit_layer_start(decoder.layer_id, self.runtime.context)

    def _routed_forward(self, experts):
        def forward(hidden_states, topk_output, pre_quant_input=None):
            # DeepEP owns activation quantization. Never silently drop a separate
            # asynchronous pre-quant input with an unjoined producer stream.
            if pre_quant_input is not None:
                raise ValueError("MoE offload uses dispatcher-owned quantization")
            runtime = self.runtime
            context = runtime.context
            runtime.emit_after_router(experts.layer_id, topk_output, context)
            program = (
                runtime.prefill_program
                if context.phase is Phase.PREFILL
                else runtime.decode_program
            )
            output = program.run(experts, hidden_states, topk_output, context)
            output = output[..., : hidden_states.shape[-1]].contiguous()
            # Preserve FusedMoE.forward_impl's tail. The enclosing MLP still
            # owns shared experts and its own post-expert reduction/scaling.
            if experts.reduce_results and (
                experts.moe_tp_size > 1 or experts.moe_ep_size > 1
            ):
                from sglang.srt.distributed import tensor_model_parallel_all_reduce

                output = tensor_model_parallel_all_reduce(output)
            return output

        return forward

    def close(self):
        for handle in self._handles:
            handle.remove()
        for module, name, previous in self._forwards:
            if previous is None:
                delattr(module, name)
            else:
                setattr(module, name, previous)
        self._handles.clear()
        self._forwards.clear()
        self.runtime = None
