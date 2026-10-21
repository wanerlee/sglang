"""Cache/framework assembly over externally prepared native weight storage.

The physical binding callback does not install model hooks or assemble programs.
It supplies weight pools, transfer and prefill pipeline only; this module owns
the model/runtime integration. No alternate transport or eager serving fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..transfer import WeightTransferBackend
from .ascend_adapter import (
    AscendDeepEPDispatcher,
    AscendDispatchLayout,
    AscendExpertRunner,
    ExpertBankBindings,
)
from .cache_runtime import CacheExecutionContext, CacheRuntime
from .config import fixed_waves
from .interfaces import MoEBackend, RuntimeCapabilities
from .model_binding import DeepSeekV4ProgramBinding, LayerPrograms
from .prefill import PrefillWeightPipeline
from .programs import (
    DecodeLocalExpertProgram,
    DispatchOnceProgram,
    PrefillDispatchOnceProgram,
    PrefillLocalExpertProgram,
)
from .state import ResidencyState

if TYPE_CHECKING:
    import torch

    from .resources import Allocation


@dataclass
class CacheBackendBinding:
    """Physical handoff, created after the model's quantization post-load.

    prepare_weights below owns bounded checkpoint loading and native packing.
    resources reports host/hot/execution storage and pipeline allocations once.
    Both transfer bindings expose native components through ExpertBankBindings;
    transport-specific addresses/handles may be additional fields of subclasses.
    The pipeline uses its existing HOST_ONLY access and prepared bank views.
    """

    device_module: object
    device: torch.device | str
    compute_stream: object
    copy_stream: object
    transfer: WeightTransferBackend
    hot_bindings: ExpertBankBindings
    execution_bindings: ExpertBankBindings
    prefill_pipeline: PrefillWeightPipeline
    borrow_hot_arena: Callable[[], None]
    resources: tuple[Allocation, ...]
    expert_nbytes: int
    bank_slots: int
    max_dispatch_rows: int
    hidden_size: int
    output_dtype: torch.dtype
    capabilities: RuntimeCapabilities


class CacheBackend(MoEBackend):
    """DeepSeek V4 + Ascend/DeepEP binding for MoEOffloaderV2 / ModelRunner.

    prepare_weights(decoder, spec): establish host checkpoint targets.
    bind_weights(layers, catalog, config): return CacheBackendBinding.
    close_weights(): release registrations after runtime/graphs are drained.
    These are physical-storage operations, not callbacks for our missing model
    integration. Actual dispatcher, F, programs and installer are constructed here.
    """

    def __init__(self, *, model_id, prepare_weights, bind_weights, close_weights):
        self.model_id = model_id
        self.prepare_weights = prepare_weights
        self.bind_weights = bind_weights
        self.close_weights = close_weights
        self.runtime = None

    def prepare_layer(self, module, spec):
        self.prepare_weights(module, spec)

    def finalize(self, layers, catalog, config, policy):
        binding = self.bind_weights(layers, catalog, config)
        if binding.bank_slots <= 0 or config.execution_slots % binding.bank_slots:
            raise ValueError("Execution pool must contain whole banks")
        if binding.execution_bindings.bank_offsets != tuple(
            range(0, config.execution_slots, binding.bank_slots)
        ):
            raise ValueError("Native bank offsets must match cache destination slots")
        for banks, capacity in (
            (binding.hot_bindings, config.hot_slots(catalog)),
            (binding.execution_bindings, config.execution_slots),
        ):
            if not {"w13_weight", "w2_weight"} <= banks.components.keys():
                raise ValueError("Native bank requires both expert projections")
            if any(t.shape[0] != capacity for t in banks.components.values()):
                raise ValueError("Native components must cover the physical slot pool")
        state = ResidencyState.allocate(
            catalog.num_experts,
            config.hot_slots(catalog),
            config.execution_slots,
            binding.device,
        )
        # All eligible DeepSeek V4 layers share the local expert/hidden geometry.
        local_experts = catalog.offloaded_layers[0].local_experts
        if any(s.local_experts != local_experts for s in catalog.offloaded_layers):
            raise ValueError(
                "Ascend cache binding requires uniform local expert counts"
            )
        layout = AscendDispatchLayout(
            local_experts,
            binding.max_dispatch_rows,
            binding.hidden_size,
            binding.output_dtype,
            binding.device,
        )
        runtime = CacheRuntime(
            catalog=catalog,
            residency=state,
            policy=policy,
            transfer=binding.transfer,
            context=CacheExecutionContext(
                binding.device_module, binding.compute_stream, binding.copy_stream
            ),
            hot_bindings=binding.hot_bindings,
            execution_bindings=binding.execution_bindings,
            bank_slots=binding.bank_slots,
            capabilities=binding.capabilities,
            expert_nbytes=binding.expert_nbytes,
            resources=tuple(binding.resources) + layout.allocations(),
            borrow_hot_arena=binding.borrow_hot_arena,
            model_binding=DeepSeekV4ProgramBinding(layers),
            prefill_pipeline=binding.prefill_pipeline,
            close_weights=self.close_weights,
        )
        self.runtime = runtime  # Preserve partially assembled work for cleanup.
        decode, prefill = {}, {}
        for decoder, spec in layers:
            experts = decoder.mlp.experts
            dispatcher = AscendDeepEPDispatcher(experts.dispatcher)
            native_f = AscendExpertRunner(experts)
            local = DecodeLocalExpertProgram(
                runtime.access,
                lambda layer, selected, wave, context: runtime.make_execution_request(
                    layer.layer_id, selected.routes, wave, context
                ),
                native_f,
            )
            decode[spec.layer_id] = DispatchOnceProgram(
                fixed_waves(
                    local_experts,
                    binding.bank_slots,
                    config.execution_slots // binding.bank_slots,
                ),
                dispatcher,
                local,
                layout,
                local_experts=local_experts,
            )
            prefill[spec.layer_id] = PrefillDispatchOnceProgram(
                dispatcher,
                PrefillLocalExpertProgram(binding.prefill_pipeline, native_f),
                layout,
                local_experts=local_experts,
                pipeline=binding.prefill_pipeline,
            )
        runtime.decode_program = LayerPrograms(decode)
        runtime.prefill_program = LayerPrograms(prefill)
        runtime.initialize(self.model_id)
        return runtime

    def close_unbound(self):
        if self.runtime is not None:
            self.runtime.synchronize()
            self.runtime.close()
            self.runtime = None
        else:
            self.close_weights()
