"""Assemble native V4 FP4 storage, ACL/MemFabric transport and Ascend execution."""

from __future__ import annotations

import logging
from dataclasses import fields
from types import SimpleNamespace

from ...transfer import (
    CopySourceKind,
    DeviceCopyPlan,
    HostCopyPlan,
    HostCopyRegion,
)
from ..cache_metadata import CacheRouteStats
from ..config import Phase
from ..diagnostics import BatchDiagnostics
from ..execution import (
    CacheUpdateCoordinator,
    ExpertWeightAccess,
    MaterializationPlan,
)
from ..interfaces import MoEBackend
from ..layer_execution import (
    BoundExecutionLayer,
    LayerExecutionRequest,
    whole_layer_bank_layout,
)
from ..layer_prefetch import LayerPrefetchPipeline
from ..models import DeepSeekV4Adapter
from ..residency import TensorResidencyOps
from ..resources import ResourceKind, tensor_allocations
from ..runtime import BoundMoERuntime, DecodeStateSnapshot
from ..state import ResidencyState
from .config import validate_native_environment
from .format_resolver import resolve_npu_format
from .runtime import NpuDeviceRuntime
from .transfer import (
    MemFabricAddressCatalog,
    MemFabricDeviceTransferOps,
    MemFabricTransferBindings,
)

logger = logging.getLogger(__name__)


class NpuMoEBackend(MoEBackend):
    """One model/rank, fixed expert format, single immutable registered master."""

    def __init__(self, settings, config, hf_config, device, *, model_adapter=None):
        import torch

        validate_native_environment()
        self.settings = settings
        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("Native V4 offload requires an explicit NPU device")
        self.model_adapter = model_adapter or DeepSeekV4Adapter()
        self.model_plan = self.model_adapter.plan_model(hf_config, config)
        if len({s.local_experts for s in self.model_plan.offloaded_layers}) != 1:
            raise ValueError(
                "The native execution arena requires uniform local expert counts"
            )
        local_experts = self.model_plan.offloaded_layers[0].local_experts
        whole_layer_bank_layout(
            local_experts, config.hot_slots(self.model_plan), config.execution_slots
        )
        self.format = self.host_pool = self.device_runtime = None
        self.layers = {}
        self.runtime = None
        self._unbound_resources = []
        self._closed = False

    def prepare_expert_weights(
        self,
        layer,
        *,
        method,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        from .format import ExpertLayerStorage
        from .storage import MemFabricHostPool

        layer_id = layer.layer_id
        if self.runtime is not None or self._closed or layer_id in self.layers:
            raise RuntimeError("Expert storage must be registered exactly once")
        spec = self.model_plan.layer(layer_id)
        format_type = resolve_npu_format(layer, method, resident=spec.resident)
        if spec.resident:
            return False
        if (
            num_experts != spec.local_experts
            or layer.moe_tp_size != 1
            or layer.num_fused_shared_experts
        ):
            raise ValueError(
                "Native V4 storage requires unchanged local routed expert ownership"
            )
        dimensions = (hidden_size, intermediate_size_per_partition)
        if self.format is None:
            from .native_acl import get_native_acl

            self.native_acl = get_native_acl(self.settings.native_extension_directory)
            self.device_runtime = NpuDeviceRuntime(self.device)
            # One-expert NZ format probe and conversion workspace, never E experts.
            self.format = format_type(
                *dimensions,
                self.device,
                native_acl=self.native_acl,
                workspace_limit_bytes=self.settings.workspace_limit_bytes,
            )
            self._dimensions = dimensions
            required = sum(
                self.format.host_nbytes(s.local_experts)
                for s in self.model_plan.offloaded_layers
            )
            self.host_pool = MemFabricHostPool(
                required,
                self.device.index,
                physical_bytes=self.settings.host_pool_gib * 1024**3,
                native_acl=self.native_acl,
            )
        elif dimensions != self._dimensions or type(self.format) is not format_type:
            raise ValueError(
                "Offloaded V4 layers must share one physical expert format"
            )
        storage = ExpertLayerStorage(
            layer, spec, self.format, self.host_pool, extra_weight_attrs
        )
        self.layers[layer_id] = storage
        return True

    def prepare_layer(self, module, spec):
        storage = self.layers.get(spec.layer_id)
        if (
            storage is None
            or getattr(
                self.model_adapter.expert_module(module), "_moe_offload_storage", None
            )
            is not storage
        ):
            raise RuntimeError("Routed weights were not intercepted before allocation")
        if storage.spec != spec:
            raise ValueError("Constructed expert ownership differs from the catalog")

    def finalize(self, layers, catalog, config, policy):
        import torch

        from .experts import NPUExpertBank, build_npu_layer_programs
        from .native_acl import AclHostTransferOps

        if self.runtime is not None or self._closed:
            raise RuntimeError("Backend already finalized or closed")
        if catalog != self.model_plan:
            raise ValueError(
                "Constructed layers differ from the pre-allocation model plan"
            )
        if set(self.layers) != {s.layer_id for s in catalog.offloaded_layers}:
            raise ValueError("Incomplete native expert storage registration")
        for storage in self.layers.values():
            storage.require_sealed()
        k = config.hot_slots(catalog)
        banks = whole_layer_bank_layout(
            catalog.offloaded_layers[0].local_experts, k, config.execution_slots
        )
        initial_keys = policy.initial_keys(
            k, getattr(self.settings, "offline_model_id", None)
        )
        state = ResidencyState.allocate(
            catalog.num_experts, k, config.execution_slots, self.device
        )
        error = torch.zeros((), dtype=torch.int32, device=self.device)
        hot = self.format.allocate_bank(k, ResourceKind.HOT_WEIGHTS, "hot")
        execution = self.format.allocate_bank(
            config.execution_slots, ResourceKind.EXECUTION_WEIGHTS, "execution"
        )
        self._unbound_resources.extend((hot, execution))
        component_names = tuple(self.format.layouts)
        hosts = {
            name: tuple(
                cap
                for spec in catalog.offloaded_layers
                for cap in self.layers[spec.layer_id].component_capabilities[name]
            )
            for name in component_names
        }
        address_catalog = MemFabricAddressCatalog(
            hosts,
            hot.component_regions,
            {name: self.format.layouts[name].nbytes for name in component_names},
            device=self.device,
        )
        self._unbound_resources.append(address_catalog)
        bindings = []

        def bind(bank, capacity, name, valid=None, *, fixed_destination_slots=None):
            result = MemFabricTransferBindings(
                address_catalog,
                bank.component_regions,
                capacity=capacity,
                hot_generations=state.slot_generation,
                error_status=error,
                name=name,
                fixed_destination_slots=fixed_destination_slots,
            )
            if valid is not None:
                result.expert_bank = NPUExpertBank(bank.quant_info(), valid, error)
            bindings.append(result)
            return result

        update = bind(hot, k, "hot.update")
        bound_layers = []
        for spec in catalog.offloaded_layers:
            count = spec.local_experts
            transfer_binding = bind(
                execution.view(0, count),
                count,
                f"decode.layer.{spec.layer_id}",
                state.execution_valid[:count],
                fixed_destination_slots=tuple(range(count)),
            )
            bound_layers.append(
                BoundExecutionLayer(spec.layer_id, 0, 0, transfer_binding)
            )
        pools = {"hot": hot, "execution": execution}
        a, b = (pools[pool].view(offset, count) for pool, offset, count in banks)
        prefill_banks = {0: a, 1: b}
        prefill_valid = {
            index: torch.ones(bank.capacity, dtype=torch.bool, device=self.device)
            for index, bank in prefill_banks.items()
        }

        prefill_bindings = {
            index: SimpleNamespace(
                bank=bank,
                bank_id=index,
                error_status=error,
                hot_generations=state.slot_generation,
                expert_bank=NPUExpertBank(
                    bank.quant_info(), prefill_valid[index], error
                ),
            )
            for index, bank in prefill_banks.items()
        }
        layer_plans = {}
        for index, spec in enumerate(catalog.offloaded_layers):
            bank_id = index % 2
            binding = prefill_bindings[bank_id]
            bank, storage = binding.bank, self.layers[spec.layer_id]
            regions = tuple(
                HostCopyRegion(
                    storage.component_capabilities[name][expert],
                    bank.component_regions[name][expert],
                    0,
                    0,
                    self.format.layouts[name].nbytes,
                )
                for name in component_names
                for expert in range(spec.local_experts)
            )
            layer_plans[(spec.layer_id, bank_id)] = MaterializationPlan(
                HostCopyPlan(regions, (storage, bank)), binding
            )

        def prepare_prefill(request, context):
            return layer_plans[(request.layer_id, request.bank_id)]

        cache_stats = CacheRouteStats.allocate(self.device)
        ops = TensorResidencyOps(
            state,
            catalog,
            execution_layers=bound_layers,
            update_bindings=update,
            error_status=error,
            prepare_prefill=prepare_prefill,
            # LFU uses step_access_count independently. Disabled batch logging
            # must also omit its per-layer reductions from the decode graph.
            cache_stats=cache_stats if self.settings.batch_log_interval else None,
        )

        # Native operations retain into the current invocation. Capture heads
        # install a new context before any program.
        def retain(resource):
            self.runtime.context.retain(resource)

        host_ops = AclHostTransferOps(self.native_acl, retain, source_location="host")
        device_ops = MemFabricDeviceTransferOps(retain=retain)
        access = ExpertWeightAccess(ops, host_ops=host_ops, device_ops=device_ops)
        policy_state = policy.allocate_state(catalog.num_experts, self.device)
        targets = torch.full((k,), -1, dtype=torch.int64, device=self.device)
        update_due = torch.zeros((), dtype=torch.bool, device=self.device)
        updates = CacheUpdateCoordinator(
            policy, policy_state, state, targets, access, due_out=update_due
        )
        snapshot = DecodeStateSnapshot(state, policy, policy_state, error_status=error)
        restore_plan = DeviceCopyPlan(
            torch.full(
                (k,), int(CopySourceKind.HOST), dtype=torch.int64, device=self.device
            ),
            torch.full((k,), -1, dtype=torch.int64, device=self.device),
            torch.arange(k, dtype=torch.int64, device=self.device),
            torch.zeros(k, dtype=torch.bool, device=self.device),
            torch.full((k,), -1, dtype=torch.int64, device=self.device),
            torch.zeros((), dtype=torch.int32, device=self.device),
            k,
        )

        def reload_weights(saved, context):
            restore_plan.source_index.copy_(saved.slot_to_key)
            restore_plan.valid.copy_(saved.slot_valid)
            restore_plan.count.copy_(restore_plan.valid.sum(dtype=torch.int32))
            ready = context.record_compute_completion()
            ready.completion.wait_on(context.copy_stream)
            return device_ops.materialize_into(
                restore_plan, update, context.copy_stream
            )

        programs, program_resources = {}, []

        # A/B belong to the arena, so one submitter owns their complete reader
        # history across layers. Layer identity is part of each session/request.
        def profile(name, layer_id, bank_id, context):
            return torch.profiler.record_function(
                f"moe.{context.phase.value}.{name} "
                f"forward={getattr(context, 'forward_sequence', 'capture')} "
                f"layer={layer_id} bank={bank_id} "
                f"experts={catalog.layer(layer_id).local_experts}"
            )

        profile_scope = profile if self.settings.layer_profile else None
        for plan in layer_plans.values():
            host_ops.prepare(plan.copy_plan, self.device)
        pipeline = LayerPrefetchPipeline(
            access,
            [s.layer_id for s in catalog.offloaded_layers],
            make_request=lambda layer, bank, ctx: LayerExecutionRequest(
                layer, None, Phase.PREFILL, bank
            ),
            record_copy_completion=lambda ctx: self.device_runtime.record(
                ctx.copy_stream
            ),
            profile=profile_scope,
        )
        pipelines = (pipeline,)
        shared_layout = None
        for module, spec in layers:
            programs[spec.layer_id] = build_npu_layer_programs(
                self.model_adapter.expert_module(module),
                access=access,
                residency=ops,
                pipeline=pipeline,
                max_dispatch_rows=self.settings.max_dispatch_rows,
                device=self.device,
                shared_layout=shared_layout,
                profile=profile_scope,
            )
            if shared_layout is None:
                shared_layout = programs[spec.layer_id].layout
                program_resources.extend(programs[spec.layer_id].allocations())
        resources = (
            list(self.host_pool.allocations())
            + list(hot.allocations())
            + list(execution.allocations())
        )
        resources += (
            list(state.allocations())
            + list(ops.allocations())
            + list(snapshot.allocations())
            + list(cache_stats.allocations())
        )
        for binding in bindings:
            resources.extend(binding.allocations())
        address_catalog.seal()
        resources.extend(address_catalog.allocations())
        tensors = [
            ("error", error),
            ("target_keys", targets),
            ("update_due", update_due),
        ]
        tensors.extend(
            ("policy." + f.name, getattr(policy_state, f.name))
            for f in fields(policy_state)
        )
        tensors.extend(
            ("restore." + f.name, getattr(restore_plan, f.name))
            for f in fields(restore_plan)
            if f.name != "capacity"
        )
        tensors.extend(
            (f"prefill.valid.{key}", tensor) for key, tensor in prefill_valid.items()
        )
        resources.extend(tensor_allocations("npu.", tensors))
        resources.extend(program_resources)
        diagnostics = None
        if self.device.type == "npu" and getattr(
            self.settings, "batch_log_interval", 1
        ):
            try:
                diagnostics = BatchDiagnostics(
                    cache_stats,
                    error,
                    rank=config.ep_rank,
                    device=self.device,
                    interval=self.settings.batch_log_interval,
                    capacity=self.settings.batch_log_capacity,
                    device_scope=lambda: torch.npu.device(self.device),
                )
                resources.extend(diagnostics.allocations())
            except Exception:
                logger.warning("MoE offload batch logging unavailable", exc_info=True)
        logger.info(
            "MoE execution modes: prefill=%s decode=%s local_experts=%s "
            "execution_weight_bytes=%s prefill_bank_bytes=%s,%s "
            "(banks alias existing pools; full-layer activation peak needs NPU validation)",
            self.settings.prefill_mode,
            self.settings.decode_mode,
            catalog.offloaded_layers[0].local_experts,
            config.execution_slots * self.format.expert_nbytes,
            a.capacity * self.format.expert_nbytes,
            b.capacity * self.format.expert_nbytes,
        )
        logger.info(
            "MoE offload runtime bound: rank=%s device=%s layers=%s experts=%s "
            "hot_slots=%s execution_slots=%s expert_bytes=%s "
            "prefill_A=%s prefill_B=%s batch_log_interval=%s",
            config.ep_rank,
            self.device,
            len(layers),
            catalog.num_experts,
            k,
            config.execution_slots,
            self.format.expert_nbytes,
            a.capacity,
            b.capacity,
            getattr(self.settings, "batch_log_interval", 1) if diagnostics else 0,
        )

        def release():
            # close_after_drain has stopped submissions and drained device reads.
            host_ops.clear_cache()
            layer_plans.clear()
            prefill_bindings.clear()
            self._release()

        self.runtime = BoundMoERuntime(
            device_runtime=self.device_runtime,
            residency=state,
            residency_ops=ops,
            cache_updates=updates,
            cache_stats=cache_stats,
            diagnostics=diagnostics,
            snapshot=snapshot,
            reload_weights=reload_weights,
            layer_programs=programs,
            resources=resources,
            expert_nbytes=self.format.expert_nbytes,
            storage=tuple(self.layers.values()),
            pipelines=pipelines,
            release=release,
        )
        if initial_keys is not None:
            try:
                self.runtime.initialize_hot_cache(initial_keys)
            except BaseException:
                self.runtime.close()
                self.runtime = None
                raise
        return self.runtime

    def _release(self):
        for storage in self.layers.values():
            storage.close()
        if self.host_pool is not None:
            self.host_pool.close()
        self._unbound_resources.clear()
        self._closed = True

    def close_unbound(self):
        if self._closed:
            return
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
            return
        if self.device_runtime is not None:
            self.device_runtime.synchronize()
        self._release()
