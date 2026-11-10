"""Expert weights for the Ascend MoE runner.

Communication and token layout stay in the mainline dispatcher/runner. Decode
stably packs active experts into one execution bank; eager prefill alternates
complete layers across two bank views. Cache publication follows the whole forward.
"""

import logging
import math
from contextlib import contextmanager

import torch

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils.offloader import BaseOffloader

from ...transfer import EventCompletion, TransferTicket
from ..config import Phase
from ..execution import CacheUpdateCoordinator, ExpertWeightAccess
from ..residency import TensorResidencyOps
from ..resources import ResourceKind
from ..state import ResidencyState
from .execution_mode import CachedExecution, PrefillOnlyExecution
from .format import ExpertLayerStorage
from .model_adapter import build_expert_catalog, get_model_id, get_quant_spec
from .native_acl import get_native_acl
from .options import MAX_PHYSICAL_WEIGHT_GROUPS, AscendOffloadOptions
from .storage import MemFabricHostPool
from .transfer import (
    MemFabricAddressCatalog,
    MemFabricTransferBindings,
    _group_pack_copy,
)
from .w8a8_format import NpuV4W8A8Format

logger = logging.getLogger(__name__)


class _Context:
    """A cache transaction on the current stream, retired after its completion."""

    phase = Phase.DECODE
    capturing = False

    def __init__(self):
        self.compute_stream = self.copy_stream = torch.npu.current_stream()
        self.invocation_id = object()
        self.resources = []

    def retain(self, resource):
        self.resources.append(resource)

    retain_completion = retain

    def record_compute_completion(self):
        event = torch.npu.Event()
        event.record(self.compute_stream)
        ticket = TransferTicket(EventCompletion(event))
        self.retain(ticket)
        return ticket


class AscendOpsOffloader(BaseOffloader):
    def __init__(self, hf_config, path):
        p = get_parallel()
        self.model_id = get_model_id(hf_config)
        self.options = AscendOffloadOptions.from_path(
            path,
            model_id=self.model_id,
            disaggregation_mode=get_disagg().disaggregation_mode,
        )
        self.cfg, self.hf_config = self.options.values, hf_config
        moe = get_exec().moe
        backend = moe.moe_a2a_backend
        use_deepep = backend == "deepep" and self.options.prefill_only
        if backend != "ascend_ops" and not use_deepep:
            raise ValueError(
                "Expert offload requires Ascend Ops; DeepEP is supported only "
                "with mode=prefill_only"
            )
        self.execution_mode = (
            PrefillOnlyExecution(
                use_deepep=use_deepep,
                resident_ratio=self.options.prefill_resident_ratio,
            )
            if self.options.prefill_only
            else CachedExecution()
        )
        if (
            p.tp_size != p.moe_ep_size
            or p.moe_tp_size != 1
            or p.moe_dp_size != 1
            or p.nnodes != 1
        ):
            raise ValueError(
                "Expert offload requires single-node stage TP=EP and MoE TP=DP=1"
            )
        self.execution_mode.validate_topology(self)
        if not get_schedule().disable_overlap_schedule:
            raise ValueError(
                "Single execution bank requires --disable-overlap-schedule"
            )
        if (
            moe.enable_eplb
            or moe.ep_num_redundant_experts
            or moe.init_expert_location != "trivial"
            or get_exec().overlap.enable_two_batch_overlap
            or get_exec().overlap.enable_single_batch_overlap
        ):
            raise ValueError(
                "Initial fixed-slot offload requires trivial expert placement and does not support EPLB, redundant experts, or batch overlap"
            )
        self.layer_range = self._stage_layer_range(hf_config, p)
        self.catalog = build_expert_catalog(
            hf_config,
            p.moe_ep_size,
            p.moe_ep_rank,
            offload_all_routed=self.options.prefill_only,
            layer_range=self.layer_range,
        )
        self.local_experts = self.catalog.offloaded_layers[0].local_experts
        self.copy_capacity = self.local_experts
        self.execution_slots = self.cfg.get(
            "execution_slots",
            self.execution_mode.default_execution_slots(self.local_experts),
        )
        # Packed copies use at most 256 local experts on A3.
        if self.copy_capacity > 256:
            raise ValueError("Packed expert copies support at most 256 local experts")
        self.execution_mode.configure(self, path)
        from .prefill import whole_layer_bank_layout

        if not (self.options.prefill_only and self.execution_mode.resident_experts):
            whole_layer_bank_layout(
                self.local_experts, self.hot_slots, self.execution_slots
            )
        self.device = torch.device("npu", get_device().gpu_id)
        self.storages = {}
        self.format = self.host_pool = None
        self.ready = False
        self.step = 0

        logger.info(
            "Ascend expert offload: mode=%s a2a=%s pp=%d/%d layers=[%d,%d) "
            "ep=%d/%d routed_layers=%d local_experts=%d",
            self.options.mode,
            backend,
            p.pp_rank if p.pp_size > 1 else 0,
            p.pp_size,
            *self.layer_range,
            p.moe_ep_rank,
            p.moe_ep_size,
            len(self.catalog.offloaded_layers),
            self.local_experts,
        )

    @staticmethod
    def _stage_layer_range(hf_config, parallel):
        """Use the model's PP partition, rejecting empty/inverted stages early."""
        count = hf_config.num_hidden_layers
        if type(count) is not int or count <= 0:
            raise ValueError("num_hidden_layers must be a positive integer")
        rank = parallel.pp_rank if parallel.pp_size > 1 else 0
        if not 0 <= rank < parallel.pp_size:
            raise ValueError("Invalid PP rank for expert offload")
        spans = [
            get_pp_indices(count, i, parallel.pp_size) for i in range(parallel.pp_size)
        ]
        previous = 0
        for start, end in spans:
            if start != previous or not 0 <= start < end <= count:
                raise ValueError(
                    "Expert offload requires nonempty contiguous PP stages"
                )
            previous = end
        if previous != count:
            raise ValueError("Expert offload PP partition must cover the model")
        return spans[rank]

    def prepare_layer(self, layer, num_experts, hidden_size, intermediate_size, attrs):
        spec = self.catalog.layer(layer.layer_id)
        if spec.resident:
            return False
        if num_experts != spec.local_experts:
            raise ValueError(
                "Offload expects only rank-local routed experts; "
                "use --disable-shared-experts-fusion"
            )
        quant_spec = get_quant_spec(layer.w13_scheme, layer.w2_scheme)
        self.execution_mode.validate_quantization(self, quant_spec)
        if self.model_id != "GLM-5.2" and quant_spec.kind != "w8a8":
            raise ValueError(
                "DeepSeek V4 offload currently supports ModelSlim W8A8 only"
            )
        if self.format is None:
            native = get_native_acl(self.cfg.get("native_extension_directory"))
            self.quant_spec = quant_spec
            if quant_spec.kind == "w4a8":
                from .w4a8_format import NpuW4A8Format

                self.format = NpuW4A8Format(
                    hidden_size,
                    intermediate_size,
                    self.device,
                    native_acl=native,
                    group_size=quant_spec.group_size,
                    tp_size=quant_spec.tp_size,
                    activation_use_clip=quant_spec.activation_use_clip,
                )
            else:
                self.format = NpuV4W8A8Format(
                    hidden_size, intermediate_size, self.device, native_acl=native
                )
            total = sum(
                self.format.host_nbytes(s.local_experts)
                for s in self.catalog.offloaded_layers
            )
            host_gib = self.cfg.get("host_pool_gib", 64)
            if type(host_gib) is not int or host_gib <= 0:
                raise ValueError("host_pool_gib must be a positive integer per rank")
            if total > host_gib * 1024**3:
                raise ValueError(
                    f"{self.model_id} {quant_spec.kind} needs at least "
                    f"{math.ceil(total / 1024**3)} GiB host_pool_gib per rank "
                    f"for expert storage; configured {host_gib} GiB"
                )
            self.host_pool = MemFabricHostPool(
                total,
                self.device.index,
                physical_bytes=host_gib * 1024**3,
                native_acl=native,
            )
        if quant_spec != self.quant_spec:
            raise ValueError(
                "This execution bank requires identical expert quantization"
            )
        if (hidden_size, intermediate_size) != (
            self.format.hidden_size,
            self.format.intermediate_size,
        ):
            raise ValueError("This execution bank requires identical expert shapes")
        storage = ExpertLayerStorage(layer, spec, self.format, self.host_pool, attrs)
        self.storages[layer.layer_id] = storage
        layer._ascend_ops_offloader = self
        return True

    def post_init(self):
        if set(self.storages) != {s.layer_id for s in self.catalog.offloaded_layers}:
            raise RuntimeError("Not all planned expert layers were intercepted")
        for storage in self.storages.values():
            storage.require_sealed()
        self.execution_mode.post_init(self)

    @staticmethod
    def _new_context():
        return _Context()

    def _init_cached_runtime(self):
        # The physical-slot GMM API accepts at most 1024 weight groups.
        # Extra execution slots retain the packed path: direct Hot addressing
        # expects the Hot bank to start immediately after the local experts.
        self.direct_hot = (
            self.quant_spec.supports_direct_hot
            and get_parallel().moe_ep_size == 4
            and not get_spec().speculative_algorithm
            and self.execution_slots == self.copy_capacity
            and self.copy_capacity + self.hot_slots <= MAX_PHYSICAL_WEIGHT_GROUPS
        )
        self.execution_keys = None
        if self.direct_hot:
            self.union = self.format.allocate_bank(
                self.copy_capacity + self.hot_slots,
                ResourceKind.EXECUTION_WEIGHTS,
                "ascend.union",
            )
            self.execution = self.union.view(0, self.copy_capacity)
            self.hot = self.union.view(self.copy_capacity, self.hot_slots)
            self.direct_groups = torch.zeros(
                (self.copy_capacity + self.hot_slots, 2),
                dtype=torch.int64,
                device=self.device,
            )
            self.execution_keys = torch.full(
                (self.copy_capacity,), -1, dtype=torch.int64, device=self.device
            )
            self.quant_info = self.union.quant_info()
        else:
            self.execution = self.format.allocate_bank(
                self.execution_slots, ResourceKind.EXECUTION_WEIGHTS, "ascend.execution"
            )
            self.hot = self.format.allocate_bank(
                self.hot_slots, ResourceKind.HOT_WEIGHTS, "ascend.hot"
            )
            self.quant_info = self.execution.view(0, self.copy_capacity).quant_info()
        names = tuple(self.format.layouts)
        hosts = {
            name: tuple(
                cap
                for s in self.catalog.offloaded_layers
                for cap in self.storages[s.layer_id].component_capabilities[name]
            )
            for name in names
        }
        self.addresses = MemFabricAddressCatalog(
            hosts,
            self.hot.component_regions,
            {n: self.format.layouts[n].nbytes for n in names},
            device=self.device,
        )
        self.sources = torch.stack(
            [torch.cat(pair) for pair in self.addresses.source_tables]
        )
        self.destinations = torch.tensor(
            [
                [
                    r.address
                    for r in self.execution.component_regions[n][: self.copy_capacity]
                ]
                for n in names
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self.lengths = torch.tensor(
            [[self.format.layouts[n].nbytes] * self.copy_capacity for n in names],
            dtype=torch.int32,
            device=self.device,
        )
        self._init_copy_plan(names)
        self.error = torch.zeros((), dtype=torch.int32, device=self.device)
        self.state = ResidencyState.allocate(
            self.catalog.num_experts, self.hot_slots, self.local_experts, self.device
        )
        self.policy_state = self.policy.allocate_state(
            self.catalog.num_experts, self.device
        )
        self.target_keys = torch.tensor(
            self.initial_keys, dtype=torch.int64, device=self.device
        )
        bindings = MemFabricTransferBindings(
            self.addresses,
            self.hot.component_regions,
            capacity=self.hot_slots,
            hot_generations=self.state.slot_generation,
            error_status=self.error,
        )
        self.residency = TensorResidencyOps(
            self.state,
            self.catalog,
            execution_waves=[],
            update_bindings=bindings,
            error_status=self.error,
        )
        self.context = _Context()
        from .cache_transfer import FixedSlotCacheTransfer

        device_ops = FixedSlotCacheTransfer(bindings, retain=self._retain)
        self.access = ExpertWeightAccess(self.residency, device_ops=device_ops)
        self.due = torch.zeros((), dtype=torch.bool, device=self.device)
        self.coordinator = CacheUpdateCoordinator(
            self.policy,
            self.policy_state,
            self.state,
            self.target_keys,
            self.access,
            due_out=self.due,
        )
        self.state.update_enabled.fill_(True)
        self.access.apply_update(
            self.target_keys, self.context
        ).completion.synchronize()
        self._publish_sources()
        from .cache_update import FusedCacheUpdate

        self.cache_update = FusedCacheUpdate(self)
        self._check_error()
        self.context.resources.clear()
        self.state.update_enabled.fill_(False)  # warmup/capture never trains LFU
        self._init_decode_mask()
        from .prefill import AscendPrefillWeights

        self.prefill = AscendPrefillWeights(self)
        self._capture_cache_tails()
        self.ready = True
        logger.info(
            "Ascend expert offload: local=%d hot=%d/%d execution=%.3f GiB host=%.3f GiB; "
            "W2 overlap=%s, fused metadata=%s, Hot direct/execution reuse=%s",
            self.local_experts,
            self.hot_slots,
            self.catalog.num_experts,
            self.format.expert_nbytes * self.execution_slots / 1024**3,
            self.host_pool.used_bytes / 1024**3,
            True,
            True,
            self.direct_hot,
        )

    def _retain(self, value):
        self.context.retain(value)

    def _init_decode_mask(self):
        parallel = get_parallel()
        if parallel.attn_cp_size != 1:
            raise ValueError("Decode row mask requires attention CP=1")
        if parallel.tp_size > parallel.attn_dp_size and parallel.disable_attn_tp_gather:
            # That opt-out leaves the runner count replicated, while a padded
            # MoE boundary can still split rows over attention TP.
            raise ValueError(
                "Mixed offload with attention TP>1 requires attention-TP gather; "
                "remove --disable-attn-tp-gather"
            )
        capacity = self.cfg.get("decode_mask_capacity", 128)
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("decode_mask_capacity must be positive")
        # Every captured bucket shares these allocations. Only contents change.
        self.decode_row_indices = torch.arange(
            capacity, dtype=torch.int32, device=self.device
        )
        self.decode_active_rows = torch.zeros(
            capacity, dtype=torch.bool, device=self.device
        )

    def decode_row_mask(self, padded_bs):
        if self.decode_active_rows is None:
            return None
        if padded_bs > self.decode_active_rows.numel():
            raise ValueError("MC2 padded batch exceeds decode_mask_capacity")
        get_bound = (
            get_forward_context().moe_num_token_non_padded
            if has_forward_context()
            else None
        )
        bound = get_bound() if get_bound is not None else None
        if bound is None:
            raise RuntimeError("Decode row mask requires a MoE-local valid token count")
        mask = self.decode_active_rows[:padded_bs]
        # Resolve after eager padding / graph bucket selection, exactly as top-k
        # does. The comparison is captured, so replay reads the updated LOCAL
        # count without a host read or a Python callback. A zero-source rank
        # still participates in MC2 and may receive other ranks' expert work.
        torch.lt(self.decode_row_indices[:padded_bs], bound, out=mask)
        return mask

    def start_prefill_layer(self, layer_id):
        self.prefill.pipeline.start(layer_id, self.context)

    def prepare_weights(self, layer_id, runner_input):
        if not self.ready:
            raise RuntimeError("Expert bank is not initialized")
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch

        if get_is_extend_in_batch():
            return self.prefill.weights(layer_id, runner_input)
        return self.execution_mode.prepare_decode_weights(self, layer_id, runner_input)

    def _launch_copies(self, layer_id, copies):
        ready, first, second = self.copy_events[layer_id]
        compute = torch.npu.current_stream()
        # This frontier also follows the previous layer's GMM2, so neither the
        # execution weights nor their packed counts are overwritten while read.
        ready.record(compute)
        with torch.npu.stream(self.weight_copy_stream):
            self.weight_copy_stream.wait_event(ready)
            self._submit_copies(copies[0])
            first.record(self.weight_copy_stream)
            self._submit_copies(copies[1])
            second.record(self.weight_copy_stream)
        compute.wait_event(first)

    def _init_copy_plan(self, names):
        # Checkpoint-only W4 metadata is not read by GMM. W8 offsets are
        # separately validated as zero and remain valid without decode copies.
        selected = []
        for i, name in enumerate(names):
            if self.format.layouts[name].runtime_name is None:
                continue
            if name.endswith("_offset"):
                self.execution.tensors[name].zero_()
            else:
                selected.append(i)
        phases = [
            [i for i in selected if names[i].startswith(prefix)]
            for prefix in ("w13", "w2")
        ]
        self.weight_copy_stream = torch.npu.Stream(device=self.device)
        self.copy_events = {
            s.layer_id: tuple(torch.npu.Event() for _ in range(3))
            for s in self.catalog.offloaded_layers
        }
        self.fused_components = tuple(selected)
        # Encode the selected table rows as one integer. The field width is
        # derived from the schema, so no component index is truncated.
        self.fused_component_bits = max(1, (len(names) - 1).bit_length())
        self.fused_component_map = sum(
            component << (position * self.fused_component_bits)
            for position, component in enumerate(selected)
        )
        if self.direct_hot:
            self._init_direct_copy_plan(phases)
            return
        self.fused_sources = torch.empty(
            (len(selected), self.copy_capacity),
            dtype=torch.int64,
            device=self.device,
        )
        self.copy_groups = torch.empty(
            self.copy_capacity, dtype=torch.int64, device=self.device
        )
        self.copy_scan = torch.tensor(
            self.copy_capacity, dtype=torch.int32, device=self.device
        )
        self.fused_descriptors = []
        for phase in phases:
            descriptors = []
            for component in phase:
                # Each component packs the same expert sequence into its own
                # destination prefix. Flattening components together would put
                # the scale's first output at slot M of the weight address table.
                packed = torch.zeros_like(self.copy_groups)
                descriptors.append(
                    (
                        self.fused_sources[selected.index(component)],
                        self.destinations[component],
                        self.lengths[component],
                        self.copy_scan,
                        self.copy_groups,
                        packed,
                    )
                )
            self.fused_descriptors.append(descriptors)
        # GMM reads the first W13 copy's output after the W13 event. All other
        # calls own separate outputs, especially W2 copies overlapping GMM1.
        self.packed_counts = self.fused_descriptors[0][0][-1]

    def _init_direct_copy_plan(self, phases):
        # Direct Hot GMM indices address fixed physical slots. Keep one copy
        # group per slot, including zero-byte copies, so misses are never packed.
        self.fixed_groups = torch.ones(256, dtype=torch.int64, device=self.device)
        self.fused_sources = torch.empty(
            len(self.fused_components) * self.copy_capacity,
            dtype=torch.int64,
            device=self.device,
        )
        self.fused_lengths = torch.empty_like(self.fused_sources, dtype=torch.uint32)
        self.fused_counts = torch.empty(
            self.copy_capacity, dtype=torch.int64, device=self.device
        )
        self.fused_descriptors = []
        chunk = max(1, 256 // self.copy_capacity)
        offset = 0
        for phase in phases:
            descriptors = []
            for begin in range(0, len(phase), chunk):
                components = torch.tensor(
                    phase[begin : begin + chunk], device=self.device
                )
                count = components.numel() * self.copy_capacity
                descriptors.append(
                    (
                        self.fused_sources[offset : offset + count],
                        self.destinations.index_select(0, components).flatten(),
                        self.fused_lengths[offset : offset + count],
                        torch.tensor(count, dtype=torch.uint32, device=self.device),
                        self.fixed_groups[:count],
                        torch.zeros(count, dtype=torch.int64, device=self.device),
                    )
                )
                offset += count
            self.fused_descriptors.append(descriptors)

    def _prepare_fused(self, layer_id, runner_input):
        import triton

        from .small_batch_kernels import prepare_fixed_copies, prepare_packed_copies

        counts = runner_input.expert_tokens
        if (
            counts.shape != (self.local_experts,)
            or counts.dtype != torch.int64
            or not counts.is_contiguous()
        ):
            raise ValueError(
                "Fused copy metadata expects one int64 count per local expert"
            )
        if runner_input.group_list_type not in (0, 1):
            raise ValueError("Unsupported group list type")
        start = self.catalog.key(layer_id, 0)
        end = start + self.local_experts
        if self.direct_hot:
            prepare_fixed_copies[(1,)](
                counts,
                self.error,
                self.state.update_enabled,
                self.state.step_access_count[start:end],
                self.published_sources[:, start:end],
                self.lengths,
                self.fused_counts,
                self.fused_sources,
                self.fused_lengths,
                EXPERTS=self.local_experts,
                SOURCE_STRIDE=self.catalog.num_experts,
                START=0,
                COMPONENT_COUNT=len(self.fused_components),
                SKIP_OFFSETS=True,
                COMPONENT_MAP=self.fused_component_map,
                COMPONENT_BITS=self.fused_component_bits,
                CUMULATIVE=runner_input.group_list_type == 0,
                BLOCK=triton.next_power_of_2(self.local_experts),
                DIRECT_HOT=True,
                HotSlots=self.state.key_to_slot,
                HOT_CAPACITY=self.hot_slots,
                Groups=self.direct_groups,
                ExecutionKeys=self.execution_keys,
                KEY_START=start,
            )
            runner_input.expert_tokens = self.direct_groups
            runner_input.group_list_type = 2
        else:
            prepare_packed_copies[(1,)](
                counts,
                self.error,
                self.state.update_enabled,
                self.state.step_access_count[start:end],
                self.published_sources[:, start:end],
                self.copy_groups,
                self.fused_sources,
                EXPERTS=self.local_experts,
                SOURCE_STRIDE=self.catalog.num_experts,
                START=0,
                COMPONENT_COUNT=len(self.fused_components),
                SKIP_OFFSETS=True,
                COMPONENT_MAP=self.fused_component_map,
                COMPONENT_BITS=self.fused_component_bits,
                CUMULATIVE=runner_input.group_list_type == 0,
                BLOCK=triton.next_power_of_2(self.local_experts),
            )
            # Dispatch already sorts rows by local expert. Removing zero-token
            # groups preserves row order, including the original combine metadata.
            runner_input.expert_tokens = self.packed_counts
            runner_input.group_list_type = 1
        self._launch_copies(layer_id, self.fused_descriptors)
        return self.quant_info

    def _submit_copies(self, descriptors):
        for source, dst, lengths, scan, groups, packed in descriptors:
            # Packed GMM reads this fixed-capacity output, whose unused tail
            # the copy kernel leaves untouched. Direct Hot GMM uses separate
            # physical-slot metadata and never reads the copy output.
            if not self.direct_hot:
                packed.zero_()
            status = _group_pack_copy(
                source, dst, lengths, scan, groups, packed, self.device
            )
            if status is not None and status != 0:
                raise RuntimeError(f"Expert copy submission failed: {status}")

    def wait_for_w2(self, layer_id):
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch

        if get_is_extend_in_batch():
            self.prefill.pipeline.wait_for_w2(layer_id, self.context)
            return
        torch.npu.current_stream().wait_event(self.copy_events[layer_id][2])

    def _publish_sources(self):
        # Cache placement is immutable throughout a forward. Publish one stable
        # device table after initial fill / cache maintenance; graph replays read
        # these addresses without rebuilding forty identical placement queries.
        keys = torch.arange(self.catalog.num_experts, device=self.device)
        slots = self.state.key_to_slot
        safe = slots.clamp(0, self.hot_slots - 1)
        hit = slots >= 0
        coherent = self.state.slot_valid.to(torch.int32).gather(0, safe).bool() & (
            self.state.slot_to_key.gather(0, safe) == keys
        )
        bad = (slots < -1) | (slots >= self.hot_slots) | (hit & ~coherent)
        self.error.bitwise_or_(bad.any().to(torch.int32))
        indices = torch.where(hit, safe + self.catalog.num_experts + 2, keys + 1)
        values = self.sources.gather(
            1, indices.unsqueeze(0).expand(self.sources.shape[0], -1)
        )
        if not hasattr(self, "published_sources"):
            self.published_sources = torch.empty_like(values)
        self.published_sources.copy_(values)

    def _cache_tail(self, maintenance):
        if maintenance:
            self.policy.observe(
                self.policy_state,
                self.state.step_access_count,
                self.state.update_enabled,
                self.due,
            )
            self.policy.select_hot_set(
                self.policy_state, self.state.key_to_slot, self.target_keys
            )
            self.policy.finish_step(self.policy_state)
            self.cache_update.apply()
        else:
            self.coordinator.emit_step_end(self.context, maintenance=maintenance)
        if maintenance:
            self._publish_sources()

    def _capture_cache_tails(self):
        # Capture with updates disabled: neither usage nor placement may change
        # during warmup. Each graph owns a separate pool and retains all raw
        # pointer descriptors for its entire lifetime.
        self.cache_tail_graphs = {}
        self.cache_tail_contexts = {}
        previous = self.context
        for maintenance in (False, True):
            self.context = _Context()
            self._cache_tail(maintenance)
            self.context.record_compute_completion().completion.synchronize()
            self.context.resources.clear()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                self.context = _Context()
                self.context.capturing = True
                self._cache_tail(maintenance)
            torch.npu.synchronize()
            self.cache_tail_graphs[maintenance] = graph
            self.cache_tail_contexts[maintenance] = self.context
        self.context = previous
        self._check_error()

    def _check_error(self):
        flag = self.error.clone()
        torch.distributed.all_reduce(
            flag,
            op=torch.distributed.ReduceOp.MAX,
            group=get_moe_ep_group().device_group,
        )
        code = int(flag.item())
        if code:
            raise RuntimeError(
                f"Expert offload device fault (all EP ranks stopped): {code}"
            )

    def forward_scope(self, forward_batch=None):
        return self.execution_mode.forward_scope(self, forward_batch)

    @contextmanager
    def _cached_forward_scope(self, forward_batch=None):
        if not self.ready:
            yield
            return
        self.context = _Context()
        self.state.step_access_count.zero_()
        self.state.update_enabled.fill_(True)
        try:
            # RuntimeContext's flag is installed later by _forward_raw; use this
            # batch's global phase, including locally idle/decode attention ranks.
            if forward_batch is not None and forward_batch.is_extend_in_batch:
                # Prefill overwrites execution banks independently of decode's
                # reuse keys. Invalidate before any prefetch, including on idle
                # attention ranks, so the next decode cannot reuse stale bytes.
                if self.execution_keys is not None:
                    self.execution_keys.fill_(-1)
                self.prefill.pipeline.begin_forward(self.context)
            yield
            self.prefill.pipeline.finish_forward(self.context)
            self.step += 1
            maintenance = self.step % self.policy.config.update_interval == 0
            self.cache_tail_graphs[maintenance].replay()
            # One error check per whole forward, outside the captured graph.
            # It also retires cache transactions before shared-bank reuse.
            self._check_error()
        finally:
            try:
                self.prefill.restore_hot()
            finally:
                self.context.compute_stream.wait_stream(self.weight_copy_stream)
                self.context.record_compute_completion().completion.synchronize()
                self.context.resources.clear()
                self.prefill.pipeline.reset()
                self.state.update_enabled.fill_(False)
