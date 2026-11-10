"""Eager prefill over two native weight banks, with optional static residency.

Adapted from codex/moe-prefill-opt-20261102 (2dc7903d0b). Each bank is reused
only after its layer's combine. Dispatch never waits for future weights;
each projection joins its own copy immediately before consuming that bank.
Borrowed Hot bytes are restored before decode or cache maintenance can consume
the published cache addresses.
"""

from dataclasses import dataclass

from ...transfer import HostCopyPlan, HostCopyRegion, TransferTicket


@dataclass(frozen=True)
class PrefillLayerTransfer:
    w13: TransferTicket
    w2: TransferTicket


def _projection_copy_plans(storage, bank, layouts, source_slice=slice(None)):
    groups = tuple(
        tuple(name for name in layouts if name.startswith(prefix))
        for prefix in ("w13_", "w2_")
    )
    if not all(groups) or sum(map(len, groups)) != len(layouts):
        raise ValueError("Prefill requires complete W13 and W2 component groups")
    return tuple(
        HostCopyPlan(
            tuple(
                HostCopyRegion(source, destination, 0, 0, layouts[name].nbytes)
                for name in group
                for source, destination in zip(
                    storage.component_capabilities[name][source_slice],
                    bank.component_regions[name],
                    strict=True,
                )
            ),
            (storage, bank),
        )
        for group in groups
    )


def whole_layer_bank_layout(local_experts, hot_slots, execution_slots):
    if local_experts <= 0 or execution_slots < local_experts:
        raise ValueError(
            "Whole-layer prefill requires execution_slots >= local_experts"
        )
    first = ("execution", 0, local_experts)
    if hot_slots >= local_experts:
        return first, ("hot", 0, local_experts)
    if execution_slots >= 2 * local_experts:
        return first, ("execution", local_experts, local_experts)
    raise ValueError(
        "Whole-layer prefill needs a second bank: provide Hot slots >= "
        "local_experts or execution_slots >= 2 * local_experts"
    )


class LayerPrefetchPipeline:
    """Ordered host submission with per-bank device completion dependencies."""

    def __init__(self, layer_ids, *, load, copy_stream):
        self.layer_ids = tuple(layer_ids)
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("Prefill requires unique ordered offloaded layers")
        self.load, self.copy_stream = load, copy_stream
        self.reset()

    def reset(self):
        self.invocation_id = None
        self.current = 0
        self.started = False
        self.ready = {}

    def _require_current(self, layer, context):
        if (
            self.invocation_id is not context.invocation_id
            or self.current >= len(self.layer_ids)
            or self.layer_ids[self.current] != layer
        ):
            raise RuntimeError("Consume prefetched layers once in model order")

    def _submit(self, index, context, reader=None):
        if reader is not None:
            reader.completion.wait_on(self.copy_stream)
        self.ready[index] = self.load(self.layer_ids[index], index % 2, context)

    def begin_forward(self, context):
        if self.invocation_id is not None:
            raise RuntimeError("Reset the previous prefill before beginning a forward")
        self.invocation_id = context.invocation_id
        # Record before any model work so resident layers and attention can hide H2D.
        context.record_compute_completion().completion.wait_on(self.copy_stream)
        self._submit(0, context)
        if len(self.layer_ids) > 1:
            self._submit(1, context)

    def start(self, layer, context):
        if self.invocation_id is None:
            raise RuntimeError("Begin prefill at forward entry before starting a layer")
        self._require_current(layer, context)
        if self.started:
            raise RuntimeError("Prefill layer entry was already consumed")
        # Dispatch consumes tokens, not weights. Waiting here for the next
        # layer would also stall the host at the dispatcher's count readback,
        # serializing this layer's MoE with an unrelated future H2D.
        self.started = True

    def acquire(self, layer, context):
        self._require_current(layer, context)
        if not self.started:
            raise RuntimeError("Start the prefill layer before acquiring weights")
        self.ready[self.current].w13.completion.wait_on(context.compute_stream)
        return self.current % 2

    def wait_for_w2(self, layer, context):
        self._require_current(layer, context)
        if not self.started:
            raise RuntimeError("Start the prefill layer before acquiring weights")
        self.ready[self.current].w2.completion.wait_on(context.compute_stream)

    def retire(self, layer, reader, context):
        self._require_current(layer, context)
        if not self.started:
            raise RuntimeError("Cannot retire a prefill layer before its entry")
        following = self.current + 2
        if following < len(self.layer_ids):
            self._submit(following, context, reader)
        del self.ready[self.current]
        self.current += 1
        self.started = False

    def finish_forward(self, context):
        if self.invocation_id is not None and (
            self.invocation_id is not context.invocation_id
            or self.current != len(self.layer_ids)
        ):
            raise RuntimeError("Prefill ended before every offloaded layer completed")


class AscendPrefillWeights:

    def __init__(self, owner, *, resident_experts=0):
        import torch

        from .native_acl import AclHostTransferOps

        self.owner = owner
        # All readers and the next layer's update run on the compute stream.
        self.counts = torch.empty(
            owner.local_experts, dtype=torch.int64, device=owner.device
        )
        specs = owner.catalog.offloaded_layers
        self.resident_experts = resident_experts
        self.resident_slots = len(specs) * resident_experts
        if resident_experts:
            cold_experts = owner.local_experts - resident_experts
            self.banks = tuple(
                owner.execution.view(
                    self.resident_slots + bank * cold_experts, cold_experts
                )
                for bank in range(2)
            )
            quant_info = owner.execution.quant_info()
            self.quant_info = (quant_info, quant_info)
            self.groups = torch.zeros(
                (owner.execution.capacity, 2), dtype=torch.int64, device=owner.device
            )
            self.borrows_hot = False
        else:
            layout = whole_layer_bank_layout(
                owner.local_experts, owner.hot_slots, owner.execution.capacity
            )
            pools = {"execution": owner.execution, "hot": owner.hot}
            self.banks = tuple(
                pools[pool].view(offset, count) for pool, offset, count in layout
            )
            self.quant_info = tuple(bank.quant_info() for bank in self.banks)
            self.borrows_hot = layout[1][0] == "hot"
        self.hot_dirty = False
        self.host_ops = AclHostTransferOps(
            owner.format.adapter.native_acl, owner._retain
        )
        if resident_experts:
            self._populate_resident(specs)
        self.plans = {}
        for index, spec in enumerate(specs):
            storage, bank = owner.storages[spec.layer_id], self.banks[index % 2]
            plans = _projection_copy_plans(
                storage, bank, owner.format.layouts, slice(resident_experts, None)
            )
            self.plans[spec.layer_id] = plans
            for plan in plans:
                self.host_ops.prepare(plan, owner.device)
        self.pipeline = LayerPrefetchPipeline(
            (spec.layer_id for spec in specs),
            load=self._load,
            copy_stream=owner.weight_copy_stream,
        )
        if self.borrows_hot:
            self._init_hot_restore()

    def _populate_resident(self, specs):
        owner, count = self.owner, self.resident_experts
        stream = owner.weight_copy_stream
        try:
            for index, spec in enumerate(specs):
                bank = owner.execution.view(index * count, count)
                plans = _projection_copy_plans(
                    owner.storages[spec.layer_id],
                    bank,
                    owner.format.layouts,
                    slice(count),
                )
                for plan in plans:
                    self.host_ops.materialize_into(plan, bank, stream)
        finally:
            # Retain descriptors and both allocations through partial submission.
            stream.synchronize()
            self.host_ops.clear_cache()
            owner.context.resources.clear()

    def _init_hot_restore(self):
        import torch

        from ...transfer import CopySourceKind, DeviceCopyPlan
        from .cache_transfer import FixedSlotCacheTransfer
        from .transfer import MemFabricTransferBindings

        owner, count = self.owner, self.owner.local_experts
        self.restore_bindings = MemFabricTransferBindings(
            owner.addresses,
            self.banks[1].component_regions,
            capacity=count,
            hot_generations=owner.state.slot_generation,
            error_status=owner.error,
        )
        self.restore_ops = FixedSlotCacheTransfer(
            self.restore_bindings, retain=owner._retain
        )
        self.restore_plan = DeviceCopyPlan(
            torch.full(
                (count,),
                int(CopySourceKind.HOST),
                dtype=torch.int64,
                device=owner.device,
            ),
            owner.state.slot_to_key[:count],
            torch.arange(count, dtype=torch.int64, device=owner.device),
            owner.state.slot_valid[:count],
            owner.state.slot_generation[:count],
            torch.zeros((), dtype=torch.int32, device=owner.device),
            count,
        )

    def _load(self, layer, bank, context):
        if bank == 1 and self.borrows_hot:
            # Mark before submission, which may enqueue only a partial bundle.
            self.hot_dirty = True
        # Record W13 readiness before enqueueing W2 on the same FIFO stream.
        # GMM1 and activation may then execute while W2 is still being copied.
        w13, w2 = (
            self.host_ops.materialize_into(
                plan, self.banks[bank], self.owner.weight_copy_stream
            )
            for plan in self.plans[layer]
        )
        return PrefillLayerTransfer(w13, w2)

    def weights(self, layer, runner_input):
        import torch
        import triton

        from .small_batch_kernels import prepare_prefill_counts

        owner = self.owner
        counts = runner_input.expert_tokens
        if (
            counts.shape != (owner.local_experts,)
            or counts.dtype != torch.int64
            or not counts.is_contiguous()
        ):
            raise ValueError("Prefill requires one int64 count per local expert")
        if runner_input.group_list_type not in (0, 1):
            raise ValueError("Unsupported prefill group list type")
        # A dedicated prefill worker has no decode-cache usage state.
        state = owner.state
        start = owner.catalog.key(layer, 0)
        group_tokens, group_type, slot_args = self.counts, 1, {}
        if self.resident_experts:
            # The same layer ordinal selects its immutable prefix and the
            # alternating cold bank. Groups stay in dispatch's logical order.
            index = self.pipeline.current
            group_tokens, group_type = self.groups, 2
            slot_args = {
                "Groups": self.groups,
                "ResidentBase": index * self.resident_experts,
                "ColdBase": self.resident_slots + (index % 2) * self.banks[0].capacity,
                "RESIDENT_EXPERTS": self.resident_experts,
            }
        prepare_prefill_counts[(1,)](
            counts,
            owner.error,
            state.update_enabled if state is not None else None,
            (
                state.step_access_count[start : start + owner.local_experts]
                if state is not None
                else None
            ),
            self.counts,
            EXPERTS=owner.local_experts,
            CUMULATIVE=runner_input.group_list_type == 0,
            BLOCK=triton.next_power_of_2(owner.local_experts),
            TRACK_USAGE=state is not None,
            **slot_args,
        )
        runner_input.expert_tokens, runner_input.group_list_type = (
            group_tokens,
            group_type,
        )
        # Keep the dispatch-owned input alive through asynchronous consumption.
        owner.context.retain(counts)
        bank = self.pipeline.acquire(layer, owner.context)
        return self.quant_info[bank]

    def finish_layer(self, layer):
        """Called after the dispatcher has enqueued the complete prefill combine."""
        owner = self.owner
        # The compute stream already waits for the return collective and carries
        # finalization. Gate bank reuse on that frontier, not just the last GMM.
        self.pipeline.retire(
            layer, owner.context.record_compute_completion(), owner.context
        )
        if self.pipeline.current == len(self.pipeline.layer_ids):
            self.restore_hot()

    def restore_hot(self):
        if not self.hot_dirty:
            return
        import torch

        owner = self.owner
        stream = torch.npu.current_stream()
        stream.wait_stream(owner.weight_copy_stream)
        self.restore_plan.count.copy_(self.restore_plan.valid.sum(dtype=torch.int32))
        self.restore_ops.materialize_into(
            self.restore_plan, self.restore_bindings, stream
        )
        self.hot_dirty = False
