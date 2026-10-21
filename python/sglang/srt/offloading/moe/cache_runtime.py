"""Cache-side runtime wiring with injected accelerator programs and transport.

This is not an NPU backend. Native expert kernels, dispatch decoding, physical
bindings, streams and device-plan copies are supplied at backend.finalize().
The same orchestration is exercised by a delayed CPU transport in unit tests.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace

from ..transfer import EventCompletion, TransferTicket
from .cache_metadata import CacheMetadataAdapter
from .cache_residency import TensorResidencyOps
from .config import Phase
from .interfaces import BoundMoERuntime, ForwardCompletion, LocalExpertProgram
from .resources import Allocation, ResourceKind


@dataclass(frozen=True)
class CacheExecutionContext:
    device_module: object
    compute_stream: object
    copy_stream: object

    def fence(self):
        event = self.device_module.Event()
        event.record(self.compute_stream)
        return TransferTicket(EventCompletion(event), (event,))


@dataclass(frozen=True)
class LocalRoutes:
    """Normalized post-dispatch assignments; IDs are local to this rank/layer."""

    expert_ids: object
    valid: object


class CacheLocalExpertProgram(LocalExpertProgram):
    """Concrete D -> count -> prepare bank -> native F link for ComposedDFCProgram.

    route_adapter(layer, dispatch_output, wave) -> LocalRoutes is format-specific.
    run_experts keeps the existing LocalExpertProgram.run argument signature; it
    reads the fixed bank selected by wave.bank and must join readers on compute.
    It must not repeat routing statistics or weight preparation.
    """

    def __init__(self, runtime, route_adapter, run_experts):
        self.runtime = runtime
        self.route_adapter = route_adapter
        self.run_experts = run_experts

    def run(self, layer, dispatch_output, wave, context):
        routes = self.route_adapter(layer, dispatch_output, wave)
        self.runtime.prepare_local_experts(layer.layer_id, routes, wave)
        return self.run_experts(layer, dispatch_output, wave, context)


class CacheRuntime(BoundMoERuntime):
    """Concrete cache lifecycle; hardware capabilities must be supplied honestly.

    backend.finalize creates this, builds decode_program with
    CacheLocalExpertProgram, initializes hot weights, then returns the runtime.
    install_programs binds those programs to the model (native backend concern).
    The baseline serializes copies against readers; pipelined overlap can replace
    the scheduling after the transport contract is validated.
    """

    def __init__(
        self,
        *,
        catalog,
        residency,
        policy,
        transfer,
        context,
        hot_bindings,
        execution_bindings,
        bank_slots,
        capabilities,
        expert_nbytes,
        resources,
        prefill_program,
        install_programs,
        borrow_hot_arena,
    ):
        import torch

        if bank_slots <= 0 or residency.execution_key.numel() % bank_slots:
            raise ValueError("Execution slots must contain whole fixed-size banks")
        self.capabilities = capabilities
        self.expert_nbytes = expert_nbytes
        self.residency = residency
        self.cache = CacheMetadataAdapter(catalog, residency, policy)
        self.residency_ops = TensorResidencyOps(residency)
        self.transfer = transfer
        self.context = context
        self.hot_bindings = hot_bindings
        self.execution_bindings = execution_bindings
        self.bank_slots = bank_slots
        self.prefill_program = prefill_program
        self.decode_program = None  # Bound before install_model_hooks.
        self._install_programs = install_programs
        self._borrow_hot_arena = borrow_hot_arena
        self._enabled = torch.zeros_like(residency.update_enabled)
        self._demand = torch.full_like(residency.execution_key, -1)
        self._layers = {
            layer.layer_id: (catalog.key(layer.layer_id, 0), layer.local_experts)
            for layer in catalog.offloaded_layers
        }
        extra = []
        for name, tensor in (("enabled", self._enabled), ("demand", self._demand)):
            extra.append(
                Allocation(
                    f"cache_runtime.{name}",
                    ResourceKind.METADATA,
                    tensor.untyped_storage().nbytes(),
                    (str(tensor.device), tensor.untyped_storage().data_ptr()),
                    tensor,
                )
            )
        self.resources = (
            tuple(resources)
            + residency.allocations()
            + self.cache.allocations()
            + self.residency_ops.allocations()
            + tuple(extra)
        )
        self._runner = None
        self._phase = Phase.DECODE
        self._initialized = False
        self._tickets = []
        self._capture_tickets = []
        self._capturing = False
        self._serving_context = context
        # Host schedule mirror, never read back from a device tensor. Advance
        # once per successful serving forward, including graph replay. Prefill
        # and capture leave it unchanged, just like the saved device LFU cycle.
        self._step_mod = 0
        self._adjust_eager = False

    def _materialize(self, plan, bindings):
        # Includes plan production and all previous native F readers. Thus reused
        # plans/banks cannot be overwritten while an earlier copy/F reads them.
        produced = self.context.fence()
        produced.completion.wait_on(self.context.copy_stream)
        ticket = self.transfer.materialize_into(
            plan, bindings, self.context.copy_stream
        )
        tickets = self._capture_tickets if self._capturing else self._tickets
        tickets.extend((produced, ticket))
        return ticket

    def initialize(self, model_id):
        """Host-only initial TopK -> actual copy -> publish, before serving/capture."""
        import torch

        if self._initialized:
            raise RuntimeError("Cache weights are already initialized")
        keys = self.cache.policy.initial_keys(
            self.residency.slot_to_key.numel(), model_id
        )
        with self.context.device_module.stream(self.context.compute_stream):
            plan = self.residency_ops.prepare_refill(
                torch.tensor(
                    keys, dtype=torch.int64, device=self.residency.key_to_slot.device
                )
            )
            ticket = self._materialize(plan.copies, self.hot_bindings)
            self.residency_ops.commit_update(plan, ticket, self.context)
            self.context.fence().completion.synchronize()
        self._tickets.clear()
        self._initialized = True

    def install_model_hooks(self, offloader, model_runner):
        if (
            self._runner is not None
            or not self._initialized
            or self.decode_program is None
        ):
            raise RuntimeError("Initialize weights and programs before installing once")
        if (
            self.capabilities.full_decode_graph
            and not self.transfer.supports_device_plans
        ):
            raise ValueError("Full decode graph requires device-plan transfer support")
        self._install_programs(model_runner, self.prefill_program, self.decode_program)
        model_runner.capture_head_hooks.append(self._capture_head)
        model_runner.capture_tail_hooks.append(self._capture_tail)
        self._runner = model_runner

    def _capture_head(self, runner, forward_batch, num_tokens):
        # _enabled is zero outside serving. Do not capture a fill_(False): the
        # serving host writes this fixed-address input before each real replay.
        self._capturing = True
        self.context = replace(
            self._serving_context,
            compute_stream=self.context.device_module.current_stream(),
        )
        self.emit_step_begin(self.context)

    def _capture_tail(self, runner, output, forward_batch, num_tokens):
        try:
            self.emit_step_end(self.context)
        finally:
            self._capturing = False
            self.context = self._serving_context

    @contextmanager
    def forward_scope(self, owner, forward_batch):
        mode = getattr(forward_batch, "global_forward_mode", None)
        mode = forward_batch.forward_mode if mode is None else mode
        if not (mode.is_decode() or mode.is_idle() or mode.name in ("EXTEND", "MIXED")):
            raise NotImplementedError(f"Cache runtime does not support {mode.name}")
        counts = getattr(forward_batch, "original_global_num_tokens_cpu", None)
        if counts is None:
            counts = getattr(forward_batch, "global_num_tokens_cpu", None)
        has_tokens = (
            any(n > 0 for n in counts)
            if counts is not None
            else (forward_batch.input_ids.numel() > 0)
        )
        # global_forward_mode is TBO-only and is normally None here. The
        # scheduler's is_extend_in_batch is shared across ranks even when a
        # local rank is IDLE or DECODE while peers are processing prefill.
        if not has_tokens:
            phase = owner.phase
        elif getattr(forward_batch, "is_extend_in_batch", False) or mode.name in (
            "EXTEND",
            "MIXED",
        ):
            phase = Phase.PREFILL
        else:
            phase = Phase.DECODE
        enabled = has_tokens and phase is Phase.DECODE
        next_step = (self._step_mod + 1) % self.cache.policy.config.update_interval
        token = owner.begin_forward(phase, self.context.compute_stream)
        self._phase = phase
        self._adjust_eager = enabled and next_step == 0
        try:
            with self.context.device_module.stream(self.context.compute_stream):
                self._enabled.fill_(enabled)
                yield
                # Fresh events after eager submission/replay, not capture tickets.
                fence = self.context.fence()
                complete = TransferTicket(
                    fence.completion, (fence, tuple(self._tickets))
                )
                owner.finish_forward(
                    token, ForwardCompletion(complete, complete, complete)
                )
                if enabled:
                    self._step_mod = next_step
                self._tickets.clear()
        except BaseException:
            owner.fail_forward(token)
            raise
        finally:
            with self.context.device_module.stream(self.context.compute_stream):
                self._enabled.zero_()
            self._capturing = False
            self._adjust_eager = False
            self.context = self._serving_context

    @contextmanager
    def eager_step_scope(self, forward_batch):
        # ModelRunner enters only around an actual eager model invocation.
        # Captured decode has its own head/tail, so replay never double-counts.
        if self._phase is Phase.DECODE:
            self.emit_step_begin(self.context)
        yield
        if self._phase is Phase.DECODE:
            self.emit_step_end(self.context, adjust=self._adjust_eager)

    def emit_step_begin(self, context):
        self.cache.begin_step(self._enabled)

    def emit_layer_start(self, layer_id, context):
        # Prefill prefetch belongs to the supplied prefill program.
        pass

    def emit_after_router(self, layer_id, routing, context):
        # Actual rank-local assignments are recorded AFTER dispatch below.
        pass

    def prepare_local_experts(self, layer_id, routes, wave):
        import torch

        offset, size = self._layers[layer_id]
        width = wave.local_end - wave.local_start
        start = wave.bank * self.bank_slots
        if not (
            0 <= wave.local_start < wave.local_end <= size
            and width <= self.bank_slots
            and 0 <= start
            and start + self.bank_slots <= self._demand.numel()
        ):
            raise ValueError("Wave does not fit the declared execution bank")
        self.cache.record_local_routes(
            layer_id,
            routes.expert_ids,
            routes.valid,
            local_start=wave.local_start,
            local_end=wave.local_end,
        )
        ids = routes.expert_ids.reshape(-1)
        active = (
            routes.valid.reshape(-1)
            & (ids >= wave.local_start)
            & (ids < wave.local_end)
        )
        # Count assignments for LFU above; deduplicate only the load demand.
        counts = torch.zeros(size + 1, dtype=torch.int64, device=ids.device)
        counts.scatter_add_(
            0, torch.where(active, ids.to(torch.int64), size), active.to(torch.int64)
        )
        experts = torch.arange(wave.local_start, wave.local_end, device=ids.device)
        self._demand.fill_(-1)
        self._demand[start : start + width].copy_(
            torch.where(counts[experts] > 0, experts + offset, -1)
        )
        plan = self.residency_ops.resolve_execution(self._demand, self.context)
        ticket = self._materialize(plan, self.execution_bindings)
        self.residency_ops.publish_execution(ticket, self.context)

    def emit_step_end(self, context, *, adjust=True):
        # adjust is a host scheduling decision for eager only. Captured graphs
        # retain all operations and gate actual swaps using device update_due.
        self.cache.finish_step(select_hot_set=adjust)
        if not adjust:
            return
        plan = self.residency_ops.prepare_update(self.cache.target_keys, context)
        ticket = self._materialize(plan.copies, self.hot_bindings)
        self.residency_ops.commit_update(plan, ticket, context)

    def snapshot_decode_state(self):
        with self.context.device_module.stream(self.context.compute_stream):
            self.cache.save_decode_metadata()
            return self.context.fence()

    def borrow_hot_arena(self):
        self._borrow_hot_arena()

    def restore_decode_state(self):
        with self.context.device_module.stream(self.context.compute_stream):
            plan = self.residency_ops.prepare_refill(self.cache.snapshot.slot_to_key)
            ticket = self._materialize(plan.copies, self.hot_bindings)
            ticket.completion.wait_on(self.context.compute_stream)
            self.cache.restore_decode_metadata_after_refill()
            return self.context.fence()

    def synchronize(self):
        self.context.device_module.synchronize()
        self._tickets.clear()

    def close(self):
        # Owner synchronizes and releases graph leases before calling close.
        if self._runner is not None:
            self._runner.capture_head_hooks.remove(self._capture_head)
            self._runner.capture_tail_hooks.remove(self._capture_tail)
            self._runner = None
        self._tickets.clear()
        self._capture_tickets.clear()
