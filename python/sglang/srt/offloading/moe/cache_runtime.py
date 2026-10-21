"""Cache-side runtime wiring with injected accelerator programs and transport.

CacheBackend assembles the model/dispatch/native-F adapters at finalize().
Physical bindings, streams and device-plan copies come from the weight backend.
The same orchestration is exercised by a delayed CPU transport in unit tests.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

from ..transfer import EventCompletion, JoinedCompletion, TransferTicket
from .cache_metadata import CacheMetadataAdapter
from .cache_residency import TensorResidencyOps
from .config import Phase
from .execution import CacheUpdateCoordinator, ExecutionRequest, ExpertWeightAccess
from .interfaces import BoundMoERuntime, ExecutionContext, ForwardCompletion
from .resources import Allocation, ResourceKind


@dataclass(eq=False)
class CacheExecutionContext(ExecutionContext):
    device_module: object
    compute_stream: object
    copy_stream: object
    phase: Phase = Phase.DECODE
    capturing: bool = False
    _resources: list = field(default_factory=list, init=False, repr=False)
    _lock: object = field(default_factory=Lock, init=False, repr=False)

    def retain(self, resource):
        with self._lock:
            self._resources.append(resource)

    def retained(self):
        with self._lock:
            return tuple(self._resources)

    def clear(self):
        with self._lock:
            self._resources.clear()

    def record_compute_completion(self):
        event = self.device_module.Event()
        event.record(self.compute_stream)
        return TransferTicket(EventCompletion(event), (event,))

    def finish(self):
        # Record after this eager submission/replay, never reuse a capture event
        # as host evidence that a later replay completed.
        output = self.record_compute_completion()
        retained = self.retained()
        idle = TransferTicket(
            JoinedCompletion(
                (output.completion,)
                + tuple(r.completion for r in retained if isinstance(r, TransferTicket))
            ),
            (output, retained),
        )
        return ForwardCompletion(output, output, idle)


class CacheRuntime(BoundMoERuntime):
    """Concrete cache lifecycle; hardware capabilities must be supplied honestly.

    backend.finalize creates this, builds decode_program with
    DecodeLocalExpertProgram, initializes hot weights, then returns the runtime.
    DeepSeekV4ProgramBinding installs those programs on the actual model.
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
        borrow_hot_arena,
        model_binding=None,
        prefill_pipeline=None,
        close_weights=None,
    ):
        import torch

        if bank_slots <= 0 or residency.execution_key.numel() % bank_slots:
            raise ValueError("Execution slots must contain whole fixed-size banks")
        self.capabilities = capabilities
        self.expert_nbytes = expert_nbytes
        self.residency = residency
        self.cache = CacheMetadataAdapter(catalog, residency, policy)
        self.residency_ops = TensorResidencyOps(
            residency, hot_bindings, execution_bindings
        )
        self.transfer = transfer
        self.access = ExpertWeightAccess(self.residency_ops, transfer)
        self.cache_updates = CacheUpdateCoordinator(
            policy,
            self.cache.policy_state,
            residency,
            self.cache.target_keys,
            self.access,
        )
        self.context = context
        self.bank_slots = bank_slots
        # Programs depend on this runtime's access and are assembled by backend
        # before initialize/install. Model installation has one explicit owner.
        self.prefill_program = None
        self.decode_program = None
        self.prefill_pipeline = prefill_pipeline
        self._borrow_hot_arena = borrow_hot_arena
        self._enabled = torch.zeros_like(residency.update_enabled)
        self._demand = torch.full_like(residency.execution_key, -1)
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
        self.model_binding = model_binding
        self.close_weights = close_weights
        self._initialized = False
        self._capture_contexts = []
        self._serving_context = context
        # Host schedule mirror, never read back from a device tensor. Advance
        # once per successful serving forward, including graph replay. Prefill
        # and capture leave it unchanged, just like the saved device LFU cycle.
        self._step_mod = 0
        self._adjust_eager = False

    def _refill(self, slot_keys):
        # Exact-slot lifecycle operation, not LFU admission or a fake forward.
        plan = self.residency_ops.prepare_refill(slot_keys, self.context)
        self.context.retain(plan)
        for dependency in plan.copy_dependencies:
            dependency.wait_on(self.context.copy_stream)
        ticket = self.transfer.materialize_into(
            plan.copy_plan, plan.bindings, self.context.copy_stream
        )
        self.context.retain(ticket)
        return plan, ticket

    def initialize(self, model_id):
        """Host-only initial TopK -> actual copy -> publish, before serving/capture."""
        import torch

        if self._initialized:
            raise RuntimeError("Cache weights are already initialized")
        keys = self.cache.policy.initial_keys(
            self.residency.slot_to_key.numel(), model_id
        )
        with self.context.device_module.stream(self.context.compute_stream):
            plan, ticket = self._refill(
                torch.tensor(
                    keys, dtype=torch.int64, device=self.residency.key_to_slot.device
                )
            )
            published = self.residency_ops.commit_update(plan, ticket, self.context)
            self.context.retain(published)
            published.completion.synchronize()
        self.context.clear()
        self._initialized = True

    def install_model_hooks(self, offloader, model_runner):
        if (
            self._runner is not None
            or not self._initialized
            or self.decode_program is None
            or self.prefill_program is None
            or self.model_binding is None
        ):
            raise RuntimeError("Initialize weights and programs before installing once")
        if (
            self.capabilities.full_decode_graph
            and not self.transfer.supports_device_plans
        ):
            raise ValueError("Full decode graph requires device-plan transfer support")
        self.model_binding.install(self)
        model_runner.capture_head_hooks.append(self._capture_head)
        model_runner.capture_tail_hooks.append(self._capture_tail)
        self._runner = model_runner

    def _capture_head(self, runner, forward_batch, num_tokens):
        # _enabled is zero outside serving. Do not capture a fill_(False): the
        # serving host writes this fixed-address input before each real replay.
        self.context = CacheExecutionContext(
            self._serving_context.device_module,
            self.context.device_module.current_stream(),
            self._serving_context.copy_stream,
            phase=Phase.DECODE,
            capturing=True,
        )
        self._capture_contexts.append(self.context)
        self.emit_step_begin(self.context)

    def _capture_tail(self, runner, output, forward_batch, num_tokens):
        try:
            self.emit_step_end(self.context)
        finally:
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
        previous = self.context
        self.context = CacheExecutionContext(
            previous.device_module,
            previous.compute_stream,
            previous.copy_stream,
            phase=phase,
        )
        # Phase switching may have retained refill plans in the old context.
        for resource in previous.retained():
            self.context.retain(resource)
        previous.clear()
        self._serving_context = self.context
        self._adjust_eager = enabled and next_step == 0
        try:
            with self.context.device_module.stream(self.context.compute_stream):
                self._enabled.fill_(enabled)
                yield
                owner.finish_forward(token, self.context.finish())
                if enabled:
                    self._step_mod = next_step
                # Owner's completion retains the resources until idle.
                self.context.clear()
        except BaseException:
            owner.fail_forward(token)
            raise
        finally:
            with self.context.device_module.stream(self.context.compute_stream):
                self._enabled.zero_()
            self._adjust_eager = False
            self.context = self._serving_context

    @contextmanager
    def eager_step_scope(self, forward_batch):
        # ModelRunner enters only around an actual eager model invocation.
        # Captured decode has its own head/tail, so replay never double-counts.
        if self.context.phase is Phase.DECODE:
            self.emit_step_begin(self.context)
        yield
        if self.context.phase is Phase.DECODE:
            self.emit_step_end(self.context, adjust=self._adjust_eager)

    def emit_step_begin(self, context):
        self.cache.begin_step(self._enabled)

    def emit_after_router(self, layer_id, routing, context):
        # Actual rank-local assignments are recorded AFTER dispatch below.
        pass

    def make_execution_request(self, layer_id, routes, wave, context):
        import torch

        offset, size = self.cache.layer_layout[layer_id]
        width = wave.local_end - wave.local_start
        start = wave.bank * self.bank_slots
        if not (
            0 <= wave.local_start < wave.local_end <= size
            and width <= self.bank_slots
            and 0 <= start
            and start + self.bank_slots <= self._demand.numel()
        ):
            raise ValueError("Wave does not fit the declared execution bank")
        counts = self.cache.record_local_routes(
            layer_id,
            routes.expert_ids,
            routes.valid,
            local_start=wave.local_start,
            local_end=wave.local_end,
        )
        experts = torch.arange(
            wave.local_start, wave.local_end, device=routes.expert_ids.device
        )
        self._demand.fill_(-1)
        self._demand[start : start + width].copy_(
            torch.where(counts[experts] > 0, experts + offset, -1)
        )
        return ExecutionRequest(layer_id, self._demand, wave, context.phase)

    def emit_step_end(self, context, *, adjust=True):
        ticket = self.cache_updates.emit_step_end(context, adjust=adjust)
        # Join publication streams into the captured tail as well as eager F.
        # A fresh post-replay compute event then covers residency publication.
        ticket.completion.wait_on(context.compute_stream)
        context.retain(ticket)
        return ticket

    def snapshot_decode_state(self):
        with self.context.device_module.stream(self.context.compute_stream):
            self.cache.save_decode_metadata()
            return self.context.record_compute_completion()

    def borrow_hot_arena(self):
        self._borrow_hot_arena()

    def restore_decode_state(self):
        with self.context.device_module.stream(self.context.compute_stream):
            plan, ticket = self._refill(self.cache.snapshot.slot_to_key)
            ticket.completion.wait_on(self.context.compute_stream)
            self.cache.restore_decode_metadata_after_refill()
            published = self.context.record_compute_completion()
            return TransferTicket(published.completion, (published, ticket, plan))

    def synchronize(self):
        # abort stops/join prefill submitters before draining their device work.
        pipeline = self.prefill_pipeline
        if pipeline is not None and self.context.phase is Phase.PREFILL:
            self.context.retain(pipeline.abort(self.context))
        self.context.device_module.synchronize()
        for resource in self.context.retained():
            if isinstance(resource, TransferTicket):
                resource.completion.synchronize()
        self.context.clear()

    def close(self):
        # Owner synchronizes and releases graph leases before calling close.
        if self.model_binding is not None:
            self.model_binding.close()
            self.model_binding = None
        if self._runner is not None:
            self._runner.capture_head_hooks.remove(self._capture_head)
            self._runner.capture_tail_hooks.remove(self._capture_tail)
            self._runner = None
        self.context.clear()
        self._capture_contexts.clear()
        if self.close_weights is not None:
            close, self.close_weights = self.close_weights, None
            close()
