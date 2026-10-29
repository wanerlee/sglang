"""Compose residency with transfers without owning weights or choosing hot experts.

Context.retain must pin objects before submission, including failed submissions,
until resources_idle (and graph destruction for captured storage). The context is
runtime-owned; these helpers never poll device values or manage a private pool.
"""

from dataclasses import dataclass
from typing import Any, Tuple

from ..transfer import (
    Completion,
    DeviceCopyPlan,
    HostCopyPlan,
    ImmediateCompletion,
    JoinedCompletion,
    TransferTicket,
)
from .config import ExpertWave, Phase


@dataclass(frozen=True)
class ExecutionRequest:
    layer_id: int
    local_demand: Any
    wave: ExpertWave
    phase: Phase
    # Explicit per-layer decode preparation; None preserves standalone planners.
    layer_plan: Any = None

    def __post_init__(self):
        if self.layer_id < 0 or not isinstance(self.phase, Phase):
            raise ValueError("An execution request needs a layer and supported phase")


@dataclass(frozen=True)
class MaterializationPlan:
    copy_plan: Any
    bindings: Any
    copy_dependencies: Tuple[Completion, ...] = ()
    keepalive: Tuple[Any, ...] = ()
    # Backend reservation metadata. Decode dependencies cover the preceding
    # compute frontier and every prior target reader, including other banks.
    state: Any = None

    def __post_init__(self):
        object.__setattr__(self, "copy_dependencies", tuple(self.copy_dependencies))
        object.__setattr__(self, "keepalive", tuple(self.keepalive))


@dataclass
class WeightLease:
    reservations: Tuple[MaterializationPlan, ...]
    bindings: Any
    ticket: TransferTicket
    manager_token: Any
    context_token: Any
    read_done: Any = None
    transferred: bool = False


@dataclass(frozen=True)
class LocalExecution:
    """Runner result; read_done covers every stream that reads expert weights."""

    output: Any
    read_done: TransferTicket


def require_prefill(context):
    if context.phase is not Phase.PREFILL or context.capturing:
        raise RuntimeError("Prefill pipeline requires eager prefill execution")


class ExpertWeightAccess:
    """Stage, acquire and release temporary weights independently of hot admission."""

    def __init__(self, residency_ops, *, host_ops=None, device_ops=None):
        self.residency_ops = residency_ops
        self.host_ops = host_ops
        self.device_ops = device_ops
        self._token = object()

    def prepare_decode_layer(self, layer_id, local_demand, context):
        return self.residency_ops.prepare_decode_layer(layer_id, local_demand, context)

    def finish_decode_layer(self, layer_plan, context):
        self.residency_ops.finish_decode_layer(layer_plan, context)

    def stage(self, request, context):
        if request.phase is not context.phase:
            raise ValueError("Execution request phase does not match context")
        if request.phase is Phase.PREFILL:
            require_prefill(context)
        context.retain((self, request))
        reservation = self.residency_ops.resolve_execution(request, context)
        if not isinstance(reservation, MaterializationPlan):
            raise TypeError("resolve_execution must return MaterializationPlan")
        copied = self._materialize(reservation, context)
        published = self.residency_ops.publish_execution(reservation, copied, context)
        ready = self._join(copied, published, context)
        lease = WeightLease(
            (reservation,),
            reservation.bindings,
            ready,
            self._token,
            context.invocation_id,
        )
        context.retain(lease)
        return lease

    def combine_prefill(self, leases, bindings, context, *, ready=None):
        """Join submitted microbatches into the sealed large FFN group, without copy.

        Bindings must be a validated view of those reservations. The pipeline
        validates contiguous expert coverage/format; this helper transfers usage
        rights and joins completion. Child leases cannot be consumed separately.
        An optional ready ticket must fence every child copy/publication; only
        an explicitly ordered backend may replace the individual dependencies.
        """
        require_prefill(context)
        leases = tuple(leases)
        if not leases or len({id(lease) for lease in leases}) != len(leases):
            raise ValueError("A prefill group needs distinct nonempty leases")
        for lease in leases:
            self._check_lease(lease)
            if lease.context_token is not context.invocation_id:
                raise ValueError("Cannot combine leases from different invocations")
        if ready is not None and not isinstance(ready, TransferTicket):
            raise TypeError("Prefill group readiness must be a TransferTicket")
        ticket = TransferTicket(
            (
                ready.completion
                if ready is not None
                else JoinedCompletion(lease.ticket.completion for lease in leases)
            ),
            (*leases, ready) if ready is not None else leases,
        )
        grouped = WeightLease(
            tuple(r for lease in leases for r in lease.reservations),
            bindings,
            ticket,
            self._token,
            context.invocation_id,
        )
        context.retain(grouped)
        context.retain_completion(ticket)
        for lease in leases:
            lease.transferred = True
        return grouped

    def wait_ready(self, lease, compute_stream, *, context):
        self._check_lease(lease)
        if lease.context_token is not context.invocation_id:
            raise ValueError("Weight lease belongs to a different invocation")
        context.wait_for(lease.ticket, compute_stream)
        return lease.bindings

    def release_after(self, lease, read_done, context):
        self._check_lease(lease)
        if lease.context_token is not context.invocation_id:
            raise ValueError("Weight lease belongs to a different invocation")
        if not isinstance(read_done, TransferTicket):
            raise TypeError("Reader completion must be a TransferTicket")
        # Retain before calling a backend which can fail after emitting work.
        context.retain_completion(read_done)
        # A partial retirement poisons the invocation; do not submit it twice.
        lease.read_done = read_done
        for reservation in lease.reservations:
            self.residency_ops.finish_execution(reservation, read_done, context)

    def apply_update(self, target_keys, context, *, update_enabled=None):
        if context.phase is not Phase.DECODE:
            raise RuntimeError("Hot-set updates are decode-only")
        context.retain((self, target_keys))
        update = self.residency_ops.prepare_update(
            target_keys, context, update_enabled=update_enabled
        )
        if not isinstance(update, MaterializationPlan):
            raise TypeError("prepare_update must return MaterializationPlan")
        copied = self._materialize(update, context)
        committed = self.residency_ops.commit_update(update, copied, context)
        return self._join(copied, committed, context)

    def _check_lease(self, lease):
        if lease.manager_token is not self._token:
            raise ValueError("Foreign weight lease")
        if lease.transferred:
            raise RuntimeError("Weight lease was transferred into a prefill group")
        if lease.read_done is not None:
            raise RuntimeError("Weight lease already has a final reader")

    def _materialize(self, plan, context):
        # Must survive even if the driver raises after partial submission.
        context.retain(plan)
        if context.phase is Phase.PREFILL:
            require_prefill(context)
            expected = HostCopyPlan
            ops = self.host_ops
        elif context.phase is Phase.DECODE:
            expected = DeviceCopyPlan
            ops = self.device_ops
        else:
            raise ValueError("Unsupported execution phase")
        if not isinstance(plan.copy_plan, expected):
            raise TypeError(f"This phase requires {expected.__name__}")
        if ops is None:
            raise RuntimeError(
                f"No transfer implementation bound for {context.phase.value}"
            )
        for dependency in plan.copy_dependencies:
            dependency.wait_on(context.copy_stream)
        ticket = ops.materialize_into(
            plan.copy_plan, plan.bindings, context.copy_stream
        )
        if not isinstance(ticket, TransferTicket):
            raise TypeError("Materialization must return TransferTicket")
        retained = TransferTicket(ticket.completion, (ticket, plan))
        context.retain_completion(retained)
        return retained

    @staticmethod
    def _join(copied, published, context):
        if not isinstance(published, TransferTicket):
            raise TypeError("Residency publication must return TransferTicket")
        # publish_execution/commit_update must wait for the complete bundle.
        # Its publication frontier therefore covers copied, while both tickets
        # still retain every source/destination owner until retirement.
        ticket = TransferTicket(
            published.completion,
            (copied, published),
        )
        context.retain_completion(ticket)
        return ticket


class CacheUpdateCoordinator:
    """Bound runtime helper for graph-tail policy -> transfer -> publication.

    Buffers are bound before capture. The concrete ResidencyOps must honor the
    runtime's device update_enabled guard, including during warmup/capture.
    Returned tickets describe emitted dependencies, not later replay completion.
    """

    def __init__(
        self, policy, policy_state, residency_state, target_keys, access, *, due_out
    ):
        self.policy = policy
        self.policy_state = policy_state
        self.residency_state = residency_state
        self.target_keys = target_keys
        self.access = access
        self.due_out = due_out

    def emit_step_end(self, context, *, maintenance=True):
        if context.phase is Phase.PREFILL:
            require_prefill(context)
            return TransferTicket(ImmediateCompletion())
        if context.phase is not Phase.DECODE:
            raise ValueError("Unsupported execution phase")
        context.retain(self)
        self.policy.observe(
            self.policy_state,
            self.residency_state.step_access_count,
            self.residency_state.update_enabled,
            self.due_out,
        )
        # This is a capture-time graph variant, never a Python branch on a
        # device scalar. Ordinary replay contains no selection/transfer tail.
        if not maintenance:
            return context.record_compute_completion()
        self.policy.select_hot_set(
            self.policy_state, self.residency_state.key_to_slot, self.target_keys
        )
        self.policy.finish_step(self.policy_state)
        return self.access.apply_update(
            self.target_keys, context, update_enabled=self.due_out
        )
