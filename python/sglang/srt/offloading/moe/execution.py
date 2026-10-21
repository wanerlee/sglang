"""Compose residency with transfers without owning weights or choosing hot experts.

Context.retain must pin objects before submission, including failed submissions,
until resources_idle (and graph destruction for captured storage). The context is
runtime-owned; these helpers never poll device values or manage a private pool.
"""

from dataclasses import dataclass
from enum import Enum
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


class SourcePolicy(Enum):
    HOST_ONLY = "host_only"
    PREFER_RESIDENT = "prefer_resident"


@dataclass(frozen=True)
class ExecutionRequest:
    layer_id: int
    local_demand: Any
    wave: ExpertWave
    phase: Phase
    source_policy: SourcePolicy = SourcePolicy.PREFER_RESIDENT

    def __post_init__(self):
        if self.layer_id < 0 or not isinstance(self.phase, Phase):
            raise ValueError("An execution request needs a layer and supported phase")
        if not isinstance(self.source_policy, SourcePolicy):
            raise ValueError("Invalid source policy")
        if (
            self.phase is Phase.PREFILL
            and self.source_policy is not SourcePolicy.HOST_ONLY
        ):
            raise ValueError("Prefill must reload all local experts from host")


@dataclass(frozen=True)
class MaterializationPlan:
    copy_plan: Any
    bindings: Any
    copy_dependencies: Tuple[Completion, ...] = ()
    keepalive: Tuple[Any, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "copy_dependencies", tuple(self.copy_dependencies))
        object.__setattr__(self, "keepalive", tuple(self.keepalive))


@dataclass(frozen=True)
class ExecutionReservation(MaterializationPlan):
    # Backend-owned fixed device metadata or an eager reservation/generation.
    state: Any = None


@dataclass(frozen=True)
class UpdatePlan(MaterializationPlan):
    state: Any = None


@dataclass
class WeightLease:
    reservations: Tuple[ExecutionReservation, ...]
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

    def __init__(self, residency_ops, transfer_backend):
        self.residency_ops = residency_ops
        self.transfer_backend = transfer_backend
        self._token = object()

    def stage(self, request, context):
        if request.phase is not context.phase:
            raise ValueError("Execution request phase does not match context")
        if request.phase is Phase.PREFILL:
            require_prefill(context)
        context.retain((self, request))
        reservation = self.residency_ops.resolve_execution(request, context)
        if not isinstance(reservation, ExecutionReservation):
            raise TypeError("resolve_execution must return ExecutionReservation")
        copied = self._materialize(reservation, context)
        published = self.residency_ops.publish_execution(reservation, copied, context)
        ready = self._join(copied, published, context)
        lease = WeightLease(
            (reservation,), reservation.bindings, ready, self._token, context
        )
        context.retain(lease)
        return lease

    def combine_prefill(self, leases, bindings, context):
        """Join submitted microbatches into the sealed large FFN group, without copy.

        Bindings must be a validated view of those reservations. The pipeline
        validates contiguous expert coverage/format; this helper transfers usage
        rights and joins completion. Child leases cannot be consumed separately.
        """
        require_prefill(context)
        leases = tuple(leases)
        if not leases or len({id(lease) for lease in leases}) != len(leases):
            raise ValueError("A prefill group needs distinct nonempty leases")
        for lease in leases:
            self._check_lease(lease)
            if lease.context_token is not context:
                raise ValueError("Cannot combine leases from different invocations")
        ticket = TransferTicket(
            JoinedCompletion(lease.ticket.completion for lease in leases), leases
        )
        grouped = WeightLease(
            tuple(r for lease in leases for r in lease.reservations),
            bindings,
            ticket,
            self._token,
            context,
        )
        context.retain(grouped)
        context.retain(ticket)
        for lease in leases:
            lease.transferred = True
        return grouped

    def wait_ready(self, lease, compute_stream):
        self._check_lease(lease)
        lease.ticket.completion.wait_on(compute_stream)
        return lease.bindings

    def release_after(self, lease, read_done, context):
        self._check_lease(lease)
        if lease.context_token is not context:
            raise ValueError("Weight lease belongs to a different invocation")
        if not isinstance(read_done, TransferTicket):
            raise TypeError("Reader completion must be a TransferTicket")
        # Retain before calling a backend which can fail after emitting work.
        context.retain(read_done)
        # A partial retirement poisons the invocation; do not submit it twice.
        lease.read_done = read_done
        for reservation in lease.reservations:
            self.residency_ops.finish_execution(reservation, read_done, context)

    def apply_update(self, target_keys, context):
        if context.phase is not Phase.DECODE:
            raise RuntimeError("Hot-set updates are decode-only")
        context.retain((self, target_keys))
        update = self.residency_ops.prepare_update(target_keys, context)
        if not isinstance(update, UpdatePlan):
            raise TypeError("prepare_update must return UpdatePlan")
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
            submit = self.transfer_backend.materialize_host_into
        elif context.phase is Phase.DECODE:
            expected = DeviceCopyPlan
            submit = self.transfer_backend.materialize_into
        else:
            raise ValueError("Unsupported execution phase")
        if not isinstance(plan.copy_plan, expected):
            raise TypeError(f"This phase requires {expected.__name__}")
        if context.phase is Phase.DECODE:
            # Even different banks cannot prefetch a later wave during prior F.
            frontier = context.record_compute_completion()
            context.retain(frontier)
            frontier.completion.wait_on(context.copy_stream)
        for dependency in plan.copy_dependencies:
            dependency.wait_on(context.copy_stream)
        ticket = submit(plan.copy_plan, plan.bindings, context.copy_stream)
        if not isinstance(ticket, TransferTicket):
            raise TypeError("Materialization must return TransferTicket")
        retained = TransferTicket(ticket.completion, (ticket, plan))
        context.retain(retained)
        return retained

    @staticmethod
    def _join(copied, published, context):
        if not isinstance(published, TransferTicket):
            raise TypeError("Residency publication must return TransferTicket")
        ticket = TransferTicket(
            JoinedCompletion((copied.completion, published.completion)),
            (copied, published),
        )
        context.retain(ticket)
        return ticket


class CacheUpdateCoordinator:
    """Bound runtime helper for graph-tail policy -> transfer -> publication.

    Buffers are bound before capture. The concrete ResidencyOps must honor the
    runtime's device update_enabled guard, including during warmup/capture.
    Returned tickets describe emitted dependencies, not later replay completion.
    """

    def __init__(self, policy, policy_state, residency_state, target_keys, access):
        self.policy = policy
        self.policy_state = policy_state
        self.residency_state = residency_state
        self.target_keys = target_keys
        self.access = access

    def emit_step_end(self, context, *, adjust=True):
        if context.phase is Phase.PREFILL:
            require_prefill(context)
            return TransferTicket(ImmediateCompletion())
        if context.phase is not Phase.DECODE:
            raise ValueError("Unsupported execution phase")
        context.retain(self)
        self.policy.update_statistics(
            self.policy_state,
            self.residency_state.step_access_count,
            self.residency_state.update_enabled,
        )
        if adjust:
            self.policy.select_hot_set(
                self.policy_state, self.residency_state.key_to_slot, self.target_keys
            )
        self.policy.finish_step(self.policy_state)
        if not adjust:
            return TransferTicket(ImmediateCompletion())
        return self.access.apply_update(self.target_keys, context)
