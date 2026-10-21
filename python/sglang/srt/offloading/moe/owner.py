"""One owner, two phase controllers, explicit graph/transfer lifetimes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from threading import RLock

from .config import Phase
from .resources import AllocationLedger, ResourceKind, SlotLeaseManager


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return call


class OwnerState(Enum):
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class ForwardToken:
    owner_token: object
    sequence: int
    phase: Phase


class PhaseController:
    def __init__(self, owner, phase, program):
        self.owner = owner
        self.phase = phase
        self.program = program

    def enter(self):
        self.owner.switch_phase(self.phase)


class PrefillController(PhaseController):
    def __init__(self, owner):
        super().__init__(owner, Phase.PREFILL, owner.runtime.prefill_program)


class DecodeController(PhaseController):
    def __init__(self, owner):
        super().__init__(owner, Phase.DECODE, owner.runtime.decode_program)


class GraphLease:
    """Held by a graph bucket until its graph has been destroyed."""

    def __init__(self, owner, key):
        self.owner = owner
        self.key = key

    def close(self):
        if self.owner is None:
            return
        self.owner._release_graph(self.key)
        self.owner = None


class MoEOffloadOwner:
    def __init__(self, config, catalog, runtime):
        self._lock = RLock()
        if any(
            (layer.ep_size, layer.ep_rank) != (config.ep_size, config.ep_rank)
            for layer in catalog.layers
        ):
            raise ValueError("Owner configuration and expert catalog disagree on EP")
        runtime.capabilities.require_decode()
        if runtime.expert_nbytes <= 0:
            raise ValueError("Backend must report physical bytes of a complete expert")
        self.config = config
        self.catalog = catalog
        self.runtime = runtime
        hot_slots = config.hot_slots(catalog)
        self.allocation_ledger = AllocationLedger(
            {
                ResourceKind.HOST_WEIGHTS: catalog.num_experts * runtime.expert_nbytes,
                ResourceKind.HOT_WEIGHTS: hot_slots * runtime.expert_nbytes,
                ResourceKind.EXECUTION_WEIGHTS: config.execution_slots
                * runtime.expert_nbytes,
                ResourceKind.WORKSPACE: config.workspace_limit_bytes,
                ResourceKind.METADATA: config.metadata_limit_bytes,
            },
            runtime.resources,
        )
        for kind in (
            ResourceKind.HOST_WEIGHTS,
            ResourceKind.HOT_WEIGHTS,
            ResourceKind.EXECUTION_WEIGHTS,
        ):
            if (
                self.allocation_ledger.usage[kind]
                != self.allocation_ledger.limits[kind]
            ):
                raise ValueError(f"{kind.value} allocation does not match slot budget")
        self.hot_slot_leases = SlotLeaseManager(hot_slots)
        self.execution_slot_leases = SlotLeaseManager(config.execution_slots)
        self.state = OwnerState.READY
        self.phase = Phase.DECODE
        self.prefill = PrefillController(self)
        self.decode = DecodeController(self)
        self._token = object()
        self._sequence = 0
        self._active = None
        self._pending = []
        self._transition_ticket = None
        self._snapshot_saved = False
        self._graphs = set()

    def _require_ready(self):
        if self.state is not OwnerState.READY:
            raise RuntimeError(f"Offload owner is {self.state.value}")

    def _require_no_forward(self):
        if self._active is not None:
            raise RuntimeError("Concurrent forwards or transitions are not allowed")

    @_serialized
    def wait_idle(self):
        self._require_no_forward()
        try:
            for completion in self._pending:
                completion.resources_idle.completion.synchronize()
            self._pending.clear()
        except BaseException:
            self.state = OwnerState.FAILED
            raise

    @_serialized
    def switch_phase(self, phase):
        self._require_ready()
        self._require_no_forward()
        if not isinstance(phase, Phase):
            raise ValueError("Unsupported offload phase")
        if phase is self.phase:
            return
        self.wait_idle()
        try:
            if phase is Phase.PREFILL:
                self._transition_ticket = self.runtime.snapshot_decode_state()
                self._transition_ticket.completion.synchronize()
                self._snapshot_saved = True
                self.runtime.borrow_hot_arena()
            else:
                if not self._snapshot_saved:
                    raise RuntimeError("Cannot restore decode without a snapshot")
                self._transition_ticket = self.runtime.restore_decode_state()
                self._transition_ticket.completion.synchronize()
                self._snapshot_saved = False
            self.phase = phase
            self._transition_ticket = None
        except BaseException:
            # A backend may already have enqueued work. Retain all resources until
            # explicit close() drains every stream, including unreported work.
            self.state = OwnerState.FAILED
            raise

    @_serialized
    def begin_forward(self, phase, compute_stream):
        self._require_ready()
        self._require_no_forward()
        self.switch_phase(phase)
        # Host submissions may be asynchronous, but device forwards are serialized
        # by this join. Do not poll LFU/miss state or host-synchronize each step.
        try:
            self._pending = [
                item
                for item in self._pending
                if not item.resources_idle.completion.done()
            ]
            if self._pending:
                self._pending[-1].resources_idle.completion.wait_on(compute_stream)
        except BaseException:
            self.state = OwnerState.FAILED
            raise
        self._sequence += 1
        token = ForwardToken(self._token, self._sequence, phase)
        self._active = token
        return token

    def _check_token(self, token):
        if (
            token is None
            or token is not self._active
            or token.owner_token is not self._token
        ):
            raise ValueError("Foreign, stale or already finished forward token")

    @_serialized
    def finish_forward(self, token, completion):
        self._check_token(token)
        self._pending.append(completion)
        self._active = None

    @_serialized
    def fail_forward(self, token):
        self._check_token(token)
        self.state = OwnerState.FAILED
        self._active = None
        # runtime.synchronize() in close covers work that failed before returning
        # a completion ticket. No new forward is allowed after partial execution.

    @_serialized
    def acquire_graph(self, key):
        self._require_ready()
        self._require_no_forward()
        if key in self._graphs:
            raise ValueError("Graph bucket already bound")
        self._graphs.add(key)
        return GraphLease(self, key)

    @_serialized
    def _release_graph(self, key):
        self.wait_idle()
        # Also cover capture/warmup work, which is not a serving forward ticket.
        try:
            self.runtime.synchronize()
            self._graphs.remove(key)
        except BaseException:
            self.state = OwnerState.FAILED
            raise

    @_serialized
    def close(self):
        if self.state is OwnerState.CLOSED:
            return
        self._require_no_forward()
        if self._graphs:
            raise RuntimeError("Destroy graph buckets and release graph leases first")
        try:
            self.runtime.synchronize()
            self.wait_idle()
            self.hot_slot_leases.require_empty()
            self.execution_slot_leases.require_empty()
            self.runtime.close()
        except BaseException:
            self.state = OwnerState.FAILED
            raise
        self._transition_ticket = None
        # Release allocation keepalives only after the backend is quiescent.
        self.allocation_ledger = None
        self.state = OwnerState.CLOSED
