"""Reusable invocation and exact-restore machinery for accelerator bindings.

No device is initialized by importing this module. Native stream/event factories
and weight reloads are supplied by the binding; none of these helpers certifies
an accelerator graph or creates another expert-weight allocation.
"""

from contextlib import contextmanager
from dataclasses import fields
from threading import RLock, get_ident

from ..transfer import Completion, JoinedCompletion, TransferTicket
from .config import Phase
from .interfaces import ExecutionContext, ForwardCompletion
from .resources import Allocation, ResourceKind
from .state import ResidencyState


class ManagedExecutionContext(ExecutionContext):
    """Thread-safe keepalives and fresh completion events for one submission.

    Stop/join host workers before finish(). failure_drain must stop workers and
    drain *every* device stream, including partially submitted native calls.
    On a failed drain the context deliberately keeps all resources alive.
    Capture contexts remain pinned by their graph bucket, not by replay tickets.
    """

    def __init__(
        self,
        phase,
        *,
        compute_stream,
        copy_stream,
        record_completion,
        failure_drain,
        capturing=False,
    ):
        if not isinstance(phase, Phase):
            raise ValueError("Unsupported execution phase")
        if phase is Phase.PREFILL and capturing:
            raise ValueError("Prefill cannot be captured")
        self.phase = phase
        self.capturing = capturing
        self.compute_stream = compute_stream
        self.copy_stream = copy_stream
        self._record = record_completion
        self._failure_drain = failure_drain
        self._lock = RLock()
        self._resources = []
        self._temporary_resources = []
        self._submission_thread = get_ident()
        self._dependencies = []
        self._closed = False
        self._completion = None

    def retain(self, resource):
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "Invocation already closed; join workers before finish"
                )
            if isinstance(resource, (TransferTicket, Completion)):
                # Captured events and transfer keepalives outlive tensor scopes.
                self._resources.append(resource)
                self._dependencies.append(
                    resource.completion
                    if isinstance(resource, TransferTicket)
                    else resource
                )
            elif self._temporary_resources and get_ident() == self._submission_thread:
                self._temporary_resources[-1].append(resource)
            else:
                self._resources.append(resource)

    @contextmanager
    def temporary_scope(self):
        """Release submission-thread wave/layer temporaries in prefill and decode.

        Only use after joining off-stream readers onto compute_stream. Ordinary
        tensor allocation/free then follows the same allocator rules as resident
        inference; graph pools keep addresses reserved without pinning every
        intermediate Tensor. Owners/tickets keep registrations and events alive.
        Background submitters bypass these scopes, retaining their resources for
        the invocation even while the submission thread has an open scope.
        """
        if get_ident() != self._submission_thread:
            raise RuntimeError("Only the submission thread may open temporary scopes")
        resources = []
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already closed")
            self._temporary_resources.append(resources)
        try:
            yield
        except BaseException:
            # Partial submission may have no usable event. Do not release these
            # resources until the runtime drains every device/communication stream.
            with self._lock:
                self._resources.extend(resources)
            raise
        finally:
            with self._lock:
                self._temporary_resources.pop()

    def record_compute_completion(self):
        return self._record_on(self.compute_stream)

    def _record_on(self, stream):
        ticket = self._record(stream)
        if not isinstance(ticket, TransferTicket):
            raise TypeError("Completion factory must return TransferTicket")
        self.retain(ticket)
        return ticket

    def finish(self, *, output_ready=None, residency_ready=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already finished")
            if self._temporary_resources:
                raise RuntimeError("Close temporary scopes before finishing invocation")
            # Fresh records on *this* submission; capture-time tickets cannot
            # establish completion of a later replay.
            output_ready = output_ready or self.record_compute_completion()
            residency_ready = residency_ready or self.record_compute_completion()
            for ticket in (output_ready, residency_ready):
                if not isinstance(ticket, TransferTicket):
                    raise TypeError("Forward completion requires TransferTicket")
                self.retain(ticket)
            copy_done = self._record_on(self.copy_stream)
            if self.capturing:
                # CaptureEnd requires every auxiliary stream's final task to
                # rejoin the origin stream, including this completion record.
                copy_done.completion.wait_on(self.compute_stream)
            idle = TransferTicket(
                JoinedCompletion(tuple(self._dependencies)),
                tuple(self._resources),
            )
            self._completion = ForwardCompletion(output_ready, residency_ready, idle)
            self._closed = True
            return self._completion

    def release_completed(self):
        """Release a completed invocation's keepalives outside device capture.

        Used for eager graph warmups. Captured invocations must stay pinned until
        their graph is destroyed. Keep every reference if completion waiting fails.
        """
        with self._lock:
            if not self._closed or self._completion is None:
                raise RuntimeError("Finish the invocation before releasing resources")
            completion = self._completion
        completion.resources_idle.completion.synchronize()
        with self._lock:
            self._resources.clear()
            self._dependencies.clear()
            # Leases can refer back to this context. Drop the completion's copy
            # of the keepalives too, so warmup tensors do not wait for cyclic GC.
            self._completion = None

    def abort(self):
        """Drain failed submissions before releasing any retained Python object."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already finished")
        # Do not hold _lock: a worker being joined may still call retain().
        self._failure_drain()
        with self._lock:
            for completion in self._dependencies:
                completion.synchronize()
            self._closed = True
            self._resources.clear()
            self._dependencies.clear()


class DecodeStateSnapshot:
    """Preallocated metadata snapshot with weights reloaded into original slots.

    reload_weights(saved_residency, context) must load every saved valid slot
    from the immutable host master and return its complete bundle ticket. It
    must not publish live maps. This helper restores maps/policy only after the
    reload dependency; a failure leaves the owner poisoned for explicit teardown.
    """

    def __init__(self, residency, policy, policy_state, *, error_status):
        import torch

        self.residency = residency
        self.policy = policy
        self.policy_state = policy_state
        if (
            error_status.shape != ()
            or error_status.dtype != torch.int32
            or error_status.device != residency.key_to_slot.device
        ):
            raise ValueError("Snapshot needs the shared scalar device error_status")
        self.error_status = error_status
        self.saved = ResidencyState(
            **{
                field.name: torch.empty_like(getattr(residency, field.name))
                for field in fields(residency)
            }
        )
        self.saved_policy = policy.allocate_state(
            residency.key_to_slot.numel(), residency.key_to_slot.device
        )
        self._snapshot = None

    def save(self, context):
        if context.capturing:
            raise RuntimeError("Mode snapshots are outside graph capture")
        if self._snapshot is not None:
            raise RuntimeError("Decode snapshot already exists")
        context.retain(self)
        # Mode changes already drain the owner. Observe device faults at this
        # eager boundary, never from dynamic decode planning or graph replay.
        context.record_compute_completion().completion.synchronize()
        self._require_healthy()
        self._copy_residency(self.residency, self.saved)
        self.policy.copy_state(self.policy_state, self.saved_policy)
        ticket = context.record_compute_completion()
        self._snapshot = TransferTicket(ticket.completion, (ticket, self))
        context.retain(self._snapshot)
        return self._snapshot

    def restore(self, context, reload_weights):
        if context.capturing or self._snapshot is None:
            raise RuntimeError("Restore requires an eager saved decode snapshot")
        context.retain(self)
        # Establish on both streams: restore lowering may inspect saved maps
        # on compute before scheduling weight loads on copy.
        self._snapshot.completion.wait_on(context.compute_stream)
        self._snapshot.completion.wait_on(context.copy_stream)
        copied = reload_weights(self.saved, context)
        if not isinstance(copied, TransferTicket):
            raise TypeError("Snapshot reload must return TransferTicket")
        context.retain(copied)
        # Native submission success does not imply a valid copy: device lowering
        # can latch a fault and mask every write. A transition is eager and can
        # wait here; never publish saved valid maps for an unsuccessful reload.
        copied.completion.synchronize()
        self._require_healthy()
        copied.completion.wait_on(context.compute_stream)
        self._copy_residency(self.saved, self.residency)
        self.policy.copy_state(self.saved_policy, self.policy_state)
        published = context.record_compute_completion()
        ticket = TransferTicket(
            JoinedCompletion((copied.completion, published.completion)),
            (self, self._snapshot, copied, published),
        )
        context.retain(ticket)
        self._snapshot = None
        return ticket

    def _require_healthy(self):
        if self.error_status.item() != 0:
            raise RuntimeError("Device residency/transfer fault during mode transition")

    @staticmethod
    def _copy_residency(source, destination):
        for field in fields(source):
            getattr(destination, field.name).copy_(getattr(source, field.name))

    def allocations(self):
        """Include both saved residency and policy in the owner's metadata budget."""
        tensors = [
            ("residency." + field.name, getattr(self.saved, field.name))
            for field in fields(self.saved)
        ]
        # Cache policies can expose custom allocation reporting; the bundled
        # TensorLFUPolicy is a dataclass of tensors.
        if hasattr(self.saved_policy, "allocations"):
            policy_allocations = tuple(self.saved_policy.allocations())
        else:
            policy_allocations = ()
            tensors.extend(
                ("policy." + field.name, getattr(self.saved_policy, field.name))
                for field in fields(self.saved_policy)
            )
        result, seen = (
            list(policy_allocations),
            {x.storage_key for x in policy_allocations},
        )
        for name, tensor in tensors:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if storage.nbytes() and key not in seen:
                seen.add(key)
                result.append(
                    Allocation(
                        "snapshot." + name,
                        ResourceKind.METADATA,
                        storage.nbytes(),
                        key,
                        self,
                    )
                )
        return tuple(result)
