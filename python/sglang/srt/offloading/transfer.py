"""Backend-neutral transfers into caller-owned storage.

Host-planned copies are useful for V2 and prefill. They do not implement dynamic
decode replay: a DeviceCopyPlan requires an explicitly supplied device lowering.
No transport may allocate an unaccounted expert-weight destination or staging copy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple


class Completion(ABC):
    """A dependency covering every stream used by the associated operation."""

    @abstractmethod
    def wait_on(self, stream: Any) -> None:
        """Enqueue a device wait; do not synchronize the host."""

    @abstractmethod
    def synchronize(self) -> None:
        """Host wait, reserved for transitions, teardown and diagnostics."""

    @abstractmethod
    def done(self) -> bool:
        """Nonblocking host query; never use this to select graph work."""


class ImmediateCompletion(Completion):
    def wait_on(self, stream):
        pass

    def synchronize(self):
        pass

    def done(self):
        return True


class EventCompletion(Completion):
    """Works with CUDA/NPU events exposing query/synchronize/wait_event."""

    def __init__(self, event):
        self.event = event

    def wait_on(self, stream):
        stream.wait_event(self.event)

    def synchronize(self):
        self.event.synchronize()

    def done(self):
        return self.event.query()


@dataclass(frozen=True)
class TransferTicket:
    completion: Completion
    # Retain storage, registration handles, plans and destinations until complete.
    keepalive: Tuple[Any, ...] = ()


class JoinedCompletion(Completion):
    """Join dependencies without inventing a host-side completion signal."""

    def __init__(self, completions):
        self.completions = tuple(completions)

    def wait_on(self, stream):
        for completion in self.completions:
            completion.wait_on(stream)

    def synchronize(self):
        for completion in self.completions:
            completion.synchronize()

    def done(self):
        return all(completion.done() for completion in self.completions)


@dataclass(frozen=True)
class HostCopyRegion:
    """Byte ranges relative to backend-recognized allocation bases, not tensor shapes.

    The backend validates physical layout, registration, direction and aliasing.
    No dtype conversion or weight staging is implied by this descriptor.
    """

    source: Any
    destination: Any
    source_offset: int
    destination_offset: int
    nbytes: int
    source_capacity: int
    destination_capacity: int

    def __post_init__(self):
        if (
            min(
                self.source_offset,
                self.destination_offset,
                self.nbytes,
                self.source_capacity,
                self.destination_capacity,
            )
            < 0
        ):
            raise ValueError("Copy byte ranges must be nonnegative")
        if self.source_offset + self.nbytes > self.source_capacity:
            raise ValueError("Copy exceeds source allocation")
        if self.destination_offset + self.nbytes > self.destination_capacity:
            raise ValueError("Copy exceeds destination allocation")


@dataclass(frozen=True)
class HostCopyPlan:
    """Eager-only descriptors covering complete expert bundles, including scales."""

    regions: Tuple[HostCopyRegion, ...]
    keepalive: Tuple[Any, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "keepalive", tuple(self.keepalive))


@dataclass(frozen=True)
class DeviceCopyPlan:
    """Fixed-capacity tensors; active entries/count are device values.

    source_kind distinguishes HOST, HOT_SLOT and NONE. expected_generation guards
    reused slots. An expert entry includes all weights and quantization metadata.
    The backend validates shapes/dtypes once when binding, outside capture.
    """

    source_kind: Any
    source_index: Any
    destination_slot: Any
    valid: Any
    expected_generation: Any
    count: Any
    capacity: int


class HostTransferOps(ABC):
    """Host-planned copies into owned storage; never a dynamic replay fallback.

    An ACL batch implementation must preserve torch_npu host queue ordering,
    reject dependent/overlapping batch entries and record bundle readiness only
    after every submitted copy. A failed submission may still have live writes.
    """

    @abstractmethod
    def materialize_into(self, plan: HostCopyPlan, bindings, stream) -> TransferTicket:
        """Submit on stream; retain registrations/descriptors through completion."""


class DeviceTransferOps(ABC):
    """Hardware lowering for device-selected sources and fixed destinations."""

    @abstractmethod
    def materialize_into(self, plan, bindings, stream) -> TransferTicket:
        """Emit graph operations. No .item(), .tolist() or CPU route decisions."""


class WeightTransferBackend(ABC):
    def materialize_host_into(self, plan, bindings, stream) -> TransferTicket:
        """Optional eager entry point; existing device-only backends stay valid."""
        raise NotImplementedError("Host-planned transfers require HostTransferOps")

    @property
    @abstractmethod
    def supports_device_plans(self) -> bool:
        pass

    @abstractmethod
    def materialize_into(self, plan, bindings, stream) -> TransferTicket:
        pass


class ComposedTransferBackend(WeightTransferBackend):
    """Independent host/device implementations sharing fixed-storage semantics.

    Having device ops does not certify full graph support. The runtime must
    validate that separately. Neither entry point allocates expert storage.
    """

    def __init__(self, *, host_ops=None, device_ops=None):
        self.host_ops = host_ops
        self.device_ops = device_ops

    @property
    def supports_device_plans(self):
        return self.device_ops is not None

    def materialize_host_into(self, plan, bindings, stream):
        if not isinstance(plan, HostCopyPlan):
            raise TypeError("Host transfers require HostCopyPlan")
        return self._materialize(self.host_ops, plan, bindings, stream)

    def materialize_into(self, plan, bindings, stream):
        if not isinstance(plan, DeviceCopyPlan):
            raise TypeError("Device transfers require DeviceCopyPlan")
        return self._materialize(self.device_ops, plan, bindings, stream)

    @staticmethod
    def _materialize(ops, plan, bindings, stream):
        if ops is None:
            raise NotImplementedError(
                "The requested transfer implementation is unbound"
            )
        ticket = ops.materialize_into(plan, bindings, stream)
        if not isinstance(ticket, TransferTicket):
            raise TypeError("Transfer implementations must return TransferTicket")
        return TransferTicket(ticket.completion, (ticket, plan, bindings, ops))


class TorchTransferBackend(WeightTransferBackend):
    """Ordinary PyTorch copies, optionally extended with device-plan lowering.

    The copy stream and event are caller-owned. A supplied lowering must retain
    any host registration handle in its ticket; pinning alone is not registration.
    """

    def __init__(self, device_ops=None):
        self.device_ops = device_ops

    @property
    def supports_device_plans(self):
        return self.device_ops is not None

    def materialize_into(self, plan, bindings, stream):
        if self.device_ops is None:
            raise NotImplementedError(
                "Dynamic graph H2D requires DeviceTransferOps; static copy_ is "
                "not a decode fallback."
            )
        return self.device_ops.materialize_into(plan, bindings, stream)

    @staticmethod
    def allocate_and_copy(source, device):
        """Legacy V2 whole-parameter API; allocation is explicit at this boundary."""
        return source.to(device=device, non_blocking=True)

    @staticmethod
    def copy_into(copies, *, device_module, stream, event):
        """Copy host-planned (source, destination) tensor pairs into fixed storage.

        Callers establish prior-reader dependencies on stream before calling.
        Do not use this host loop to choose dynamic replay sources.
        """
        copies = tuple(copies)
        for source, destination in copies:
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError("Weight transfers must preserve shape and dtype")
            if source.device.type == "cpu" and not source.is_pinned():
                raise ValueError("Asynchronous H2D requires pinned host weights")
        with device_module.stream(stream):
            for source, destination in copies:
                destination.copy_(source, non_blocking=True)
            event.record(stream)
        return TransferTicket(EventCompletion(event), (copies,))
