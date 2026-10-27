"""ACL prefill and MemFabric decode adapters for fixed, caller-owned storage.

ACL requires an explicitly injected SDK-matched native submitter. It must place
waits, batch submission and event recording in torch_npu host-queue order, retain
descriptors inside any asynchronous host callback, and release the GIL while
blocking. There is deliberately no guessed ctypes ABI or raw-stream fallback.

MemFabric's local group_pack_copy copies input i to *packed* output j. Lowering
below builds that packed destination prefix on the device. Tensor metadata is
preallocated at binding time; intermediate tensor expressions use the framework
allocator and must be included in a runtime's capture workspace accounting.
No adapter alone certifies torch_npu operator coverage or full decode graphs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ...transfer import (
    Completion,
    CopySourceKind,
    DeviceCopyPlan,
    DeviceTransferOps,
    EventCompletion,
    HostCopyPlan,
    HostTransferOps,
    TransferTicket,
)
from ..resources import Allocation, ResourceKind
from ..storage import HostMemoryCapability

_UINT32_MAX = 2**32 - 1
_INT64_MAX = 2**63 - 1
# The referenced kernel reserves 176 KiB for data, plus aligned metadata arrays.
# 256 entries fit within both 192 KiB and 256 KiB UB configurations.
_MAX_GROUP_ENTRIES = 256


logger = logging.getLogger(__name__)


def _range(address, nbytes):
    if type(address) is not int or type(nbytes) is not int:
        raise TypeError("Byte addresses and capacities must be integers")
    if address <= 0 or nbytes < 0 or address + nbytes > _INT64_MAX:
        raise ValueError("Invalid or unrepresentable memory range")
    return address, address + nbytes


def _overlap(first, second):
    return first[0] < second[1] and second[0] < first[1]


def _reject_overlapping(ranges):
    ordered = sorted(r for r in ranges if r[0] < r[1])
    for previous, current in zip(ordered, ordered[1:]):
        if _overlap(previous, current):
            raise ValueError("Copy destinations overlap")


@dataclass(frozen=True)
class DeviceMemoryRegion:
    """Verified physical byte extent in a device allocation.

    A format adapter supplies physical (possibly padded) size and retains its
    owner. from_tensor is suitable only for ordinary contiguous tensor storage;
    opaque NPU formats require an SDK-aware format adapter's explicit extent.
    """

    address: int
    nbytes: int
    device: object
    owner: object

    def __post_init__(self):
        _range(self.address, self.nbytes)
        if self.owner is None:
            raise ValueError("Device storage must retain its allocation owner")

    @classmethod
    def from_tensor(cls, tensor):
        if not tensor.is_contiguous():
            raise ValueError("Physical copy regions require contiguous storage")
        storage = tensor.untyped_storage()
        offset = tensor.data_ptr() - storage.data_ptr()
        logical_bytes = tensor.numel() * tensor.element_size()
        if offset < 0 or offset + logical_bytes > storage.nbytes():
            raise ValueError("Tensor view exceeds its physical allocation")
        return cls(tensor.data_ptr(), logical_bytes, tensor.device, tensor)


@dataclass(frozen=True)
class AclCopyBatch:
    """Native shim input: H2D only, one HOST/DEVICE location pair per entry.

    The compiled shim translates these Python-owned tuples to its SDK's arrays
    and location attributes, covering every entry with increasing indices.
    It owns the translated arrays until its queued task and DMA have completed.
    """

    source_addresses: tuple
    destination_addresses: tuple
    sizes: tuple
    destination_capacities: tuple
    keepalive: tuple
    direction: str = "host_to_device"


class TransferSubmissionError(RuntimeError):
    """Submission may be partial. Retained resources require a stream drain."""

    def __init__(self, message, completion_ticket=None):
        super().__init__(message)
        self.completion_ticket = completion_ticket


def _ticket(record_completion, stream, resources, retain):
    completion = record_completion(stream)
    if not isinstance(completion, Completion):
        raise TypeError("record_completion must return a Completion")
    ticket = TransferTicket(completion, resources)
    retain(ticket)
    return ticket


def _submission_failure(error, record_completion, stream, resources, retain):
    # Recording can fail too (e.g. a poisoned stream). Pre-submit retain remains
    # authoritative; the runtime must drain its streams before releasing it.
    try:
        ticket = _ticket(record_completion, stream, resources, retain)
    except Exception:
        ticket = None
    return TransferSubmissionError(
        f"Transfer submission failed; drain retained work before reuse: {error}",
        ticket,
    )


class AclBatchHostTransferOps(HostTransferOps):
    """Validate, bound and submit eager ACL batches without weight staging.

    native_submit(batch, stream) returns None/0 on success or raises/returns a
    nonzero status on failure. The explicit queue_ordering_verified declaration
    is the binder's assertion that its SDK-specific shim preserves host queue
    order; this class cannot establish that property from a raw ACL handle.
    retain must remain valid after exceptions and be thread-safe for prefetch.
    """

    def __init__(
        self,
        *,
        native_submit,
        record_completion,
        retain,
        queue_ordering_verified=False,
        stream_context=None,
        max_batch_entries=256,
    ):
        if not queue_ordering_verified:
            raise ValueError("ACL requires a verified torch_npu queue-ordered shim")
        if type(max_batch_entries) is not int or max_batch_entries <= 0:
            raise ValueError("ACL batch capacity must be positive")
        for callback in (native_submit, record_completion, retain):
            if not callable(callback):
                raise TypeError("ACL submission/lifetime callbacks are required")
        self.native_submit = native_submit
        self.record_completion = record_completion
        self.retain = retain
        self.stream_context = stream_context or _npu_stream_context
        self.max_batch_entries = max_batch_entries

    def _lower(self, plan):
        entries, destinations, mapped_sources = [], [], []
        destination_device = None
        for region in plan.regions:
            if any(
                type(value) is not int
                for value in (
                    region.source_offset,
                    region.destination_offset,
                    region.nbytes,
                    region.source_capacity,
                    region.destination_capacity,
                )
            ):
                raise TypeError("ACL byte ranges and capacities must be integers")
            source, destination = region.source, region.destination
            if not isinstance(source, HostMemoryCapability):
                raise TypeError("ACL source requires a HostMemoryCapability")
            if not source.acl_async_readable:
                raise ValueError(
                    "Host registration does not permit asynchronous ACL reads"
                )
            if not isinstance(destination, DeviceMemoryRegion):
                raise TypeError("ACL destination requires a DeviceMemoryRegion")
            if (
                getattr(
                    destination.device, "type", str(destination.device).split(":")[0]
                )
                != "npu"
            ):
                raise ValueError("ACL prefill supports host-to-NPU copies only")
            if (
                destination_device is not None
                and destination.device != destination_device
            ):
                raise ValueError("An ACL batch must target a single device")
            destination_device = destination.device
            if region.source_capacity > source.nbytes:
                raise ValueError("Declared source capacity exceeds registered storage")
            if region.destination_capacity > destination.nbytes:
                raise ValueError("Declared destination capacity exceeds storage")
            # Revalidate even when a caller constructed a mutable/foreign plan.
            if min(region.source_offset, region.destination_offset, region.nbytes) < 0:
                raise ValueError("Copy ranges must be nonnegative")
            if region.source_offset + region.nbytes > region.source_capacity:
                raise ValueError("Copy exceeds registered source capacity")
            if region.destination_offset + region.nbytes > region.destination_capacity:
                raise ValueError("Copy exceeds destination capacity")
            src = source.host_address + region.source_offset
            dst = destination.address + region.destination_offset
            _range(src, region.nbytes)
            if source.device_address is not None:
                mapped_sources.append(
                    _range(source.device_pointer(src, region.nbytes), region.nbytes)
                )
            destinations.append(_range(dst, region.nbytes))
            if region.nbytes:
                entries.append(
                    (
                        src,
                        dst,
                        region.nbytes,
                        region.destination_capacity - region.destination_offset,
                        source,
                        destination,
                    )
                )
        _reject_overlapping(destinations)
        if any(
            _overlap(source, destination)
            for source in mapped_sources
            for destination in destinations
        ):
            raise ValueError("Copy destination overlaps mapped immutable host storage")
        batches = []
        for begin in range(0, len(entries), self.max_batch_entries):
            chunk = entries[begin : begin + self.max_batch_entries]
            batches.append(
                AclCopyBatch(
                    tuple(e[0] for e in chunk),
                    tuple(e[1] for e in chunk),
                    tuple(e[2] for e in chunk),
                    tuple(e[3] for e in chunk),
                    tuple((e[4], e[5]) for e in chunk),
                )
            )
        return tuple(batches)

    def materialize_into(self, plan, bindings, stream):
        if not isinstance(plan, HostCopyPlan):
            raise TypeError("ACL transfers require a HostCopyPlan")
        batches = self._lower(plan)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "MoE offload acl.submit: batches=%s regions=%s bytes=%s stream=%s",
                len(batches),
                len(plan.regions),
                sum(region.nbytes for region in plan.regions),
                getattr(stream, "stream_id", None),
            )
        resources = (plan, bindings, batches, self)
        # This happens before the first native call, including a call that throws
        # after submitting a partial batch and never returns a completion ticket.
        self.retain(resources)
        with self.stream_context(stream):
            try:
                for batch in batches:
                    status = self.native_submit(batch, stream)
                    if status is not None and status != 0:
                        raise RuntimeError(f"ACL batch failed with status {status}")
                return _ticket(self.record_completion, stream, resources, self.retain)
            except Exception as error:
                raise _submission_failure(
                    error, self.record_completion, stream, resources, self.retain
                ) from error


@dataclass(frozen=True)
class MemFabricCopyComponent:
    """One physical component per expert, with equal byte extent across experts.

    Include separate components for all weights, quantization scales and padding.
    Each source/destination region starts at that expert's component; no implicit
    contiguous expert stride or dtype conversion is assumed.
    """

    host_sources: tuple
    hot_sources: tuple
    destinations: tuple
    nbytes: int
    name: str = "weights"

    def __post_init__(self):
        for field in ("host_sources", "hot_sources", "destinations"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if type(self.nbytes) is not int or not 0 <= self.nbytes <= _UINT32_MAX:
            raise ValueError("MemFabric component size must fit uint32 bytes")
        for source in self.host_sources:
            if not isinstance(source, HostMemoryCapability):
                raise TypeError("Mapped host sources require HostMemoryCapability")
            if not source.acl_async_readable:
                raise ValueError(
                    "Shared host master must also support asynchronous ACL reads"
                )
            source.device_pointer(source.host_address, self.nbytes)
        for region in self.hot_sources + self.destinations:
            if not isinstance(region, DeviceMemoryRegion):
                raise TypeError("Device sources/targets require DeviceMemoryRegion")
            if self.nbytes > region.nbytes:
                raise ValueError("Component exceeds its physical storage capacity")


class MemFabricTransferBindings:
    """Prebind address tables and native metadata; never construct during replay.

    error_status is the runtime's shared scalar int32 fault flag. Residency
    publication and FFN validity must be gated by it; only a drained/reset runtime
    may clear it. hot_generations is canonical slot-generation state.
    """

    def __init__(
        self,
        components,
        *,
        capacity,
        hot_generations,
        error_status,
        device,
        max_batch_entries=256,
        name=None,
    ):
        import torch

        self.components = tuple(components)
        if not self.components:
            raise ValueError("At least one complete-bundle component is required")
        if type(capacity) is not int or capacity < 0:
            raise ValueError("Plan capacity must be nonnegative")
        if (
            type(max_batch_entries) is not int
            or not 0 < max_batch_entries <= _MAX_GROUP_ENTRIES
        ):
            raise ValueError("MemFabric batch capacity must be in [1, 256]")
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("MemFabric requires NPU storage")
        if self.device.type == "npu" and self.device.index is None:
            raise ValueError("Bind MemFabric to an explicit NPU device index")
        self.name = name if name is not None else f"memfabric.{id(self):x}"
        self.capacity, self.max_batch_entries = capacity, max_batch_entries
        first = self.components[0]
        self.host_count = len(first.host_sources)
        self.hot_count = len(first.hot_sources)
        self.destination_count = len(first.destinations)
        self.hot_generations, self.error_status = hot_generations, error_status
        self._validate_tensor(hot_generations, (self.hot_count,), torch.int64)
        self._validate_tensor(error_status, (), torch.int32)
        self._tensors = []

        def tensor(values, dtype):
            value = torch.tensor(values, dtype=dtype, device=self.device)
            self._tensors.append(value)
            return value

        self.destination_writes = tensor(
            [0] * (self.destination_count + 1), torch.int64
        )
        self.mask = tensor([False] * capacity, torch.bool)
        self.tables = []
        self.batches = []
        destination_ranges, host_ranges, hot_ranges = [], [], []
        for component in self.components:
            if (
                len(component.host_sources) != self.host_count
                or len(component.hot_sources) != self.hot_count
                or len(component.destinations) != self.destination_count
            ):
                raise ValueError(
                    "All components must describe the same expert/slot counts"
                )
            for region in component.hot_sources + component.destinations:
                if torch.device(region.device) != self.device:
                    raise ValueError(
                        "All component regions must reside on the bound device"
                    )
            host = [
                s.device_pointer(s.host_address, component.nbytes)
                for s in component.host_sources
            ]
            hot = [s.address for s in component.hot_sources]
            destinations = [s.address for s in component.destinations]
            for pointer in host + hot + destinations:
                _range(pointer, component.nbytes)
            host_ranges.extend(_range(p, component.nbytes) for p in host)
            hot_ranges.extend(_range(p, component.nbytes) for p in hot)
            destination_ranges.extend(_range(p, component.nbytes) for p in destinations)
            # A zero sentinel supports empty source pools and invalid device indices
            # without an out-of-bounds gather before fault gating.
            self.tables.append(
                (
                    tensor([0] + host, torch.int64),
                    tensor([0] + hot, torch.int64),
                    tensor([0] + destinations, torch.int64),
                )
            )
            component_batches = []
            for begin in range(0, capacity, max_batch_entries):
                size = min(max_batch_entries, capacity - begin)
                component_batches.append(
                    (
                        begin,
                        begin + size,
                        tensor(
                            [0] * size, torch.int64
                        ),  # source pointers, uncompressed
                        tensor([0] * size, torch.int64),  # packed destination pointers
                        tensor([component.nbytes] * size, torch.uint32),
                        tensor(size, torch.uint32),  # fixed native scan length
                        tensor([0] * size, torch.int64),  # native group mask
                        tensor(
                            [0] * size, torch.int64
                        ),  # packed groups; cleared each replay
                    )
                )
            self.batches.append(tuple(component_batches))
        _reject_overlapping(destination_ranges)
        if any(_overlap(h, d) for h in host_ranges for d in destination_ranges):
            raise ValueError("Mapped immutable host storage overlaps a destination")
        # Tail hot-set updates use HOST only. If targets alias the hot arena, any
        # HOT_SLOT copy is conservatively rejected to prevent batch copy cycles.
        self.hot_targets_alias = any(
            _overlap(h, d) for h in hot_ranges for d in destination_ranges
        )

    def _validate_tensor(self, value, shape, dtype):
        if value.shape != shape or value.dtype != dtype or value.device != self.device:
            raise ValueError(
                "Transfer metadata shape/dtype/device does not match binding"
            )
        if not value.is_contiguous():
            raise ValueError("Transfer metadata must be contiguous")

    def validate_plan(self, plan):
        import torch

        if not isinstance(plan, DeviceCopyPlan) or plan.capacity != self.capacity:
            raise ValueError("Device plan does not match fixed transfer capacity")
        for name in (
            "source_kind",
            "source_index",
            "destination_slot",
            "expected_generation",
        ):
            self._validate_tensor(getattr(plan, name), (self.capacity,), torch.int64)
        self._validate_tensor(plan.valid, (self.capacity,), torch.bool)
        self._validate_tensor(plan.count, (), torch.int32)

    def allocations(self):
        """Native tables/scratch only; owners count weights and canonical state."""
        return tuple(
            Allocation(
                f"{self.name}.metadata.{index}",
                ResourceKind.METADATA,
                value.untyped_storage().nbytes(),
                (str(value.device), value.untyped_storage().data_ptr()),
                value,
            )
            for index, value in enumerate(self._tensors)
            if value.untyped_storage().nbytes()
        )


def _npu_stream_context(stream):
    import torch
    import torch_npu  # noqa: F401

    return torch.npu.stream(stream)


def _npu_completion(stream):
    import torch
    import torch_npu  # noqa: F401

    event = torch.npu.Event()
    event.record(stream)
    return EventCompletion(event)


def _group_pack_copy(*args):
    from memfabric_hybrid import offload

    return offload.group_pack_copy(*args)


class MemFabricDeviceTransferOps(DeviceTransferOps):
    """Lower fixed device plans; callers supply complete-bundle bindings.

    Native statuses report submission only. Completion is recorded after all
    component/batch calls on the caller's copy stream. Generation/index/count
    errors poison shared error_status before any kernel is allowed to write.
    """

    def __init__(
        self,
        *,
        retain,
        group_pack_copy=None,
        record_completion=None,
        stream_context=None,
    ):
        if not callable(retain):
            raise TypeError("A runtime resource-retention callback is required")
        self.retain = retain
        self.group_pack_copy = group_pack_copy or _group_pack_copy
        self.record_completion = record_completion or _npu_completion
        self.stream_context = stream_context or _npu_stream_context

    @staticmethod
    def _mask(plan, bindings):
        import torch

        host = plan.source_kind == int(CopySourceKind.HOST)
        hot = plan.source_kind == int(CopySourceKind.HOT_SLOT)
        source_in_range = (plan.source_index >= 0) & (
            (host & (plan.source_index < bindings.host_count))
            | (hot & (plan.source_index < bindings.hot_count))
        )
        destination_in_range = (plan.destination_slot >= 0) & (
            plan.destination_slot < bindings.destination_count
        )
        if bindings.hot_count:
            generation = bindings.hot_generations.gather(
                0, plan.source_index.clamp(0, bindings.hot_count - 1)
            )
            source_in_range &= (~hot) | (generation == plan.expected_generation)
        if bindings.hot_targets_alias:
            source_in_range &= ~hot
        invalid = (plan.valid & ~(source_in_range & destination_in_range)).any()
        invalid |= plan.count != plan.valid.sum(dtype=torch.int32)
        # Destination slot duplication is unsafe even when split into bounded
        # calls: publication cannot assign two expert identities to one slot.
        bindings.destination_writes.zero_()
        indices = torch.where(destination_in_range, plan.destination_slot + 1, 0)
        bindings.destination_writes.scatter_add_(0, indices, plan.valid.to(torch.int64))
        invalid |= (bindings.destination_writes[1:] > 1).any()
        bindings.error_status.bitwise_or_(invalid.to(torch.int32))
        bindings.mask.copy_(plan.valid & (bindings.error_status == 0))

    def materialize_into(self, plan, bindings, stream):
        import torch

        if not isinstance(bindings, MemFabricTransferBindings):
            raise TypeError("MemFabric requires preallocated MemFabricTransferBindings")
        if bindings.device.type != "npu":
            raise ValueError("MemFabric lowering requires NPU storage")
        stream_device = getattr(stream, "device", None)
        if stream_device is None or torch.device(stream_device) != bindings.device:
            raise ValueError(
                "MemFabric copy stream must belong to the bound NPU device"
            )
        bindings.validate_plan(plan)
        logger.debug(
            "MoE offload memfabric.emit: binding=%s capacity=%s "
            "components=%s stream=%s",
            bindings.name,
            plan.capacity,
            len(bindings.components),
            getattr(stream, "stream_id", None),
        )
        resources = (plan, bindings, self)
        self.retain(resources)
        with self.stream_context(stream):
            try:
                self._mask(plan, bindings)
                # Every weight/scale component uses the same logical copy plan.
                # Select table indices once; entry zero gates inactive/faulted
                # lanes before gathering any component's physical addresses.
                from_host = plan.source_kind == int(CopySourceKind.HOST)
                source_index = torch.where(bindings.mask, plan.source_index + 1, 0)
                host_index = source_index.clamp(0, bindings.host_count)
                hot_index = source_index.clamp(0, bindings.hot_count)
                destination_index = torch.where(
                    bindings.mask, plan.destination_slot + 1, 0
                )
                for batch_index, first in enumerate(bindings.batches[0]):
                    begin, end = first[:2]
                    selected = bindings.mask[begin:end].to(torch.int64)
                    packed_index = (selected.cumsum(0) - 1).clamp_min(0)
                    for tables, batches in zip(bindings.tables, bindings.batches):
                        host_table, hot_table, destination_table = tables
                        _, _, src, dst, lengths, scan, groups, packed = batches[
                            batch_index
                        ]
                        host_ptr = host_table.gather(0, host_index[begin:end])
                        hot_ptr = hot_table.gather(0, hot_index[begin:end])
                        torch.where(from_host[begin:end], host_ptr, hot_ptr, out=src)
                        groups.copy_(selected)
                        destinations = destination_table.gather(
                            0, destination_index[begin:end]
                        )
                        # Native output j corresponds to the j-th selected i.
                        # Zero additions from inactive entries cannot overwrite an
                        # active destination, including entirely empty masks.
                        dst.zero_()
                        dst.scatter_add_(0, packed_index, destinations)
                        packed.zero_()
                        status = self.group_pack_copy(
                            src, dst, lengths, scan, groups, packed, bindings.device
                        )
                        if status is not None and status != 0:
                            raise RuntimeError(
                                f"MemFabric group_pack_copy failed with status {status}"
                            )
                return _ticket(self.record_completion, stream, resources, self.retain)
            except Exception as error:
                raise _submission_failure(
                    error, self.record_completion, stream, resources, self.retain
                ) from error
