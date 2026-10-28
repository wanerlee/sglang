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
from types import MappingProxyType

from ...transfer import (
    Completion,
    CopySourceKind,
    DeviceCopyPlan,
    DeviceTransferOps,
    EventCompletion,
    TransferTicket,
)
from ..resources import tensor_allocations
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


def _ranges_overlap(first, second):
    """Check two physical interval sets once, without a Cartesian product."""
    first = sorted(r for r in first if r[0] < r[1])
    second = sorted(r for r in second if r[0] < r[1])
    i = j = 0
    while i < len(first) and j < len(second):
        if _overlap(first[i], second[j]):
            return True
        if first[i][1] <= second[j][0]:
            i += 1
        else:
            j += 1
    return False


class MemFabricAddressCatalog:
    """One certified source catalog; bindings supply only physical destinations."""

    def __init__(
        self,
        host_sources,
        hot_sources,
        component_nbytes,
        *,
        device,
        name="moe.addresses",
    ):
        import torch

        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("MemFabric requires an explicit NPU device index")
        if (
            not component_nbytes
            or set(host_sources) != set(component_nbytes)
            or set(hot_sources) != set(component_nbytes)
        ):
            raise ValueError("Complete source bundles must match the component schema")
        self.component_nbytes = MappingProxyType(dict(component_nbytes))
        self.host_sources = tuple(
            tuple(host_sources[name]) for name in component_nbytes
        )
        self.hot_sources = tuple(tuple(hot_sources[name]) for name in component_nbytes)
        self.host_count, self.hot_count = len(self.host_sources[0]), len(
            self.hot_sources[0]
        )
        self.name, self._tensors, self._destinations = name, [], {}
        self._sealed = False
        self._host_ranges, self._hot_ranges = [], []
        tables = []
        for nbytes, hosts, hot in zip(
            self.component_nbytes.values(), self.host_sources, self.hot_sources
        ):
            if type(nbytes) is not int or not 0 <= nbytes <= _UINT32_MAX:
                raise ValueError("MemFabric component size must fit uint32 bytes")
            if (len(hosts), len(hot)) != (self.host_count, self.hot_count):
                raise ValueError("Components must describe the same source counts")
            pointers = []
            for source in hosts:
                if (
                    not isinstance(source, HostMemoryCapability)
                    or not source.acl_async_readable
                ):
                    raise ValueError(
                        "Host sources require certified asynchronous reads"
                    )
                pointers.append(source.device_pointer(source.host_address, nbytes))
            self._validate_device_regions(hot, nbytes)
            self._host_ranges.extend(_range(p, nbytes) for p in pointers)
            self._hot_ranges.extend(_range(r.address, nbytes) for r in hot)
            tables.append(
                (self._table(pointers), self._table([r.address for r in hot]))
            )
        self.source_tables = tuple(tables)

    def _validate_device_regions(self, regions, nbytes):
        import torch

        for region in regions:
            if not isinstance(region, DeviceMemoryRegion):
                raise TypeError("Device storage requires DeviceMemoryRegion")
            if torch.device(region.device) != self.device or nbytes > region.nbytes:
                raise ValueError("Device region has an incompatible device or capacity")

    def _table(self, pointers):
        import torch

        # Sentinel zero makes invalid gathers safe before fault gating.
        value = torch.tensor([0] + pointers, dtype=torch.int64, device=self.device)
        self._tensors.append(value)
        return value

    def bind(self, destinations):
        if set(destinations) != set(self.component_nbytes):
            raise ValueError("Destination bundle differs from the address catalog")
        regions = tuple(tuple(destinations[name]) for name in self.component_nbytes)
        count = len(regions[0])
        for nbytes, component in zip(self.component_nbytes.values(), regions):
            if len(component) != count:
                raise ValueError("Components must describe the same destination count")
            self._validate_device_regions(component, nbytes)
        # Retained owners make physical view identity safe against address reuse.
        key = tuple(
            tuple((r.address, r.nbytes) for r in component) for component in regions
        )
        cached = self._destinations.get(key)
        if cached is None:
            if self._sealed:
                raise RuntimeError(
                    "Address catalog is sealed; bind views before capture"
                )
            ranges = [
                _range(r.address, nbytes)
                for nbytes, component in zip(self.component_nbytes.values(), regions)
                for r in component
            ]
            _reject_overlapping(ranges)
            if _ranges_overlap(self._host_ranges, ranges):
                raise ValueError("Mapped immutable host storage overlaps a destination")
            tables = tuple(
                self._table([r.address for r in component]) for component in regions
            )
            cached = tables, _ranges_overlap(self._hot_ranges, ranges), regions
            self._destinations[key] = cached
        destination_tables, aliases_hot, _ = cached
        return (
            tuple(
                (*sources, target)
                for sources, target in zip(self.source_tables, destination_tables)
            ),
            aliases_hot,
            count,
        )

    def seal(self):
        """Freeze physical table allocations before ledger creation and capture."""
        self._sealed = True

    def allocations(self):
        return tensor_allocations(
            self.name + ".", ((str(i), value) for i, value in enumerate(self._tensors))
        )


class MemFabricTransferBindings:
    """Prebind address tables and native metadata; never construct during replay.

    error_status is the runtime's shared scalar int32 fault flag. Residency
    publication and FFN validity must be gated by it; only a drained/reset runtime
    may clear it. hot_generations is canonical slot-generation state.
    """

    def __init__(
        self,
        address_catalog,
        destinations,
        *,
        capacity,
        hot_generations,
        error_status,
        max_batch_entries=256,
        name=None,
    ):
        import torch

        if type(capacity) is not int or capacity < 0:
            raise ValueError("Plan capacity must be nonnegative")
        if (
            type(max_batch_entries) is not int
            or not 0 < max_batch_entries <= _MAX_GROUP_ENTRIES
        ):
            raise ValueError("MemFabric batch capacity must be in [1, 256]")
        self.address_catalog = address_catalog
        self.device = address_catalog.device
        self.name = name if name is not None else f"memfabric.{id(self):x}"
        self.capacity, self.max_batch_entries = capacity, max_batch_entries
        self.host_count, self.hot_count = (
            address_catalog.host_count,
            address_catalog.hot_count,
        )
        self.tables, self.hot_targets_alias, self.destination_count = (
            address_catalog.bind(destinations)
        )
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
        self.batches = []
        for nbytes in address_catalog.component_nbytes.values():
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
                        tensor([nbytes] * size, torch.uint32),
                        tensor(size, torch.uint32),  # fixed native scan length
                        tensor([0] * size, torch.int64),  # native group mask
                        tensor(
                            [0] * size, torch.int64
                        ),  # packed groups; cleared each replay
                    )
                )
            self.batches.append(tuple(component_batches))

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
        """Only mutable scratch: the arena counts the shared catalog once."""
        return tensor_allocations(
            self.name + ".metadata.",
            ((str(i), value) for i, value in enumerate(self._tensors)),
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
            len(bindings.address_catalog.component_nbytes),
            getattr(stream, "stream_id", None),
        )
        resources = (plan, bindings, self)
        self.retain(resources)
        with self.stream_context(stream):
            try:
                self._mask(plan, bindings)
                # Bundle components share logical indices and validity. Compute
                # them once on the copy stream, while keeping native mutable
                # groups/packed buffers private to each component and batch.
                source_index = plan.source_index + 1
                host_indices = source_index.clamp(0, bindings.host_count)
                hot_indices = source_index.clamp(0, bindings.hot_count)
                destination_indices = (plan.destination_slot + 1).clamp(
                    0, bindings.destination_count
                )
                from_host = plan.source_kind == int(CopySourceKind.HOST)
                packed_indices = tuple(
                    (bindings.mask[begin:end].to(torch.int64).cumsum(0) - 1).clamp_min(0)
                    for begin, end, *_ in bindings.batches[0]
                )
                for tables, batches in zip(bindings.tables, bindings.batches):
                    host_table, hot_table, destination_table = tables
                    for batch, packed_index in zip(batches, packed_indices):
                        begin, end, src, dst, lengths, scan, groups, packed = batch
                        host_ptr = host_table.gather(0, host_indices[begin:end])
                        hot_ptr = hot_table.gather(0, hot_indices[begin:end])
                        selected = bindings.mask[begin:end]
                        src.copy_(
                            torch.where(
                                selected,
                                torch.where(
                                    from_host[begin:end],
                                    host_ptr,
                                    hot_ptr,
                                ),
                                0,
                            )
                        )
                        groups.copy_(selected)
                        destinations = destination_table.gather(
                            0,
                            destination_indices[begin:end],
                        )
                        # Native output j corresponds to the j-th selected i.
                        # Zero additions from inactive entries cannot overwrite an
                        # active destination, including entirely empty masks.
                        dst.zero_()
                        dst.scatter_add_(
                            0, packed_index, torch.where(selected, destinations, 0)
                        )
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
