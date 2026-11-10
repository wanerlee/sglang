"""One MemFabric host allocation with explicit ACL registration ownership.

The reference MemFabric LOCAL allocator reserves GiB-sized physical pools. Its
Python ``empty(pin_memory=True)`` ignores that flag, so it cannot certify ACL
asynchronous reads. This owner registers the actual pool slab with ACL and uses
MemFabric's separate HVA-to-DVA lookup. There is no second host weight copy.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from dataclasses import dataclass

from ..resources import Allocation, ResourceKind
from ..storage import HostMemoryCapability

_GIB = 1024**3


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class HostAllocation:
    tensor: object
    capability: HostMemoryCapability


class MemFabricHostPool:
    """Explicitly closed, process-owned LOCAL pool.

    Close only after all streams, graphs and host submitters have been drained
    and all layer views retired. Deliberately no destructor: freeing memory from
    a GC finalizer cannot establish device-reader completion. On unregister
    failure the slab remains allocated and close can be retried safely.
    """

    _lock = threading.RLock()
    _active = None

    def __init__(
        self,
        total_bytes,
        device_index,
        *,
        physical_bytes,
        native_acl=None,
    ):
        import torch

        if type(total_bytes) is not int or total_bytes <= 0:
            raise ValueError("Host slab size must be a positive byte count")
        if type(device_index) is not int or device_index < 0:
            raise ValueError("An explicit NPU device index is required")
        if (
            type(physical_bytes) is not int
            or physical_bytes <= 0
            or physical_bytes % _GIB
        ):
            raise ValueError("MemFabric physical budget must be an explicit whole GiB")
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self.nbytes = _align(total_bytes, self.page_size)
        if self.nbytes > physical_bytes:
            raise ValueError("Registered slab exceeds the physical host budget")
        hicache = sys.modules.get("sglang.srt.mem_cache.pool_host.npu_memfabric")
        hicache_active = getattr(hicache, "_memfabric_state", {}).get(
            "initialized", False
        )
        if (
            hicache_active
            or os.getenv("SGLANG_HICACHE_HOST_MEM_BACKEND", "").lower() == "memfabric"
        ):
            raise ValueError(
                "MoE and HiCache cannot both own the MemFabric offload pool"
            )
        self.physical_bytes = physical_bytes
        self.device_index = device_index
        self._offset = 0
        self._initialized = self._registered = self._closed = False
        self._address = 0
        self._tensor = self._buffer = None
        if native_acl is None:
            from .native_acl import get_native_acl

            native_acl = get_native_acl()
        from memfabric_hybrid import offload

        self.native_acl, self.offload = native_acl, offload
        with self._lock:
            if MemFabricHostPool._active is not None:
                raise RuntimeError("The MemFabric LOCAL pool already has an owner")
            MemFabricHostPool._active = self
            try:
                config = offload.OffloadConfig()
                config.device_id = device_index
                config.reserve_size = config.alloc_size = physical_bytes
                config.world_size = 1
                # Some MemFabric builds expose only world_size for LOCAL pools.
                if hasattr(config, "local_world_size"):
                    config.local_world_size = 1
                config.rank_id = 0
                config.flags = 0
                # config.flags = offload.OFFLOAD_FLAG_GIANT_PAGE
                config.scene = offload.Scene.LOCAL
                status = offload.initialize(config)
                if status not in (None, 0):
                    raise RuntimeError(f"MemFabric initialization failed: {status}")
                self._initialized = True
                self._address = int(offload.malloc(self.nbytes))
                if self._address <= 0 or self._address % self.page_size:
                    raise RuntimeError("MemFabric did not return a page-aligned slab")
                device_address = int(offload.get_dva(self._address))
                last = int(offload.get_dva(self._address + self.nbytes - 1))
                if device_address <= 0 or last != device_address + self.nbytes - 1:
                    raise RuntimeError(
                        "MemFabric did not certify one linear device mapping"
                    )
                # MF already owns the device mapping. Register only PINNED with
                # ACL, avoiding a second mapping and its potentially distinct DVA.
                native_acl.register_host(
                    self._address, self.nbytes, device_index, False
                )
                self._registered = True
                self._buffer = (ctypes.c_ubyte * self.nbytes).from_address(
                    self._address
                )
                # frombuffer's storage retains the buffer; the buffer owns this
                # pool so an escaped Tensor also keeps its registration alive.
                self._buffer._moe_pool = self
                self._tensor = torch.frombuffer(self._buffer, dtype=torch.uint8)
                self.capability = HostMemoryCapability(
                    self._address,
                    self.nbytes,
                    True,
                    device_address,
                    self,
                    allocation_nbytes=physical_bytes,
                )
            except BaseException:
                # Do not free an allocation if unregister itself fails. _active
                # deliberately retains the owner for failure diagnosis/cleanup.
                self.close()
                raise

    def allocate(self, nbytes, alignment=64):
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError("Host component size must be positive")
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("Host alignment must be a positive power of two")
        with self._lock:
            if self._closed:
                raise RuntimeError("Host pool is closed")
            start = _align(self._offset, alignment)
            if start + nbytes > self.nbytes:
                raise MemoryError("The fixed host slab is exhausted")
            self._offset = start + nbytes
            return HostAllocation(
                self._tensor.narrow(0, start, nbytes), self.capability
            )

    @property
    def used_bytes(self):
        return self._offset

    def allocations(self):
        if self._closed:
            raise RuntimeError("Host pool is closed")
        return (
            Allocation(
                "moe.host_pool",
                ResourceKind.HOST_WEIGHTS,
                self.physical_bytes,
                ("cpu", self._address),
                self,
            ),
        )

    def close(self):
        """Release after the caller proves all readers and layer views retired."""
        with self._lock:
            if self._closed:
                return
            if self._registered:
                self.native_acl.unregister_host(self._address, self.device_index)
                self._registered = False
            if self._address:
                self.offload.free(self._address)
                self._address = 0
            if self._initialized:
                self.offload.uninitialize()
                self._initialized = False
            self._tensor = self._buffer = None
            self._closed = True
            if MemFabricHostPool._active is self:
                MemFabricHostPool._active = None
