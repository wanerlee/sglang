"""Pinned native expert storage, independent of CUDA UVA or NPU registration."""

from types import MappingProxyType

from .resources import Allocation, ResourceKind


class HostExpertStore:
    """Own caller-created host bundles without copying or changing their format.

    Tensors are writable during checkpoint loading/repacking. seal() freezes the
    bindings; contents are logically immutable afterward. A registration object
    owns any backend host-memory registration and stays alive with this store.
    Physical allocations are counted once, even when bundle views share storage.
    """

    def __init__(self):
        self._components = {}
        self._registrations = []
        self._sealed = False

    def add(self, layer_id, name, tensor):
        if self._sealed:
            raise RuntimeError("Host expert storage is sealed")
        key = (layer_id, name)
        if key in self._components:
            raise ValueError("Duplicate expert component")
        if tensor.device.type != "cpu" or not tensor.is_pinned():
            raise ValueError("Host expert components must use pinned CPU storage")
        self._components[key] = tensor

    def retain_registration(self, registration):
        if self._sealed:
            raise RuntimeError("Register host memory before sealing graph bindings")
        self._registrations.append(registration)

    def seal(self):
        if not self._components:
            raise ValueError("Cannot seal empty expert storage")
        self._sealed = True

    @property
    def components(self):
        return MappingProxyType(self._components)

    def allocations(self):
        if not self._sealed:
            raise RuntimeError(
                "Seal host storage before finalizing its allocation ledger"
            )
        seen, allocations = set(), []
        for (layer_id, name), tensor in self._components.items():
            storage = tensor.untyped_storage()
            identity = ("cpu", storage.data_ptr())
            if identity in seen:
                continue
            seen.add(identity)
            allocations.append(
                Allocation(
                    f"host.{layer_id}.{name}",
                    ResourceKind.HOST_WEIGHTS,
                    storage.nbytes(),
                    identity,
                    self,
                )
            )
        return tuple(allocations)
