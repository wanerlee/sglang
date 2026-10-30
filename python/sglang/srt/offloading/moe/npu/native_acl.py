"""Build/load the header-matched ACL shim and bind it to eager prefill.

Building is explicit and requires the installed CANN SDK and torch_npu. Importing
this module on CPU does not import either library or start a build. A successful
host-queue flush provides enqueue ordering, not device completion or graph proof.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import os
import shlex
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from ...transfer import HostCopyPlan, HostTransferOps
from ..storage import HostMemoryCapability
from .transfer import (
    DeviceMemoryRegion,
    _npu_completion,
    _npu_stream_context,
    _range,
    _ranges_overlap,
    _reject_overlapping,
    _submission_failure,
    _ticket,
)

logger = logging.getLogger(__name__)

_MODULE_NAME = "_sglang_moe_acl"
_NATIVE = Path(__file__).with_name("native")
_LOAD_LOCK = RLock()
_LOADED = {}


def probe_batch_abi(include_directory, *, compiler=None, work_directory=None):
    """Compile exact function-pointer casts; resolve SDK overloads without guessing.

    Prefer the C form with failIndex. The nine-argument C++ form is accepted only
    if the installed header itself declares it with the expected full signature.
    Linking is left to the real extension build; compile probes cannot certify a
    shared library or hardware implementation.
    """
    command = shlex.split(compiler or os.environ.get("CXX", "c++"))
    if not command:
        raise ValueError("A C++ compiler command is required")
    errors = []
    with tempfile.TemporaryDirectory(dir=work_directory, prefix="acl-abi-") as tmp:
        tmp = Path(tmp)
        for arity in (10, 9):
            source = tmp / f"probe_{arity}.cpp"
            failure_parameter = "size_t*, " if arity == 10 else ""
            source.write_text(
                "#include <acl/acl.h>\n#include <cstddef>\n"
                "using Fn = aclError (*)(void**, size_t*, void**, size_t*, size_t, "
                "aclrtMemcpyBatchAttr*, size_t*, size_t, "
                + failure_parameter
                + "aclrtStream);\n"
                "Fn checked = static_cast<Fn>(&aclrtMemcpyBatchAsync);\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                command
                + [
                    "-std=c++17",
                    "-I",
                    str(include_directory),
                    "-c",
                    str(source),
                    "-o",
                    str(tmp / f"probe_{arity}.o"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return arity
            errors.append(f"{arity} arguments: {result.stderr[-3000:]}")
    raise RuntimeError(
        "Installed CANN headers expose no supported aclrtMemcpyBatchAsync ABI. "
        "No ctypes/dynamic-symbol fallback is allowed.\n" + "\n".join(errors)
    )


def build_native_acl(build_directory, *, cann_home=None, verbose=False):
    """Explicitly compile/link against this process's torch/torch_npu and CANN."""
    import torch_npu
    from torch.utils.cpp_extension import load

    home = Path(
        cann_home
        or os.environ.get("ASCEND_HOME_PATH")
        or os.environ.get("ASCEND_TOOLKIT_HOME")
        or "/usr/local/Ascend/ascend-toolkit/latest"
    ).resolve()
    include = home / "include"
    library = home / "lib64"
    if not (include / "acl" / "acl.h").is_file():
        raise RuntimeError(f"CANN SDK header missing: {include / 'acl/acl.h'}")
    if not (library / "libascendcl.so").is_file():
        raise RuntimeError(
            f"CANN runtime library missing: {library / 'libascendcl.so'}"
        )
    build_directory = Path(build_directory).resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    abi = probe_batch_abi(include, work_directory=build_directory)
    npu = Path(torch_npu.__file__).resolve().parent
    module = load(
        name=_MODULE_NAME,
        sources=[str(_NATIVE / "acl_batch.cpp")],
        build_directory=str(build_directory),
        extra_include_paths=[
            str(include),
            str(npu / "include"),
            str(npu / "include" / "third_party" / "hccl" / "inc"),
        ],
        extra_cflags=[
            "-std=c++17",
            "-fvisibility=hidden",
            f"-DSGLANG_ACL_BATCH_ABI={abi}",
        ],
        extra_ldflags=[
            f"-L{library}",
            "-lascendcl",
            f"-L{npu / 'lib'}",
            "-ltorch_npu",
            f"-Wl,-rpath,{library}",
            f"-Wl,-rpath,{npu / 'lib'}",
        ],
        verbose=verbose,
    )
    with _LOAD_LOCK:
        _LOADED[str(build_directory)] = module
    return module


def get_native_acl(build_directory=None):
    """Load a built extension; absence never triggers compilation or eager fallback."""
    # torch_npu loads its matching shared libraries and device registration first.
    import torch_npu  # noqa: F401

    key = str(Path(build_directory).resolve()) if build_directory is not None else None
    with _LOAD_LOCK:
        if key in _LOADED:
            return _LOADED[key]
        module = _load_native_acl(key)
        _LOADED[key] = module
        return module


def _load_native_acl(directory):
    if directory is None:
        try:
            return importlib.import_module(_MODULE_NAME)
        except ImportError as error:
            raise RuntimeError(
                "ACL extension is unbuilt. Run python -m "
                "sglang.srt.offloading.moe.npu.native_acl --build-directory PATH "
                "on the target CANN/torch_npu host and pass that directory."
            ) from error
    directory = Path(directory)
    candidates = [
        directory / (_MODULE_NAME + suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (directory / (_MODULE_NAME + suffix)).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one built ACL extension in " + str(directory)
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load ACL extension: " + str(candidates[0]))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AclHostTransferOps(HostTransferOps):
    """Validate eager bundles and retain native descriptors through queued H2D.

    The module certifies torch_npu host-queue ordering, not device completion.
    Source location describes the allocator, independently of its registration.
    Runtime/pipeline grants the stream one submitting thread at a time.
    """

    def __init__(
        self,
        module,
        retain,
        *,
        source_location="host",
        source_location_id=0,
        max_batch_entries=256,
        max_cached_plans=1024,
    ):
        if not callable(retain):
            raise TypeError("A durable runtime retention callback is required")
        if source_location not in ("host", "host_numa", "unregistered"):
            raise ValueError("Unsupported ACL source allocation location")
        if type(source_location_id) is not int or not 0 <= source_location_id < 2**32:
            raise ValueError("ACL source location ID must fit uint32")
        if type(max_batch_entries) is not int or max_batch_entries <= 0:
            raise ValueError("ACL batch capacity must be positive")
        if type(max_cached_plans) is not int or max_cached_plans < 0:
            raise ValueError("ACL descriptor cache capacity must be nonnegative")
        info = module.api_info()
        if info.get("queue_ordering") != "torch_npu_stream_flush" or info.get(
            "device_synchronize"
        ):
            raise ValueError(
                "Loaded ACL shim lacks the expected host-queue ordering contract"
            )
        self.module, self.retain = module, retain
        self.source_location, self.source_location_id = (
            source_location,
            source_location_id,
        )
        self.max_batch_entries = max_batch_entries
        self.max_cached_plans = max_cached_plans
        self._prepared = OrderedDict()
        self._submission_lock = RLock()

    def clear_cache(self):
        """Drop cached address owners after the runtime has drained its streams."""
        with self._submission_lock:
            self._prepared.clear()

    def prepare(self, plan, device):
        """Validate/lower immutable descriptors before serving, without copying."""
        if not isinstance(plan, HostCopyPlan):
            raise TypeError("ACL descriptor preparation requires HostCopyPlan")
        with self._submission_lock:
            self._prepared_batches(plan, device)

    def _prepared_batches(self, plan, device):
        # HostCopyPlan/regions/capabilities are immutable. Keep the plan alive in
        # each entry so its identity cannot be reused for unrelated addresses.
        key = (device.index, id(plan))
        cached = self._prepared.get(key)
        if cached is not None:
            self._prepared.move_to_end(key)
            return cached[1]
        batches = self._lower(plan, device)
        if self.max_cached_plans:
            self._prepared[key] = (plan, batches)
            if len(self._prepared) > self.max_cached_plans:
                # Eviction drops only cache ownership; in-flight tickets still
                # own every descriptor and source/destination allocation.
                self._prepared.popitem(last=False)
        return batches

    def _lower(self, plan, device):
        entries, destinations, mapped_sources = [], [], []
        for region in plan.regions:
            source, destination = region.source, region.destination
            if (
                not isinstance(source, HostMemoryCapability)
                or not source.acl_async_readable
            ):
                raise ValueError(
                    "ACL source requires certified asynchronous host reads"
                )
            if (
                not isinstance(destination, DeviceMemoryRegion)
                or destination.device != device
            ):
                raise ValueError(
                    "ACL destination must belong to the submitting stream device"
                )
            src = source.host_address + region.source_offset
            dst = destination.address + region.destination_offset
            # Region construction checked immutable capacities; registration and
            # address ABI certification belong to this concrete transport.
            source.validate_range(src, region.nbytes)
            _range(src, region.nbytes)
            destinations.append(_range(dst, region.nbytes))
            if source.device_address is not None:
                mapped_sources.append(
                    _range(source.device_pointer(src, region.nbytes), region.nbytes)
                )
            if region.nbytes:
                entries.append(
                    (
                        src,
                        dst,
                        region.nbytes,
                        destination.nbytes - region.destination_offset,
                    )
                )
        _reject_overlapping(destinations)
        if _ranges_overlap(mapped_sources, destinations):
            raise ValueError("Copy destination overlaps mapped immutable host storage")
        batches = []
        for begin in range(0, len(entries), self.max_batch_entries):
            chunk = entries[begin : begin + self.max_batch_entries]
            batches.append(
                self.module.Batch(
                    *(tuple(entry[index] for entry in chunk) for index in range(4)),
                    device.index,
                    self.source_location,
                    self.source_location_id,
                )
            )
        return tuple(batches)

    def materialize_into(self, plan, bindings, stream):
        # Native Batch reuses a mutable fail_index while the GIL is released.
        # Serialize submissions, including cache lookup and completion recording.
        with self._submission_lock:
            return self._materialize_into(plan, bindings, stream)

    def _materialize_into(self, plan, bindings, stream):
        if not isinstance(plan, HostCopyPlan):
            raise TypeError("ACL transfers require HostCopyPlan")
        device = getattr(stream, "device", None)
        index, stream_id = (
            getattr(device, "index", None),
            getattr(stream, "stream_id", None),
        )
        if (
            getattr(device, "type", None) != "npu"
            or type(index) is not int
            or not 0 <= index <= 127
            or type(stream_id) is not int
        ):
            raise ValueError(
                "Native ACL requires a managed NPU stream with an explicit device"
            )
        batches = self._prepared_batches(plan, device)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "MoE offload acl.submit: batches=%s regions=%s bytes=%s stream=%s",
                len(batches),
                len(plan.regions),
                sum(region.nbytes for region in plan.regions),
                stream_id,
            )
        # Each region owns both allocations. Native descriptors and the complete
        # plan survive even a partial submission that raises without a ticket.
        resources = (plan, bindings, batches, stream, self)
        self.retain(resources)
        with _npu_stream_context(stream):
            try:
                for batch in batches:
                    self.module.submit(batch, stream_id)
                return _ticket(_npu_completion, stream, resources, self.retain)
            except Exception as error:
                raise _submission_failure(
                    error, _npu_completion, stream, resources, self.retain
                ) from error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the CANN/torch_npu ACL batch extension"
    )
    parser.add_argument("--build-directory", required=True)
    parser.add_argument("--cann-home")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    result = build_native_acl(
        arguments.build_directory,
        cann_home=arguments.cann_home,
        verbose=arguments.verbose,
    )
    print(result.api_info())
