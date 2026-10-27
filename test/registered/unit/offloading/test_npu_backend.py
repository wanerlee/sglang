"""CPU assembly contracts: real residency, transfer metadata and allocation ledger.

Only native allocation, device events, byte-copy kernels and EP/FFN construction
are simulated. These tests do not claim accelerator or collective correctness.
"""

import ctypes
import sys
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.offloading.moe.config import (
    ExpertCatalog,
    ExpertLayerSpec,
    ExpertWave,
    MoEOffloadConfig,
    Phase,
)
from sglang.srt.offloading.moe.execution import ExecutionRequest
from sglang.srt.offloading.moe.npu import backend as npu_backend
from sglang.srt.offloading.moe.npu import experts as npu_experts
from sglang.srt.offloading.moe.npu import native_acl
from sglang.srt.offloading.moe.npu import storage as npu_storage
from sglang.srt.offloading.moe.npu.backend import NpuMoEBackend
from sglang.srt.offloading.moe.npu.transfer import (
    DeviceMemoryRegion,
    MemFabricCopyComponent,
    MemFabricDeviceTransferOps,
    MemFabricTransferBindings,
    _group_pack_copy,
)
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.policy import TensorLFUPolicy
from sglang.srt.offloading.moe.resources import Allocation, ResourceKind
from sglang.srt.offloading.moe.storage import HostExpertStore, HostMemoryCapability
from sglang.srt.offloading.transfer import (
    DeviceCopyPlan,
    ImmediateCompletion,
    TransferTicket,
)


@dataclass
class QuantInfo:
    w13_weight: object
    w2_weight: object
    w13_weight_scale: object
    w2_weight_scale: object


class Bank:
    def __init__(self, tensors, kind, name, owner=None):
        self.tensors, self.kind, self.name, self.owner = tensors, kind, name, owner
        self.capacity = next(iter(tensors.values())).shape[0]
        self.component_regions = {
            name: tuple(DeviceMemoryRegion.from_tensor(row) for row in tensor)
            for name, tensor in tensors.items()
        }

    def view(self, start, count):
        if start < 0 or count < 0 or start + count > self.capacity:
            raise ValueError("Bank slice exceeds physical allocation")
        return Bank(
            {
                name: tensor.narrow(0, start, count)
                for name, tensor in self.tensors.items()
            },
            self.kind,
            self.name,
            owner=self,
        )

    def quant_info(self):
        return QuantInfo(*(self.tensors[name] for name in Format.component_nbytes))

    def allocations(self):
        if self.owner is not None:
            return ()
        return tuple(
            Allocation(
                f"{self.name}.{name}",
                self.kind,
                tensor.untyped_storage().nbytes(),
                (str(tensor.device), tensor.untyped_storage().data_ptr()),
                tensor,
            )
            for name, tensor in self.tensors.items()
            if tensor.numel()
        )


class Format:
    component_nbytes = {
        "w13_weight": 8,
        "w2_weight": 8,
        "w13_weight_scale_inv": 4,
        "w2_weight_scale_inv": 4,
    }
    expert_nbytes = sum(component_nbytes.values())

    def __init__(self):
        self.banks = []

    def allocate_bank(self, capacity, kind, name):
        bank = Bank(
            {
                key: torch.zeros(capacity, size, dtype=torch.uint8)
                for key, size in self.component_nbytes.items()
            },
            kind,
            name,
        )
        self.banks.append(bank)
        return bank


class HostPool:
    def __init__(self, nbytes, events):
        self.tensor = torch.zeros(nbytes, dtype=torch.uint8)
        self.events = events
        self.capability = HostMemoryCapability(
            self.tensor.data_ptr(), nbytes, True, self.tensor.data_ptr(), self
        )

    def allocations(self):
        return (
            Allocation(
                "host.master",
                ResourceKind.HOST_WEIGHTS,
                self.tensor.numel(),
                ("cpu", self.tensor.data_ptr()),
                self,
            ),
        )

    def close(self):
        self.events.append("pool.close")


class Storage:
    def __init__(self, spec, pool, offset, events):
        self.spec, self.events = spec, events
        self.host_store = HostExpertStore()
        self.component_capabilities = {}
        for component, nbytes in Format.component_nbytes.items():
            view = pool.tensor.narrow(0, offset, spec.local_experts * nbytes)
            self.host_store.add(
                spec.layer_id, component, view, capability=pool.capability
            )
            capabilities = []
            for local in range(spec.local_experts):
                # Distinct complete expert bytes make incorrect layer flattening
                # or bank offsets visible to copy tests.
                view[local * nbytes : (local + 1) * nbytes].fill_(
                    spec.layer_id * 10 + local
                )
                address = view.data_ptr() + local * nbytes
                capabilities.append(
                    HostMemoryCapability(address, nbytes, True, address, pool)
                )
            self.component_capabilities[component] = tuple(capabilities)
            offset += view.numel()
        self.host_store.seal()
        self.end = offset

    def close(self):
        self.events.append(f"storage.{self.spec.layer_id}.close")


class Device:
    def __init__(self, events):
        self.events, self.copy_stream, self.compute = events, object(), object()

    def current_stream(self):
        return self.compute

    def record(self, stream):
        return TransferTicket(ImmediateCompletion(), (stream,))

    def stream_scope(self, stream):
        return nullcontext()

    def synchronize(self):
        self.events.append("device.synchronize")


class HostOps:
    def __init__(self, retain):
        self.retain, self.plans = retain, []

    def materialize_into(self, plan, bindings, stream):
        self.retain((plan, bindings))
        self.plans.append(plan)
        for region in plan.regions:
            source, destination = region.source, region.destination
            source.validate_range(
                source.host_address + region.source_offset, region.nbytes
            )
            if region.destination_offset + region.nbytes > destination.nbytes:
                raise ValueError("Physical destination overflow")
            ctypes.memmove(
                destination.address + region.destination_offset,
                source.host_address + region.source_offset,
                region.nbytes,
            )
        return TransferTicket(ImmediateCompletion(), (plan, bindings))


class DeviceOps:
    """Test-only byte transport; production never enables CPU MemFabric."""

    def __init__(self, retain):
        self.retain = retain

    def materialize_into(self, plan, binding, stream):
        assert plan.count.dtype == torch.int32
        self.retain((plan, binding))
        MemFabricDeviceTransferOps._mask(plan, binding)
        for lane in range(plan.capacity):
            if not bool(binding.mask[lane]):
                continue
            index = int(plan.source_index[lane])
            destination = int(plan.destination_slot[lane])
            for component in binding.components:
                if int(plan.source_kind[lane]) == 1:
                    source = component.host_sources[index].host_address
                else:
                    source = component.hot_sources[index].address
                ctypes.memmove(
                    component.destinations[destination].address,
                    source,
                    component.nbytes,
                )
        return TransferTicket(ImmediateCompletion(), (plan, binding))


class Assembly:
    """Native seams are scoped; all metadata/ownership code is the real code."""

    def __init__(self, ratio=0.5, slots=4, group=2, layer_count=3, local_experts=5):
        self.events, self.bindings, self.builds = [], [], []
        self.catalog = ExpertCatalog(
            tuple(
                ExpertLayerSpec(layer, local_experts * 2, 2, 1)
                for layer in range(3, 3 + layer_count)
            )
        )
        self.config = MoEOffloadConfig(
            ratio, slots, 2, 1, workspace_limit_bytes=2**20, metadata_limit_bytes=2**20
        )
        self.policy = TensorLFUPolicy()
        backend = self.backend = NpuMoEBackend.__new__(NpuMoEBackend)
        backend.settings = SimpleNamespace(
            prefill_group_size=group,
            prefill_microbatch_size=1,
            prefill_max_inflight=2,
            max_dispatch_rows=32,
        )
        backend.config, backend.device = self.config, torch.device("cpu")
        backend.format = Format()
        backend.host_pool = HostPool(
            self.catalog.num_experts * Format.expert_nbytes, self.events
        )
        backend.device_runtime = Device(self.events)
        backend.layers, backend._unbound_resources = {}, []
        backend.runtime, backend._closed, backend.native_acl = None, False, object()
        self.layers, offset = [], 0
        for spec in self.catalog.offloaded_layers:
            storage = Storage(spec, backend.host_pool, offset, self.events)
            offset = storage.end
            backend.layers[spec.layer_id] = storage
            expert = SimpleNamespace(
                layer_id=spec.layer_id, _moe_offload_storage=storage
            )
            self.layers.append(
                (SimpleNamespace(mlp=SimpleNamespace(experts=expert)), spec)
            )
        self.shared_workspace = torch.zeros(32, dtype=torch.uint8)
        self.fail_layer = None

    def bind(self, *args, **kwargs):
        components = tuple(args[0])
        first = components[0]
        result = SimpleNamespace(
            **kwargs,
            components=components,
            host_count=len(first.host_sources),
            hot_count=len(first.hot_sources),
            destination_count=len(first.destinations),
            hot_targets_alias=first.hot_sources == first.destinations,
            mask=torch.zeros(kwargs["capacity"], dtype=torch.bool),
            destination_writes=torch.zeros(
                len(first.destinations) + 1, dtype=torch.int64
            ),
            allocations=lambda: (),
        )
        self.bindings.append(result)
        return result

    def build(self, experts, **kwargs):
        if experts.layer_id == self.fail_layer:
            raise RuntimeError("injected EP builder failure")
        shared = kwargs["shared_layout"]
        layout = (
            shared
            if shared is not None
            else SimpleNamespace(workspace=self.shared_workspace)
        )
        allocation = Allocation(
            "dispatch.shared",
            ResourceKind.WORKSPACE,
            self.shared_workspace.numel(),
            ("cpu", self.shared_workspace.data_ptr()),
            self.shared_workspace,
        )
        result = SimpleNamespace(
            prefill=object(),
            decode=object(),
            layout=layout,
            allocations=lambda: (allocation,),
            **{key: value for key, value in kwargs.items() if key != "shared_layout"},
        )
        self.builds.append((experts.layer_id, result, shared))
        return result

    def host_ops(self, **kwargs):
        self.host_arguments = kwargs
        self.host = HostOps(kwargs["retain"])
        return self.host

    def finalize(self):
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    npu_backend, "MemFabricTransferBindings", side_effect=self.bind
                )
            )
            stack.enter_context(
                mock.patch.object(
                    npu_backend,
                    "MemFabricDeviceTransferOps",
                    side_effect=lambda **kwargs: DeviceOps(**kwargs),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    native_acl, "make_acl_host_ops", side_effect=self.host_ops
                )
            )
            stack.enter_context(
                mock.patch.object(
                    npu_experts, "build_npu_layer_programs", side_effect=self.build
                )
            )
            return self.backend.finalize(
                self.layers, self.catalog, self.config, self.policy
            )

    def close(self):
        if self.backend.runtime is not None:
            self.backend.runtime.close()
        else:
            self.backend.close_unbound()


class MemFabricHostPoolTests(unittest.TestCase):
    def test_initialization_requests_giant_pages_for_local_pool(self):
        offload = SimpleNamespace(
            OffloadConfig=SimpleNamespace,
            OFFLOAD_FLAG_GIANT_PAGE=1,
            Scene=SimpleNamespace(LOCAL="local"),
            # Stop at the native allocation boundary; no real pool is allocated.
            initialize=mock.Mock(return_value=23),
        )
        with (
            mock.patch.dict(
                sys.modules, {"memfabric_hybrid": SimpleNamespace(offload=offload)}
            ),
            mock.patch.dict(
                npu_storage.os.environ, {"SGLANG_HICACHE_HOST_MEM_BACKEND": ""}
            ),
            mock.patch.object(
                npu_storage.os, "sysconf", return_value=4096, create=True
            ),
            mock.patch.object(npu_storage, "_check_host_registration_platform"),
            self.assertRaisesRegex(RuntimeError, "MemFabric initialization failed: 23"),
        ):
            npu_storage.MemFabricHostPool(
                4096, 3, physical_bytes=48 * 1024**3, native_acl=mock.Mock()
            )

        offload.initialize.assert_called_once()
        config = offload.initialize.call_args.args[0]
        self.assertEqual(config.flags, offload.OFFLOAD_FLAG_GIANT_PAGE)
        self.assertEqual(config.rank_id, 0)
        self.assertEqual(config.device_id, 3)
        self.assertEqual((config.world_size, config.local_world_size), (1, 1))
        self.assertEqual(config.scene, offload.Scene.LOCAL)
        self.assertEqual(config.reserve_size, 48 * 1024**3)
        self.assertEqual(config.alloc_size, config.reserve_size)
        self.assertIsNone(npu_storage.MemFabricHostPool._active)


class NpuBackendAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.assemblies = []

    def tearDown(self):
        for assembly in self.assemblies:
            assembly.close()

    def assemble(self, **kwargs):
        assembly = Assembly(**kwargs)
        self.assemblies.append(assembly)
        return assembly

    def test_physical_resources_are_counted_once_across_layers_and_bank_views(self):
        for ratio in (0.0, 0.25, 0.75, 1.0):
            with self.subTest(ratio=ratio):
                assembly = self.assemble(ratio=ratio)
                runtime = assembly.finalize()
                owner = MoEOffloadOwner(assembly.config, assembly.catalog, runtime)
                usage = owner.allocation_ledger.usage
                self.assertEqual(
                    usage[ResourceKind.HOST_WEIGHTS], 15 * Format.expert_nbytes
                )
                self.assertEqual(
                    usage[ResourceKind.HOT_WEIGHTS],
                    int(15 * ratio) * Format.expert_nbytes,
                )
                self.assertEqual(
                    usage[ResourceKind.EXECUTION_WEIGHTS], 4 * Format.expert_nbytes
                )
                self.assertEqual(usage[ResourceKind.WORKSPACE], 32)
                self.assertEqual(len(assembly.backend.format.banks), 2)
                self.assertEqual(len(assembly.bindings), 7)  # tail + 2 waves per layer
                self.assertEqual(
                    [x.capacity for x in assembly.bindings[1:]], [4, 1] * 3
                )
                self.assertTrue(
                    all(
                        result.layout is assembly.builds[0][1].layout
                        for _, result, _ in assembly.builds
                    )
                )
                self.assertEqual(assembly.host_arguments["source_location"], "host")
                self.assertIs(
                    assembly.host_arguments["module"], assembly.backend.native_acl
                )

    def test_all_bindings_share_fault_and_generation_state_and_catalog_order(self):
        assembly = self.assemble()
        runtime = assembly.finalize()
        for binding in assembly.bindings:
            self.assertIs(binding.error_status, runtime.ops.error_status)
            self.assertIs(binding.hot_generations, runtime.residency.slot_generation)
            for name, component in zip(Format.component_nbytes, binding.components):
                expected = tuple(
                    cap
                    for spec in assembly.catalog.offloaded_layers
                    for cap in assembly.backend.layers[
                        spec.layer_id
                    ].component_capabilities[name]
                )
                self.assertEqual(component.host_sources, expected)
        for _, program, _ in assembly.builds:
            self.assertIs(program.residency, runtime.ops)

    def test_prefill_bank_offsets_copy_correct_layer_and_combine_adjacent_views(self):
        for ratio in (0.0, 0.5):
            with self.subTest(ratio=ratio):
                assembly = self.assemble(ratio=ratio)
                runtime = assembly.finalize()
                context = runtime._new_context(Phase.PREFILL)
                runtime._context = context
                pipeline = runtime.pipelines[1]
                access = pipeline.access
                leases = []
                for expert in range(2):
                    wave = ExpertWave(expert, expert, expert + 1, 0)
                    request = pipeline.make_request(4, wave, expert, context)
                    lease = access.stage(request, context)
                    leases.append(lease)
                    for tensor in lease.bindings.bank.tensors.values():
                        self.assertTrue(torch.all(tensor == 40 + expert))
                whole = ExpertWave(0, 0, 2, 0)
                combined = pipeline.combine_bindings(4, whole, leases, context)
                self.assertEqual(combined.bank.capacity, 2)
                for tensor in combined.bank.tensors.values():
                    self.assertEqual(tensor[:, 0].tolist(), [40, 41])
                with self.assertRaisesRegex(ValueError, "adjacent"):
                    pipeline.combine_bindings(4, whole, list(reversed(leases)), context)
                a = combined.bank
                b = pipeline.make_request(
                    4, ExpertWave(1, 2, 4, 1), 0, context
                ).local_demand.binding.bank
                for name in Format.component_nbytes:
                    a_ranges = {
                        (region.address, region.nbytes)
                        for region in a.component_regions[name]
                    }
                    b_ranges = {
                        (region.address, region.nbytes)
                        for region in b.component_regions[name]
                    }
                    self.assertTrue(a_ranges.isdisjoint(b_ranges))
                for lease in leases:
                    access.release_after(
                        lease, context.record_compute_completion(), context
                    )

    def test_decode_transfer_and_restore_reuse_fixed_bank_storage(self):
        assembly = self.assemble(ratio=0.25)
        runtime = assembly.finalize()
        context = runtime._new_context(Phase.DECODE, capturing=True)
        runtime._context = context
        runtime.ops.begin_step(context)
        access = runtime.pipelines[0].access
        targets = torch.tensor([0, 5, 14], dtype=torch.int64)
        access.apply_update(targets, context)
        hot, execution = assembly.backend.format.banks
        before = {name: tensor.data_ptr() for name, tensor in hot.tensors.items()}
        for tensor in hot.tensors.values():
            self.assertEqual(tensor[:, 0].tolist(), [30, 40, 54])
        wave = assembly.builds[0][1].decode_waves[0]
        lease = access.stage(
            ExecutionRequest(3, torch.tensor([2, 0, 0, 1, 0]), wave, Phase.DECODE),
            context,
        )
        for tensor in execution.tensors.values():
            self.assertEqual(tensor[[0, 3], 0].tolist(), [30, 33])
        self.assertEqual(runtime.residency.slot_to_key.tolist(), [0, 5, 14])
        access.release_after(lease, context.record_compute_completion(), context)
        runtime.reload_weights(
            SimpleNamespace(
                slot_to_key=torch.tensor([4, 9, 10]),
                slot_valid=torch.ones(3, dtype=torch.bool),
            ),
            context,
        )
        for name, tensor in hot.tensors.items():
            self.assertEqual(tensor[:, 0].tolist(), [34, 44, 50])
            self.assertEqual(tensor.data_ptr(), before[name])
        self.assertEqual(int(runtime.ops.error_status), 0)

    def test_finalize_failure_drains_before_retiring_host_views_and_pool(self):
        assembly = self.assemble()
        assembly.fail_layer = 4
        with self.assertRaisesRegex(RuntimeError, "EP builder"):
            assembly.finalize()
        self.assertFalse(assembly.backend._closed)
        self.assertIsNone(assembly.backend.runtime)
        self.assertEqual(assembly.events, [])
        assembly.backend.close_unbound()
        self.assertEqual(
            assembly.events,
            [
                "device.synchronize",
                "storage.3.close",
                "storage.4.close",
                "storage.5.close",
                "pool.close",
            ],
        )
        self.assertEqual(assembly.backend._unbound_resources, [])
        assembly.backend.close_unbound()
        self.assertEqual(len(assembly.events), 5)

    def test_incomplete_or_unsealed_host_master_fails_before_bank_allocation(self):
        assembly = self.assemble()
        removed = assembly.backend.layers.pop(5)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            assembly.finalize()
        self.assertEqual(assembly.backend.format.banks, [])
        assembly.backend.layers[5] = removed
        removed.host_store._sealed = False
        with self.assertRaisesRegex(RuntimeError, "Seal"):
            assembly.finalize()
        self.assertEqual(assembly.backend.format.banks, [])

    def test_insufficient_prefill_storage_rejects_then_cleanup_is_safe(self):
        assembly = self.assemble(ratio=0.0, slots=3, group=2)
        with self.assertRaisesRegex(ValueError, "Prefill requires"):
            assembly.finalize()
        assembly.backend.close_unbound()
        self.assertEqual(assembly.events[-1], "pool.close")

    def test_zero_hot_pool_restore_has_no_hidden_weight_allocation(self):
        assembly = self.assemble(ratio=0.0)
        runtime = assembly.finalize()
        context = runtime._new_context(Phase.DECODE, capturing=True)
        runtime._context = context
        runtime.ops.begin_step(context)
        runtime.pipelines[0].access.apply_update(
            torch.empty(0, dtype=torch.int64), context
        )
        runtime.reload_weights(
            SimpleNamespace(
                slot_to_key=torch.empty(0, dtype=torch.int64),
                slot_valid=torch.empty(0, dtype=torch.bool),
            ),
            context,
        )
        self.assertEqual(assembly.backend.format.banks[0].allocations(), ())
        self.assertEqual(len(assembly.backend.format.banks), 2)
        self.assertEqual(int(runtime.ops.error_status), 0)

    def test_restore_uses_shared_sticky_fault_to_suppress_all_component_writes(self):
        assembly = self.assemble(ratio=0.25)
        runtime = assembly.finalize()
        context = runtime._new_context(Phase.DECODE, capturing=True)
        runtime._context = context
        runtime.ops.error_status.fill_(7)
        hot = assembly.backend.format.banks[0]
        for tensor in hot.tensors.values():
            tensor.fill_(199)
        runtime.reload_weights(
            SimpleNamespace(
                slot_to_key=torch.tensor([0, 5, 14]),
                slot_valid=torch.ones(3, dtype=torch.bool),
            ),
            context,
        )
        self.assertEqual(int(runtime.ops.error_status), 7)
        self.assertTrue(
            all(bool(torch.all(tensor == 199)) for tensor in hot.tensors.values())
        )


class TestMemFabricPublicWrapper(unittest.TestCase):
    def test_passes_tensors_to_public_offload_wrapper(self):
        package = ModuleType("memfabric_hybrid")
        native = mock.Mock(return_value=0)
        tensors = tuple(torch.zeros(1, dtype=torch.int64) for _ in range(6))
        device = SimpleNamespace(index=5)

        def public_wrapper(*args):
            for actual, expected in zip(args[:6], tensors):
                self.assertIs(actual, expected)
            self.assertIs(args[6], device)
            return native(*(value.data_ptr() for value in args[:6]), args[6].index)

        package.offload = SimpleNamespace(group_pack_copy=public_wrapper)
        # The internal module must not be imported: use the package's public API.
        with mock.patch.dict(
            sys.modules,
            {"memfabric_hybrid": package, "memfabric_hybrid.mf_acc_offload": None},
        ):
            self.assertEqual(_group_pack_copy(*tensors, device), 0)
        native.assert_called_once_with(*(value.data_ptr() for value in tensors), 5)


class TestMemFabricDeviceLowering(unittest.TestCase):
    """Exercise real bindings/lowering with CPU storage and a byte-copy stand-in."""

    def bind(self, *, components=4, capacity=5, host_count=3, hot_count=2):
        device = SimpleNamespace(type="npu", index=0)
        bundles = []
        for component in range(components):
            width = component + 3
            host = torch.arange(host_count * width, dtype=torch.uint8).reshape(
                host_count, width
            )
            hot = torch.full((hot_count, width), 100 + component, dtype=torch.uint8)
            destination = torch.full((4, width), 231, dtype=torch.uint8)
            bundles.append(
                MemFabricCopyComponent(
                    tuple(
                        HostMemoryCapability(
                            row.data_ptr(), width, True, row.data_ptr(), host
                        )
                        for row in host
                    ),
                    tuple(DeviceMemoryRegion.from_tensor(row) for row in hot),
                    tuple(DeviceMemoryRegion.from_tensor(row) for row in destination),
                    width,
                )
            )
        tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return tensor(*args, **kwargs)

        def validate(value, shape, dtype):
            self.assertEqual(value.shape, shape)
            self.assertEqual(value.dtype, dtype)
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(value.is_contiguous())

        # Only emulate NPU allocation/device identity, not address preparation.
        with (
            mock.patch.object(torch, "device", return_value=device),
            mock.patch.object(torch, "tensor", side_effect=cpu_tensor),
            mock.patch.object(
                MemFabricTransferBindings, "_validate_tensor", side_effect=validate
            ),
        ):
            binding = MemFabricTransferBindings(
                bundles,
                capacity=capacity,
                hot_generations=torch.arange(hot_count, dtype=torch.int64),
                error_status=torch.zeros((), dtype=torch.int32),
                device=device,
                max_batch_entries=2,
            )
        binding._validate_tensor = validate
        return binding

    def plan(self):
        # Sparse selections cross batches; unused lanes deliberately have bad IDs.
        return DeviceCopyPlan(
            torch.tensor([1, 0, 2, 0, 1]),
            torch.tensor([2, -999, 1, 999, 0]),
            torch.tensor([3, -1, 1, 999, 0]),
            torch.tensor([True, False, True, False, True]),
            torch.tensor([-1, -1, 1, -1, -1]),
            torch.tensor(3, dtype=torch.int32),
            5,
        )

    def run_copy(self, plan, binding):
        def native(src, dst, lengths, scan, groups, packed, device):
            self.assertIs(device, binding.device)
            self.assertTrue(bool((packed == 0).all()))
            selected = [i for i in range(int(scan)) if int(groups[i])]
            self.assertTrue(bool((dst[len(selected) :] == 0).all()))
            for j, i in enumerate(selected):
                ctypes.memmove(int(dst[j]), int(src[i]), int(lengths[i]))
            # Native output must be cleared before the next submission/replay.
            packed.fill_(17)
            return 0

        ops = MemFabricDeviceTransferOps(
            retain=lambda value: None,
            group_pack_copy=native,
            record_completion=lambda stream: ImmediateCompletion(),
            stream_context=lambda stream: nullcontext(),
        )
        stream = SimpleNamespace(device=binding.device)
        with mock.patch.object(torch, "device", return_value=binding.device):
            return ops.materialize_into(plan, binding, stream)

    def test_mixed_sources_sparse_destinations_and_changed_plan(self):
        binding, plan = self.bind(), self.plan()
        addresses = tuple(t.data_ptr() for t in binding._tensors)
        for _ in range(2):
            self.run_copy(plan, binding)
            for component in binding.components:
                for lane in range(plan.capacity):
                    if not bool(plan.valid[lane]):
                        continue
                    index = int(plan.source_index[lane])
                    source = (
                        component.host_sources[index].host_address
                        if int(plan.source_kind[lane]) == 1
                        else component.hot_sources[index].address
                    )
                    destination = component.destinations[
                        int(plan.destination_slot[lane])
                    ].address
                    self.assertEqual(
                        ctypes.string_at(destination, component.nbytes),
                        ctypes.string_at(source, component.nbytes),
                    )
                self.assertEqual(
                    ctypes.string_at(
                        component.destinations[2].address, component.nbytes
                    ),
                    bytes([231]) * component.nbytes,
                )
            plan.source_index[0] = 1
            plan.source_kind[2] = 1
            plan.source_index[2] = 2
        self.assertEqual(addresses, tuple(t.data_ptr() for t in binding._tensors))
        self.assertEqual(int(binding.error_status), 0)

    def test_empty_plan_clears_previously_populated_metadata(self):
        binding, plan = self.bind(), self.plan()
        self.run_copy(plan, binding)
        plan.valid.zero_()
        plan.count.zero_()
        plan.source_index.fill_(2**62)
        plan.destination_slot.fill_(-999)
        self.run_copy(plan, binding)
        for batches in binding.batches:
            for _, _, src, dst, _, _, groups, _ in batches:
                self.assertTrue(bool((src == 0).all()))
                self.assertTrue(bool((dst == 0).all()))
                self.assertTrue(bool((groups == 0).all()))
        self.assertEqual(int(binding.error_status), 0)

    def test_invalid_plan_suppresses_all_component_writes(self):
        for fault in (
            "source",
            "destination",
            "generation",
            "count",
            "duplicate",
            "kind",
            "latched",
        ):
            with self.subTest(fault=fault):
                binding, plan = self.bind(), self.plan()
                if fault == "source":
                    plan.source_index[0] = 999
                elif fault == "destination":
                    plan.destination_slot[0] = -1
                elif fault == "generation":
                    plan.expected_generation[2] = 999
                elif fault == "count":
                    plan.count.fill_(4)
                elif fault == "duplicate":
                    plan.destination_slot[2] = plan.destination_slot[0]
                elif fault == "kind":
                    plan.source_kind[0] = 999
                else:
                    binding.error_status.fill_(1)
                self.run_copy(plan, binding)
                self.assertNotEqual(int(binding.error_status), 0)
                for component in binding.components:
                    for region in component.destinations:
                        self.assertEqual(
                            ctypes.string_at(region.address, region.nbytes),
                            bytes([231]) * region.nbytes,
                        )

    def test_empty_source_pools_and_zero_capacity(self):
        for host_count, hot_count in ((0, 2), (3, 0), (0, 0)):
            with self.subTest(host_count=host_count, hot_count=hot_count):
                binding = self.bind(host_count=host_count, hot_count=hot_count)
                plan = self.plan()
                plan.valid.zero_()
                plan.count.zero_()
                if host_count or hot_count:
                    plan.source_kind[0] = 1 if host_count else 2
                    plan.source_index[0] = 0
                    plan.expected_generation[0] = 0
                    plan.valid[0] = True
                    plan.count.fill_(1)
                self.run_copy(plan, binding)
                self.assertEqual(int(binding.error_status), 0)
        binding = self.bind(capacity=0)
        plan = self.plan()
        empty = DeviceCopyPlan(
            *(
                t[:0]
                for t in (
                    plan.source_kind,
                    plan.source_index,
                    plan.destination_slot,
                    plan.valid,
                    plan.expected_generation,
                )
            ),
            torch.tensor(0, dtype=torch.int32),
            0,
        )
        self.run_copy(empty, binding)

    def test_indices_and_scans_are_shared_across_components(self):
        counts = []
        original_cumsum = torch.Tensor.cumsum
        for components in (1, 4):
            binding, plan = self.bind(components=components), self.plan()
            scans = []

            def cumsum(tensor, *args, **kwargs):
                scans.append(tensor.numel())
                return original_cumsum(tensor, *args, **kwargs)

            with (
                mock.patch.object(torch, "where", wraps=torch.where) as where,
                mock.patch.object(torch.Tensor, "cumsum", cumsum),
            ):
                self.run_copy(plan, binding)
            counts.append((where.call_count, scans))
        # Three shared where calls (fault mask, source and destination indices),
        # plus one address selection per component/batch. Scans stay per batch.
        self.assertEqual(counts, [(6, [2, 2, 1]), (15, [2, 2, 1])])


class TestNPUDispatchBytePacking(unittest.TestCase):
    def layout(self):
        return npu_experts.NPUDispatchLayout(
            2, 256, 8, "cpu", torch.zeros((), dtype=torch.int32)
        )

    def test_padding_fill_accepts_npu_dtypes_and_reuses_workspace(self):
        layout = self.layout()
        source = torch.arange(256, dtype=torch.int64).to(torch.uint8).reshape(8, 32)
        indices = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3])
        original_fill = torch.Tensor.masked_fill_

        def npu_fill(tensor, mask, value):
            if tensor.dtype == torch.uint8:
                raise RuntimeError("selfRef not implemented for DT_UINT8")
            return original_fill(tensor, mask, value)

        with mock.patch.object(torch.Tensor, "masked_fill_", npu_fill):
            result = layout._gather_bytes(
                source, indices, torch.tensor(3), "scale_bytes"
            )

        expected = source.index_select(0, indices)
        expected[3:].zero_()
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.data_ptr(), layout._owned["scale_bytes"].data_ptr())

    def test_activation_and_scale_payload_bits_and_padding(self):
        layout = self.layout()
        indices = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3])
        dtypes = [
            torch.uint8,
            torch.int8,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
        if hasattr(torch, "float8_e8m0fnu"):
            dtypes.append(torch.float8_e8m0fnu)
        for name, width in (("input_bytes", 256), ("scale_bytes", 32)):
            payload = (
                torch.arange(8 * width).reshape(8, width) * 17
                + torch.arange(8).unsqueeze(1) * 31
            ).to(torch.uint8)
            for dtype in dtypes:
                source = payload.view(dtype)
                for count in (0, 3, 8):
                    with self.subTest(workspace=name, dtype=dtype, count=count):
                        storage = layout._owned[name]
                        storage.fill_(171)
                        result = layout._gather_bytes(
                            source, indices, torch.tensor(count), name
                        )
                        expected = payload.index_select(0, indices)
                        expected[count:].zero_()
                        self.assertEqual(result.dtype, dtype)
                        self.assertEqual(result.shape, source.shape)
                        self.assertEqual(result.data_ptr(), storage.data_ptr())
                        self.assertTrue(torch.equal(result.view(torch.uint8), expected))
                        self.assertTrue(torch.equal(source.view(torch.uint8), payload))
                        self.assertTrue(bool((storage[payload.numel() :] == 171).all()))


if __name__ == "__main__":
    unittest.main()
