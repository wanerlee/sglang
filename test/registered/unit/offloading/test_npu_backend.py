"""CPU assembly contracts: real residency, transfer metadata and allocation ledger.

Only native allocation, device events, byte-copy kernels and EP/FFN construction
are simulated. These tests do not claim accelerator or collective correctness.
"""

import ctypes
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
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
from sglang.srt.offloading.moe.npu.backend import NpuMoEBackend
from sglang.srt.offloading.moe.npu.transfer import (
    DeviceMemoryRegion,
    MemFabricAddressCatalog,
    MemFabricDeviceTransferOps,
    MemFabricTransferBindings,
    TransferSubmissionError,
)
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.policy import TensorLFUPolicy
from sglang.srt.offloading.moe.resources import Allocation, ResourceKind
from sglang.srt.offloading.moe.storage import HostMemoryCapability
from sglang.srt.offloading.transfer import (
    HostCopyPlan,
    HostCopyRegion,
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
    layouts = {
        name: SimpleNamespace(nbytes=size) for name, size in component_nbytes.items()
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
        self._sealed = True
        self.component_capabilities = {}
        for component, nbytes in Format.component_nbytes.items():
            view = pool.tensor.narrow(0, offset, spec.local_experts * nbytes)
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
        self.end = offset

    def require_sealed(self):
        if not self._sealed:
            raise RuntimeError("Seal host storage before binding device access")

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

    def clear_cache(self):
        self.plans.clear()

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
            batch_log_interval=0,
            batch_log_capacity=64,
        )
        backend.config, backend.device = self.config, torch.device("cpu")
        backend.model_plan = self.catalog
        backend.model_adapter = SimpleNamespace(
            expert_module=lambda layer: layer.mlp.experts
        )
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

    def address_catalog(self, hosts, hot, sizes, **kwargs):
        return SimpleNamespace(
            hosts=hosts,
            hot=hot,
            sizes=sizes,
            allocations=lambda: (),
            seal=lambda: None,
            **kwargs,
        )

    def bind(self, catalog, destinations, **kwargs):
        components = tuple(
            SimpleNamespace(
                host_sources=catalog.hosts[name],
                hot_sources=catalog.hot[name],
                destinations=destinations[name],
                nbytes=size,
            )
            for name, size in catalog.sizes.items()
        )
        first = components[0]
        result = SimpleNamespace(
            **kwargs,
            components=components,
            address_catalog=catalog,
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

    def host_ops(self, module, retain, **kwargs):
        self.host_arguments = dict(module=module, retain=retain, **kwargs)
        self.host = HostOps(retain)
        return self.host

    def finalize(self, finalize=None):
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    npu_backend,
                    "MemFabricAddressCatalog",
                    side_effect=self.address_catalog,
                )
            )
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
                    native_acl, "AclHostTransferOps", side_effect=self.host_ops
                )
            )
            stack.enter_context(
                mock.patch.object(
                    npu_experts, "build_npu_layer_programs", side_effect=self.build
                )
            )
            if finalize is not None:
                return finalize()
            return self.backend.finalize(
                self.layers, self.catalog, self.config, self.policy
            )

    def close(self):
        if self.backend.runtime is not None:
            # These tests drive planners directly, outside forward_scope. End
            # their synthetic invocation before invoking the real close gate.
            context = self.backend.runtime._context
            if context is not None:
                if context.capturing:
                    # Only the capture flag is simulated here; this fixture
                    # never owns an actual device graph to reset.
                    self.backend.runtime.synchronize()
                    context.dispose_after_drain(graph_destroyed=True)
                elif not context._closed:
                    self.backend.runtime.synchronize()
                    context.dispose_after_drain()
                self.backend.runtime._context = None
            self.backend.runtime.close()
        else:
            self.backend.close_unbound()


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
                self.assertEqual(len(runtime.pipelines), 1)
                pipeline = runtime.pipelines[0]
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

    def test_prefill_reuses_physical_plans_but_not_reservation_state(self):
        assembly = self.assemble()
        runtime = assembly.finalize()
        pipeline = runtime.pipelines[0]
        context = runtime._new_context(Phase.PREFILL)
        runtime._context = context
        wave = ExpertWave(0, 0, 2, 0)
        request = pipeline.make_request(3, wave, 0, context)
        repeated = pipeline.make_request(3, wave, 0, context)
        self.assertIs(request.local_demand.binding, repeated.local_demand.binding)
        first = pipeline.access.stage(request, context)
        pipeline.access.release_after(
            first, context.record_compute_completion(), context
        )
        second = pipeline.access.stage(repeated, context)
        self.assertIs(assembly.host.plans[0], assembly.host.plans[1])
        self.assertIs(first.bindings, second.bindings)
        self.assertIsNot(first.reservations[0].state, second.reservations[0].state)
        self.assertTrue(first.reservations[0].state.retired)
        self.assertFalse(second.reservations[0].state.retired)
        pipeline.access.release_after(
            second, context.record_compute_completion(), context
        )
        # Physical bank views are shared across layers, host source plans are not.
        third = pipeline.access.stage(
            pipeline.make_request(4, wave, 0, context), context
        )
        self.assertIsNot(assembly.host.plans[0], assembly.host.plans[2])
        for tensor in third.bindings.bank.tensors.values():
            self.assertEqual(tensor[:, 0].tolist(), [40, 41])
        pipeline.access.release_after(
            third, context.record_compute_completion(), context
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
        removed._sealed = False
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


@dataclass(frozen=True)
class SimulatedNpuDevice:
    index: int = 0
    type: str = "npu"

    def __str__(self):
        return f"npu:{self.index}"


class TestAddressCatalogMetadata(unittest.TestCase):
    """Real metadata storage and validation, replacing only the NPU allocator."""

    def setUp(self):
        self.device = SimulatedNpuDevice()
        self.owner = object()
        original_device, original_tensor = torch.device, torch.tensor

        def device(value):
            return (
                value
                if isinstance(value, SimulatedNpuDevice)
                else original_device(value)
            )

        def tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(torch, "device", side_effect=device))
        stack.enter_context(mock.patch.object(torch, "tensor", side_effect=tensor))
        stack.enter_context(
            mock.patch.object(torch.Tensor, "device", property(lambda _: self.device))
        )
        self.sizes = {"weight": 64, "scale": 128}
        self.hosts = {
            name: tuple(
                HostMemoryCapability(
                    100000 + c * 4096 + i * 128,
                    size,
                    True,
                    1000 + c * 4096 + i * 128,
                    self.owner,
                )
                for i in range(3)
            )
            for c, (name, size) in enumerate(self.sizes.items())
        }
        self.hot = self.regions(10000, 2)
        self.destinations = self.regions(20000, 2)
        self.catalog = MemFabricAddressCatalog(
            self.hosts, self.hot, self.sizes, device=self.device
        )
        self.generations = torch.tensor([0, 0], dtype=torch.int64)
        self.error = torch.tensor(0, dtype=torch.int32)

    def regions(self, base, count):
        return {
            name: tuple(
                DeviceMemoryRegion(
                    base + c * 4096 + i * 128, size, self.device, self.owner
                )
                for i in range(count)
            )
            for c, (name, size) in enumerate(self.sizes.items())
        }

    def bind(self, destinations=None):
        return MemFabricTransferBindings(
            self.catalog,
            self.destinations if destinations is None else destinations,
            capacity=3,
            hot_generations=self.generations,
            error_status=self.error,
        )

    def test_shared_readonly_tables_and_independent_mutable_scratch(self):
        first, second = self.bind(), self.bind()
        for a, b in zip(first.tables, second.tables):
            self.assertEqual([t.data_ptr() for t in a], [t.data_ptr() for t in b])
        for c, (name, _) in enumerate(self.sizes.items()):
            self.assertEqual(
                first.tables[c][0].tolist(),
                [0] + [source.device_address for source in self.hosts[name]],
            )
            self.assertEqual(
                first.tables[c][2].tolist(),
                [0] + [region.address for region in self.destinations[name]],
            )
        first_scratch = {a.storage_key for a in first.allocations()}
        second_scratch = {a.storage_key for a in second.allocations()}
        self.assertTrue(first_scratch.isdisjoint(second_scratch))
        catalog_allocations = self.catalog.allocations()
        self.assertEqual(
            sum(a.nbytes for a in catalog_allocations), 2 * (4 + 3 + 3) * 8
        )
        self.assertEqual(len({a.storage_key for a in catalog_allocations}), 6)
        self.assertTrue(
            first_scratch.isdisjoint(a.storage_key for a in catalog_allocations)
        )
        first.batches[0][0][2].fill_(987)
        self.assertTrue(torch.all(second.batches[0][0][2] == 0))
        self.assertIs(first.hot_generations, second.hot_generations)
        self.assertIs(first.error_status, second.error_status)

    def test_seal_allows_existing_views_and_rejects_new_physical_tables(self):
        before = self.bind()
        self.catalog.seal()
        after = self.bind()
        self.assertEqual(before.tables[0][2].data_ptr(), after.tables[0][2].data_ptr())
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            self.bind(self.regions(30000, 2))

    def test_destination_validation_precedes_table_allocation(self):
        before = len(self.catalog.allocations())
        invalid = dict(self.destinations)
        invalid["weight"] = (self.destinations["weight"][0],) * 2
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.bind(invalid)
        source = self.hosts["weight"][0]
        invalid["weight"] = (
            DeviceMemoryRegion(source.device_address, 64, self.device, self.owner),
            self.destinations["weight"][1],
        )
        with self.assertRaisesRegex(ValueError, "immutable host"):
            self.bind(invalid)
        invalid = dict(self.destinations)
        invalid["scale"] = invalid["scale"][:1]
        with self.assertRaisesRegex(ValueError, "same destination count"):
            self.bind(invalid)
        self.assertEqual(len(self.catalog.allocations()), before)
        self.assertTrue(self.bind(self.hot).hot_targets_alias)

    def test_source_schema_and_mapping_validation_are_not_repeated_per_binding(self):
        with self.assertRaisesRegex(ValueError, "component schema"):
            MemFabricAddressCatalog({}, self.hot, self.sizes, device=self.device)
        hosts = dict(self.hosts)
        hosts["weight"] = (HostMemoryCapability(1000, 64, True, None, self.owner),) * 3
        with self.assertRaisesRegex(ValueError, "no certified device mapping"):
            MemFabricAddressCatalog(hosts, self.hot, self.sizes, device=self.device)
        self.bind()
        with mock.patch.object(
            HostMemoryCapability,
            "device_pointer",
            side_effect=AssertionError("source validation repeated"),
        ):
            self.bind()


class TestAclHostTransfer(unittest.TestCase):
    def setUp(self):
        self.device = SimulatedNpuDevice()
        self.retained = []
        self.module = SimpleNamespace(
            api_info=lambda: {
                "queue_ordering": "torch_npu_stream_flush",
                "device_synchronize": False,
            },
            Batch=mock.Mock(side_effect=lambda *args: SimpleNamespace(arguments=args)),
            submit=mock.Mock(),
        )
        self.ops = native_acl.AclHostTransferOps(
            self.module, self.retained.append, max_batch_entries=1
        )
        self.stream = SimpleNamespace(device=self.device, stream_id=19)
        self.owner = object()
        self.source = HostMemoryCapability(1000, 256, True, 10000, self.owner)
        self.destination = DeviceMemoryRegion(20000, 256, self.device, self.owner)
        self.plan = HostCopyPlan(
            tuple(
                HostCopyRegion(self.source, self.destination, i, i, 64) for i in (0, 64)
            )
        )
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            mock.patch.object(
                native_acl, "_npu_stream_context", side_effect=lambda _: nullcontext()
            )
        )
        self.completion = stack.enter_context(
            mock.patch.object(
                native_acl, "_npu_completion", return_value=ImmediateCompletion()
            )
        )

    def test_native_descriptor_capacity_is_derived_from_physical_region(self):
        ticket = self.ops.materialize_into(self.plan, None, self.stream)
        self.assertIsInstance(ticket, TransferTicket)
        self.assertEqual(
            self.module.Batch.call_args_list[1].args,
            ((1064,), (20064,), (64,), (192,), 0, "host", 0),
        )
        self.assertEqual(self.module.submit.call_count, 2)
        self.assertIs(self.retained[0][0], self.plan)
        self.assertIs(self.retained[-1], ticket)
        with self.assertRaisesRegex(ValueError, "source allocation"):
            HostCopyRegion(self.source, self.destination, 250, 0, 64)

    def test_partial_submission_retains_complete_plan_and_native_batches(self):
        self.module.submit.side_effect = [None, RuntimeError("submit failed")]
        with self.assertRaises(TransferSubmissionError) as raised:
            self.ops.materialize_into(self.plan, None, self.stream)
        resources = self.retained[0]
        self.assertIs(resources[0], self.plan)
        self.assertEqual(len(resources[2]), 2)
        self.assertIs(raised.exception.completion_ticket, self.retained[-1])
        self.assertIs(resources[0].regions[0].source.registration, self.owner)

    def test_repeated_plan_reuses_native_descriptors_and_records_fresh_completion(self):
        first = self.ops.materialize_into(self.plan, None, self.stream)
        first_batches = self.retained[0][2]
        with mock.patch.object(
            self.ops, "_lower", side_effect=AssertionError("rebuilt")
        ):
            second = self.ops.materialize_into(self.plan, None, self.stream)
        self.assertIsNot(first, second)
        self.assertEqual(self.completion.call_count, 2)
        self.assertEqual(self.module.Batch.call_count, 2)
        self.assertEqual(self.module.submit.call_count, 4)
        self.assertIs(self.retained[2][2], first_batches)
        # Cache hits must still validate the device/stream contract.
        with self.assertRaisesRegex(ValueError, "managed NPU stream"):
            self.ops.materialize_into(
                self.plan, None, SimpleNamespace(device=self.device)
            )

    def test_cache_eviction_preserves_inflight_descriptors_and_clear_rebuilds(self):
        self.ops.max_cached_plans = 1
        first = self.ops.materialize_into(self.plan, None, self.stream)
        original_batches = self.retained[0][2]
        other = HostCopyPlan((self.plan.regions[0],))
        self.ops.materialize_into(other, None, self.stream)
        self.assertEqual(len(self.ops._prepared), 1)
        self.assertIs(first.keepalive[2], original_batches)
        self.ops.materialize_into(self.plan, None, self.stream)
        self.assertEqual(self.module.Batch.call_count, 5)
        self.ops.clear_cache()
        self.assertEqual(len(self.ops._prepared), 0)
        self.ops.materialize_into(self.plan, None, self.stream)
        self.assertEqual(self.module.Batch.call_count, 7)

    def test_disabled_descriptor_cache_revalidates_every_submission(self):
        self.ops.max_cached_plans = 0
        for _ in range(2):
            self.ops.materialize_into(self.plan, None, self.stream)
        self.assertEqual(len(self.ops._prepared), 0)
        self.assertEqual(self.module.Batch.call_count, 4)

    def test_failed_completion_record_keeps_presubmission_retention(self):
        self.module.submit.side_effect = RuntimeError("submit failed")
        self.completion.side_effect = RuntimeError("stream poisoned")
        with self.assertRaises(TransferSubmissionError) as raised:
            self.ops.materialize_into(self.plan, None, self.stream)
        self.assertIsNone(raised.exception.completion_ticket)
        self.assertEqual(len(self.retained), 1)
        self.assertIs(self.retained[0][0], self.plan)

    def test_invalid_aliases_and_stream_are_rejected_before_submission(self):
        for destination in (
            self.destination,
            DeviceMemoryRegion(10000, 256, self.device, self.owner),
        ):
            regions = (HostCopyRegion(self.source, destination, 0, 0, 64),) * 2
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.ops.materialize_into(HostCopyPlan(regions), None, self.stream)
        mapped = DeviceMemoryRegion(10000, 256, self.device, self.owner)
        with self.assertRaisesRegex(ValueError, "immutable host"):
            self.ops.materialize_into(
                HostCopyPlan((HostCopyRegion(self.source, mapped, 0, 0, 64),)),
                None,
                self.stream,
            )
        with self.assertRaisesRegex(ValueError, "managed NPU stream"):
            self.ops.materialize_into(
                self.plan, None, SimpleNamespace(device=self.device)
            )
        self.assertEqual(self.module.submit.call_count, 0)
        self.assertEqual(self.retained, [])


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
