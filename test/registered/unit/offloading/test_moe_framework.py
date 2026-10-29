"""CPU contracts for the MoE framework; accelerator graph validation is separate."""

import copy
import gc
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.offloading.base import BaseOffloader
from sglang.srt.offloading.moe.config import (
    ExpertCatalog,
    ExpertLayerSpec,
    MoEOffloadConfig,
    Phase,
    fixed_waves,
)
from sglang.srt.offloading.moe.interfaces import (
    ForwardCompletion,
    MoEBackend,
    RuntimeCapabilities,
)
from sglang.srt.offloading.moe.offloader import DeepSeekV4Adapter, MoEOffloaderV2
from sglang.srt.offloading.moe.owner import MoEOffloadOwner, OwnerState
from sglang.srt.offloading.moe.periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy
from sglang.srt.offloading.moe.policy import TensorLFUPolicy
from sglang.srt.offloading.moe.programs import DispatchOnceProgram
from sglang.srt.offloading.moe.resources import (
    Allocation,
    AllocationLedger,
    ResourceKind,
)
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.moe.storage import HostMemoryCapability
from sglang.srt.offloading.transfer import (
    Completion,
    ImmediateCompletion,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def config(ratio=0.5):
    return MoEOffloadConfig(ratio, execution_slots=2, ep_size=2, ep_rank=1, tp_size=2)


def catalog():
    return ExpertCatalog(tuple(ExpertLayerSpec(i, 8, 2, 1, i < 3) for i in range(5)))


def immediate():
    return TransferTicket(ImmediateCompletion())


def completed_step(ticket=None):
    return ForwardCompletion(immediate(), immediate(), ticket or immediate())


class DelayedCompletion(Completion):
    def __init__(self):
        self.ready = False
        self.waits = []
        self.syncs = 0

    def wait_on(self, stream):
        self.waits.append(stream)

    def synchronize(self):
        self.syncs += 1
        self.ready = True

    def done(self):
        return self.ready


class FakeRuntime:
    """Metadata-only runtime double; never presented as a hardware backend."""

    capabilities = RuntimeCapabilities(True, True, True)
    expert_nbytes = 16
    prefill_program = object()
    decode_program = object()

    def __init__(self, cfg=None, cat=None):
        cfg, cat = cfg or config(), cat or catalog()
        self.resources = tuple(
            Allocation(kind.value, kind, slots * self.expert_nbytes, kind, object())
            for kind, slots in (
                (ResourceKind.HOST_WEIGHTS, cat.num_experts),
                (ResourceKind.HOT_WEIGHTS, cfg.hot_slots(cat)),
                (ResourceKind.EXECUTION_WEIGHTS, cfg.execution_slots),
            )
        )
        self.actions = []
        self.metadata = {
            "slots": [7, 2, -1, 5],
            "lfu": [1, 4, 8],
            "generation": [3, 2, 0, 6],
        }
        self.weights = ["expert7", "expert2", None, "expert5"]
        self.saved = None
        self.fail_restore = False

    def install_model_hooks(self, offloader, model_runner):
        self.actions.append("install")

    def forward_scope(self, owner, forward_batch):
        return nullcontext()

    def emit_step_begin(self, context):
        pass

    def emit_layer_start(self, layer_id, context):
        pass

    def emit_after_router(self, layer_id, routing, context):
        pass

    def emit_step_end(self, context):
        pass

    def snapshot_decode_state(self):
        self.actions.append("snapshot")
        self.saved = copy.deepcopy(self.metadata)
        return immediate()

    def borrow_hot_arena(self):
        self.actions.append("borrow")

    def restore_decode_state(self):
        self.actions.append("restore")
        if self.fail_restore:
            raise RuntimeError("transfer failed after submission")
        self.metadata = copy.deepcopy(self.saved)
        self.weights = [
            None if key < 0 else f"expert{key}" for key in self.saved["slots"]
        ]
        return immediate()

    def synchronize(self):
        self.actions.append("synchronize")

    def close(self):
        self.synchronize()
        self.close_after_drain()

    def close_after_drain(self):
        self.actions.append("close")


class FakeBackend(MoEBackend):
    def __init__(self):
        self.prepared = []
        self.closed = 0

    def prepare_layer(self, module, spec):
        self.prepared.append(spec.layer_id)

    def finalize(self, layers, cat, cfg, policy):
        self.runtime = FakeRuntime(cfg, cat)
        return self.runtime

    def close_unbound(self):
        self.closed += 1


class IdentityAdapter:
    def describe(self, spec):
        return spec


class TestIdentityAndBudget(unittest.TestCase):
    def test_rank_local_capacity_excludes_resident_layers(self):
        cat = catalog()
        self.assertEqual(cat.num_experts, 8)
        self.assertEqual(config().hot_slots(cat), 4)
        self.assertEqual(cat.key(3, 3), 3)
        self.assertEqual(cat.key(4, 0), 4)
        with self.assertRaises(ValueError):
            cat.key(0, 0)

    def test_ep_rejects_foreign_experts(self):
        layer = catalog().layers[-1]
        self.assertEqual(layer.local_id(4), 0)
        with self.assertRaises(ValueError):
            layer.local_id(3)

    def test_config_rejects_unsupported_topology_and_capacity(self):
        for kwargs in (
            dict(cache_ratio=float("nan")),
            dict(cache_ratio=1.1),
            dict(execution_slots=0),
            dict(moe_tp_size=2),
            dict(ep_size=1, ep_rank=0, tp_size=2),
        ):
            args = dict(cache_ratio=0.5, execution_slots=2, ep_size=2, ep_rank=1)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MoEOffloadConfig(**args)

    def test_catalog_rejects_mixed_ranks_and_duplicate_layers(self):
        for layers in (
            (ExpertLayerSpec(3, 8, 2, 0), ExpertLayerSpec(4, 8, 2, 1)),
            (ExpertLayerSpec(3, 8, 2, 1), ExpertLayerSpec(3, 8, 2, 1)),
        ):
            with self.assertRaises(ValueError):
                ExpertCatalog(layers)

    def test_budget_rejects_hidden_duplicate_storage_and_overspend(self):
        a = Allocation("bank", ResourceKind.EXECUTION_WEIGHTS, 16, "storage", object())
        b = Allocation("view", ResourceKind.EXECUTION_WEIGHTS, 16, "storage", object())
        with self.assertRaises(ValueError):
            AllocationLedger({ResourceKind.EXECUTION_WEIGHTS: 32}, (a, b))
        with self.assertRaises(ValueError):
            AllocationLedger({ResourceKind.EXECUTION_WEIGHTS: 8}, (a,))

    def test_owner_requires_complete_weight_accounting(self):
        runtime = FakeRuntime()
        runtime.resources = ()
        with self.assertRaises(ValueError):
            MoEOffloadOwner(config(), catalog(), runtime)

    def test_host_capability_requires_owner_and_bounded_mapping(self):
        with self.assertRaises(ValueError):
            HostMemoryCapability(1000, 64, True, 2000, None)
        capability = HostMemoryCapability(1000, 64, True, 2000, object())
        self.assertEqual(capability.device_pointer(1016, 48), 2016)
        with self.assertRaises(ValueError):
            capability.device_pointer(1016, 49)
        unmapped = HostMemoryCapability(1000, 64, True, None, object())
        with self.assertRaises(ValueError):
            unmapped.device_pointer(1000, 1)

    def test_residency_metadata_is_accounted_and_starts_invalid(self):
        state = ResidencyState.allocate(8, 4, 2, "cpu")
        self.assertEqual(state.key_to_slot.tolist(), [-1] * 8)
        self.assertFalse(state.update_enabled.item())
        allocations = state.allocations()
        total = sum(item.nbytes for item in allocations)
        allocation_ledger = AllocationLedger(
            {ResourceKind.METADATA: total}, allocations
        )
        self.assertEqual(allocation_ledger.usage[ResourceKind.METADATA], total)

    def test_deepseek_adapter_keeps_hash_and_first_three_layers_resident(self):
        for layer_id, is_hash, expected in (
            (0, False, True),
            (2, True, True),
            (3, False, False),
        ):
            experts = SimpleNamespace(
                moe_tp_size=1,
                _num_global_routed=8,
                moe_ep_size=2,
                moe_ep_rank=1,
                _num_local_routed=4,
            )
            mlp = SimpleNamespace(
                is_deepseek_v4=True, layer_id=layer_id, is_hash=is_hash, experts=experts
            )
            spec = DeepSeekV4Adapter().describe(SimpleNamespace(mlp=mlp))
            self.assertEqual(spec.resident, expected)
            experts.num_fused_shared_experts = 1
            with self.assertRaises(ValueError):
                DeepSeekV4Adapter().describe(SimpleNamespace(mlp=mlp))


class TestLifetime(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.owner = MoEOffloadOwner(config(), catalog(), self.runtime)

    def test_no_concurrent_forward_or_transition(self):
        token = self.owner.begin_forward(Phase.DECODE, "compute")
        with self.assertRaises(RuntimeError):
            self.owner.begin_forward(Phase.DECODE, "compute")
        with self.assertRaises(RuntimeError):
            self.owner.switch_phase(Phase.PREFILL)
        with self.assertRaises(RuntimeError):
            self.owner.close()
        self.owner.finish_forward(token, completed_step())
        with self.assertRaises(ValueError):
            self.owner.finish_forward(token, completed_step())

    def test_next_step_waits_on_idle_without_host_synchronization(self):
        delayed = DelayedCompletion()
        token = self.owner.begin_forward(Phase.DECODE, "compute")
        self.owner.finish_forward(token, completed_step(TransferTicket(delayed)))
        token = self.owner.begin_forward(Phase.DECODE, "next_compute")
        self.assertEqual(delayed.waits, ["next_compute"])
        self.assertEqual(delayed.syncs, 0)
        self.owner.finish_forward(token, completed_step())
        self.owner.close()
        self.assertEqual(delayed.syncs, 0)
        self.assertIn("synchronize", self.runtime.actions)

    def test_transfer_retains_source_until_idle(self):
        class Source:
            pass

        source = Source()
        reference = weakref.ref(source)
        delayed = DelayedCompletion()
        token = self.owner.begin_forward(Phase.DECODE, None)
        self.owner.finish_forward(
            token, completed_step(TransferTicket(delayed, (source,)))
        )
        del source
        gc.collect()
        self.assertIsNotNone(reference())
        delayed.ready = True
        token = self.owner.begin_forward(Phase.DECODE, None)
        gc.collect()
        self.assertIsNone(reference())
        self.owner.finish_forward(token, completed_step())

    def test_prefill_snapshots_once_and_restores_exact_slots_and_policy(self):
        metadata, weights = (
            copy.deepcopy(self.runtime.metadata),
            list(self.runtime.weights),
        )
        self.owner.switch_phase(Phase.PREFILL)
        self.runtime.metadata["slots"][:] = [-1] * 4
        self.runtime.metadata["lfu"][:] = [0] * 3
        self.runtime.weights[:] = ["temporary"] * 4
        self.owner.switch_phase(Phase.PREFILL)
        self.assertEqual(self.runtime.actions, ["snapshot", "borrow"])
        self.owner.switch_phase(Phase.DECODE)
        self.assertEqual(self.runtime.metadata, metadata)
        self.assertEqual(self.runtime.weights, weights)
        self.owner.switch_phase(Phase.DECODE)
        self.assertEqual(self.runtime.actions.count("restore"), 1)

    def test_transition_drains_inflight_forward(self):
        delayed = DelayedCompletion()
        token = self.owner.begin_forward(Phase.DECODE, None)
        self.owner.finish_forward(token, completed_step(TransferTicket(delayed)))
        self.owner.switch_phase(Phase.PREFILL)
        self.assertEqual(delayed.syncs, 1)

    def test_graph_lease_prevents_teardown(self):
        self.owner.acquire_graph("batch8")
        with self.assertRaises(RuntimeError):
            self.owner.close()
        with self.assertRaises(ValueError):
            self.owner.acquire_graph("batch8")
        self.runtime.synchronize()
        self.owner._release_graph("batch8")
        with self.assertRaises(KeyError):
            self.owner._release_graph("batch8")
        self.owner.close()
        self.owner.close()
        self.assertEqual(self.runtime.actions.count("close"), 1)

    def test_failed_restore_poisoned_until_drained_close(self):
        self.owner.switch_phase(Phase.PREFILL)
        self.runtime.fail_restore = True
        with self.assertRaises(RuntimeError):
            self.owner.switch_phase(Phase.DECODE)
        self.assertIs(self.owner.state, OwnerState.FAILED)
        with self.assertRaises(RuntimeError):
            self.owner.begin_forward(Phase.PREFILL, None)
        self.owner.close()
        self.assertEqual(self.runtime.actions[-2:], ["synchronize", "close"])

    def test_failed_submission_cannot_start_another_forward(self):
        token = self.owner.begin_forward(Phase.DECODE, None)
        self.owner.fail_forward(token)
        with self.assertRaises(RuntimeError):
            self.owner.begin_forward(Phase.DECODE, None)
        self.owner.close()

    def test_failed_drain_retains_owner_until_retry(self):
        synchronize = self.runtime.synchronize

        def fail():
            raise RuntimeError("device stream has not drained")

        self.runtime.synchronize = fail
        with self.assertRaises(RuntimeError):
            self.owner.close()
        self.assertIsNotNone(self.owner.allocation_ledger)
        self.assertIs(self.owner.state, OwnerState.FAILED)
        with self.assertRaises(RuntimeError):
            self.owner.begin_forward(Phase.DECODE, None)
        self.runtime.synchronize = synchronize
        self.owner.close()
        self.assertIsNone(self.owner.allocation_ledger)


class TestLFU(unittest.TestCase):
    def test_global_popularity_counts_misses_and_preserves_resident_ties(self):
        policy = TensorLFUPolicy()
        state = policy.allocate_state(6, "cpu")
        policy.observe(
            state,
            torch.tensor([1, 8, 8, 0, 9, 0]),
            torch.tensor(True),
            torch.empty((), dtype=torch.bool),
        )
        target = torch.empty(3, dtype=torch.int64)
        policy.select_hot_set(state, torch.tensor([-1, -1, 0, -1, -1, -1]), target)
        self.assertEqual(target.tolist(), [4, 2, 1])

    def test_aging_snapshot_restore_and_disabled_updates(self):
        policy = TensorLFUPolicy(aging_interval=2)
        state = policy.allocate_state(3, "cpu")
        saved = policy.allocate_state(3, "cpu")
        policy.observe(
            state,
            torch.tensor([2, 4, 6]),
            torch.tensor(True),
            torch.empty((), dtype=torch.bool),
        )
        policy.copy_state(state, saved)
        policy.observe(
            state,
            torch.tensor([10, 10, 10]),
            torch.tensor(False),
            torch.empty((), dtype=torch.bool),
        )
        self.assertTrue(torch.equal(state.frequency, saved.frequency))
        self.assertTrue(torch.equal(state.steps, saved.steps))
        policy.observe(
            state,
            torch.tensor([0, 2, 0]),
            torch.tensor(True),
            torch.empty((), dtype=torch.bool),
        )
        self.assertEqual(state.frequency.tolist(), [1, 3, 3])
        policy.copy_state(saved, state)
        self.assertEqual(state.frequency.tolist(), [2, 4, 6])

    def test_cold_cache_and_zero_capacity(self):
        policy = TensorLFUPolicy()
        state = policy.allocate_state(3, "cpu")
        for capacity in (0, 3):
            target = torch.empty(capacity, dtype=torch.int64)
            policy.select_hot_set(state, torch.full((3,), -1), target)
            self.assertEqual(target.tolist(), [-1] * capacity)

    def test_matches_python_lfu_over_many_steps(self):
        generator = torch.Generator().manual_seed(7)
        policy = TensorLFUPolicy(aging_interval=4)
        state = policy.allocate_state(13, "cpu")
        counts = [0] * 13
        resident = []
        target = torch.empty(4, dtype=torch.int64)
        for step in range(1, 41):
            accesses = torch.randint(0, 3, (13,), generator=generator)
            counts = [old + new for old, new in zip(counts, accesses.tolist())]
            if step % 4 == 0:
                counts = [value // 2 for value in counts]
            mapping = torch.full((13,), -1)
            for slot, key in enumerate(resident):
                mapping[key] = slot
            policy.observe(
                state, accesses, torch.tensor(True), torch.empty((), dtype=torch.bool)
            )
            policy.select_hot_set(state, mapping, target)
            expected = sorted(
                range(13), key=lambda key: (-counts[key], key not in resident, key)
            )[:4]
            expected = [key if counts[key] else -1 for key in expected]
            self.assertEqual(target.tolist(), expected)
            resident = [key for key in expected if key >= 0]

    def test_frequency_saturates_without_overflow(self):
        policy = TensorLFUPolicy()
        state = policy.allocate_state(1, "cpu")
        state.frequency.fill_(torch.iinfo(torch.int64).max // 4)
        policy.observe(
            state,
            torch.tensor([torch.iinfo(torch.int64).max]),
            torch.tensor(True),
            torch.empty((), dtype=torch.bool),
        )
        self.assertEqual(state.frequency.item(), torch.iinfo(torch.int64).max // 4)


class TestIntegration(unittest.TestCase):
    def test_periodic_policy_binds_after_catalog_registration(self):
        backend = FakeBackend()
        original_finalize = backend.finalize

        def finalize(layers, cat, cfg, policy):
            self.assertIsInstance(policy, PeriodicLFUPolicy)
            self.assertEqual(policy.catalog, cat)
            state = policy.allocate_state(cat.num_experts, "cpu")
            target = torch.empty(cfg.hot_slots(cat), dtype=torch.int64)
            slots = torch.tensor([0, -1, 1, -1, 2, -1, -1, 3])
            # A bound runtime must use this sequence at each valid step tail.
            for _ in range(4):
                policy.observe(
                    state,
                    torch.tensor([0, 3, 0, 0, 0, 2, 0, 0]),
                    torch.tensor(True),
                    torch.empty((), dtype=torch.bool),
                )
                policy.select_hot_set(state, slots, target)
                policy.finish_step(state)
            # Cross-layer LFU replaces tied cold incumbents globally by canonical
            # key order; it does not reserve one eviction within each layer.
            self.assertEqual(target.tolist(), [1, 4, 5, 7])
            return original_finalize(layers, cat, cfg, policy)

        backend.finalize = finalize
        facade = MoEOffloaderV2(
            config(),
            backend,
            adapter=IdentityAdapter(),
            periodic_lfu_config=PeriodicLFUConfig("unused.json"),
        )
        facade.wrap_modules(catalog().layers)
        facade.post_init()
        facade.close()
        with self.assertRaises(ValueError):
            MoEOffloaderV2(
                config(),
                FakeBackend(),
                policy=TensorLFUPolicy(),
                periodic_lfu_config=PeriodicLFUConfig("unused.json"),
            )

    def test_legacy_base_is_noop(self):
        offloader = BaseOffloader()
        self.assertEqual(offloader.wrap_modules(iter([1, 2])), [1, 2])
        offloader.post_init()
        offloader.bind_model_runner(None)
        with offloader.forward_scope(None):
            self.assertFalse(offloader.forbid_copy_engine_usage)

    def test_facade_registers_only_eligible_layers_and_binds_once(self):
        backend = FakeBackend()
        facade = MoEOffloaderV2(config(), backend, adapter=IdentityAdapter())
        layers = list(catalog().layers)
        self.assertEqual(facade.wrap_modules(iter(layers)), layers)
        self.assertEqual(backend.prepared, [3, 4])
        self.assertTrue(facade.forbid_copy_engine_usage)
        facade.post_init()
        with self.assertRaises(RuntimeError):
            facade.forward_scope(None)
        runner = SimpleNamespace(model=torch.nn.Module())
        facade.bind_model_runner(runner)
        self.assertIn("Online weight updates", runner.model._derived_weight_cache_error)
        with facade.forward_scope(None):
            pass
        with self.assertRaises(RuntimeError):
            facade.post_init()
        facade.close()

    def test_missing_graph_backend_fails_before_serving(self):
        backend = FakeBackend()
        facade = MoEOffloaderV2(config(), backend, adapter=IdentityAdapter())
        facade.wrap_modules(catalog().layers)
        original_finalize = backend.finalize

        def finalize(*args):
            runtime = original_finalize(*args)
            runtime.capabilities = RuntimeCapabilities()
            return runtime

        backend.finalize = finalize
        with self.assertRaises(NotImplementedError):
            facade.post_init()
        self.assertEqual(backend.runtime.actions, ["synchronize", "close"])

    def test_partial_initialization_retains_runtime_if_cleanup_cannot_drain(self):
        backend = FakeBackend()
        facade = MoEOffloaderV2(config(), backend, adapter=IdentityAdapter())
        facade.wrap_modules(catalog().layers)
        original_finalize = backend.finalize

        def finalize(*args):
            runtime = original_finalize(*args)
            runtime.capabilities = RuntimeCapabilities()

            def fail():
                raise RuntimeError("device stream has not drained")

            runtime.synchronize = fail
            return runtime

        backend.finalize = finalize
        with self.assertRaises(RuntimeError):
            facade.post_init()
        self.assertIs(facade._runtime, backend.runtime)
        backend.runtime.synchronize = lambda: None
        facade.close()
        self.assertIsNone(facade._runtime)
        self.assertEqual(backend.runtime.actions, ["close"])

    def test_dispatch_once_order_and_single_common_tail(self):
        calls = []

        class Dispatch:
            def dispatch(self, hidden, routing, context):
                calls.append("D")
                return object()

            def combine(self, local, context):
                calls.append("C")
                return local

            def finish(self, result, context):
                calls.append("tail")
                return result

        class Layout:
            def bind(self, dispatched, context):
                return []

            def select_wave(self, layout, wave, context):
                return wave.index

            def write_wave(self, layout, wave, output, context):
                layout.append(output)

            def combine_input(self, layout, context):
                return sum(layout)

        class Local:
            def run(self, layer, dispatched, wave, context):
                calls.append(("F", wave.index))
                return dispatched + 1

        waves = fixed_waves(5, 2, bank_count=2)
        self.assertEqual(
            [(w.local_start, w.local_end, w.bank) for w in waves],
            [(0, 2, 0), (2, 4, 1), (4, 5, 0)],
        )
        program = DispatchOnceProgram(
            waves, Dispatch(), Local(), Layout(), local_experts=5
        )
        context = SimpleNamespace(
            phase=Phase.DECODE, capturing=True, temporary_scope=nullcontext
        )
        self.assertEqual(program.run(3, None, None, context), 6)
        self.assertEqual(calls, ["D", ("F", 0), ("F", 1), ("F", 2), "C", "tail"])


if __name__ == "__main__":
    unittest.main()
