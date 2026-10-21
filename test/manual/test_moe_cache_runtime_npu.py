"""Real NPU copies and compute for the cache caller, without a serving backend.

    PYTHONPATH=python python test/manual/test_moe_cache_runtime_npu.py -v
    MOE_TEST_NPU_DEVICE=1 PYTHONPATH=python python test/manual/test_moe_cache_runtime_npu.py -v

Requires a matching PyTorch / torch_npu / CANN environment. The eager test
transport reads plans on the host: it is deliberately NOT graph compatible.
No model download, native MoE kernel or multi-rank EP process is needed.
Direct execution exits 2 if hardware tests are skipped, not a successful check.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe import (
    CacheExecutionContext,
    CacheLocalExpertProgram,
    CacheRuntime,
    CopySource,
    ExpertCatalog,
    ExpertLayerSpec,
    LocalRoutes,
    PeriodicLFUConfig,
    PeriodicLFUPolicy,
)
from sglang.srt.offloading.moe.config import fixed_waves
from sglang.srt.offloading.moe.interfaces import RuntimeCapabilities, WaveDispatcher
from sglang.srt.offloading.moe.programs import ComposedDFCProgram
from sglang.srt.offloading.moe.resources import Allocation, ResourceKind
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.transfer import (
    Completion,
    TorchTransferBackend,
    TransferTicket,
    WeightTransferBackend,
)

DIM = 8
INITIAL_KEYS = [0, 2, 4, 7]
REPLACED_KEYS = [1, 2, 5, 7]
L3_ROUTES = ([1, 1, 0], [1, 2, 2], [1, 1, 3, 2], [1, 1, 1, 0, 2, 2, 3, 3])
L4_ROUTES = ([1, 1, 3], [1, 3, 2], [1, 1, 3], [1, 1, 0, 2, 3])


def make_weights():
    """Each key owns two distinct matrices, so copying half a bundle fails."""
    base = torch.arange(DIM * DIM, dtype=torch.float32).reshape(DIM, DIM) / 256
    return torch.stack(
        [
            torch.stack(
                (
                    torch.eye(DIM) * (key + 1) / 8 + base,
                    torch.eye(DIM) * (key + 2) / 8 - base / 2,
                )
            )
            for key in range(8)
        ]
    )


def padded_routes(assignments, batch_size):
    ids = torch.full((batch_size * 2,), -1, dtype=torch.int64)
    ids[: len(assignments)] = torch.tensor(assignments, dtype=torch.int64)
    ids = ids.reshape(batch_size, 2)
    return LocalRoutes(ids, ids >= 0)


def cpu_reference(hidden, routes, weights, offset):
    """Independent token/expert loop, deliberately separate from bank execution."""
    result = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for choice in range(routes.expert_ids.shape[1]):
            if routes.valid[token, choice]:
                key = offset + int(routes.expert_ids[token, choice])
                up, down = weights[key]
                result[token] += torch.relu(hidden[token] @ up) @ down * 0.5
    return result


@dataclass
class CopyAudit:
    pool: str
    entries: list
    before_mapping: torch.Tensor
    expected: list
    waits: int = 0
    after_wait_mapping: object = None
    after_wait_valid: object = None
    after_wait_weights: object = None


class AuditedCompletion(Completion):
    """Use the actual NPU event; audit tensors are read only at test boundaries."""

    def __init__(self, inner, audit, bindings, device_module):
        self.inner, self.audit = inner, audit
        self.bindings, self.device_module = bindings, device_module

    def wait_on(self, stream):
        self.inner.wait_on(stream)
        self.audit.waits += 1
        # These copies enqueue AFTER the real device wait but BEFORE the caller
        # publishes. No .cpu(), .item() or synchronize after H2D submission here.
        with self.device_module.stream(stream):
            self.audit.after_wait_mapping = self.bindings.state.key_to_slot.clone()
            self.audit.after_wait_valid = self.bindings.state.slot_valid.clone()
            self.audit.after_wait_weights = self.bindings.destination.clone()

    def synchronize(self):
        self.inner.synchronize()

    def done(self):
        return self.inner.done()


class EagerNPUTransfer(WeightTransferBackend):
    """Test-only HOST/HOT_SLOT copies using pinned host memory and real events.

    Reading plans on the host adds synchronization BEFORE submission. Never use
    this implementation to claim dynamic graph support or benchmark overlap.
    """

    supports_device_plans = False

    def __init__(self, device_module):
        self.device_module = device_module
        self.audits = []

    def materialize_into(self, plan, bindings, stream):
        # All diagnostic host reads occur before scheduling the weight copies.
        valid = plan.valid.detach().cpu().tolist()
        kinds = plan.source_kind.detach().cpu().tolist()
        sources = plan.source_index.detach().cpu().tolist()
        destinations = plan.destination_slot.detach().cpu().tolist()
        generations = plan.expected_generation.detach().cpu().tolist()
        count = int(plan.count.detach().cpu())
        assert count == sum(valid)
        old_keys = bindings.state.slot_to_key.detach().cpu().tolist()
        old_valid = bindings.state.slot_valid.detach().cpu().tolist()
        old_generation = bindings.state.slot_generation.detach().cpu().tolist()
        entries, expected, copies = [], [], []
        for i in range(plan.capacity):
            if not valid[i]:
                continue
            kind, source, destination = kinds[i], sources[i], destinations[i]
            if kind == CopySource.HOST:
                bundle = bindings.host[source]
                key = source
                assert bundle.device.type == "cpu" and bundle.is_pinned()
            else:
                assert kind == CopySource.HOT_SLOT
                assert old_valid[source] and old_generation[source] == generations[i]
                bundle = bindings.hot[source]
                key = old_keys[source]
                assert bundle.device.type == "npu"
            if bindings.pool == "hot":
                assert not old_valid[destination], (
                    "Victim must be invalid before overwrite"
                )
            assert bindings.destination.device.type == "npu"
            entries.append((kind, source, destination))
            expected.append((destination, bindings.oracle[key].clone()))
            # Copy every component, then record ONE event covering the bundle.
            for component in range(2):
                copies.append(
                    (bundle[component], bindings.destination[destination, component])
                )
        audit = CopyAudit(
            bindings.pool, entries, bindings.state.key_to_slot.clone(), expected
        )
        self.audits.append(audit)
        event = self.device_module.Event()
        ticket = TorchTransferBackend.copy_into(
            copies, device_module=self.device_module, stream=stream, event=event
        )
        completion = AuditedCompletion(
            ticket.completion, audit, bindings, self.device_module
        )
        return TransferTicket(completion, (ticket, bindings, audit))


class ToyDispatcher(WaveDispatcher):
    """Local scripted dispatch only: no HCCL/DeepEP or distributed correctness claim."""

    def mask_routes(self, routing, wave, context):
        return LocalRoutes(
            routing.expert_ids,
            routing.valid
            & (routing.expert_ids >= wave.local_start)
            & (routing.expert_ids < wave.local_end),
        )

    def dispatch(self, hidden_states, routing, context):
        return SimpleNamespace(hidden=hidden_states, routes=routing)

    def combine(self, local_output, context):
        return local_output

    def accumulator(self, hidden_states, context):
        return torch.zeros_like(hidden_states)

    def accumulate(self, accumulator, partial, context):
        accumulator.add_(partial)

    def finish(self, accumulator, context):
        return accumulator


class TestNPUCacheRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch_npu") is None:
            raise unittest.SkipTest(
                "torch_npu is not installed; NPU flow NOT validated"
            )
        import torch_npu

        if not torch.npu.is_available():
            raise unittest.SkipTest(
                "No Ascend device available; NPU flow NOT validated"
            )
        index = int(os.environ.get("MOE_TEST_NPU_DEVICE", "0"))
        torch.npu.set_device(index)
        cls.device = f"npu:{index}"
        print(
            f"NPU cache flow: torch={torch.__version__}, torch_npu={torch_npu.__version__}, device={cls.device}",
            flush=True,
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        profile = Path(self.temporary.name) / "routes.json"
        profile.write_text(
            json.dumps(
                {
                    "model_id": "npu-cache-toy",
                    "layers": {
                        "3": {"4": 100, "6": 90},
                        "4": {"4": 80, "7": 70},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.catalog = ExpertCatalog(tuple(ExpertLayerSpec(i, 8, 2, 1) for i in (3, 4)))
        self.oracle = make_weights()
        self.host = torch.empty(self.oracle.shape, dtype=torch.float32, pin_memory=True)
        self.host.copy_(self.oracle)
        self.assertTrue(self.host.is_pinned())
        self.compute = torch.npu.Stream()
        self.copy = torch.npu.Stream()
        self.assertNotEqual(self.compute, self.copy)
        self.transfer = EagerNPUTransfer(torch.npu)
        self.compute_audits = []
        with torch.npu.stream(self.compute):
            state = ResidencyState.allocate(8, 4, 2, self.device)
            self.hot = torch.full(
                (4, 2, DIM, DIM), torch.nan, dtype=torch.float32, device=self.device
            )
            self.execution = torch.full(
                (2, 2, DIM, DIM), torch.nan, dtype=torch.float32, device=self.device
            )
            bindings = dict(
                host=self.host, hot=self.hot, state=state, oracle=self.oracle
            )
            hot_binding = SimpleNamespace(pool="hot", destination=self.hot, **bindings)
            execution_binding = SimpleNamespace(
                pool="execution", destination=self.execution, **bindings
            )
            resources = tuple(
                Allocation(
                    kind.value,
                    kind,
                    tensor.numel() * tensor.element_size(),
                    (str(tensor.device), tensor.data_ptr()),
                    tensor,
                )
                for kind, tensor in (
                    (ResourceKind.HOST_WEIGHTS, self.host),
                    (ResourceKind.HOT_WEIGHTS, self.hot),
                    (ResourceKind.EXECUTION_WEIGHTS, self.execution),
                )
            )
            self.runtime = CacheRuntime(
                catalog=self.catalog,
                residency=state,
                policy=PeriodicLFUPolicy(PeriodicLFUConfig(str(profile)), self.catalog),
                transfer=self.transfer,
                context=CacheExecutionContext(torch.npu, self.compute, self.copy),
                hot_bindings=hot_binding,
                execution_bindings=execution_binding,
                bank_slots=2,
                capabilities=RuntimeCapabilities(
                    full_decode_graph=False,
                    dynamic_expert_selection=False,
                    canonical_residency=True,
                ),
                expert_nbytes=2 * DIM * DIM * self.host.element_size(),
                resources=resources,
                prefill_program=None,
                install_programs=lambda *args: None,
                borrow_hot_arena=self.borrow_hot,
            )
            self.addCleanup(self.close_runtime)
            self.runtime.decode_program = ComposedDFCProgram(
                fixed_waves(4, 2),
                ToyDispatcher(),
                CacheLocalExpertProgram(
                    self.runtime,
                    lambda layer, output, wave: output.routes,
                    self.native_experts,
                ),
            )
            self.runtime.initialize("npu-cache-toy")
        self.original_addresses = self.addresses()

    def close_runtime(self):
        self.runtime.synchronize()
        self.runtime.close()

    def addresses(self):
        state = self.runtime.residency
        return tuple(
            tensor.data_ptr()
            for tensor in (
                self.hot,
                self.execution,
                state.key_to_slot,
                state.slot_to_key,
                state.slot_valid,
                state.slot_generation,
                self.runtime.cache.policy_state.usage,
            )
        )

    def native_experts(self, layer, dispatched, wave, context):
        hidden, routes = dispatched.hidden, dispatched.routes
        ids = routes.expert_ids.reshape(-1)
        active = routes.valid.reshape(-1)
        lanes = (ids - wave.local_start).clamp(0, 1) + wave.bank * 2
        # Only bank weights on NPU participate in the result, never host oracle.
        selected = self.execution[lanes]
        x = hidden.repeat_interleave(routes.expert_ids.shape[1], dim=0).unsqueeze(1)
        inner = torch.relu(torch.bmm(x, selected[:, 0]))
        output = torch.bmm(inner, selected[:, 1]).squeeze(1)
        output = torch.where(active.unsqueeze(1), output * 0.5, 0)
        state = self.runtime.residency
        self.compute_audits.append(
            (
                active.clone(),
                state.execution_valid[lanes].clone(),
                state.execution_key[lanes].clone(),
                (ids + self.catalog.key(layer.layer_id, 0)).clone(),
            )
        )
        return output.reshape(hidden.shape[0], 2, DIM).sum(dim=1)

    def borrow_hot(self):
        # Stand-in for prefill overwriting borrowed storage; not a prefill model.
        with torch.npu.stream(self.compute):
            self.hot.fill_(-999)
            self.execution.fill_(-777)

    def decode(self, l3, l4, *, enabled=True):
        batch_size = max(1, (max(len(l3), len(l4)) + 1) // 2)
        cpu_hidden = (
            torch.arange(batch_size * DIM, dtype=torch.float32).reshape(batch_size, DIM)
            / 32
        )
        reference = cpu_hidden
        with torch.npu.stream(self.compute):
            hidden = cpu_hidden.to(self.device)
            self.runtime._enabled.fill_(
                enabled
            )  # Host input normally set by forward_scope.
            self.runtime.emit_step_begin(self.runtime.context)
            for layer, assignments in zip(self.catalog.layers, (l3, l4)):
                cpu_routes = padded_routes(assignments, batch_size)
                routes = LocalRoutes(
                    cpu_routes.expert_ids.to(self.device),
                    cpu_routes.valid.to(self.device),
                )
                hidden = self.runtime.decode_program.run(
                    layer, hidden, routes, self.runtime.context
                )
                reference = cpu_reference(
                    reference,
                    cpu_routes,
                    self.oracle,
                    self.catalog.key(layer.layer_id, 0),
                )
            self.runtime.emit_step_end(self.runtime.context)
            self.runtime.context.fence().completion.synchronize()
        torch.testing.assert_close(hidden.cpu(), reference, rtol=1e-3, atol=1e-4)
        self.validate_audits()
        self.assertEqual(self.addresses(), self.original_addresses)
        self.runtime._tickets.clear()  # All test work has completed at the fence.

    def validate_audits(self):
        for audit in self.transfer.audits:
            self.assertGreaterEqual(
                audit.waits, 1, "Caller must wait before publish/read"
            )
            torch.testing.assert_close(
                audit.after_wait_mapping.cpu(), audit.before_mapping.cpu()
            )
            weights = audit.after_wait_weights.cpu()
            for destination, expected in audit.expected:
                # Copies are bit-preserving for both matrices, no tolerance needed.
                torch.testing.assert_close(
                    weights[destination], expected, rtol=0, atol=0
                )
                if audit.pool == "hot":
                    self.assertFalse(bool(audit.after_wait_valid[destination].cpu()))
        for active, valid, keys, expected_keys in self.compute_audits:
            active = active.cpu()
            self.assertTrue(bool(valid.cpu()[active].all()))
            torch.testing.assert_close(keys.cpu()[active], expected_keys.cpu()[active])

    def assert_hot(self, keys):
        state = self.runtime.residency
        self.assertEqual(state.slot_to_key.cpu().tolist(), keys)
        reverse = [-1] * 8
        for slot, key in enumerate(keys):
            reverse[key] = slot
        self.assertEqual(state.key_to_slot.cpu().tolist(), reverse)
        self.assertTrue(bool(state.slot_valid.cpu().all()))
        torch.testing.assert_close(self.hot.cpu(), self.oracle[keys], rtol=0, atol=0)

    def heat_four_steps(self):
        for step in range(4):
            self.decode(L3_ROUTES[step], L4_ROUTES[step])
            self.assert_hot(INITIAL_KEYS if step < 3 else REPLACED_KEYS)

    def test_real_h2d_d2d_lfu_and_next_step_hits(self):
        self.assert_hot(INITIAL_KEYS)
        self.validate_audits()
        self.heat_four_steps()
        state = self.runtime.residency
        self.assertEqual(state.slot_generation.cpu().tolist(), [2, 1, 2, 1])
        torch.testing.assert_close(
            self.runtime.cache.policy_state.usage.cpu(),
            torch.tensor([2, 8, 5, 3, 1, 7, 2, 4]) * 0.7,
        )
        entries = [
            entry
            for audit in self.transfer.audits
            if audit.pool == "execution"
            for entry in audit.entries
        ]
        self.assertTrue(any(kind == CopySource.HOST for kind, _, _ in entries))
        self.assertTrue(any(kind == CopySource.HOT_SLOT for kind, _, _ in entries))
        start = len(self.transfer.audits)
        self.decode([1, 1], [1, 1])
        self.assertEqual(int(self.runtime.cache.stats.hits.cpu()), 4)
        self.assertEqual(int(self.runtime.cache.stats.misses.cpu()), 0)
        loads = [
            audit for audit in self.transfer.audits[start:] if audit.pool == "execution"
        ]
        self.assertEqual([len(audit.entries) for audit in loads], [1, 0, 1, 0])
        self.assertTrue(
            all(
                kind == CopySource.HOT_SLOT
                for audit in loads
                for kind, _, _ in audit.entries
            )
        )

    def test_copy_completion_and_mapping_publication_are_separate(self):
        state = self.runtime.residency
        before = state.key_to_slot.cpu().clone()
        with torch.npu.stream(self.compute):
            state.update_enabled.fill_(True)
            target = torch.tensor(REPLACED_KEYS, dtype=torch.int64, device=self.device)
            plan = self.runtime.residency_ops.prepare_update(
                target, self.runtime.context
            )
            ticket = self.runtime._materialize(plan.copies, self.runtime.hot_bindings)
            # Diagnostic host wait: explicitly stop between copying and commit.
            ticket.completion.synchronize()
        torch.testing.assert_close(state.key_to_slot.cpu(), before)
        self.assertEqual(state.slot_to_key.cpu().tolist(), INITIAL_KEYS)
        self.assertEqual(state.slot_valid.cpu().tolist(), [False, True, False, True])
        torch.testing.assert_close(
            self.hot.cpu(), self.oracle[REPLACED_KEYS], rtol=0, atol=0
        )
        with torch.npu.stream(self.compute):
            self.runtime.residency_ops.commit_update(plan, ticket, self.runtime.context)
            self.runtime.context.fence().completion.synchronize()
        self.assert_hot(REPLACED_KEYS)
        self.validate_audits()

    def test_prefill_borrow_and_exact_refill_restore(self):
        self.heat_four_steps()
        self.decode([1], [1])  # Preserve a nonzero LFU cycle position.
        state = self.runtime.residency
        names = ("key_to_slot", "slot_to_key", "slot_valid", "slot_generation")
        saved = {name: getattr(state, name).cpu().clone() for name in names}
        usage = self.runtime.cache.policy_state.usage.cpu().clone()
        step_mod = int(self.runtime.cache.policy_state.step_mod.cpu())
        self.runtime.snapshot_decode_state().completion.synchronize()
        self.runtime.borrow_hot_arena()
        self.compute.synchronize()
        self.assertTrue(bool((self.hot.cpu() == -999).all()))
        start = len(self.transfer.audits)
        self.runtime.restore_decode_state().completion.synchronize()
        for name, expected in saved.items():
            torch.testing.assert_close(getattr(state, name).cpu(), expected)
        torch.testing.assert_close(self.runtime.cache.policy_state.usage.cpu(), usage)
        self.assertEqual(int(self.runtime.cache.policy_state.step_mod.cpu()), step_mod)
        self.assert_hot(REPLACED_KEYS)
        self.assertEqual(state.execution_key.cpu().tolist(), [-1, -1])
        self.assertFalse(bool(state.execution_valid.cpu().any()))
        refill = self.transfer.audits[start:]
        self.assertEqual(len(refill), 1)
        self.assertEqual(
            refill[0].entries,
            [
                (int(CopySource.HOST), key, slot)
                for slot, key in enumerate(REPLACED_KEYS)
            ],
        )
        self.validate_audits()
        self.assertEqual(self.addresses(), self.original_addresses)
        self.decode([1, 2], [1, 3])
        self.assertEqual(int(self.runtime.cache.stats.hits.cpu()), 4)


if __name__ == "__main__":
    program = unittest.main(exit=False)
    if program.result.skipped:
        print(
            "NPU flow was NOT validated: required hardware/environment is missing.",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(0 if program.result.wasSuccessful() else 1)
