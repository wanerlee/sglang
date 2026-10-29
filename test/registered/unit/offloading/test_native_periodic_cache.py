"""Periodic cache on the production NPU assembly, using test-only byte copies.

These contracts exercise real plans, publication, snapshots and policy state.
They do not certify native MemFabric, FP4 FFN or NPU graph capture.
"""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from test_npu_backend import Assembly

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.execution import ExecutionRequest
from sglang.srt.offloading.moe.npu.config import (
    NpuOffloadSettings,
    create_npu_offloader,
)
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy
from sglang.srt.offloading.moe.runtime import BoundMoERuntime
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNativePeriodicCache(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.profile = Path(self.directory.name) / "profile.json"
        # EP rank 1 owns global IDs 4..7. The initial rank-wide Top-K gives
        # two slots to each layer, while online usage starts independently.
        self.profile.write_text(
            json.dumps(
                {
                    "model_id": "toy",
                    "layers": {"3": {"4": 100, "6": 90}, "4": {"4": 80, "7": 70}},
                }
            )
        )
        self.assembly = Assembly(ratio=0.5, layer_count=2, local_experts=4)
        self.addCleanup(self.assembly.close)
        self.assembly.backend.settings.offline_model_id = "toy"
        self.assembly.policy = PeriodicLFUPolicy(
            PeriodicLFUConfig(str(self.profile)), self.assembly.catalog
        )

    def run_step(self, runtime, demand, *, enabled=True):
        context = runtime._new_context(Phase.DECODE, capturing=True)
        runtime._context = context
        runtime.residency.update_enabled.fill_(enabled)
        runtime.ops.begin_step(context)
        try:
            access = runtime.cache_updates.access
            for spec in self.assembly.catalog.offloaded_layers:
                counts = torch.tensor(demand.get(spec.layer_id, [0] * 4))
                for wave in runtime.layer_programs[spec.layer_id].decode_waves:
                    lease = access.stage(
                        ExecutionRequest(spec.layer_id, counts, wave, Phase.DECODE),
                        context,
                    )
                    access.wait_ready(lease, context.compute_stream, context=context)
                    access.release_after(
                        lease, context.record_compute_completion(), context
                    )
            tail = runtime.cache_updates.emit_step_end(context)
            result = context.finish(residency_ready=tail)
            # These are eager CPU emissions of the capture program, with no
            # actual graph retaining addresses after the completed step.
            result.resources_idle.completion.synchronize()
            context.dispose_completed(graph_destroyed=True)
        finally:
            runtime._context = None

    def test_offline_topk_loads_complete_bundles_without_an_online_step(self):
        runtime = self.assembly.finalize()
        self.assertEqual(runtime.residency.slot_to_key.tolist(), [0, 2, 4, 7])
        self.assertTrue(bool(runtime.residency.slot_valid.all()))
        for tensor in self.assembly.backend.format.banks[0].tensors.values():
            self.assertEqual(tensor[:, 0].tolist(), [30, 32, 40, 43])
        state = runtime.cache_updates.policy_state
        self.assertEqual(int(state.usage.count_nonzero()), 0)
        self.assertEqual(int(state.step_mod), 0)
        self.assertFalse(bool(state.update_due))
        self.assertEqual(int(runtime.cache_stats.accesses), 0)
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            runtime.initialize_hot_cache([0, 2, 4, 7])

    def test_four_native_steps_replace_across_layers_and_decay_once(self):
        self.assembly.backend.settings.batch_log_interval = 1
        runtime = self.assembly.finalize()
        demand = {3: [0, 5, 0, 0], 4: [0, 0, 7, 0]}
        for _ in range(3):
            self.run_step(runtime, demand)
            self.assertEqual(runtime.residency.slot_to_key.tolist(), [0, 2, 4, 7])
        self.run_step(runtime, demand)
        # Both victims have zero usage; the rank-wide tie break evicts keys 0
        # and 2, even though both originally belonged to the same layer.
        self.assertEqual(set(runtime.residency.slot_to_key.tolist()), {1, 4, 6, 7})
        self.assertEqual(int(runtime.cache_updates.policy_state.step_mod), 0)
        torch.testing.assert_close(
            runtime.cache_updates.policy_state.usage,
            torch.tensor([0, 14, 0, 0, 0, 0, 19.6, 0]),
        )
        self.assertEqual(int(runtime.cache_stats.hits), 0)
        self.run_step(runtime, demand)
        self.assertEqual(int(runtime.cache_stats.accesses), 12)
        self.assertEqual(int(runtime.cache_stats.hits), 12)
        self.assertEqual(int(runtime.cache_stats.misses), 0)

    def test_assignment_statistics_count_repeated_routes_and_clear_per_batch(self):
        # Disabled batch logging intentionally omits route-statistics kernels.
        self.assembly.backend.settings.batch_log_interval = 1
        runtime = self.assembly.finalize()
        self.run_step(runtime, {3: [2, 3, 0, 0], 4: [0, 0, 5, 7]})
        stats = runtime.cache_stats
        self.assertEqual(
            (int(stats.accesses), int(stats.hits), int(stats.misses)), (17, 9, 8)
        )
        pointers = [item.storage_key for item in stats.allocations()]
        self.run_step(runtime, {3: [0, 0, 1, 0]})
        self.assertEqual(
            (int(stats.accesses), int(stats.hits), int(stats.misses)), (1, 1, 0)
        )
        self.assertEqual([item.storage_key for item in stats.allocations()], pointers)

    def test_capture_gate_preserves_offline_weights_usage_and_clock(self):
        runtime = self.assembly.finalize()
        before = runtime.residency.slot_to_key.clone()
        for _ in range(5):
            self.run_step(runtime, {3: [0, 10, 0, 0]}, enabled=False)
        torch.testing.assert_close(runtime.residency.slot_to_key, before)
        self.assertEqual(int(runtime.cache_updates.policy_state.usage.sum()), 0)
        self.assertEqual(int(runtime.cache_updates.policy_state.step_mod), 0)
        self.assertEqual(int(runtime.cache_stats.accesses), 0)

    def test_serving_gate_skips_global_idle_but_counts_an_active_peer_batch(self):
        runtime = self.assembly.finalize()
        owner = MoEOffloadOwner(self.assembly.config, self.assembly.catalog, runtime)
        mode = SimpleNamespace(is_decode_or_idle=lambda: True)
        for counts, expected_step in (([0, 0], 0), ([0, 3], 1)):
            batch = SimpleNamespace(
                forward_mode=mode,
                input_ids=torch.empty(0, dtype=torch.int64),
                original_global_num_tokens_cpu=counts,
            )
            with runtime.forward_scope(owner, batch):
                # Stand in for the already captured step nodes, without rerunning
                # model hooks or marking a second eager step in forward_scope.
                runtime.claim_decode_graph()
                runtime.ops.begin_step(runtime.context)
                runtime.cache_updates.emit_step_end(runtime.context)
            self.assertEqual(
                int(runtime.cache_updates.policy_state.step_mod), expected_step
            )
            self.assertEqual(int(runtime.cache_stats.accesses), 0)
        owner.close()

    def test_prefill_borrow_restores_original_slots_and_periodic_cycle(self):
        runtime = self.assembly.finalize()
        self.run_step(runtime, {3: [0, 5, 0, 0]})
        saved_keys = runtime.residency.slot_to_key.clone()
        saved_usage = runtime.cache_updates.policy_state.usage.clone()
        owner = MoEOffloadOwner(self.assembly.config, self.assembly.catalog, runtime)
        owner.switch_phase(Phase.PREFILL)
        for tensor in self.assembly.backend.format.banks[0].tensors.values():
            tensor.fill_(199)
        owner.switch_phase(Phase.DECODE)
        torch.testing.assert_close(runtime.residency.slot_to_key, saved_keys)
        torch.testing.assert_close(
            runtime.cache_updates.policy_state.usage, saved_usage
        )
        self.assertEqual(int(runtime.cache_updates.policy_state.step_mod), 1)
        for tensor in self.assembly.backend.format.banks[0].tensors.values():
            self.assertEqual(tensor[:, 0].tolist(), [30, 32, 40, 43])
        owner.close()

    def test_device_fault_during_initialization_fails_and_releases_resources(self):
        original = self.assembly.bind

        def faulty(*args, **kwargs):
            binding = original(*args, **kwargs)
            binding.error_status.fill_(1)
            return binding

        self.assembly.bind = faulty
        with self.assertRaisesRegex(RuntimeError, "cache initialization"):
            self.assembly.finalize()
        self.assertTrue(self.assembly.backend._closed)
        self.assertIsNone(self.assembly.backend.runtime)

    def test_initial_profile_model_mismatch_fails_before_bank_allocation(self):
        self.assembly.backend.settings.offline_model_id = "different-model"
        with self.assertRaisesRegex(ValueError, "model_id"):
            self.assembly.finalize()
        self.assertEqual(self.assembly.backend.format.banks, [])

    @staticmethod
    def dp_batch(mode_name, *, width=8, rank=1, prefill=False, counts=None):
        if counts is None:
            counts = [3] + [0] * (width - 1)
        mode = SimpleNamespace(
            name=mode_name,
            is_decode_or_idle=lambda: mode_name in ("DECODE", "IDLE"),
            is_extend_without_speculative=lambda: mode_name == "EXTEND",
        )
        rows = 0 if mode_name == "IDLE" else counts[rank]
        return SimpleNamespace(
            forward_mode=mode,
            is_extend_in_batch=prefill,
            original_global_num_tokens_cpu=counts,
            global_num_tokens_cpu=list(counts),
            input_ids=torch.ones(rows, dtype=torch.int64),
        )

    def test_attention_dp_prefill_selects_host_phase_on_active_and_idle_ranks(self):
        runtime = self.assembly.finalize()
        owner = MoEOffloadOwner(self.assembly.config, self.assembly.catalog, runtime)
        try:
            for width in (8, 16):
                for rank in range(width):
                    with self.subTest(width=width, rank=rank):
                        local = "EXTEND" if rank == 0 else "IDLE"
                        batch = self.dp_batch(
                            local, width=width, rank=rank, prefill=True
                        )
                        mode = batch.forward_mode
                        with runtime.forward_scope(owner, batch):
                            self.assertIs(runtime.context.phase, Phase.PREFILL)
                            self.assertIs(owner.phase, Phase.PREFILL)
                            self.assertFalse(bool(runtime.residency.update_enabled))
                            # Exercise the actual eager entry guard, not only the
                            # phase helper. No captured-decode scope is installed.
                            self.assertIsNone(runtime._capture)
                            runtime._require_emission()
                            self.assertIs(batch.forward_mode, mode)
                        self.assertEqual(
                            int(runtime.cache_updates.policy_state.step_mod), 0
                        )
                        owner.switch_phase(Phase.DECODE)
        finally:
            owner.close()

    def test_attention_dp_global_prefill_also_overrides_a_local_decode(self):
        runtime = self.assembly.finalize()
        owner = MoEOffloadOwner(self.assembly.config, self.assembly.catalog, runtime)
        try:
            for width in (8, 16):
                counts = [3, 1] + [0] * (width - 2)
                batch = self.dp_batch(
                    "DECODE", width=width, prefill=True, counts=counts
                )
                with runtime.forward_scope(owner, batch):
                    self.assertIs(runtime.context.phase, Phase.PREFILL)
                    runtime._require_emission()
                    self.assertEqual(batch.forward_mode.name, "DECODE")
                self.assertEqual(int(runtime.cache_updates.policy_state.step_mod), 0)
                owner.switch_phase(Phase.DECODE)
        finally:
            owner.close()

    def test_global_idle_preserves_prefill_banks_and_period_despite_padding(self):
        runtime = self.assembly.finalize()
        owner = MoEOffloadOwner(self.assembly.config, self.assembly.catalog, runtime)
        try:
            owner.switch_phase(Phase.PREFILL)
            for tensor in self.assembly.backend.format.banks[0].tensors.values():
                tensor.fill_(199)
            transfers = len(self.assembly.host.plans)
            for width in (8, 16):
                batch = self.dp_batch("IDLE", width=width, counts=[0] * width)
                batch.global_num_tokens_cpu = [8] * width
                batch.input_ids = torch.ones(8, dtype=torch.int64)
                with runtime.forward_scope(owner, batch):
                    self.assertIs(runtime.context.phase, Phase.PREFILL)
                    self.assertFalse(bool(runtime.residency.update_enabled))
                self.assertIs(owner.phase, Phase.PREFILL)
                self.assertEqual(len(self.assembly.host.plans), transfers)
                self.assertEqual(int(runtime.cache_updates.policy_state.step_mod), 0)
                for tensor in self.assembly.backend.format.banks[0].tensors.values():
                    self.assertTrue(bool((tensor == 199).all()))
        finally:
            owner.close()

    def test_idle_prefill_restores_cache_before_active_peer_decode_replay(self):
        runtime = self.assembly.finalize()
        owner = MoEOffloadOwner(self.assembly.config, self.assembly.catalog, runtime)
        keys = runtime.residency.slot_to_key.clone()
        usage = runtime.cache_updates.policy_state.usage.clone()
        try:
            for width in (8, 16):
                before = int(runtime.cache_updates.policy_state.step_mod)
                with runtime.forward_scope(
                    owner, self.dp_batch("IDLE", width=width, prefill=True)
                ):
                    runtime._require_emission()
                    bank = self.assembly.backend.format.banks[0]
                    for tensor in bank.tensors.values():
                        tensor.fill_(199)
                self.assertEqual(
                    int(runtime.cache_updates.policy_state.step_mod), before
                )
                batch = self.dp_batch(
                    "IDLE", width=width, counts=[1] + [0] * (width - 1)
                )
                with runtime.forward_scope(owner, batch):
                    self.assertIs(runtime.context.phase, Phase.DECODE)
                    self.assertTrue(bool(runtime.residency.update_enabled))
                    torch.testing.assert_close(runtime.residency.slot_to_key, keys)
                    torch.testing.assert_close(
                        runtime.cache_updates.policy_state.usage, usage
                    )
                    bank = self.assembly.backend.format.banks[0]
                    for tensor in bank.tensors.values():
                        self.assertEqual(tensor[:, 0].tolist(), [30, 32, 40, 43])
                    # Stand in for replay's recorded nodes. The serving scope
                    # must not add a second policy step on exit.
                    runtime.claim_decode_graph()
                    runtime.ops.begin_step(runtime.context)
                    runtime.cache_updates.emit_step_end(runtime.context)
                self.assertEqual(
                    int(runtime.cache_updates.policy_state.step_mod), before + 1
                )
        finally:
            owner.close()

    def test_attention_dp_keeps_rejecting_real_eager_decode_and_idle(self):
        runtime = self.assembly.finalize()
        owner = MoEOffloadOwner(self.assembly.config, self.assembly.catalog, runtime)
        try:
            for local in ("DECODE", "IDLE"):
                batch = self.dp_batch(local, counts=[1, 1] + [0] * 6)
                with runtime.forward_scope(owner, batch):
                    with self.assertRaisesRegex(RuntimeError, "captured full graph"):
                        runtime._require_emission()
                    # After checking the eager guard, stand in for the graph
                    # replay required to complete a successful serving scope.
                    runtime.claim_decode_graph()
        finally:
            owner.close()

    def test_global_prefill_does_not_admit_unsupported_local_forward_modes(self):
        runtime = self.assembly.finalize()
        owner = MoEOffloadOwner(self.assembly.config, self.assembly.catalog, runtime)
        try:
            batch = self.dp_batch("TARGET_VERIFY", prefill=True)
            with mock.patch.object(owner, "begin_forward") as begin:
                with self.assertRaisesRegex(ValueError, "ordinary extend"):
                    with runtime.forward_scope(owner, batch):
                        self.fail("Unsupported mode entered the serving scope")
                begin.assert_not_called()
        finally:
            owner.close()


class TestNativePeriodicConfig(unittest.TestCase):
    def settings(self, **kwargs):
        values = dict(
            cache_ratio=0.25,
            execution_slots=16,
            host_pool_gib=128,
            workspace_limit_bytes=2**30,
            metadata_limit_bytes=2**20,
            max_dispatch_rows=32768,
            offline_stats_path="profile.json",
            offline_model_id="toy",
        )
        values.update(kwargs)
        return NpuOffloadSettings(**values)

    def test_profile_and_model_label_are_optional(self):
        self.settings(offline_stats_path=None)
        self.settings(offline_model_id=None)
        settings = self.settings(offline_stats_path=None, offline_model_id=None)
        assembly = Assembly(ratio=0.5, layer_count=2, local_experts=4)
        self.addCleanup(assembly.close)
        policy = PeriodicLFUPolicy(settings.policy_config(), assembly.catalog)
        assembly.policy = policy
        runtime = assembly.finalize()
        self.assertEqual(runtime.residency.slot_to_key.tolist(), [0, 1, 4, 5])
        self.assertTrue(bool(runtime.residency.slot_valid.all()))
        for tensor in assembly.backend.format.banks[0].tensors.values():
            self.assertEqual(tensor[:, 0].tolist(), [30, 31, 40, 41])
        self.assertEqual(int(runtime.cache_updates.policy_state.usage.sum()), 0)
        settings = self.settings(
            cache_ratio=0, offline_stats_path=None, offline_model_id=None
        )
        catalog = Assembly(ratio=0, layer_count=2, local_experts=4)
        self.addCleanup(catalog.close)
        policy = PeriodicLFUPolicy(settings.policy_config(), catalog.catalog)
        self.assertEqual(policy.initial_keys(0, None), ())

    def test_relative_profile_path_and_policy_parameters_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offload.json"
            path.write_text(
                json.dumps(asdict(self.settings(update_interval=8, max_swaps=12)))
            )
            settings = NpuOffloadSettings.from_file(path)
            self.assertEqual(
                settings.offline_stats_path,
                str((Path(directory) / "profile.json").resolve()),
            )
            self.assertEqual(settings.policy_config().update_interval, 8)
            self.assertEqual(settings.policy_config().max_swaps, 12)
        with self.assertRaises(ValueError):
            self.settings(update_interval=0)

    def test_native_factory_selects_periodic_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offload.json"
            path.write_text(json.dumps(asdict(self.settings())))
            parallel = SimpleNamespace(
                moe_ep_size=2,
                moe_ep_rank=1,
                tp_size=2,
                moe_tp_size=1,
                pp_size=1,
                moe_dp_size=1,
            )
            with mock.patch(
                "sglang.srt.offloading.moe.npu.backend.NpuMoEBackend"
            ) as backend:
                offloader = create_npu_offloader(
                    path, hf_config=object(), parallel=parallel, device="npu:0"
                )
            self.assertIs(offloader.backend, backend.return_value)
            self.assertIsInstance(offloader.periodic_lfu_config, PeriodicLFUConfig)
            self.assertEqual(offloader.periodic_lfu_config.update_interval, 4)

    def test_global_work_uses_unpadded_metadata_even_on_a_locally_idle_rank(self):
        empty = SimpleNamespace(
            input_ids=torch.ones(8), original_global_num_tokens_cpu=[0, 0]
        )
        active = SimpleNamespace(
            input_ids=torch.empty(0), original_global_num_tokens_cpu=[0, 3]
        )
        self.assertFalse(BoundMoERuntime._has_global_work(empty))
        self.assertTrue(BoundMoERuntime._has_global_work(active))

    def test_native_factory_accepts_attention_dp8_and_dp16_with_moe_dp1(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offload.json"
            path.write_text(json.dumps(asdict(self.settings())))
            for width in (8, 16):
                parallel = SimpleNamespace(
                    moe_ep_size=width,
                    moe_ep_rank=width - 1,
                    tp_size=width,
                    moe_tp_size=1,
                    pp_size=1,
                    moe_dp_size=1,
                    attn_dp_size=width,
                )
                with mock.patch("sglang.srt.offloading.moe.npu.backend.NpuMoEBackend"):
                    offloader = create_npu_offloader(
                        path, hf_config=object(), parallel=parallel, device="npu:0"
                    )
                self.assertEqual(offloader.config.ep_size, width)
                self.assertEqual(offloader.config.tp_size, width)


if __name__ == "__main__":
    unittest.main()
