"""Mixed Prefill resident bytes must survive Decode and cold-bank reuse."""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.offloading.moe.npu import ascend_ops
from sglang.srt.offloading.moe.npu.execution_mode import CachedExecution
from sglang.srt.offloading.moe.npu.format import ExpertBank
from sglang.srt.offloading.moe.npu.model_adapter import ExpertQuantSpec
from sglang.srt.offloading.moe.npu.prefill import (
    AscendPrefillWeights,
    mixed_resident_bank_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from test_ascend_prefill_only import (
    _Completion,
    _Context,
    _Stream,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMixedPrefillResidency(CustomTestCase):
    def test_dsv4_mixed_configuration_keeps_decode_budget(self):
        from test_ascend_prefill_only import TestOffloadModeWiring

        # Reuse the fixture's profile ID; its empty frequencies rank by key.
        values = {
            "offline_stats_path": "profile.json",
            "offline_model_id": "GLM-5.2",
            "cache_ratio": 0.25,
            "prefill_resident_ratio": 0.25,
        }
        owner = TestOffloadModeWiring._construct(
            values,
            "null",
            model_changes={"model_type": "deepseek_v4", "num_hidden_layers": 43},
            parallel_changes={"tp_size": 8, "moe_ep_size": 8, "attn_dp_size": 2},
        )
        self.assertIsInstance(owner.execution_mode, CachedExecution)
        self.assertEqual(owner.execution_slots, 32)
        self.assertEqual(owner.hot_slots, 320)
        self.assertEqual(len(owner.catalog.offloaded_layers), 40)
        self.assertEqual(owner.initial_keys, tuple(range(320)))
        with self.assertRaisesRegex(ValueError, "execution_slots >= local_experts"):
            TestOffloadModeWiring._construct(
                {**values, "execution_slots": 1},
                "null",
                model_changes={"model_type": "deepseek_v4", "num_hidden_layers": 43},
            )

    def _allocated_owner(self, *, direct, ratio):
        owner = object.__new__(ascend_ops.AscendOpsOffloader)
        owner.device = torch.device("cpu")
        owner.cfg = {"prefill_resident_ratio": ratio}
        owner.quant_spec = ExpertQuantSpec("w8a8")
        owner.local_experts = owner.copy_capacity = owner.execution_slots = 4
        owner.hot_slots = 4
        owner.catalog = SimpleNamespace(
            offloaded_layers=tuple(SimpleNamespace(layer_id=i) for i in (3, 4, 5))
        )
        names = ("w13_weight", "w2_weight")
        owner.format = SimpleNamespace(
            layouts={name: SimpleNamespace(nbytes=1) for name in names},
            adapter=SimpleNamespace(native_acl=object()),
            quant_info=lambda tensors: tensors,
        )
        allocations = []

        def allocate(capacity, kind, name):
            bank = ExpertBank(
                owner.format,
                {n: torch.zeros((capacity, 1), dtype=torch.uint8) for n in names},
                capacity,
                kind,
                name,
            )
            allocations.append(bank)
            return bank

        owner.format.allocate_bank = allocate
        owner.storages = {
            layer: SimpleNamespace(
                component_capabilities={
                    name: tuple(
                        SimpleNamespace(value=layer * 16 + component * 4 + e, nbytes=1)
                        for e in range(4)
                    )
                    for component, name in enumerate(names)
                }
            )
            for layer in (3, 4, 5)
        }
        # Run the real allocation branch, stopping before unrelated MemFabric
        # address/Decode graph setup. No metadata or kernel is stubbed as passing.
        with (
            patch.object(
                ascend_ops,
                "get_parallel",
                return_value=SimpleNamespace(moe_ep_size=4 if direct else 8),
            ),
            patch.object(
                ascend_ops,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch.object(
                ascend_ops,
                "MemFabricAddressCatalog",
                side_effect=RuntimeError("stop after allocation"),
            ),
            self.assertRaisesRegex(RuntimeError, "stop after allocation"),
        ):
            owner._init_cached_runtime()
        return owner, allocations[0]

    def test_decode_layout_and_direct_hot_choice_are_preserved(self):
        for direct in (False, True):
            for ratio in (0, 0.25, 0.5):
                with self.subTest(direct=direct, ratio=ratio):
                    owner, pool = self._allocated_owner(direct=direct, ratio=ratio)
                    resident_slots = 3 * int(4 * ratio)
                    self.assertEqual(owner.direct_hot, direct)
                    self.assertEqual(owner.execution.capacity, 4)
                    self.assertEqual(owner.hot.capacity, 4)
                    for name in pool.tensors:
                        root = pool.tensors[name].data_ptr()
                        self.assertEqual(
                            owner.execution.tensors[name].data_ptr(),
                            root + resident_slots,
                        )
                        owner.execution.tensors[name].fill_(201)
                        owner.hot.tensors[name].fill_(202)
                        self.assertEqual(
                            pool.tensors[name][:resident_slots].sum().item(), 0
                        )
                        if direct:
                            self.assertEqual(owner.union.capacity, 8)
                            self.assertEqual(
                                owner.hot.tensors[name].data_ptr(),
                                root + resident_slots + 4,
                            )

    def test_prefill_decode_prefill_preserves_residents_hot_and_expert_order(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                owner, pool = self._allocated_owner(direct=direct, ratio=0.25)
                _, offsets = mixed_resident_bank_layout(4, 3, 1, 4, 4 if direct else 0)
                copy_stream = _Stream()
                copy_stream.synchronize = lambda: _Completion(
                    copy_stream.tail
                ).synchronize()
                owner.weight_copy_stream = copy_stream
                owner.context = _Context()
                host_ops = Mock()

                def submit(plan, bank, stream):
                    owner.context.retain(plan)
                    for region in plan.regions:
                        stream.enqueue(
                            lambda r=region: ctypes.memset(
                                r.destination.address, r.source.value, r.nbytes
                            )
                        )
                    return SimpleNamespace(completion=_Completion(stream.tail))

                host_ops.materialize_into.side_effect = submit
                with patch(
                    "sglang.srt.offloading.moe.npu.native_acl.AclHostTransferOps",
                    return_value=host_ops,
                ):
                    weights = AscendPrefillWeights(
                        owner,
                        resident_experts=1,
                        resident_pool=pool,
                        cold_offsets=offsets,
                    )
                self.assertFalse(weights.borrows_hot)
                self.assertEqual(host_ops.materialize_into.call_count, 6)
                for value in owner.hot.tensors.values():
                    value.fill_(202)

                for _ in range(2):
                    owner.context = _Context()
                    weights.pipeline.begin_forward(owner.context)
                    for index, layer in enumerate((3, 4, 5)):
                        weights.pipeline.start(layer, owner.context)
                        bank = weights.pipeline.acquire(layer, owner.context)
                        weights.pipeline.wait_for_w2(layer, owner.context)
                        _Completion(owner.context.compute_stream.tail).synchronize()
                        for name, tensor in pool.tensors.items():
                            expected = [
                                c.value
                                for c in owner.storages[layer].component_capabilities[
                                    name
                                ]
                            ]
                            actual = [tensor[index, 0].item()] + weights.banks[
                                bank
                            ].tensors[name][:, 0].tolist()
                            self.assertEqual(actual, expected)
                            self.assertTrue(torch.all(owner.hot.tensors[name] == 202))
                        weights.finish_layer(layer)
                    weights.pipeline.finish_forward(owner.context)
                    weights.pipeline.reset()
                    # Decode may overwrite every execution slot. It must not
                    # touch residents needed by the next Prefill chunk/request.
                    for name, tensor in owner.execution.tensors.items():
                        tensor.fill_(201)
                        self.assertEqual(
                            pool.tensors[name][:3, 0].tolist(),
                            [
                                owner.storages[layer]
                                .component_capabilities[name][0]
                                .value
                                for layer in (3, 4, 5)
                            ],
                        )
                self.assertEqual(host_ops.materialize_into.call_count, 18)
                # Six initial resident bytes, then 18 cold bytes per forward.
                self.assertEqual(
                    sum(
                        sum(r.nbytes for r in call.args[0].regions)
                        for call in host_ops.materialize_into.call_args_list
                    ),
                    6 + 2 * 18,
                )

    def test_type2_capacity_is_checked_without_disabling_direct_hot(self):
        self.assertEqual(mixed_resident_bank_layout(32, 40, 8, 32), (368, (320, 344)))
        with self.assertRaisesRegex(ValueError, "Reduce prefill_resident_ratio"):
            mixed_resident_bank_layout(64, 40, 32, 64, 720)

    def test_quantization_guard_and_zero_ratio_keep_existing_decode_support(self):
        from sglang.srt.offloading.moe.npu import execution_mode

        owner = SimpleNamespace(cfg={"prefill_resident_ratio": 0.25})
        for mtp in (None, "NEXTN"):
            with patch.object(
                execution_mode,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=mtp),
            ):
                if mtp is None:
                    CachedExecution().validate_quantization(
                        owner, ExpertQuantSpec("w8a8")
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "without MTP"):
                        CachedExecution().validate_quantization(
                            owner, ExpertQuantSpec("w8a8")
                        )
                with self.assertRaisesRegex(ValueError, "W8A8"):
                    CachedExecution().validate_quantization(
                        owner, ExpertQuantSpec("w4a8")
                    )
                CachedExecution().validate_quantization(
                    SimpleNamespace(cfg={}), ExpertQuantSpec("w4a8")
                )


if __name__ == "__main__":
    unittest.main()
