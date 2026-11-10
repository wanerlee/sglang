"""CPU contracts for sparse GLM catalogs and ModelSlim host checkpoint bundles.

The CPU adapter below is a memory/accelerator boundary; format recipes,
checkpoint writes, storage sealing and byte-budget checks run production code.
Native NZ conversion and GMM equivalence have a separate NPU manual test.
"""

import ctypes
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.offloading.moe.config import ExpertLayerSpec
from sglang.srt.offloading.moe.npu.format import ExpertLayerStorage
from sglang.srt.offloading.moe.npu.model_adapter import (
    build_expert_catalog,
    get_quant_spec,
)
from sglang.srt.offloading.moe.npu.prefill import _projection_copy_plans
from sglang.srt.offloading.moe.npu.storage import HostAllocation
from sglang.srt.offloading.moe.npu.w4a8_format import NpuW4A8Format
from sglang.srt.offloading.moe.resources import ResourceKind
from sglang.srt.offloading.moe.storage import HostMemoryCapability
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestAscendModelCatalog(CustomTestCase):
    @staticmethod
    def _glm(**changes):
        config = dict(
            model_type="glm_moe_dsa",
            architectures=["GlmMoeDsaForCausalLM"],
            num_hidden_layers=12,
            first_k_dense_replace=3,
            moe_layer_freq=2,
            n_routed_experts=16,
        )
        config.update(changes)
        return SimpleNamespace(**config)

    def test_sparse_layers_preserve_model_ids_and_dense_layers_have_no_cache_keys(self):
        """Dense gaps must not allocate host experts or shift the dispatch layer ID."""
        catalog = build_expert_catalog(self._glm(), ep_size=4, ep_rank=2)
        self.assertEqual(
            tuple(layer.layer_id for layer in catalog.layers), (4, 6, 8, 10)
        )
        self.assertTrue(catalog.layer(4).resident)
        self.assertEqual(
            tuple(layer.layer_id for layer in catalog.offloaded_layers), (6, 8, 10)
        )
        self.assertEqual(catalog.num_experts, 12)
        self.assertEqual(catalog.layer(6).global_start, 8)
        self.assertEqual(catalog.layer(6).local_id(11), 3)
        self.assertEqual(catalog.key(6, 3), 3)
        self.assertEqual(catalog.key(10, 0), 8)
        for layer in (0, 3, 4, 5, 11):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                catalog.key(layer, 0)

    def test_no_dense_prefix_still_keeps_first_moe_resident(self):
        catalog = build_expert_catalog(
            self._glm(first_k_dense_replace=0, moe_layer_freq=3), 4, 0
        )
        self.assertEqual(
            tuple(layer.layer_id for layer in catalog.layers), (0, 3, 6, 9)
        )
        self.assertEqual(
            tuple(layer.layer_id for layer in catalog.offloaded_layers), (3, 6, 9)
        )

    def test_v4_hash_layers_keep_the_existing_residency_contract(self):
        for hash_layers, first_offloaded in ((0, 3), (5, 5)):
            with self.subTest(hash_layers=hash_layers):
                config = SimpleNamespace(
                    model_type="deepseek_v4",
                    num_hidden_layers=8,
                    n_routed_experts=16,
                    num_hash_layers=hash_layers,
                )
                catalog = build_expert_catalog(config, 4, 0)
                self.assertEqual(catalog.offloaded_layers[0].layer_id, first_offloaded)
                self.assertEqual(catalog.num_experts, (8 - first_offloaded) * 4)

    def test_stage_catalogs_cover_only_local_layers_and_stage_local_experts(self):
        config = self._glm(num_hidden_layers=78, moe_layer_freq=1, n_routed_experts=256)
        spans = ((0, 18), (18, 38), (38, 62), (62, 78))
        stage_layers = []
        for span, expected_layers in zip(spans, (15, 20, 24, 16)):
            owned_experts = set()
            for rank in range(4):
                with self.subTest(span=span, rank=rank):
                    catalog = build_expert_catalog(
                        config, 4, rank, offload_all_routed=True, layer_range=span
                    )
                    ids = tuple(layer.layer_id for layer in catalog.layers)
                    self.assertEqual(ids, tuple(range(max(span[0], 3), span[1])))
                    self.assertEqual(len(catalog.offloaded_layers), expected_layers)
                    self.assertEqual(catalog.num_experts, expected_layers * 64)
                    self.assertEqual(catalog.key(ids[0], 0), 0)
                    self.assertEqual(catalog.key(ids[-1], 63), catalog.num_experts - 1)
                    for layer in catalog.layers:
                        self.assertFalse(layer.resident)
                        self.assertEqual(layer.local_experts, 64)
                        self.assertEqual(layer.global_start, rank * 64)
                    first = catalog.layers[0]
                    owned_experts.update(
                        range(
                            first.global_start, first.global_start + first.local_experts
                        )
                    )
                    if rank == 0:
                        stage_layers.extend(ids)
            self.assertEqual(owned_experts, set(range(256)))
        self.assertEqual(stage_layers, list(range(3, 78)))

    def test_stage_filter_preserves_sparse_ids_and_global_resident_prefix(self):
        # The first routed layer is globally 4. A later stage must not make its
        # first local routed layer resident or renumber its global layer ID.
        for all_routed in (False, True):
            with self.subTest(all_routed=all_routed):
                catalog = build_expert_catalog(
                    self._glm(),
                    4,
                    2,
                    offload_all_routed=all_routed,
                    layer_range=(5, 10),
                )
                self.assertEqual(
                    tuple(layer.layer_id for layer in catalog.layers), (6, 8)
                )
                self.assertEqual(catalog.offloaded_layers, catalog.layers)
                self.assertEqual(catalog.key(6, 0), 0)
                self.assertEqual(catalog.key(8, 3), 7)
                for outside in (4, 10):
                    with self.assertRaisesRegex(
                        ValueError, "Unregistered expert layer"
                    ):
                        catalog.layer(outside)
        prefix = build_expert_catalog(self._glm(), 4, 0, layer_range=(0, 8))
        self.assertTrue(prefix.layer(4).resident)
        self.assertEqual(
            tuple(layer.layer_id for layer in prefix.offloaded_layers), (6,)
        )

    def test_invalid_stage_range_and_stages_without_offloaded_layers_are_rejected(self):
        invalid_ranges = (
            (),
            (0,),
            (0, 1, 2),
            (0, 13),
            (-1, 3),
            (3, 3),
            (4, 3),
            (0, True),
            (0, 3.0),
            "0,3",
        )
        for span in invalid_ranges:
            with self.subTest(span=span), self.assertRaisesRegex(
                ValueError, "layer_range"
            ):
                build_expert_catalog(self._glm(), 4, 0, layer_range=span)
        # Dense-only and resident-only stages have no execution-bank owner in V1.
        for span, all_routed in (((0, 4), True), ((4, 5), False), ((5, 6), True)):
            with self.subTest(span=span), self.assertRaisesRegex(
                ValueError, r"nonresident MoE layer in layer range"
            ):
                build_expert_catalog(
                    self._glm(), 4, 0, offload_all_routed=all_routed, layer_range=span
                )

    def test_invalid_sparse_catalog_is_rejected_before_bank_allocation(self):
        cases = (
            dict(model_type="glm4_moe"),
            dict(moe_layer_freq=0),
            dict(moe_layer_freq=-1),
            dict(first_k_dense_replace=-1),
            dict(first_k_dense_replace=12),
            dict(num_hidden_layers=5),  # Only resident layer 4 remains.
            dict(n_routed_experts=15),
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_expert_catalog(self._glm(**changes), 4, 0)


class TestAscendQuantSelection(CustomTestCase):
    @staticmethod
    def _scheme(kind, **changes):
        scheme = type(kind, (), {})()
        scheme.__dict__.update(
            group_size=0,
            tp_size=4,
            activation_use_clip=False,
            is_per_channel_weight=True,
        )
        scheme.__dict__.update(changes)
        return scheme

    def test_packed_int4_never_enables_int8_only_direct_hot_gmm(self):
        for kind, expected, direct in (
            ("ModelSlimW8A8Int8MoE", "w8a8", True),
            ("ModelSlimW4A8Int8MoE", "w4a8", False),
        ):
            with self.subTest(kind=kind):
                spec = get_quant_spec(self._scheme(kind), self._scheme(kind))
                self.assertEqual(spec.kind, expected)
                self.assertEqual(spec.supports_direct_hot, direct)
        grouped = self._scheme(
            "ModelSlimW4A8Int8MoE", group_size=128, is_per_channel_weight=False
        )
        spec = get_quant_spec(grouped, grouped)
        self.assertEqual((spec.group_size, spec.tp_size), (128, 4))
        self.assertFalse(spec.supports_direct_hot)

    def test_speculative_target_allocates_packed_bank_even_for_w8_ep4(self):
        from sglang.srt.offloading.moe.npu import ascend_ops

        class AllocationReached(Exception):
            pass

        # Stop at the first device allocation: post_init must choose its real
        # execution layout before any NPU tensor or cache state is allocated.
        for algorithm, capacity, name in (
            (None, 12, "ascend.union"),
            ("EAGLE", 4, "ascend.execution"),
        ):
            with self.subTest(algorithm=algorithm):
                from sglang.srt.offloading.moe.npu.execution_mode import (
                    CachedExecution,
                )

                owner = object.__new__(ascend_ops.AscendOpsOffloader)
                owner.execution_mode = CachedExecution()
                owner.catalog = build_expert_catalog(
                    TestAscendModelCatalog._glm(), 4, 0
                )
                owner.storages = {
                    layer.layer_id: SimpleNamespace(require_sealed=lambda: None)
                    for layer in owner.catalog.offloaded_layers
                }
                w8 = self._scheme("ModelSlimW8A8Int8MoE")
                owner.quant_spec = get_quant_spec(w8, w8)
                owner.execution_slots = owner.copy_capacity = 4
                owner.hot_slots = 8
                allocations = []

                def allocate(*args):
                    allocations.append(args)
                    raise AllocationReached

                owner.format = SimpleNamespace(allocate_bank=allocate)
                with (
                    patch.object(
                        ascend_ops,
                        "get_parallel",
                        return_value=SimpleNamespace(moe_ep_size=4),
                    ),
                    patch.object(
                        ascend_ops,
                        "get_spec",
                        return_value=SimpleNamespace(speculative_algorithm=algorithm),
                    ),
                    self.assertRaises(AllocationReached),
                ):
                    owner.post_init()
                self.assertEqual(
                    allocations, [(capacity, ResourceKind.EXECUTION_WEIGHTS, name)]
                )
                self.assertEqual(owner.direct_hot, algorithm is None)

    def test_mixed_projection_and_unsupported_quantization_cannot_share_a_bank(self):
        w8 = self._scheme("ModelSlimW8A8Int8MoE")
        w4 = self._scheme("ModelSlimW4A8Int8MoE")
        invalid = (
            (w8, w4),
            (w4, self._scheme("ModelSlimW4A8Int8MoE", tp_size=8)),
            (
                w4,
                self._scheme(
                    "ModelSlimW4A8Int8MoE", group_size=64, is_per_channel_weight=False
                ),
            ),
            (w4, self._scheme("ModelSlimW4A8MXFP4MoE")),
            (w4, self._scheme("ModelSlimW4A8Int8MoE", activation_use_clip=True)),
            (w4, self._scheme("ModelSlimW4A8Int8MoE", group_size=64)),
            (w4, self._scheme("ModelSlimW4A8Int8MoE", tp_size=3)),
        )
        for first, second in invalid:
            with self.subTest(second=second.__dict__), self.assertRaises(ValueError):
                get_quant_spec(first, second)


class TestModelSlimW4Metadata(CustomTestCase):
    @staticmethod
    def _select(group_size, kind="W4A8_DYNAMIC"):
        from sglang.srt.hardware_backend.npu.quantization import moe_methods
        from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

        config = object.__new__(ModelSlimConfig)
        prefix = "model.layers.3.mlp.experts"
        config.quant_description = {
            f"{prefix}.0.{projection}.weight": kind
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        if group_size is not None:
            config.quant_description["group_size"] = group_size
        # Constructors bind the unavailable torch.ops.npu quantizer. Metadata
        # selection and real scheme.create_weights remain under test.
        with patch.object(
            moe_methods,
            "HiddenStatesDynamicQuant",
            return_value=SimpleNamespace(),
            create=True,
        ):
            return config.get_moe_scheme(torch.nn.Module(), prefix)

    def test_exported_group_size_reaches_both_checkpoint_parameter_layouts(self):
        """Ignoring export metadata loses second scales before offload can load them."""
        from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w4a8_int8_moe import (
            ModelSlimW4A8Int8MoE,
        )

        for size in (None, 0, 64, "64"):
            with self.subTest(group_size=size):
                first, second = self._select(size)
                self.assertIsInstance(first, ModelSlimW4A8Int8MoE)
                self.assertIsInstance(second, ModelSlimW4A8Int8MoE)
                layer = torch.nn.Module()
                first.create_weights(layer, 2, 128, 256)
                second.create_weights(layer, 2, 128, 256)
                if size in (64, "64"):
                    self.assertEqual(
                        tuple(layer.w13_weight_scale_second.shape), (2, 512, 2)
                    )
                    self.assertEqual(
                        tuple(layer.w2_weight_scale_second.shape), (2, 128, 4)
                    )
                    self.assertEqual(get_quant_spec(first, second).group_size, 64)
                else:
                    self.assertFalse(hasattr(layer, "w13_weight_scale_second"))
                    self.assertTrue(first.is_per_channel_weight)
                    self.assertTrue(second.is_per_channel_weight)

    def test_invalid_group_metadata_cannot_silently_select_per_channel(self):
        for size in (True, 64.0, -1, "oops"):
            with self.subTest(group_size=size), self.assertRaises(ValueError):
                self._select(size)
        # Global metadata on a mixed-format checkpoint must not change W8A8.
        first, second = self._select("not-used", kind="W8A8_DYNAMIC")
        self.assertEqual(get_quant_spec(first, second).kind, "w8a8")

    def test_resident_draft_uses_own_scheme_after_target_offloader_is_replaced(self):
        """The shared process registry must not attach target host storage to draft."""
        from sglang.srt.layers.quantization.modelslim.modelslim import (
            ModelSlimFusedMoEMethod,
        )
        from sglang.srt.utils import offloader

        target_owner = object()
        target = torch.nn.Module()
        target._ascend_ops_offloader = target_owner
        for kind in ("W8A8_DYNAMIC", "W4A8_DYNAMIC"):
            with self.subTest(kind=kind):
                first, second = self._select(64, kind=kind)
                draft = torch.nn.Module()
                draft.w13_scheme, draft.w2_scheme = first, second
                method = object.__new__(ModelSlimFusedMoEMethod)
                with patch.object(offloader, "_instance", target_owner):
                    # ModelRunner installs a fresh NoopOffloader before draft
                    # construction while the target retains its private owner.
                    offloader.set_offloader(offloader.NoopOffloader())
                    method.create_weights(draft, 2, 128, 256, torch.bfloat16)
                expected = (2, 256, 128) if kind == "W4A8_DYNAMIC" else (2, 512, 128)
                self.assertEqual(tuple(draft.w13_weight.shape), expected)
                self.assertEqual(draft.w13_weight.dtype, torch.int8)
                self.assertFalse(hasattr(draft, "_moe_offload_storage"))
                self.assertFalse(hasattr(draft, "_ascend_ops_offloader"))
                self.assertIs(target._ascend_ops_offloader, target_owner)

    def test_non_square_packed_weights_use_output_axis_for_group_scale_bits(self):
        """Raw [E,N/2,K] must produce [E,K/group,N], not infer N from K."""
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            NPUW4A8Int8MoEMethod,
        )

        kernel = object.__new__(NPUW4A8Int8MoEMethod)
        for n, k in ((512, 128), (128, 256)):
            with self.subTest(n=n, k=k):
                weight = torch.empty((2, n // 2, k), dtype=torch.int8)
                channel = torch.linspace(0.001, 0.05, 2 * n).reshape(2, n, 1)
                second = torch.linspace(0.2, 1.0, 2 * n * (k // 64)).reshape(
                    2, n, k // 64
                )
                expected = (
                    (channel * second)
                    .to(torch.float16)
                    .float()
                    .transpose(1, 2)
                    .contiguous()
                )
                bits = expected.numpy().view(np.uint32).astype(np.int64)
                with patch.object(
                    torch.Tensor, "npu", lambda value, *a, **kw: value, create=True
                ):
                    actual = kernel._process_scale(weight, channel, second, False)
                self.assertEqual(tuple(actual.shape), (2, k // 64, n))
                torch.testing.assert_close(
                    actual, torch.from_numpy(bits), rtol=0, atol=0
                )


class _CpuFormatAdapter:
    """Simulated accelerator allocation/cast boundary, never an NPU oracle.

    The distinct output payload lets the test detect omitted/misaligned copies.
    Tensor shapes model the resident ABI; native numerical parity is manual.
    """

    def __init__(self, device, native_acl, *, is_per_channel_weight):
        self.is_per_channel_weight = is_per_channel_weight
        self.converted = []

    def to_device(self, tensor):
        return tensor.clone()

    def synchronize(self):
        pass

    def begin_workspace(self):
        return 0

    def end_workspace(self, baseline):
        return 0

    def prepare(
        self, weight, scale, offset, bias, scale_second=None, offset_second=None
    ):
        weight = weight.transpose(1, 2).contiguous().view(torch.int32)
        channel = scale.transpose(1, 2).contiguous()
        if self.is_per_channel_weight:
            converted_scale = channel.view(torch.int32).to(torch.int64)
        else:
            converted_scale = scale_second.transpose(1, 2).contiguous()
            converted_scale = converted_scale.view(torch.int32).to(torch.int64)
        # The native scale path creates a new compact NumPy-backed carrier;
        # cloning a one-expert padded CPU view can retain its unused stride 0.
        compact_scale = torch.empty(converted_scale.shape, dtype=torch.int64)
        compact_scale.copy_(converted_scale)
        converted_scale = compact_scale
        converted_bias = bias.sum(dim=-1)
        if self.is_per_channel_weight:
            result = (weight, converted_scale, offset, converted_bias)
        else:
            result = (
                weight,
                scale,
                offset,
                converted_bias,
                converted_scale,
                offset_second,
            )
        self.converted.append(tuple(value.clone() for value in result))
        return result

    def physical_nbytes(self, tensor):
        return tensor.untyped_storage().nbytes()

    def copy_to_host(self, destination, source, nbytes):
        ctypes.memmove(destination, source.data_ptr(), nbytes)

    def empty_component(self, name, layout, capacity):
        return torch.empty((capacity,) + layout.runtime_shape[1:], dtype=layout.dtype)


class _CpuHostPool:
    """Certified host-allocation boundary; the storage implementation is real."""

    def __init__(self, nbytes):
        self.tensor = torch.empty(nbytes, dtype=torch.uint8)
        self.offset = 0
        self.capability = HostMemoryCapability(
            self.tensor.data_ptr(), nbytes, True, self.tensor.data_ptr(), self
        )

    def allocate(self, nbytes, alignment=64):
        start = (self.offset + alignment - 1) // alignment * alignment
        self.offset = start + nbytes
        return HostAllocation(self.tensor.narrow(0, start, nbytes), self.capability)


class TestAscendW4Checkpoint(CustomTestCase):
    @staticmethod
    def _format(**kwargs):
        with patch(
            "sglang.srt.offloading.moe.npu.w4a8_format._TorchNpuW4A8FormatAdapter",
            _CpuFormatAdapter,
        ):
            return NpuW4A8Format(128, 256, "cpu", tp_size=4, **kwargs)

    def test_packed_gate_up_shards_do_not_use_unpacked_metadata_offsets(self):
        """Packed W1/W3 occupy half as many rows as their per-channel metadata."""
        format = self._format()
        for name, width in (("w13_weight", 128), ("w13_weight_scale", 1)):
            component = format.layouts[name]
            target = torch.zeros(
                component.raw_shape[1:], dtype=component.checkpoint_dtype
            )
            rows = 128 if name.endswith("_weight") else 256
            first = torch.full((rows, width), 3, dtype=component.checkpoint_dtype)
            second = torch.full((rows, width), 7, dtype=component.checkpoint_dtype)
            format.write_checkpoint(name, target, second, "w3", {})
            format.write_checkpoint(name, target, first, "w1", {})
            torch.testing.assert_close(
                target, torch.cat((first, second)), rtol=0, atol=0
            )
        with self.assertRaisesRegex(ValueError, "packed checkpoint"):
            format.write_checkpoint(
                "w13_weight",
                torch.zeros((256, 128), dtype=torch.int8),
                torch.zeros((256, 128), dtype=torch.int8),
                "w1",
                {},
            )

    def test_group_scale_carrier_and_bias_shrink_remain_fully_budgeted(self):
        """INT64 runtime scales grow while TP bias shrinks; charge both extents."""
        format = self._format(group_size=64)
        weight = format.layouts["w13_weight"]
        self.assertEqual(weight.raw_shape, (1, 256, 128))
        self.assertEqual(weight.runtime_shape, (1, 128, 64))
        self.assertEqual(weight.dtype, torch.int32)
        self.assertEqual(weight.nbytes, 32768)
        scale = format.layouts["w2_weight_scale_second"]
        self.assertEqual(scale.raw_shape, (1, 128, 4))
        self.assertEqual(scale.runtime_shape, (1, 4, 128))
        self.assertEqual(scale.dtype, torch.int64)
        self.assertEqual(scale.host_nbytes, 4096)
        bias = format.layouts["w2_scale_bias"]
        self.assertEqual(bias.raw_shape, (1, 128, 4))
        self.assertEqual(bias.runtime_shape, (1, 128))
        self.assertEqual((bias.nbytes, bias.host_nbytes), (512, 2048))
        self.assertEqual(
            format.host_nbytes(3),
            3 * sum(c.host_nbytes for c in format.layouts.values()),
        )
        runtime = [c.runtime_name for c in format.layouts.values() if c.runtime_name]
        self.assertEqual(runtime.count("w2_weight_scale"), 1)
        self.assertIsNone(format.layouts["w2_weight_scale"].runtime_name)
        self.assertEqual(scale.runtime_name, "w2_weight_scale")

    def test_malformed_metadata_is_rejected_without_overwriting_checkpoint(self):
        format = self._format(group_size=64)
        cases = (
            ("w2_weight_scale", torch.full((128, 1), -1.0)),
            ("w2_weight_scale_second", torch.full((128, 4), float("nan"))),
            ("w2_scale_bias", torch.zeros((128, 16))),
            ("w2_weight_offset_second", torch.zeros((128, 3))),
            ("w2_weight", torch.zeros((64, 256), dtype=torch.uint8)),
        )
        for name, value in cases:
            component = format.layouts[name]
            target = torch.full(
                component.raw_shape[1:], 9, dtype=component.checkpoint_dtype
            )
            before = target.clone()
            with self.subTest(name=name), self.assertRaises(ValueError):
                format.write_checkpoint(name, target, value, "w2", {})
            torch.testing.assert_close(target, before, rtol=0, atol=0)
        # Biases and ignored resident offsets can be negative; only scales
        # have a nonnegative domain restriction.
        for name in ("w2_scale_bias", "w2_weight_offset"):
            layout = format.layouts[name]
            target = torch.zeros(layout.raw_shape[1:])
            format.write_checkpoint(name, target, torch.full_like(target, -2), "w2", {})
            self.assertTrue(target.eq(-2).all())

    def test_invalid_shape_group_clip_and_workspace_are_rejected(self):
        for kwargs in (
            dict(group_size=-1),
            dict(group_size=96),
            dict(activation_use_clip=True),
            dict(workspace_limit_bytes=1),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._format(**kwargs)

    def test_sealed_host_master_roundtrips_every_projection_component(self):
        """Missing bias/group scales or raw/runtime stride confusion corrupt reloads."""
        for group_size in (0, 64):
            with self.subTest(group_size=group_size):
                format = self._format(group_size=group_size)
                experts = 2
                pool = _CpuHostPool(format.host_nbytes(experts))
                layer = torch.nn.Module()
                storage = ExpertLayerStorage(
                    layer, ExpertLayerSpec(6, experts, 1, 0), format, pool, {}
                )
                try:
                    for index, (name, layout) in enumerate(format.layouts.items()):
                        parameter = getattr(layer, name)
                        for expert in range(experts):
                            value = torch.full(
                                layout.raw_shape[1:],
                                index + expert + 1,
                                dtype=layout.checkpoint_dtype,
                            )
                            parts = (
                                value.chunk(2, 0)
                                if len(layout.shards) == 2
                                else (value,)
                            )
                            for shard, part in zip(layout.shards, parts, strict=True):
                                parameter.weight_loader(
                                    parameter, part, shard_id=shard, expert_id=expert
                                )
                    self.assertEqual(pool.offset, format.host_nbytes(experts))
                    format.adapter.converted.clear()
                    storage.finalize()
                    storage.require_sealed()
                    self.assertEqual(storage.raw_components, {})
                    bank = format.allocate_bank(
                        experts, ResourceKind.EXECUTION_WEIGHTS, "test"
                    )
                    for plan in _projection_copy_plans(storage, bank, format.layouts):
                        for region in plan.regions:
                            ctypes.memmove(
                                region.destination.address,
                                region.source.host_address,
                                region.nbytes,
                            )
                    for expert in range(experts):
                        for projection, names in enumerate(
                            format.schema.conversion_groups
                        ):
                            expected = format.adapter.converted[2 * expert + projection]
                            for name, value in zip(names, expected, strict=True):
                                torch.testing.assert_close(
                                    bank.tensors[name][expert : expert + 1],
                                    value,
                                    rtol=0,
                                    atol=0,
                                )
                    module = SimpleNamespace(AscendQuantInfo=SimpleNamespace)
                    with patch.dict(
                        sys.modules, {"sglang.srt.layers.moe.moe_runner.ascend": module}
                    ):
                        quant = bank.quant_info()
                    for prefix in ("w13", "w2"):
                        name = f"{prefix}_weight_scale" + (
                            "_second" if group_size else ""
                        )
                        self.assertIs(
                            getattr(quant, f"{prefix}_weight_scale"), bank.tensors[name]
                        )
                        self.assertIs(
                            getattr(quant, f"{prefix}_scale_bias"),
                            bank.tensors[f"{prefix}_scale_bias"],
                        )
                        self.assertFalse(hasattr(quant, f"{prefix}_weight_offset"))
                    with self.assertRaisesRegex(RuntimeError, "not writable"):
                        layer.w2_weight.weight_loader(None, None)
                finally:
                    storage.close()


class TestAscendFormatCopyPlan(CustomTestCase):
    def test_decode_descriptors_copy_bias_and_runtime_group_scales(self):
        """Component indices cannot assume W8's weight/scale/offset triplets."""
        from sglang.srt.offloading.moe.npu.ascend_ops import AscendOpsOffloader

        for group_size, expected in (
            (0, (0, 1, 3, 4, 5, 7)),
            (64, (0, 3, 4, 6, 9, 10)),
        ):
            with self.subTest(group_size=group_size):
                format = TestAscendW4Checkpoint._format(group_size=group_size)
                names = tuple(format.layouts)
                owner = object.__new__(AscendOpsOffloader)
                owner.format = format
                owner.device = torch.device("cpu")
                owner.copy_capacity = 4
                owner.direct_hot = False
                owner.catalog = SimpleNamespace(
                    offloaded_layers=[SimpleNamespace(layer_id=6)]
                )
                owner.execution = format.allocate_bank(
                    4, ResourceKind.EXECUTION_WEIGHTS, "test"
                )
                owner.destinations = (
                    torch.arange(len(names) * 4).reshape(len(names), 4) + 100
                )
                owner.lengths = (
                    torch.arange(len(names) * 4, dtype=torch.int32).reshape(
                        len(names), 4
                    )
                    + 1000
                )
                npu = SimpleNamespace(Stream=lambda **kw: object(), Event=object)
                with patch.object(torch, "npu", npu, create=True):
                    owner._init_copy_plan(names)
                self.assertEqual(owner.fused_components, expected)
                decoded = tuple(
                    (owner.fused_component_map >> (p * owner.fused_component_bits))
                    & ((1 << owner.fused_component_bits) - 1)
                    for p in range(len(expected))
                )
                self.assertEqual(decoded, expected)
                owner.fused_sources.copy_(
                    torch.arange(len(expected) * 4).reshape(len(expected), 4) + 2000
                )
                descriptors = [
                    entry for phase in owner.fused_descriptors for entry in phase
                ]
                self.assertEqual(tuple(map(len, owner.fused_descriptors)), (3, 3))
                for position, (component, entry) in enumerate(
                    zip(expected, descriptors, strict=True)
                ):
                    torch.testing.assert_close(entry[0], owner.fused_sources[position])
                    torch.testing.assert_close(entry[1], owner.destinations[component])
                    torch.testing.assert_close(entry[2], owner.lengths[component])
                # W2 progress must never mutate GMM1's packed-count output.
                self.assertEqual(
                    len({entry[-1].data_ptr() for entry in descriptors}), 6
                )
                self.assertIs(owner.packed_counts, descriptors[0][-1])


if __name__ == "__main__":
    unittest.main()
