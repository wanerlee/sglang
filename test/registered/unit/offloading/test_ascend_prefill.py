"""CPU stream contracts and interpreted Triton metadata; no NPU required."""

import ctypes
import random
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import triton.language as tl
from triton.runtime.interpreter import InterpretedFunction

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.offloading.moe.npu.prefill import (
    AscendPrefillWeights,
    LayerPrefetchPipeline,
    PrefillLayerTransfer,
    whole_layer_bank_layout,
)
from sglang.srt.offloading.moe.npu.small_batch_kernels import prepare_prefill_counts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Command:
    def __init__(self, dependencies, action):
        self.dependencies, self.action = dependencies, action
        self.done = self.running = False

    def run(self):
        if self.done:
            return
        if self.running:
            raise AssertionError("Cyclic stream dependency")
        self.running = True
        for dependency in self.dependencies:
            dependency.run()
        self.action()
        self.done, self.running = True, False


class _Stream:
    def __init__(self):
        self.tail = None
        self.waits = []

    def enqueue(self, action, extra=()):
        dependencies = tuple(extra) + (() if self.tail is None else (self.tail,))
        self.tail = _Command(dependencies, action)

    def wait_stream(self, other):
        if other.tail is not None:
            self.enqueue(lambda: None, (other.tail,))

    def drain(self):
        if self.tail is not None:
            self.tail.run()


class _Completion:
    def __init__(self, command):
        self.command = command

    def wait_on(self, stream):
        stream.waits.append(self.command)
        if self.command is not None:
            stream.enqueue(lambda: None, (self.command,))

    def synchronize(self):
        if self.command is not None:
            self.command.run()


class _Context:
    def __init__(self):
        self.invocation_id = object()
        self.compute_stream = _Stream()
        self.resources = []

    def retain(self, value):
        self.resources.append(value)

    def record_compute_completion(self):
        return SimpleNamespace(completion=_Completion(self.compute_stream.tail))


class TestLayerPrefetchPipeline(CustomTestCase):
    def test_capacity_never_silently_shrinks_hot_or_overlaps_banks(self):
        self.assertEqual(
            whole_layer_bank_layout(4, 6, 4), (("execution", 0, 4), ("hot", 0, 4))
        )
        self.assertEqual(
            whole_layer_bank_layout(4, 2, 8), (("execution", 0, 4), ("execution", 4, 4))
        )
        for capacity in ((4, 2, 4), (4, 6, 3), (0, 6, 4)):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                whole_layer_bank_layout(*capacity)

    def test_delayed_copy_and_reader_preserve_complete_layer_results(self):
        """Draining copies first must not overwrite either projection still in use."""
        for count in (1, 2, 3, 6):
            with self.subTest(layers=count):
                context, copy = _Context(), _Stream()
                banks, output, submitted = [[None, None], [None, None]], [], []
                layers = tuple(range(3, 3 + count))

                def load(layer, bank, ctx):
                    submitted.append(layer)
                    copy.enqueue(lambda: banks[bank].__setitem__(0, (layer, 0.5)))
                    first = SimpleNamespace(completion=_Completion(copy.tail))
                    copy.enqueue(lambda: banks[bank].__setitem__(1, (layer + 1, 2.0)))
                    return PrefillLayerTransfer(
                        first, SimpleNamespace(completion=_Completion(copy.tail))
                    )

                pipeline = LayerPrefetchPipeline(layers, load=load, copy_stream=copy)
                pipeline.begin_forward(context)
                for index, layer in enumerate(layers):
                    pipeline.start(layer, context)
                    if index == 0:
                        self.assertEqual(submitted, list(layers[:2]))
                    bank = pipeline.acquire(layer, context)
                    intermediate = []

                    def first_read(layer=layer, bank=bank, intermediate=intermediate):
                        w13, scale = banks[bank][0]
                        self.assertEqual(w13, layer)
                        intermediate.append(w13 * scale)

                    context.compute_stream.enqueue(first_read)
                    pipeline.wait_for_w2(layer, context)

                    def second_read(layer=layer, bank=bank, intermediate=intermediate):
                        w2, scale = banks[bank][1]
                        self.assertEqual(w2, layer + 1)
                        output.append(intermediate[0] * w2 * scale)

                    context.compute_stream.enqueue(second_read)
                    reader = context.record_compute_completion()
                    pipeline.retire(layer, reader, context)
                pipeline.finish_forward(context)
                # Forces bank-reuse waits to protect pending compute readers.
                copy.drain()
                context.compute_stream.drain()
                self.assertEqual(output, [layer * (layer + 1) for layer in layers])
                self.assertEqual(submitted, list(layers))
                self.assertEqual(len(copy.waits), 1 + max(0, count - 2))
                self.assertEqual(pipeline.ready, {})

    def test_current_layer_progresses_before_future_or_second_projection_copy(self):
        """Future H2D must not gate dispatch; W2 must not gate the first GMM."""
        context, copy = _Context(), _Stream()
        banks, submitted, observed = [[None, None], [None, None]], [], []

        def load(layer, bank, ctx):
            submitted.append(layer)
            copy.enqueue(lambda: banks[bank].__setitem__(0, layer))
            first = SimpleNamespace(completion=_Completion(copy.tail))
            copy.enqueue(lambda: banks[bank].__setitem__(1, layer + 1))
            return PrefillLayerTransfer(
                first, SimpleNamespace(completion=_Completion(copy.tail))
            )

        pipeline = LayerPrefetchPipeline((3, 4, 5), load=load, copy_stream=copy)
        pipeline.begin_forward(context)
        current, future = pipeline.ready[0], pipeline.ready[1]
        pipeline.start(3, context)
        context.compute_stream.enqueue(lambda: observed.append("dispatch"))
        context.compute_stream.drain()
        self.assertEqual(observed, ["dispatch"])
        self.assertEqual(banks, [[None, None], [None, None]])

        bank = pipeline.acquire(3, context)
        context.compute_stream.enqueue(lambda: observed.append(banks[bank][0] * 2))
        context.compute_stream.drain()
        self.assertEqual(observed, ["dispatch", 6])
        self.assertFalse(current.w2.completion.command.done)
        self.assertFalse(future.w13.completion.command.done)

        pipeline.wait_for_w2(3, context)
        context.compute_stream.enqueue(
            lambda: observed.append(observed[-1] * banks[bank][1])
        )
        context.compute_stream.enqueue(lambda: observed.append("combine"))
        context.compute_stream.drain()
        self.assertEqual(observed, ["dispatch", 6, 24, "combine"])
        self.assertFalse(future.w13.completion.command.done)
        self.assertFalse(future.w2.completion.command.done)
        pipeline.retire(3, context.record_compute_completion(), context)
        # The next refill is submitted while the existing lookahead is pending,
        # so this layer's compute/combine can fill that transfer's idle window.
        self.assertEqual(submitted, [3, 4, 5])
        self.assertFalse(future.w2.completion.command.done)
        copy.drain()
        self.assertEqual(banks, [[5, 6], [4, 5]])

    def test_empty_layer_reuses_bank_safely_without_waiting_for_w2(self):
        """An empty EP rank skips GMM2 but FIFO still protects both bank phases."""
        context, copy, writes = _Context(), _Stream(), []

        def load(layer, bank, ctx):
            copy.enqueue(lambda: writes.append((bank, layer, "w13")))
            first = SimpleNamespace(completion=_Completion(copy.tail))
            copy.enqueue(lambda: writes.append((bank, layer, "w2")))
            return PrefillLayerTransfer(
                first, SimpleNamespace(completion=_Completion(copy.tail))
            )

        pipeline = LayerPrefetchPipeline((3, 4, 5), load=load, copy_stream=copy)
        pipeline.begin_forward(context)
        pending = pipeline.ready[0].w2.completion.command
        pipeline.start(3, context)
        pipeline.acquire(3, context)
        context.compute_stream.enqueue(lambda: writes.append((0, 3, "combine")))
        context.compute_stream.drain()
        self.assertFalse(pending.done)
        pipeline.retire(3, context.record_compute_completion(), context)
        copy.drain()
        self.assertLess(writes.index((0, 3, "combine")), writes.index((0, 5, "w13")))
        self.assertLess(writes.index((0, 3, "w2")), writes.index((0, 5, "w13")))

    def test_partial_forward_and_wrong_layer_are_rejected(self):
        context, copy = _Context(), _Stream()
        pipeline = LayerPrefetchPipeline(
            (3, 4),
            load=lambda *args: PrefillLayerTransfer(
                SimpleNamespace(completion=_Completion(None)),
                SimpleNamespace(completion=_Completion(None)),
            ),
            copy_stream=copy,
        )
        with self.assertRaisesRegex(RuntimeError, "forward entry"):
            pipeline.start(3, context)
        pipeline.begin_forward(context)
        with self.assertRaisesRegex(RuntimeError, "model order"):
            pipeline.start(4, context)
        with self.assertRaisesRegex(RuntimeError, "previous prefill"):
            pipeline.begin_forward(_Context())
        pipeline.start(3, context)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            pipeline.start(3, context)
        with self.assertRaisesRegex(RuntimeError, "model order"):
            pipeline.acquire(4, context)
        with self.assertRaisesRegex(RuntimeError, "every offloaded"):
            pipeline.finish_forward(context)
        with self.assertRaisesRegex(RuntimeError, "model order"):
            pipeline.acquire(3, _Context())

    def test_next_forward_cannot_reuse_previous_ready_event(self):
        copy, submitted = _Stream(), []

        def load(layer, bank, context):
            submitted.append(context.invocation_id)
            return PrefillLayerTransfer(
                SimpleNamespace(completion=_Completion(None)),
                SimpleNamespace(completion=_Completion(None)),
            )

        pipeline = LayerPrefetchPipeline((3,), load=load, copy_stream=copy)
        for _ in range(2):
            context = _Context()
            pipeline.begin_forward(context)
            pipeline.start(3, context)
            pipeline.acquire(3, context)
            pipeline.retire(3, context.record_compute_completion(), context)
            pipeline.finish_forward(context)
            pipeline.reset()
        self.assertEqual(len(submitted), 2)
        self.assertIsNot(submitted[0], submitted[1])


class TestPrefillProjectionPlans(CustomTestCase):
    def test_phase_completion_includes_projection_scales_and_offsets(self):
        """An early W13 event must publish its quantization bytes, but not await W2."""
        from sglang.srt.offloading.moe.npu.prefill import _projection_copy_plans

        names = (
            "w2_weight",
            "w13_weight",
            "w2_scale",
            "w13_scale",
            "w2_offset",
            "w13_offset",
        )
        layouts = {name: SimpleNamespace(nbytes=8) for name in names}
        sources = {
            name: tuple(
                SimpleNamespace(
                    nbytes=8,
                    payload=torch.full(
                        (8,), 10 * index + expert + 1, dtype=torch.uint8
                    ),
                )
                for expert in range(2)
            )
            for index, name in enumerate(names)
        }
        destinations = {
            name: tuple(
                SimpleNamespace(nbytes=8, payload=torch.zeros(8, dtype=torch.uint8))
                for _ in range(2)
            )
            for name in names
        }
        storage = SimpleNamespace(component_capabilities=sources)
        bank = SimpleNamespace(component_regions=destinations)
        plans = _projection_copy_plans(storage, bank, layouts)
        copy = _Stream()

        def submit(plan, bank, stream):
            for region in plan.regions:
                stream.enqueue(
                    lambda region=region: region.destination.payload.copy_(
                        region.source.payload
                    )
                )
            return SimpleNamespace(completion=_Completion(stream.tail))

        weights = object.__new__(AscendPrefillWeights)
        weights.borrows_hot, weights.hot_dirty = False, False
        weights.owner = SimpleNamespace(weight_copy_stream=copy)
        weights.banks, weights.plans = (bank,), {3: plans}
        weights.host_ops = SimpleNamespace(materialize_into=submit)
        transfer = weights._load(3, 0, _Context())
        transfer.w13.completion.synchronize()
        for name in names:
            for source, destination in zip(sources[name], destinations[name]):
                if name.startswith("w13_"):
                    self.assertTrue(torch.equal(source.payload, destination.payload))
                else:
                    self.assertEqual(destination.payload.count_nonzero().item(), 0)
        self.assertFalse(transfer.w2.completion.command.done)
        transfer.w2.completion.synchronize()
        for name in names:
            for source, destination in zip(sources[name], destinations[name]):
                self.assertTrue(torch.equal(source.payload, destination.payload))

        # A format change must not silently omit an unclassified component or
        # create a ready event for an absent projection.
        for invalid in (
            {"w13_weight": layouts["w13_weight"]},
            {**layouts, "unexpected_scale": layouts["w13_scale"]},
        ):
            with self.subTest(components=tuple(invalid)), self.assertRaises(ValueError):
                _projection_copy_plans(storage, bank, invalid)


class TestAscendPrefillWeights(CustomTestCase):
    def setUp(self):
        super().setUp()
        # Execute the production kernel on CPU, not a second implementation.
        kernel = InterpretedFunction(prepare_prefill_counts.fn)
        self.kernel_patch = patch(
            "sglang.srt.offloading.moe.npu.small_batch_kernels.prepare_prefill_counts",
            kernel,
        )
        self.kernel_patch.start()
        self.addCleanup(self.kernel_patch.stop)
        # tl.sum is itself a jitted helper when imported outside interpreter mode.
        sum_patch = patch.object(tl, "sum", InterpretedFunction(tl.sum.fn))
        sum_patch.start()
        self.addCleanup(sum_patch.stop)

    def test_w4_decode_metadata_selects_bias_and_folded_scale_components(self):
        """A W8 triplet assumption must not route W4 checkpoint-only metadata."""
        from sglang.srt.offloading.moe.npu.small_batch_kernels import (
            prepare_packed_copies,
        )

        kernel = InterpretedFunction(prepare_packed_copies.fn)
        experts, stride, start = 4, 12, 4
        components = (0, 3, 4, 6, 9, 10)
        sources = torch.arange(12 * stride, dtype=torch.int64).reshape(12, stride)
        counts = torch.tensor([3, 0, 2, 0], dtype=torch.int64)
        groups = torch.full_like(counts, -1)
        selected = torch.full((len(components) * experts,), -1, dtype=torch.int64)
        accesses = torch.full((stride,), 7, dtype=torch.int64)
        error = torch.zeros((), dtype=torch.int32)
        enabled = torch.ones((), dtype=torch.bool)
        kernel[(1,)](
            counts,
            error,
            enabled,
            accesses,
            sources,
            groups,
            selected,
            EXPERTS=experts,
            SOURCE_STRIDE=stride,
            START=start,
            COMPONENT_COUNT=len(components),
            SKIP_OFFSETS=False,
            CUMULATIVE=False,
            BLOCK=experts,
            COMPONENT_MAP=sum(c << (4 * p) for p, c in enumerate(components)),
            COMPONENT_BITS=4,
        )
        torch.testing.assert_close(groups, counts)
        torch.testing.assert_close(
            selected.reshape(len(components), experts),
            sources[list(components), start : start + experts],
        )
        expected_accesses = torch.full((stride,), 7, dtype=torch.int64)
        expected_accesses[start : start + experts] += counts
        torch.testing.assert_close(accesses, expected_accesses)
        self.assertEqual(error.item(), 0)

    def test_cumulative_counts_keep_zero_groups_and_cache_usage_in_layer_order(self):
        weights = object.__new__(AscendPrefillWeights)
        weights.counts = torch.empty(4, dtype=torch.int64)
        weights.resident_experts = 0
        context = _Context()
        usage = torch.zeros(8, dtype=torch.int64)
        weights.owner = SimpleNamespace(
            local_experts=4,
            error=torch.zeros((), dtype=torch.int32),
            state=SimpleNamespace(
                step_access_count=usage, update_enabled=torch.tensor(True)
            ),
            catalog=SimpleNamespace(key=lambda layer, expert: 4 + expert),
            context=context,
        )
        weights.quant_info = (object(), object())
        weights.pipeline = SimpleNamespace(acquire=lambda layer, ctx: 1)
        inputs = SimpleNamespace(
            expert_tokens=torch.tensor([0, 3, 3, 5]), group_list_type=0
        )
        quant = weights.weights(4, inputs)
        self.assertIs(quant, weights.quant_info[1])
        self.assertEqual(inputs.expert_tokens.tolist(), [0, 3, 0, 2])
        self.assertEqual(inputs.group_list_type, 1)
        self.assertEqual(usage.tolist(), [0, 0, 0, 0, 0, 3, 0, 2])
        weights.owner.state.update_enabled.fill_(False)
        weights.weights(4, inputs)
        self.assertEqual(usage.tolist(), [0, 0, 0, 0, 0, 3, 0, 2])
        weights.owner.error.fill_(1)
        weights.weights(4, inputs)
        self.assertEqual(inputs.expert_tokens.tolist(), [0, 0, 0, 0])

    def test_prefill_only_counts_need_no_lfu_state(self):
        weights = object.__new__(AscendPrefillWeights)
        weights.counts = torch.empty(4, dtype=torch.int64)
        weights.resident_experts = 0
        context = _Context()
        weights.owner = SimpleNamespace(
            local_experts=4,
            error=torch.zeros((), dtype=torch.int32),
            state=None,
            catalog=SimpleNamespace(key=lambda layer, expert: expert),
            context=context,
        )
        weights.quant_info = (object(), object())
        weights.pipeline = SimpleNamespace(acquire=lambda layer, ctx: 0)
        inputs = SimpleNamespace(
            expert_tokens=torch.tensor([0, 3, 3, 5]), group_list_type=0
        )
        original = inputs.expert_tokens
        self.assertIs(weights.weights(0, inputs), weights.quant_info[0])
        self.assertEqual(inputs.expert_tokens.tolist(), [0, 3, 0, 2])
        self.assertEqual(inputs.group_list_type, 1)
        self.assertIs(context.resources[0], original)
        weights.owner.error.fill_(1)
        weights.weights(0, inputs)
        self.assertEqual(inputs.expert_tokens.tolist(), [0, 0, 0, 0])

    def test_physical_groups_keep_logical_counts_and_clear_previous_layer(self):
        from sglang.srt.offloading.moe.npu import small_batch_kernels

        weights = object.__new__(AscendPrefillWeights)
        weights.resident_experts, weights.resident_slots = 1, 3
        weights.counts = torch.empty(3, dtype=torch.int64)
        weights.groups = torch.zeros((7, 2), dtype=torch.int64)
        weights.banks = (SimpleNamespace(capacity=2),) * 2
        weights.quant_info = (object(),) * 2
        weights.pipeline = SimpleNamespace(current=1, acquire=lambda layer, ctx: 1)
        weights.owner = SimpleNamespace(
            local_experts=3,
            error=torch.zeros((), dtype=torch.int32),
            state=None,
            catalog=SimpleNamespace(key=lambda layer, expert: expert),
            context=_Context(),
        )

        with (
            patch.object(
                small_batch_kernels,
                "_store_slot_count_pairs",
                InterpretedFunction(small_batch_kernels._store_slot_count_pairs.fn),
            ),
            patch.object(tl, "cumsum", InterpretedFunction(tl.cumsum.fn)),
        ):
            for counts, expected, pairs in (
                ([2, 2, 5], [2, 0, 3], [[1, 2], [6, 3]]),
                ([0, 4, 4], [0, 4, 0], [[5, 4]]),
                ([0, 0, 0], [0, 0, 0], []),
            ):
                inputs = SimpleNamespace(
                    expert_tokens=torch.tensor(counts, dtype=torch.int64),
                    group_list_type=0,
                )
                self.assertIs(weights.weights(4, inputs), weights.quant_info[1])
                self.assertEqual(inputs.group_list_type, 2)
                self.assertEqual(weights.counts.tolist(), expected)
                self.assertEqual(
                    inputs.expert_tokens.tolist(), pairs + [[0, 0]] * (7 - len(pairs))
                )

    def test_prefill_count_kernel_without_usage_checks_invalid_and_padded_groups(self):
        from sglang.srt.offloading.moe.npu.small_batch_kernels import (
            prepare_prefill_counts as kernel,
        )

        for raw, cumulative, expected, failed in (
            ([2, 0, 3], False, [2, 0, 3], False),
            ([2, 2, 5], True, [2, 0, 3], False),
            ([0, 0, 0], True, [0, 0, 0], False),
            ([2, -1, 3], False, [0, 0, 0], True),
            ([2, 1, 5], True, [0, 0, 0], True),
        ):
            with self.subTest(raw=raw, cumulative=cumulative):
                counts = torch.tensor(raw, dtype=torch.int64)
                error = torch.zeros((), dtype=torch.int32)
                output = torch.full((5,), -9, dtype=torch.int64)
                kernel[(1,)](
                    counts,
                    error,
                    None,
                    None,
                    output[1:4],
                    EXPERTS=3,
                    CUMULATIVE=cumulative,
                    BLOCK=4,
                    TRACK_USAGE=False,
                )
                self.assertEqual(counts.tolist(), raw)
                self.assertEqual(output.tolist(), [-9, *expected, -9])
                self.assertEqual(error.item(), int(failed))

    def test_count_kernel_preserves_padding_and_latches_invalid_groups(self):
        # Three experts exercise the masked fourth lane. Invalid counts must
        # stop every group, including after a later valid layer is submitted.
        from sglang.srt.offloading.moe.npu.small_batch_kernels import (
            prepare_prefill_counts as kernel,
        )

        for raw, cumulative, expected, failed in (
            ([2, 0, 3], False, [2, 0, 3], False),
            ([2, 2, 5], True, [2, 0, 3], False),
            ([0, 0, 0], True, [0, 0, 0], False),
            ([2, -1, 3], False, [0, 0, 0], True),
            ([2, 1, 5], True, [0, 0, 0], True),
        ):
            with self.subTest(raw=raw, cumulative=cumulative):
                counts = torch.tensor(raw, dtype=torch.int64)
                error = torch.zeros((), dtype=torch.int32)
                enabled = torch.tensor(True)
                usage = torch.tensor([100, 1, 2, 3, 100], dtype=torch.int64)
                out = torch.full((5,), -9, dtype=torch.int64)

                def launch():
                    kernel[(1,)](
                        counts,
                        error,
                        enabled,
                        usage[1:4],
                        out[1:4],
                        EXPERTS=3,
                        CUMULATIVE=cumulative,
                        BLOCK=4,
                    )

                launch()
                self.assertEqual(counts.tolist(), raw)
                self.assertEqual(out.tolist(), [-9, *expected, -9])
                self.assertEqual(
                    usage.tolist(),
                    [100, *[a + b for a, b in zip((1, 2, 3), expected)], 100],
                )
                self.assertEqual(error.item(), int(failed))
                previous_usage = usage.clone()
                counts.copy_(torch.tensor([1, 2, 3]))
                enabled.fill_(False)
                launch()
                self.assertTrue(torch.equal(usage, previous_usage))
                if failed:
                    self.assertEqual(out.tolist(), [-9, 0, 0, 0, -9])
                    self.assertEqual(error.item(), 1)

    def test_borrowed_hot_restores_current_slot_owners_after_partial_submission(self):
        """Cache metadata keeps its owners when a prefill H2D fails midway."""
        weights = object.__new__(AscendPrefillWeights)
        copy, compute = _Stream(), _Stream()
        host = torch.tensor([[11, 12], [21, 22], [31, 32]])
        hot = torch.tensor([[31, 32], [11, 12], [91, 92]])
        keys, valid = torch.tensor([2, 0]), torch.tensor([True, True])
        weights.owner = SimpleNamespace(weight_copy_stream=copy)
        weights.borrows_hot, weights.hot_dirty = True, False
        weights.banks, weights.plans = (object(), object()), {4: (object(), object())}
        weights.restore_plan = SimpleNamespace(
            source_index=keys, valid=valid, count=torch.zeros((), dtype=torch.int32)
        )
        weights.restore_bindings = object()

        def partial_copy(*args):
            copy.enqueue(lambda: hot[0].fill_(-1))
            raise RuntimeError("partial H2D")

        def restore(plan, bindings, stream):
            stream.enqueue(lambda: hot[:2].copy_(host[plan.source_index]))

        weights.host_ops = SimpleNamespace(materialize_into=partial_copy)
        weights.restore_ops = SimpleNamespace(materialize_into=restore)
        with self.assertRaisesRegex(RuntimeError, "partial H2D"):
            weights._load(4, 1, _Context())
        self.assertTrue(weights.hot_dirty)
        with patch.object(
            torch, "npu", SimpleNamespace(current_stream=lambda: compute), create=True
        ):
            weights.restore_hot()
        compute.drain()
        copy.drain()
        self.assertEqual(hot.tolist(), [[31, 32], [11, 12], [91, 92]])
        self.assertEqual(keys.tolist(), [2, 0])
        self.assertEqual(weights.restore_plan.count.item(), 2)
        self.assertFalse(weights.hot_dirty)


class TestPrefillAclDescriptors(CustomTestCase):
    def test_old_extension_requires_rebuild_instead_of_silent_flush(self):
        from sglang.srt.offloading.moe.npu.native_acl import AclHostTransferOps

        old_native = SimpleNamespace(
            api_info=lambda: {"queue_ordering": "torch_npu_stream_flush"}
        )
        with self.assertRaisesRegex(ValueError, "[Rr]ebuild"):
            AclHostTransferOps(old_native, lambda value: None)

    def test_preparation_covers_complete_bundle_and_reuses_native_descriptors(self):
        from sglang.srt.offloading.moe.npu.native_acl import AclHostTransferOps
        from sglang.srt.offloading.moe.npu.transfer import DeviceMemoryRegion
        from sglang.srt.offloading.moe.storage import HostMemoryCapability
        from sglang.srt.offloading.transfer import HostCopyPlan, HostCopyRegion

        batches = []
        device = SimpleNamespace(type="npu", index=0)

        def batch(*args):
            batches.append(args)
            return object()

        native = SimpleNamespace(
            api_info=lambda: {"queue_ordering": "torch_npu_op_queue"},
            Batch=batch,
        )
        ops = AclHostTransferOps(native, lambda value: None, max_batch_entries=3)
        regions = tuple(
            HostCopyRegion(
                HostMemoryCapability(4096 + index * 128, 64, True, None, native),
                DeviceMemoryRegion(16384 + index * 64, 64, device, native),
                0,
                0,
                64,
            )
            for index in range(12)
        )
        plan = HostCopyPlan(regions)
        ops.prepare(plan, device)
        self.assertEqual(len(batches), 4)
        self.assertEqual(
            [address for b in batches for address in b[0]],
            [r.source.host_address for r in regions],
        )
        self.assertEqual(
            [address for b in batches for address in b[1]],
            [r.destination.address for r in regions],
        )
        self.assertEqual([size for b in batches for size in b[2]], [64] * 12)
        ops.prepare(plan, device)
        self.assertEqual(len(batches), 4)

    def _lower_copy_regions(self, regions, *, max_batch_entries=256):
        from sglang.srt.offloading.moe.npu.native_acl import AclHostTransferOps
        from sglang.srt.offloading.transfer import HostCopyPlan

        device = SimpleNamespace(type="npu", index=0)
        batches = []

        def batch(*args):
            batches.append(args)
            return args

        native = SimpleNamespace(
            api_info=lambda: {"queue_ordering": "torch_npu_op_queue"}, Batch=batch
        )
        ops = AclHostTransferOps(
            native, lambda value: None, max_batch_entries=max_batch_entries
        )
        plan = HostCopyPlan(regions)
        ops.prepare(plan, device)
        return batches, ops, plan, device

    def _copy_regions_for_buffers(self, source, destination, ranges, *, offsets=False):
        from sglang.srt.offloading.moe.npu.transfer import DeviceMemoryRegion
        from sglang.srt.offloading.moe.storage import HostMemoryCapability
        from sglang.srt.offloading.transfer import HostCopyRegion

        device = SimpleNamespace(type="npu", index=0)
        return tuple(
            HostCopyRegion(
                HostMemoryCapability(
                    ctypes.addressof(source) + (0 if offsets else src),
                    len(source) if offsets else max(1, size),
                    True,
                    None,
                    source,
                ),
                DeviceMemoryRegion(
                    ctypes.addressof(destination) + (0 if offsets else dst),
                    len(destination) if offsets else max(1, size),
                    device,
                    destination,
                ),
                src if offsets else 0,
                dst if offsets else 0,
                size,
            )
            for src, dst, size in ranges
        )

    def _execute_copy_descriptors(self, batches):
        for srcs, dsts, sizes, capacities, *_ in batches:
            for src, dst, size, capacity in zip(srcs, dsts, sizes, capacities):
                self.assertLessEqual(size, capacity)
                ctypes.memmove(dst, src, size)

    def test_coalesced_ranges_preserve_bytes_and_untouched_gaps(self):
        rng = random.Random(24072471)
        for case in range(32):
            with self.subTest(case=case):
                source = (ctypes.c_ubyte * 1024)(
                    *range(256), *range(256), *range(256), *range(256)
                )
                destination = (ctypes.c_ubyte * 1024)(*([0xA5] * 1024))
                ranges, src, dst = [], 3, 5
                expected = bytearray(destination)
                for _ in range(32):
                    size = rng.randrange(1, 17)
                    ranges.append((src, dst, size))
                    expected[dst : dst + size] = bytes(source)[src : src + size]
                    src += size + rng.choice((0, 0, 0, 1, 3))
                    dst += size + rng.choice((0, 0, 0, 2, 4))
                regions = self._copy_regions_for_buffers(
                    source, destination, ranges, offsets=bool(case % 2)
                )
                batches, ops, plan, device = self._lower_copy_regions(
                    regions, max_batch_entries=3
                )
                self.assertTrue(all(len(batch[0]) <= 3 for batch in batches))
                self.assertEqual(
                    sum(sum(batch[2]) for batch in batches),
                    sum(size for _, _, size in ranges),
                )
                self._execute_copy_descriptors(batches)
                self.assertEqual(bytes(destination), bytes(expected))
                self.assertIs(ops._prepared[(device.index, id(plan))][0], plan)
                self.assertEqual(plan.regions, regions)
                # Repeated use must reuse the lowered descriptor owners.
                descriptor_batches = len(batches)
                ops.prepare(plan, device)
                self.assertEqual(len(batches), descriptor_batches)

    def test_w8a8_scale_padding_stays_out_of_packed_destinations(self):
        regions, allocations, expected = [], [], []
        # Reduced byte extents preserve the real layout's FP32-host/BF16-runtime
        # scale stride, six components and 32 experts, without a large allocation.
        for component, (size, stride) in enumerate(
            ((16, 16), (2, 4), (4, 4), (8, 8), (2, 4), (4, 4))
        ):
            source = (ctypes.c_ubyte * (32 * stride))(*([0xCD] * (32 * stride)))
            destination = (ctypes.c_ubyte * (32 * size + 8))(
                *([0xA5] * (32 * size + 8))
            )
            valid = bytearray()
            for expert in range(32):
                payload = bytes(
                    (component * 31 + expert * 7 + byte) % 251 for byte in range(size)
                )
                source[expert * stride : expert * stride + size] = payload
                valid.extend(payload)
            ranges = [
                (expert * stride, expert * size + 4, size) for expert in range(32)
            ]
            regions.extend(self._copy_regions_for_buffers(source, destination, ranges))
            allocations.append((source, destination))
            expected.append(b"\xa5" * 4 + valid + b"\xa5" * 4)
        batches, _, _, _ = self._lower_copy_regions(regions)
        self.assertEqual(len(regions), 192)
        self.assertEqual(sum(len(batch[0]) for batch in batches), 68)
        self._execute_copy_descriptors(batches)
        for (_, destination), reference in zip(allocations, expected):
            self.assertEqual(bytes(destination), reference)

    def test_contiguous_addresses_do_not_merge_across_allocation_owners(self):
        from dataclasses import replace

        source = (ctypes.c_ubyte * 16)(*range(16))
        destination = (ctypes.c_ubyte * 16)()
        regions = list(
            self._copy_regions_for_buffers(
                source, destination, [(i * 4, i * 4, 4) for i in range(4)]
            )
        )
        # Identity is required: owners may compare equal despite being distinct.
        registrations = (
            SimpleNamespace(storage=source),
            SimpleNamespace(storage=source),
        )
        owners = (
            SimpleNamespace(storage=destination),
            SimpleNamespace(storage=destination),
        )
        for i in range(4):
            regions[i] = replace(
                regions[i],
                source=replace(
                    regions[i].source, registration=registrations[0 if i < 2 else 1]
                ),
                destination=replace(
                    regions[i].destination, owner=owners[0 if i == 0 else 1]
                ),
            )
        batches, _, _, _ = self._lower_copy_regions(regions)
        self.assertEqual([size for batch in batches for size in batch[2]], [4, 4, 8])
        self._execute_copy_descriptors(batches)
        self.assertEqual(bytes(destination), bytes(source))

    def test_invalid_late_regions_are_rejected_before_native_descriptors(self):
        from dataclasses import replace

        from sglang.srt.offloading.moe.npu.native_acl import AclHostTransferOps
        from sglang.srt.offloading.transfer import HostCopyPlan

        source = (ctypes.c_ubyte * 16)(*range(16))
        destination = (ctypes.c_ubyte * 16)()
        regions = self._copy_regions_for_buffers(
            source, destination, [(0, 0, 4), (4, 4, 4)]
        )
        cases = (
            (
                "host reads",
                replace(
                    regions[1],
                    source=replace(regions[1].source, acl_async_readable=False),
                ),
            ),
            (
                "submitting stream device",
                replace(
                    regions[1],
                    destination=replace(
                        regions[1].destination,
                        device=SimpleNamespace(type="npu", index=1),
                    ),
                ),
            ),
            (
                "destinations overlap",
                replace(
                    regions[1],
                    destination=replace(
                        regions[1].destination,
                        address=ctypes.addressof(destination) + 2,
                    ),
                ),
            ),
            (
                "immutable host storage",
                replace(
                    regions[1],
                    source=replace(
                        regions[1].source, device_address=ctypes.addressof(destination)
                    ),
                ),
            ),
        )
        for error, invalid in cases:
            with self.subTest(error=error):
                batches = []
                native = SimpleNamespace(
                    api_info=lambda: {"queue_ordering": "torch_npu_op_queue"},
                    Batch=lambda *args: batches.append(args),
                )
                ops = AclHostTransferOps(native, lambda value: None)
                with self.assertRaisesRegex(ValueError, error):
                    ops.prepare(
                        HostCopyPlan((regions[0], invalid)),
                        regions[0].destination.device,
                    )
                self.assertEqual(batches, [])

    def test_empty_region_does_not_copy_or_change_adjacent_extent(self):
        source = (ctypes.c_ubyte * 8)(*range(8))
        destination = (ctypes.c_ubyte * 10)(*([0xA5] * 10))
        regions = self._copy_regions_for_buffers(
            source, destination, [(0, 1, 4), (4, 5, 0), (4, 5, 4)]
        )
        batches, _, _, _ = self._lower_copy_regions(regions)
        self.assertEqual([size for batch in batches for size in batch[2]], [8])
        self._execute_copy_descriptors(batches)
        self.assertEqual(bytes(destination), b"\xa5" + bytes(source) + b"\xa5")


class TestAscendPrefillIntegration(CustomTestCase):
    def _combine_call(self, owner, layer, mode, values):
        from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
            AscendOpsCombineInput,
            AscendOpsDispatcher,
            _AllGatherState,
            _AllToAllVState,
            _MC2State,
        )

        dispatcher = object.__new__(AscendOpsDispatcher)
        dispatcher._offload_layer = SimpleNamespace(layer_id=layer)
        dispatcher.hidden_size = values.shape[1]
        if owner is not None:
            dispatcher._offload_layer._ascend_ops_offloader = owner
        rows = values.shape[0]
        if mode == "allgather":
            state = _AllGatherState(object(), [rows], values.dtype)
        elif mode == "alltoallv":
            state = _AllToAllVState(
                object(),
                [rows],
                [rows],
                torch.ones((rows, 1)),
                torch.zeros((rows, 1), dtype=torch.int32),
                torch.arange(rows, dtype=torch.int32),
                values.shape,
            )
        else:
            dispatcher._ep_comm_name = "test"
            dispatcher.ep_size = dispatcher.num_experts = 1
            dispatcher.ep_rank = 0
            dispatcher.moe_ascend_ops_config = SimpleNamespace(ll_combine_alg="")
            state = _MC2State(max(0, rows - 1), None, None, None, None, None, None)
        return dispatcher, AscendOpsCombineInput(values, state)

    @contextmanager
    def _combine_device(self, context, communication, trace):
        """Model a collective's stream handoff, not the dispatcher's decisions."""

        def collective(output, values, **kwargs):
            trace.append("combine-submit")
            communication.wait_stream(context.compute_stream)

            def complete():
                output.copy_(values)
                trace.append("combine-done")

            communication.enqueue(complete)
            # Synchronous dist calls make their output available to subsequent
            # work on the caller stream without waiting on the Python thread.
            context.compute_stream.wait_stream(communication)

        def finalize(values, *args, **kwargs):
            context.compute_stream.enqueue(lambda: trace.append("finalize-done"))
            return values

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_rank",
                return_value=0,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.all_to_all_single",
                side_effect=collective,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.reduce_scatter_tensor",
                side_effect=collective,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.npu_moe_finalize_routing",
                side_effect=finalize,
            ),
        ):
            yield

    def _forward_owner(self, *, fail_second_load=False, fail_w2=False):
        from sglang.srt.offloading.moe.npu.ascend_ops import AscendOpsOffloader

        context, copy = _Context(), _Stream()
        banks, submitted = [torch.tensor([0]), torch.tensor([99])], []
        from sglang.srt.offloading.moe.npu.execution_mode import CachedExecution

        owner = object.__new__(AscendOpsOffloader)
        owner.execution_mode = CachedExecution()
        owner.ready, owner.step, owner.decode_active_rows = True, 0, None
        owner.weight_copy_stream = copy
        owner.execution_keys = None
        owner.state = SimpleNamespace(
            step_access_count=torch.zeros(3, dtype=torch.int64),
            update_enabled=torch.tensor(False),
        )
        owner.policy = SimpleNamespace(config=SimpleNamespace(update_interval=4))
        owner._check_error = lambda: None
        owner.cache_tail_graphs = {
            due: SimpleNamespace(replay=lambda: None) for due in (False, True)
        }
        weights = object.__new__(AscendPrefillWeights)
        owner.prefill, weights.owner = weights, owner
        weights.borrows_hot, weights.hot_dirty = True, False
        weights.banks = banks
        weights.plans = {layer: ((layer, "w13"), (layer, "w2")) for layer in (3, 4, 5)}
        weights.restore_plan = SimpleNamespace(
            valid=torch.tensor([True]), count=torch.zeros((), dtype=torch.int32)
        )
        weights.restore_bindings = object()

        def load(plan, bank, stream):
            layer, phase = plan
            if phase == "w13":
                submitted.append(layer)
            stream.enqueue(lambda: bank.fill_(layer))
            if layer == 4 and (fail_second_load or (fail_w2 and phase == "w2")):
                raise RuntimeError("partial initial H2D")
            return SimpleNamespace(completion=_Completion(stream.tail))

        def restore(plan, bindings, stream):
            stream.enqueue(lambda: banks[1].fill_(99))

        weights.host_ops = SimpleNamespace(materialize_into=load)
        weights.restore_ops = SimpleNamespace(materialize_into=restore)
        weights.pipeline = LayerPrefetchPipeline(
            (3, 4, 5), load=weights._load, copy_stream=copy
        )
        return owner, context, banks, submitted

    def test_forward_entry_prefetch_precedes_model_on_all_prefill_ranks(self):
        """Initial copies must be submitted before even resident-layer attention."""
        for global_extend, local_extend, idle in (
            (True, True, False),
            (True, False, True),
            (True, False, False),
        ):
            with self.subTest(
                global_extend=global_extend, local_extend=local_extend, idle=idle
            ):
                owner, context, banks, submitted = self._forward_owner()
                # Decode identities become stale as soon as prefill overwrites
                # the shared execution bank, including on locally idle ranks.
                owner.execution_keys = torch.tensor([31, 32], dtype=torch.int64)
                copy_keys = []
                materialize = owner.prefill.host_ops.materialize_into

                def snapshot_keys(plan, bank, stream):
                    copy_keys.append(owner.execution_keys.clone())
                    return materialize(plan, bank, stream)

                owner.prefill.host_ops.materialize_into = snapshot_keys
                batch = SimpleNamespace(
                    is_extend_in_batch=global_extend,
                    forward_mode=SimpleNamespace(
                        is_extend_without_speculative=lambda: local_extend,
                        is_idle=lambda: idle,
                    ),
                )
                observed = []
                with (
                    patch(
                        "sglang.srt.offloading.moe.npu.ascend_ops._Context",
                        return_value=context,
                    ),
                    patch(
                        "sglang.srt.layers.dp_attention.get_is_extend_in_batch",
                        return_value=False,
                    ),
                    patch.object(
                        torch,
                        "npu",
                        SimpleNamespace(current_stream=lambda: context.compute_stream),
                        create=True,
                    ),
                ):
                    with owner.forward_scope(batch):
                        self.assertEqual(submitted, [3, 4])
                        self.assertEqual(len(copy_keys), 4)
                        self.assertTrue(
                            all(
                                torch.equal(keys, torch.full_like(keys, -1))
                                for keys in copy_keys
                            )
                        )
                        # Represents attention / resident MoE before the first
                        # offloaded dispatcher can consume a layer entry.
                        context.compute_stream.enqueue(
                            lambda: observed.append("resident")
                        )
                        owner.weight_copy_stream.drain()
                        self.assertEqual(observed, [])
                        self.assertEqual([bank.item() for bank in banks], [3, 4])
                        for layer in (3, 4, 5):
                            owner.start_prefill_layer(layer)
                            bank = owner.prefill.pipeline.acquire(layer, context)
                            context.compute_stream.enqueue(
                                lambda bank=bank: observed.append(banks[bank].item())
                            )
                            owner.prefill.finish_layer(layer)
                self.assertEqual(submitted, [3, 4, 5])
                self.assertEqual(observed, ["resident", 3, 4, 5])
                self.assertEqual(banks[1].item(), 99)
                self.assertIsNone(owner.prefill.pipeline.invocation_id)

    def test_decode_verify_and_global_idle_ignore_stale_prefill_flag(self):
        for name, idle in (
            ("decode", False),
            ("target_verify", False),
            ("idle", True),
            ("local_extend_label", False),
        ):
            with self.subTest(mode=name):
                owner, context, _, submitted = self._forward_owner()
                owner.execution_keys = torch.tensor([31, 32], dtype=torch.int64)
                previous_keys = owner.execution_keys.clone()
                batch = SimpleNamespace(
                    is_extend_in_batch=False,
                    forward_mode=SimpleNamespace(
                        is_extend_without_speculative=lambda: (
                            name == "local_extend_label"
                        ),
                        is_idle=lambda: idle,
                    ),
                )
                with (
                    patch(
                        "sglang.srt.offloading.moe.npu.ascend_ops._Context",
                        return_value=context,
                    ),
                    patch(
                        "sglang.srt.layers.dp_attention.get_is_extend_in_batch",
                        return_value=True,
                    ),
                ):
                    with owner.forward_scope(batch):
                        self.assertEqual(submitted, [])
                self.assertEqual(submitted, [])
                self.assertEqual(owner.step, 1)
                self.assertTrue(torch.equal(owner.execution_keys, previous_keys))

    def test_initial_prefetch_failure_restores_hot_before_model_entry(self):
        """Failure in either Hot projection must restore decode's published bytes."""
        for phase in ("w13", "w2"):
            with self.subTest(failed_phase=phase):
                owner, context, banks, submitted = self._forward_owner(
                    fail_second_load=phase == "w13", fail_w2=phase == "w2"
                )
                batch = SimpleNamespace(is_extend_in_batch=True)
                entered = False
                with (
                    patch(
                        "sglang.srt.offloading.moe.npu.ascend_ops._Context",
                        return_value=context,
                    ),
                    patch.object(
                        torch,
                        "npu",
                        SimpleNamespace(current_stream=lambda: context.compute_stream),
                        create=True,
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "partial initial H2D"):
                        with owner.forward_scope(batch):
                            entered = True
                self.assertFalse(entered)
                self.assertEqual(submitted, [3, 4])
                self.assertEqual(banks[1].item(), 99)
                self.assertFalse(owner.prefill.hot_dirty)
                self.assertIsNone(owner.prefill.pipeline.invocation_id)
                self.assertFalse(owner.state.update_enabled.item())
                self.assertEqual(owner.step, 0)

    def test_prefill_w2_wait_does_not_pull_in_future_layer(self):
        from sglang.srt.offloading.moe.npu.ascend_ops import AscendOpsOffloader

        owner, context, banks, _ = self._forward_owner()
        owner.context = context
        owner.prefill.pipeline.begin_forward(context)
        owner.start_prefill_layer(3)
        owner.prefill.pipeline.acquire(3, context)
        context.compute_stream.drain()
        current = owner.prefill.pipeline.ready[0]
        future = owner.prefill.pipeline.ready[1]
        self.assertFalse(current.w2.completion.command.done)
        with patch(
            "sglang.srt.layers.dp_attention.get_is_extend_in_batch",
            return_value=True,
        ):
            AscendOpsOffloader.wait_for_w2(owner, 3)
        context.compute_stream.drain()
        self.assertTrue(current.w2.completion.command.done)
        self.assertFalse(future.w13.completion.command.done)
        self.assertEqual([bank.item() for bank in banks], [3, 99])

    def test_decode_keeps_packed_preparation_and_w2_event(self):
        from sglang.srt.offloading.moe.npu.ascend_ops import AscendOpsOffloader
        from sglang.srt.offloading.moe.npu.execution_mode import CachedExecution

        owner = object.__new__(AscendOpsOffloader)
        owner.execution_mode = CachedExecution()
        owner.ready = True
        owner._prepare_fused = Mock(return_value="packed decode weights")
        owner.prefill = SimpleNamespace(
            weights=Mock(side_effect=AssertionError("prefill"))
        )
        event, stream, inputs = object(), Mock(), object()
        owner.copy_events = {4: (None, None, event)}
        with (
            patch(
                "sglang.srt.layers.dp_attention.get_is_extend_in_batch",
                return_value=False,
            ),
            patch.object(
                torch,
                "npu",
                SimpleNamespace(current_stream=lambda: stream),
                create=True,
            ),
        ):
            self.assertEqual(owner.prepare_weights(4, inputs), "packed decode weights")
            owner.wait_for_w2(4)
        owner._prepare_fused.assert_called_once_with(4, inputs)
        stream.wait_event.assert_called_once_with(event)
        owner.prefill.weights.assert_not_called()

    def test_dispatch_progresses_with_pending_prefetch_even_for_empty_ranks(self):
        """Count collectives must not inherit a dependency on future weights."""
        from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
            AscendOpsDispatcher,
        )

        for mode in ("allgather", "alltoallv"):
            for rows in (0, 2):
                with self.subTest(mode=mode, local_rows=rows):
                    owner, context, banks, _ = self._forward_owner()
                    owner.context = context
                    owner.prefill.pipeline.begin_forward(context)
                    owner.prefill.pipeline.ready[0].w2.completion.synchronize()
                    self.assertEqual([b.item() for b in banks], [3, 99])
                    communication, observed = _Stream(), []
                    dispatcher = object.__new__(AscendOpsDispatcher)
                    dispatcher._offload_layer = SimpleNamespace(
                        layer_id=3, _ascend_ops_offloader=owner
                    )
                    dispatcher.moe_ascend_ops_config = SimpleNamespace(prefill=mode)
                    dispatcher.tp_group = object()
                    payload = torch.zeros((rows, 2))

                    def dispatch_device(values, weights, ids, group):
                        communication.wait_stream(context.compute_stream)
                        communication.enqueue(
                            lambda: observed.append([b.item() for b in banks])
                        )
                        context.compute_stream.wait_stream(communication)
                        return values

                    dispatcher._dispatch_allgather = dispatch_device
                    dispatcher._dispatch_alltoallv = dispatch_device
                    with patch(
                        "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_is_extend_in_batch",
                        return_value=True,
                    ):
                        result = dispatcher.dispatch(payload, (None, None, None))
                    self.assertIs(result, payload)
                    self.assertEqual([b.item() for b in banks], [3, 99])
                    self.assertEqual(observed, [])
                    communication.drain()
                    self.assertEqual(observed, [[3, 99]])
                    self.assertFalse(owner.weight_copy_stream.tail.done)

    def test_refill_waits_for_combine_submission_and_device_completion(self):
        """Refilling a consumed bank must neither delay nor race its return collective."""
        for mode in ("allgather", "alltoallv"):
            for rows in (0, 2):
                with self.subTest(mode=mode, received_rows=rows):
                    owner, context, banks, submitted = self._forward_owner()
                    owner.context = context
                    communication, trace = _Stream(), []
                    load = owner.prefill.host_ops.materialize_into

                    def tracked_load(plan, bank, stream):
                        layer, phase = plan
                        trace.append(f"submit-{layer}-{phase}")
                        load(plan, bank, stream)
                        stream.enqueue(lambda: trace.append(f"copy-{layer}-{phase}"))
                        return SimpleNamespace(completion=_Completion(stream.tail))

                    owner.prefill.host_ops.materialize_into = tracked_load
                    owner.prefill.pipeline.begin_forward(context)
                    owner.start_prefill_layer(3)
                    bank = owner.prefill.pipeline.acquire(3, context)
                    values = torch.zeros((rows, 2))
                    context.compute_stream.enqueue(
                        lambda: values.fill_(banks[bank].item())
                    )
                    trace.clear()
                    dispatcher, inputs = self._combine_call(owner, 3, mode, values)
                    with self._combine_device(context, communication, trace):
                        result = dispatcher.combine(inputs)

                    self.assertEqual(submitted, [3, 4, 5])
                    self.assertLess(
                        trace.index("combine-submit"), trace.index("submit-5-w13")
                    )
                    self.assertFalse(communication.tail.done)
                    self.assertEqual(owner.prefill.pipeline.current, 1)
                    # Adversarial scheduling: run the copy stream first. Its new
                    # H2D must still execute the queued return communication
                    # before overwriting bank A, including on an empty rank.
                    owner.weight_copy_stream.drain()
                    self.assertTrue(communication.tail.done)
                    self.assertLess(
                        trace.index("combine-done"), trace.index("copy-5-w13")
                    )
                    if mode == "alltoallv":
                        self.assertLess(
                            trace.index("finalize-done"), trace.index("copy-5-w13")
                        )
                    self.assertTrue(torch.equal(result, torch.full_like(values, 3)))
                    self.assertEqual(banks[0].item(), 5)

    def test_combine_failure_leaves_bank_owned_by_current_layer(self):
        """A failed return collective must not advance or overwrite its layer bank."""
        owner, context, _, submitted = self._forward_owner()
        owner.context = context
        owner.prefill.pipeline.begin_forward(context)
        owner.start_prefill_layer(3)
        owner.prefill.pipeline.acquire(3, context)
        copy_tail = owner.weight_copy_stream.tail
        dispatcher, inputs = self._combine_call(
            owner, 3, "alltoallv", torch.ones((2, 2))
        )
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.all_to_all_single",
            side_effect=RuntimeError("collective enqueue failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "collective enqueue failed"):
                dispatcher.combine(inputs)
        self.assertEqual(owner.prefill.pipeline.current, 0)
        self.assertTrue(owner.prefill.pipeline.started)
        self.assertEqual(submitted, [3, 4])
        self.assertIs(owner.weight_copy_stream.tail, copy_tail)

    def test_decode_and_resident_combine_do_not_require_prefill_session(self):
        from sglang.srt.layers.moe.token_dispatcher import ascend_ops

        def unexpected_retire(layer):
            raise AssertionError("decode cannot consume a prefill bank")

        owner = SimpleNamespace(prefill=SimpleNamespace(finish_layer=unexpected_retire))
        values = torch.tensor([[1.0, 2.0], [99.0, 99.0]])
        dispatcher, inputs = self._combine_call(owner, 3, "mc2", values)
        with patch.object(
            ascend_ops,
            "torch_npu",
            SimpleNamespace(npu_moe_distribute_combine_v2=lambda x, *a, **kw: x * 2),
            create=True,
        ):
            result = dispatcher.combine(inputs)
        self.assertTrue(torch.equal(result, torch.tensor([[2.0, 4.0]])))

        for mode in ("allgather", "alltoallv"):
            with self.subTest(resident_mode=mode):
                context, communication, trace = _Context(), _Stream(), []
                dispatcher, inputs = self._combine_call(None, 0, mode, values)
                with self._combine_device(context, communication, trace):
                    result = dispatcher.combine(inputs)
                context.compute_stream.drain()
                self.assertTrue(torch.equal(result, values))

    def test_hot_restore_follows_combine_and_precedes_maintenance_and_decode(self):
        """Borrowed Hot must survive return communication, then be ready for cache users."""
        owner, context, banks, submitted = self._forward_owner()
        owner.step = 3
        weights, communication, trace, observed = owner.prefill, _Stream(), [], []

        def restore(plan, bindings, stream):
            def complete():
                banks[1].fill_(99)
                trace.append("restore")

            stream.enqueue(complete)

        weights.restore_ops = SimpleNamespace(materialize_into=restore)

        def check_hot():
            self.assertEqual(banks[1].item(), 99)
            trace.append("cache")

        owner.cache_tail_graphs = {
            due: SimpleNamespace(
                replay=lambda: context.compute_stream.enqueue(check_hot)
            )
            for due in (False, True)
        }
        with (
            patch(
                "sglang.srt.offloading.moe.npu.ascend_ops._Context",
                side_effect=lambda: context,
            ),
            patch.object(
                torch,
                "npu",
                SimpleNamespace(current_stream=lambda: context.compute_stream),
                create=True,
            ),
            self._combine_device(context, communication, trace),
        ):
            with owner.forward_scope(SimpleNamespace(is_extend_in_batch=True)):
                for layer in (3, 4, 5):
                    owner.start_prefill_layer(layer)
                    bank = weights.pipeline.acquire(layer, context)
                    values = torch.zeros((1, 2))
                    context.compute_stream.enqueue(
                        lambda bank=bank, values=values: values.fill_(
                            banks[bank].item()
                        )
                    )
                    dispatcher, inputs = self._combine_call(
                        owner, layer, "alltoallv", values
                    )
                    result = dispatcher.combine(inputs)
                    context.compute_stream.enqueue(
                        lambda result=result: observed.append(result[0, 0].item())
                    )
            self.assertEqual(observed, [3, 4, 5])
            self.assertEqual(trace.count("combine-done"), 3)
            self.assertGreater(
                trace.index("restore"),
                max(i for i, item in enumerate(trace) if item == "finalize-done"),
            )
            self.assertLess(trace.index("restore"), trace.index("cache"))
            self.assertFalse(weights.hot_dirty)
            self.assertIsNone(weights.pipeline.invocation_id)
            self.assertFalse(owner.state.update_enabled.item())
            self.assertEqual(context.resources, [])

            # MTP verifies every candidate row, not merely batch_size rows.
            # It uses the restored decode bank without submitting a new
            # prefill transfer; a following idle forward clears the row mask.
            context = _Context()
            previous_submitted = list(submitted)
            owner.decode_active_rows = torch.ones(16, dtype=torch.bool)
            owner.decode_row_indices = torch.arange(16, dtype=torch.int32)
            valid_rows = torch.tensor(6, dtype=torch.int32)
            mask_context = ForwardContext(
                attn_backend=None, moe_num_token_non_padded=lambda: valid_rows
            )
            verify = SimpleNamespace(
                is_extend_in_batch=False,
                batch_size=1,
                input_ids=torch.arange(6),
                forward_mode=SimpleNamespace(is_idle=lambda: False),
            )
            with owner.forward_scope(verify), forward_context(mask_context):
                owner.decode_row_mask(16)
                self.assertEqual(
                    owner.decode_active_rows.tolist(), [True] * 6 + [False] * 10
                )
                context.compute_stream.enqueue(check_hot)
            self.assertEqual(trace[-3:], ["cache", "cache", "cache"])
            self.assertEqual(submitted, previous_submitted)
            idle = SimpleNamespace(
                is_extend_in_batch=False,
                input_ids=torch.arange(6),
                forward_mode=SimpleNamespace(is_idle=lambda: True),
            )
            valid_rows.zero_()
            with owner.forward_scope(idle), forward_context(mask_context):
                owner.decode_row_mask(16)
                self.assertFalse(owner.decode_active_rows.any().item())
            self.assertEqual(submitted, previous_submitted)

    def test_allgather_keeps_native_capacity_without_host_count_read(self):
        from sglang.srt.layers.moe.moe_runner.ascend import _pre_permute_allgather

        payload = torch.arange(16, dtype=torch.int8).reshape(4, 4)
        scale = torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2)
        routed = SimpleNamespace(
            hidden_states=payload,
            scale=scale,
            expert_tokens=torch.tensor([1, 1]),
            expanded_row_idx=torch.arange(4, dtype=torch.int32),
        )
        output = SimpleNamespace(
            hidden_states=payload,
            hidden_states_scale=scale,
            topk_ids=torch.tensor([[0, 2], [3, 1]]),
            topk_weights=torch.ones((2, 2)),
            quant_mode=-1,
            group_list_type=1,
            forward_state=SimpleNamespace(),
        )
        config = SimpleNamespace(
            num_local_experts=2,
            moe_ep_rank=0,
            layer=SimpleNamespace(_ascend_ops_offloader=object()),
        )
        with (
            patch(
                "sglang.srt.layers.moe.moe_runner.ascend.npu_moe_init_routing",
                return_value=routed,
            ),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("host sync")),
        ):
            inputs = _pre_permute_allgather(output, config, {})
        self.assertIs(inputs.hidden_states, payload)
        self.assertIs(inputs.hidden_states_scale, scale)
        self.assertEqual(output.forward_state.num_routes, 4)
        self.assertEqual(inputs.expert_tokens.tolist(), [1, 1])

    def test_empty_receive_leaves_bank_for_combine_without_launching_ffn(self):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore

        retired = []
        owner = SimpleNamespace(
            prepare_weights=lambda *args: object(),
            prefill=SimpleNamespace(finish_layer=retired.append),
        )
        core = object.__new__(AscendRunnerCore)
        core.config = SimpleNamespace(
            layer=SimpleNamespace(layer_id=3, _ascend_ops_offloader=owner)
        )
        inputs = SimpleNamespace(
            hidden_states=torch.empty((0, 4), dtype=torch.int8),
            hidden_states_scale=torch.empty(0),
            expert_tokens=torch.zeros(2, dtype=torch.int64),
            group_list_type=1,
        )
        with patch(
            "sglang.srt.layers.moe.moe_runner.ascend.get_is_extend_in_batch",
            return_value=True,
        ):
            result = core.run(inputs, None, {})
        self.assertEqual(tuple(result.hidden_states.shape), (0, 4))
        self.assertEqual(result.hidden_states.dtype, torch.bfloat16)
        self.assertEqual(retired, [])

    def test_ffn_preserves_results_without_refilling_before_combine(self):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore

        sequence = []

        def projection(value, label, factor):
            sequence.append(label)
            return value * factor

        owner = SimpleNamespace(
            prepare_weights=lambda *args: object(),
            wait_for_w2=lambda layer: sequence.append("w2-ready"),
            prefill=SimpleNamespace(
                finish_layer=lambda layer: sequence.append("retired")
            ),
        )
        core = object.__new__(AscendRunnerCore)
        core.config = SimpleNamespace(
            layer=SimpleNamespace(
                layer_id=3,
                _ascend_ops_offloader=owner,
                w13_kernel=SimpleNamespace(
                    apply=lambda quant, x, counts, **kw: projection(x, "w13", 2)
                ),
                w2_kernel=SimpleNamespace(
                    apply=lambda quant, x, counts, **kw: projection(x, "w2", 3)
                ),
            )
        )
        core.activation = SimpleNamespace(_apply_activation=lambda x: (x.relu(), None))
        inputs = SimpleNamespace(
            hidden_states=torch.tensor([[1.0, -1.0, 2.0, 3.0]]),
            hidden_states_scale=None,
            expert_tokens=torch.tensor([0, 1]),
            group_list_type=1,
        )
        with (
            patch(
                "sglang.srt.layers.moe.moe_runner.ascend.get_is_extend_in_batch",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.moe.moe_runner.ascend._uses_fused_gmm1",
                return_value=False,
            ),
        ):
            result = core.run(inputs, None, {})
        self.assertTrue(
            torch.equal(result.hidden_states, inputs.hidden_states.relu() * 6)
        )
        self.assertEqual(sequence, ["w13", "w2-ready", "w2"])

        # The same runner must not retire/overwrite banks during Decode.
        sequence.clear()
        with (
            patch(
                "sglang.srt.layers.moe.moe_runner.ascend.get_is_extend_in_batch",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.moe.moe_runner.ascend._uses_fused_gmm1",
                return_value=False,
            ),
        ):
            decoded = core.run(inputs, None, {})
        self.assertTrue(torch.equal(decoded.hidden_states, result.hidden_states))
        self.assertEqual(sequence, ["w13", "w2-ready", "w2"])

    def test_activation_uses_logical_counts_for_prefill_and_direct_hot_decode(self):
        """Physical decode slots must not leak into activation or later prefill."""
        from sglang.srt.layers.moe.moe_runner import ascend

        class CPUGroupedActivation(ascend.NPUSitu):
            def _apply_activation(self, values, *, group_list, group_list_type):
                if group_list_type == 0:
                    counts = torch.diff(group_list, prepend=group_list.new_zeros(1))
                elif group_list_type == 1:
                    counts = group_list
                else:
                    raise AssertionError("Activation cannot consume physical slots")
                multipliers = torch.repeat_interleave(torch.tensor([2.0, 5.0]), counts)
                return values * multipliers[:, None], None

        for prefill, physical_slots in ((True, False), (True, True), (False, True)):
            for original_type in (0, 1):
                with self.subTest(
                    prefill=prefill,
                    physical_slots=physical_slots,
                    original_type=original_type,
                ):
                    logical_counts = torch.tensor(
                        [1, 3] if original_type == 0 else [1, 2], dtype=torch.int64
                    )
                    prepared_counts = (
                        torch.tensor([[1, 1], [5, 2], [0, 0]], dtype=torch.int64)
                        if physical_slots
                        else torch.tensor([1, 2], dtype=torch.int64)
                    )
                    prepared_type = 2 if physical_slots else 1

                    def prepare(layer, inputs):
                        inputs.expert_tokens = prepared_counts
                        inputs.group_list_type = prepared_type
                        return None

                    owner = SimpleNamespace(
                        direct_hot=not (prefill and physical_slots),
                        prefill=SimpleNamespace(counts=torch.tensor([1, 2])),
                        # A previous decode's logical counts are unrelated to
                        # the next prefill, even though direct_hot stays enabled.
                        fused_counts=(
                            torch.tensor([3, 0], dtype=torch.int64)
                            if prefill
                            else logical_counts.clone()
                        ),
                        prepare_weights=prepare,
                        wait_for_w2=lambda layer: None,
                    )
                    core = object.__new__(ascend.AscendRunnerCore)
                    core.config = SimpleNamespace(
                        layer=SimpleNamespace(
                            layer_id=3,
                            _ascend_ops_offloader=owner,
                            w13_kernel=SimpleNamespace(
                                apply=lambda q, x, c, **kw: x * 2
                            ),
                            w2_kernel=SimpleNamespace(
                                apply=lambda q, x, c, **kw: x * 3
                            ),
                        )
                    )
                    core.activation = object.__new__(CPUGroupedActivation)
                    inputs = SimpleNamespace(
                        hidden_states=torch.ones((3, 1)),
                        hidden_states_scale=None,
                        expert_tokens=logical_counts,
                        group_list_type=original_type,
                    )
                    with (
                        patch.object(
                            ascend, "get_is_extend_in_batch", return_value=prefill
                        ),
                        patch.object(ascend, "_uses_fused_gmm1", return_value=False),
                    ):
                        result = core.run(inputs, None, {})
                    self.assertTrue(
                        torch.equal(
                            result.hidden_states, torch.tensor([[12.0], [30.0], [30.0]])
                        )
                    )
                    self.assertIs(inputs.expert_tokens, prepared_counts)
                    self.assertEqual(inputs.group_list_type, prepared_type)

    def test_prefill_dispatch_consumes_layer_entry_on_idle_rank(self):
        from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
            AscendOpsDispatcher,
        )

        sequence = []
        dispatcher = object.__new__(AscendOpsDispatcher)
        dispatcher._offload_layer = SimpleNamespace(
            layer_id=3,
            _ascend_ops_offloader=SimpleNamespace(
                start_prefill_layer=lambda layer: sequence.append("enter-prefill")
            ),
        )
        dispatcher.moe_ascend_ops_config = SimpleNamespace(prefill="alltoallv")
        dispatcher.tp_group = object()

        def dispatch(*args):
            sequence.append("dispatch")
            return "received peer tokens"

        dispatcher._dispatch_alltoallv = dispatch
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_is_extend_in_batch",
            return_value=True,
        ):
            result = dispatcher.dispatch(
                torch.empty((0, 4)), (torch.empty((0, 2)), torch.empty((0, 2)), None)
            )
        self.assertEqual(result, "received peer tokens")
        self.assertEqual(sequence, ["enter-prefill", "dispatch"])

        sequence.clear()
        dispatcher._dispatch_low_latency = dispatch
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_is_extend_in_batch",
            return_value=False,
        ):
            dispatcher.dispatch(
                torch.ones((1, 4)), (torch.ones((1, 2)), torch.zeros((1, 2)), None)
            )
        self.assertEqual(sequence, ["dispatch"])


if __name__ == "__main__":
    unittest.main()
