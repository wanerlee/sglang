"""Dedicated P-worker lifecycle contracts with CPU stream boundaries.

The production facade, execution strategy and prefetch pipeline run unchanged;
only device allocation, stream/event submission and host copies are replaced.
"""

import ctypes
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.offloading.moe.npu import ascend_ops
from sglang.srt.offloading.moe.npu.execution_mode import (
    CachedExecution,
    PrefillOnlyExecution,
)
from sglang.srt.offloading.moe.npu.prefill import (
    AscendPrefillWeights,
    LayerPrefetchPipeline,
    PrefillLayerTransfer,
)
from sglang.srt.offloading.moe.resources import ResourceKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Command:
    def __init__(self, dependencies, action):
        self.dependencies, self.action, self.done = dependencies, action, False

    def run(self):
        if not self.done:
            for dependency in self.dependencies:
                dependency.run()
            self.action()
            self.done = True


class _Stream:
    def __init__(self):
        self.tail = None

    def enqueue(self, action, extra=()):
        dependencies = tuple(extra) + (() if self.tail is None else (self.tail,))
        self.tail = _Command(dependencies, action)

    def wait_stream(self, stream):
        if stream.tail is not None:
            self.enqueue(lambda: None, (stream.tail,))


class _Completion:
    def __init__(self, command):
        self.command = command

    def wait_on(self, stream):
        if self.command is not None:
            stream.enqueue(lambda: None, (self.command,))

    def synchronize(self):
        if self.command is not None:
            self.command.run()


class _Context:
    def __init__(self):
        self.compute_stream = _Stream()
        self.invocation_id = object()
        self.resources = []

    def retain(self, value):
        self.resources.append(value)

    def record_compute_completion(self):
        return SimpleNamespace(completion=_Completion(self.compute_stream.tail))


class TestPrefillOnlyRuntime(CustomTestCase):
    @staticmethod
    def _batch(*, global_extend=True, local_idle=False, local_extend=True):
        return SimpleNamespace(
            is_extend_in_batch=global_extend,
            forward_mode=SimpleNamespace(
                is_idle=lambda: local_idle,
                is_extend_without_speculative=lambda: local_extend,
            ),
        )

    @staticmethod
    def _owner(*, fail_copy=None, layer_ids=(0, 2, 4)):
        owner = object.__new__(ascend_ops.AscendOpsOffloader)
        owner.execution_mode = PrefillOnlyExecution()
        owner.ready, owner.step = True, 0
        owner.state = owner.hot = owner.decode_active_rows = None
        owner.error = torch.zeros((), dtype=torch.int32)
        owner.weight_copy_stream = _Stream()
        owner._check_error = Mock()
        context = _Context()
        owner._new_context = Mock(return_value=context)
        submitted, completed = [], []
        weights = object.__new__(AscendPrefillWeights)
        owner.prefill, weights.owner = weights, owner
        weights.borrows_hot = weights.hot_dirty = False

        def load(layer, bank, ctx):
            submitted.append(layer)
            ctx.retain((layer, bank))
            stream = owner.weight_copy_stream
            stream.enqueue(lambda: completed.append((layer, "w13")))
            first = SimpleNamespace(completion=_Completion(stream.tail))
            stream.enqueue(lambda: completed.append((layer, "w2")))
            if layer == fail_copy:
                raise RuntimeError("partial H2D submission")
            return PrefillLayerTransfer(
                first, SimpleNamespace(completion=_Completion(stream.tail))
            )

        weights.pipeline = LayerPrefetchPipeline(
            layer_ids, load=load, copy_stream=owner.weight_copy_stream
        )
        return owner, context, submitted, completed

    def _consume_layers(self, owner, context, completed):
        for layer in owner.prefill.pipeline.layer_ids:
            owner.start_prefill_layer(layer)
            owner.prefill.pipeline.acquire(layer, context)
            owner.prefill.pipeline.wait_for_w2(layer, context)
            context.compute_stream.enqueue(
                lambda layer=layer: completed.append((layer, "combine"))
            )
            owner.prefill.finish_layer(layer)

    def test_all_attention_ranks_follow_global_prefill_and_first_layer_is_streamed(
        self,
    ):
        for local_idle, local_extend in ((False, True), (True, False), (False, False)):
            with self.subTest(local_idle=local_idle, local_extend=local_extend):
                owner, context, submitted, completed = self._owner()
                batch = self._batch(local_idle=local_idle, local_extend=local_extend)
                with owner.forward_scope(batch):
                    self.assertEqual(submitted, [0, 2])
                    self.assertEqual(completed, [])
                    self._consume_layers(owner, context, completed)
                    self.assertTrue(context.resources)
                self.assertEqual(submitted, [0, 2, 4])
                self.assertEqual(
                    [layer for layer, phase in completed if phase == "combine"],
                    [0, 2, 4],
                )
                self.assertLess(
                    completed.index((0, "combine")), completed.index((4, "w13"))
                )
                self.assertEqual(context.resources, [])
                self.assertIsNone(owner.prefill.pipeline.invocation_id)
                owner._check_error.assert_called_once()

    def test_unexpected_global_idle_is_rejected_without_copies_or_error_collective(
        self,
    ):
        owner, _, submitted, completed = self._owner()
        with self.assertRaisesRegex(ValueError, "globally extending"):
            with owner.forward_scope(self._batch(global_extend=False, local_idle=True)):
                self.fail("Unexpected global idle entered the model")
        self.assertEqual(submitted, [])
        self.assertEqual(completed, [])
        owner._check_error.assert_not_called()

    def test_decode_is_rejected_before_model_or_weight_submission(self):
        owner, _, submitted, completed = self._owner()
        with self.assertRaisesRegex((RuntimeError, ValueError), "[Pp]refill|decode"):
            with owner.forward_scope(
                self._batch(global_extend=False, local_extend=False)
            ):
                self.fail("A P-only worker entered a decode model forward")
        self.assertEqual(submitted, [])
        self.assertEqual(completed, [])
        with patch(
            "sglang.srt.layers.dp_attention.get_is_extend_in_batch", return_value=False
        ):
            with self.assertRaisesRegex(
                (RuntimeError, ValueError), "[Pp]refill|decode"
            ):
                owner.prepare_weights(0, object())

    def test_partial_initial_prefetch_drains_before_reset_and_resource_release(self):
        owner, context, submitted, completed = self._owner(fail_copy=2)
        with self.assertRaisesRegex(RuntimeError, "partial H2D submission"):
            with owner.forward_scope(self._batch()):
                self.fail("Model entered after a failed initial prefetch")
        self.assertEqual(submitted, [0, 2])
        self.assertEqual(completed, [(0, "w13"), (0, "w2"), (2, "w13"), (2, "w2")])
        self.assertEqual(context.resources, [])
        self.assertIsNone(owner.prefill.pipeline.invocation_id)
        self.assertEqual(owner.prefill.pipeline.ready, {})
        owner._check_error.assert_not_called()

    def test_model_failure_drains_and_allows_a_fresh_forward(self):
        owner, context, submitted, completed = self._owner()
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            with owner.forward_scope(self._batch()):
                raise RuntimeError("model failure")
        self.assertEqual(len(completed), 4)
        self.assertEqual(context.resources, [])
        self.assertIsNone(owner.prefill.pipeline.invocation_id)
        next_context = _Context()
        owner._new_context.return_value = next_context
        with owner.forward_scope(self._batch()):
            self._consume_layers(owner, next_context, completed)
        self.assertEqual(submitted, [0, 2, 0, 2, 4])
        self.assertEqual(next_context.resources, [])
        owner._check_error.assert_called_once()

    def test_incomplete_forward_is_rejected_and_pending_copies_are_drained(self):
        owner, context, _, completed = self._owner()
        with self.assertRaisesRegex(RuntimeError, "every offloaded"):
            with owner.forward_scope(self._batch()):
                pass
        self.assertEqual(len(completed), 4)
        self.assertEqual(context.resources, [])
        self.assertIsNone(owner.prefill.pipeline.invocation_id)
        owner._check_error.assert_not_called()

    def test_device_error_still_retires_forward_resources(self):
        owner, context, _, completed = self._owner()
        owner._check_error.side_effect = RuntimeError("invalid expert count")
        with self.assertRaisesRegex(RuntimeError, "invalid expert count"):
            with owner.forward_scope(self._batch()):
                self._consume_layers(owner, context, completed)
        self.assertEqual(context.resources, [])
        self.assertIsNone(owner.prefill.pipeline.invocation_id)
        self.assertEqual(len(completed), 9)

    def test_stage_local_sequence_restarts_at_bank_zero_for_each_chunk(self):
        for layers in ((3, 4, 5), (18, 19, 20), (38, 40, 42), (62, 63, 64)):
            with self.subTest(layers=layers):
                owner, context, submitted, completed = self._owner(layer_ids=layers)
                for _ in range(2):
                    with owner.forward_scope(self._batch()):
                        self.assertEqual(owner.prefill.pipeline.current, 0)
                        self.assertEqual(context.resources[0], (layers[0], 0))
                        self._consume_layers(owner, context, completed)
                    self.assertEqual(context.resources, [])
                    self.assertIsNone(owner.prefill.pipeline.invocation_id)
                self.assertEqual(submitted, list(layers) * 2)

    def test_deepep_combine_completion_fences_layer_refill(self):
        from sglang.srt.batch_overlap import two_batch_overlap
        from sglang.srt.layers.moe.token_dispatcher import deepep

        owner, context, submitted, completed = self._owner()
        owner.execution_mode = PrefillOnlyExecution(use_deepep=True)
        owner.device = torch.device("cpu")
        owner.local_experts, owner.execution_slots = 4, 8
        owner.format = SimpleNamespace(allocate_bank=Mock(), expert_nbytes=256)
        owner.host_pool = SimpleNamespace(used_bytes=3072)
        layers = owner.prefill.pipeline.layer_ids
        owner.catalog = SimpleNamespace(
            offloaded_layers=[SimpleNamespace(layer_id=layer) for layer in layers]
        )
        owner.storages = {}
        comm = _Stream()

        def submit_comm(layer, phase):
            tail = context.compute_stream.tail
            comm.enqueue(
                lambda: completed.append((layer, phase)),
                () if tail is None else (tail,),
            )
            return SimpleNamespace(
                current_stream_wait=lambda: context.compute_stream.wait_stream(comm)
            )

        def dispatch(layer, hidden, ids, weights, previous_event):
            self.assertTrue(owner.prefill.pipeline.started)
            return hidden, ids, weights, [0] * 4, submit_comm(layer, "dispatch")

        def combine(layer, output, previous_event):
            return output, submit_comm(layer, "combine")

        with (
            patch.object(
                torch,
                "npu",
                SimpleNamespace(Stream=Mock(return_value=owner.weight_copy_stream)),
                create=True,
            ),
            patch(
                "sglang.srt.offloading.moe.npu.prefill.AscendPrefillWeights",
                return_value=owner.prefill,
            ),
            patch.object(
                two_batch_overlap,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_deepep=lambda: True),
            ),
            patch.object(two_batch_overlap, "is_tbo_enabled", return_value=False),
            patch.object(deepep, "get_is_extend_in_batch", return_value=True),
            patch.object(deepep, "_is_npu", True),
            patch.object(deepep, "use_deepep", True),
            patch.object(
                deepep, "Buffer", SimpleNamespace(capture=lambda: None), create=True
            ),
            patch.object(
                deepep._DeepEPDispatcherImplNormal, "set_deepep_dispatcher_dtype"
            ),
        ):
            for layer in layers:
                dispatcher = two_batch_overlap.MaybeTboDeepEPDispatcher(
                    group=object(),
                    router_topk=2,
                    num_experts=16,
                    num_local_experts=4,
                    hidden_size=4,
                    deepep_mode=deepep.DeepEPMode.NORMAL,
                    async_finish=True,
                )
                normal = dispatcher._inners[0]._normal_dispatcher
                normal._dispatch_core = partial(dispatch, layer)
                normal._combine_core = partial(combine, layer)
                owner.storages[layer] = SimpleNamespace(
                    require_sealed=Mock(),
                    layer=SimpleNamespace(layer_id=layer, dispatcher=dispatcher),
                )
            owner.post_init()
            topk = SimpleNamespace(
                topk_ids=torch.empty((0, 2), dtype=torch.int32),
                topk_weights=torch.empty((0, 2)),
            )
            for chunk in range(2):
                completed.clear()
                batch = self._batch(local_idle=True, local_extend=False)
                with owner.forward_scope(batch):
                    for layer in layers:
                        dispatcher = owner.storages[layer].layer.dispatcher
                        received = dispatcher.dispatch(
                            hidden_states=torch.empty((0, 4), dtype=torch.bfloat16),
                            topk_output=topk,
                        )
                        owner.prefill.pipeline.acquire(layer, context)
                        owner.prefill.pipeline.wait_for_w2(layer, context)
                        dispatcher.combine(
                            combine_input=deepep.DeepEPNormalCombineInput(
                                received.hidden_states,
                                received.topk_ids,
                                received.topk_weights,
                            )
                        )
                        if layer == layers[0]:
                            self.assertNotIn((layer, "combine"), completed)
                            # Drain the copy stream first to check that the refill
                            # cannot complete before the queued combine.
                            owner.weight_copy_stream.tail.run()
                            self.assertLess(
                                completed.index((layer, "combine")),
                                completed.index((layers[2], "w13")),
                            )
                self.assertEqual(submitted, list(layers) * (chunk + 1))
                self.assertEqual(context.resources, [])


class TestDeepEPNormalOffloadInputs(CustomTestCase):
    def test_dummy_receive_row_bypasses_gmm_and_preserves_combine_shape(self):
        from sglang.srt.layers.moe.moe_runner import ascend
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPNormalDispatchOutput,
        )

        owner = SimpleNamespace(prepare_weights=Mock())
        config = SimpleNamespace(
            num_local_experts=4,
            layer=SimpleNamespace(layer_id=62, _ascend_ops_offloader=owner),
        )
        core = object.__new__(ascend.AscendRunnerCore)
        core.config = config
        for rows in (0, 1):
            with self.subTest(rows=rows):
                state = {}
                dispatch = DeepEPNormalDispatchOutput(
                    torch.full((rows, 4), 42, dtype=torch.int8),
                    torch.empty(rows),
                    None,
                    None,
                    [0, 0, 0, 0],
                )
                inputs = ascend.pre_permute_deepep_normal_to_ascend(
                    dispatch, None, config, state
                )
                with patch.object(ascend, "get_is_extend_in_batch", return_value=True):
                    output = core.run(inputs, None, state)
                combine = ascend.post_permute_ascend_to_deepep_normal(
                    output, None, config, state
                )
                torch.testing.assert_close(
                    combine.hidden_states, torch.zeros((rows, 4), dtype=torch.bfloat16)
                )

    def test_grouped_int8_input_and_metadata_validation(self):
        from sglang.srt.layers.moe.moe_runner.ascend import (
            pre_permute_deepep_normal_to_ascend,
        )
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPNormalDispatchOutput,
        )

        config = SimpleNamespace(
            num_local_experts=4,
            layer=SimpleNamespace(_ascend_ops_offloader=object()),
        )
        dispatch = DeepEPNormalDispatchOutput(
            torch.zeros((3, 4), dtype=torch.int8),
            torch.ones(3),
            None,
            None,
            [2, 0, 1, 0],
        )
        inputs = pre_permute_deepep_normal_to_ascend(dispatch, None, config, {})
        self.assertEqual(inputs.group_list_type, 1)
        self.assertEqual(inputs.expert_tokens.tolist(), [2, 0, 1, 0])
        self.assertIs(inputs.hidden_states_scale, dispatch.hidden_states_scale)
        for bad_dispatch, message in (
            (
                dispatch._replace(num_recv_tokens_per_expert=[2, 2, 3, 3]),
                "expert tokens",
            ),
            (dispatch._replace(num_recv_tokens_per_expert=[3]), "local expert"),
            (dispatch._replace(hidden_states_scale=None), "one scale per row"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                pre_permute_deepep_normal_to_ascend(bad_dispatch, None, config, {})


class TestPrefillOnlyConfiguration(CustomTestCase):
    def test_attention_tp4_dp4_is_supported_without_a_decode_dp_ep_requirement(self):
        from sglang.srt.offloading.moe.npu import execution_mode

        mode, owner = PrefillOnlyExecution(), SimpleNamespace()
        parallel = SimpleNamespace(tp_size=16, attn_dp_size=4, attn_cp_size=1)
        graphs = SimpleNamespace(
            prefill=SimpleNamespace(backend="disabled"),
            decode=SimpleNamespace(backend="disabled"),
        )
        with (
            patch.object(execution_mode, "get_parallel", return_value=parallel),
            patch.object(
                execution_mode,
                "get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(cuda_graph_config=graphs)
                ),
            ),
            patch.object(
                execution_mode.envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER,
                "get",
                return_value=False,
            ),
        ):
            mode.configure(owner, "unused.json")
        self.assertEqual(owner.hot_slots, 0)
        self.assertEqual(mode.default_execution_slots(16), 32)
        self.assertFalse(hasattr(owner, "policy"))

    def test_graphs_and_unsynchronized_dp_fail_before_allocation(self):
        from sglang.srt.offloading.moe.npu import execution_mode

        for cp, dp, prefill_graph, decode_graph, skip_sync in (
            (1, 3, "disabled", "disabled", False),
            (1, 4, "piecewise", "disabled", False),
            (1, 4, "disabled", "native", False),
            (1, 4, "disabled", "disabled", True),
        ):
            with self.subTest(
                cp=cp,
                dp=dp,
                prefill=prefill_graph,
                decode=decode_graph,
                skip_sync=skip_sync,
            ):
                parallel = SimpleNamespace(tp_size=16, attn_dp_size=dp, attn_cp_size=cp)
                graphs = SimpleNamespace(
                    prefill=SimpleNamespace(backend=prefill_graph),
                    decode=SimpleNamespace(backend=decode_graph),
                )
                with (
                    patch.object(execution_mode, "get_parallel", return_value=parallel),
                    patch.object(
                        execution_mode,
                        "get_exec",
                        return_value=SimpleNamespace(
                            graph=SimpleNamespace(cuda_graph_config=graphs)
                        ),
                    ),
                    patch.object(
                        execution_mode.envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER,
                        "get",
                        return_value=skip_sync,
                    ),
                    self.assertRaises(ValueError),
                ):
                    PrefillOnlyExecution().configure(SimpleNamespace(), "unused.json")


class TestOffloadModeWiring(CustomTestCase):
    @staticmethod
    def _construct(
        values,
        disaggregation_mode,
        *,
        parallel_changes=None,
        model_changes=None,
        moe_changes=None,
        speculative_algorithm=None,
    ):
        from sglang.srt.offloading.moe.npu import execution_mode

        config = SimpleNamespace(
            model_type="glm_moe_dsa",
            architectures=["GlmMoeDsaForCausalLM"],
            num_hidden_layers=78,
            first_k_dense_replace=3,
            moe_layer_freq=1,
            n_routed_experts=256,
        )
        parallel = SimpleNamespace(
            tp_size=16,
            moe_ep_size=16,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_dp_size=1,
            pp_size=1,
            pp_rank=0,
            nnodes=1,
            dp_size=1,
            attn_dp_size=4,
            attn_cp_size=1,
            attn_tp_size=4,
            enable_prefill_cp=False,
            disable_attn_tp_gather=False,
            cp_strategy=None,
            enable_cp_tp_group_sharing=False,
            enable_dsa_cache_layer_split=False,
            enable_kv_cache_sharding=False,
            cp_kv_layout=None,
        )
        parallel.__dict__.update(parallel_changes or {})
        config.__dict__.update(model_changes or {})
        execution = SimpleNamespace(
            moe=SimpleNamespace(
                moe_a2a_backend="ascend_ops",
                moe_ascend_ops_config=None,
                moe_runner_backend="auto",
                deepep_mode="normal",
                enable_eplb=False,
                ep_num_redundant_experts=0,
                init_expert_location="trivial",
            ),
            overlap=SimpleNamespace(
                enable_two_batch_overlap=False,
                enable_single_batch_overlap=False,
            ),
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(backend="disabled"),
                    decode=SimpleNamespace(backend="disabled"),
                )
            ),
        )
        execution.moe.__dict__.update(moe_changes or {})
        cpu = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "offload.json"
            path.write_text(json.dumps(values))
            (directory / "profile.json").write_text(
                json.dumps(
                    {
                        "model_id": "GLM-5.2",
                        "layers": {},
                    }
                )
            )
            with (
                patch.object(ascend_ops, "get_parallel", return_value=parallel),
                patch.object(execution_mode, "get_parallel", return_value=parallel),
                patch.object(ascend_ops, "get_exec", return_value=execution),
                patch.object(execution_mode, "get_exec", return_value=execution),
                patch.object(
                    execution_mode,
                    "get_spec",
                    return_value=SimpleNamespace(
                        speculative_algorithm=speculative_algorithm
                    ),
                ),
                patch.object(
                    ascend_ops, "get_device", return_value=SimpleNamespace(gpu_id=0)
                ),
                patch.object(
                    ascend_ops,
                    "get_disagg",
                    return_value=SimpleNamespace(
                        disaggregation_mode=disaggregation_mode
                    ),
                ),
                patch.object(
                    ascend_ops,
                    "get_schedule",
                    return_value=SimpleNamespace(disable_overlap_schedule=True),
                ),
                patch.object(
                    execution_mode.envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER,
                    "get",
                    return_value=False,
                ),
                patch.object(torch, "device", return_value=cpu),
            ):
                return ascend_ops.AscendOpsOffloader(config, path)

    def test_prefill_configuration_selects_all_routed_layers_and_two_banks(self):
        owner = self._construct(
            {"mode": "prefill_only", "host_pool_gib": 48}, "prefill"
        )
        self.assertIsInstance(owner.execution_mode, PrefillOnlyExecution)
        self.assertEqual(len(owner.catalog.offloaded_layers), 75)
        self.assertEqual(owner.catalog.offloaded_layers[0].layer_id, 3)
        self.assertEqual(owner.catalog.num_experts, 1200)
        self.assertEqual(owner.local_experts, 16)
        self.assertEqual(owner.execution_slots, 32)
        self.assertEqual(owner.hot_slots, 0)
        self.assertFalse(hasattr(owner, "policy"))
        self.assertIsNone(owner.format)
        self.assertIsNone(owner.host_pool)
        self.assertFalse(owner.ready)

    def test_deepep_normal_uses_the_same_stage_catalog_and_banks(self):
        from sglang.srt.offloading.moe.npu.model_adapter import ExpertQuantSpec

        with patch.dict(
            "os.environ",
            {
                "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
                "MOE_EXPERT_TOKEN_NUMS_TYPE": "1",
                "SGLANG_ZBAL_LOCAL_MEM_SIZE": "0",
            },
        ):
            owner = self._construct(
                {"mode": "prefill_only"},
                "prefill",
                parallel_changes=self._pipeline_parallel(pp_rank=3),
                moe_changes={"moe_a2a_backend": "deepep"},
            )
        self.assertEqual(owner.layer_range, (62, 78))
        self.assertEqual(owner.local_experts, 64)
        self.assertEqual(owner.execution_slots, 128)
        self.assertEqual(owner.hot_slots, 0)
        owner.execution_mode.validate_quantization(owner, ExpertQuantSpec("w8a8"))
        with self.assertRaisesRegex(ValueError, "W8A8"):
            owner.execution_mode.validate_quantization(owner, ExpertQuantSpec("w4a8"))

    def test_partial_residency_uses_stage_budget_for_both_dispatchers(self):
        values = {
            "mode": "prefill_only",
            "prefill_resident_ratio": 0.25,
            "execution_slots": 480,
        }
        with patch.dict(
            "os.environ",
            {
                "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
                "MOE_EXPERT_TOKEN_NUMS_TYPE": "1",
                "SGLANG_ZBAL_LOCAL_MEM_SIZE": "0",
            },
        ):
            for backend in ("ascend_ops", "deepep"):
                with self.subTest(backend=backend):
                    owner = self._construct(
                        values,
                        "prefill",
                        parallel_changes=self._pipeline_parallel(pp_rank=3),
                        moe_changes={"moe_a2a_backend": backend},
                    )
                    self.assertEqual(owner.execution_mode.resident_experts, 16)
                    self.assertEqual(owner.execution_slots, 352)
                    self.assertIsNone(owner.host_pool)
            with self.assertRaisesRegex(ValueError, "at least 480"):
                self._construct(
                    {**values, "execution_slots": 479},
                    "prefill",
                    parallel_changes=self._pipeline_parallel(pp_rank=2),
                )
            with self.assertRaisesRegex(ValueError, "requires Ascend Ops alltoallv"):
                self._construct(
                    values,
                    "prefill",
                    parallel_changes=self._pipeline_parallel(pp_rank=3),
                    moe_changes={"moe_ascend_ops_config": '{"prefill":"allgather"}'},
                )

    def test_deepep_rejects_incompatible_modes_and_counts(self):
        for mode in ("auto", "low_latency"):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, "mode normal"),
            ):
                self._construct(
                    {"mode": "prefill_only"},
                    "prefill",
                    moe_changes={"moe_a2a_backend": "deepep", "deepep_mode": mode},
                )
        with self.assertRaisesRegex(ValueError, "mode=prefill_only"):
            self._construct({}, "null", moe_changes={"moe_a2a_backend": "deepep"})
        with (
            patch.dict(
                "os.environ",
                {"MOE_EXPERT_TOKEN_NUMS_TYPE": "0", "SGLANG_ZBAL_LOCAL_MEM_SIZE": "0"},
            ),
            self.assertRaisesRegex(ValueError, "MOE_EXPERT_TOKEN_NUMS_TYPE=1"),
        ):
            self._construct(
                {"mode": "prefill_only"},
                "prefill",
                moe_changes={"moe_a2a_backend": "deepep"},
            )

    def test_existing_mixed_configuration_keeps_resident_prefix_and_lfu(self):
        owner = self._construct({"offline_stats_path": "profile.json"}, "null")
        self.assertIsInstance(owner.execution_mode, CachedExecution)
        self.assertTrue(owner.catalog.layer(3).resident)
        self.assertEqual(len(owner.catalog.offloaded_layers), 74)
        self.assertEqual(owner.catalog.offloaded_layers[0].layer_id, 4)
        self.assertEqual(owner.catalog.num_experts, 1184)
        self.assertEqual(owner.execution_slots, 16)
        self.assertEqual(owner.hot_slots, 296)
        self.assertEqual(len(owner.initial_keys), 296)
        self.assertEqual(owner.policy.config.update_interval, 4)
        self.assertFalse(owner.ready)

    @staticmethod
    def _pipeline_parallel(**changes):
        values = dict(
            tp_size=4,
            moe_ep_size=4,
            pp_size=4,
            attn_dp_size=1,
            attn_cp_size=4,
            attn_tp_size=1,
            enable_prefill_cp=True,
            cp_strategy="zigzag",
        )
        values.update(changes)
        return values

    def test_stage_catalogs_match_model_partition_before_device_allocation(self):
        from sglang.srt.offloading.moe.npu import execution_mode

        for cp in (1, 4):
            for rank, span in enumerate(((0, 18), (18, 38), (38, 62), (62, 78))):
                with (
                    self.subTest(cp=cp, rank=rank),
                    patch.dict(
                        "os.environ", {"SGLANG_PP_LAYER_PARTITION": "18,20,24,16"}
                    ),
                ):
                    owner = self._construct(
                        {"mode": "prefill_only"},
                        "prefill",
                        parallel_changes=self._pipeline_parallel(
                            pp_rank=rank,
                            attn_cp_size=cp,
                            attn_tp_size=4 // cp,
                            enable_prefill_cp=cp > 1,
                        ),
                    )
                    self.assertEqual(owner.layer_range, span)
                    self.assertEqual(
                        [s.layer_id for s in owner.catalog.offloaded_layers],
                        list(range(max(3, span[0]), span[1])),
                    )
                    self.assertEqual(owner.local_experts, 64)
                    self.assertEqual(owner.execution_slots, 128)
                    self.assertEqual(owner.hot_slots, 0)
                    self.assertIsNone(owner.host_pool)
                    self.assertIsNone(owner.format)
                    self.assertIsInstance(
                        owner.execution_mode, execution_mode.PrefillOnlyExecution
                    )

    def test_unsupported_pipeline_topologies_fail_before_native_resources(self):
        cases = (
            dict(pp_size=2),
            dict(tp_size=8, moe_ep_size=8),
            dict(moe_ep_size=16),
            dict(moe_dp_size=2),
            dict(nnodes=2),
            dict(dp_size=2),
            dict(attn_dp_size=2),
            dict(attn_cp_size=2),
            dict(attn_tp_size=2),
            dict(cp_strategy="interleave"),
            dict(enable_prefill_cp=False),
            dict(disable_attn_tp_gather=True),
            dict(enable_cp_tp_group_sharing=True),
            dict(enable_dsa_cache_layer_split=True),
            dict(enable_kv_cache_sharding=True),
            dict(cp_kv_layout="sharded"),
            dict(pp_rank=4),
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._construct(
                    {"mode": "prefill_only"},
                    "prefill",
                    parallel_changes=self._pipeline_parallel(**changes),
                )
        for model, algorithm in (("deepseek_v4", None), ("glm_moe_dsa", "EAGLE")):
            with (
                self.subTest(model=model, algorithm=algorithm),
                self.assertRaises(ValueError),
            ):
                self._construct(
                    {"mode": "prefill_only"},
                    "prefill",
                    parallel_changes=self._pipeline_parallel(),
                    model_changes=dict(model_type=model),
                    speculative_algorithm=algorithm,
                )

    def test_mixed_mode_still_rejects_pipeline_and_cp(self):
        for changes in (self._pipeline_parallel(), dict(attn_cp_size=4)):
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, "Mixed"),
            ):
                self._construct({}, "null", parallel_changes=changes)

    def test_bad_or_dense_only_partition_is_rejected_before_host_allocation(self):
        for partition in ("18,-1,45,16", "18,20,24", "0,38,24,16", "2,20,24,32"):
            with (
                self.subTest(partition=partition),
                patch.dict("os.environ", {"SGLANG_PP_LAYER_PARTITION": partition}),
                self.assertRaises(ValueError),
            ):
                self._construct(
                    {"mode": "prefill_only"},
                    "prefill",
                    parallel_changes=self._pipeline_parallel(),
                )

    def test_pipelined_prefill_rejects_w4_before_format_allocation(self):
        from sglang.srt.offloading.moe.npu import execution_mode
        from sglang.srt.offloading.moe.npu.model_adapter import ExpertQuantSpec

        owner = SimpleNamespace()
        with patch.object(
            execution_mode, "get_parallel", return_value=SimpleNamespace(pp_size=4)
        ):
            with self.assertRaisesRegex(ValueError, "W8A8"):
                PrefillOnlyExecution().validate_quantization(
                    owner, ExpertQuantSpec("w4a8")
                )
            PrefillOnlyExecution().validate_quantization(owner, ExpertQuantSpec("w8a8"))


class TestPrefillOnlyInitialization(CustomTestCase):
    @staticmethod
    def _owner():
        owner = object.__new__(ascend_ops.AscendOpsOffloader)
        owner.execution_mode = PrefillOnlyExecution()
        owner.device, owner.model_id = torch.device("cpu"), "GLM-5.2"
        owner.local_experts = owner.copy_capacity = 4
        owner.execution_slots, owner.hot_slots = 8, 0
        owner.catalog = SimpleNamespace(
            num_experts=12,
            offloaded_layers=tuple(SimpleNamespace(layer_id=i) for i in (0, 2, 4)),
        )
        owner.storages = {i: SimpleNamespace(require_sealed=Mock()) for i in (0, 2, 4)}
        owner.format = SimpleNamespace(
            allocate_bank=Mock(return_value=object()), expert_nbytes=256
        )
        owner.host_pool = SimpleNamespace(used_bytes=3072)
        owner._capture_cache_tails = Mock(
            side_effect=AssertionError("cache graphs on P")
        )
        owner._init_copy_plan = Mock(side_effect=AssertionError("decode copies on P"))
        owner._new_context = Mock(return_value=_Context())
        return owner

    def test_only_execution_bank_is_allocated_without_decode_or_cache_state(self):
        owner = self._owner()
        with (
            patch.object(torch, "npu", SimpleNamespace(Stream=Mock()), create=True),
            patch(
                "sglang.srt.offloading.moe.npu.prefill.AscendPrefillWeights"
            ) as weights,
        ):
            owner.post_init()
        owner.format.allocate_bank.assert_called_once_with(
            8, ResourceKind.EXECUTION_WEIGHTS, "ascend.execution"
        )
        for storage in owner.storages.values():
            storage.require_sealed.assert_called_once()
        weights.assert_called_once_with(owner, resident_experts=0)
        self.assertTrue(owner.ready)
        self.assertIsNone(owner.hot)
        self.assertIsNone(owner.state)
        self.assertIsNone(owner.decode_active_rows)
        self.assertIsNone(owner.decode_row_mask(128))
        self.assertFalse(owner.direct_hot)
        owner._capture_cache_tails.assert_not_called()
        owner._init_copy_plan.assert_not_called()
        for name in ("addresses", "policy_state", "cache_update", "cache_tail_graphs"):
            self.assertFalse(hasattr(owner, name), name)

    def test_resident_bytes_survive_cold_bank_reuse_across_chunks(self):
        from sglang.srt.offloading.moe.npu.format import ExpertBank

        owner = self._owner()
        owner.execution_mode.resident_experts = 1
        owner.execution_slots = 9
        layers = tuple(owner.storages)
        names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
        owner.format.layouts = {name: SimpleNamespace(nbytes=1) for name in names}
        owner.format.adapter = SimpleNamespace(native_acl=object())
        owner.format.quant_info = lambda tensors: tensors
        tensors = {name: torch.zeros((9, 1), dtype=torch.uint8) for name in names}
        owner.format.allocate_bank.return_value = ExpertBank(
            owner.format, tensors, 9, ResourceKind.EXECUTION_WEIGHTS, "test"
        )
        for index, layer in enumerate(layers):
            owner.storages[layer].component_capabilities = {
                name: tuple(
                    SimpleNamespace(nbytes=1, value=32 * component + 4 * index + e + 1)
                    for e in range(4)
                )
                for component, name in enumerate(names)
            }

        copy_stream = _Stream()
        copy_stream.synchronize = lambda: _Completion(copy_stream.tail).synchronize()
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
        with (
            patch.object(
                torch,
                "npu",
                SimpleNamespace(Stream=lambda **kw: copy_stream),
                create=True,
            ),
            patch(
                "sglang.srt.offloading.moe.npu.native_acl.AclHostTransferOps",
                return_value=host_ops,
            ),
        ):
            owner.post_init()

        self.assertEqual(host_ops.materialize_into.call_count, 2 * len(layers))
        self.assertEqual(owner.context.resources, [])
        self.assertEqual([bank.capacity for bank in owner.prefill.banks], [3, 3])
        owner._new_context = _Context
        owner._check_error = Mock()
        for _ in range(2):
            with owner.forward_scope(TestPrefillOnlyRuntime._batch()):
                for index, layer in enumerate(layers):
                    owner.start_prefill_layer(layer)
                    bank = owner.prefill.pipeline.acquire(layer, owner.context)
                    owner.prefill.pipeline.wait_for_w2(layer, owner.context)
                    _Completion(owner.context.compute_stream.tail).synchronize()
                    cold_start = len(layers) + bank * 3
                    for name in names:
                        expected = [
                            source.value
                            for source in owner.storages[layer].component_capabilities[
                                name
                            ]
                        ]
                        self.assertEqual(tensors[name][index, 0].item(), expected[0])
                        self.assertEqual(
                            tensors[name][cold_start : cold_start + 3, 0].tolist(),
                            expected[1:],
                        )
                    owner.prefill.finish_layer(layer)
        self.assertEqual(host_ops.materialize_into.call_count, 6 * len(layers))

    def test_unsealed_or_missing_layers_fail_before_device_allocation(self):
        for unsealed in (False, True):
            with self.subTest(unsealed=unsealed):
                owner = self._owner()
                if unsealed:
                    owner.storages[2].require_sealed.side_effect = RuntimeError(
                        "unsealed"
                    )
                else:
                    del owner.storages[2]
                with self.assertRaises(RuntimeError):
                    owner.post_init()
                owner.format.allocate_bank.assert_not_called()


if __name__ == "__main__":
    unittest.main()
