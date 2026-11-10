"""Zigzag's logical MoE rows across physical CP padding; no NPU required."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.moe import topk as npu_topk
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import padding, utils, zigzag
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner import eager_runner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestZigzagMoeRows(CustomTestCase):
    @staticmethod
    def _batch(extend_lens, *, prefix_lens=None):
        tokens = sum(extend_lens)
        prefix_lens = prefix_lens or [0] * len(extend_lens)
        batch = SimpleNamespace(
            input_ids=torch.arange(tokens),
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=list(extend_lens),
            seq_lens_cpu=[p + n for p, n in zip(prefix_lens, extend_lens)],
            attn_cp_metadata=None,
            global_num_tokens_cpu=None,
            num_token_non_padded=torch.tensor(tokens, dtype=torch.int32),
            global_num_token_non_padded=torch.tensor(tokens, dtype=torch.int32),
        )
        return batch

    def _prepare(self, batch, rank):
        strategy = zigzag.ZigzagCPStrategy(4)
        with (
            patch.object(
                cp_base,
                "get_parallel",
                return_value=SimpleNamespace(attn_cp_rank=rank),
            ),
            patch.object(
                zigzag, "get_device", return_value=SimpleNamespace(device="cpu")
            ),
            patch.object(utils, "get_cp_strategy", return_value=strategy),
            patch.object(
                utils,
                "get_parallel",
                return_value=SimpleNamespace(enable_cp_tp_group_sharing=False),
            ),
            patch.object(padding, "get_cp_padding_align_size", return_value=8),
        ):
            utils.prepare_cp_forward(batch)
            local_rows = strategy.shard_hidden_states(batch.input_ids, batch)
            count = strategy.moe_num_token_non_padded(batch)
            # Different layers must not allocate a new device scalar.
            self.assertIs(count, strategy.moe_num_token_non_padded(batch))
        return strategy, local_rows, count

    def test_uneven_and_batched_prefix_inputs_keep_physical_payload(self):
        for extend_lens, prefix_lens, expected in (
            ([9], [0], [3, 2, 2, 2]),
            ([16], [0], [4, 4, 4, 4]),
            ([9, 10], [31, 200], [6, 5, 4, 4]),
        ):
            for rank, valid in enumerate(expected):
                with self.subTest(extend_lens=extend_lens, rank=rank):
                    batch = self._batch(extend_lens, prefix_lens=prefix_lens)
                    _, local, count = self._prepare(batch, rank)
                    metadata = batch.attn_cp_metadata
                    self.assertEqual(metadata.per_rank_logical_token, expected)
                    self.assertEqual(metadata.per_rank_actual_token, [8] * 4)
                    self.assertEqual(local.shape, (8,))
                    self.assertEqual(count.dtype, torch.int32)
                    self.assertEqual(count.shape, ())
                    self.assertEqual(count.item(), valid)
                    self.assertTrue(
                        torch.equal(
                            local[valid:], torch.zeros(8 - valid, dtype=local.dtype)
                        )
                    )

    def test_forward_batch_routes_local_count_and_topk_retains_ids(self):
        for rank, valid in enumerate([3, 2, 2, 2]):
            batch = self._batch([9])
            strategy, _, expected_count = self._prepare(batch, rank)
            # This is the real ForwardBatch callback exposed to MoE top-k.
            with (
                patch.object(
                    moe_utils,
                    "is_moe_input_scattered_across_dp_ranks",
                    return_value=True,
                ),
                patch.object(cp_base, "get_cp_strategy", return_value=strategy),
                patch.object(
                    cp_base,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_cp_rank=rank),
                ),
            ):
                count = ForwardBatch.moe_num_token_non_padded(batch)
            self.assertIs(count, expected_count)
            ids = torch.arange(16, dtype=torch.int32).reshape(8, 2)
            original_ids = ids.clone()
            weights = torch.full((8, 2), 0.5)
            recorder = Mock()
            with (
                patch.object(
                    npu_topk,
                    "topk_ids_logical_to_physical",
                    side_effect=lambda value, _: value,
                ),
                patch.object(npu_topk, "capture_routed_experts_if_allowed"),
                patch.object(
                    npu_topk,
                    "get_global_expert_distribution_recorder",
                    return_value=recorder,
                ),
            ):
                output = npu_topk._finalize_topk_npu(
                    topk_ids=ids,
                    topk_weights=weights,
                    router_logits=torch.zeros(8, 256),
                    topk_config=SimpleNamespace(num_shared_topk_slots=0),
                    num_token_non_padded=count,
                    expert_location_dispatch_info=None,
                    layer_id=18,
                )
            self.assertTrue(torch.equal(output.topk_ids, original_ids))
            self.assertEqual(output.topk_weights.shape, (8, 2))
            self.assertTrue(torch.all(output.topk_weights[:valid] == 0.5))
            self.assertTrue(torch.all(output.topk_weights[valid:] == 0))
            recorded = recorder.on_select_experts.call_args.kwargs["topk_ids"]
            self.assertTrue(torch.all(recorded[valid:] == -1))

    def test_next_chunk_uses_a_fresh_local_count(self):
        old = self._batch([9])
        _, _, old_count = self._prepare(old, 0)
        next_chunk = self._batch([17], prefix_lens=[9])
        _, _, new_count = self._prepare(next_chunk, 0)
        self.assertEqual(old_count.item(), 3)
        self.assertEqual(new_count.item(), 5)
        self.assertIsNot(new_count, old_count)

    def test_unpadded_metadata_uses_actual_local_rows(self):
        batch = self._batch([9])
        batch.attn_cp_metadata = zigzag.ZigzagContextParallelMetadata(
            per_rank_actual_token=[3, 2, 2, 2]
        )
        strategy = zigzag.ZigzagCPStrategy(4)
        with patch.object(
            cp_base, "get_parallel", return_value=SimpleNamespace(attn_cp_rank=1)
        ):
            self.assertEqual(strategy.moe_num_token_non_padded(batch).item(), 2)

    def test_short_input_eager_fallback_clears_old_cp_metadata(self):
        for extend_lens in ([7], [9, 7]):
            batch = self._batch(extend_lens)
            # Exercise the real eager entrypoint with deliberately stale metadata.
            batch.attn_cp_metadata = zigzag.ZigzagContextParallelMetadata(
                per_rank_actual_token=[8] * 4,
                per_rank_logical_token=[3, 2, 2, 2],
                moe_local_token_count=torch.tensor(2, dtype=torch.int32),
            )
            batch.positions = batch.input_ids
            batch.needs_forward_metadata_init = lambda: False
            strategy = zigzag.ZigzagCPStrategy(4)
            self.assertFalse(strategy.can_apply(len(batch.input_ids), batch))
            observed = []

            def model_forward(_, __, fb, **kwargs):
                observed.append(fb.attn_cp_metadata)
                return ForwardBatch.moe_num_token_non_padded(fb)

            runner = SimpleNamespace(
                enable_pdmux=False,
                load_batch=lambda fb, _: fb,
                model_runner=SimpleNamespace(
                    _extend_forward_kwargs=lambda *_: {},
                    device_timer=None,
                    model=SimpleNamespace(forward=model_forward),
                ),
            )
            with (
                patch.object(utils, "get_cp_strategy", return_value=strategy),
                patch.object(
                    moe_utils,
                    "is_moe_input_scattered_across_dp_ranks",
                    return_value=True,
                ),
                patch.object(
                    eager_runner,
                    "device_timer_ctx",
                    side_effect=lambda *_: nullcontext(),
                ),
                patch.object(eager_runner, "get_forward_context", return_value=None),
                patch.object(
                    eager_runner,
                    "forward_context",
                    side_effect=lambda *_args, **_kwargs: nullcontext(),
                ),
            ):
                count = eager_runner.EagerRunner._execute_extend(runner, batch)
            self.assertEqual(observed, [None])
            self.assertIs(count, batch.num_token_non_padded)
            self.assertEqual(count.item(), sum(extend_lens))


if __name__ == "__main__":
    unittest.main()
