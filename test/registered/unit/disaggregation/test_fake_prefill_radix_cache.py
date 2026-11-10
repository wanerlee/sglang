"""Fake prefill may publish local KV only when explicitly enabled."""

import os
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestFakePrefillRadixCache(unittest.TestCase):
    @staticmethod
    def make_req(
        *, host=FAKE_BOOTSTRAP_HOST, mode=DisaggregationMode.PREFILL, overrides=None
    ):
        return Req(
            rid="fake-prefill-cache",
            origin_input_text="",
            origin_input_ids=array("q", range(512)),
            sampling_params=SamplingParams(max_new_tokens=1),
            bootstrap_host=host,
            bootstrap_room=0,
            disagg_mode=mode,
            positional_embed_overrides=overrides,
        )

    def test_opt_in_is_disabled_when_not_set(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE", None)
            self.assertFalse(envs.SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE.get())
            self.assertTrue(self.make_req().skip_radix_cache_insert)

    def test_opt_in_is_limited_to_fake_prefill_without_embedding_overrides(self):
        for enabled, host, mode, overrides, skip in (
            (False, FAKE_BOOTSTRAP_HOST, DisaggregationMode.PREFILL, None, True),
            (True, FAKE_BOOTSTRAP_HOST, DisaggregationMode.PREFILL, None, False),
            (True, FAKE_BOOTSTRAP_HOST, DisaggregationMode.DECODE, None, True),
            (True, FAKE_BOOTSTRAP_HOST, DisaggregationMode.NULL, None, True),
            (True, FAKE_BOOTSTRAP_HOST, None, None, True),
            (True, "192.0.2.1", DisaggregationMode.PREFILL, None, False),
            (False, "192.0.2.1", DisaggregationMode.PREFILL, None, False),
            (True, FAKE_BOOTSTRAP_HOST, DisaggregationMode.PREFILL, object(), True),
            (True, "192.0.2.1", DisaggregationMode.PREFILL, object(), True),
        ):
            with self.subTest(
                enabled=enabled, host=host, mode=mode, overrides=overrides
            ):
                with envs.SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE.override(enabled):
                    req = self.make_req(host=host, mode=mode, overrides=overrides)
                self.assertEqual(req.skip_radix_cache_insert, skip)

    def test_checkpoint_and_release_publish_only_when_enabled(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                with envs.SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE.override(enabled):
                    req = self.make_req()
                req.extend_end = 384
                req.kv = SimpleNamespace(
                    holds_kv=True,
                    is_kv_released=False,
                    cache_protected_len=0,
                    kv_allocated_len=512,
                    mark_kv_released=Mock(),
                )
                req.owned_kv_len = Mock(return_value=512)
                req.refresh_fill_ids = Mock()
                tree = SimpleNamespace(
                    claim_kv_row=Mock(return_value=False),
                    supports_mamba=Mock(return_value=False),
                    checkpoint=Mock(),
                    free_kv_row=Mock(),
                    unlock=Mock(),
                    on_release=Mock(),
                    req_to_token_pool=object(),
                    token_to_kv_pool_allocator=object(),
                )
                row_store = SimpleNamespace(release_row=Mock())
                checkpoint_kv_cache(req, tree)
                self.assertEqual(req.prefix_len, 384)
                with (
                    patch(
                        "sglang.srt.mem_cache.common.kv_storage_of",
                        return_value=row_store,
                    ),
                    patch(
                        "sglang.srt.mem_cache.common._release_overallocated_kv_indices"
                    ),
                ):
                    release_kv_cache(req, tree, checkpoint=True)
                expected = [384, 512] if enabled else []
                self.assertEqual(
                    [call.kwargs["up_to"] for call in tree.checkpoint.call_args_list],
                    expected,
                )
                tree.on_release.assert_called_once_with(req, checkpointed=enabled)
                row_store.release_row.assert_called_once_with(req)
                req.kv.mark_kv_released.assert_called_once()

    def test_abort_release_without_checkpoint_never_publishes(self):
        with envs.SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE.override(True):
            req = self.make_req()
        req.kv = SimpleNamespace(
            holds_kv=True,
            is_kv_released=False,
            cache_protected_len=0,
            kv_allocated_len=512,
            mark_kv_released=Mock(),
        )
        req.owned_kv_len = Mock(return_value=512)
        tree = SimpleNamespace(
            claim_kv_row=Mock(return_value=False),
            checkpoint=Mock(),
            free_kv_row=Mock(),
            unlock=Mock(),
            on_release=Mock(),
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=object(),
        )
        with (
            patch(
                "sglang.srt.mem_cache.common.kv_storage_of",
                return_value=SimpleNamespace(release_row=Mock()),
            ),
            patch("sglang.srt.mem_cache.common._release_overallocated_kv_indices"),
        ):
            release_kv_cache(req, tree, checkpoint=False)
        tree.checkpoint.assert_not_called()
        tree.on_release.assert_called_once_with(req, checkpointed=False)

    def test_matching_a_later_request_finds_the_published_prefix(self):
        class Pool:
            def __init__(self):
                self.req_to_token = torch.arange(512, dtype=torch.int64).view(1, -1)

            def write(self, indices, values):
                self.req_to_token[indices] = values

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                pool = Pool()
                allocator = SimpleNamespace(
                    page_size=128,
                    device="cpu",
                    free_segment=Mock(),
                )
                tree = RadixCache(
                    CacheInitParams(
                        disable=False,
                        req_to_token_pool=pool,
                        token_to_kv_pool_allocator=allocator,
                        page_size=128,
                    )
                )
                with envs.SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE.override(enabled):
                    req = self.make_req()
                req.kv.req_pool_idx = 0
                req.full_untruncated_fill_ids = array("q", range(512))
                req.extend_end = 512
                checkpoint_kv_cache(req, tree)

                next_input = array("q", range(512)) + array("q", [1000] * 128)
                match = tree.match_prefix(MatchPrefixParams(key=RadixKey(next_input)))
                self.assertEqual(match.device_prefix_len, 512 if enabled else 0)
                tree.unlock(req.lock)


if __name__ == "__main__":
    unittest.main()
