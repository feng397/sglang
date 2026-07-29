import types
import unittest

from sglang.srt.models.deepseek_common.utils import (
    compute_dsv4_index_cache_descriptor,
    dsv4_index_cache_enabled,
    dsv4_index_cache_producer_layer_ids,
    validate_dsv4_index_cache_pd_compatibility,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hf_config(index_topk_freq=1, index_topk_pattern=None):
    # 43-layer DeepSeek-V4-Flash-like layout: 21 C4 layers interleaved with C128.
    compress_ratios = [0, 0] + [4, 128] * 20 + [4]
    return types.SimpleNamespace(
        num_hidden_layers=43,
        compress_ratios=compress_ratios,
        index_topk=512,
        index_head_dim=128,
        index_n_heads=64,
        index_topk_freq=index_topk_freq,
        index_topk_pattern=index_topk_pattern,
    )


class TestDSV4IndexCacheDescriptor(CustomTestCase):
    def test_layout_signature_ignores_freq(self):
        # layout_signature must depend only on layout fields, not freq/pattern,
        # so a P freq=1 + D freq=4 gray-rollout is not rejected on layout.
        sig_p, _ = compute_dsv4_index_cache_descriptor(
            _hf_config(index_topk_freq=1), fp4_indexer_enabled=True, page_size=64
        )
        sig_d, _ = compute_dsv4_index_cache_descriptor(
            _hf_config(index_topk_freq=4), fp4_indexer_enabled=True, page_size=64
        )
        self.assertEqual(sig_p, sig_d)

    def test_layout_signature_changes_with_page_size(self):
        sig_a, _ = compute_dsv4_index_cache_descriptor(
            _hf_config(), fp4_indexer_enabled=True, page_size=64
        )
        sig_b, _ = compute_dsv4_index_cache_descriptor(
            _hf_config(), fp4_indexer_enabled=True, page_size=128
        )
        self.assertNotEqual(sig_a, sig_b)

    def test_layout_signature_changes_with_fp4(self):
        sig_a, _ = compute_dsv4_index_cache_descriptor(
            _hf_config(), fp4_indexer_enabled=True, page_size=64
        )
        sig_b, _ = compute_dsv4_index_cache_descriptor(
            _hf_config(), fp4_indexer_enabled=False, page_size=64
        )
        self.assertNotEqual(sig_a, sig_b)

    def test_freq1_is_all_producers(self):
        cfg = _hf_config(index_topk_freq=1)
        producers = dsv4_index_cache_producer_layer_ids(cfg.compress_ratios, 1)
        self.assertEqual(len(producers), 21)
        self.assertFalse(dsv4_index_cache_enabled(cfg.compress_ratios, 1))

    def test_freq4_producer_subset(self):
        cfg = _hf_config(index_topk_freq=4)
        producers = dsv4_index_cache_producer_layer_ids(cfg.compress_ratios, 4)
        # 21 C4 layers, FSSS pattern -> 6 producers.
        self.assertEqual(len(producers), 6)
        self.assertTrue(dsv4_index_cache_enabled(cfg.compress_ratios, 4))

    def test_pd_p_freq1_d_freq4_is_safe(self):
        sig_p, prod_p = compute_dsv4_index_cache_descriptor(
            _hf_config(index_topk_freq=1), fp4_indexer_enabled=True, page_size=64
        )
        sig_d, prod_d = compute_dsv4_index_cache_descriptor(
            _hf_config(index_topk_freq=4), fp4_indexer_enabled=True, page_size=64
        )
        # P is the superset of producers; D requires a subset. No raise.
        validate_dsv4_index_cache_pd_compatibility(
            prefill_layout_signature=sig_p,
            prefill_producer_layer_ids=prod_p,
            decode_layout_signature=sig_d,
            decode_producer_layer_ids=prod_d,
        )

    def test_pd_p_freq4_d_freq1_is_rejected(self):
        sig_p, prod_p = compute_dsv4_index_cache_descriptor(
            _hf_config(index_topk_freq=4), fp4_indexer_enabled=True, page_size=64
        )
        sig_d, prod_d = compute_dsv4_index_cache_descriptor(
            _hf_config(index_topk_freq=1), fp4_indexer_enabled=True, page_size=64
        )
        with self.assertRaisesRegex(RuntimeError, "producer coverage"):
            validate_dsv4_index_cache_pd_compatibility(
                prefill_layout_signature=sig_p,
                prefill_producer_layer_ids=prod_p,
                decode_layout_signature=sig_d,
                decode_producer_layer_ids=prod_d,
            )

    def test_pd_layout_mismatch_is_rejected(self):
        sig_p, prod_p = compute_dsv4_index_cache_descriptor(
            _hf_config(), fp4_indexer_enabled=True, page_size=128
        )
        sig_d, prod_d = compute_dsv4_index_cache_descriptor(
            _hf_config(), fp4_indexer_enabled=True, page_size=64
        )
        with self.assertRaisesRegex(RuntimeError, "layout mismatch"):
            validate_dsv4_index_cache_pd_compatibility(
                prefill_layout_signature=sig_p,
                prefill_producer_layer_ids=prod_p,
                decode_layout_signature=sig_d,
                decode_producer_layer_ids=prod_d,
            )


if __name__ == "__main__":
    unittest.main()
