import unittest

from sglang.srt.models.deepseek_common.utils import (
    compute_dsv4_index_topk_flags,
    get_dsv4_c4_layer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV4IndexCacheUtils(CustomTestCase):
    COMPRESS_RATIOS = [0, 0, 4, 128, 4, 128, 4, 128, 4, 0]

    def test_get_c4_layer_ids(self):
        self.assertEqual(get_dsv4_c4_layer_ids(self.COMPRESS_RATIOS), [2, 4, 6, 8])

    def test_index_topk_freq(self):
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 2, 4),
            (False, True),
        )
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 4, 4),
            (True, True),
        )
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 6, 4),
            (True, True),
        )
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 8, 4),
            (True, False),
        )

    def test_index_topk_pattern_for_c4_layers(self):
        pattern = "FSSF"

        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 2, 1, pattern),
            (False, True),
        )
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 4, 1, pattern),
            (True, True),
        )
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 6, 1, pattern),
            (True, False),
        )
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 8, 1, pattern),
            (False, False),
        )

    def test_index_topk_pattern_for_all_layers(self):
        pattern = ["F"] * len(self.COMPRESS_RATIOS)
        pattern[4] = "S"
        pattern[6] = "S"

        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 2, 1, pattern),
            (False, True),
        )
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 4, 1, pattern),
            (True, True),
        )
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 6, 1, pattern),
            (True, False),
        )
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 8, 1, pattern),
            (False, False),
        )

    def test_invalid_index_topk_freq(self):
        with self.assertRaisesRegex(ValueError, "index_topk_freq must be positive"):
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 2, 0)

    def test_none_index_topk_freq_defaults_to_one(self):
        # A JSON override of index_topk_freq to null (None) must behave like
        # freq=1 (all producers), not crash on `None <= 0`.
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 2, None),
            (False, False),
        )
        self.assertEqual(
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 8, None),
            (False, False),
        )

    def test_negative_index_topk_freq_rejected(self):
        with self.assertRaisesRegex(ValueError, "index_topk_freq must be positive"):
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 2, -2)

    def test_invalid_index_topk_pattern_value(self):
        with self.assertRaisesRegex(ValueError, "only supports 'F'.*'S'"):
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 2, 1, "FXSF")

    def test_invalid_index_topk_pattern_length(self):
        with self.assertRaisesRegex(ValueError, "length must either match"):
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 2, 1, "FS")

    def test_first_c4_layer_must_be_full_c4_only_pattern(self):
        # C4-only pattern (len == num C4 layers) whose first C4 layer is 'S'.
        with self.assertRaisesRegex(ValueError, "first C4 layer must be 'F'"):
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 2, 1, "SFFF")

    def test_first_c4_layer_must_be_full_full_layer_pattern(self):
        # Full-layer pattern (len == num_hidden_layers) whose first C4 layer
        # (compress_ratio==4 at index 2) is 'S'.
        pattern = ["F"] * len(self.COMPRESS_RATIOS)
        pattern[2] = "S"  # first C4 layer
        with self.assertRaisesRegex(ValueError, "first C4 layer must be 'F'"):
            compute_dsv4_index_topk_flags(self.COMPRESS_RATIOS, 2, 1, pattern)


if __name__ == "__main__":
    unittest.main()
