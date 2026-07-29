"""B200 extra CI: DeepSeek-V4-Flash FP4 + attn-CP + EAGLE with IndexCache.

Mirrors test_deepseek_v4_flash_fp4_b200_cp.py (the production-shaped recipe:
TP=4, attn-CP=4, DSA prefill CP round-robin-split, EAGLE 1-step spec, FP4
checkpoint) but adds the DeepSeek V4 IndexCache override
(index_topk_freq=4). This exercises the C4 producer/shared forward paths and
the raw->page transform under the real CP + FP4 + speculative production
combination, which the single-node FP8 e2e does not cover.

Registry: extra-b-test-4-gpu-b200 (label-gated extra CI, 4x B200).
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=260, stage="extra-b", runner_config="deepep-4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
INDEX_CACHE_OVERRIDE = '{"index_topk_freq": 4}'


class TestDSV4FlashFP4B200CP_IndexCache(
    SpecDecodingMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """FP4 + attn-CP=4 + EAGLE with IndexCache freq=4 (accuracy tradeoff)."""

    # IndexCache is a precision-for-throughput approximation, so hold GSM8K to
    # the same production bar the non-IndexCache CP recipe uses.
    gsm8k_accuracy_thres = 0.93
    accept_length_thres = 1.8
    bs_1_speed_thres = 100

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--attn-cp-size",
                "4",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--enable-dsa-prefill-context-parallel",
                "--dsa-prefill-cp-mode",
                "round-robin-split",
                "--moe-runner-backend",  # for fp4 checkpoint
                "flashinfer_mxfp4",
                # Enable the C4 FP4 indexer (the MoE backend above is the expert
                # path and does NOT turn this on). Production P/D runs the FP4
                # indexer, and IndexCache must be validated against it, not FP8.
                # Requires SM100 (B200), which this runner provides.
                "--enable-deepseek-v4-fp4-indexer",
                "--json-model-override-args",
                INDEX_CACHE_OVERRIDE,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
