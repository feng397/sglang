"""8-GPU extra-b CI: DeepSeek-V4-Flash FP8 P/D + HiSparse with IndexCache.

Mirrors test_disaggregation_hisparse.py (the production V4 PD + HiSparse
recipe: TP=4, page-size 256, chunked prefill, decode-side HiSparse C4 cache
offload) and adds the DeepSeek V4 IndexCache override (index_topk_freq=4) on
BOTH prefill and decode.

This is the combination that most directly stresses producer indexer-cache
handling: the prefill C4 producer layers generate the indexer K/state that is
transferred to decode, and HiSparse manages the C4 sparse pages the shared
layers translate raw top-k against. It also exercises the P/D IndexCache
descriptor bootstrap handshake under the HiSparse cache path.

Registry: extra-b, 8x H200 (4 prefill + 4 decode).
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=1000, stage="extra-b", runner_config="deepep-8-gpu-h200")

DSV4_FLASH_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
DSV4_FLASH_LOADER_CONFIG = '{"enable_multithread_load": true, "num_threads": 64}'
DSV4_HISPARSE_CONFIG = (
    '{"top_k":512,"device_buffer_size":4096,"host_to_device_ratio":2}'
)
INDEX_CACHE_OVERRIDE = '{"index_topk_freq": 4}'

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
}


class TestDisaggregationDSV4HiSparseIndexCache(
    PDDisaggregationServerBase, GSM8KMixin
):
    # IndexCache is a precision-for-throughput approximation; hold GSM8K to the
    # same production bar the non-IndexCache HiSparse PD recipe uses.
    gsm8k_accuracy_thres = 0.93
    gsm8k_num_questions = 200
    gsm8k_num_shots = 20

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model = try_cached_model(DSV4_FLASH_MODEL)
        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            4,
            "--page-size",
            256,
            "--chunked-prefill-size",
            8192,
            "--max-running-requests",
            16,
            "--mem-fraction-static",
            0.9,
            "--skip-server-warmup",
            "--reasoning-parser",
            "deepseek-v4",
            "--tool-call-parser",
            "deepseekv4",
            "--model-loader-extra-config",
            DSV4_FLASH_LOADER_CONFIG,
            "--json-model-override-args",
            INDEX_CACHE_OVERRIDE,
            "--watchdog-timeout",
            "900",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=DSV4_FLASH_ENV,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            4,
            "--base-gpu-id",
            4,
            "--page-size",
            256,
            "--chunked-prefill-size",
            8192,
            "--max-running-requests",
            16,
            "--mem-fraction-static",
            0.9,
            "--skip-server-warmup",
            "--reasoning-parser",
            "deepseek-v4",
            "--tool-call-parser",
            "deepseekv4",
            "--model-loader-extra-config",
            DSV4_FLASH_LOADER_CONFIG,
            "--enable-hisparse",
            "--hisparse-config",
            DSV4_HISPARSE_CONFIG,
            "--json-model-override-args",
            INDEX_CACHE_OVERRIDE,
            "--watchdog-timeout",
            "900",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=DSV4_FLASH_ENV,
        )


if __name__ == "__main__":
    unittest.main()
