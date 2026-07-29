"""Unit test for the DeepSeek V4 IndexCache descriptor consistency check in
the PD bootstrap server's PUT handler (CommonKVBootstrapServer).

Drives `_handle_route_put` directly (no HTTP server thread) to assert the
all-or-none descriptor enforcement added for IndexCache: once any prefill rank
reports a descriptor, a rank that omits it (or reports it after ranks without
one, or reports a different layout/producer mask) is rejected with 400 and is
NOT counted toward readiness.
"""

import asyncio
import json
import unittest

from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _FakeGetRequest:
    """Mimics the topology GET (all -1 rank query) used by decode.

    `wants_descriptor` mirrors a new-image decode advertising the
    dsv4_indexcache_desc capability param; an old-image decode omits it.
    """

    def __init__(self, wants_descriptor: bool = False):
        self.query = {
            "prefill_dp_rank": "-1",
            "prefill_cp_rank": "-1",
            "target_tp_rank": "-1",
            "target_pp_rank": "-1",
        }
        if wants_descriptor:
            self.query["dsv4_indexcache_desc"] = "1"


def _base_payload(**overrides):
    payload = {
        "attn_tp_size": 1,
        "attn_tp_rank": 0,
        "attn_cp_size": 1,
        "attn_cp_rank": 0,
        "attn_dp_size": 1,
        "attn_dp_rank": 0,
        "pp_size": 1,
        "pp_rank": 0,
        "system_dp_size": 1,
        "system_dp_rank": 0,
        "rank_ip": "127.0.0.1",
        "rank_port": 10000,
        "page_size": 64,
        "kv_cache_dtype": "auto",
        "load_balance_method": "follow_bootstrap_room",
        "dsv4_index_cache_layout_signature": None,
        "dsv4_index_cache_producer_layer_ids": None,
    }
    # attn_tp_rank / rank_port typically vary per rank so each registers a
    # distinct slot; let callers override any field.
    payload.update(overrides)
    return payload


class TestBootstrapIndexCacheDescriptorConsistency(CustomTestCase):
    def _new_server(self):
        # Bypass __init__ so no HTTP server thread is started; set only the
        # attributes _handle_route_put / _handle_route_get touch.
        server = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
        server.lock = asyncio.Lock()
        server.attn_tp_size = None
        server.attn_cp_size = None
        server.dp_size = None
        server.pp_size = None
        server.page_size = None
        server.kv_cache_dtype = None
        server.follow_bootstrap_room = None
        server.enable_dsa_cache_layer_split = None
        server.prefill_http_port = None
        server.dsv4_index_cache_layout_signature = None
        server.dsv4_index_cache_producer_layer_ids = None
        server._dsv4_descriptor_seen = False
        server.prefill_port_table = {}
        server.room_to_dp_rank = {}
        server._registered_count = 0
        return server

    def _put(self, server, **overrides):
        req = _FakeRequest(_base_payload(**overrides))
        return asyncio.run(server._handle_route_put(req))

    def test_all_ranks_report_same_descriptor_ok(self):
        server = self._new_server()
        r1 = self._put(
            server,
            attn_tp_rank=0,
            rank_port=10000,
            dsv4_index_cache_layout_signature="sigA",
            dsv4_index_cache_producer_layer_ids=[2, 6],
        )
        r2 = self._put(
            server,
            attn_tp_rank=1,
            rank_port=10001,
            dsv4_index_cache_layout_signature="sigA",
            dsv4_index_cache_producer_layer_ids=[2, 6],
        )
        self.assertEqual(r1.status, 200)
        self.assertEqual(r2.status, 200)
        self.assertEqual(server._registered_count, 2)

    def test_descriptor_then_missing_rejected(self):
        server = self._new_server()
        r1 = self._put(
            server,
            attn_tp_rank=0,
            rank_port=10000,
            dsv4_index_cache_layout_signature="sigA",
            dsv4_index_cache_producer_layer_ids=[2, 6],
        )
        self.assertEqual(r1.status, 200)
        self.assertEqual(server._registered_count, 1)
        # Second rank omits the descriptor -> 400, count unchanged.
        r2 = self._put(server, attn_tp_rank=1, rank_port=10001)
        self.assertEqual(r2.status, 400)
        self.assertEqual(server._registered_count, 1)

    def test_missing_then_descriptor_rejected(self):
        server = self._new_server()
        r1 = self._put(server, attn_tp_rank=0, rank_port=10000)
        self.assertEqual(r1.status, 200)
        self.assertEqual(server._registered_count, 1)
        # Later rank reports a descriptor after a rank without one -> 400.
        r2 = self._put(
            server,
            attn_tp_rank=1,
            rank_port=10001,
            dsv4_index_cache_layout_signature="sigA",
            dsv4_index_cache_producer_layer_ids=[2, 6],
        )
        self.assertEqual(r2.status, 400)
        self.assertEqual(server._registered_count, 1)

    def test_layout_signature_mismatch_rejected(self):
        server = self._new_server()
        self._put(
            server,
            attn_tp_rank=0,
            rank_port=10000,
            dsv4_index_cache_layout_signature="sigA",
            dsv4_index_cache_producer_layer_ids=[2, 6],
        )
        r2 = self._put(
            server,
            attn_tp_rank=1,
            rank_port=10001,
            dsv4_index_cache_layout_signature="sigB",
            dsv4_index_cache_producer_layer_ids=[2, 6],
        )
        self.assertEqual(r2.status, 400)
        self.assertEqual(server._registered_count, 1)

    def test_producer_mask_mismatch_rejected(self):
        server = self._new_server()
        self._put(
            server,
            attn_tp_rank=0,
            rank_port=10000,
            dsv4_index_cache_layout_signature="sigA",
            dsv4_index_cache_producer_layer_ids=[2, 6],
        )
        r2 = self._put(
            server,
            attn_tp_rank=1,
            rank_port=10001,
            dsv4_index_cache_layout_signature="sigA",
            dsv4_index_cache_producer_layer_ids=[2, 8],
        )
        self.assertEqual(r2.status, 400)
        self.assertEqual(server._registered_count, 1)

    def test_non_v4_all_missing_ok(self):
        server = self._new_server()
        r1 = self._put(server, attn_tp_rank=0, rank_port=10000)
        r2 = self._put(server, attn_tp_rank=1, rank_port=10001)
        self.assertEqual(r1.status, 200)
        self.assertEqual(r2.status, 200)
        self.assertEqual(server._registered_count, 2)
        self.assertFalse(server._dsv4_descriptor_seen)

    def test_get_omits_descriptor_for_old_decode_even_with_descriptor(self):
        # A V4 prefill at freq=1 still reports a real layout signature, so the
        # descriptor IS stored. An OLD-image decode's topology GET omits the
        # capability param, so the response must NOT include the descriptor
        # keys (else its strict PrefillServerInfo(**data) raises TypeError).
        server = self._new_server()
        self._put(
            server,
            attn_tp_rank=0,
            rank_port=10000,
            dsv4_index_cache_layout_signature="sigA",
            dsv4_index_cache_producer_layer_ids=[2, 6],
        )
        resp = asyncio.run(
            server._handle_route_get(_FakeGetRequest(wants_descriptor=False))
        )
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.body.decode())
        self.assertNotIn("dsv4_index_cache_layout_signature", body)
        self.assertNotIn("dsv4_index_cache_producer_layer_ids", body)

    def test_get_omits_descriptor_for_old_decode_non_v4(self):
        # Non-V4 fleet (no descriptor stored) + old decode: keys still absent.
        server = self._new_server()
        self._put(server, attn_tp_rank=0, rank_port=10000)
        resp = asyncio.run(
            server._handle_route_get(_FakeGetRequest(wants_descriptor=False))
        )
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.body.decode())
        self.assertNotIn("dsv4_index_cache_layout_signature", body)
        self.assertNotIn("dsv4_index_cache_producer_layer_ids", body)

    def test_get_includes_descriptor_for_capable_decode(self):
        # A NEW-image decode advertises the capability param, so it receives
        # the descriptor keys (needed for layout + producer-subset validation,
        # including the P freq=1 + D freq=4 gray upgrade).
        server = self._new_server()
        self._put(
            server,
            attn_tp_rank=0,
            rank_port=10000,
            dsv4_index_cache_layout_signature="sigA",
            dsv4_index_cache_producer_layer_ids=[2, 6],
        )
        resp = asyncio.run(
            server._handle_route_get(_FakeGetRequest(wants_descriptor=True))
        )
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.body.decode())
        self.assertEqual(body["dsv4_index_cache_layout_signature"], "sigA")
        self.assertEqual(body["dsv4_index_cache_producer_layer_ids"], [2, 6])

    def test_get_capable_decode_non_v4_keys_present_as_none(self):
        # A new decode against a non-V4 prefill: capability param present, so
        # the keys ARE included (as None); the new PrefillServerInfo defaults
        # them to None anyway and the D-side check is skipped.
        server = self._new_server()
        self._put(server, attn_tp_rank=0, rank_port=10000)
        resp = asyncio.run(
            server._handle_route_get(_FakeGetRequest(wants_descriptor=True))
        )
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.body.decode())
        self.assertIsNone(body["dsv4_index_cache_layout_signature"])
        self.assertIsNone(body["dsv4_index_cache_producer_layer_ids"])


if __name__ == "__main__":
    unittest.main()
