# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang FPM (Forward Pass Metrics) relay.

The SglangFpmRelay bridges SGLang scheduler KvMetrics to Dynamo's
ForwardPassMetrics event plane, enabling the Dynamic Planner to
auto-scale SGLang workers. This is AMD-additive code.

These tests verify:
1. KvMetrics → ForwardPassMetrics conversion correctness
2. ZMQ publisher lifecycle (start/publish/shutdown)
3. End-to-end relay with a mock KvMetrics source
"""

import time

import msgspec
import pytest
import zmq

pytestmark = [
    pytest.mark.rocm,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
    pytest.mark.pre_merge,
]


@pytest.fixture
def mock_kv_metrics():
    """Create a mock SGLang KvMetrics-like object."""

    class MockKvMetrics:
        def __init__(
            self,
            request_active_slots=10,
            num_requests_waiting=5,
            kv_active_blocks=200,
            kv_total_blocks=1000,
            gpu_cache_usage_perc=0.2,
            data_parallel_rank=0,
        ):
            self.request_active_slots = request_active_slots
            self.num_requests_waiting = num_requests_waiting
            self.kv_active_blocks = kv_active_blocks
            self.kv_total_blocks = kv_total_blocks
            self.gpu_cache_usage_perc = gpu_cache_usage_perc
            self.data_parallel_rank = data_parallel_rank

    return MockKvMetrics


class TestKvMetricsToFpm:
    """Test KvMetrics → ForwardPassMetrics conversion."""

    def test_basic_conversion(self, mock_kv_metrics):
        try:
            from dynamo.sglang.fpm_relay import kv_metrics_to_fpm
        except ImportError:
            pytest.skip("dynamo.sglang.fpm_relay not importable")

        kv = mock_kv_metrics(
            request_active_slots=10,
            num_requests_waiting=5,
            kv_active_blocks=200,
            data_parallel_rank=0,
        )

        fpm = kv_metrics_to_fpm(kv, worker_id="test-worker-1", wall_time=0.1)

        assert fpm.worker_id == "test-worker-1"
        assert fpm.dp_rank == 0
        assert fpm.wall_time == pytest.approx(0.1)
        assert fpm.scheduled_requests.num_decode_requests == 10
        assert fpm.scheduled_requests.sum_decode_kv_tokens == 200
        assert fpm.queued_requests.num_prefill_requests == 5

    def test_conversion_with_dp_rank(self, mock_kv_metrics):
        try:
            from dynamo.sglang.fpm_relay import kv_metrics_to_fpm
        except ImportError:
            pytest.skip("dynamo.sglang.fpm_relay not importable")

        kv = mock_kv_metrics(data_parallel_rank=3)
        fpm = kv_metrics_to_fpm(kv, worker_id="worker-dp3", wall_time=0.05)
        assert fpm.dp_rank == 3

    def test_conversion_zero_values(self, mock_kv_metrics):
        try:
            from dynamo.sglang.fpm_relay import kv_metrics_to_fpm
        except ImportError:
            pytest.skip("dynamo.sglang.fpm_relay not importable")

        kv = mock_kv_metrics(
            request_active_slots=0,
            num_requests_waiting=0,
            kv_active_blocks=0,
        )
        fpm = kv_metrics_to_fpm(kv, worker_id="idle-worker", wall_time=0.0)
        assert fpm.scheduled_requests.num_decode_requests == 0
        assert fpm.queued_requests.num_prefill_requests == 0

    def test_conversion_none_values(self, mock_kv_metrics):
        """SGLang may send None for some fields."""
        try:
            from dynamo.sglang.fpm_relay import kv_metrics_to_fpm
        except ImportError:
            pytest.skip("dynamo.sglang.fpm_relay not importable")

        kv = mock_kv_metrics()
        kv.request_active_slots = None
        kv.num_requests_waiting = None
        kv.kv_active_blocks = None
        kv.data_parallel_rank = None

        fpm = kv_metrics_to_fpm(kv, worker_id="null-worker", wall_time=0.0)
        assert fpm.dp_rank == 0
        assert fpm.scheduled_requests.num_decode_requests == 0


class TestFpmPublisherLifecycle:
    """Test the ZMQ publisher thread start/stop."""

    def test_publisher_starts_and_stops(self):
        try:
            from dynamo.sglang.fpm_relay import _SglangFpmPublisherThread
        except ImportError:
            pytest.skip("dynamo.sglang.fpm_relay not importable")

        pub = _SglangFpmPublisherThread(
            endpoint="tcp://127.0.0.1:*",
            worker_id="test-lifecycle",
            dp_rank=0,
        )
        assert pub._thread.is_alive()
        pub.shutdown()
        assert not pub._thread.is_alive()


class TestSglangFpmRelayE2E:
    """End-to-end test: relay publishes, subscriber receives."""

    def test_relay_publishes_metrics(self, mock_kv_metrics):
        try:
            from dynamo.common.forward_pass_metrics import ForwardPassMetrics, decode
            from dynamo.sglang.fpm_relay import SglangFpmRelay
        except ImportError:
            pytest.skip("FPM relay modules not importable")

        import os

        test_port = 29999
        os.environ["DYN_FORWARDPASS_METRIC_PORT"] = str(test_port)

        try:
            relay = SglangFpmRelay(worker_id="e2e-test", dp_rank=0)

            ctx = zmq.Context.instance()
            sub = ctx.socket(zmq.SUB)
            sub.connect(f"tcp://127.0.0.1:{test_port}")
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.setsockopt(zmq.RCVTIMEO, 5000)

            time.sleep(0.5)

            kv = mock_kv_metrics(
                request_active_slots=42,
                num_requests_waiting=7,
                kv_active_blocks=500,
            )
            relay.on_kv_metrics(kv)

            try:
                parts = sub.recv_multipart()
                assert len(parts) == 3
                _topic, _seq, payload = parts
                fpm = decode(payload)
                assert fpm.worker_id == "e2e-test"
                assert fpm.scheduled_requests.num_decode_requests == 42
                assert fpm.queued_requests.num_prefill_requests == 7
            except zmq.Again:
                pytest.fail("Timed out waiting for FPM message from relay")
            finally:
                sub.close()
                relay.shutdown()
        finally:
            os.environ.pop("DYN_FORWARDPASS_METRIC_PORT", None)
