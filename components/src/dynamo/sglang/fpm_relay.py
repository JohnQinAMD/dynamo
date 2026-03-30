# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang Forward Pass Metrics (FPM) relay.

Bridges SGLang scheduler metrics to Dynamo's ForwardPassMetrics event plane,
enabling the Dynamic Planner to perform load-based scaling with SGLang workers.

Data flow::

    SGLang scheduler child process:
        SchedulerMetricsMixin._emit_kv_metrics() -> ZMQ PUSH (IPC)

    Dynamo parent process (this module):
        SglangFpmRelay (ZMQ PULL) -> construct ForwardPassMetrics
            -> _FpmPublisherThread -> ZMQ PUB (localhost)

    FpmEventRelay (Rust bridge):
        ZMQ SUB -> EventPublisher -> Event Plane (NATS)

    Consumer (planner):
        FpmEventSubscriber -> decode() -> ForwardPassMetrics

The SGLang KvMetrics provides:
    - request_active_slots (num running requests)
    - num_requests_waiting (queued requests)
    - kv_active_blocks / kv_total_blocks (cache utilization)
    - gpu_cache_usage_perc
    - data_parallel_rank

We map these to ForwardPassMetrics fields for the planner's load-based
scaling regression models (PrefillRegressionModel / DecodeRegressionModel).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from itertools import count
from typing import TYPE_CHECKING

import msgspec.structs
import zmq

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
    encode,
)

if TYPE_CHECKING:
    from sglang.srt.observability.scheduler_metrics_mixin import KvMetrics

logger = logging.getLogger(__name__)

DEFAULT_FPM_PORT = 20380
ENV_FPM_PORT = "DYN_FORWARDPASS_METRIC_PORT"


class _SglangFpmPublisherThread:
    """Background thread that publishes ForwardPassMetrics over ZMQ PUB.

    Same pattern as vLLM's _FpmPublisherThread but fed by SGLang KvMetrics.
    """

    SHUTDOWN_TIMEOUT: float = 1.0
    HEARTBEAT_INTERVAL: float = 1.0

    def __init__(
        self,
        endpoint: str,
        worker_id: str,
        dp_rank: int,
        max_queue_size: int = 10_000,
    ) -> None:
        self._queue: queue.Queue[ForwardPassMetrics | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._seq = count()
        self._worker_id = worker_id
        self._dp_rank = dp_rank

        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(endpoint)

        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="sglang-fpm-publisher"
        )
        self._thread.start()
        logger.info(
            "SGLang FPM publisher bound on %s (worker_id=%s, dp_rank=%d)",
            endpoint,
            worker_id,
            dp_rank,
        )

    def publish(self, metrics: ForwardPassMetrics) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(metrics)
        except queue.Full:
            pass

    def shutdown(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            self._pub.close(linger=0)
        except Exception:
            pass

    def _run(self) -> None:
        topic = b""
        last_publish = time.monotonic()

        while self._running or not self._queue.empty():
            try:
                metrics = self._queue.get(timeout=self.HEARTBEAT_INTERVAL)
                if metrics is None:
                    break
            except queue.Empty:
                if time.monotonic() - last_publish >= self.HEARTBEAT_INTERVAL:
                    metrics = ForwardPassMetrics(
                        worker_id=self._worker_id,
                        dp_rank=self._dp_rank,
                    )
                else:
                    continue

            try:
                seq = next(self._seq)
                metrics = msgspec.structs.replace(metrics, counter_id=seq)
                payload = encode(metrics)
                seq_bytes = seq.to_bytes(8, "big")
                self._pub.send_multipart(
                    (topic, seq_bytes, payload), flags=zmq.NOBLOCK
                )
                last_publish = time.monotonic()
            except zmq.Again:
                pass
            except Exception:
                logger.warning("SGLang FPM publisher send failed", exc_info=True)


def kv_metrics_to_fpm(
    kv_metrics: "KvMetrics",
    worker_id: str,
    wall_time: float,
) -> ForwardPassMetrics:
    """Convert SGLang KvMetrics to Dynamo ForwardPassMetrics.

    SGLang's KvMetrics provides aggregate scheduler state. We map it to
    ForwardPassMetrics fields that the planner's regression models use:

    - num_decode_requests → request_active_slots (running requests)
    - sum_decode_kv_tokens → kv_active_blocks (active KV cache)
    - num_prefill_requests → 0 (SGLang doesn't separate in KvMetrics)
    - queued requests → num_requests_waiting
    """
    dp_rank = kv_metrics.data_parallel_rank or 0
    num_running = kv_metrics.request_active_slots or 0
    num_waiting = kv_metrics.num_requests_waiting or 0
    kv_active = kv_metrics.kv_active_blocks or 0

    return ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=dp_rank,
        wall_time=wall_time,
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=num_running,
            sum_decode_kv_tokens=kv_active,
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=num_waiting,
        ),
    )


class SglangFpmRelay:
    """Relay SGLang KvMetrics → ForwardPassMetrics via ZMQ PUB.

    Integrates with DynamoSglangPublisher to intercept KvMetrics and
    additionally emit ForwardPassMetrics for the Dynamic Planner.

    Usage in dynamo.sglang init_llm.py::

        fpm_relay = SglangFpmRelay(worker_id=connection_id, dp_rank=0)
        # In the publisher.run() loop, after receiving kv_metrics:
        fpm_relay.on_kv_metrics(kv_metrics)
    """

    def __init__(self, worker_id: str, dp_rank: int = 0) -> None:
        base_port = int(os.environ.get(ENV_FPM_PORT, str(DEFAULT_FPM_PORT)))
        port = base_port + dp_rank
        self._publisher = _SglangFpmPublisherThread(
            f"tcp://*:{port}",
            worker_id=worker_id,
            dp_rank=dp_rank,
        )
        self._worker_id = worker_id
        self._last_time: float = 0.0

    def on_kv_metrics(self, kv_metrics: "KvMetrics") -> None:
        """Called each time SGLang emits KvMetrics. Converts and publishes FPM."""
        now = time.monotonic()
        wall_time = now - self._last_time if self._last_time > 0 else 0.0
        self._last_time = now

        fpm = kv_metrics_to_fpm(kv_metrics, self._worker_id, wall_time)
        self._publisher.publish(fpm)

    def shutdown(self) -> None:
        self._publisher.shutdown()
