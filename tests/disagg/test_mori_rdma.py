# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoRI RDMA disaggregated serving integration tests.

These tests validate the prefill/decode disaggregation pipeline using MoRI
as the KV cache transfer backend over Pensando Pollara 400 ionic NICs.

Requirements:
- 2 nodes with MI355X GPUs and ionic RDMA NICs
- Matching ionic subnets between nodes
- MoRI and SGLang installed
- QoS/DCQCN configured per ROCm docs

These are multi-node tests — they require the DISAGG_PREFILL_HOST and
DISAGG_DECODE_HOST environment variables to specify the node addresses,
or they will be skipped.

Skipped entirely on NVIDIA/CUDA systems.
"""

import json
import os
import subprocess
import time

import pytest

try:
    import torch

    _is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
except ImportError:
    _is_hip = os.path.exists("/opt/rocm")

if not _is_hip:
    pytest.skip("MoRI RDMA tests require AMD ROCm (HIP)", allow_module_level=True)

pytestmark = [
    pytest.mark.rocm,
    pytest.mark.mi355x,
    pytest.mark.gpu_8,
    pytest.mark.e2e,
    pytest.mark.weekly,
]


def _get_disagg_hosts():
    """Return (prefill_host, decode_host) from env or None."""
    prefill = os.environ.get("DISAGG_PREFILL_HOST")
    decode = os.environ.get("DISAGG_DECODE_HOST")
    if not prefill or not decode:
        return None
    return prefill, decode


def _wait_for_server(host: str, port: int, timeout: int = 300) -> bool:
    """Poll a server's /health endpoint until ready."""
    import urllib.request

    deadline = time.time() + timeout
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _send_chat_request(
    host: str,
    port: int,
    prompt: str = "What is 2+2?",
    max_tokens: int = 32,
    stream: bool = False,
) -> dict:
    """Send an OpenAI-compatible chat request and return the response."""
    import urllib.request

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


class TestMoriRdmaDisagg:
    """Multi-node MoRI RDMA disaggregated inference tests.

    These tests are SKIPPED unless DISAGG_PREFILL_HOST and
    DISAGG_DECODE_HOST are set. They assume the disagg Dynamo stack is
    already running (started by the test harness or manually).
    """

    @pytest.fixture(autouse=True)
    def _check_multi_node(self):
        hosts = _get_disagg_hosts()
        if hosts is None:
            pytest.skip(
                "Multi-node disagg tests require DISAGG_PREFILL_HOST and "
                "DISAGG_DECODE_HOST environment variables"
            )
        self.prefill_host, self.decode_host = hosts
        self.frontend_port = int(os.environ.get("DISAGG_FRONTEND_PORT", "8000"))

    def test_frontend_reachable(self):
        """Verify the Dynamo frontend is accepting connections."""
        frontend_host = os.environ.get("DISAGG_FRONTEND_HOST", self.prefill_host)
        assert _wait_for_server(
            frontend_host, self.frontend_port, timeout=30
        ), f"Frontend at {frontend_host}:{self.frontend_port} not reachable"

    def test_basic_inference(self):
        """Send a basic request through the disagg pipeline."""
        frontend_host = os.environ.get("DISAGG_FRONTEND_HOST", self.prefill_host)
        resp = _send_chat_request(frontend_host, self.frontend_port)

        assert "choices" in resp, f"No choices in response: {resp}"
        assert len(resp["choices"]) > 0
        content = resp["choices"][0]["message"]["content"]
        assert len(content) > 0, "Empty response content"

    def test_concurrent_requests(self):
        """Send multiple concurrent requests to validate disagg under load."""
        import concurrent.futures

        frontend_host = os.environ.get("DISAGG_FRONTEND_HOST", self.prefill_host)
        num_requests = int(os.environ.get("DISAGG_CONCURRENCY", "4"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as pool:
            futures = [
                pool.submit(
                    _send_chat_request,
                    frontend_host,
                    self.frontend_port,
                    f"Count from 1 to {i + 1}",
                )
                for i in range(num_requests)
            ]

            results = []
            errors = []
            for f in concurrent.futures.as_completed(futures, timeout=300):
                try:
                    results.append(f.result())
                except Exception as e:
                    errors.append(str(e))

        assert len(errors) == 0, f"Errors during concurrent requests: {errors}"
        assert len(results) == num_requests
        for r in results:
            assert "choices" in r
            assert len(r["choices"][0]["message"]["content"]) > 0


class TestMoriRdmaSingleNode:
    """Single-node MoRI-related validation (no multi-node required)."""

    def test_mori_library_loads(self):
        """Verify the MoRI shared library can be loaded."""
        try:
            import mori  # noqa: F401
        except ImportError:
            pytest.skip("mori package not installed")

    def test_sglang_disagg_modes_available(self):
        """Verify SGLang exposes prefill/decode disaggregation modes."""
        try:
            from sglang.srt.server_args import ServerArgs
        except ImportError:
            pytest.skip("sglang not installed")

        args = ServerArgs.__dataclass_fields__
        assert "disaggregation_mode" in args or hasattr(
            ServerArgs, "disaggregation_mode"
        ), "SGLang ServerArgs missing disaggregation_mode"

    def test_rocm_env_vars_set(self):
        """Verify critical ROCm env vars for disagg inference."""
        critical_vars = {
            "SGLANG_AITER_MLA_PERSIST": "False",
        }
        for var, expected in critical_vars.items():
            val = os.environ.get(var)
            if val is None:
                pytest.skip(f"{var} not set")
            assert val == expected, f"{var}={val}, expected {expected}"
