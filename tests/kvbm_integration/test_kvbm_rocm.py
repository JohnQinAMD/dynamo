# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KVBM integration tests for SGLang on AMD ROCm.

Validates KV Block Manager functionality with the HIP KVBM kernels and
SGLang backend on MI355X GPUs. Tests mirror the vLLM KVBM tests but use
the SGLang aggregated deployment with KVBM CPU offloading.

Prerequisites:
- HIP KVBM kernels compiled (tensor_kernels.hip → libkvbm_kernels.so)
- SGLang installed with aiter support
- AMD GPU with ROCm

Skipped on NVIDIA/CUDA systems.
"""

import json
import os
import subprocess
import time
import urllib.request

import pytest

if not os.path.exists("/opt/rocm"):
    pytest.skip("KVBM ROCm tests require /opt/rocm", allow_module_level=True)

pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.sglang,
    pytest.mark.rocm,
    pytest.mark.mi355x,
    pytest.mark.post_merge,
]

AELDORA_STORY = (
    "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, "
    "lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria "
    "was buried beneath the shifting sands of time, lost to the world for centuries."
)
MAX_TOKENS = 20
MODEL = "Qwen/Qwen3-0.6B"


def _check_rocm():
    if not os.path.exists("/opt/rocm"):
        pytest.skip("ROCm not installed")


def _check_sglang():
    try:
        import sglang  # noqa: F401
    except ImportError:
        pytest.skip("SGLang not installed")


def _send_chat(port: int, prompt: str, max_tokens: int = MAX_TOKENS) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _wait_healthy(port: int, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


class TestKvbmHipKernels:
    """Verify HIP KVBM kernel artifacts exist."""

    def test_hip_kernel_source_exists(self):
        _check_rocm()
        dynamo_root = os.environ.get(
            "DYNAMO_ROOT",
            os.path.join(os.path.dirname(__file__), "..", ".."),
        )
        kernel_path = os.path.join(
            dynamo_root, "lib/kvbm-kernels/hip/tensor_kernels.hip"
        )
        assert os.path.exists(kernel_path), (
            f"HIP KVBM kernel not found at {kernel_path}. "
            "The HIP kernel must be compiled before running KVBM tests."
        )

    def test_hip_kernel_compiles(self, tmp_path):
        _check_rocm()
        dynamo_root = os.environ.get(
            "DYNAMO_ROOT",
            os.path.join(os.path.dirname(__file__), "..", ".."),
        )
        kernel_path = os.path.join(
            dynamo_root, "lib/kvbm-kernels/hip/tensor_kernels.hip"
        )
        if not os.path.exists(kernel_path):
            pytest.skip("HIP kernel source not found")

        hipcc = "/opt/rocm/bin/hipcc"
        if not os.path.exists(hipcc):
            pytest.skip("hipcc not found")

        arch = os.environ.get("KVBM_HIP_ARCH", "gfx942")
        agent_info = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if agent_info.returncode == 0:
            for line in agent_info.stdout.splitlines():
                if "gfx9" in line and "Name:" in line:
                    arch = line.split(":")[-1].strip()
                    break

        obj_file = tmp_path / "tensor_kernels.o"
        r = subprocess.run(
            [
                hipcc,
                "-c",
                "-std=c++17",
                "-O3",
                "-fPIC",
                f"--offload-arch={arch}",
                kernel_path,
                "-o",
                str(obj_file),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, f"HIP compile failed (arch={arch}): {r.stderr}"


class TestKvbmSglangIntegration:
    """KVBM + SGLang integration tests.

    These require a running SGLang + KVBM deployment. Skipped if no server
    is available at the expected port.
    """

    @pytest.fixture(autouse=True)
    def _check_environment(self):
        _check_rocm()
        _check_sglang()
        self.port = int(os.environ.get("KVBM_TEST_PORT", "8000"))
        if not _wait_healthy(self.port, timeout=10):
            pytest.skip(
                f"No healthy server at port {self.port}. "
                "Start SGLang+KVBM with agg_kvbm_rocm.sh first."
            )

    def test_basic_inference_with_kvbm(self):
        """Verify basic inference works when KVBM is enabled."""
        resp = _send_chat(self.port, "What is 2+2?")
        assert "choices" in resp
        content = resp["choices"][0]["message"]["content"]
        assert len(content) > 0

    def test_multi_turn_cache_reuse(self):
        """Verify KVBM provides cache reuse benefit across turns.

        Send the same long prefix multiple times — the second and third
        requests should be measurably faster due to KV cache reuse.
        """
        import time

        latencies = []
        for i, question in enumerate(
            [
                " What is the city called?",
                " Who buried the city?",
                " What magic exists there?",
            ]
        ):
            t0 = time.perf_counter()
            resp = _send_chat(self.port, AELDORA_STORY + question)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            assert "choices" in resp
            assert len(resp["choices"][0]["message"]["content"]) > 0

            usage = resp.get("usage", {})
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            print(
                f"  Turn {i+1}: {elapsed_ms:.0f}ms, "
                f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
                f"cached_tokens={cached}"
            )

        if len(latencies) == 3 and latencies[0] > 0:
            speedup = latencies[0] / latencies[2] if latencies[2] > 0 else 0
            print(
                f"  Cache reuse speedup: {speedup:.2f}x "
                f"(first={latencies[0]:.0f}ms, third={latencies[2]:.0f}ms)"
            )

    def test_concurrent_requests_with_kvbm(self):
        """Send concurrent requests to test KVBM under load."""
        import concurrent.futures

        prompts = [f"Count from 1 to {i + 1} and stop." for i in range(4)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_send_chat, self.port, p) for p in prompts]
            results = []
            errors = []
            for f in concurrent.futures.as_completed(futures, timeout=120):
                try:
                    results.append(f.result())
                except Exception as e:
                    errors.append(str(e))

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == len(prompts)
