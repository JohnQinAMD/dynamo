# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ROCm-specific SGLang serve integration tests.

These tests use the ROCm launch scripts (examples/backends/sglang/launch/rocm/)
which include AMD-specific settings: HIP_VISIBLE_DEVICES, VLLM_ROCM_USE_AITER,
--page-size 16, etc.

Counterpart to test_sglang.py for AMD GPUs. Skipped on NVIDIA/CUDA systems.
"""

import dataclasses
import logging
import os
from dataclasses import dataclass, field

import pytest

if not os.path.exists("/opt/rocm"):
    pytest.skip("ROCm serve tests require /opt/rocm", allow_module_level=True)

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    chat_payload_default,
    completion_payload_default,
    metric_payload_default,
)

logger = logging.getLogger(__name__)


@dataclass
class SGLangROCmConfig(EngineConfig):
    """Configuration for SGLang on ROCm test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["SGLANG:EngineCore"])


rocm_sglang_dir = os.path.join(
    WORKSPACE_DIR, "examples/backends/sglang/launch/rocm"
)

rocm_sglang_configs = {
    "rocm_aggregated": SGLangROCmConfig(
        name="rocm_aggregated",
        directory=rocm_sglang_dir,
        script_name="agg_rocm.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.max_vram_gib(6.5),
            pytest.mark.timeout(300),
            pytest.mark.pre_merge,
            pytest.mark.rocm,
            pytest.mark.mi355x,
        ],
        model="Qwen/Qwen3-0.6B",
        env={
            "HIP_VISIBLE_DEVICES": "0",
            "SGLANG_AITER_MLA_PERSIST": "False",
        },
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=4, backend="sglang"),
        ],
    ),
    "rocm_aggregated_kvbm": SGLangROCmConfig(
        name="rocm_aggregated_kvbm",
        directory=rocm_sglang_dir,
        script_name="agg_kvbm_rocm.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.max_vram_gib(8.0),
            pytest.mark.timeout(300),
            pytest.mark.post_merge,
            pytest.mark.rocm,
            pytest.mark.mi355x,
            pytest.mark.kvbm,
        ],
        model="Qwen/Qwen3-0.6B",
        env={
            "HIP_VISIBLE_DEVICES": "0",
            "SGLANG_AITER_MLA_PERSIST": "False",
            "DYN_KVBM_CPU_CACHE_GB": "2",
        },
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
}


@pytest.fixture(params=params_with_model_mark(rocm_sglang_configs))
def rocm_sglang_config_test(request):
    """Fixture that provides different ROCm SGLang test configurations"""
    return rocm_sglang_configs[request.param]


@pytest.mark.e2e
@pytest.mark.sglang
@pytest.mark.rocm
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_sglang_rocm_deployment(
    rocm_sglang_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """Test SGLang deployment on AMD ROCm GPUs using ROCm launch scripts."""
    if not os.path.exists("/opt/rocm"):
        pytest.skip("ROCm not installed — skipping ROCm serve test")

    config = dataclasses.replace(
        rocm_sglang_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
