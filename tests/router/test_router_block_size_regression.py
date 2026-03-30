# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the KV Router block_size fix in multi_worker.rs.

SGLang uses page_size=1 by default, which caused an assertion panic
(`assert!(block_size > 1)`) in the upstream Rust KV Router. The AMD
fix changes this to gracefully default to block_size=16 when block_size <= 1.

This test verifies that block_size=1 does NOT crash the router and that
routing still works correctly with small block sizes.
"""

import logging
from typing import Any, Dict

import pytest

from tests.router.common import _test_router_basic
from tests.router.helper import generate_random_suffix
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)

MODEL_NAME = ROUTER_MODEL_NAME

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.rocm,
    pytest.mark.model(MODEL_NAME),
]

NUM_MOCKERS = 2
SPEEDUP_RATIO = 10.0
NUM_REQUESTS = 20

TEST_PAYLOAD: Dict[str, Any] = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "Hello, world!",
        }
    ],
    "stream": True,
    "max_tokens": 10,
}


def _build_mocker_command(
    endpoint: str,
    num_workers: int,
    block_size: int,
) -> list[str]:
    return [
        "python",
        "-m",
        "dynamo.mocker",
        "--model-path",
        MODEL_NAME,
        "--endpoint",
        endpoint,
        "--num-workers",
        str(num_workers),
        "--speedup-ratio",
        str(SPEEDUP_RATIO),
        "--block-size",
        str(block_size),
    ]


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "block_size",
    [
        pytest.param(1, id="block_size_1_sglang_default"),
        pytest.param(2, id="block_size_2"),
        pytest.param(16, id="block_size_16_standard"),
    ],
)
def test_kv_router_survives_small_block_size(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    block_size,
):
    """Verify KV Router does not panic with block_size=1 (SGLang default).

    This is a regression test for the multi_worker.rs fix. Before the fix,
    block_size=1 caused a Rust panic. After the fix, the router gracefully
    defaults to block_size=16 internally.
    """
    namespace_suffix = generate_random_suffix()
    namespace = f"test-blocksize-{namespace_suffix}"
    endpoint = f"dyn://{namespace}.mocker.generate"

    command = _build_mocker_command(
        endpoint=endpoint,
        num_workers=NUM_MOCKERS,
        block_size=block_size,
    )

    ports = allocate_ports(1, 9100)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port = ports[0]

    mocker_process = ManagedProcess(
        command=command,
        timeout=60,
        display_output=True,
        health_check_ports=[],
        health_check_urls=[],
        log_dir=request.node.name,
        terminate_all_matching_process_names=False,
    )

    with mocker_process:

        class _Workers:
            def __init__(self):
                self.namespace = namespace
                self.component_name = "mocker"
                self.endpoint = endpoint
                self.num_workers = NUM_MOCKERS
                self.dp_size = None
                self.data_parallel_size = None

        workers = _Workers()

        _test_router_basic(
            engine_workers=workers,
            block_size=max(block_size, 2),
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
            request_plane="tcp",
            router_mode="kv",
            min_initial_workers=NUM_MOCKERS,
        )
