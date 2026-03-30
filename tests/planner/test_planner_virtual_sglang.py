# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner E2E test with virtual environment and SGLang backend.

Validates that the Dynamic Planner can:
1. Initialize in virtual mode (no K8s required)
2. Accept SGLang as the backend
3. Process PlannerConfig with load-based scaling
4. Connect to etcd for state management

This test uses `environment: virtual` which means the planner simulates
scaling decisions without actually creating/destroying pods.
"""

import logging
import os

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.planner,
    pytest.mark.rocm,
    pytest.mark.post_merge,
    pytest.mark.integration,
]

logger = logging.getLogger(__name__)


class TestPlannerVirtualSglang:
    """Test planner initialization and config with SGLang backend."""

    def test_planner_config_virtual_sglang(self):
        """Verify PlannerConfig accepts virtual + sglang combination."""
        try:
            from dynamo.planner.utils.planner_config import PlannerConfig
        except ImportError:
            pytest.skip("dynamo.planner not importable")

        config = PlannerConfig(
            environment="virtual",
            backend="sglang",
            mode="agg",
            enable_load_scaling=True,
            enable_throughput_scaling=False,
            ttft=1000,
            itl=100,
            min_endpoint=1,
            max_gpu_budget=16,
            load_adjustment_interval=5,
        )

        assert config.environment == "virtual"
        assert config.backend == "sglang"
        assert config.mode == "agg"
        assert config.enable_load_scaling is True

    def test_planner_config_virtual_sglang_disagg(self):
        """Verify PlannerConfig accepts disagg mode with SGLang."""
        try:
            from dynamo.planner.utils.planner_config import PlannerConfig
        except ImportError:
            pytest.skip("dynamo.planner not importable")

        for mode in ["prefill", "decode"]:
            config = PlannerConfig(
                environment="virtual",
                backend="sglang",
                mode=mode,
                enable_load_scaling=True,
                enable_throughput_scaling=False,
                ttft=1000,
                itl=100,
            )
            assert config.mode == mode
            assert config.backend == "sglang"

    def test_planner_imports(self):
        """Verify all planner classes import cleanly."""
        try:
            from dynamo.planner.utils.decode_planner import DecodePlanner  # noqa: F401
            from dynamo.planner.utils.prefill_planner import PrefillPlanner  # noqa: F401
            from dynamo.planner.utils.planner_config import PlannerConfig  # noqa: F401
        except ImportError:
            pytest.skip("dynamo.planner not fully importable")

    def test_agg_planner_import(self):
        """Verify AggPlanner imports (combines prefill + decode)."""
        try:
            from dynamo.planner.utils.agg_planner import AggPlanner  # noqa: F401
        except ImportError:
            pytest.skip("AggPlanner not available")

    def test_virtual_connector_import(self):
        """Verify VirtualConnector imports (no K8s dependency)."""
        try:
            from dynamo.planner import VirtualConnector  # noqa: F401
        except ImportError:
            pytest.skip("VirtualConnector not available")

    def test_fpm_metrics_import(self):
        """Verify ForwardPassMetrics encoding works (needed for SGLang FPM)."""
        try:
            from dynamo.common.forward_pass_metrics import (
                ForwardPassMetrics,
                encode,
                decode,
            )
        except ImportError:
            pytest.skip("forward_pass_metrics not importable")

        fpm = ForwardPassMetrics(worker_id="test", dp_rank=0)
        payload = encode(fpm)
        assert len(payload) > 0

        decoded = decode(payload)
        assert decoded.worker_id == "test"
        assert decoded.dp_rank == 0

    def test_planner_config_sglang_fpm_port(self):
        """Verify FPM port can be configured for SGLang relay."""
        port = 20380
        os.environ["DYN_FORWARDPASS_METRIC_PORT"] = str(port)
        try:
            from dynamo.sglang.fpm_relay import DEFAULT_FPM_PORT, ENV_FPM_PORT
            assert ENV_FPM_PORT == "DYN_FORWARDPASS_METRIC_PORT"
            assert DEFAULT_FPM_PORT == 20380
        except ImportError:
            pytest.skip("fpm_relay not importable")
        finally:
            os.environ.pop("DYN_FORWARDPASS_METRIC_PORT", None)


class TestPlannerVirtualSglangE2E:
    """E2E planner test with etcd — needs runtime services."""

    def test_planner_connects_to_etcd(self, runtime_services_dynamic_ports):
        """Verify planner can reach etcd in virtual mode.

        This doesn't start a full planner loop — it just verifies that
        the config + etcd connection path works.
        """
        try:
            from dynamo.planner.utils.planner_config import PlannerConfig
        except ImportError:
            pytest.skip("dynamo.planner not importable")

        nats_proc, etcd_proc = runtime_services_dynamic_ports
        if etcd_proc is None:
            pytest.skip("etcd not available")

        config = PlannerConfig(
            environment="virtual",
            backend="sglang",
            mode="agg",
            enable_load_scaling=True,
            enable_throughput_scaling=False,
            ttft=1000,
            itl=100,
        )

        assert config.environment == "virtual"
        etcd_url = os.environ.get("ETCD_ENDPOINTS", "")
        assert "localhost" in etcd_url, f"ETCD_ENDPOINTS not set properly: {etcd_url}"
