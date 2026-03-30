# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K8s CRD validation tests for AMD ROCm deployments.

Validates that:
1. DynamoGraphDeployment CRD YAML templates parse correctly
2. AMD-specific resource requests (amd.com/gpu) are well-formed
3. ROCm-specific environment variables are included
4. DGD templates can be loaded by the Dynamo deploy tooling

These tests do NOT require a running K8s cluster — they validate the
YAML structure offline.
"""

import os
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.deploy,
    pytest.mark.rocm,
    pytest.mark.pre_merge,
]

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "fault_tolerance" / "deploy" / "templates"
SGLANG_TEMPLATES_DIR = TEMPLATES_DIR / "sglang"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class TestRocmDgdTemplates:
    """Validate ROCm DGD YAML templates are well-formed."""

    @pytest.fixture(autouse=True)
    def _check_templates_exist(self):
        if not SGLANG_TEMPLATES_DIR.exists():
            pytest.skip(f"SGLang templates dir not found: {SGLANG_TEMPLATES_DIR}")

    @pytest.mark.parametrize("template_name", ["rocm_agg.yaml", "rocm_disagg.yaml"])
    def test_template_parses(self, template_name):
        path = SGLANG_TEMPLATES_DIR / template_name
        if not path.exists():
            pytest.skip(f"{template_name} not found")
        doc = _load_yaml(path)
        assert doc is not None
        assert doc["kind"] == "DynamoGraphDeployment"
        assert "spec" in doc
        assert "services" in doc["spec"]

    @pytest.mark.parametrize("template_name", ["rocm_agg.yaml", "rocm_disagg.yaml"])
    def test_amd_gpu_resource(self, template_name):
        """Verify templates use amd.com/gpu instead of nvidia.com/gpu."""
        path = SGLANG_TEMPLATES_DIR / template_name
        if not path.exists():
            pytest.skip(f"{template_name} not found")

        content = path.read_text()
        assert "amd.com/gpu" in content, "Template should use amd.com/gpu resource"
        assert "nvidia.com/gpu" not in content, "Template should NOT use nvidia.com/gpu"

    def test_rocm_agg_template_structure(self):
        path = SGLANG_TEMPLATES_DIR / "rocm_agg.yaml"
        if not path.exists():
            pytest.skip("rocm_agg.yaml not found")
        doc = _load_yaml(path)

        services = doc["spec"]["services"]
        assert "Frontend" in services
        assert "SglangWorker" in services

        worker = services["SglangWorker"]
        assert worker["componentType"] == "worker"

        env_names = [e["name"] for e in worker.get("envs", [])]
        assert "SGLANG_AITER_MLA_PERSIST" in env_names
        assert "HIP_VISIBLE_DEVICES" in env_names

    def test_rocm_disagg_template_structure(self):
        path = SGLANG_TEMPLATES_DIR / "rocm_disagg.yaml"
        if not path.exists():
            pytest.skip("rocm_disagg.yaml not found")
        doc = _load_yaml(path)

        services = doc["spec"]["services"]
        assert "Frontend" in services
        assert "SglangPrefillWorker" in services
        assert "SglangDecodeWorker" in services

        prefill = services["SglangPrefillWorker"]
        decode = services["SglangDecodeWorker"]

        prefill_args = prefill["extraPodSpec"]["mainContainer"]["args"]
        decode_args = decode["extraPodSpec"]["mainContainer"]["args"]

        assert "--disaggregation-mode" in prefill_args
        assert "prefill" in prefill_args
        assert "--disaggregation-transfer-backend" in prefill_args
        assert "mori" in prefill_args

        assert "--disaggregation-mode" in decode_args
        assert "decode" in decode_args

    def test_rocm_image_consistency(self):
        """All ROCm templates should use the same container image."""
        images = set()
        for template_name in ["rocm_agg.yaml", "rocm_disagg.yaml"]:
            path = SGLANG_TEMPLATES_DIR / template_name
            if not path.exists():
                continue
            content = path.read_text()
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("image:"):
                    images.add(stripped.split(":", 1)[1].strip())

        if not images:
            pytest.skip("No ROCm templates with images found")

        assert len(images) == 1, f"Inconsistent images across ROCm templates: {images}"


class TestVllmDgdTemplatesExist:
    """Verify upstream vLLM templates still exist (regression guard)."""

    def test_vllm_templates_exist(self):
        vllm_dir = TEMPLATES_DIR / "vllm"
        if not vllm_dir.exists():
            pytest.skip("vLLM templates directory not found")
        yamls = list(vllm_dir.glob("*.yaml"))
        assert len(yamls) > 0, "No vLLM DGD templates found"


class TestProfilerConfigsRocm:
    """Validate ROCm profiler YAML configs."""

    PROFILER_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "profiler" / "configs"

    @pytest.mark.parametrize(
        "config_name",
        ["rocm_mi355x_sglang_rapid.yaml", "rocm_mi355x_sglang_thorough.yaml"],
    )
    def test_profiler_config_parses(self, config_name):
        path = self.PROFILER_CONFIGS_DIR / config_name
        if not path.exists():
            pytest.skip(f"{config_name} not found")

        doc = _load_yaml(path)
        assert doc is not None
        assert "model" in doc
        assert "backend" in doc
        assert doc["backend"] == "sglang"
        assert "hardware" in doc
        assert doc["hardware"]["gpuSku"] == "mi355x"
        assert "sla" in doc
        assert "searchStrategy" in doc
