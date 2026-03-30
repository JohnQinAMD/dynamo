# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ROCm build and GPU detection validation tests.

Verifies that AMD GPU hardware is visible, HIP kernels compile, RIXL
libraries are installed, and the Dynamo gpu_utils module correctly
detects the AMD backend. These tests are the foundation for all other
ROCm tests — if they fail, nothing else will work.

Skipped on NVIDIA/CUDA systems.
"""

import os
import shutil
import subprocess

import pytest

_is_rocm = os.path.exists("/opt/rocm") or shutil.which("rocm-smi") is not None
if not _is_rocm:
    pytest.skip("ROCm tests require /opt/rocm or rocm-smi", allow_module_level=True)

pytestmark = [
    pytest.mark.rocm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def _has_rocm_smi() -> bool:
    return shutil.which("rocm-smi") is not None or os.path.exists(
        "/opt/rocm/bin/rocm-smi"
    )


def _has_hipcc() -> bool:
    return shutil.which("hipcc") is not None or os.path.exists("/opt/rocm/bin/hipcc")


def _rocm_smi(*args: str) -> subprocess.CompletedProcess:
    cmd = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"
    return subprocess.run(
        [cmd, *args],
        capture_output=True,
        text=True,
        timeout=15,
    )


class TestRocmEnvironment:
    """Verify the ROCm software stack is present."""

    @pytest.mark.skipif(not _has_rocm_smi(), reason="rocm-smi not found")
    def test_rocm_smi_detects_gpus(self):
        r = _rocm_smi("--showproductname")
        assert r.returncode == 0, f"rocm-smi failed: {r.stderr}"
        output = r.stdout
        assert output.strip(), "rocm-smi returned empty output"

    @pytest.mark.skipif(not _has_hipcc(), reason="hipcc not found")
    def test_hipcc_version(self):
        hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
        r = subprocess.run(
            [hipcc, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert r.returncode == 0, f"hipcc --version failed: {r.stderr}"
        assert "HIP" in r.stdout or "hip" in r.stdout.lower()

    def test_rocm_path_exists(self):
        if not os.path.exists("/opt/rocm"):
            pytest.skip("ROCm not installed at /opt/rocm")
        assert os.path.isdir("/opt/rocm"), "/opt/rocm is not a directory"


class TestHipKernelCompilation:
    """Verify that HIP KVBM kernels can compile on this platform."""

    KERNEL_SOURCE = "lib/kvbm-kernels/hip/tensor_kernels.hip"

    @pytest.fixture(autouse=True)
    def _check_prerequisites(self):
        if not _has_hipcc():
            pytest.skip("hipcc not available")
        dynamo_root = os.environ.get(
            "DYNAMO_ROOT",
            os.path.join(os.path.dirname(__file__), "..", ".."),
        )
        kernel_path = os.path.join(dynamo_root, self.KERNEL_SOURCE)
        if not os.path.exists(kernel_path):
            pytest.skip(f"HIP kernel source not found: {kernel_path}")
        self.kernel_path = kernel_path
        self.hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"

    @pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
    def test_hip_kernel_compiles(self, arch, tmp_path):
        obj_file = tmp_path / f"tensor_kernels_{arch}.o"
        r = subprocess.run(
            [
                self.hipcc,
                "-c",
                "-std=c++17",
                "-O3",
                "-fPIC",
                f"--offload-arch={arch}",
                self.kernel_path,
                "-o",
                str(obj_file),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, f"HIP compile failed for {arch}: {r.stderr}"
        assert obj_file.exists(), f"Object file not created for {arch}"

    def test_shared_library_links(self, tmp_path):
        obj_file = tmp_path / "tensor_kernels.o"
        so_file = tmp_path / "libkvbm_kernels.so"

        r = subprocess.run(
            [
                self.hipcc,
                "-c",
                "-std=c++17",
                "-O3",
                "-fPIC",
                "--offload-arch=gfx942",
                self.kernel_path,
                "-o",
                str(obj_file),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            pytest.skip(f"Compile step failed: {r.stderr}")

        r = subprocess.run(
            [self.hipcc, "-shared", "-o", str(so_file), str(obj_file), "-lamdhip64"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode == 0, f"Linking failed: {r.stderr}"
        assert so_file.exists()

        nm_r = subprocess.run(
            ["nm", "-D", str(so_file)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "kvbm_kernels" in nm_r.stdout, "Expected kvbm_kernels symbols not found"


class TestDynamoGpuUtils:
    """Verify the Dynamo gpu_utils module detects AMD GPUs correctly."""

    def test_detect_gpu_backend(self):
        try:
            from dynamo.common.gpu_utils import detect_gpu_backend
        except ImportError:
            pytest.skip("dynamo.common.gpu_utils not importable")

        backend = detect_gpu_backend()
        if not _has_rocm_smi() and not shutil.which("nvidia-smi"):
            assert backend == "none"
        elif _has_rocm_smi():
            assert backend == "amd", f"Expected 'amd', got '{backend}'"

    @pytest.mark.gpu_1
    def test_get_gpu_count(self):
        try:
            from dynamo.common.gpu_utils import detect_gpu_backend, get_gpu_count
        except ImportError:
            pytest.skip("dynamo.common.gpu_utils not importable")

        if detect_gpu_backend() == "none":
            pytest.skip("No GPU backend available")

        count = get_gpu_count()
        assert count > 0, f"Expected at least 1 GPU, got {count}"

    @pytest.mark.gpu_1
    def test_get_gpu_info(self):
        try:
            from dynamo.common.gpu_utils import detect_gpu_backend, get_gpu_info
        except ImportError:
            pytest.skip("dynamo.common.gpu_utils not importable")

        if detect_gpu_backend() == "none":
            pytest.skip("No GPU backend available")

        info = get_gpu_info(0)
        assert info is not None, "get_gpu_info(0) returned None"
        assert info.memory_total_mb > 0, "GPU reports 0 MB VRAM"
        assert info.name, "GPU name is empty"


class TestPytorchGpu:
    """Verify PyTorch can see GPUs via HIP."""

    @pytest.mark.gpu_1
    def test_torch_cuda_available(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        if not torch.cuda.is_available():
            pytest.skip("torch.cuda not available (no GPU)")

        count = torch.cuda.device_count()
        assert count > 0, "PyTorch sees 0 GPUs"

    @pytest.mark.gpu_1
    def test_torch_tensor_gpu_compute(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        if not torch.cuda.is_available():
            pytest.skip("torch.cuda not available (no GPU)")

        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        assert c.shape == (256, 256)
        assert c.device.type == "cuda"


class TestRixlLibraries:
    """Verify RIXL (ROCm port of NIXL) is installed."""

    def test_rixl_library_exists(self):
        rixl_prefix = os.environ.get("NIXL_PREFIX", "/opt/rocm/rixl")
        lib_paths = [
            os.path.join(rixl_prefix, "lib", "libnixl.so"),
            os.path.join(rixl_prefix, "lib", "x86_64-linux-gnu", "libnixl.so"),
        ]
        found = any(os.path.exists(p) for p in lib_paths)
        if not found:
            pytest.skip(
                f"RIXL library not found at any of: {lib_paths}. "
                "Set NIXL_PREFIX if installed elsewhere."
            )

    def test_rixl_python_importable(self):
        try:
            import nixl  # noqa: F401
        except ImportError:
            pytest.skip("nixl Python package not installed")
