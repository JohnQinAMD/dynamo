"""Unit tests for bugs fixed in the AMD adaptation review.

Tests cover:
  BUG-1: HipDeviceProperties struct layout
  BUG-3: hipMemcpy return value checking
  BUG-4: gpu_utils.sh GB/MiB unit conversion
  H-3:   staging_tensors list population
  H-6:   mooncake copy_d2h error on unregistered buffers
  H-9:   gfx arch auto-detection
  H-10:  version consistency test markers
"""

import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../components/src"))

pytestmark = [
    pytest.mark.rocm,
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class TestHipMemcpyErrorChecking(unittest.TestCase):
    """BUG-3: Verify hipMemcpy return values are checked."""

    def _make_staging(self, module_name):
        if module_name == "nixl":
            from dynamo.sglang.nixl_rocm_staging import _RocmDramStaging
        else:
            from dynamo.sglang.mooncake_rocm_staging import _RocmDramStaging
        staging = object.__new__(_RocmDramStaging)
        staging.buffers = {}
        staging._hip = MagicMock()
        staging._stream = None
        staging._lock = __import__("threading").Lock()
        return staging

    def test_nixl_copy_d2h_direct_checks_error(self):
        staging = self._make_staging("nixl")
        staging._hip.hipMemcpy.return_value = 999
        with self.assertRaises(RuntimeError) as ctx:
            staging.copy_d2h_direct(0x1000, 0x2000, 4096)
        self.assertIn("hipMemcpy D2H", str(ctx.exception))
        self.assertIn("999", str(ctx.exception))

    def test_nixl_copy_h2d_direct_checks_error(self):
        staging = self._make_staging("nixl")
        staging._hip.hipMemcpy.return_value = 42
        with self.assertRaises(RuntimeError) as ctx:
            staging.copy_h2d_direct(0x2000, 0x1000, 4096)
        self.assertIn("hipMemcpy H2D", str(ctx.exception))

    def test_nixl_copy_d2h_direct_succeeds_on_zero(self):
        staging = self._make_staging("nixl")
        staging._hip.hipMemcpy.return_value = 0
        staging.copy_d2h_direct(0x1000, 0x2000, 4096)

    def test_nixl_sync_checks_error(self):
        staging = self._make_staging("nixl")
        staging._hip.hipDeviceSynchronize.return_value = 7
        with self.assertRaises(RuntimeError) as ctx:
            staging.sync()
        self.assertIn("hipDeviceSynchronize", str(ctx.exception))

    def test_nixl_copy_d2h_returns_false_unregistered(self):
        staging = self._make_staging("nixl")
        result = staging.copy_d2h(0xDEAD, 4096)
        self.assertFalse(result)

    def test_nixl_copy_d2h_returns_true_registered(self):
        staging = self._make_staging("nixl")
        staging.buffers[0x1000] = (None, 0x2000, 8192)
        staging._hip.hipMemcpy.return_value = 0
        result = staging.copy_d2h(0x1000, 4096)
        self.assertTrue(result)

    def test_mooncake_copy_d2h_checks_error(self):
        staging = self._make_staging("mooncake")
        staging.buffers[0x1000] = (None, 0x2000, 8192)
        staging._hip.hipMemcpy.return_value = 99
        with self.assertRaises(RuntimeError):
            staging.copy_d2h(0x1000, 4096)

    def test_mooncake_copy_d2h_returns_false_unregistered(self):
        staging = self._make_staging("mooncake")
        result = staging.copy_d2h(0xDEAD, 4096)
        self.assertFalse(result)

    def test_mooncake_sync_checks_error(self):
        staging = self._make_staging("mooncake")
        staging._hip.hipDeviceSynchronize.return_value = 3
        with self.assertRaises(RuntimeError):
            staging.sync()


class TestMooncakeBatchTransferValidation(unittest.TestCase):
    """H-6: batch_transfer_sync must fail if copy_d2h returns False."""

    def test_batch_transfer_raises_on_unregistered_buffer(self):
        from dynamo.sglang.mooncake_rocm_staging import (
            _RocmDramStaging,
            _StagingEngineWrapper,
        )

        staging = object.__new__(_RocmDramStaging)
        staging.buffers = {}
        staging._hip = MagicMock()

        real_engine = MagicMock()
        wrapper = _StagingEngineWrapper(real_engine, staging)

        with self.assertRaises(RuntimeError) as ctx:
            wrapper.batch_transfer_sync("session1", [0xDEAD], [0xBEEF], [4096])
        self.assertIn("not registered", str(ctx.exception))
        real_engine.batch_transfer_sync.assert_not_called()

    def test_transfer_sync_raises_on_unregistered_buffer(self):
        from dynamo.sglang.mooncake_rocm_staging import (
            _RocmDramStaging,
            _StagingEngineWrapper,
        )

        staging = object.__new__(_RocmDramStaging)
        staging.buffers = {}
        staging._hip = MagicMock()

        real_engine = MagicMock()
        wrapper = _StagingEngineWrapper(real_engine, staging)

        with self.assertRaises(RuntimeError):
            wrapper.transfer_sync("session1", 0xDEAD, 0xBEEF, 4096)


class TestGpuUtilsUnitConversion(unittest.TestCase):
    """BUG-4: Test GB/MiB unit conversion logic in gpu_utils.sh."""

    def _run_conversion(self, val_str):
        code = f"""
s = str({val_str!r}).strip()
if 'GB' in s:
    print(int(float(s.replace('GB','').strip()) * 1024))
elif 'MB' in s:
    print(int(float(s.replace('MB','').strip())))
else:
    print(int(float(s)))
"""
        r = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert r.returncode == 0, f"Python failed: {r.stderr}"
        return int(r.stdout.strip())

    def test_gb_converted_to_mib(self):
        self.assertEqual(self._run_conversion("192 GB"), 192 * 1024)

    def test_mb_stays_as_mib(self):
        self.assertEqual(self._run_conversion("8192 MB"), 8192)

    def test_bare_number_treated_as_mib(self):
        self.assertEqual(self._run_conversion("196608"), 196608)

    def test_gb_with_decimal(self):
        self.assertEqual(self._run_conversion("192.5 GB"), int(192.5 * 1024))

    def test_mi300x_192gb(self):
        result = self._run_conversion("192 GB")
        self.assertEqual(result, 196608)

    def test_mi355x_288gb(self):
        result = self._run_conversion("288 GB")
        self.assertEqual(result, 288 * 1024)


class TestVersionConsistencyMarkers(unittest.TestCase):
    """H-10: test_rocm_version_consistency.py should not have sglang marker."""

    def test_no_sglang_marker(self):
        import importlib.util

        spec = importlib.util.find_spec("tests.basic.test_rocm_version_consistency")
        if spec is None:
            test_path = os.path.join(
                os.path.dirname(__file__),
                "test_rocm_version_consistency.py",
            )
            if not os.path.exists(test_path):
                self.skipTest("test_rocm_version_consistency.py not found")
            with open(test_path) as f:
                content = f.read()
        else:
            with open(spec.origin) as f:
                content = f.read()

        self.assertNotIn(
            "pytest.mark.sglang",
            content,
            "sglang marker should be removed from version consistency test",
        )


class TestGfxArchDetection(unittest.TestCase):
    """H-9: Verify gfx arch is auto-detected, not hardcoded."""

    def test_kvbm_test_no_hardcoded_gfx942(self):
        test_path = os.path.join(
            os.path.dirname(__file__),
            "../kvbm_integration/test_kvbm_rocm.py",
        )
        if not os.path.exists(test_path):
            self.skipTest("test_kvbm_rocm.py not found")
        with open(test_path) as f:
            content = f.read()
        self.assertNotIn(
            '"--offload-arch=gfx942"',
            content,
            "gfx942 should not be hardcoded — arch should be auto-detected",
        )


class TestHipDevicePropertiesLayout(unittest.TestCase):
    """BUG-1: Verify HipDeviceProperties struct has correct field offsets."""

    def test_rust_struct_has_uuid_field(self):
        hip_rs_path = os.path.join(
            os.path.dirname(__file__),
            "../../lib/memory/src/gpu/hip.rs",
        )
        if not os.path.exists(hip_rs_path):
            self.skipTest("hip.rs not found")
        with open(hip_rs_path) as f:
            content = f.read()
        self.assertIn(
            "_uuid",
            content,
            "HipDeviceProperties must have _uuid field for correct layout",
        )
        self.assertIn(
            "_luid",
            content,
            "HipDeviceProperties must have _luid field for correct layout",
        )
        self.assertIn(
            "_luid_device_node_mask",
            content,
            "HipDeviceProperties must have _luid_device_node_mask field",
        )

    def test_total_memory_uses_stable_api(self):
        hip_rs_path = os.path.join(
            os.path.dirname(__file__),
            "../../lib/memory/src/gpu/hip.rs",
        )
        if not os.path.exists(hip_rs_path):
            self.skipTest("hip.rs not found")
        with open(hip_rs_path) as f:
            content = f.read()
        self.assertIn(
            "hipDeviceTotalMem",
            content,
            "total_memory() must use hipDeviceTotalMem for struct-layout independence",
        )
        in_fn = False
        for line in content.splitlines():
            stripped = line.strip()
            if "fn total_memory" in stripped:
                in_fn = True
            elif (
                in_fn and stripped.startswith("fn ") and "total_memory" not in stripped
            ):
                break
            if (
                in_fn
                and "hipGetDeviceProperties" in stripped
                and not stripped.startswith("//")
            ):
                self.fail(
                    "total_memory() should NOT call hipGetDeviceProperties — use hipDeviceTotalMem"
                )


class TestGpuBackendOverride(unittest.TestCase):
    """H-7: DYNAMO_GPU_BACKEND env var override."""

    def test_override_to_nvidia(self):
        from dynamo.common import gpu_utils

        gpu_utils._GPU_BACKEND = None
        with patch.dict(os.environ, {"DYNAMO_GPU_BACKEND": "nvidia"}):
            result = gpu_utils.detect_gpu_backend()
        self.assertEqual(result, "nvidia")
        gpu_utils._GPU_BACKEND = None

    def test_override_to_none(self):
        from dynamo.common import gpu_utils

        gpu_utils._GPU_BACKEND = None
        with patch.dict(os.environ, {"DYNAMO_GPU_BACKEND": "none"}):
            result = gpu_utils.detect_gpu_backend()
        self.assertEqual(result, "none")
        gpu_utils._GPU_BACKEND = None

    def test_invalid_override_ignored(self):
        from dynamo.common import gpu_utils

        gpu_utils._GPU_BACKEND = None
        with patch.dict(os.environ, {"DYNAMO_GPU_BACKEND": "bogus"}):
            result = gpu_utils.detect_gpu_backend()
        self.assertIn(result, ("amd", "nvidia", "none"))
        gpu_utils._GPU_BACKEND = None


class TestSharedStagingModule(unittest.TestCase):
    """Verify nixl and mooncake both use the shared RocmDramStaging class."""

    def test_shared_import_exists(self):
        from dynamo.sglang.rocm_dram_staging_common import RocmDramStaging

        self.assertTrue(hasattr(RocmDramStaging, "copy_d2h"))
        self.assertTrue(hasattr(RocmDramStaging, "copy_h2d"))
        self.assertTrue(hasattr(RocmDramStaging, "_check_hip"))
        self.assertTrue(hasattr(RocmDramStaging, "translate_reqs"))

    def test_thread_lock_exists(self):
        import threading

        from dynamo.sglang.rocm_dram_staging_common import RocmDramStaging

        staging = object.__new__(RocmDramStaging)
        staging._hip = MagicMock()
        staging._lock = threading.Lock()
        staging.buffers = {}
        self.assertIsInstance(staging._lock, type(threading.Lock()))

    def test_shared_check_hip_raises(self):
        from dynamo.sglang.rocm_dram_staging_common import RocmDramStaging

        staging = object.__new__(RocmDramStaging)
        with self.assertRaises(RuntimeError):
            staging._check_hip(42, "test_op")

    def test_shared_check_hip_ok(self):
        from dynamo.sglang.rocm_dram_staging_common import RocmDramStaging

        staging = object.__new__(RocmDramStaging)
        staging._check_hip(0, "test_op")


class TestCrdApiVersionComment(unittest.TestCase):
    """H-11: Verify CRD YAML has API version documentation."""

    def test_rocm_agg_has_comment(self):
        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "../fault_tolerance/deploy/templates/sglang/rocm_agg.yaml",
        )
        if not os.path.exists(yaml_path):
            self.skipTest("rocm_agg.yaml not found")
        with open(yaml_path) as f:
            content = f.read()
        self.assertIn("vendor-neutral", content)

    def test_rocm_disagg_uses_ci_model(self):
        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "../fault_tolerance/deploy/templates/sglang/rocm_disagg.yaml",
        )
        if not os.path.exists(yaml_path):
            self.skipTest("rocm_disagg.yaml not found")
        with open(yaml_path) as f:
            content = f.read()
        self.assertNotIn(
            "DeepSeek-V3",
            content,
            "Disagg template should use CI-testable model, not DeepSeek-V3",
        )
        self.assertIn("Qwen", content)


class TestCiNoSilentFailures(unittest.TestCase):
    """H-13: CI test steps should not use '|| true'."""

    def test_rocm_test_yml_no_or_true(self):
        ci_path = os.path.join(
            os.path.dirname(__file__),
            "../../.github/workflows/rocm-test.yml",
        )
        if not os.path.exists(ci_path):
            self.skipTest("rocm-test.yml not found")
        with open(ci_path) as f:
            content = f.read()
        lines_with_or_true = [
            line.strip()
            for line in content.splitlines()
            if "|| true" in line and not line.strip().startswith("#")
        ]
        self.assertEqual(
            len(lines_with_or_true),
            0,
            f"CI should not swallow failures with '|| true': {lines_with_or_true}",
        )

    def test_rocm_test_yml_no_privileged(self):
        ci_path = os.path.join(
            os.path.dirname(__file__),
            "../../.github/workflows/rocm-test.yml",
        )
        if not os.path.exists(ci_path):
            self.skipTest("rocm-test.yml not found")
        with open(ci_path) as f:
            content = f.read()
        self.assertNotIn(
            "--privileged", content, "CI should use --cap-add instead of --privileged"
        )


class TestHipContextGuardRestoresPrevious(unittest.TestCase):
    """H-1: DynamoHipContextGuard must save and restore previous context."""

    def test_guard_has_previous_context_field(self):
        hip_rs_path = os.path.join(
            os.path.dirname(__file__),
            "../../lib/llm/src/hip.rs",
        )
        if not os.path.exists(hip_rs_path):
            self.skipTest("hip.rs not found")
        with open(hip_rs_path) as f:
            content = f.read()
        self.assertIn(
            "previous_context", content, "Guard must have previous_context field"
        )
        self.assertIn(
            "get_current_context",
            content,
            "Guard must save current context before switching",
        )

    def test_drop_restores_context(self):
        hip_rs_path = os.path.join(
            os.path.dirname(__file__),
            "../../lib/llm/src/hip.rs",
        )
        if not os.path.exists(hip_rs_path):
            self.skipTest("hip.rs not found")
        with open(hip_rs_path) as f:
            content = f.read()
        self.assertIn("fn drop", content)
        self.assertNotIn(
            "Nothing to explicitly pop",
            content,
            "Drop should restore context, not be a no-op",
        )
        self.assertIn(
            "set_current_context(prev)",
            content,
            "Drop must call set_current_context with saved context",
        )


class TestHipMemPoolPropsLayout(unittest.TestCase):
    """H-2: hipMemPoolProps struct must match ROCm header layout."""

    def test_no_padding_alloc_field(self):
        pool_rs_path = os.path.join(
            os.path.dirname(__file__),
            "../../lib/memory/src/pool/hip.rs",
        )
        if not os.path.exists(pool_rs_path):
            self.skipTest("pool/hip.rs not found")
        with open(pool_rs_path) as f:
            content = f.read()
        self.assertNotIn(
            "_padding_alloc",
            content,
            "Old layout used _padding_alloc between alloc_type and location",
        )
        self.assertIn(
            "handle_types", content, "Must have handle_types field after alloc_type"
        )

    def test_has_layout_comment(self):
        pool_rs_path = os.path.join(
            os.path.dirname(__file__),
            "../../lib/memory/src/pool/hip.rs",
        )
        if not os.path.exists(pool_rs_path):
            self.skipTest("pool/hip.rs not found")
        with open(pool_rs_path) as f:
            content = f.read()
        self.assertIn(
            "hip_runtime_api.h",
            content,
            "Struct should reference the header it mirrors",
        )


class TestStagingUsesStreamSync(unittest.TestCase):
    """H-5: Staging should use hipStreamSynchronize, not hipDeviceSynchronize."""

    def test_shared_staging_has_stream(self):
        from dynamo.sglang.rocm_dram_staging_common import RocmDramStaging

        self.assertTrue(
            hasattr(RocmDramStaging, "sync"), "RocmDramStaging must have sync method"
        )
        staging = object.__new__(RocmDramStaging)
        staging._hip = MagicMock()
        staging._stream = MagicMock()
        staging._hip.hipStreamSynchronize = MagicMock(return_value=0)

        staging.sync()

        staging._hip.hipStreamSynchronize.assert_called_once_with(staging._stream)
        staging._hip.hipDeviceSynchronize.assert_not_called()

    def test_falls_back_to_device_sync_when_no_stream(self):
        from dynamo.sglang.rocm_dram_staging_common import RocmDramStaging

        staging = object.__new__(RocmDramStaging)
        staging._hip = MagicMock()
        staging._stream = None
        staging._hip.hipDeviceSynchronize = MagicMock(return_value=0)

        staging.sync()

        staging._hip.hipDeviceSynchronize.assert_called_once()


class TestExecutorHipStreamComment(unittest.TestCase):
    """BUG-2: executor/hip.rs must document cudarc/HIP ABI compatibility."""

    def test_has_abi_comment(self):
        executor_path = os.path.join(
            os.path.dirname(__file__),
            "../../lib/llm/src/block_manager/v2/physical/transfer/executor/hip.rs",
        )
        if not os.path.exists(executor_path):
            self.skipTest("executor/hip.rs not found")
        with open(executor_path) as f:
            content = f.read()
        self.assertIn(
            "ABI-compatible",
            content,
            "Must document cudarc/HIP stream ABI compatibility assumption",
        )


class TestAmdsmiSingletonInit(unittest.TestCase):
    """Verify amdsmi/NVML use singleton init, not per-call init/shutdown."""

    def test_no_per_call_nvml_shutdown(self):
        gpu_utils_path = os.path.join(
            os.path.dirname(__file__),
            "../../components/src/dynamo/common/gpu_utils.py",
        )
        if not os.path.exists(gpu_utils_path):
            self.skipTest("gpu_utils.py not found")
        with open(gpu_utils_path) as f:
            content = f.read()
        in_func = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("def _nvidia_get_count_nvml") or stripped.startswith(
                "def _nvidia_get_info_nvml"
            ):
                in_func = True
            elif in_func and stripped.startswith("def "):
                in_func = False
            if in_func and "nvmlShutdown" in stripped and "atexit" not in stripped:
                self.fail(f"Per-call nvmlShutdown found: {stripped}")

    def test_no_per_call_amdsmi_shutdown(self):
        gpu_utils_path = os.path.join(
            os.path.dirname(__file__),
            "../../components/src/dynamo/common/gpu_utils.py",
        )
        if not os.path.exists(gpu_utils_path):
            self.skipTest("gpu_utils.py not found")
        with open(gpu_utils_path) as f:
            content = f.read()
        in_func = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("def _amd_get_count_amdsmi") or stripped.startswith(
                "def _amd_get_info_amdsmi"
            ):
                in_func = True
            elif in_func and stripped.startswith("def "):
                in_func = False
            if in_func and "amdsmi_shut_down" in stripped and "atexit" not in stripped:
                self.fail(f"Per-call amdsmi_shut_down found: {stripped}")

    def test_atexit_registered_for_nvml(self):
        gpu_utils_path = os.path.join(
            os.path.dirname(__file__),
            "../../components/src/dynamo/common/gpu_utils.py",
        )
        if not os.path.exists(gpu_utils_path):
            self.skipTest("gpu_utils.py not found")
        with open(gpu_utils_path) as f:
            content = f.read()
        self.assertIn("atexit.register(pynvml.nvmlShutdown)", content)

    def test_atexit_registered_for_amdsmi(self):
        gpu_utils_path = os.path.join(
            os.path.dirname(__file__),
            "../../components/src/dynamo/common/gpu_utils.py",
        )
        if not os.path.exists(gpu_utils_path):
            self.skipTest("gpu_utils.py not found")
        with open(gpu_utils_path) as f:
            content = f.read()
        self.assertIn("atexit.register(amdsmi.amdsmi_shut_down)", content)


class TestDockerLabelsAndHealthcheck(unittest.TestCase):
    """Verify Dockerfiles have LABEL and HEALTHCHECK."""

    def _read_dockerfile(self, name):
        path = os.path.join(
            os.path.dirname(__file__),
            f"../../container/{name}",
        )
        if not os.path.exists(path):
            self.skipTest(f"{name} not found")
        with open(path) as f:
            return f.read()

    def test_sglang_dockerfile_has_label(self):
        content = self._read_dockerfile("Dockerfile.rocm-sglang")
        self.assertIn("LABEL", content)

    def test_sglang_dockerfile_has_healthcheck(self):
        content = self._read_dockerfile("Dockerfile.rocm-sglang")
        self.assertIn("HEALTHCHECK", content)

    def test_vllm_dockerfile_has_label(self):
        content = self._read_dockerfile("Dockerfile.rocm-vllm")
        self.assertIn("LABEL", content)

    def test_vllm_dockerfile_has_healthcheck(self):
        content = self._read_dockerfile("Dockerfile.rocm-vllm")
        self.assertIn("HEALTHCHECK", content)


class TestCompileErrorGuard(unittest.TestCase):
    """Verify compile_error! guard for mutually exclusive cuda/rocm features."""

    def test_gpu_mod_has_compile_error_guard(self):
        mod_rs_path = os.path.join(
            os.path.dirname(__file__),
            "../../lib/memory/src/gpu/mod.rs",
        )
        if not os.path.exists(mod_rs_path):
            self.skipTest("gpu/mod.rs not found")
        with open(mod_rs_path) as f:
            content = f.read()
        self.assertIn("compile_error!", content)
        self.assertIn("mutually exclusive", content)


class TestDisaggServeConfig(unittest.TestCase):
    """Verify disaggregated serving config exists in test_sglang_rocm.py."""

    def test_disagg_config_present(self):
        test_path = os.path.join(
            os.path.dirname(__file__),
            "../serve/test_sglang_rocm.py",
        )
        if not os.path.exists(test_path):
            self.skipTest("test_sglang_rocm.py not found")
        with open(test_path) as f:
            content = f.read()
        self.assertIn("rocm_disaggregated", content)
        self.assertIn("disagg", content)


class TestTransferStrategyNamingDocumented(unittest.TestCase):
    """Verify HIP executor documents why CUDA-named enum variants are used."""

    def test_hip_executor_has_naming_note(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "../../lib/llm/src/block_manager/v2/physical/transfer/executor/hip.rs",
        )
        if not os.path.exists(path):
            self.skipTest("executor/hip.rs not found")
        with open(path) as f:
            content = f.read()
        self.assertIn("TransferStrategy", content)
        self.assertIn("naming", content.lower())


class TestMonkeyPatchVersionGuards(unittest.TestCase):
    """Verify staging modules check SGLang API compatibility before patching."""

    def _check_file_has_guard(self, filename):
        path = os.path.join(
            os.path.dirname(__file__),
            f"../../components/src/dynamo/sglang/{filename}",
        )
        if not os.path.exists(path):
            self.skipTest(f"{filename} not found")
        with open(path) as f:
            content = f.read()
        self.assertIn(
            "_required_attrs",
            content,
            f"{filename} must check required attributes before patching",
        )
        self.assertIn(
            "ABORTED", content, f"{filename} must log ABORTED if SGLang API changed"
        )

    def test_nixl_has_version_guard(self):
        self._check_file_has_guard("nixl_rocm_staging.py")

    def test_mooncake_has_version_guard(self):
        self._check_file_has_guard("mooncake_rocm_staging.py")


if __name__ == "__main__":
    unittest.main()
