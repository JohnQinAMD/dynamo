"""
Test ROCm DRAM staging monkey-patch for SGLang nixl connector.

Tests run in 3 tiers:
  1. Pure-Python unit tests (no GPU) — validate _RocmDramStaging logic
  2. GPU tests — validate hipMemcpy D2H / H2D via staging pool
  3. Integration test — validate monkey-patch hooks NixlKVManager correctly

Run:
    python3 test_nixl_rocm_staging.py
"""

import ctypes
import os
import sys
import unittest

import pytest

# Ensure dynamo components are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../components/src"))

pytestmark = [
    pytest.mark.rocm,
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class TestRocmDramStagingUnit(unittest.TestCase):
    """Pure-Python tests for _RocmDramStaging (no GPU required)."""

    def _make_staging(self):
        """Create a staging instance with mock buffers (no GPU needed)."""
        import torch

        from dynamo.sglang.nixl_rocm_staging import _RocmDramStaging

        staging = object.__new__(_RocmDramStaging)
        staging.buffers = {}
        staging._hip = None

        # Regular CPU tensors (no pin_memory — we only test address math)
        buf_a = torch.empty(4096, dtype=torch.uint8, device="cpu")
        buf_b = torch.empty(2048, dtype=torch.uint8, device="cpu")
        # Fake GPU base addresses
        gpu_a, gpu_b = 0x7F0000000000, 0x7F0000100000
        staging.buffers[gpu_a] = (buf_a, buf_a.data_ptr(), 4096)
        staging.buffers[gpu_b] = (buf_b, buf_b.data_ptr(), 2048)
        return staging, gpu_a, gpu_b, buf_a, buf_b

    def test_translate_base(self):
        staging, gpu_a, gpu_b, buf_a, buf_b = self._make_staging()
        self.assertEqual(staging.translate_base(gpu_a), buf_a.data_ptr())
        self.assertEqual(staging.translate_base(gpu_b), buf_b.data_ptr())
        # Unknown GPU ptr passes through
        self.assertEqual(staging.translate_base(0xDEADBEEF), 0xDEADBEEF)

    def test_translate_ptrs(self):
        staging, gpu_a, gpu_b, buf_a, buf_b = self._make_staging()
        result = staging.translate_ptrs([gpu_a, gpu_b, 0x1234])
        self.assertEqual(result, [buf_a.data_ptr(), buf_b.data_ptr(), 0x1234])

    def test_translate_reqs_numpy(self):
        import numpy as np

        staging, gpu_a, gpu_b, buf_a, buf_b = self._make_staging()
        reqs = np.array(
            [
                [gpu_a + 100, 256, 3],  # in buf_a → should translate
                [gpu_b + 50, 128, 3],  # in buf_b → should translate
                [0xAAAA, 64, 5],  # not in any buffer → pass through
            ],
            dtype=np.int64,
        )
        translated = staging.translate_reqs(reqs)
        # buf_a address
        self.assertEqual(translated[0, 0], buf_a.data_ptr() + 100)
        self.assertEqual(translated[0, 2], 0)  # CPU device
        # buf_b address
        self.assertEqual(translated[1, 0], buf_b.data_ptr() + 50)
        self.assertEqual(translated[1, 2], 0)
        # passthrough
        self.assertEqual(translated[2, 0], 0xAAAA)
        self.assertEqual(translated[2, 2], 5)

    def test_translate_reqs_tuples(self):
        staging, gpu_a, gpu_b, buf_a, buf_b = self._make_staging()
        reqs = [
            (gpu_a + 200, 512, 2),
            (0xBBBB, 64, 7),
        ]
        translated = staging.translate_reqs(reqs)
        self.assertEqual(translated[0], (buf_a.data_ptr() + 200, 512, 0))
        self.assertEqual(translated[1], (0xBBBB, 64, 7))

    def test_translate_reqs_empty_numpy(self):
        import numpy as np

        staging, *_ = self._make_staging()
        empty = np.empty((0, 3), dtype=np.int64)
        result = staging.translate_reqs(empty)
        self.assertEqual(result.size, 0)

    def test_translate_reqs_4tuple(self):
        staging, gpu_a, _, buf_a, _ = self._make_staging()
        reqs = [(gpu_a, 1024, 1, "tag")]
        translated = staging.translate_reqs(reqs)
        self.assertEqual(translated[0], (buf_a.data_ptr(), 1024, 0, "tag"))


class TestRocmDramStagingGPU(unittest.TestCase):
    """GPU tests — require ROCm + torch with HIP."""

    @classmethod
    def setUpClass(cls):
        import torch

        if not (hasattr(torch.version, "hip") and torch.version.hip):
            raise unittest.SkipTest("Not a ROCm build of PyTorch")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No GPU available")

    def test_create_staging_buffer(self):
        import torch

        from dynamo.sglang.nixl_rocm_staging import _RocmDramStaging

        staging = _RocmDramStaging()

        gpu_tensor = torch.randn(1024, device="cuda:0", dtype=torch.float32)
        gpu_ptr = gpu_tensor.data_ptr()
        size = gpu_tensor.nelement() * gpu_tensor.element_size()

        host_ptr = staging.create(gpu_ptr, size)
        self.assertNotEqual(host_ptr, gpu_ptr)
        self.assertIn(gpu_ptr, staging.buffers)
        _, stored_host, stored_size = staging.buffers[gpu_ptr]
        self.assertEqual(stored_host, host_ptr)
        self.assertEqual(stored_size, size)

    def test_copy_d2h_and_h2d_roundtrip(self):
        import torch

        from dynamo.sglang.nixl_rocm_staging import _RocmDramStaging

        staging = _RocmDramStaging()

        # Create GPU tensor with known data
        original = torch.arange(256, device="cuda:0", dtype=torch.float32)
        gpu_ptr = original.data_ptr()
        nbytes = original.nelement() * original.element_size()

        staging.create(gpu_ptr, nbytes)

        # GPU → DRAM
        staging.copy_d2h(gpu_ptr, nbytes)
        staging.sync()

        # Verify host buffer has the data
        _, host_base, _ = staging.buffers[gpu_ptr]
        host_array = (ctypes.c_float * 256).from_address(host_base)
        for i in range(10):
            self.assertAlmostEqual(host_array[i], float(i), places=5)

        # Modify GPU tensor
        original.fill_(999.0)
        torch.cuda.synchronize()

        # DRAM → GPU (restores original data)
        staging.copy_h2d(gpu_ptr, nbytes)
        staging.sync()

        # Verify GPU has the original data back
        result = original.cpu()
        for i in range(10):
            self.assertAlmostEqual(result[i].item(), float(i), places=5)

    def test_copy_d2h_skips_unknown_addr(self):
        from dynamo.sglang.nixl_rocm_staging import _RocmDramStaging

        staging = _RocmDramStaging()
        # Should not crash — silently skips unknown addresses
        staging.copy_d2h(0xDEADBEEF, 64)
        staging.copy_h2d(0xDEADBEEF, 64)


class TestMonkeyPatch(unittest.TestCase):
    """Test that patch_nixl_for_rocm hooks the right methods."""

    def test_patch_is_idempotent(self):
        import dynamo.sglang.nixl_rocm_staging as mod

        # Reset patch state
        mod._PATCHED = False
        orig_enabled = mod._STAGING_ENABLED

        # If sglang is not importable, patch should warn and return
        mod._STAGING_ENABLED = True
        try:
            mod.patch_nixl_for_rocm()
        except Exception:
            pass  # sglang may not be installed

        # Second call should be no-op (idempotent)
        mod.patch_nixl_for_rocm()
        mod._STAGING_ENABLED = orig_enabled

    def test_staging_disabled_when_not_rocm(self):
        import dynamo.sglang.nixl_rocm_staging as mod

        mod._PATCHED = False
        orig = mod._STAGING_ENABLED
        mod._STAGING_ENABLED = False
        mod.patch_nixl_for_rocm()
        # Should have done nothing
        mod._STAGING_ENABLED = orig


class TestMonkeyPatchIntegration(unittest.TestCase):
    """Integration test — requires sglang importable."""

    @classmethod
    def setUpClass(cls):
        try:
            from sglang.srt.disaggregation.nixl.conn import NixlKVManager  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("sglang not installed")

    def test_methods_are_patched(self):
        from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVReceiver

        import dynamo.sglang.nixl_rocm_staging as mod

        mod._PATCHED = False
        mod._STAGING_ENABLED = True
        mod.patch_nixl_for_rocm()

        # Verify register_buffer_to_engine was replaced
        # (original has "VRAM" in source, patched doesn't)
        import inspect

        src = inspect.getsource(NixlKVManager.register_buffer_to_engine)
        self.assertIn("DRAM", src)
        self.assertNotIn('"VRAM"', src)

        # Verify receiver poll was wrapped
        poll_src = inspect.getsource(NixlKVReceiver.poll)
        self.assertIn("_staging_kv_indices", poll_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
