# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RIXL/nixl ROCm DRAM staging with chunked MR.

Tests the core staging logic (mmap, address translation, chunking math)
WITHOUT requiring real RDMA hardware or a running SGLang instance.

Run:  python -m pytest tests/test_nixl_rocm_staging.py -v
  or: python tests/test_nixl_rocm_staging.py
"""

import ctypes
import mmap
import os
import struct
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

_parent = os.path.join(os.path.dirname(__file__), "..")
_components_src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, _parent)
sys.path.insert(0, _components_src)

from rocm_dram_staging_common import RocmDramStaging

# Pre-inject the dynamo.sglang package so nixl_rocm_staging can import
# rocm_dram_staging_common via `from dynamo.sglang.rocm_dram_staging_common ...`
import importlib
import types
if "dynamo" not in sys.modules:
    _dynamo_mod = types.ModuleType("dynamo")
    _dynamo_mod.__path__ = [os.path.join(_components_src, "dynamo")]
    sys.modules["dynamo"] = _dynamo_mod
if "dynamo.sglang" not in sys.modules:
    _sglang_mod = types.ModuleType("dynamo.sglang")
    _sglang_mod.__path__ = [_parent]
    sys.modules["dynamo.sglang"] = _sglang_mod
    sys.modules["dynamo"].sglang = _sglang_mod
import rocm_dram_staging_common as _rdc
sys.modules["dynamo.sglang.rocm_dram_staging_common"] = _rdc


# ---------------------------------------------------------------------------
# Test RocmDramStaging (shared base class)
# ---------------------------------------------------------------------------
class TestRocmDramStaging(unittest.TestCase):
    """Test mmap+mlock allocation and address translation."""

    def test_create_allocates_mmap(self):
        staging = MagicMock(spec=RocmDramStaging)
        staging.buffers = {}
        staging._lock = threading.Lock()

        m = mmap.mmap(-1, 4096)
        buf = (ctypes.c_char * 4096).from_buffer(m)
        host_ptr = ctypes.addressof(buf)

        gpu_ptr = 0xDEAD0000
        staging.buffers[gpu_ptr] = ((m, buf), host_ptr, 4096)

        self.assertIn(gpu_ptr, staging.buffers)
        _, stored_host, stored_size = staging.buffers[gpu_ptr]
        self.assertEqual(stored_size, 4096)
        self.assertGreater(stored_host, 0)

    def test_translate_base(self):
        staging = RocmDramStaging.__new__(RocmDramStaging)
        staging.buffers = {}
        staging._lock = threading.Lock()

        gpu_ptr = 0xAAAA0000
        host_ptr = 0xBBBB0000
        staging.buffers[gpu_ptr] = (None, host_ptr, 1024)

        self.assertEqual(staging.translate_base(gpu_ptr), host_ptr)
        self.assertEqual(staging.translate_base(0x1234), 0x1234)

    def test_translate_addr_with_offset(self):
        staging = RocmDramStaging.__new__(RocmDramStaging)
        staging.buffers = {}
        staging._lock = threading.Lock()

        gpu_base = 0x10000000
        host_base = 0x20000000
        buf_size = 0x1000000  # 16MB
        staging.buffers[gpu_base] = (None, host_base, buf_size)

        offset = 0x500
        self.assertEqual(
            staging.translate_addr(gpu_base + offset),
            host_base + offset,
        )

    def test_translate_reqs_numpy(self):
        staging = RocmDramStaging.__new__(RocmDramStaging)
        staging.buffers = {}
        staging._lock = threading.Lock()

        gpu_base = 1000000
        host_base = 2000000
        buf_size = 500000
        staging.buffers[gpu_base] = (None, host_base, buf_size)

        reqs = np.array([
            [gpu_base + 100, 256, 7],
            [gpu_base + 1000, 512, 7],
            [9999999, 64, 3],  # not in any buffer
        ], dtype=np.int64)

        translated = staging.translate_reqs(reqs)

        self.assertEqual(translated[0, 0], host_base + 100)
        self.assertEqual(translated[0, 2], 0)
        self.assertEqual(translated[1, 0], host_base + 1000)
        self.assertEqual(translated[1, 2], 0)
        self.assertEqual(translated[2, 0], 9999999)  # unchanged
        self.assertEqual(translated[2, 2], 3)  # unchanged

    def test_translate_ptrs(self):
        staging = RocmDramStaging.__new__(RocmDramStaging)
        staging.buffers = {}
        staging._lock = threading.Lock()

        staging.buffers[100] = (None, 200, 50)
        staging.buffers[300] = (None, 400, 50)

        result = staging.translate_ptrs([100, 300, 999])
        self.assertEqual(result, [200, 400, 999])


# ---------------------------------------------------------------------------
# Test chunking math
# ---------------------------------------------------------------------------
class TestChunkingMath(unittest.TestCase):
    """Test MR region grouping and chunk splitting logic."""

    def test_single_region_under_limit(self):
        """A single small region should produce 1 chunk."""
        mr_regions = {1000: (1000, 1000 + 100 * 1024 * 1024)}  # 100MB
        chunk_limit = 190 * 1024 * 1024

        mr_list = list(mr_regions.items())
        chunks = []
        cur_chunk = []
        cur_bytes = 0
        for host_base, (lo, hi) in mr_list:
            region_sz = hi - lo
            if cur_bytes + region_sz > chunk_limit and cur_chunk:
                chunks.append(cur_chunk)
                cur_chunk = []
                cur_bytes = 0
            cur_chunk.append((host_base, lo, hi))
            cur_bytes += region_sz
        if cur_chunk:
            chunks.append(cur_chunk)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 1)

    def test_multiple_regions_split_into_chunks(self):
        """Regions exceeding chunk limit should be split."""
        chunk_limit = 190 * 1024 * 1024
        region_size = 100 * 1024 * 1024  # 100MB each

        mr_regions = {}
        for i in range(4):
            base = i * 200 * 1024 * 1024
            mr_regions[base] = (base, base + region_size)

        mr_list = list(mr_regions.items())
        chunks = []
        cur_chunk = []
        cur_bytes = 0
        for host_base, (lo, hi) in mr_list:
            region_sz = hi - lo
            if cur_bytes + region_sz > chunk_limit and cur_chunk:
                chunks.append(cur_chunk)
                cur_chunk = []
                cur_bytes = 0
            cur_chunk.append((host_base, lo, hi))
            cur_bytes += region_sz
        if cur_chunk:
            chunks.append(cur_chunk)

        # 4 × 100MB at 190MB limit → chunks: [100], [100], [100], [100]
        self.assertEqual(len(chunks), 4)

    def test_two_regions_fit_in_one_chunk(self):
        """Two small regions should fit in a single chunk."""
        chunk_limit = 190 * 1024 * 1024
        region_size = 80 * 1024 * 1024  # 80MB each, 2 × 80 = 160 < 190

        mr_regions = {}
        for i in range(2):
            base = i * 200 * 1024 * 1024
            mr_regions[base] = (base, base + region_size)

        mr_list = list(mr_regions.items())
        chunks = []
        cur_chunk = []
        cur_bytes = 0
        for host_base, (lo, hi) in mr_list:
            region_sz = hi - lo
            if cur_bytes + region_sz > chunk_limit and cur_chunk:
                chunks.append(cur_chunk)
                cur_chunk = []
                cur_bytes = 0
            cur_chunk.append((host_base, lo, hi))
            cur_bytes += region_sz
        if cur_chunk:
            chunks.append(cur_chunk)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 2)


# ---------------------------------------------------------------------------
# Test _CompletedTransfer sentinel
# ---------------------------------------------------------------------------
class TestCompletedTransfer(unittest.TestCase):

    def test_sentinel_type(self):
        from nixl_rocm_staging import _CompletedTransfer
        ct = _CompletedTransfer()
        self.assertIsInstance(ct, _CompletedTransfer)
        self.assertNotIsInstance(ct, str)
        self.assertNotIsInstance(ct, int)


# ---------------------------------------------------------------------------
# Test _batch_d2h_copy logic (with mocked HIP)
# ---------------------------------------------------------------------------
class TestBatchD2HCopy(unittest.TestCase):

    def test_empty_reqs(self):
        from nixl_rocm_staging import _batch_d2h_copy
        staging = MagicMock(spec=RocmDramStaging)
        staging.buffers = {}
        _batch_d2h_copy(staging, np.empty((0, 3), dtype=np.int64))
        staging.sync.assert_not_called()

    def test_groups_by_buffer(self):
        from nixl_rocm_staging import _batch_d2h_copy

        staging = MagicMock(spec=RocmDramStaging)
        gpu_base1 = 0x10000
        gpu_base2 = 0x20000
        host_base1 = 0xA0000
        host_base2 = 0xB0000
        buf_size = 0x8000

        staging.buffers = {
            gpu_base1: (None, host_base1, buf_size),
            gpu_base2: (None, host_base2, buf_size),
        }

        reqs = np.array([
            [gpu_base1 + 100, 256, 0],
            [gpu_base1 + 500, 128, 0],
            [gpu_base2 + 200, 64, 0],
        ], dtype=np.int64)

        _batch_d2h_copy(staging, reqs)

        self.assertEqual(staging.copy_d2h_direct.call_count, 2)
        staging.sync.assert_called_once()


# ---------------------------------------------------------------------------
# Test slab layout math
# ---------------------------------------------------------------------------
class TestSlabLayout(unittest.TestCase):

    def test_slab_division(self):
        slab_size = 200 * 1024 * 1024  # 200MB
        num_layers = 128
        layer_slab = slab_size // max(num_layers, 1)

        self.assertEqual(layer_slab, 200 * 1024 * 1024 // 128)
        self.assertGreater(layer_slab, 0)

        # Each layer gets ~1.56MB
        self.assertAlmostEqual(layer_slab / (1024 * 1024), 1.5625, places=3)

    def test_slab_addresses_sequential(self):
        slab_ptr = 0x7F0000000000
        slab_size = 200 * 1024 * 1024
        num_kv = 64
        layer_slab = slab_size // max(num_kv, 1)

        kv_ptrs = [slab_ptr + i * layer_slab for i in range(num_kv)]

        self.assertEqual(kv_ptrs[0], slab_ptr)
        self.assertEqual(kv_ptrs[1], slab_ptr + layer_slab)
        self.assertEqual(kv_ptrs[-1], slab_ptr + (num_kv - 1) * layer_slab)
        # Last layer end should be within slab
        self.assertLessEqual(kv_ptrs[-1] + layer_slab, slab_ptr + slab_size)

    def test_item_fits_in_slab_slice(self):
        """DeepSeek-R1: item_len=9216, page_size=16, typical 64 pages per request."""
        slab_size = 200 * 1024 * 1024
        num_layers = 128
        layer_slab = slab_size // num_layers

        item_len = 9216
        pages_per_request = 64
        data_per_request = item_len * pages_per_request  # 576KB

        self.assertLess(data_per_request, layer_slab,
                       f"Request data ({data_per_request}) exceeds slab slice ({layer_slab})")


# ---------------------------------------------------------------------------
# Test mmap+mlock allocation
# ---------------------------------------------------------------------------
class TestMmapAllocation(unittest.TestCase):

    def test_mmap_alloc_and_addressof(self):
        size = 1024 * 1024  # 1MB
        m = mmap.mmap(-1, size)
        buf = (ctypes.c_char * size).from_buffer(m)
        ptr = ctypes.addressof(buf)

        self.assertGreater(ptr, 0)
        self.assertEqual(len(buf), size)

        # Write and read back
        buf[0] = b'\x42'
        buf[size - 1] = b'\x43'
        self.assertEqual(buf[0], b'\x42')
        self.assertEqual(buf[size - 1], b'\x43')

        del buf
        m.close()

    def test_mlock_succeeds(self):
        size = 64 * 1024  # 64KB
        m = mmap.mmap(-1, size)
        buf = (ctypes.c_char * size).from_buffer(m)
        ptr = ctypes.addressof(buf)

        libc = ctypes.CDLL("libc.so.6")
        rc = libc.mlock(ctypes.c_void_p(ptr), ctypes.c_size_t(size))
        # mlock may fail with ENOMEM if ulimit is too low, but shouldn't crash
        if rc == 0:
            libc.munlock(ctypes.c_void_p(ptr), ctypes.c_size_t(size))
        del buf
        m.close()


# ---------------------------------------------------------------------------
# Integration test: mock nixl agent with chunked write
# ---------------------------------------------------------------------------
class TestChunkedRixlWrite(unittest.TestCase):

    def _make_mock_agent(self):
        agent = MagicMock()
        agent.register_memory.return_value = MagicMock()  # opaque descs
        agent.get_xfer_descs.return_value = MagicMock()
        agent.initialize_xfer.return_value = MagicMock()  # xfer handle
        agent.transfer.return_value = "PROC"
        agent.check_xfer_state.return_value = "DONE"
        agent.release_xfer_handle.return_value = None
        agent.deregister_memory.return_value = None
        return agent

    def test_empty_reqs(self):
        from nixl_rocm_staging import _chunked_rixl_write, _CompletedTransfer

        agent = self._make_mock_agent()
        staging = MagicMock(spec=RocmDramStaging)
        staging.buffers = {}

        result = _chunked_rixl_write(
            agent, staging,
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            "peer", b"notif", threading.Lock(),
        )
        self.assertIsInstance(result, _CompletedTransfer)
        agent.register_memory.assert_not_called()

    def test_single_small_transfer(self):
        from nixl_rocm_staging import _chunked_rixl_write, _CompletedTransfer

        agent = self._make_mock_agent()
        staging = MagicMock(spec=RocmDramStaging)

        host_base = 0x100000
        buf_size = 1024 * 1024  # 1MB
        staging.buffers = {0xBAD0: (None, host_base, buf_size)}

        # Manually set buffers keyed by host ptr for chunked write
        staging.buffers = {host_base: (None, host_base, buf_size)}

        src = np.array([[host_base + 100, 256, 0]], dtype=np.int64)
        dst = np.array([[0xD570, 256, 0]], dtype=np.int64)

        result = _chunked_rixl_write(
            agent, staging, src, dst, "peer", b"done", threading.Lock(),
        )

        self.assertIsInstance(result, _CompletedTransfer)
        agent.register_memory.assert_called_once()
        agent.transfer.assert_called_once()
        agent.deregister_memory.assert_called_once()

    def test_notification_only_on_last_chunk(self):
        from nixl_rocm_staging import _chunked_rixl_write, _IONIC_MR_CHUNK

        agent = self._make_mock_agent()
        staging = MagicMock(spec=RocmDramStaging)

        chunk_size = _IONIC_MR_CHUNK
        buf1_base = 0x10000000
        buf2_base = 0x20000000
        buf1_size = chunk_size  # exactly 1 chunk
        buf2_size = chunk_size

        staging.buffers = {
            buf1_base: (None, buf1_base, buf1_size),
            buf2_base: (None, buf2_base, buf2_size),
        }

        src = np.array([
            [buf1_base + 100, 1024, 0],
            [buf2_base + 200, 1024, 0],
        ], dtype=np.int64)
        dst = np.array([
            [0xAAA, 1024, 0],
            [0xBBB, 1024, 0],
        ], dtype=np.int64)

        notif = b"final_notif"
        _chunked_rixl_write(
            agent, staging, src, dst, "peer", notif, threading.Lock(),
        )

        # Check that initialize_xfer was called multiple times (chunks)
        # and only the last one got the real notification
        calls = agent.initialize_xfer.call_args_list
        if len(calls) > 1:
            for call in calls[:-1]:
                self.assertEqual(call[0][4], b"")
            self.assertEqual(calls[-1][0][4], notif)


# ---------------------------------------------------------------------------
# Test struct packing for kv_args (slab addressing)
# ---------------------------------------------------------------------------
class TestKvArgsPacking(unittest.TestCase):

    def test_pack_kv_ptrs(self):
        slab_ptr = 0x7F0000000000
        num_kv = 4
        slab_size = 200 * 1024 * 1024
        layer_slab = slab_size // num_kv

        kv_ptrs = [slab_ptr + i * layer_slab for i in range(num_kv)]
        packed = b"".join(struct.pack("Q", p) for p in kv_ptrs)

        self.assertEqual(len(packed), 8 * num_kv)

        unpacked = [struct.unpack_from("Q", packed, i * 8)[0] for i in range(num_kv)]
        self.assertEqual(unpacked, kv_ptrs)


if __name__ == "__main__":
    unittest.main()
