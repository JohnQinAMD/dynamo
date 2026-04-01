# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ROCm DRAM staging monkey-patch for SGLang nixl connector.

On AMD GPUs with Pensando ionic NICs, RIXL/nixl cannot ibv_reg_mr() on
GPU VRAM (no GPU Direct RDMA).  This module patches NixlKVManager at
runtime so that:

  1. KV buffers are registered as pinned host (DRAM) instead of VRAM.
  2. Before each RDMA send, relevant GPU blocks are hipMemcpy'd to DRAM.
  3. After each RDMA receive completes, DRAM data is hipMemcpy'd back to GPU.

Nothing in sglang-amd source is modified — all hooks are injected here.

Usage (from dynamo sglang integration, or at container startup):

    from dynamo.sglang.nixl_rocm_staging import patch_nixl_for_rocm
    patch_nixl_for_rocm()       # idempotent, safe to call multiple times

Or set the env var before launching SGLang:

    export SGLANG_NIXL_ROCM_STAGING=1

Data flow with patch active:

    Prefill GPU KV ─hipMemcpy D2H─▶ Prefill DRAM staging
        ──── RIXL RDMA WRITE ────▶
    Decode DRAM staging ─hipMemcpy H2D─▶ Decode GPU KV
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
from typing import Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-detect ROCm
# ---------------------------------------------------------------------------
_IS_ROCM = False
try:
    import torch as _torch

    _IS_ROCM = hasattr(_torch.version, "hip") and _torch.version.hip is not None
except ImportError:
    pass

_STAGING_ENABLED = _IS_ROCM or os.environ.get(
    "SGLANG_NIXL_ROCM_STAGING", ""
).lower() in ("1", "true", "yes")

_PATCHED = False  # guard against double-patch


# ---------------------------------------------------------------------------
# DRAM staging pool
# ---------------------------------------------------------------------------
class _RocmDramStaging:
    """Pinned host mirrors for GPU VRAM buffers.

    Each GPU buffer gets a same-sized pinned host allocation.  Addresses
    are translated 1:1 (gpu_base+offset → host_base+offset) so all
    existing index arithmetic in the connector is preserved.
    """

    def __init__(self):
        try:
            self._hip = ctypes.CDLL("libamdhip64.so")
        except OSError:
            raise RuntimeError(
                "libamdhip64.so not found — ROCm DRAM staging requires the HIP runtime"
            )
        # gpu_base → (pinned_tensor, host_ptr, size)
        self.buffers: dict[int, tuple] = {}

    def create(self, gpu_ptr: int, size: int) -> int:
        """Record GPU buffer metadata for lazy staging. Returns a placeholder host ptr.
        
        In lazy mode (ionic), we don't allocate the full DRAM mirror upfront.
        Instead, small staging regions are allocated per-transfer in get_staging_region().
        """
        # Store metadata only — no allocation
        self.buffers[gpu_ptr] = (None, gpu_ptr, size)  # (tensor=None, placeholder=gpu_ptr, size)
        logger.info(
            "ROCm DRAM staging: %d MB GPU %s (lazy — no allocation yet)",
            size // (1024 * 1024),
            hex(gpu_ptr),
        )
        return gpu_ptr  # return GPU ptr as placeholder — translate_base will map it

    def get_staging_region(self, gpu_addr: int, nbytes: int):
        """Allocate a small pinned DRAM region for a single transfer chunk.
        Returns (host_ptr, tensor) — caller must keep tensor alive until transfer completes.
        """
        import torch
        t = torch.empty(nbytes, dtype=torch.uint8, device="cpu").pin_memory()
        host_ptr = t.data_ptr()
        # Copy GPU → DRAM
        self.copy_d2h_direct(gpu_addr, host_ptr, nbytes)
        self.sync()
        return host_ptr, t

    def copy_d2h_direct(self, gpu_addr: int, host_addr: int, nbytes: int):
        """hipMemcpy D2H between arbitrary addresses."""
        self._hip.hipMemcpy(
            self._ctypes.c_void_p(host_addr),
            self._ctypes.c_void_p(gpu_addr),
            self._ctypes.c_size_t(nbytes),
            self._ctypes.c_int(2),  # D2H
        )

    def copy_h2d_direct(self, host_addr: int, gpu_addr: int, nbytes: int):
        """hipMemcpy H2D between arbitrary addresses."""
        self._hip.hipMemcpy(
            self._ctypes.c_void_p(gpu_addr),
            self._ctypes.c_void_p(host_addr),
            self._ctypes.c_size_t(nbytes),
            self._ctypes.c_int(1),  # H2D
        )

    # -- address translation --------------------------------------------------

    def translate_base(self, gpu_ptr: int) -> int:
        e = self.buffers.get(gpu_ptr)
        return e[1] if e else gpu_ptr

    def translate_ptrs(self, ptrs: list) -> list:
        return [self.translate_base(p) for p in ptrs]

    def translate_reqs(self, reqs):
        """Translate GPU addrs → DRAM staging addrs.

        Accepts an (N,3) numpy array **or** a list of (addr,len,dev[,tag]) tuples.
        Non-matching addresses (e.g. remote dst) pass through unchanged.
        """
        if isinstance(reqs, np.ndarray):
            if reqs.size == 0:
                return reqs
            result = reqs.copy()
            for gpu_base, (_, host_base, buf_size) in self.buffers.items():
                mask = (result[:, 0] >= gpu_base) & (
                    result[:, 0] < gpu_base + buf_size
                )
                result[mask, 0] = result[mask, 0] - gpu_base + host_base
                result[mask, 2] = 0  # CPU device
            return result

        out = []
        for item in reqs:
            addr, length, dev = item[0], item[1], item[2]
            for gpu_base, (_, host_base, buf_size) in self.buffers.items():
                if gpu_base <= addr < gpu_base + buf_size:
                    addr = addr - gpu_base + host_base
                    dev = 0
                    break
            out.append(
                (addr, length, dev) if len(item) == 3 else (addr, length, dev, item[3])
            )
        return out

    # -- GPU ↔ DRAM copies ----------------------------------------------------

    def copy_d2h(self, gpu_addr: int, nbytes: int):
        """hipMemcpy Device→Host.  Silently skips addresses not in any buffer."""
        for gpu_base, (_, host_base, buf_size) in self.buffers.items():
            if gpu_base <= gpu_addr < gpu_base + buf_size:
                offset = gpu_addr - gpu_base
                self._hip.hipMemcpy(
                    ctypes.c_void_p(host_base + offset),
                    ctypes.c_void_p(gpu_addr),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_int(2),  # hipMemcpyDeviceToHost
                )
                return

    def copy_h2d(self, gpu_addr: int, nbytes: int):
        """hipMemcpy Host→Device.  Silently skips addresses not in any buffer."""
        for gpu_base, (_, host_base, buf_size) in self.buffers.items():
            if gpu_base <= gpu_addr < gpu_base + buf_size:
                offset = gpu_addr - gpu_base
                self._hip.hipMemcpy(
                    ctypes.c_void_p(gpu_addr),
                    ctypes.c_void_p(host_base + offset),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_int(1),  # hipMemcpyHostToDevice
                )
                return

    def sync(self):
        self._hip.hipDeviceSynchronize()


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def patch_nixl_for_rocm():
    """Monkey-patch SGLang's NixlKVManager / NixlKVReceiver for ROCm DRAM staging.

    Safe to call multiple times (idempotent).  Does nothing when not on ROCm
    and ``SGLANG_NIXL_ROCM_STAGING`` is not set.
    """
    global _PATCHED
    if _PATCHED:
        return
    if not _STAGING_ENABLED:
        logger.debug("ROCm staging not enabled, skipping nixl patch")
        return

    try:
        from sglang.srt.disaggregation.nixl.conn import (
            NixlKVManager,
            NixlKVReceiver,
        )
    except ImportError:
        logger.warning("sglang nixl connector not available, skipping ROCm patch")
        return

    logger.info("Applying ROCm DRAM staging monkey-patch to NixlKVManager")

    # ---- save originals -----------------------------------------------------
    _orig_mgr_init = NixlKVManager.__init__
    _orig_register = NixlKVManager.register_buffer_to_engine
    _orig_recv_init = NixlKVReceiver.init
    _orig_recv_poll = NixlKVReceiver.poll
    _orig_recv_register_kv = NixlKVReceiver._register_kv_args

    # ---- 1. NixlKVManager.__init__ ------------------------------------------
    def _patched_mgr_init(self, *args, **kwargs):
        # Set staging object BEFORE original __init__ (which calls
        # register_buffer_to_engine internally).
        self._rocm_staging = _RocmDramStaging()
        _orig_mgr_init(self, *args, **kwargs)

        # Wrap agent.get_xfer_descs once so ALL transfer methods that call
        # get_xfer_descs("VRAM") go through the staging pipeline automatically.
        _orig_get = self.agent.get_xfer_descs
        staging = self._rocm_staging

        def _staging_get_xfer_descs(reqs, mem_type):
            if mem_type == "VRAM":
                # Lazy staging: allocate small DRAM regions per-transfer,
                # copy GPU→DRAM, register with RIXL, get xfer descs.

                reg_addrs = []
                staging_tensors = []  # keep alive until transfer completes

                if isinstance(reqs, np.ndarray):
                    items = [(int(reqs[i, 0]), int(reqs[i, 1])) for i in range(len(reqs))]
                else:
                    items = [(int(r[0]), int(r[1])) for r in reqs]

                for gpu_addr, nbytes in items:
                    host_ptr, tensor = staging.get_staging_region(gpu_addr, nbytes)
                    reg_addrs.append((host_ptr, nbytes, 0, ""))
                    staging_tensors.append(tensor)

                if reg_addrs:
                    try:
                        per_xfer_descs = self.agent.register_memory(reg_addrs, "DRAM")
                        if hasattr(self, "_active_mr_descs"):
                            self._active_mr_descs.append(per_xfer_descs)
                        if not hasattr(self, "_staging_tensors"):
                            self._staging_tensors = []
                        self._staging_tensors.extend(staging_tensors)
                        return _orig_get(per_xfer_descs, "DRAM")
                    except Exception as e:
                        logger.warning("Lazy MR registration failed: %s", e)
                        # Fallback: try without staging
                        return _orig_get(reqs, "DRAM")
                return _orig_get(reqs, mem_type)
            return _orig_get(reqs, mem_type)

        self.agent.get_xfer_descs = _staging_get_xfer_descs
        logger.info("ROCm staging: agent.get_xfer_descs wrapped")

    NixlKVManager.__init__ = _patched_mgr_init

    # ---- 2. register_buffer_to_engine ---------------------------------------
    def _patched_register(self):
        staging: _RocmDramStaging = self._rocm_staging

        # Solution A: lazy MR registration for ionic NICs.
        # ionic has ~3GB total MR capacity — we cannot pre-register all KV buffers.
        # Instead: create DRAM staging buffers but skip ibv_reg_mr.
        # MRs will be registered per-transfer in _staging_get_xfer_descs.

        # Create DRAM staging buffers (no registration yet)
        for gpu_ptr, data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            staging.create(gpu_ptr, data_len)
        logger.info(
            "ROCm staging: created %d DRAM mirrors (lazy MR — no pre-registration)",
            len(self.kv_args.kv_data_ptrs),
        )

        # Aux buffers are small — register normally
        aux_addrs = []
        for ptr, length in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            aux_addrs.append((ptr, length, 0, ""))
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM")
        if not self.aux_descs:
            raise Exception("NIXL DRAM registration failed for aux tensors")

        # State buffers — create staging but skip registration
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            for gpu_ptr, data_len in zip(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            ):
                staging.create(gpu_ptr, data_len)

        # Track per-transfer registrations for cleanup
        self._active_mr_descs = []
        self.kv_descs = True  # placeholder — actual descs created per-transfer
        self.state_descs = True

    NixlKVManager.register_buffer_to_engine = _patched_register

    # ---- 3. _staging_post_copy_h2d (new helper) ----------------------------
    def _staging_post_copy_h2d(
        self, kv_indices: npt.NDArray[np.int32]
    ):
        """Copy received tokens from DRAM staging → GPU after RDMA receive."""
        staging: _RocmDramStaging = self._rocm_staging
        for buf_idx, (gpu_ptr, data_len) in enumerate(
            zip(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)
        ):
            if gpu_ptr not in staging.buffers:
                continue
            if self.is_mla_backend:
                item_len = self.kv_args.kv_item_lens[buf_idx]
            else:
                item_len = self.kv_args.kv_item_lens[buf_idx // 2]
            for idx in kv_indices:
                offset = int(idx) * item_len
                if offset + item_len <= data_len:
                    staging.copy_h2d(gpu_ptr + offset, item_len)
        staging.sync()

    NixlKVManager._staging_post_copy_h2d = _staging_post_copy_h2d

    # ---- 4. NixlKVReceiver._register_kv_args --------------------------------
    def _patched_register_kv_args(self):
        staging: Optional[_RocmDramStaging] = self.kv_mgr._rocm_staging
        from sglang.srt.disaggregation.nixl.conn import GUARD

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)

            # Advertise DRAM staging ptrs so prefill writes into DRAM
            kv_ptrs = staging.translate_ptrs(self.kv_mgr.kv_args.kv_data_ptrs)
            state_ptrs = staging.translate_ptrs(self.kv_mgr.kv_args.state_data_ptrs)
            advertised_gpu_id = 0  # CPU / DRAM

            packed_kv = b"".join(struct.pack("Q", p) for p in kv_ptrs)
            packed_aux = b"".join(
                struct.pack("Q", p) for p in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state = b"".join(struct.pack("Q", p) for p in state_ptrs)

            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        self.kv_mgr.agent.get_agent_metadata(),
                        packed_kv,
                        packed_aux,
                        packed_state,
                        str(advertised_gpu_id).encode("ascii"),
                        str(self.kv_mgr.attn_tp_size).encode("ascii"),
                        str(self.kv_mgr.kv_args.engine_rank).encode("ascii"),
                        str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii"),
                    ]
                )

    NixlKVReceiver._register_kv_args = _patched_register_kv_args

    # ---- 5. NixlKVReceiver.init — store kv_indices for post-copy ------------
    def _patched_recv_init(self, kv_indices, aux_index=None, state_indices=None):
        self._staging_kv_indices = kv_indices.copy()
        return _orig_recv_init(self, kv_indices, aux_index, state_indices)

    NixlKVReceiver.init = _patched_recv_init

    # ---- 6. NixlKVReceiver.poll — DRAM→GPU after transfer done --------------
    def _patched_recv_poll(self):
        from sglang.srt.disaggregation.base.conn import KVPoll

        result = _orig_recv_poll(self)
        if result == KVPoll.Success and hasattr(self, "_staging_kv_indices"):
            # DRAM → GPU copy
            self.kv_mgr._staging_post_copy_h2d(self._staging_kv_indices)
            del self._staging_kv_indices

            # Deregister per-transfer MRs to free ionic MR slots
            if hasattr(self.kv_mgr, "_active_mr_descs"):
                for descs in self.kv_mgr._active_mr_descs:
                    try:
                        self.kv_mgr.agent.deregister_memory(descs)
                    except Exception:
                        pass
                self.kv_mgr._active_mr_descs.clear()
        return result

    NixlKVReceiver.poll = _patched_recv_poll

    _PATCHED = True
    logger.info("ROCm DRAM staging patch applied successfully")


# ---------------------------------------------------------------------------
# Auto-patch on import when env var is set
# ---------------------------------------------------------------------------
if _STAGING_ENABLED:
    try:
        patch_nixl_for_rocm()
    except Exception:
        logger.debug("Auto-patch deferred (sglang not yet importable)", exc_info=True)
