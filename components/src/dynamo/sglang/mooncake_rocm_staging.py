# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ROCm DRAM staging monkey-patch for SGLang Mooncake connector.

On AMD GPUs with Pensando ionic NICs, Mooncake cannot ibv_reg_mr() on
GPU VRAM (no GPU Direct RDMA).  This module patches MooncakeKVManager at
runtime so that:

  1. KV buffers are registered as pinned host (DRAM) instead of VRAM.
  2. Before each RDMA send, relevant GPU blocks are hipMemcpy'd to DRAM.
  3. After each RDMA receive completes, DRAM data is hipMemcpy'd back to GPU.

Nothing in sglang-amd source is modified — all hooks are injected here.

Usage (from dynamo sglang integration, or at container startup):

    from dynamo.sglang.mooncake_rocm_staging import patch_mooncake_for_rocm
    patch_mooncake_for_rocm()       # idempotent, safe to call multiple times

Or set the env var before launching SGLang:

    export SGLANG_MOONCAKE_ROCM_STAGING=1

Data flow with patch active:

    Prefill GPU KV ─hipMemcpy D2H─▶ Prefill DRAM staging
        ──── Mooncake RDMA WRITE ────▶
    Decode DRAM staging ─hipMemcpy H2D─▶ Decode GPU KV
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
from typing import List, Optional

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
    "SGLANG_MOONCAKE_ROCM_STAGING", ""
).lower() in ("1", "true", "yes")

_PATCHED = False  # guard against double-patch


# ---------------------------------------------------------------------------
# DRAM staging pool  (reused from nixl_rocm_staging design)
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
        """Allocate a pinned host buffer mirroring *gpu_ptr*.  Returns host ptr."""
        import torch

        t = torch.empty(size, dtype=torch.uint8, device="cpu").pin_memory()
        host_ptr = t.data_ptr()
        self.buffers[gpu_ptr] = (t, host_ptr, size)
        logger.info(
            "ROCm DRAM staging: %d MB  GPU %s → host %s",
            size // (1024 * 1024),
            hex(gpu_ptr),
            hex(host_ptr),
        )
        return host_ptr

    # -- address translation --------------------------------------------------

    def translate_addr(self, addr: int) -> int:
        """Translate a single GPU address to its DRAM staging counterpart."""
        for gpu_base, (_, host_base, buf_size) in self.buffers.items():
            if gpu_base <= addr < gpu_base + buf_size:
                return addr - gpu_base + host_base
        return addr

    def translate_addrs(self, addrs: list) -> list:
        return [self.translate_addr(a) for a in addrs]

    def translate_ptrs(self, ptrs: list) -> list:
        """Translate base pointers (exact-match on gpu_base)."""
        return [self.buffers[p][1] if p in self.buffers else p for p in ptrs]

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
# Engine wrapper — intercepts RDMA transfers to bounce through DRAM
# ---------------------------------------------------------------------------
class _StagingEngineWrapper:
    """Thin proxy around MooncakeTransferEngine that stages GPU↔DRAM.

    Only ``batch_transfer_sync`` is intercepted; every other attribute
    delegates to the real engine so the rest of the stack is unaffected.
    The real engine singleton is never mutated.
    """

    def __init__(self, real_engine, staging: _RocmDramStaging):
        self._real = real_engine
        self._staging = staging

    def __getattr__(self, name):
        return getattr(self._real, name)

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
    ) -> int:
        staging = self._staging
        for src, length in zip(buffers, lengths):
            staging.copy_d2h(src, length)
        staging.sync()
        translated_src = staging.translate_addrs(buffers)
        return self._real.batch_transfer_sync(
            session_id, translated_src, peer_buffer_addresses, lengths
        )

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        staging = self._staging
        staging.copy_d2h(buffer, length)
        staging.sync()
        translated = staging.translate_addr(buffer)
        return self._real.transfer_sync(
            session_id, translated, peer_buffer_address, length
        )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def patch_mooncake_for_rocm():
    """Monkey-patch SGLang's MooncakeKVManager / MooncakeKVReceiver for ROCm DRAM staging.

    Safe to call multiple times (idempotent).  Does nothing when not on ROCm
    and ``SGLANG_MOONCAKE_ROCM_STAGING`` is not set.
    """
    global _PATCHED
    if _PATCHED:
        return
    if not _STAGING_ENABLED:
        logger.debug("ROCm staging not enabled, skipping mooncake patch")
        return

    try:
        from sglang.srt.disaggregation.mooncake.conn import (
            MooncakeKVManager,
            MooncakeKVReceiver,
        )
    except ImportError:
        logger.warning("sglang mooncake connector not available, skipping ROCm patch")
        return

    logger.info("Applying ROCm DRAM staging monkey-patch to MooncakeKVManager")

    # ---- save originals -----------------------------------------------------
    _orig_mgr_init = MooncakeKVManager.__init__
    _orig_register = MooncakeKVManager.register_buffer_to_engine
    _orig_recv_register_kv = MooncakeKVReceiver._register_kv_args
    _orig_recv_init = MooncakeKVReceiver.init
    _orig_recv_poll = MooncakeKVReceiver.poll

    # ---- 1. MooncakeKVManager.__init__ --------------------------------------
    def _patched_mgr_init(self, *args, **kwargs):
        self._rocm_staging = _RocmDramStaging()
        _orig_mgr_init(self, *args, **kwargs)
        # Wrap the engine so ALL transfer calls (batch_transfer_sync,
        # transfer_sync) go through DRAM staging automatically.
        # The real singleton is not mutated — only self.engine is replaced.
        self.engine = _StagingEngineWrapper(self.engine, self._rocm_staging)
        logger.info("ROCm staging: engine wrapped with DRAM staging proxy")

    MooncakeKVManager.__init__ = _patched_mgr_init

    # ---- 2. register_buffer_to_engine — register DRAM mirrors ---------------
    def _patched_register(self):
        staging: _RocmDramStaging = self._rocm_staging

        # KV buffers → DRAM staging
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            dram_ptrs = []
            dram_lens = []
            for gpu_ptr, data_len in zip(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            ):
                host_ptr = staging.create(gpu_ptr, data_len)
                dram_ptrs.append(host_ptr)
                dram_lens.append(data_len)
            self.engine.batch_register(dram_ptrs, dram_lens)
            logger.info(
                "ROCm staging: registered %d KV buffers as DRAM", len(dram_ptrs)
            )

        # Aux buffers — already CPU/DRAM, register directly
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )

        # State buffers (Mamba, SWA) → DRAM staging
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            dram_ptrs = []
            dram_lens = []
            for gpu_ptr, data_len in zip(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            ):
                host_ptr = staging.create(gpu_ptr, data_len)
                dram_ptrs.append(host_ptr)
                dram_lens.append(data_len)
            self.engine.batch_register(dram_ptrs, dram_lens)
            logger.info(
                "ROCm staging: registered %d state buffers as DRAM", len(dram_ptrs)
            )

    MooncakeKVManager.register_buffer_to_engine = _patched_register

    # ---- 3. Post-receive H2D helper (attached to manager) -------------------
    def _staging_post_copy_h2d(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        """Copy received slots from DRAM staging → GPU after RDMA receive."""
        staging: _RocmDramStaging = self._rocm_staging

        # KV buffers
        for buf_idx, gpu_ptr in enumerate(self.kv_args.kv_data_ptrs):
            if gpu_ptr not in staging.buffers:
                continue
            item_len = self.kv_args.kv_item_lens[buf_idx]
            data_len = self.kv_args.kv_data_lens[buf_idx]
            for idx in kv_indices:
                offset = int(idx) * item_len
                if offset + item_len <= data_len:
                    staging.copy_h2d(gpu_ptr + offset, item_len)

        # State buffers (if present)
        if (
            state_indices
            and self.kv_args.state_data_ptrs
            and self.kv_args.state_item_lens
        ):
            for buf_idx, gpu_ptr in enumerate(self.kv_args.state_data_ptrs):
                if gpu_ptr not in staging.buffers:
                    continue
                item_len = self.kv_args.state_item_lens[buf_idx]
                data_len = self.kv_args.state_data_lens[buf_idx]
                for idx in state_indices:
                    offset = int(idx) * item_len
                    if offset + item_len <= data_len:
                        staging.copy_h2d(gpu_ptr + offset, item_len)

        staging.sync()

    MooncakeKVManager._staging_post_copy_h2d = _staging_post_copy_h2d

    # ---- 4. MooncakeKVReceiver._register_kv_args — advertise DRAM addrs -----
    def _patched_register_kv_args(self):
        staging: _RocmDramStaging = self.kv_mgr._rocm_staging

        for bootstrap_info in self.bootstrap_infos:
            # Translate KV ptrs: GPU → DRAM staging
            kv_ptrs = staging.translate_ptrs(self.kv_mgr.kv_args.kv_data_ptrs)
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in kv_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            # Translate state ptrs: GPU → DRAM staging
            state_ptrs = staging.translate_ptrs(self.kv_mgr.kv_args.state_data_ptrs)
            packed_state_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in state_ptrs
            )
            packed_state_item_lens = b"".join(
                struct.pack("I", item_len)
                for item_len in self.kv_mgr.kv_args.state_item_lens
            )
            state_dim_per_tensor = getattr(
                self.kv_mgr.kv_args, "state_dim_per_tensor", []
            )
            packed_state_dim_per_tensor = b"".join(
                struct.pack("I", dim) for dim in state_dim_per_tensor
            )

            tp_rank = self.kv_mgr.kv_args.engine_rank
            kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]

            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        packed_state_data_ptrs,
                        str(tp_rank).encode("ascii"),
                        str(self.kv_mgr.attn_tp_size).encode("ascii"),
                        str(kv_item_len).encode("ascii"),
                        packed_state_item_lens,
                        packed_state_dim_per_tensor,
                    ]
                )

    MooncakeKVReceiver._register_kv_args = _patched_register_kv_args

    # ---- 5. MooncakeKVReceiver.init — stash indices for post-copy -----------
    def _patched_recv_init(self, kv_indices, aux_index=None, state_indices=None):
        self._staging_kv_indices = kv_indices.copy()
        self._staging_state_indices = (
            list(state_indices) if state_indices is not None else None
        )
        return _orig_recv_init(self, kv_indices, aux_index, state_indices)

    MooncakeKVReceiver.init = _patched_recv_init

    # ---- 6. MooncakeKVReceiver.poll — DRAM→GPU after transfer done ----------
    def _patched_recv_poll(self):
        from sglang.srt.disaggregation.base.conn import KVPoll

        result = _orig_recv_poll(self)
        if result == KVPoll.Success and hasattr(self, "_staging_kv_indices"):
            self.kv_mgr._staging_post_copy_h2d(
                self._staging_kv_indices,
                state_indices=getattr(self, "_staging_state_indices", None),
            )
            del self._staging_kv_indices
            if hasattr(self, "_staging_state_indices"):
                del self._staging_state_indices
        return result

    MooncakeKVReceiver.poll = _patched_recv_poll

    _PATCHED = True
    logger.info("ROCm DRAM staging patch for Mooncake applied successfully")


# ---------------------------------------------------------------------------
# Auto-patch on import when env var is set
# ---------------------------------------------------------------------------
if _STAGING_ENABLED:
    try:
        patch_mooncake_for_rocm()
    except Exception:
        logger.debug("Auto-patch deferred (sglang not yet importable)", exc_info=True)
