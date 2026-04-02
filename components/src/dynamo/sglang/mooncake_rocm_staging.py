# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ROCm DRAM staging monkey-patch for SGLang Mooncake connector.

On AMD GPUs with Pensando ionic NICs, Mooncake cannot ibv_reg_mr() on
GPU VRAM (no GPU Direct RDMA).  This module patches MooncakeKVManager at
runtime so that:

  1. KV buffers are mirrored in host DRAM (mmap+mlock, NOT hipHostMalloc).
  2. Before each RDMA send, GPU blocks are hipMemcpy'd to DRAM, then
     registered, transferred, and unregistered in chunks that fit within
     the ionic per-device MR limit (~250 MB).
  3. After each RDMA receive completes, DRAM data is hipMemcpy'd back to GPU.

Nothing in sglang-amd source is modified — all hooks are injected here.

Usage (from dynamo sglang integration, or at container startup):

    from dynamo.sglang.mooncake_rocm_staging import patch_mooncake_for_rocm
    patch_mooncake_for_rocm()       # idempotent, safe to call multiple times

Or set the env var before launching SGLang:

    export SGLANG_MOONCAKE_ROCM_STAGING=1

Data flow with patch active:

    Prefill GPU KV ─hipMemcpy D2H─▶ Prefill DRAM staging
        ──── Mooncake RDMA WRITE (chunked MR) ────▶
    Decode DRAM staging ─hipMemcpy H2D─▶ Decode GPU KV

ionic ``ibv_reg_mr()`` limitations handled by chunked registration:
  - EINVAL on ``hipHostMalloc``-backed memory → use mmap+mlock instead
  - ~199 MB max single MR, ~250 MB total per device → register/transfer/unregister
    in chunks ≤ 190 MB so the MR slot is reused for every chunk
"""

from __future__ import annotations

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
# DRAM staging pool — shared implementation (after _IS_ROCM detection)
# ---------------------------------------------------------------------------
from dynamo.sglang.rocm_dram_staging_common import (  # noqa: E402
    RocmDramStaging as _RocmDramStaging,
)


# ---------------------------------------------------------------------------
# Engine wrapper — intercepts RDMA transfers to bounce through DRAM
# ---------------------------------------------------------------------------
_IONIC_MR_CHUNK = int(os.environ.get("MOONCAKE_IONIC_MR_CHUNK_MB", "190")) * 1024 * 1024


class _StagingEngineWrapper:
    """Thin proxy around MooncakeTransferEngine that stages GPU↔DRAM.

    Transfers go through DRAM staging with chunked ibv_reg_mr registration
    to stay within ionic's ~250 MB per-device MR limit.  Each chunk is
    registered, transferred via RDMA, then immediately unregistered so the
    MR slot can be reused for the next chunk.
    """

    def __init__(self, real_engine, staging: _RocmDramStaging):
        self._real = real_engine
        self._staging = staging
        self._mr_lock = __import__("threading").Lock()

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _chunked_rdma_write(
        self, session_id: str, host_ptr: int, peer_ptr: int, length: int
    ) -> int:
        """Register → RDMA write → unregister in ionic-safe chunks."""
        chunk = _IONIC_MR_CHUNK
        offset = 0
        while offset < length:
            n = min(chunk, length - offset)
            src = host_ptr + offset
            dst = peer_ptr + offset
            with self._mr_lock:
                self._real.register(src, n)
                ret = self._real.transfer_sync(session_id, src, dst, n)
                self._real.deregister(src)
            if ret < 0:
                logger.error(
                    "Chunked RDMA write failed at offset %d/%d (ret=%d)",
                    offset, length, ret,
                )
                return ret
            offset += n
        return 0

    def _resolve_host_ptr(self, addr: int, length: int) -> int:
        """Get host pointer for a buffer: D2H copy for GPU, passthrough for CPU."""
        staging = self._staging
        if staging.copy_d2h(addr, length):
            return staging.translate_addr(addr)
        # Not a staged GPU buffer — might be an aux/CPU buffer already in DRAM.
        # Return the address as-is for direct RDMA.
        return addr

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
    ) -> int:
        host_ptrs = []
        for src, length in zip(buffers, lengths):
            host_ptrs.append(self._resolve_host_ptr(src, length))
        self._staging.sync()

        for host_ptr, peer, length in zip(host_ptrs, peer_buffer_addresses, lengths):
            ret = self._chunked_rdma_write(session_id, host_ptr, peer, length)
            if ret < 0:
                return ret
        return 0

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        host_ptr = self._resolve_host_ptr(buffer, length)
        self._staging.sync()
        return self._chunked_rdma_write(
            session_id, host_ptr, peer_buffer_address, length
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

    _required_attrs = [
        (MooncakeKVManager, "__init__"),
        (MooncakeKVManager, "register_buffer_to_engine"),
        (MooncakeKVReceiver, "_register_kv_args"),
        (MooncakeKVReceiver, "init"),
        (MooncakeKVReceiver, "poll"),
    ]
    for cls, attr in _required_attrs:
        if not hasattr(cls, attr):
            logger.error(
                "ROCm DRAM staging patch ABORTED: %s.%s not found. "
                "SGLang API may have changed — update the monkey-patch.",
                cls.__name__,
                attr,
            )
            return

    logger.info("Applying ROCm DRAM staging monkey-patch to MooncakeKVManager")

    # ---- 0. Keep RDMA transport (chunked MR handles ionic limits) ----------
    logger.info(
        "ROCm staging: using RDMA transport with chunked MR registration "
        "(ionic limit ~250 MB/device, chunk size %d MB)",
        _IONIC_MR_CHUNK // (1024 * 1024),
    )

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

    # ---- 2. register_buffer_to_engine — create DRAM mirrors + receive slab --
    _SLAB_MB = int(os.environ.get("MOONCAKE_RECV_SLAB_MB", "200"))

    def _patched_register(self):
        staging: _RocmDramStaging = self._rocm_staging

        # KV buffers → DRAM staging mirrors
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            for gpu_ptr, data_len in zip(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            ):
                staging.create(gpu_ptr, data_len)
            logger.info(
                "ROCm staging: created %d KV DRAM mirrors",
                len(self.kv_args.kv_data_ptrs),
            )

        # Aux buffers — small CPU/DRAM, register directly
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )

        # State buffers → DRAM staging mirrors
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            for gpu_ptr, data_len in zip(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            ):
                staging.create(gpu_ptr, data_len)

        # Pre-register a receive slab for decode side.  This avoids
        # per-request ibv_reg_mr in _patched_recv_init.
        # The slab is a fixed mmap region that stays registered for the
        # lifetime of the server.  Prefill RDMA-writes into it, then
        # decode copies slab → DRAM → GPU.
        slab_size = _SLAB_MB * 1024 * 1024
        import mmap as _mmap
        import ctypes as _ctypes
        slab_mmap = _mmap.mmap(-1, slab_size)
        slab_buf = (_ctypes.c_char * slab_size).from_buffer(slab_mmap)
        slab_ptr = _ctypes.addressof(slab_buf)
        libc = _RocmDramStaging._get_libc()
        libc.mlock(_ctypes.c_void_p(slab_ptr), _ctypes.c_size_t(slab_size))
        real_engine = self.engine._real if hasattr(self.engine, "_real") else self.engine
        real_engine.register(slab_ptr, slab_size)
        self._recv_slab = (slab_mmap, slab_buf, slab_ptr, slab_size)
        logger.info(
            "ROCm staging: pre-registered %d MB receive slab at %s",
            _SLAB_MB, hex(slab_ptr),
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

    # ---- 4. MooncakeKVReceiver._register_kv_args — advertise slab or DRAM ---
    def _patched_register_kv_args(self):
        staging: _RocmDramStaging = self.kv_mgr._rocm_staging

        # If receive slab exists, advertise slab-based addresses.
        # The prefill writes into the slab; decode copies slab→DRAM→GPU.
        if hasattr(self.kv_mgr, "_recv_slab"):
            _, _, slab_ptr, slab_size = self.kv_mgr._recv_slab
            # Map each KV layer to a slab offset.  For ISL=1024 with
            # page_size=16 and item_len~9216: each layer's active data
            # per request is small (~pages * item_len < slab_size / num_layers).
            kv_ptrs = []
            offset = 0
            for data_len in self.kv_mgr.kv_args.kv_data_lens:
                kv_ptrs.append(slab_ptr + offset)
                # Each layer gets a proportional slab slice
                layer_slab = slab_size // max(len(self.kv_mgr.kv_args.kv_data_lens), 1)
                offset += layer_slab
            logger.debug(
                "ROCm staging: advertising slab addresses for %d KV layers",
                len(kv_ptrs),
            )
        else:
            # Fallback: translate GPU → DRAM staging (original behavior)
            kv_ptrs = staging.translate_ptrs(self.kv_mgr.kv_args.kv_data_ptrs)

        for bootstrap_info in self.bootstrap_infos:
            packed_kv_data_ptrs = b"".join(struct.pack("Q", ptr) for ptr in kv_ptrs)
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

    # ---- 5. MooncakeKVReceiver.init — skip MR if slab is pre-registered -----
    def _patched_recv_init(self, kv_indices, aux_index=None, state_indices=None):
        self._staging_kv_indices = kv_indices.copy()
        self._staging_state_indices = (
            list(state_indices) if state_indices is not None else None
        )
        self._staging_registered_regions = []

        if hasattr(self.kv_mgr, "_recv_slab"):
            # Slab is pre-registered — no per-request MR needed
            pass
        elif len(kv_indices) > 0:
            # Fallback: dynamic MR registration per request
            staging: _RocmDramStaging = self.kv_mgr._rocm_staging
            real_engine = self.kv_mgr.engine._real
            for buf_idx, gpu_ptr in enumerate(self.kv_mgr.kv_args.kv_data_ptrs):
                if gpu_ptr not in staging.buffers:
                    continue
                _, host_base, buf_size = staging.buffers[gpu_ptr]
                item_len = self.kv_mgr.kv_args.kv_item_lens[buf_idx]
                min_idx = int(min(kv_indices))
                max_idx = int(max(kv_indices))
                start = min_idx * item_len
                end = (max_idx + 1) * item_len
                if end <= buf_size:
                    region_addr = host_base + start
                    region_len = end - start
                    real_engine.register(region_addr, region_len)
                    self._staging_registered_regions.append(
                        (region_addr, region_len)
                    )

        return _orig_recv_init(self, kv_indices, aux_index, state_indices)

    MooncakeKVReceiver.init = _patched_recv_init

    # ---- 6. MooncakeKVReceiver.poll — slab→DRAM→GPU or DRAM→GPU -------------
    def _patched_recv_poll(self):
        from sglang.srt.disaggregation.base.conn import KVPoll
        import ctypes as _ctypes

        result = _orig_recv_poll(self)
        if result == KVPoll.Success and hasattr(self, "_staging_kv_indices"):
            staging: _RocmDramStaging = self.kv_mgr._rocm_staging

            if hasattr(self.kv_mgr, "_recv_slab"):
                # Slab mode: copy slab → DRAM staging → GPU
                _, _, slab_ptr, slab_size = self.kv_mgr._recv_slab
                num_layers = len(self.kv_mgr.kv_args.kv_data_ptrs)
                layer_slab = slab_size // max(num_layers, 1)

                for buf_idx, gpu_ptr in enumerate(self.kv_mgr.kv_args.kv_data_ptrs):
                    if gpu_ptr not in staging.buffers:
                        continue
                    _, host_base, buf_size = staging.buffers[gpu_ptr]
                    item_len = self.kv_mgr.kv_args.kv_item_lens[buf_idx]
                    slab_layer_base = slab_ptr + buf_idx * layer_slab

                    for idx in self._staging_kv_indices:
                        offset = int(idx) * item_len
                        if offset + item_len <= buf_size and offset + item_len <= layer_slab:
                            # slab → DRAM
                            _ctypes.memmove(
                                host_base + offset,
                                slab_layer_base + offset,
                                item_len,
                            )
                            # DRAM → GPU
                            staging.copy_h2d(gpu_ptr + offset, item_len)
                staging.sync()
            else:
                # Non-slab mode: DRAM → GPU directly
                self.kv_mgr._staging_post_copy_h2d(
                    self._staging_kv_indices,
                    state_indices=getattr(self, "_staging_state_indices", None),
                )
                # Unregister dynamic MR regions
                real_engine = self.kv_mgr.engine._real
                for host_addr, length in getattr(
                    self, "_staging_registered_regions", []
                ):
                    real_engine.deregister(host_addr)

            del self._staging_kv_indices
            if hasattr(self, "_staging_registered_regions"):
                del self._staging_registered_regions
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
