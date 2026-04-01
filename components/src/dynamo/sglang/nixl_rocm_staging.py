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
# DRAM staging pool — shared implementation (after _IS_ROCM detection)
# ---------------------------------------------------------------------------
from dynamo.sglang.rocm_dram_staging_common import (  # noqa: E402
    RocmDramStaging as _RocmDramStaging,
)


class _MrLruCache:
    """LRU cache for RIXL memory registrations on ionic NICs.

    ionic has ~3GB total MR capacity. Instead of registering/deregistering
    per transfer (Solution A, ~100ms overhead each), cache recently-used
    MRs and evict LRU entries when approaching the limit.
    """

    MAX_REGISTERED_BYTES = 2 * 1024 * 1024 * 1024  # 2GB safety (ionic ~3GB limit)

    def __init__(self, agent):
        self.agent = agent
        self._cache = {}  # (addr, size) → (descs, tensor, access_count)
        self._total_bytes = 0
        self._access_counter = 0

    def get_or_register(self, host_ptr: int, nbytes: int, tensor):
        """Return registered descs for (host_ptr, nbytes). Register if not cached."""
        key = (host_ptr, nbytes)
        self._access_counter += 1

        if key in self._cache:
            descs, _, _ = self._cache[key]
            self._cache[key] = (descs, tensor, self._access_counter)
            return descs

        # Need to register — evict LRU entries if over budget
        while self._total_bytes + nbytes > self.MAX_REGISTERED_BYTES and self._cache:
            self._evict_lru()

        try:
            descs = self.agent.register_memory([(host_ptr, nbytes, 0, "")], "DRAM")
            self._cache[key] = (descs, tensor, self._access_counter)
            self._total_bytes += nbytes
            return descs
        except Exception as e:
            # Registration failed — try evicting more
            logger.warning("MR registration failed (%s), evicting...", e)
            while (
                self._cache and self._total_bytes + nbytes > self.MAX_REGISTERED_BYTES
            ):
                self._evict_lru()
            try:
                descs = self.agent.register_memory([(host_ptr, nbytes, 0, "")], "DRAM")
                self._cache[key] = (descs, tensor, self._access_counter)
                self._total_bytes += nbytes
                return descs
            except Exception:
                logger.error("MR registration failed even after eviction")
                return None

    def _evict_lru(self):
        """Evict the least recently used MR entry."""
        if not self._cache:
            return
        lru_key = min(self._cache, key=lambda k: self._cache[k][2])
        descs, _, _ = self._cache.pop(lru_key)
        try:
            self.agent.deregister_memory(descs)
        except Exception:
            pass
        self._total_bytes -= lru_key[1]
        logger.debug(
            "MR LRU evict: %s (%d bytes, total now %d)",
            hex(lru_key[0]),
            lru_key[1],
            self._total_bytes,
        )

    def clear(self):
        """Deregister all cached MRs."""
        for key, (descs, _, _) in list(self._cache.items()):
            try:
                self.agent.deregister_memory(descs)
            except Exception:
                pass
        self._cache.clear()
        self._total_bytes = 0


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
        from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVReceiver
    except ImportError:
        logger.warning("sglang nixl connector not available, skipping ROCm patch")
        return

    _required_attrs = [
        (NixlKVManager, "__init__"),
        (NixlKVManager, "register_buffer_to_engine"),
        (NixlKVReceiver, "init"),
        (NixlKVReceiver, "poll"),
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

                staging_tensors = []  # keep alive until transfer completes

                if isinstance(reqs, np.ndarray):
                    items = [
                        (int(reqs[i, 0]), int(reqs[i, 1])) for i in range(len(reqs))
                    ]
                else:
                    items = [(int(r[0]), int(r[1])) for r in reqs]

                # Solution B: LRU MR cache — register or reuse cached MRs
                mr_cache = getattr(self, "_mr_cache", None)
                all_xfer_tuples = []

                for gpu_addr, nbytes in items:
                    host_ptr, tensor = staging.get_staging_region(gpu_addr, nbytes)
                    staging_tensors.append(tensor)

                    if mr_cache:
                        mr_cache.get_or_register(host_ptr, nbytes, tensor)
                    else:
                        try:
                            self.agent.register_memory(
                                [(host_ptr, nbytes, 0, "")], "DRAM"
                            )
                        except Exception as e:
                            logger.warning("MR registration failed: %s", e)

                    all_xfer_tuples.append((host_ptr, nbytes, 0))

                if all_xfer_tuples:
                    return _orig_get(all_xfer_tuples, "DRAM")
                return _orig_get(reqs, mem_type)
            return _orig_get(reqs, mem_type)

        # Create MR LRU cache for prefill-side lazy registration
        self._mr_cache = _MrLruCache(self.agent)

        self.agent.get_xfer_descs = _staging_get_xfer_descs
        logger.info("ROCm staging: agent.get_xfer_descs wrapped (LRU MR cache)")

    NixlKVManager.__init__ = _patched_mgr_init

    # ---- 2. register_buffer_to_engine ---------------------------------------
    def _patched_register(self):
        staging: _RocmDramStaging = self._rocm_staging
        MAX_CHUNK = 512 * 1024 * 1024  # 512MB per MR chunk (ionic safe limit)
        is_decode = self.disaggregation_mode.value == "decode"

        if is_decode:
            # DECODE side: pre-allocate DRAM + register in chunks.
            # Decode needs stable addresses for RDMA WRITE targets.
            # Register in 512MB chunks to stay under ionic ~3GB MR limit.
            kv_addrs = []
            for gpu_ptr, data_len in zip(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            ):
                host_ptr = staging.create(gpu_ptr, data_len, allocate=True)
                offset = 0
                while offset < data_len:
                    chunk = min(MAX_CHUNK, data_len - offset)
                    kv_addrs.append((host_ptr + offset, chunk, 0, ""))
                    offset += chunk
            self.kv_descs = self.agent.register_memory(kv_addrs, "DRAM")
            logger.info(
                "ROCm staging DECODE: registered %d chunks (%d buffers)",
                len(kv_addrs),
                len(self.kv_args.kv_data_ptrs),
            )
            if not self.kv_descs:
                raise Exception("NIXL DRAM registration failed for KV tensors (decode)")
        else:
            # PREFILL side: lazy — create metadata only, no DRAM allocation.
            # MRs registered per-transfer in _staging_get_xfer_descs.
            for gpu_ptr, data_len in zip(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            ):
                staging.create(gpu_ptr, data_len, allocate=False)
            logger.info(
                "ROCm staging PREFILL: %d lazy buffers (no pre-alloc)",
                len(self.kv_args.kv_data_ptrs),
            )
            self.kv_descs = True  # placeholder

        # Aux buffers are small — always register
        aux_addrs = []
        for ptr, length in zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens):
            aux_addrs.append((ptr, length, 0, ""))
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM")
        if not self.aux_descs:
            raise Exception("NIXL DRAM registration failed for aux tensors")

        # State buffers — same pattern as KV
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            if is_decode:
                state_addrs = []
                for gpu_ptr, data_len in zip(
                    self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
                ):
                    host_ptr = staging.create(gpu_ptr, data_len, allocate=True)
                    offset = 0
                    while offset < data_len:
                        chunk = min(MAX_CHUNK, data_len - offset)
                        state_addrs.append((host_ptr + offset, chunk, 0, ""))
                        offset += chunk
                self.state_descs = self.agent.register_memory(state_addrs, "DRAM")
            else:
                for gpu_ptr, data_len in zip(
                    self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
                ):
                    staging.create(gpu_ptr, data_len, allocate=False)
                self.state_descs = True

        self._active_mr_descs = []

    NixlKVManager.register_buffer_to_engine = _patched_register

    # ---- 3. _staging_post_copy_h2d (new helper) ----------------------------
    def _staging_post_copy_h2d(self, kv_indices: npt.NDArray[np.int32]):
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

            # MR LRU cache handles deregistration — no per-transfer cleanup needed
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
