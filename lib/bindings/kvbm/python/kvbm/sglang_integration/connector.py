# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KVBM connector for SGLang.

Integrates Dynamo's KV Block Manager (KVBM) with SGLang's HiRadixCache
to provide GPU→CPU KV cache offloading. KVBM manages a CPU DRAM pool that
stores evicted KV blocks, allowing them to be reloaded on cache hits in
subsequent turns without recomputation.

The connector replaces SGLang's default HostKVCache with a KVBM-managed
host pool, reusing the existing HiRadixCache write-through / write-back
eviction machinery.

Environment variables:
    DYN_KVBM_CPU_CACHE_GB : float
        Size of the CPU DRAM pool in GB (default: 0, meaning disabled).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Optional

import torch

from kvbm import BlockManager
from kvbm.utils import is_dyn_runtime_enabled, nvtx_annotate

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)

DistributedRuntime = None
if is_dyn_runtime_enabled():
    from dynamo.runtime import DistributedRuntime

_ENV_CPU_CACHE_GB = "DYN_KVBM_CPU_CACHE_GB"


def _get_cpu_cache_gb() -> float:
    val = os.environ.get(_ENV_CPU_CACHE_GB, "0")
    try:
        gb = float(val)
    except ValueError:
        logger.warning(
            "%s='%s' is not a valid float, defaulting to 0", _ENV_CPU_CACHE_GB, val
        )
        gb = 0.0
    return gb


class SglangKvbmConnector:
    """
    Bridge between KVBM's BlockManager and SGLang's hierarchical cache.

    This connector allocates a pinned CPU memory pool via KVBM's BlockManager
    and exposes the ``backup_from_device`` / ``load_to_device`` primitives that
    SGLang's HiRadixCache cache controller calls during eviction and reload.

    Lifecycle:
        1. Instantiated once per SGLang worker process.
        2. ``register_kv_caches`` is called after the GPU KV pool is created,
           which initialises the KVBM BlockManager with the correct geometry.
        3. The connector is then used by the cache controller for all
           GPU↔CPU transfers.

    Thread-safety: all mutable state is protected by ``self._lock``.
    """

    def __init__(
        self,
        worker_id: int = 0,
        device_id: int = 0,
    ):
        self._worker_id = worker_id
        self._device_id = device_id
        self._block_manager: Optional[BlockManager] = None
        self._registered = False
        self._lock = threading.Lock()

        drt: Optional[object] = None
        if is_dyn_runtime_enabled():
            drt = DistributedRuntime.detached()
        self._drt = drt

        self._cpu_cache_gb = _get_cpu_cache_gb()
        if self._cpu_cache_gb > 0:
            logger.info(
                "KVBM SGLang connector: CPU cache pool = %.1f GB (worker %d, device %d)",
                self._cpu_cache_gb,
                worker_id,
                device_id,
            )
        else:
            logger.info(
                "KVBM SGLang connector: CPU cache disabled (%s not set or 0)",
                _ENV_CPU_CACHE_GB,
            )

    @property
    def enabled(self) -> bool:
        return self._cpu_cache_gb > 0

    @property
    def registered(self) -> bool:
        return self._registered

    @nvtx_annotate(category="sglang")
    def register_kv_caches(
        self,
        device_pool: "KVCache",
        page_size: int,
        num_layers: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """
        Register the GPU KV cache geometry with KVBM and allocate the CPU pool.

        Must be called exactly once, after the SGLang device pool is initialised.

        Args:
            device_pool: SGLang's GPU-side KVCache instance.
            page_size:   Number of tokens per KV page.
            num_layers:  Number of transformer layers.
            head_num:    Number of KV attention heads per layer.
            head_dim:    Dimension of each attention head.
            dtype:       Element dtype of the KV cache (e.g. ``torch.float16``).
        """
        if not self.enabled:
            logger.warning("register_kv_caches called but KVBM CPU cache is disabled")
            return

        with self._lock:
            if self._registered:
                raise RuntimeError("KVBM SGLang connector: already registered")

            inner_dim = 2 * head_num * head_dim  # K + V per token per layer

            bytes_per_block = page_size * inner_dim * dtype.itemsize * num_layers
            host_num_blocks = int(self._cpu_cache_gb * 1e9 / bytes_per_block)

            if host_num_blocks <= 0:
                logger.error(
                    "KVBM: computed 0 host blocks for %.1f GB "
                    "(block size = %d bytes). Disabling CPU offload.",
                    self._cpu_cache_gb,
                    bytes_per_block,
                )
                return

            dtype_str = {
                torch.float16: "fp16",
                torch.bfloat16: "bf16",
                torch.float32: "fp32",
                torch.float8_e4m3fn: "fp8_e4m3",
            }.get(dtype, "fp16")

            self._block_manager = BlockManager(
                worker_id=self._worker_id,
                num_layer=num_layers,
                page_size=page_size,
                inner_dim=inner_dim,
                dtype=dtype_str,
                host_num_blocks=host_num_blocks,
                device_num_blocks=None,
                device_id=self._device_id,
            )
            self._registered = True

            logger.info(
                "KVBM SGLang connector: registered %d host blocks "
                "(%.1f GB, page_size=%d, layers=%d, inner_dim=%d, dtype=%s)",
                host_num_blocks,
                self._cpu_cache_gb,
                page_size,
                num_layers,
                inner_dim,
                dtype_str,
            )

    @nvtx_annotate(category="sglang")
    def allocate_host_blocks(self, count: int):
        """
        Allocate ``count`` blocks in the KVBM CPU pool.

        Returns:
            A ``BlockList`` from KVBM, or ``None`` if allocation fails.
        """
        if not self._registered or self._block_manager is None:
            return None
        with self._lock:
            try:
                return self._block_manager.allocate_host_blocks_blocking(count)
            except Exception:
                logger.exception("KVBM: host block allocation failed (count=%d)", count)
                return None

    @nvtx_annotate(category="sglang")
    def backup_from_device(
        self,
        device_k_buffers: list[torch.Tensor],
        device_v_buffers: list[torch.Tensor],
        device_indices: torch.Tensor,
        host_blocks,
        num_layers: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Copy KV data from GPU paged buffers into KVBM host blocks.

        This is the GPU→CPU path invoked during HiRadixCache eviction.

        Args:
            device_k_buffers: Per-layer K tensors on GPU (list of length num_layers).
            device_v_buffers: Per-layer V tensors on GPU (list of length num_layers).
            device_indices:   Token indices into the GPU pool to copy.
            host_blocks:      KVBM BlockList previously allocated via ``allocate_host_blocks``.
            num_layers:       Number of layers to transfer.
            stream:           CUDA stream for the copy (default: current stream).
        """
        if host_blocks is None:
            return

        use_stream = stream or torch.cuda.current_stream(self._device_id)

        with torch.cuda.stream(use_stream):
            for layer_idx in range(num_layers):
                host_block = host_blocks[layer_idx] if layer_idx < len(host_blocks) else None
                if host_block is None:
                    continue
                host_tensor = torch.from_dlpack(host_block)

                k_src = device_k_buffers[layer_idx]
                v_src = device_v_buffers[layer_idx]

                src_data = torch.cat(
                    [
                        k_src[device_indices].contiguous(),
                        v_src[device_indices].contiguous(),
                    ],
                    dim=-1,
                )
                host_tensor.copy_(src_data.view(host_tensor.shape), non_blocking=True)

    @nvtx_annotate(category="sglang")
    def load_to_device(
        self,
        device_k_buffers: list[torch.Tensor],
        device_v_buffers: list[torch.Tensor],
        device_indices: torch.Tensor,
        host_blocks,
        num_layers: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Copy KV data from KVBM host blocks back into GPU paged buffers.

        This is the CPU→GPU path invoked during HiRadixCache cache-hit reload.

        Args:
            device_k_buffers: Per-layer K tensors on GPU (list of length num_layers).
            device_v_buffers: Per-layer V tensors on GPU (list of length num_layers).
            device_indices:   Token indices into the GPU pool to write.
            host_blocks:      KVBM BlockList containing the cached KV data.
            num_layers:       Number of layers to transfer.
            stream:           CUDA stream for the copy (default: current stream).
        """
        if host_blocks is None:
            return

        use_stream = stream or torch.cuda.current_stream(self._device_id)
        head_dim_k = device_k_buffers[0].shape[-1] if len(device_k_buffers) > 0 else 0

        with torch.cuda.stream(use_stream):
            for layer_idx in range(num_layers):
                host_block = host_blocks[layer_idx] if layer_idx < len(host_blocks) else None
                if host_block is None:
                    continue
                host_tensor = torch.from_dlpack(host_block)

                flat = host_tensor.flatten()
                midpoint = flat.numel() // 2
                k_data = flat[:midpoint].view_as(device_k_buffers[layer_idx][device_indices])
                v_data = flat[midpoint:].view_as(device_v_buffers[layer_idx][device_indices])

                device_k_buffers[layer_idx][device_indices] = k_data
                device_v_buffers[layer_idx][device_indices] = v_data

    def shutdown(self) -> None:
        """Release all KVBM resources."""
        with self._lock:
            self._block_manager = None
            self._registered = False
            logger.info("KVBM SGLang connector: shut down")
