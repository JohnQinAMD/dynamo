# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DRAM staging adapter for RIXL/nixl KV cache transfer on AMD GPUs.

Problem: RIXL's UCX backend cannot register GPU VRAM directly on AMD
ionic NICs (no GPU Direct RDMA support). The `ibv_reg_mr` call for
VRAM addresses fails with NIXL_ERR_BACKEND.

Solution: Allocate pinned host (DRAM) staging buffers, register those
with RIXL, and do explicit GPU↔CPU copies around each transfer.

Data flow:
  Prefill side:  GPU KV cache → hipMemcpy D2H → DRAM staging buffer
                 → RIXL RDMA transfer → remote DRAM staging buffer
  Decode side:   remote DRAM staging buffer → hipMemcpy H2D → GPU KV cache

Performance impact: ~1-2ms extra per transfer for hipMemcpy, but enables
RIXL on AMD ionic NICs where VRAM registration fails.

Usage: Wrap the SGLang nixl connector's register/transfer calls.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


class DramStagingBuffer:
    """Manages pinned host memory for DRAM-staged RDMA transfers."""

    def __init__(self, size: int, device_id: int = 0):
        """Allocate a pinned host buffer for staging.

        Args:
            size: Buffer size in bytes.
            device_id: GPU device ID for hipMemcpy operations.
        """
        self.size = size
        self.device_id = device_id

        if torch is None:
            raise RuntimeError("PyTorch required for DRAM staging")

        self.host_tensor = torch.empty(
            size, dtype=torch.uint8, device="cpu"
        ).pin_memory()
        self.host_ptr = self.host_tensor.data_ptr()
        logger.info(
            f"DramStagingBuffer: allocated {size / (1024**2):.1f} MB "
            f"pinned host memory at {hex(self.host_ptr)}"
        )

    def copy_from_gpu(self, gpu_ptr: int, offset: int, length: int) -> None:
        """Copy data from GPU VRAM to host staging buffer."""
        gpu_tensor = torch.tensor([], dtype=torch.uint8)
        gpu_storage = torch.UntypedStorage(length, device=f"cuda:{self.device_id}")
        torch.cuda.current_stream().synchronize()
        # Use low-level memcpy
        import ctypes

        ctypes.memmove(
            self.host_ptr + offset,
            gpu_ptr,
            length,
        )

    def copy_to_gpu(self, gpu_ptr: int, offset: int, length: int) -> None:
        """Copy data from host staging buffer to GPU VRAM."""
        import ctypes

        ctypes.memmove(
            gpu_ptr,
            self.host_ptr + offset,
            length,
        )


class DramStagingManager:
    """Manages multiple DRAM staging buffers for KV cache transfer.

    Replaces VRAM registration with DRAM registration in the nixl connector.
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.buffers: dict[int, DramStagingBuffer] = {}
        self._total_allocated = 0

    def get_or_create_buffer(self, buffer_id: int, size: int) -> DramStagingBuffer:
        """Get existing staging buffer or create a new one."""
        if buffer_id not in self.buffers:
            buf = DramStagingBuffer(size, self.gpu_id)
            self.buffers[buffer_id] = buf
            self._total_allocated += size
            logger.info(
                f"DramStagingManager: buffer {buffer_id}, "
                f"total={self._total_allocated / (1024**3):.2f} GB"
            )
        return self.buffers[buffer_id]

    def stage_gpu_to_dram(
        self,
        gpu_addrs: List[Tuple[int, int]],
        buffer_id: int,
    ) -> List[Tuple[int, int]]:
        """Stage GPU buffers to DRAM before RDMA transfer.

        Args:
            gpu_addrs: List of (gpu_ptr, size) pairs.
            buffer_id: Staging buffer ID.

        Returns:
            List of (dram_ptr, size) pairs for RIXL registration.
        """
        total_size = sum(size for _, size in gpu_addrs)
        buf = self.get_or_create_buffer(buffer_id, total_size)

        dram_addrs = []
        offset = 0
        for gpu_ptr, size in gpu_addrs:
            buf.copy_from_gpu(gpu_ptr, offset, size)
            dram_addrs.append((buf.host_ptr + offset, size))
            offset += size

        return dram_addrs

    def unstage_dram_to_gpu(
        self,
        gpu_addrs: List[Tuple[int, int]],
        buffer_id: int,
    ) -> None:
        """Copy received data from DRAM staging buffer back to GPU.

        Args:
            gpu_addrs: List of (gpu_ptr, size) pairs.
            buffer_id: Staging buffer ID.
        """
        buf = self.buffers.get(buffer_id)
        if buf is None:
            raise ValueError(f"No staging buffer for id={buffer_id}")

        offset = 0
        for gpu_ptr, size in gpu_addrs:
            buf.copy_to_gpu(gpu_ptr, offset, size)
            offset += size

    def cleanup(self) -> None:
        """Release all staging buffers."""
        for buf in self.buffers.values():
            del buf.host_tensor
        self.buffers.clear()
        self._total_allocated = 0
