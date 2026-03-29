// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HIP (ROCm) executor for GPU memory transfers.
//!
//! ROCm equivalent of `executor/cuda.rs` in kvbm-physical. Uses the GPU HAL's
//! `HipBackend` for async memcpy between host and device memory.
//!
//! The whole-block batched path and vectorized_copy kernel are not available on
//! HIP; this implementation uses individual `hipMemcpyAsync` calls per chunk.

use super::TransferContext;
use super::{PhysicalLayout, TransferStrategy};
use crate::BlockId;
use crate::transfer::context::TransferCompleteNotification;
use crate::transfer::{can_use_whole_block_transfer, validate_layout_compatibility};
use anyhow::{Result, anyhow};
use dynamo_memory::gpu::hip::HipBackend;
use dynamo_memory::gpu::types::*;
use dynamo_memory::gpu::GpuDevice;
use std::ops::Range;
use std::sync::Arc;

/// Execute a HIP transfer between host and device memory.
///
/// # Arguments
/// * `src` - Source physical layout
/// * `dst` - Destination physical layout
/// * `src_block_ids` - Source block IDs to transfer
/// * `dst_block_ids` - Destination block IDs to transfer
/// * `layer_range` - Optional range of layers to transfer (None = all layers)
/// * `strategy` - Transfer strategy (H2D, D2H, D2D)
/// * `hip_stream` - Optional caller-provided stream. If provided, caller manages sync.
/// * `ctx` - Transfer context
#[allow(clippy::too_many_arguments)]
pub fn execute_hip_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layer_range: Option<Range<usize>>,
    strategy: TransferStrategy,
    hip_stream: Option<StreamHandle>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    let src_layout = src.layout();
    let dst_layout = dst.layout();

    if src_layout.num_layers() != dst_layout.num_layers() {
        return Err(anyhow!(
            "Layouts have incompatible layer counts: src={}, dst={}",
            src_layout.num_layers(),
            dst_layout.num_layers()
        ));
    }

    if src_layout.outer_dim() != dst_layout.outer_dim() {
        return Err(anyhow!(
            "Layouts have incompatible outer dimensions: src={}, dst={}",
            src_layout.outer_dim(),
            dst_layout.outer_dim()
        ));
    }

    validate_layout_compatibility(src, dst)?;

    let layers = layer_range.unwrap_or(0..src_layout.num_layers());

    let caller_manages_sync = hip_stream.is_some();

    // Get stream — use caller-provided or acquire from pool and extract raw handle
    let (cudarc_stream, stream_handle) = if let Some(s) = hip_stream {
        (None, s)
    } else {
        let s = match strategy {
            TransferStrategy::CudaAsyncD2H => ctx.next_d2h_streams(),
            _ => ctx.next_h2d_streams(),
        };
        let handle = StreamHandle(unsafe { s.cu_stream() as *mut std::ffi::c_void });
        (Some(s), handle)
    };

    match strategy {
        TransferStrategy::CudaAsyncH2D
        | TransferStrategy::CudaAsyncD2H
        | TransferStrategy::CudaAsyncD2D => {
            execute_per_chunk(src, dst, src_block_ids, dst_block_ids, layers, stream_handle, strategy)?;
        }
        _ => {
            return Err(anyhow!("Invalid HIP transfer strategy: {:?}", strategy));
        }
    }

    if caller_manages_sync {
        return Ok(TransferCompleteNotification::completed());
    }

    if let Some(stream) = cudarc_stream {
        let event = stream.record_event(None)?;
        Ok(ctx.register_cuda_event(event))
    } else {
        Ok(TransferCompleteNotification::completed())
    }
}

/// Per-chunk transfer using individual hipMemcpyAsync calls.
fn execute_per_chunk(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layers: Range<usize>,
    stream: StreamHandle,
    strategy: TransferStrategy,
) -> Result<()> {
    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        for layer_id in layers.clone() {
            for outer_id in 0..src.layout().outer_dim() {
                let src_region = src.memory_region(src_block_id, layer_id, outer_id)?;
                let dst_region = dst.memory_region(dst_block_id, layer_id, outer_id)?;

                if src_region.size() != dst_region.size() {
                    return Err(anyhow!(
                        "Size mismatch at block=({},{}), layer={}, outer={}: src={}, dst={}",
                        src_block_id, dst_block_id, layer_id, outer_id,
                        src_region.size(), dst_region.size()
                    ));
                }

                match strategy {
                    TransferStrategy::CudaAsyncH2D => {
                        HipBackend::memcpy_htod_async(
                            DevicePtr(dst_region.addr() as u64),
                            src_region.addr() as *const u8,
                            src_region.size(),
                            stream,
                        )
                        .map_err(|e| anyhow!("HIP H2D memcpy failed: {}", e))?;
                    }
                    TransferStrategy::CudaAsyncD2H => {
                        HipBackend::memcpy_dtoh_async(
                            dst_region.addr() as *mut u8,
                            DevicePtr(src_region.addr() as u64),
                            src_region.size(),
                            stream,
                        )
                        .map_err(|e| anyhow!("HIP D2H memcpy failed: {}", e))?;
                    }
                    TransferStrategy::CudaAsyncD2D => {
                        HipBackend::memcpy_dtod_async(
                            DevicePtr(dst_region.addr() as u64),
                            DevicePtr(src_region.addr() as u64),
                            src_region.size(),
                            stream,
                        )
                        .map_err(|e| anyhow!("HIP D2D memcpy failed: {}", e))?;
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
    Ok(())
}
