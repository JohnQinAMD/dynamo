// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HIP (ROCm) executor for GPU memory transfers.
//!
//! ROCm equivalent of `executor/cuda.rs`. Uses the GPU HAL's `HipBackend`
//! for async memcpy operations between host and device memory.

use super::TransferContext;
use super::{PhysicalLayout, TransferStrategy};
use crate::block_manager::v2::physical::transfer::context::TransferCompleteNotification;
use anyhow::{Result, anyhow};
use dynamo_memory::gpu::hip::HipBackend;
use dynamo_memory::gpu::types::*;
use dynamo_memory::gpu::GpuDevice;
use std::ops::Range;

/// Execute a HIP transfer between host and device memory.
///
/// This executor handles transfers involving GPU memory using HIP APIs.
/// Supports async and blocking transfers depending on the strategy.
///
/// # Arguments
/// * `src` - Source physical layout
/// * `dst` - Destination physical layout
/// * `src_block_ids` - Source block IDs to transfer
/// * `dst_block_ids` - Destination block IDs to transfer
/// * `layer_range` - Optional range of layers to transfer (None = all layers)
/// * `strategy` - Transfer strategy (H2D, D2H, D2D, async or blocking)
/// * `ctx` - Transfer context with stream
pub fn execute_hip_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layer_range: Option<Range<usize>>,
    strategy: TransferStrategy,
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

    let layers = layer_range.unwrap_or(0..src_layout.num_layers());

    let stream = match strategy {
        TransferStrategy::CudaAsyncD2H | TransferStrategy::CudaBlockingD2H => ctx.d2h_stream(),
        _ => ctx.h2d_stream(),
    };

    // Convert cudarc CudaStream to raw StreamHandle for HIP HAL
    let stream_handle = StreamHandle(unsafe { stream.cu_stream() as *mut std::ffi::c_void });

    match strategy {
        TransferStrategy::CudaAsyncH2D => {
            execute_h2d(src, dst, src_block_ids, dst_block_ids, layers, stream_handle)?;
        }
        TransferStrategy::CudaAsyncD2H => {
            execute_d2h(src, dst, src_block_ids, dst_block_ids, layers, stream_handle)?;
        }
        TransferStrategy::CudaAsyncD2D => {
            execute_d2d(src, dst, src_block_ids, dst_block_ids, layers, stream_handle)?;
        }
        TransferStrategy::CudaBlockingH2D => {
            execute_h2d(src, dst, src_block_ids, dst_block_ids, layers, stream_handle)?;
            HipBackend::stream_synchronize(stream_handle)
                .map_err(|e| anyhow!("HIP stream sync failed: {}", e))?;
        }
        TransferStrategy::CudaBlockingD2H => {
            execute_d2h(src, dst, src_block_ids, dst_block_ids, layers, stream_handle)?;
            HipBackend::stream_synchronize(stream_handle)
                .map_err(|e| anyhow!("HIP stream sync failed: {}", e))?;
        }
        _ => {
            return Err(anyhow!("Invalid HIP transfer strategy: {:?}", strategy));
        }
    }

    if matches!(
        strategy,
        TransferStrategy::CudaAsyncH2D
            | TransferStrategy::CudaAsyncD2H
            | TransferStrategy::CudaAsyncD2D
    ) {
        let event = stream.record_event(None)?;
        Ok(ctx.register_cuda_event(event))
    } else {
        Ok(TransferCompleteNotification::completed())
    }
}

/// Execute host-to-device transfer via HIP HAL.
fn execute_h2d(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layers: Range<usize>,
    stream: StreamHandle,
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

                HipBackend::memcpy_htod_async(
                    DevicePtr(dst_region.addr() as u64),
                    src_region.addr() as *const u8,
                    src_region.size(),
                    stream,
                )
                .map_err(|e| anyhow!("HIP H2D memcpy failed: {}", e))?;
            }
        }
    }
    Ok(())
}

/// Execute device-to-host transfer via HIP HAL.
fn execute_d2h(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layers: Range<usize>,
    stream: StreamHandle,
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

                HipBackend::memcpy_dtoh_async(
                    dst_region.addr() as *mut u8,
                    DevicePtr(src_region.addr() as u64),
                    src_region.size(),
                    stream,
                )
                .map_err(|e| anyhow!("HIP D2H memcpy failed: {}", e))?;
            }
        }
    }
    Ok(())
}

/// Execute device-to-device transfer via HIP HAL.
fn execute_d2d(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layers: Range<usize>,
    stream: StreamHandle,
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

                HipBackend::memcpy_dtod_async(
                    DevicePtr(dst_region.addr() as u64),
                    DevicePtr(src_region.addr() as u64),
                    src_region.size(),
                    stream,
                )
                .map_err(|e| anyhow!("HIP D2D memcpy failed: {}", e))?;
            }
        }
    }
    Ok(())
}
