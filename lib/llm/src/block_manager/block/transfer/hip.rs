// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HIP (ROCm) asynchronous memory copy operations for block transfers.
//!
//! ROCm equivalent of `transfer/cuda.rs`. Provides per-block and per-layer
//! memcpy helpers using the GPU HAL's `HipBackend` for `hipMemcpyAsync`.
//!
//! The vectorized-copy kernel path (FATBIN loading) is not available on HIP;
//! `copy_blocks_with_customized_kernel` returns `Err` so the caller can
//! fall back to the per-layer memcpy path.

use super::*;

use super::TransferError;
use crate::block_manager::block::{BlockDataProvider, BlockDataProviderMut};

use anyhow::Result;
use dynamo_memory::gpu::hip::HipBackend;
use dynamo_memory::gpu::types::*;
use dynamo_memory::gpu::GpuDevice;
use std::ops::Range;

type HipMemcpyFnPtr = unsafe fn(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: StreamHandle,
) -> Result<(), TransferError>;

fn hip_memcpy_fn_ptr(strategy: &TransferStrategy) -> Result<HipMemcpyFnPtr, TransferError> {
    match strategy {
        TransferStrategy::CudaAsyncH2D => Ok(hip_memcpy_h2d),
        TransferStrategy::CudaAsyncD2H => Ok(hip_memcpy_d2h),
        TransferStrategy::CudaAsyncD2D => Ok(hip_memcpy_d2d),
        _ => Err(TransferError::ExecutionError(
            "Unsupported copy strategy for HIP memcpy async".into(),
        )),
    }
}

/// Stub: the vectorized-copy kernel is CUDA-specific (requires FATBIN/NVRTC).
/// Returns an error so callers fall back to the per-layer memcpy path.
pub fn copy_blocks_with_customized_kernel<'a, Source, Destination>(
    _sources: &'a [Source],
    _destinations: &'a mut [Destination],
    _stream: StreamHandle,
    _ctx: &crate::block_manager::block::transfer::TransferContext,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    Err(TransferError::ExecutionError(
        "Vectorized copy kernel is not available on HIP/ROCm; use per-layer memcpy".into(),
    ))
}

/// Copy a block from a source to a destination using HIP memcpy
pub fn copy_block<'a, Source, Destination>(
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: StreamHandle,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = hip_memcpy_fn_ptr(&strategy)?;

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_view = src_data.block_view()?;
        let mut dst_view = dst_data.block_view_mut()?;

        debug_assert_eq!(src_view.size(), dst_view.size());

        unsafe {
            memcpy_fn(
                src_view.as_ptr(),
                dst_view.as_mut_ptr(),
                src_view.size(),
                stream,
            )?;
        }
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        copy_layers(
            0..src_data.num_layers(),
            sources,
            destinations,
            stream,
            strategy,
        )?;
    }
    Ok(())
}

/// Copy a range of layers from a source to a destination using HIP memcpy
pub fn copy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: StreamHandle,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = hip_memcpy_fn_ptr(&strategy)?;

    for layer_idx in layer_range {
        for outer_idx in 0..src_data.num_outer_dims() {
            let src_view = src_data.layer_view(layer_idx, outer_idx)?;
            let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

            debug_assert_eq!(src_view.size(), dst_view.size());

            unsafe {
                memcpy_fn(
                    src_view.as_ptr(),
                    dst_view.as_mut_ptr(),
                    src_view.size(),
                    stream,
                )?;
            }
        }
    }
    Ok(())
}

/// H2D Implementation
#[inline(always)]
unsafe fn hip_memcpy_h2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: StreamHandle,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source host pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination device pointer is null");

    HipBackend::memcpy_htod_async(DevicePtr(dst_ptr as u64), src_ptr, size, stream)
        .map_err(|e| TransferError::ExecutionError(format!("HIP H2D memcpy failed: {}", e)))
}

/// D2H Implementation
#[inline(always)]
unsafe fn hip_memcpy_d2h(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: StreamHandle,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination host pointer is null");

    HipBackend::memcpy_dtoh_async(dst_ptr, DevicePtr(src_ptr as u64), size, stream)
        .map_err(|e| TransferError::ExecutionError(format!("HIP D2H memcpy failed: {}", e)))
}

/// D2D Implementation
#[inline(always)]
unsafe fn hip_memcpy_d2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: StreamHandle,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination device pointer is null");
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination device memory regions must not overlap for D2D copy"
    );

    HipBackend::memcpy_dtod_async(
        DevicePtr(dst_ptr as u64),
        DevicePtr(src_ptr as u64),
        size,
        stream,
    )
    .map_err(|e| TransferError::ExecutionError(format!("HIP D2D memcpy failed: {}", e)))
}
