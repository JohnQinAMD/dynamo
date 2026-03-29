// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA GPU backend implementation wrapping `cudarc`.
//!
//! [`CudaBackend`] implements [`GpuDevice`] by delegating to `cudarc`'s
//! low-level driver result functions, keeping the HAL layer thin.

use super::types::*;
use cudarc::driver::{result, sys};
use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Converts a `cudarc` `DriverError` into our backend-agnostic [`GpuError`].
fn cuda_err(e: cudarc::driver::DriverError) -> GpuError {
    let msg = format!("{e}");
    if msg.contains("OUT_OF_MEMORY") || msg.contains("CUDA_ERROR_OUT_OF_MEMORY") {
        GpuError::OutOfMemory
    } else if msg.contains("INVALID_VALUE") {
        GpuError::InvalidValue
    } else if msg.contains("NOT_INITIALIZED") {
        GpuError::NotInitialized
    } else {
        GpuError::DriverError(msg)
    }
}

/// Wraps a raw `CUresult` into a [`GpuResult`].
fn check_cu(res: sys::CUresult) -> GpuResult<()> {
    if res == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::ErrorCode(res as i32))
    }
}

// ---------------------------------------------------------------------------
// CudaBackend — implements GpuDevice
// ---------------------------------------------------------------------------

/// Zero-sized marker type for the CUDA GPU backend (wraps cudarc).
pub struct CudaBackend;

impl super::GpuDevice for CudaBackend {
    fn device_count() -> GpuResult<i32> {
        result::device::get_count().map_err(cuda_err)
    }

    fn create_context(device_id: i32) -> GpuResult<ContextHandle> {
        let dev = result::device::get(device_id as usize).map_err(cuda_err)?;
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        check_cu(unsafe { sys::cuDevicePrimaryCtxRetain(&mut ctx, dev) })?;
        Ok(ContextHandle(ctx as *mut c_void))
    }

    fn set_current_context(ctx: ContextHandle) -> GpuResult<()> {
        check_cu(unsafe { sys::cuCtxSetCurrent(ctx.raw() as sys::CUcontext) })
    }

    fn malloc(size: usize) -> GpuResult<DevicePtr> {
        let ptr = unsafe { result::malloc_sync(size).map_err(cuda_err)? };
        Ok(DevicePtr(ptr))
    }

    fn free(ptr: DevicePtr) -> GpuResult<()> {
        unsafe { result::free_sync(ptr.raw()).map_err(cuda_err) }
    }

    fn malloc_host(size: usize) -> GpuResult<*mut u8> {
        let ptr = unsafe {
            result::malloc_host(size, sys::CU_MEMHOSTALLOC_DEVICEMAP).map_err(cuda_err)?
        };
        Ok(ptr as *mut u8)
    }

    fn free_host(ptr: *mut u8) -> GpuResult<()> {
        unsafe { result::free_host(ptr as *mut c_void).map_err(cuda_err) }
    }

    fn create_stream() -> GpuResult<StreamHandle> {
        let mut stream: sys::CUstream = std::ptr::null_mut();
        check_cu(unsafe { sys::cuStreamCreate(&mut stream, 0) })?;
        Ok(StreamHandle(stream as *mut c_void))
    }

    fn destroy_stream(stream: StreamHandle) -> GpuResult<()> {
        check_cu(unsafe { sys::cuStreamDestroy_v2(stream.raw() as sys::CUstream) })
    }

    fn stream_synchronize(stream: StreamHandle) -> GpuResult<()> {
        check_cu(unsafe { sys::cuStreamSynchronize(stream.raw() as sys::CUstream) })
    }

    fn create_event() -> GpuResult<EventHandle> {
        let mut event: sys::CUevent = std::ptr::null_mut();
        check_cu(unsafe { sys::cuEventCreate(&mut event, 0) })?;
        Ok(EventHandle(event as *mut c_void))
    }

    fn destroy_event(event: EventHandle) -> GpuResult<()> {
        check_cu(unsafe { sys::cuEventDestroy_v2(event.raw() as sys::CUevent) })
    }

    fn event_record(event: EventHandle, stream: StreamHandle) -> GpuResult<()> {
        check_cu(unsafe {
            sys::cuEventRecord(
                event.raw() as sys::CUevent,
                stream.raw() as sys::CUstream,
            )
        })
    }

    fn event_query(event: EventHandle) -> GpuResult<bool> {
        let res = unsafe { sys::cuEventQuery(event.raw() as sys::CUevent) };
        match res {
            sys::CUresult::CUDA_SUCCESS => Ok(true),
            sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            _ => Err(GpuError::ErrorCode(res as i32)),
        }
    }

    fn memcpy_async(
        dst: DevicePtr,
        src: DevicePtr,
        size: usize,
        kind: MemcpyKind,
        stream: StreamHandle,
    ) -> GpuResult<()> {
        match kind {
            MemcpyKind::DeviceToDevice => Self::memcpy_dtod_async(dst, src, size, stream),
            MemcpyKind::HostToDevice => {
                Self::memcpy_htod_async(dst, src.raw() as *const u8, size, stream)
            }
            MemcpyKind::DeviceToHost => {
                Self::memcpy_dtoh_async(dst.raw() as *mut u8, src, size, stream)
            }
            MemcpyKind::Default => Self::memcpy_dtod_async(dst, src, size, stream),
        }
    }

    fn memcpy_htod_async(
        dst: DevicePtr,
        src: *const u8,
        size: usize,
        stream: StreamHandle,
    ) -> GpuResult<()> {
        check_cu(unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst.raw(),
                src as *const c_void,
                size,
                stream.raw() as sys::CUstream,
            )
        })
    }

    fn memcpy_dtoh_async(
        dst: *mut u8,
        src: DevicePtr,
        size: usize,
        stream: StreamHandle,
    ) -> GpuResult<()> {
        check_cu(unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                dst as *mut c_void,
                src.raw(),
                size,
                stream.raw() as sys::CUstream,
            )
        })
    }

    fn memcpy_dtod_async(
        dst: DevicePtr,
        src: DevicePtr,
        size: usize,
        stream: StreamHandle,
    ) -> GpuResult<()> {
        check_cu(unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                dst.raw(),
                src.raw(),
                size,
                stream.raw() as sys::CUstream,
            )
        })
    }

    fn device_name(device_id: i32) -> GpuResult<String> {
        let dev = result::device::get(device_id as usize).map_err(cuda_err)?;
        let mut name_buf = [0i8; 256];
        check_cu(unsafe {
            sys::cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as i32, dev)
        })?;
        let name = unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) };
        Ok(name.to_string_lossy().into_owned())
    }

    fn total_memory(device_id: i32) -> GpuResult<usize> {
        let dev = result::device::get(device_id as usize).map_err(cuda_err)?;
        let mut bytes: usize = 0;
        check_cu(unsafe { sys::cuDeviceTotalMem_v2(&mut bytes, dev) })?;
        Ok(bytes)
    }
}
