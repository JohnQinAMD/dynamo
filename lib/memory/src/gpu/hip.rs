// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HIP (ROCm) GPU backend implementation via raw FFI to `libamdhip64`.
//!
//! This module provides a [`HipBackend`] type that implements [`GpuDevice`] by
//! calling HIP runtime functions directly through `extern "C"` FFI. No Rust
//! wrapper crate is required — the linker resolves symbols from `libamdhip64.so`
//! at load time.

use super::types::*;
use std::ffi::{CStr, c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// HIP error codes (from hip_runtime_api.h)
// ---------------------------------------------------------------------------

const HIP_SUCCESS: i32 = 0;
const HIP_ERROR_INVALID_VALUE: i32 = 1;
const HIP_ERROR_OUT_OF_MEMORY: i32 = 2;
const HIP_ERROR_NOT_INITIALIZED: i32 = 3;
const HIP_ERROR_NOT_SUPPORTED: i32 = 801;
// hipErrorNotReady — returned by hipEventQuery when work is still pending.
const HIP_ERROR_NOT_READY: i32 = 600;

// ---------------------------------------------------------------------------
// HIP memcpy direction constants (hipMemcpyKind)
// ---------------------------------------------------------------------------

const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
const HIP_MEMCPY_DEFAULT: i32 = 4;

// ---------------------------------------------------------------------------
// Minimal device-properties struct (only the fields we read)
// ---------------------------------------------------------------------------

/// Partial mirror of `hipDeviceProp_t`.
///
/// HIP's full struct is very large (~800+ bytes). We only access `name` and
/// `totalGlobalMem`; the trailing padding ensures we pass a buffer at least as
/// large as the real struct so the driver doesn't write out of bounds.
///
/// Layout from hip_runtime_api.h:
///   char name[256]           — offset 0
///   hipUUID uuid             — offset 256 (16 bytes)
///   char luid[8]             — offset 272
///   unsigned int luidMask    — offset 280
///   size_t totalGlobalMem    — offset 288 (with 4 bytes alignment padding)
#[repr(C)]
struct HipDeviceProperties {
    name: [c_char; 256],
    _uuid: [u8; 16],
    _luid: [u8; 8],
    _luid_device_node_mask: u32,
    _pad_align: u32,
    total_global_mem: usize,
    _padding: [u8; 2048],
}

// ---------------------------------------------------------------------------
// Raw FFI declarations — linked against libamdhip64.so
// ---------------------------------------------------------------------------

#[link(name = "amdhip64")]
extern "C" {
    fn hipGetDeviceCount(count: *mut c_int) -> i32;
    fn hipSetDevice(device_id: c_int) -> i32;
    fn hipDevicePrimaryCtxRetain(ctx: *mut *mut c_void, device: c_int) -> i32;
    fn hipCtxSetCurrent(ctx: *mut c_void) -> i32;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn hipFree(ptr: *mut c_void) -> i32;
    fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn hipHostFree(ptr: *mut c_void) -> i32;
    fn hipStreamCreate(stream: *mut *mut c_void) -> i32;
    fn hipStreamDestroy(stream: *mut c_void) -> i32;
    fn hipStreamSynchronize(stream: *mut c_void) -> i32;
    fn hipEventCreate(event: *mut *mut c_void) -> i32;
    fn hipEventDestroy(event: *mut c_void) -> i32;
    fn hipEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn hipEventQuery(event: *mut c_void) -> i32;
    fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> i32;
    fn hipGetDeviceProperties(prop: *mut HipDeviceProperties, device: c_int) -> i32;
    fn hipDeviceTotalMem(bytes: *mut usize, device: c_int) -> i32;
    fn hipGetErrorString(error: i32) -> *const c_char;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Translates a HIP error code into a [`GpuError`].
fn hip_error(code: i32) -> GpuError {
    match code {
        HIP_ERROR_OUT_OF_MEMORY => GpuError::OutOfMemory,
        HIP_ERROR_INVALID_VALUE => GpuError::InvalidValue,
        HIP_ERROR_NOT_INITIALIZED => GpuError::NotInitialized,
        HIP_ERROR_NOT_SUPPORTED => GpuError::NotSupported,
        _ => {
            let msg = unsafe {
                let ptr = hipGetErrorString(code);
                if ptr.is_null() {
                    format!("unknown HIP error {code}")
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };
            GpuError::DriverError(msg)
        }
    }
}

/// Converts our [`MemcpyKind`] to the HIP integer constant.
fn to_hip_memcpy_kind(kind: MemcpyKind) -> c_int {
    match kind {
        MemcpyKind::HostToDevice => HIP_MEMCPY_HOST_TO_DEVICE,
        MemcpyKind::DeviceToHost => HIP_MEMCPY_DEVICE_TO_HOST,
        MemcpyKind::DeviceToDevice => HIP_MEMCPY_DEVICE_TO_DEVICE,
        MemcpyKind::Default => HIP_MEMCPY_DEFAULT,
    }
}

/// Checks a HIP return code and returns `Ok(())` on success.
#[inline]
fn check(code: i32) -> GpuResult<()> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(hip_error(code))
    }
}

// ---------------------------------------------------------------------------
// HipBackend — implements GpuDevice
// ---------------------------------------------------------------------------

/// Zero-sized marker type for the HIP/ROCm GPU backend.
pub struct HipBackend;

impl super::GpuDevice for HipBackend {
    fn device_count() -> GpuResult<i32> {
        let mut count: c_int = 0;
        check(unsafe { hipGetDeviceCount(&mut count) })?;
        Ok(count)
    }

    fn create_context(device_id: i32) -> GpuResult<ContextHandle> {
        let mut ctx: *mut c_void = std::ptr::null_mut();
        check(unsafe { hipSetDevice(device_id) })?;
        check(unsafe { hipDevicePrimaryCtxRetain(&mut ctx, device_id) })?;
        Ok(ContextHandle(ctx))
    }

    fn set_current_context(ctx: ContextHandle) -> GpuResult<()> {
        check(unsafe { hipCtxSetCurrent(ctx.raw()) })
    }

    fn malloc(size: usize) -> GpuResult<DevicePtr> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { hipMalloc(&mut ptr, size) })?;
        Ok(DevicePtr(ptr as u64))
    }

    fn free(ptr: DevicePtr) -> GpuResult<()> {
        check(unsafe { hipFree(ptr.raw() as *mut c_void) })
    }

    fn malloc_host(size: usize) -> GpuResult<*mut u8> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // flags = 0 → default pinned allocation (hipHostMallocDefault)
        check(unsafe { hipHostMalloc(&mut ptr, size, 0) })?;
        Ok(ptr as *mut u8)
    }

    fn free_host(ptr: *mut u8) -> GpuResult<()> {
        check(unsafe { hipHostFree(ptr as *mut c_void) })
    }

    fn create_stream() -> GpuResult<StreamHandle> {
        let mut stream: *mut c_void = std::ptr::null_mut();
        check(unsafe { hipStreamCreate(&mut stream) })?;
        Ok(StreamHandle(stream))
    }

    fn destroy_stream(stream: StreamHandle) -> GpuResult<()> {
        check(unsafe { hipStreamDestroy(stream.raw()) })
    }

    fn stream_synchronize(stream: StreamHandle) -> GpuResult<()> {
        check(unsafe { hipStreamSynchronize(stream.raw()) })
    }

    fn create_event() -> GpuResult<EventHandle> {
        let mut event: *mut c_void = std::ptr::null_mut();
        check(unsafe { hipEventCreate(&mut event) })?;
        Ok(EventHandle(event))
    }

    fn destroy_event(event: EventHandle) -> GpuResult<()> {
        check(unsafe { hipEventDestroy(event.raw()) })
    }

    fn event_record(event: EventHandle, stream: StreamHandle) -> GpuResult<()> {
        check(unsafe { hipEventRecord(event.raw(), stream.raw()) })
    }

    fn event_query(event: EventHandle) -> GpuResult<bool> {
        let code = unsafe { hipEventQuery(event.raw()) };
        match code {
            HIP_SUCCESS => Ok(true),
            HIP_ERROR_NOT_READY => Ok(false),
            _ => Err(hip_error(code)),
        }
    }

    fn memcpy_async(
        dst: DevicePtr,
        src: DevicePtr,
        size: usize,
        kind: MemcpyKind,
        stream: StreamHandle,
    ) -> GpuResult<()> {
        check(unsafe {
            hipMemcpyAsync(
                dst.raw() as *mut c_void,
                src.raw() as *const c_void,
                size,
                to_hip_memcpy_kind(kind),
                stream.raw(),
            )
        })
    }

    fn memcpy_htod_async(
        dst: DevicePtr,
        src: *const u8,
        size: usize,
        stream: StreamHandle,
    ) -> GpuResult<()> {
        check(unsafe {
            hipMemcpyAsync(
                dst.raw() as *mut c_void,
                src as *const c_void,
                size,
                HIP_MEMCPY_HOST_TO_DEVICE,
                stream.raw(),
            )
        })
    }

    fn memcpy_dtoh_async(
        dst: *mut u8,
        src: DevicePtr,
        size: usize,
        stream: StreamHandle,
    ) -> GpuResult<()> {
        check(unsafe {
            hipMemcpyAsync(
                dst as *mut c_void,
                src.raw() as *const c_void,
                size,
                HIP_MEMCPY_DEVICE_TO_HOST,
                stream.raw(),
            )
        })
    }

    fn memcpy_dtod_async(
        dst: DevicePtr,
        src: DevicePtr,
        size: usize,
        stream: StreamHandle,
    ) -> GpuResult<()> {
        check(unsafe {
            hipMemcpyAsync(
                dst.raw() as *mut c_void,
                src.raw() as *const c_void,
                size,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
                stream.raw(),
            )
        })
    }

    fn device_name(device_id: i32) -> GpuResult<String> {
        let mut props: HipDeviceProperties = unsafe { std::mem::zeroed() };
        check(unsafe { hipGetDeviceProperties(&mut props, device_id) })?;
        let name = unsafe { CStr::from_ptr(props.name.as_ptr()) };
        Ok(name.to_string_lossy().into_owned())
    }

    fn total_memory(device_id: i32) -> GpuResult<usize> {
        // Use hipDeviceTotalMem() instead of hipGetDeviceProperties() to avoid
        // dependence on the HipDeviceProperties struct layout, which has changed
        // between ROCm versions (see BUG-1). This single-purpose API is stable.
        let mut bytes: usize = 0;
        check(unsafe { hipDeviceTotalMem(&mut bytes, device_id) })?;
        Ok(bytes)
    }
}
