// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common GPU types shared between CUDA and HIP backends.
//!
//! These types provide a backend-agnostic representation of GPU resources
//! (device pointers, streams, events, contexts) so that higher-level code
//! can operate without knowing whether the underlying runtime is CUDA or HIP.

use std::ffi::c_void;

/// Opaque device pointer (maps to `CUdeviceptr` / `hipDevicePtr_t`).
///
/// Wraps a 64-bit device address. The value is intentionally opaque — callers
/// should not dereference it from host code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    /// Returns the raw 64-bit device address.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Returns `true` if the pointer is null (address 0).
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

impl From<u64> for DevicePtr {
    fn from(addr: u64) -> Self {
        Self(addr)
    }
}

/// Opaque stream handle (maps to `CUstream` / `hipStream_t`).
#[derive(Debug, Clone, Copy)]
pub struct StreamHandle(pub *mut c_void);

impl StreamHandle {
    /// A null stream handle, representing the default/legacy stream.
    pub const NULL: Self = Self(std::ptr::null_mut());

    /// Returns the raw pointer.
    pub fn raw(self) -> *mut c_void {
        self.0
    }
}

/// Opaque event handle (maps to `CUevent` / `hipEvent_t`).
#[derive(Debug, Clone, Copy)]
pub struct EventHandle(pub *mut c_void);

impl EventHandle {
    /// Returns the raw pointer.
    pub fn raw(self) -> *mut c_void {
        self.0
    }
}

/// Opaque context handle (maps to `CUcontext` / `hipCtx_t`).
#[derive(Debug, Clone, Copy)]
pub struct ContextHandle(pub *mut c_void);

impl ContextHandle {
    /// Returns the raw pointer.
    pub fn raw(self) -> *mut c_void {
        self.0
    }
}

/// GPU memory copy direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    /// Host → Device
    HostToDevice,
    /// Device → Host
    DeviceToHost,
    /// Device → Device
    DeviceToDevice,
    /// Runtime-inferred direction
    Default,
}

/// Errors originating from GPU driver or runtime calls.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// A named driver/runtime error with a human-readable message.
    #[error("GPU driver error: {0}")]
    DriverError(String),

    /// The device ran out of memory during an allocation.
    #[error("GPU out of memory")]
    OutOfMemory,

    /// An argument was invalid (null pointer, bad size, etc.).
    #[error("GPU invalid value")]
    InvalidValue,

    /// The GPU runtime has not been initialized.
    #[error("GPU not initialized")]
    NotInitialized,

    /// The requested operation is not supported on this device/backend.
    #[error("GPU not supported")]
    NotSupported,

    /// A numeric error code that does not map to a known variant.
    #[error("GPU error code {0}")]
    ErrorCode(i32),
}

/// Convenience alias for results from GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;

// SAFETY: The handles below are opaque pointers to GPU runtime objects whose
// lifetimes are managed by the driver. They do not alias mutable host memory
// and are safe to move between threads.
unsafe impl Send for StreamHandle {}
unsafe impl Sync for StreamHandle {}
unsafe impl Send for EventHandle {}
unsafe impl Sync for EventHandle {}
unsafe impl Send for ContextHandle {}
unsafe impl Sync for ContextHandle {}
