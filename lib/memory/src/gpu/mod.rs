// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GPU Hardware Abstraction Layer (HAL).
//!
//! Provides a unified, trait-based interface for GPU operations that works with
//! both CUDA (via `cudarc`) and HIP (via raw FFI to `libamdhip64`) backends.
//!
//! # Feature gates
//!
//! * `cuda` — enables the [`cuda`] module (CUDA backend via cudarc).
//! * `rocm` — enables the [`hip`] module (HIP/ROCm backend via raw FFI).
//!
//! At most one backend should be enabled per build.

pub mod types;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "rocm")]
pub mod hip;

pub use types::*;

/// Core GPU device operations.
///
/// All methods are associated functions (no `&self`) because the underlying
/// GPU runtimes use global, thread-local driver state rather than per-object
/// state. Implementations are zero-sized marker types (`CudaBackend`,
/// `HipBackend`) that exist only to carry the impl.
pub trait GpuDevice: Send + Sync {
    /// Returns the number of visible GPU devices.
    fn device_count() -> GpuResult<i32>;

    /// Retains (or creates) a primary context for `device_id`.
    fn create_context(device_id: i32) -> GpuResult<ContextHandle>;

    /// Makes `ctx` the current context on the calling thread.
    fn set_current_context(ctx: ContextHandle) -> GpuResult<()>;

    /// Allocates `size` bytes of device memory.
    fn malloc(size: usize) -> GpuResult<DevicePtr>;

    /// Frees device memory previously allocated with [`GpuDevice::malloc`].
    fn free(ptr: DevicePtr) -> GpuResult<()>;

    /// Allocates `size` bytes of page-locked (pinned) host memory.
    fn malloc_host(size: usize) -> GpuResult<*mut u8>;

    /// Frees pinned host memory previously allocated with [`GpuDevice::malloc_host`].
    fn free_host(ptr: *mut u8) -> GpuResult<()>;

    /// Creates a new stream on the current device/context.
    fn create_stream() -> GpuResult<StreamHandle>;

    /// Destroys a stream.
    fn destroy_stream(stream: StreamHandle) -> GpuResult<()>;

    /// Blocks the calling thread until all work on `stream` completes.
    fn stream_synchronize(stream: StreamHandle) -> GpuResult<()>;

    /// Creates a new event.
    fn create_event() -> GpuResult<EventHandle>;

    /// Destroys an event.
    fn destroy_event(event: EventHandle) -> GpuResult<()>;

    /// Records `event` at the current tail of `stream`.
    fn event_record(event: EventHandle, stream: StreamHandle) -> GpuResult<()>;

    /// Returns `Ok(true)` if all work captured by `event` has completed,
    /// `Ok(false)` if work is still pending. Non-blocking.
    fn event_query(event: EventHandle) -> GpuResult<bool>;

    /// Asynchronous memcpy of `size` bytes, direction given by `kind`.
    fn memcpy_async(
        dst: DevicePtr,
        src: DevicePtr,
        size: usize,
        kind: MemcpyKind,
        stream: StreamHandle,
    ) -> GpuResult<()>;

    /// Asynchronous host-to-device copy of `size` bytes.
    fn memcpy_htod_async(
        dst: DevicePtr,
        src: *const u8,
        size: usize,
        stream: StreamHandle,
    ) -> GpuResult<()>;

    /// Asynchronous device-to-host copy of `size` bytes.
    fn memcpy_dtoh_async(
        dst: *mut u8,
        src: DevicePtr,
        size: usize,
        stream: StreamHandle,
    ) -> GpuResult<()>;

    /// Asynchronous device-to-device copy of `size` bytes.
    fn memcpy_dtod_async(
        dst: DevicePtr,
        src: DevicePtr,
        size: usize,
        stream: StreamHandle,
    ) -> GpuResult<()>;

    /// Returns a human-readable name for `device_id` (e.g. "NVIDIA A100").
    fn device_name(device_id: i32) -> GpuResult<String>;

    /// Returns the total device memory in bytes for `device_id`.
    fn total_memory(device_id: i32) -> GpuResult<usize>;
}

/// Returns the name of the GPU backend enabled at compile time.
///
/// Possible values: `"cuda"`, `"rocm"`, or `"none"`.
pub fn detect_backend() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        return "cuda";
    }

    #[cfg(feature = "rocm")]
    {
        return "rocm";
    }

    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        "none"
    }
}
