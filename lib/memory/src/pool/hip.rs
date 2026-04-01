// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HIP memory pool for efficient device memory allocation in hot paths.
//!
//! ROCm equivalent of `pool/cuda.rs`. Provides a safe wrapper around HIP's
//! stream-ordered allocation APIs (`hipMallocAsync` / `hipFreeAsync`),
//! backed by `hipMemPool`.
//!
//! # Thread Safety
//!
//! [`HipMemPool`] uses internal locking to serialize host-side calls to
//! the HIP driver. GPU-side operations remain stream-ordered and asynchronous.
//!
//! # Note
//!
//! HIP's memory pool API (`hipMemPoolCreate`) requires ROCm 5.2+.
//! If the runtime does not support memory pools, [`HipMemPoolBuilder::build`]
//! returns an error.

use anyhow::{Result, anyhow};
use dynamo_memory::gpu::hip::HipBackend;
use dynamo_memory::gpu::types::*;
use std::ffi::c_void;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Raw FFI declarations for HIP memory pool APIs
// ---------------------------------------------------------------------------

/// Partial mirror of `hipMemPoolProps`.
///
/// Layout from hip_runtime_api.h:
///   hipMemAllocationType    allocType;        // offset 0  (i32)
///   hipMemAllocationHandleType handleTypes;   // offset 4  (i32)
///   hipMemLocation          location;         // offset 8  { type: i32, id: i32 }
///   unsigned char            reserved[64];    // offset 16
///
/// Total size: 80 bytes. We add extra padding to be safe against future changes.
#[repr(C)]
#[allow(non_camel_case_types)]
struct hipMemPoolProps {
    alloc_type: i32,
    handle_types: i32,
    location_type: i32,
    location_id: i32,
    _reserved: [u8; 64],
    _extra_padding: [u8; 128],
}

#[link(name = "amdhip64")]
extern "C" {
    fn hipMemPoolCreate(pool: *mut *mut c_void, props: *const hipMemPoolProps) -> i32;
    fn hipMemPoolDestroy(pool: *mut c_void) -> i32;
    fn hipMallocFromPoolAsync(
        ptr: *mut *mut c_void,
        size: usize,
        pool: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn hipFreeAsync(ptr: *mut c_void, stream: *mut c_void) -> i32;
    fn hipStreamSynchronize(stream: *mut c_void) -> i32;
    fn hipMemPoolSetAttribute(pool: *mut c_void, attr: i32, value: *mut c_void) -> i32;
}

const HIP_SUCCESS: i32 = 0;
const HIP_MEM_ALLOCATION_TYPE_PINNED: i32 = 1;
const HIP_MEM_LOCATION_TYPE_DEVICE: i32 = 1;
// hipMemPoolAttrReleaseThreshold
const HIP_MEMPOOL_ATTR_RELEASE_THRESHOLD: i32 = 4;

/// Builder for creating a HIP memory pool with configurable parameters.
///
/// # Example
/// ```ignore
/// let pool = HipMemPoolBuilder::new(0, 64 * 1024 * 1024)
///     .release_threshold(32 * 1024 * 1024)
///     .build()?;
/// ```
pub struct HipMemPoolBuilder {
    device_id: i32,
    reserve_size: usize,
    release_threshold: Option<u64>,
}

impl HipMemPoolBuilder {
    /// Create a new builder with the required reserve size.
    ///
    /// # Arguments
    /// * `device_id` - HIP device ordinal
    /// * `reserve_size` - Number of bytes to pre-allocate to warm the pool
    pub fn new(device_id: i32, reserve_size: usize) -> Self {
        Self {
            device_id,
            reserve_size,
            release_threshold: None,
        }
    }

    /// Set the release threshold for the pool.
    ///
    /// Memory above this threshold is returned to the system when freed.
    pub fn release_threshold(mut self, threshold: u64) -> Self {
        self.release_threshold = Some(threshold);
        self
    }

    /// Build the HIP memory pool.
    ///
    /// This will:
    /// 1. Create the pool
    /// 2. Set the release threshold if configured
    /// 3. Pre-allocate and free memory to warm the pool
    pub fn build(self) -> Result<HipMemPool> {
        let mut props: hipMemPoolProps = unsafe { std::mem::zeroed() };
        props.alloc_type = HIP_MEM_ALLOCATION_TYPE_PINNED;
        props.location_type = HIP_MEM_LOCATION_TYPE_DEVICE;
        props.location_id = self.device_id;

        let mut pool: *mut c_void = std::ptr::null_mut();

        let result = unsafe { hipMemPoolCreate(&mut pool, &props) };
        if result != HIP_SUCCESS {
            return Err(anyhow!("hipMemPoolCreate failed with error code: {}", result));
        }

        if let Some(threshold) = self.release_threshold {
            let result = unsafe {
                hipMemPoolSetAttribute(
                    pool,
                    HIP_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut c_void,
                )
            };
            if result != HIP_SUCCESS {
                unsafe { hipMemPoolDestroy(pool) };
                return Err(anyhow!(
                    "hipMemPoolSetAttribute failed with error code: {}",
                    result
                ));
            }
        }

        let hip_pool = HipMemPool {
            inner: Mutex::new(pool),
        };

        // Warm the pool by pre-allocating and freeing memory
        if self.reserve_size > 0 {
            let stream = HipBackend::create_stream()
                .map_err(|e| anyhow!("Failed to create HIP stream for pool warming: {}", e))?;

            let ptr = hip_pool.alloc_async_raw(self.reserve_size, stream)?;
            hip_pool.free_async_raw(ptr, stream)?;

            let result = unsafe { hipStreamSynchronize(stream.raw()) };
            if result != HIP_SUCCESS {
                return Err(anyhow!(
                    "hipStreamSynchronize failed with error code: {}",
                    result
                ));
            }

            HipBackend::destroy_stream(stream)
                .map_err(|e| anyhow!("Failed to destroy warming stream: {}", e))?;
        }

        Ok(hip_pool)
    }
}

/// Safe wrapper around a HIP memory pool.
///
/// The pool amortizes allocation overhead by maintaining a reservoir of device memory.
/// Allocations are fast sub-allocations from this reservoir, and frees return memory
/// to the pool rather than the OS (until the release threshold is exceeded).
///
/// # Thread Safety
///
/// This type uses internal locking to serialize host-side calls to HIP driver APIs.
/// `hipMallocFromPoolAsync` is not host-thread reentrant, so concurrent calls from
/// multiple threads must be serialized.
///
/// Use [`HipMemPoolBuilder`] for configurable pool creation with pre-allocation.
pub struct HipMemPool {
    inner: Mutex<*mut c_void>,
}

// SAFETY: HipMemPool is Send because the Mutex serializes all host-side access
// to the pool handle, and HIP driver state is thread-safe when properly serialized.
unsafe impl Send for HipMemPool {}

// SAFETY: HipMemPool is Sync because all access to the pool handle goes through
// the Mutex, which serializes host-thread access.
unsafe impl Sync for HipMemPool {}

impl HipMemPool {
    /// Create a builder for a new HIP memory pool.
    pub fn builder(device_id: i32, reserve_size: usize) -> HipMemPoolBuilder {
        HipMemPoolBuilder::new(device_id, reserve_size)
    }

    /// Allocate memory from the pool asynchronously using a GPU HAL stream handle.
    ///
    /// The allocation is stream-ordered; the memory is available for use
    /// after all preceding operations on the stream complete.
    pub fn alloc_async(&self, size: usize, stream: StreamHandle) -> Result<u64> {
        self.alloc_async_raw(size, stream)
    }

    /// Allocate memory from the pool asynchronously (raw stream handle variant).
    ///
    /// # Host Serialization
    ///
    /// This method acquires an internal mutex because `hipMallocFromPoolAsync`
    /// is not host-thread reentrant.
    pub fn alloc_async_raw(&self, size: usize, stream: StreamHandle) -> Result<u64> {
        let pool = self
            .inner
            .lock()
            .map_err(|e| anyhow!("mutex poisoned: {}", e))?;

        let mut ptr: *mut c_void = std::ptr::null_mut();
        let result = unsafe { hipMallocFromPoolAsync(&mut ptr, size, *pool, stream.raw()) };

        if result != HIP_SUCCESS {
            return Err(anyhow!(
                "hipMallocFromPoolAsync failed with error code: {}",
                result
            ));
        }

        Ok(ptr as u64)
    }

    /// Free memory back to the pool asynchronously.
    ///
    /// The memory is returned to the pool's reservoir (not the OS) and can be
    /// reused by subsequent allocations. The free is stream-ordered.
    pub fn free_async(&self, ptr: u64, stream: StreamHandle) -> Result<()> {
        self.free_async_raw(ptr, stream)
    }

    /// Free memory back to the pool asynchronously (raw stream handle variant).
    pub fn free_async_raw(&self, ptr: u64, stream: StreamHandle) -> Result<()> {
        let result = unsafe { hipFreeAsync(ptr as *mut c_void, stream.raw()) };

        if result != HIP_SUCCESS {
            return Err(anyhow!(
                "hipFreeAsync failed with error code: {}",
                result
            ));
        }

        Ok(())
    }
}

impl Drop for HipMemPool {
    fn drop(&mut self) {
        let pool = self
            .inner
            .get_mut()
            .expect("mutex should not be poisoned during drop");

        let result = unsafe { hipMemPoolDestroy(*pool) };
        if result != HIP_SUCCESS {
            tracing::warn!("hipMemPoolDestroy failed with error code: {}", result);
        }
    }
}
