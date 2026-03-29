// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # HIP (ROCm) Storage Support
//!
//! ROCm equivalent of `storage/cuda.rs`. Provides HIP-specific storage
//! implementations for the block manager, backed by the GPU HAL in
//! `dynamo_memory::gpu::hip`.
//!
//! ## Features
//!
//! The following types are available when the `rocm` feature is enabled:
//! - [`HipPinnedStorage`] - Page-locked host memory for efficient GPU transfers
//! - [`HipDeviceStorage`] - Direct GPU memory allocation
//!
//! ## Storage Allocators
//!
//! - [`HipPinnedAllocator`] - Creates pinned host memory allocations
//! - [`HipDeviceAllocator`] - Creates device memory allocations
//!
//! ## HIP Context Management
//!
//! The module provides a singleton [`Hip`] type for managing HIP contexts:
//! - Thread-safe context management
//! - Lazy initialization of device contexts
//! - Automatic cleanup of resources

use super::*;

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use dynamo_memory::gpu::hip::HipBackend;
use dynamo_memory::gpu::types::*;
use dynamo_memory::gpu::GpuDevice;

/// Trait for [`Storage`] types that can be accessed by HIP
pub trait HipAccessible: Storage {}

/// Singleton for managing HIP contexts.
pub struct Hip {
    contexts: HashMap<usize, ContextHandle>,
}

impl Hip {
    fn new() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    /// Get a HIP context for a specific device_id.
    /// Returns None if the context does not exist.
    pub fn device(device_id: usize) -> Option<ContextHandle> {
        Hip::instance()
            .lock()
            .unwrap()
            .get_existing_context(device_id)
    }

    /// Get or initialize a HIP context for a specific device_id.
    pub fn device_or_create(device_id: usize) -> Result<ContextHandle, StorageError> {
        Hip::instance().lock().unwrap().get_context(device_id)
    }

    /// Check if a HIP context exists for a specific device_id.
    pub fn is_initialized(device_id: usize) -> bool {
        Hip::instance().lock().unwrap().has_context(device_id)
    }

    fn instance() -> &'static Mutex<Hip> {
        static INSTANCE: OnceLock<Mutex<Hip>> = OnceLock::new();
        INSTANCE.get_or_init(|| Mutex::new(Hip::new()))
    }

    fn get_context(&mut self, device_id: usize) -> Result<ContextHandle, StorageError> {
        if let Some(ctx) = self.contexts.get(&device_id) {
            return Ok(*ctx);
        }

        let ctx = HipBackend::create_context(device_id as i32).map_err(|e| {
            StorageError::AllocationFailed(format!(
                "Failed to create HIP context for device {}: {}",
                device_id, e
            ))
        })?;

        self.contexts.insert(device_id, ctx);
        Ok(ctx)
    }

    pub fn get_existing_context(&self, device_id: usize) -> Option<ContextHandle> {
        self.contexts.get(&device_id).copied()
    }

    pub fn has_context(&self, device_id: usize) -> bool {
        self.contexts.contains_key(&device_id)
    }
}

/// Pinned host memory storage using HIP page-locked memory.
#[derive(Debug)]
pub struct HipPinnedStorage {
    ptr: *mut u8,
    size: usize,
    device_id: i32,
    handles: RegistrationHandles,
}

// SAFETY: The pinned host pointer is obtained from hipHostMalloc and is
// safe to access from any thread.
unsafe impl Send for HipPinnedStorage {}
unsafe impl Sync for HipPinnedStorage {}

impl Local for HipPinnedStorage {}
impl SystemAccessible for HipPinnedStorage {}
impl HipAccessible for HipPinnedStorage {}

impl HipPinnedStorage {
    /// Create a new pinned storage with the given size.
    pub fn new(size: usize, device_id: Option<i32>) -> Result<Self, StorageError> {
        let dev = device_id.unwrap_or(0);

        // Ensure context is initialised for the target device
        Hip::device_or_create(dev as usize)?;

        let ptr = HipBackend::malloc_host(size).map_err(|e| {
            StorageError::AllocationFailed(format!("HIP pinned alloc failed: {}", e))
        })?;

        Ok(Self {
            ptr,
            size,
            device_id: dev,
            handles: RegistrationHandles::new(),
        })
    }

    /// Return the device ordinal this storage is associated with.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Drop for HipPinnedStorage {
    fn drop(&mut self) {
        self.handles.release();
        if let Err(e) = HipBackend::free_host(self.ptr) {
            tracing::warn!("hipHostFree failed: {}", e);
        }
    }
}

impl Storage for HipPinnedStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Pinned
    }

    fn addr(&self) -> u64 {
        self.ptr as u64
    }

    fn size(&self) -> usize {
        self.size
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl RegisterableStorage for HipPinnedStorage {
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        self.handles.register(key, handle)
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

impl StorageMemset for HipPinnedStorage {
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<(), StorageError> {
        if offset + size > self.size {
            return Err(StorageError::OperationFailed(
                "memset: offset + size > storage size".into(),
            ));
        }
        unsafe {
            let ptr = self.ptr.add(offset);
            std::ptr::write_bytes(ptr, value, size);
        }
        Ok(())
    }
}

/// Allocator for HipPinnedStorage
pub struct HipPinnedAllocator {
    device_id: i32,
}

impl Default for HipPinnedAllocator {
    fn default() -> Self {
        Hip::device_or_create(0).expect("Failed to create HIP context");
        Self { device_id: 0 }
    }
}

impl HipPinnedAllocator {
    pub fn new(device_id: usize) -> Result<Self, StorageError> {
        Hip::device_or_create(device_id)?;
        Ok(Self {
            device_id: device_id as i32,
        })
    }
}

impl StorageAllocator<HipPinnedStorage> for HipPinnedAllocator {
    fn allocate(&self, size: usize) -> Result<HipPinnedStorage, StorageError> {
        HipPinnedStorage::new(size, Some(self.device_id))
    }
}

/// HIP device memory storage
#[derive(Debug)]
pub struct HipDeviceStorage {
    ptr: DevicePtr,
    size: usize,
    device_id: i32,
    handles: RegistrationHandles,
}

// SAFETY: DevicePtr wraps an opaque device address. The pointer itself does
// not alias mutable host memory and is safe to move between threads.
unsafe impl Send for HipDeviceStorage {}
unsafe impl Sync for HipDeviceStorage {}

impl Local for HipDeviceStorage {}
impl HipAccessible for HipDeviceStorage {}

impl HipDeviceStorage {
    /// Create a new device storage with the given size.
    pub fn new(device_id: usize, size: usize) -> Result<Self, StorageError> {
        let ctx = Hip::device_or_create(device_id)?;
        HipBackend::set_current_context(ctx).map_err(|e| {
            StorageError::OperationFailed(format!("Failed to set HIP context: {}", e))
        })?;

        let ptr = HipBackend::malloc(size).map_err(|e| {
            StorageError::AllocationFailed(format!("HIP device alloc failed: {}", e))
        })?;

        Ok(Self {
            ptr,
            size,
            device_id: device_id as i32,
            handles: RegistrationHandles::new(),
        })
    }

    /// Get the device ordinal.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Storage for HipDeviceStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Device(self.device_id as u32)
    }

    fn addr(&self) -> u64 {
        self.ptr.raw()
    }

    fn size(&self) -> usize {
        self.size
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr.raw() as *const u8
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.raw() as *mut u8
    }
}

impl Drop for HipDeviceStorage {
    fn drop(&mut self) {
        self.handles.release();
        if let Err(e) = HipBackend::free(self.ptr) {
            tracing::warn!("hipFree failed: {}", e);
        }
    }
}

impl RegisterableStorage for HipDeviceStorage {
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        self.handles.register(key, handle)
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

/// Allocator for HipDeviceStorage
pub struct HipDeviceAllocator {
    device_id: usize,
}

impl Default for HipDeviceAllocator {
    fn default() -> Self {
        Hip::device_or_create(0).expect("Failed to create HIP context");
        Self { device_id: 0 }
    }
}

impl HipDeviceAllocator {
    pub fn new(device_id: usize) -> Result<Self, StorageError> {
        Hip::device_or_create(device_id)?;
        Ok(Self { device_id })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

impl StorageAllocator<HipDeviceStorage> for HipDeviceAllocator {
    fn allocate(&self, size: usize) -> Result<HipDeviceStorage, StorageError> {
        HipDeviceStorage::new(self.device_id, size)
    }
}
