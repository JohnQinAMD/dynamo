// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HIP (ROCm) GPU context and stream management.
//!
//! Parallel to `cuda.rs` — provides the same API surface using HIP FFI via
//! the GPU HAL in `dynamo_memory::gpu::hip`.
//!
//! Dynamo is typically not the primary compute driver; an external framework
//! (e.g. PyTorch with ROCm) owns the HIP context. The traits here allow
//! Dynamo to borrow and safely bind to that external context.

use dynamo_memory::gpu::hip::HipBackend;
use dynamo_memory::gpu::types::*;

use std::pin::Pin;
use std::{marker::PhantomData, sync::Arc};

/// Trait for types that can provide a HIP context handle.
pub trait DynamoHipContextProvider {
    /// # Safety
    ///
    /// This method is unsafe because it directly accesses the underlying HIP context.
    /// The caller must ensure that the context is valid and active.
    unsafe fn hip_context(&self) -> ContextHandle;

    fn bind_to_thread(&self) -> Pin<Box<DynamoHipContextGuard>> {
        unsafe { DynamoHipContextGuard::new(self.hip_context()) }
    }
}

/// Trait for types that can provide a HIP stream handle.
pub trait DynamoHipStreamProvider {
    /// # Safety
    ///
    /// This method is unsafe because it directly accesses the underlying HIP stream.
    /// The caller must ensure that the stream is valid and that the HIP context is active.
    ///
    /// Similarly, any pointers/references to data for which the stream will be accessed must
    /// have proper lifetimes and scoping, which is not guaranteed by this trait.
    unsafe fn hip_stream(&self) -> StreamHandle;

    fn context(&self) -> Arc<dyn DynamoHipContextProvider>;
}

/// A HIP context guard that ensures safe access to HIP contexts.
///
/// This guard:
/// - Cannot be moved (uses PhantomPinned)
/// - Cannot be cloned
/// - Cannot pass across async boundaries (!Send + !Sync)
/// - Provides safe access to the underlying HIP context
/// - Automatically manages context lifecycle
pub struct DynamoHipContextGuard {
    context: ContextHandle,
    _pin: std::marker::PhantomPinned,
    _not_send_sync: PhantomData<*const ()>,
}

impl DynamoHipContextGuard {
    /// Create a new context guard that sets the given context as current.
    ///
    /// # Safety
    ///
    /// The caller must ensure the context handle is valid.
    pub unsafe fn new(context: ContextHandle) -> Pin<Box<Self>> {
        HipBackend::set_current_context(context)
            .expect("Failed to set HIP context as current");

        let guard = Self {
            context,
            _pin: std::marker::PhantomPinned,
            _not_send_sync: PhantomData,
        };

        Box::pin(guard)
    }

    /// Get the raw HIP context handle.
    ///
    /// This method is safe because the guard ensures the context remains valid
    /// for its lifetime and cannot be moved or passed across async boundaries.
    pub fn context(&self) -> ContextHandle {
        self.context
    }
}

impl Drop for DynamoHipContextGuard {
    fn drop(&mut self) {
        // HIP uses hipCtxSetCurrent rather than push/pop semantics.
        // Nothing to explicitly pop on drop — the context remains current
        // until another context is set. This matches HIP runtime behavior
        // where the primary context is reference-counted.
    }
}

/// A HIP context provider that wraps an external HIP context.
pub struct ExternalHipContext {
    context: ContextHandle,
}

// SAFETY: ContextHandle wraps an opaque pointer to GPU runtime state.
// It does not alias mutable host memory and is safe to transfer between threads.
unsafe impl Send for ExternalHipContext {}
unsafe impl Sync for ExternalHipContext {}

impl ExternalHipContext {
    pub fn new(context: ContextHandle) -> Arc<Self> {
        Arc::new(Self { context })
    }

    pub fn hip_context(&self) -> ContextHandle {
        self.context
    }
}

impl DynamoHipContextProvider for ExternalHipContext {
    unsafe fn hip_context(&self) -> ContextHandle {
        self.context
    }
}

/// A HIP stream provider that wraps an external HIP stream.
pub struct ExternalHipStream {
    stream: StreamHandle,
    context: Arc<dyn DynamoHipContextProvider>,
}

impl ExternalHipStream {
    pub fn new(stream: StreamHandle, context: Arc<dyn DynamoHipContextProvider>) -> Self {
        Self { stream, context }
    }
}

impl DynamoHipStreamProvider for ExternalHipStream {
    unsafe fn hip_stream(&self) -> StreamHandle {
        self.stream
    }

    fn context(&self) -> Arc<dyn DynamoHipContextProvider> {
        self.context.clone()
    }
}
