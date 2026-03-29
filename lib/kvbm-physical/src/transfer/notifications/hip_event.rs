// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HIP event polling-based completion checker.
//!
//! ROCm equivalent of `notifications/cuda_event.rs` in kvbm-physical.
//! Uses the GPU HAL's `HipBackend::event_query` to poll HIP event status.

use anyhow::Result;
use dynamo_memory::gpu::hip::HipBackend;
use dynamo_memory::gpu::types::*;
use dynamo_memory::gpu::GpuDevice;

use super::CompletionChecker;

/// Completion checker that polls HIP event status.
pub struct HipEventChecker {
    event: EventHandle,
}

impl HipEventChecker {
    pub fn new(event: EventHandle) -> Self {
        Self { event }
    }
}

impl CompletionChecker for HipEventChecker {
    fn is_complete(&self) -> Result<bool> {
        match HipBackend::event_query(self.event) {
            Ok(true) => Ok(true),
            Ok(false) => Ok(false),
            Err(e) => Err(anyhow::anyhow!("HIP event query failed: {:?}", e)),
        }
    }
}
