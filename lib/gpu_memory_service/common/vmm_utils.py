# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified VMM utility facade — auto-detects HIP or CUDA backend.

Import from this module instead of the backend-specific modules to get
transparent GPU backend selection:

    from gpu_memory_service.common.vmm_utils import (
        check_result,
        ensure_initialized,
        get_allocation_granularity,
        align_to_granularity,
        BACKEND,
    )

Detection order: HIP first (preferred on AMD GPUs), then CUDA.
"""

from __future__ import annotations

from typing import Callable

BACKEND: str  # "hip" or "cuda"

check_result: Callable[..., None]
ensure_initialized: Callable[[], None]
get_allocation_granularity: Callable[[int], int]
align_to_granularity: Callable[[int, int], int]

try:
    from gpu_memory_service.common.hip_vmm_utils import (
        align_to_granularity,
        check_hip_result as check_result,
        ensure_hip_initialized as ensure_initialized,
        get_allocation_granularity,
    )

    BACKEND = "hip"

except ImportError:
    try:
        from gpu_memory_service.common.cuda_vmm_utils import (
            align_to_granularity,
            check_cuda_result as check_result,
            ensure_cuda_initialized as ensure_initialized,
            get_allocation_granularity,
        )

        BACKEND = "cuda"

    except ImportError as exc:
        raise ImportError(
            "Neither HIP (hip python package) nor CUDA (cuda-python bindings) "
            "is available. Install one of them to use GPU Memory Service."
        ) from exc
