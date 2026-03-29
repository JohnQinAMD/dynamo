# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified client-side VMM utility facade — auto-detects HIP or CUDA backend.

Import from this module instead of the backend-specific modules to get
transparent GPU backend selection:

    from gpu_memory_service.client.vmm_utils import (
        import_handle_from_fd,
        reserve_va,
        free_va,
        map_to_va,
        set_access,
        unmap,
        release_handle,
        validate_pointer,
        synchronize,
        set_current_device,
    )

Detection order: HIP first (preferred on AMD GPUs), then CUDA.
"""

from __future__ import annotations

try:
    from gpu_memory_service.client.hip_vmm_utils import (  # noqa: F401
        free_va,
        import_handle_from_fd,
        map_to_va,
        release_handle,
        reserve_va,
        set_access,
        set_current_device,
        synchronize,
        unmap,
        validate_pointer,
    )

except ImportError:
    from gpu_memory_service.client.cuda_vmm_utils import (  # noqa: F401
        free_va,
        import_handle_from_fd,
        map_to_va,
        release_handle,
        reserve_va,
        set_access,
        set_current_device,
        synchronize,
        unmap,
        validate_pointer,
    )
