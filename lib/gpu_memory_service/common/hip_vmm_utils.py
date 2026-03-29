# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HIP Virtual Memory Management (VMM) utility functions.

This module provides utility functions for HIP driver API operations
used by both server (GMSServerMemoryManager) and client (GMSClientMemoryManager).

HIP VMM is a 1:1 mapping of CUDA VMM — every cuMem* function has a
corresponding hipMem* equivalent with identical semantics.
"""

from hip import hip as hip_driver


def check_hip_result(result: hip_driver.hipError_t, name: str) -> None:
    """Check HIP driver API result and raise on error.

    Args:
        result: HIP driver API return code (hipError_t enum)
        name: Operation name for error message

    Raises:
        RuntimeError: If result is not hipSuccess
    """
    if result != hip_driver.hipError_t.hipSuccess:
        err_str = hip_driver.hipGetErrorString(result)
        if isinstance(err_str, tuple):
            _, err_str = err_str
        if err_str:
            err_msg = err_str.decode() if isinstance(err_str, bytes) else str(err_str)
        else:
            err_msg = str(result)
        raise RuntimeError(f"{name}: {err_msg}")


def ensure_hip_initialized() -> None:
    """Ensure HIP driver is initialized.

    Raises:
        RuntimeError: If hipInit fails
    """
    result = hip_driver.hipInit(0)
    if isinstance(result, tuple):
        (result,) = result
    check_hip_result(result, "hipInit")


def get_allocation_granularity(device: int) -> int:
    """Get VMM allocation granularity for a device.

    Args:
        device: HIP device index

    Returns:
        Allocation granularity in bytes (typically 2 MiB)
    """
    prop = hip_driver.hipMemAllocationProp()
    prop.type = hip_driver.hipMemAllocationType.hipMemAllocationTypePinned
    prop.location.type = hip_driver.hipMemLocationType.hipMemLocationTypeDevice
    prop.location.id = device
    prop.requestedHandleTypes = (
        hip_driver.hipMemAllocationHandleType.hipMemHandleTypePosixFileDescriptor
    )

    result, granularity = hip_driver.hipMemGetAllocationGranularity(
        prop,
        hip_driver.hipMemAllocationGranularity.hipMemAllocationGranularityMinimum,
    )
    check_hip_result(result, "hipMemGetAllocationGranularity")
    return int(granularity)


def align_to_granularity(size: int, granularity: int) -> int:
    """Align size up to VMM granularity.

    Args:
        size: Size in bytes
        granularity: Allocation granularity

    Returns:
        Aligned size
    """
    return ((size + granularity - 1) // granularity) * granularity
