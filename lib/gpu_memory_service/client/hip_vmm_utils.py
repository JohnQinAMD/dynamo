# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side HIP VMM utilities.

These functions wrap HIP driver API calls used by the client memory manager
for importing, mapping, and unmapping GPU memory.

HIP VMM is a 1:1 mapping of CUDA VMM — every cuMem* function has a
corresponding hipMem* equivalent with identical semantics.
"""

from __future__ import annotations

import os

from hip import hip as hip_driver
from gpu_memory_service.common.hip_vmm_utils import check_hip_result
from gpu_memory_service.common.types import GrantedLockType


def import_handle_from_fd(fd: int) -> int:
    """Import a HIP memory handle from a file descriptor.

    Closes the FD after import — the imported handle holds its own reference
    to the physical allocation. Leaving the FD open leaks a DMA-buf ref that
    prevents hipMemRelease from freeing GPU memory.

    Args:
        fd: POSIX file descriptor received via SCM_RIGHTS.

    Returns:
        HIP memory handle.
    """
    try:
        result, handle = hip_driver.hipMemImportFromShareableHandle(
            fd,
            hip_driver.hipMemAllocationHandleType.hipMemHandleTypePosixFileDescriptor,
        )
        check_hip_result(result, "hipMemImportFromShareableHandle")
        return int(handle)
    finally:
        os.close(fd)


def reserve_va(size: int, granularity: int) -> int:
    """Reserve virtual address space.

    Args:
        size: Size in bytes (should be aligned to granularity).
        granularity: VMM allocation granularity.

    Returns:
        Reserved virtual address.
    """
    result, va = hip_driver.hipMemAddressReserve(size, granularity, 0, 0)
    check_hip_result(result, "hipMemAddressReserve")
    return int(va)


def free_va(va: int, size: int) -> None:
    """Free a virtual address reservation.

    Args:
        va: Virtual address to free.
        size: Size of the reservation.
    """
    (result,) = hip_driver.hipMemAddressFree(va, size)
    check_hip_result(result, "hipMemAddressFree")


def map_to_va(va: int, size: int, handle: int) -> None:
    """Map a HIP handle to a virtual address.

    Args:
        va: Virtual address (must be reserved).
        size: Size of the mapping.
        handle: HIP memory handle.
    """
    (result,) = hip_driver.hipMemMap(va, size, 0, handle, 0)
    check_hip_result(result, "hipMemMap")


def set_access(va: int, size: int, device: int, access: GrantedLockType) -> None:
    """Set access permissions for a mapped region.

    Args:
        va: Virtual address.
        size: Size of the region.
        device: HIP device index.
        access: Access mode - RO for read-only, RW for read-write.
    """
    acc = hip_driver.hipMemAccessDesc()
    acc.location.type = hip_driver.hipMemLocationType.hipMemLocationTypeDevice
    acc.location.id = device
    acc.flags = (
        hip_driver.hipMemAccessFlags.hipMemAccessFlagsProtRead
        if access == GrantedLockType.RO
        else hip_driver.hipMemAccessFlags.hipMemAccessFlagsProtReadWrite
    )
    (result,) = hip_driver.hipMemSetAccess(va, size, [acc], 1)
    check_hip_result(result, "hipMemSetAccess")


def unmap(va: int, size: int) -> None:
    """Unmap a virtual address region.

    Args:
        va: Virtual address to unmap.
        size: Size of the mapping.
    """
    (result,) = hip_driver.hipMemUnmap(va, size)
    check_hip_result(result, "hipMemUnmap")


def release_handle(handle: int) -> None:
    """Release a HIP memory handle.

    Args:
        handle: HIP memory handle to release.
    """
    (result,) = hip_driver.hipMemRelease(handle)
    check_hip_result(result, "hipMemRelease")


def validate_pointer(va: int) -> bool:
    """Validate that a mapped VA is accessible.

    Returns True if the pointer is valid, False otherwise (logs a warning).

    Note: HIP uses hipPointerGetAttribute with hipDeviceAttributeT for
    pointer validation. The exact API may differ from CUDA; this
    implementation uses hipPointerGetAttributes as the HIP equivalent.
    """
    result, attributes = hip_driver.hipPointerGetAttributes(va)
    if result != hip_driver.hipError_t.hipSuccess:
        err_str = hip_driver.hipGetErrorString(result)
        if isinstance(err_str, tuple):
            _, err_str = err_str
        err_msg = ""
        if err_str:
            err_msg = err_str.decode() if isinstance(err_str, bytes) else str(err_str)
        import logging

        logging.getLogger(__name__).warning(
            "hipPointerGetAttributes failed for VA 0x%x: %s (%s)",
            va,
            result,
            err_msg,
        )
        return False
    return True


def synchronize() -> None:
    """Synchronize the current HIP context.

    Blocks until all preceding commands in the current context have completed.
    """
    (result,) = hip_driver.hipCtxSynchronize()
    check_hip_result(result, "hipCtxSynchronize")


def set_current_device(device: int) -> None:
    """Set the current HIP device by activating its primary context.

    Args:
        device: HIP device index.
    """
    result, ctx = hip_driver.hipDevicePrimaryCtxRetain(device)
    check_hip_result(result, "hipDevicePrimaryCtxRetain")
    (result,) = hip_driver.hipCtxSetCurrent(ctx)
    check_hip_result(result, "hipCtxSetCurrent")
