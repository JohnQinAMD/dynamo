#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ROCm GPU utility functions for launch scripts.
#
# Source:
#   source "$(dirname "$(readlink -f "$0")")/../common/rocm_utils.sh"
#   # or with SCRIPT_DIR already set:
#   source "$SCRIPT_DIR/../common/rocm_utils.sh"
#
# Functions:
#   get_rocm_gpu_count         Count available AMD GPUs
#   get_rocm_visible_devices   Return HIP_VISIBLE_DEVICES or generate device list
#   check_rocm_env             Verify ROCm tooling is available

# get_rocm_gpu_count
#
# Prints the number of AMD GPUs detected via amd-smi or rocm-smi.
# Falls back to 0 if neither tool is available.
get_rocm_gpu_count() {
    if command -v amd-smi &> /dev/null; then
        amd-smi list 2>/dev/null | grep -c "GPU"
    elif command -v rocm-smi &> /dev/null; then
        rocm-smi --showid 2>/dev/null | grep -c "GPU"
    else
        echo "0"
    fi
}

# get_rocm_visible_devices
#
# Prints the value of HIP_VISIBLE_DEVICES if set, otherwise generates
# a comma-separated list of all available GPU indices (0,1,2,...).
get_rocm_visible_devices() {
    if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
        echo "$HIP_VISIBLE_DEVICES"
    else
        local count
        count=$(get_rocm_gpu_count)
        if [[ "$count" -gt 0 ]]; then
            seq -s, 0 $((count - 1))
        else
            echo "0"
        fi
    fi
}

# check_rocm_env
#
# Verifies that ROCm tooling (amd-smi or rocm-smi) is available.
# Prints a warning to stderr if neither is found but does not exit,
# allowing containers with pre-configured HIP_VISIBLE_DEVICES to proceed.
check_rocm_env() {
    if ! command -v amd-smi &> /dev/null && ! command -v rocm-smi &> /dev/null; then
        echo "WARNING: Neither amd-smi nor rocm-smi found in PATH." >&2
        echo "GPU discovery will not work. Set HIP_VISIBLE_DEVICES manually." >&2
    fi
}

# Run check on source (non-fatal warning)
check_rocm_env

# CLI mode: only when executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ "${1:-}" == "--self-test" ]]; then
        echo "=== ROCm Utils Self-Test ==="
        echo "GPU count: $(get_rocm_gpu_count)"
        echo "Visible devices: $(get_rocm_visible_devices)"
        echo "=== Done ==="
        exit 0
    fi

    cat <<'HELP'
rocm_utils.sh — ROCm GPU utility functions

Usage:
  source rocm_utils.sh           Source for use in launch scripts
  ./rocm_utils.sh --self-test    Run self-test

Functions:
  get_rocm_gpu_count             Count available AMD GPUs
  get_rocm_visible_devices       Return HIP_VISIBLE_DEVICES or device list
  check_rocm_env                 Verify ROCm tooling availability

Environment variables:
  HIP_VISIBLE_DEVICES            Override visible GPU list (e.g. "0,1")
HELP
    exit 0
fi
