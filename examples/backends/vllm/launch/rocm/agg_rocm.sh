#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving on a single AMD ROCm GPU.
# Usage: DEVICE_PLATFORM=rocm bash agg_rocm.sh [--model <name>]

set -e
trap 'echo Cleaning up...; kill 0' EXIT

DEVICE_PLATFORM="${DEVICE_PLATFORM:-rocm}"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/rocm_utils.sh"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

# ROCm-specific environment
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export VLLM_ROCM_USE_AITER=1

MODEL="Qwen/Qwen3-0.6B"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching ROCm Aggregated Serving (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "Device:      $DEVICE_PLATFORM" \
    "HIP devices: $HIP_VISIBLE_DEVICES"

python -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm --model "$MODEL" --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
