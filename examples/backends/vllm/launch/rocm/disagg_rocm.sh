#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated serving on AMD ROCm GPUs (2 GPUs: 1 decode + 1 prefill).
# Usage: DEVICE_PLATFORM=rocm bash disagg_rocm.sh

set -e
trap 'echo Cleaning up...; kill 0' EXIT

DEVICE_PLATFORM="${DEVICE_PLATFORM:-rocm}"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/rocm_utils.sh"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

# ROCm-specific environment
export VLLM_ROCM_USE_AITER=1

MODEL="Qwen/Qwen3-0.6B"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching ROCm Disaggregated Serving (2 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Device:      $DEVICE_PLATFORM"

python -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
HIP_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --disaggregation-mode decode \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
HIP_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

wait_any_exit
