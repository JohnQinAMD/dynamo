#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated embedding model serving on AMD ROCm.
# Usage: DEVICE_PLATFORM=rocm bash agg_embed_rocm.sh [--model-path <name>]

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/rocm_utils.sh"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

MODEL="Qwen/Qwen3-Embedding-4B"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path) MODEL="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching ROCm Embedding Worker" "$MODEL" "$HTTP_PORT" \
    "HIP devices: $HIP_VISIBLE_DEVICES"

python3 -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang \
  --embedding-worker \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --use-sglang-tokenizer \
  --enable-metrics \
  "${EXTRA_ARGS[@]}" &

wait_any_exit
