#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated SGLang serving on a single AMD ROCm GPU.
# Usage: DEVICE_PLATFORM=rocm bash agg_rocm.sh [--model-path <name>]

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
ENABLE_OTEL=false

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL="$2"
            shift 2
            ;;
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path <name>  Specify model (default: $MODEL)"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Additional SGLang/Dynamo flags can be passed and will be forwarded"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

TRACE_ARGS=()
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
    TRACE_ARGS+=(--enable-trace --otlp-traces-endpoint localhost:4317)
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching ROCm SGLang Aggregated Serving" "$MODEL" "$HTTP_PORT" \
    "Device:      $DEVICE_PLATFORM" \
    "HIP devices: $HIP_VISIBLE_DEVICES"

OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &

OTEL_SERVICE_NAME=dynamo-worker DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --enable-metrics \
  "${TRACE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" &

wait_any_exit
