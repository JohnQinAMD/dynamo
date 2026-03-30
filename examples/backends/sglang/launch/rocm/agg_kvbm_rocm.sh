#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated SGLang serving with KVBM CPU KV cache offloading on AMD ROCm.
#
# KVBM (KV Block Manager) offloads evicted GPU KV cache blocks to CPU DRAM,
# enabling multi-turn conversations to reuse cached KV instead of recomputing.
#
# Usage:
#   DEVICE_PLATFORM=rocm bash agg_kvbm_rocm.sh [--model-path <name>] [--cpu-cache-gb <GB>]

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
CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-20}"
ENABLE_OTEL=false

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL="$2"
            shift 2
            ;;
        --cpu-cache-gb)
            CPU_CACHE_GB="$2"
            shift 2
            ;;
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path <name>      Specify model (default: $MODEL)"
            echo "  --cpu-cache-gb <GB>       CPU DRAM for KV cache offload (default: $CPU_CACHE_GB)"
            echo "  --enable-otel            Enable OpenTelemetry tracing"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  DYN_KVBM_CPU_CACHE_GB    CPU DRAM pool size in GB (overridden by --cpu-cache-gb)"
            echo "  HIP_VISIBLE_DEVICES      ROCm GPU device indices (default: 0)"
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
print_launch_banner "Launching ROCm SGLang + KVBM Offloading" "$MODEL" "$HTTP_PORT" \
    "Device:         $DEVICE_PLATFORM" \
    "HIP devices:    $HIP_VISIBLE_DEVICES" \
    "CPU cache pool: ${CPU_CACHE_GB} GB"

# Run ingress
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &

# Run worker with KVBM CPU offloading enabled
OTEL_SERVICE_NAME=dynamo-worker DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
DYN_KVBM_CPU_CACHE_GB="$CPU_CACHE_GB" \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --enable-metrics \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  "${TRACE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" &

wait_any_exit
