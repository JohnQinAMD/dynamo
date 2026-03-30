#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run all ROCm tests inside a container.
#
# Usage:
#   # Inside a ROCm container with Dynamo installed:
#   bash scripts/run_rocm_tests.sh [--tier 1|2|3] [--skip-build]
#
#   # From the host with Docker:
#   docker run --rm --device=/dev/kfd --device=/dev/dri \
#     --group-add video --shm-size 256G --ipc=host --privileged \
#     -v $(pwd):/workspace/dynamo \
#     rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2 \
#     bash /workspace/dynamo/scripts/run_rocm_tests.sh

set -euo pipefail

TIER="${1:-all}"
SKIP_BUILD=false
for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
        --tier) TIER="$2"; shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DYNAMO_ROOT"

PASS=0
FAIL=0
SKIP=0

run_test_group() {
    local name="$1"
    shift
    echo ""
    echo "================================================================"
    echo "  $name"
    echo "================================================================"
    if python3 -m pytest "$@" 2>&1; then
        PASS=$((PASS + 1))
        echo "  RESULT: PASS"
    else
        local rc=$?
        if [ $rc -eq 5 ]; then
            SKIP=$((SKIP + 1))
            echo "  RESULT: SKIPPED (no tests collected)"
        else
            FAIL=$((FAIL + 1))
            echo "  RESULT: FAIL (exit code $rc)"
        fi
    fi
}

# ==========================================================================
# Build
# ==========================================================================
if [ "$SKIP_BUILD" = false ]; then
    echo "Building Dynamo..."
    apt-get update -qq && apt-get install -y -qq build-essential pkg-config libclang-dev protobuf-compiler > /dev/null 2>&1 || true
    command -v rustc &>/dev/null || {
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.93.1 -q 2>&1 | tail -1
        export PATH=/root/.cargo/bin:$PATH
    }
    export PATH=/root/.cargo/bin:$PATH
    export LIBCLANG_PATH=/opt/rocm/lib/llvm/lib
    GCC_INC=$(ls -d /usr/lib/gcc/x86_64-linux-gnu/*/include 2>/dev/null | sort -V | tail -1 || echo "")
    [ -n "$GCC_INC" ] && export BINDGEN_EXTRA_CLANG_ARGS="-I${GCC_INC}"
    cd lib/bindings/python && maturin develop --release 2>&1 | tail -3
    cd "$DYNAMO_ROOT" && pip install -e . 2>&1 | tail -2
    pip install pytest pyyaml 2>&1 | tail -1
fi

export SGLANG_AITER_MLA_PERSIST=False
export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0

echo ""
echo "================================================================"
echo "  Dynamo ROCm Test Suite — Tier: $TIER"
echo "================================================================"
echo "  ROCm:     $(cat /opt/rocm/.info/version 2>/dev/null || echo 'unknown')"
echo "  Python:   $(python3 --version 2>&1)"
echo "  PyTorch:  $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "  GPUs:     $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'unknown')"
echo "================================================================"

# ==========================================================================
# Tier 1: Unit tests (no GPU)
# ==========================================================================
if [ "$TIER" = "all" ] || [ "$TIER" = "1" ]; then
    echo ""
    echo ">>> TIER 1: Unit Tests (no GPU required)"
    echo ""

    run_test_group "ROCm GPU Detection" \
        tests/basic/test_rocm_gpu_detection.py -v --tb=short

    run_test_group "ROCm Version Consistency" \
        tests/basic/test_rocm_version_consistency.py -v --tb=short

    run_test_group "K8s CRD Validation" \
        tests/deploy/test_k8s_crd_validation.py -v --tb=short

    run_test_group "Planner Config (unit)" \
        tests/planner/unit/test_planner_config.py -v --tb=short -k "not vllm"

    run_test_group "Load Predictors (unit)" \
        tests/planner/unit/test_load_predictors.py -v --tb=short

    run_test_group "Planner Virtual SGLang" \
        tests/planner/test_planner_virtual_sglang.py -v --tb=short -k "not E2E"

    run_test_group "FPM Relay SGLang" \
        tests/planner/test_fpm_relay_sglang.py -v --tb=short

    run_test_group "Mocker Config" \
        tests/mocker/test_config.py -v --tb=short
fi

# ==========================================================================
# Tier 2: GPU tests (single node)
# ==========================================================================
if [ "$TIER" = "all" ] || [ "$TIER" = "2" ]; then
    echo ""
    echo ">>> TIER 2: GPU Tests (single node)"
    echo ""

    run_test_group "Router Block Size Regression" \
        tests/router/test_router_block_size_regression.py -v --tb=short

    run_test_group "Router E2E with Mockers" \
        tests/router/test_router_e2e_with_mockers.py \
        -k "test_mocker_router[kv-nondurable-tcp]" \
        -v --tb=short --timeout=120

    run_test_group "Ionic NIC Validation" \
        tests/disagg/test_ionic_validation.py -v --tb=short

    run_test_group "KVBM HIP Kernels" \
        tests/kvbm_integration/test_kvbm_rocm.py -v --tb=short -k "TestKvbmHipKernels"
fi

# ==========================================================================
# Tier 3: E2E tests (GPU + serving)
# ==========================================================================
if [ "$TIER" = "3" ]; then
    echo ""
    echo ">>> TIER 3: E2E Tests (GPU + serving)"
    echo ""

    run_test_group "SGLang ROCm Serve" \
        tests/serve/test_sglang_rocm.py -v --tb=short --timeout=300

    run_test_group "Router E2E with SGLang" \
        tests/router/test_router_e2e_with_sglang.py -v --tb=short --timeout=300
fi

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "================================================================"
echo "  TEST SUMMARY"
echo "================================================================"
echo "  Passed:  $PASS"
echo "  Failed:  $FAIL"
echo "  Skipped: $SKIP"
echo "  Total:   $((PASS + FAIL + SKIP))"
echo "================================================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
