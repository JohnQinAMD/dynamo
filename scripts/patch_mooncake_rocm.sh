#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Apply ROCm RDMA patch to Mooncake transfer engine and rebuild.
#
# This enables Mooncake RDMA on AMD GPUs with Pensando ionic NICs by:
#   1. Detecting GPU vs CPU memory in ibv_reg_mr (returns clear error for GPU VRAM)
#   2. Auto-detecting ionic NICs and setting max_sge=2
#
# Usage (inside a container with Mooncake source):
#   bash scripts/patch_mooncake_rocm.sh
#
# The script auto-detects the Mooncake source location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_FILE="$DYNAMO_ROOT/patches/mooncake_rocm_rdma.patch"

echo "============================================"
echo "  Mooncake ROCm RDMA Patch"
echo "============================================"

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Patch file not found: $PATCH_FILE"
    exit 1
fi

# Auto-detect Mooncake source location
MC_ROOT=""
for candidate in \
    /sgl-workspace/Mooncake \
    /workspace/Mooncake \
    /opt/mooncake \
    "$HOME/Mooncake"; do
    if [ -f "$candidate/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp" ]; then
        MC_ROOT="$candidate"
        break
    fi
done

if [ -z "$MC_ROOT" ]; then
    echo "ERROR: Mooncake source not found. Searched:"
    echo "  /sgl-workspace/Mooncake"
    echo "  /workspace/Mooncake"
    echo "  /opt/mooncake"
    echo "  ~/Mooncake"
    echo ""
    echo "Set MC_ROOT=/path/to/Mooncake and retry."
    exit 1
fi

echo "Mooncake source: $MC_ROOT"

# Check if already patched
if grep -q "USE_HIP" "$MC_ROOT/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp"; then
    echo "Already patched — skipping patch step."
else
    echo "Applying patch..."
    cd "$MC_ROOT"
    patch -p1 < "$PATCH_FILE"
    echo "Patch applied successfully."
fi

# Rebuild
if [ -d "$MC_ROOT/build" ]; then
    echo "Rebuilding Mooncake..."
    cd "$MC_ROOT/build"
    make -j"$(nproc)" 2>&1 | tail -5
    echo "Build complete."

    # Reinstall Python package if setup.py exists
    if [ -f "$MC_ROOT/setup.py" ] || [ -f "$MC_ROOT/pyproject.toml" ]; then
        echo "Reinstalling Python package..."
        cd "$MC_ROOT"
        pip install -e . 2>&1 | tail -3
    fi
else
    echo "No build/ directory found. Run cmake first:"
    echo "  cd $MC_ROOT && mkdir build && cd build"
    echo "  cmake .. -DUSE_HIP=ON && make -j\$(nproc)"
fi

echo ""
echo "============================================"
echo "  Mooncake ROCm patch applied and rebuilt."
echo "  Use: --disaggregation-transfer-backend mooncake"
echo "============================================"
