#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build Dynamo maturin wheel for Python 3.12
#
# Purpose: Unblock vLLM backend tests on ROCm. The ROCm vLLM container
# (vllm/vllm-openai-rocm) uses Python 3.12, but Dynamo's maturin build
# currently targets Python 3.10 (matching the SGLang container).
#
# Usage:
#   # Inside a Python 3.12 container (e.g., rocm/vllm):
#   bash scripts/build_maturin_py312.sh
#
#   # Or specify output directory:
#   WHEEL_DIR=/tmp/wheels bash scripts/build_maturin_py312.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHEEL_DIR="${WHEEL_DIR:-$DYNAMO_ROOT/dist}"

echo "============================================"
echo "  Dynamo Maturin Build for Python 3.12"
echo "============================================"
echo "DYNAMO_ROOT: $DYNAMO_ROOT"
echo "WHEEL_DIR:   $WHEEL_DIR"
echo "Python:      $(python3 --version 2>&1)"
echo "============================================"

# Verify Python version
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PY_VER" != "3.12" ]]; then
    echo "ERROR: This script requires Python 3.12, got $PY_VER"
    echo "Run inside a Python 3.12 container (e.g., rocm/vllm:latest)"
    exit 1
fi

# Install build dependencies
echo "Installing build dependencies..."
pip install --quiet maturin patchelf

# Verify Rust is available
if ! command -v rustc &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.93.1 -q
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Set ROCm-specific env vars
export LIBCLANG_PATH="${LIBCLANG_PATH:-/opt/rocm/lib/llvm/lib}"
if [ -d "/opt/rocm/lib/llvm/lib" ]; then
    echo "Using LIBCLANG_PATH=$LIBCLANG_PATH"
fi

GCC_INC=$(ls -d /usr/lib/gcc/x86_64-linux-gnu/*/include 2>/dev/null | sort -V | tail -1 || echo "")
if [ -n "$GCC_INC" ]; then
    export BINDGEN_EXTRA_CLANG_ARGS="-I${GCC_INC}"
fi

# Build the wheel
echo "Building maturin wheel..."
mkdir -p "$WHEEL_DIR"
cd "$DYNAMO_ROOT/lib/bindings/python"
maturin build --release --out "$WHEEL_DIR" 2>&1

# Install the wheel
echo "Installing wheel..."
WHEEL_FILE=$(ls "$WHEEL_DIR"/dynamo_llm-*-cp312-*.whl 2>/dev/null | head -1)
if [ -n "$WHEEL_FILE" ]; then
    pip install "$WHEEL_FILE" --force-reinstall
    echo "PASS: Installed $WHEEL_FILE"
else
    echo "WARNING: No cp312 wheel found in $WHEEL_DIR"
    ls "$WHEEL_DIR"/*.whl 2>/dev/null || echo "No wheels at all"
    exit 1
fi

# Install dynamo Python package
echo "Installing dynamo Python package..."
cd "$DYNAMO_ROOT"
pip install -e . 2>&1

# Verify
echo "Verifying installation..."
python3 -c "
import dynamo_llm
print(f'dynamo_llm version: {dynamo_llm.__version__}')
import dynamo
print('dynamo package: OK')
" 2>&1

echo ""
echo "============================================"
echo "  Build complete!"
echo "  Wheel: $WHEEL_FILE"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. pip install vllm (ROCm version)"
echo "  2. pytest tests/ -m 'vllm and rocm' -v"
