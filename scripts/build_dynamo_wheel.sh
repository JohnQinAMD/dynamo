#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build Dynamo maturin wheel (Python 3.10+)
#
# The Rust bindings use PyO3 abi3-py310, so a SINGLE wheel works on
# Python 3.10, 3.11, 3.12, and beyond. No need for separate builds.
#
# Usage:
#   # Inside any Python >= 3.10 container:
#   bash scripts/build_dynamo_wheel.sh
#
#   # Or specify output directory:
#   WHEEL_DIR=/tmp/wheels bash scripts/build_dynamo_wheel.sh
#
#   # Use maturin develop (editable install, faster iteration):
#   MODE=develop bash scripts/build_dynamo_wheel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHEEL_DIR="${WHEEL_DIR:-$DYNAMO_ROOT/dist}"
MODE="${MODE:-build}"

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

echo "============================================"
echo "  Dynamo Maturin Build (abi3, Python >= 3.10)"
echo "============================================"
echo "DYNAMO_ROOT: $DYNAMO_ROOT"
echo "WHEEL_DIR:   $WHEEL_DIR"
echo "MODE:        $MODE"
echo "Python:      $(python3 --version 2>&1) ($PY_VER)"
echo "============================================"

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    echo "ERROR: Python >= 3.10 required, got $PY_VER"
    exit 1
fi

# Detect virtualenv
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d "/opt/venv" ]; then
        export VIRTUAL_ENV=/opt/venv
        export PATH=/opt/venv/bin:$PATH
        echo "Auto-detected VIRTUAL_ENV=/opt/venv"
    fi
fi

# Install build dependencies
echo "Installing build dependencies..."
pip install --quiet maturin patchelf 2>/dev/null || pip install --quiet --break-system-packages maturin patchelf

# Verify Rust is available
if ! command -v rustc &>/dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    else
        echo "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.93.1 -q
        source "$HOME/.cargo/env"
    fi
fi
echo "Rust: $(rustc --version 2>&1)"

# Set clang/bindgen env vars (required for nixl-sys bindgen)
export LIBCLANG_PATH="${LIBCLANG_PATH:-/opt/rocm/lib/llvm/lib}"
GCC_INC=$(find /usr/lib/gcc -name stdbool.h 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
if [ -n "$GCC_INC" ]; then
    export BINDGEN_EXTRA_CLANG_ARGS="-I${GCC_INC}"
    echo "BINDGEN fix: $GCC_INC"
fi

cd "$DYNAMO_ROOT/lib/bindings/python"

if [[ "$MODE" == "develop" ]]; then
    echo "Running maturin develop..."
    maturin develop --release 2>&1 | tail -5
    echo ""
    echo "Installing dynamo Python package..."
    cd "$DYNAMO_ROOT"
    pip install -e . 2>&1 | tail -3
else
    echo "Building maturin wheel..."
    mkdir -p "$WHEEL_DIR"
    maturin build --release --out "$WHEEL_DIR" 2>&1 | tail -5

    # The wheel uses abi3 tag: ai_dynamo_runtime-*-cp310-abi3-*.whl
    WHEEL_FILE=$(ls "$WHEEL_DIR"/ai_dynamo_runtime-*-abi3-*.whl 2>/dev/null | head -1)
    if [ -z "$WHEEL_FILE" ]; then
        WHEEL_FILE=$(ls "$WHEEL_DIR"/ai_dynamo_runtime-*.whl 2>/dev/null | head -1)
    fi

    if [ -n "$WHEEL_FILE" ]; then
        echo "Installing wheel: $WHEEL_FILE"
        pip install "$WHEEL_FILE" --force-reinstall 2>&1 | tail -3
    else
        echo "ERROR: No ai_dynamo_runtime wheel found in $WHEEL_DIR"
        ls "$WHEEL_DIR"/*.whl 2>/dev/null || echo "No wheels at all"
        exit 1
    fi

    echo "Installing dynamo Python package..."
    cd "$DYNAMO_ROOT"
    pip install -e . 2>&1 | tail -3
fi

# Verify
echo ""
echo "Verifying installation..."
python3 -c "
from dynamo.llm import ModelType
print(f'dynamo.llm: OK (Python {__import__(\"sys\").version})')
try:
    from dynamo.llm import MockEngineArgs
    print('MockEngineArgs: OK')
except ImportError:
    print('MockEngineArgs: not in this build (pre-built wheel)')
import dynamo
print(f'dynamo package: OK')
" 2>&1

echo ""
echo "============================================"
echo "  Build complete! (Python $PY_VER, abi3)"
if [[ "$MODE" != "develop" ]] && [ -n "${WHEEL_FILE:-}" ]; then
    echo "  Wheel: $WHEEL_FILE"
    echo "  This wheel works on Python 3.10, 3.11, 3.12+"
fi
echo "============================================"
