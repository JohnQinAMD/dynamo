# Building Dynamo with AMD ROCm (RIXL)

This guide explains how to build and run Dynamo on AMD GPUs using ROCm and RIXL (the ROCm port of NIXL).

## Prerequisites

- AMD Instinct MI200/MI300/MI355X GPU (or compatible)
- ROCm 6.x installed (`/opt/rocm`)
- Docker (for the container workflow) or a bare-metal ROCm environment

## Option A: Docker Container (Recommended)

The ROCm development container provides a complete build environment.

### Build the Container

```bash
cd dynamo
docker build -f container/Dockerfile.rocm-dev -t dynamo-rocm-dev .
```

### Run the Container

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  -v /path/to/rocm-dynamo:/workspace \
  dynamo-rocm-dev
```

Inside the container, proceed to the build steps below.

## Option B: Bare-Metal Setup

Install system dependencies:

```bash
sudo apt install -y build-essential cmake pkg-config autoconf automake libtool \
  rdma-core libibverbs-dev librdmacm-dev protobuf-compiler libclang-dev \
  libhwloc-dev libudev-dev patchelf ninja-build python3-pip
```

Install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Install Python build tools:

```bash
pip3 install meson pybind11 maturin uv hatchling
```

## Step 1: Build UCX with ROCm Support

RIXL requires UCX compiled with ROCm GPU-direct support:

```bash
git clone https://github.com/ROCm/ucx -b v1.19.x --depth 1
cd ucx && ./autogen.sh && mkdir build && cd build
../configure --enable-shared --disable-static --disable-doxygen-doc \
  --enable-optimizations --enable-devel-headers \
  --with-rocm=/opt/rocm --with-verbs --with-dm --enable-mt \
  --prefix=/usr/local/ucx
make -j$(nproc) && sudo make install
```

The container image already includes this step.

## Step 2: Build and Install RIXL

> **Important**: Use the [patched RIXL fork](https://github.com/JohnQinAMD/RIXL) which includes transparent DRAM staging for Pensando ionic NICs (`src/plugins/ucx/dram_staging.{h,cpp}`). On NICs without GPU Direct RDMA, VRAM registration via `ucp_mem_map` fails — the patch adds automatic fallback to pinned host memory staging with zero link-time dependency on ROCm.

```bash
# Clone the patched RIXL (includes DRAM staging for ionic NICs)
git clone https://github.com/JohnQinAMD/RIXL.git
cd RIXL
mkdir build && cd build
meson setup .. --prefix=/opt/rocm/rixl \
  -Dwith_ucx=enabled \
  -Ducx_path=/usr/local/ucx \
  -Dwith_rocm=enabled \
  -Drocm_path=/opt/rocm
ninja -j$(nproc)
sudo ninja install
```

Set environment variables:

```bash
export NIXL_PREFIX=/opt/rocm/rixl
export RIXL_PREFIX=/opt/rocm/rixl
export LD_LIBRARY_PATH=/opt/rocm/rixl/lib:$LD_LIBRARY_PATH
```

## Step 3: Enable the RIXL Cargo Patch

In the Dynamo workspace root `Cargo.toml`, uncomment the `[patch.crates-io]` section:

```toml
[patch.crates-io]
nixl-sys = { path = "../RIXL/src/bindings/rust" }
```

This redirects the `nixl-sys` crate from the upstream NIXL version (0.10.1) to RIXL's version (1.0.0), which provides the same API surface but links against ROCm/HIP and the ROCm-enabled UCX build.

The RIXL `nixl-sys` build.rs reads `NIXL_PREFIX` to locate headers and shared libraries — the same mechanism as upstream NIXL but pointed at the RIXL install prefix.

## Step 4: Build Dynamo

```bash
cd /workspace/dynamo

# Build with ROCm feature flags
cargo build --features rocm

# Or build just the memory library for testing
cargo build -p dynamo-memory --features rocm

# Build with the block manager for ROCm
cargo build -p dynamo-llm --features block-manager-rocm
```

## Step 5: Build Python Bindings

```bash
# Critical: set BINDGEN_EXTRA_CLANG_ARGS for stdbool.h resolution
export BINDGEN_EXTRA_CLANG_ARGS="-I$(find /usr/lib/gcc -name stdbool.h | head -1 | xargs dirname)"
export VIRTUAL_ENV=/opt/venv  # or your virtualenv path

cd lib/bindings/python
maturin develop --release

cd /workspace/dynamo
pip install -e .
```

> **Note**: If using the `rocm/sgl-dev` container, Rust/cargo/maturin are pre-installed.
> The `BINDGEN_EXTRA_CLANG_ARGS` fix is required because ROCm's bundled clang
> cannot find system GCC headers (`stdbool.h`) by default.

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `NIXL_PREFIX` | `/opt/rocm/rixl` | RIXL install prefix (read by nixl-sys build.rs) |
| `RIXL_PREFIX` | `/opt/rocm/rixl` | Alias for NIXL_PREFIX |
| `ROCM_PATH` | `/opt/rocm` | ROCm installation root |
| `HIP_PATH` | `/opt/rocm/hip` | HIP compiler path |
| `HIP_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | Visible GPU devices |
| `VLLM_ROCM_USE_AITER` | `1` | Enable AIter optimizations for vLLM on ROCm |

## Verifying the Build

Check that RIXL libraries are linked correctly:

```bash
ldd target/debug/libdynamo_memory.so | grep nixl
```

You should see references to `libnixl.so` under the RIXL prefix (`/opt/rocm/rixl/lib`).

Run the memory tests:

```bash
cargo test -p dynamo-memory --features testing-all-rocm
```

## Troubleshooting

**`libnixl.so not found`** — Ensure `NIXL_PREFIX` points to where RIXL is installed and that the RIXL build completed successfully. Check that `/opt/rocm/rixl/lib/libnixl.so` exists.

**UCX transport errors** — Verify UCX was built with ROCm support: `ucx_info -d | grep rocm` should show ROCm memory domains.

**Version mismatch** — The workspace `[patch.crates-io]` must be uncommented. Without it, Cargo resolves `nixl-sys = "=0.10.1"` from crates.io instead of using the RIXL path override.

**HIP not found** — Ensure `ROCM_PATH` and `HIP_PATH` are set and that `/opt/rocm/bin/hipcc` exists.
