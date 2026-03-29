# Disaggregated Serving on AMD MI355X — Setup Guide

## Overview

Two validated paths for disaggregated prefill/decode serving on MI355X:

1. **SGLang + MoRI** — Production-validated, uses MoRI for inter-node GPU RDMA
2. **Dynamo + RIXL** — Under development, uses RIXL (NIXL port) for KV cache transfer

## Path 1: SGLang + MoRI (Production-Ready)

Reference: [ROCm docs — SGLang distributed inference with MoRI](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/sglang-mori-distributed.html)

### Prerequisites

- 2+ MI355X nodes with 8x Pensando Pollara 400 AI NICs each
- Slurm cluster for job orchestration
- `rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-0113` Docker image
- DeepSeek-R1 or DeepSeek-V3 model weights on shared NFS

### Quick Launch (1P1D, 2 Nodes)

Using InferenceX scripts at `/mnt/vast/john/rocm-dynamo/InferenceX`:

```bash
# Set environment
export IMAGE="rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-0113"
export MODEL_PATH="/path/to/models"
export MODEL_NAME="DeepSeek-R1"
export IBDEVICES="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
export MORI_RDMA_TC=104

# Configure 1P1D
export PREFILL_NODES=1
export DECODE_NODES=1
export PREFILL_NUM_WORKERS=1
export DECODE_NUM_WORKERS=1
export PREFILL_TP=8
export DECODE_TP=8
export PREFILL_EP=8
export DECODE_EP=8
export PREFILL_DP_ATTN=true
export DECODE_DP_ATTN=true

# Submit via Slurm
cd /mnt/vast/john/rocm-dynamo/InferenceX/benchmarks/multi_node/amd_utils
bash submit.sh 1 1 1 1 1024 1024 "128x256x512" inf true true true true 8 8 1
```

### Key Environment Variables

From `InferenceX/benchmarks/multi_node/amd_utils/env.sh`:

```bash
# IB devices (auto-detected or set by runner)
export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7

# Network interfaces (auto-detected from default route)
export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
export NCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export NCCL_IB_HCA=$IBDEVICES

# MoRI settings
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_MORI_FP8_DISP=True
export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384
export MORI_MAX_DISPATCH_TOKENS_DECODE=160
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export MORI_IO_QP_MAX_SEND_WR=16384

# RCCL/ANP for collective operations
export LD_PRELOAD="/opt/amd-anp/build/librccl-anp.so /opt/rccl/build/release/librccl.so.1.0"
export IONIC_LOCKFREE=all
export NCCL_IB_GID_INDEX=1
export NCCL_IB_TC=104
```

### Network Architecture

Each MI355X node has 8 GPUs and 8 Pensando NICs in 1:1 mapping.
MoRI uses RDMA directly on these NICs for dispatch/combine operations.

```
Node 1 (Prefill)                 Node 2 (Decode)
┌─────────────┐                  ┌─────────────┐
│ GPU 0 ←→ NIC 0 │ ──── RDMA ──── │ NIC 0 ←→ GPU 0 │
│ GPU 1 ←→ NIC 1 │ ──── RDMA ──── │ NIC 1 ←→ GPU 1 │
│ ...            │                │ ...            │
│ GPU 7 ←→ NIC 7 │ ──── RDMA ──── │ NIC 7 ←→ GPU 7 │
└─────────────┘                  └─────────────┘
```

## Path 2: Dynamo + RIXL (Under Development)

### Current Status

| Component | Status |
|-----------|--------|
| RIXL build (UCX 1.19 + ROCm) | PASS — built in tasimage container |
| RIXL Rust bindings (nixl-sys) | PASS — cargo check with RIXL |
| RCCL 2-node (ANP, 8 GPU) | PASS — 406 GB/s busbw |
| HIP kernel (tensor_kernels) | PASS — compiled on gfx950 |
| Dynamo Dockerfiles (rocm device) | PASS — additive blocks added |
| Dynamo native extension | PENDING — needs maturin build |
| vLLM + Dynamo frontend | PENDING — needs full container |
| RIXL UCX inter-node VRAM | PENDING — needs UCX_NET_DEVICES config |

### RIXL UCX Configuration for Pensando ionic

The GID[1] addresses on the ionic devices use IPv6 link-local (fd93:...).
UCX needs explicit device configuration:

```bash
export UCX_NET_DEVICES=ionic_0:1,ionic_1:1,...,ionic_7:1
export UCX_TLS=rc_v,ud_v,sm,self
export UCX_IB_GID_INDEX=1
```

### Building Dynamo with RIXL

See `docs/amd-rocm-build.md` for the full build guide.

```bash
# In ROCm container
cd /workspace/dynamo

# Activate RIXL override
# Uncomment [patch.crates-io] nixl-sys in Cargo.toml

export NIXL_PREFIX=/opt/rocm/rixl
export ROCM_PATH=/opt/rocm
cargo build --features rocm
maturin develop --features rocm
```

## Validated RCCL Performance

Single-node 8-GPU (rccl-tests, ANP, tasimage/primus:pr-591-ainic):

| Message Size | busbw (GB/s) |
|-------------|-------------|
| 1 MB | 28 |
| 64 MB | 351 |
| 256 MB | 394 |
| 1 GB | 399 |
| 4 GB | **406** |

Key: `LD_PRELOAD=/opt/amd-anp/build/librccl-anp.so`, `IONIC_LOCKFREE=all`

## Next Steps

1. Pull SGLang MoRI Docker image and run 1P1D with small model
2. Configure UCX_NET_DEVICES for RIXL inter-node VRAM transfer
3. Build Dynamo native extension and test vLLM + Dynamo frontend
4. Compare Dynamo RIXL path vs SGLang MoRI path for KV cache transfer
