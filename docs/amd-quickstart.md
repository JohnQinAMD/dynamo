# Dynamo on AMD ROCm — Quickstart

> Get Dynamo running on AMD Instinct GPUs (MI300X / MI325X / MI355X) in minutes.

## Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | AMD Instinct MI300X, MI325X, or MI355X |
| Host driver | ROCm 7.1+ (`/opt/rocm` present) |
| Docker | 24.0+ with `--device=/dev/kfd` support |
| Networking | Pensando ionic 400Gb NICs (for disaggregated serving) |

## Install Dynamo

**Option A: Container (Recommended)**

```bash
# SGLang + Dynamo + MoRI (MI355X optimized)
docker run --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -it amdprimus/dynamo-rocm-sglang:latest
```

**Option B: Build from Source**

```bash
# Inside a ROCm container (e.g. rocm/pytorch:latest)
git clone https://github.com/ai-dynamo/dynamo.git && cd dynamo
export LIBCLANG_PATH=/opt/rocm/lib/llvm/lib
export VIRTUAL_ENV=/opt/venv

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.93.1
export PATH=/root/.cargo/bin:$PATH

# Build Dynamo wheel
cd lib/bindings/python && maturin develop --release && cd ../../..
pip install -e .
```

## Run Dynamo (Single Node)

Start the frontend, then start a worker:

```bash
# Terminal 1: Start frontend
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
sleep 2

# Terminal 2: Start SGLang worker
python3 -m dynamo.sglang \
    --model-path Qwen/Qwen3-0.6B \
    --tp-size 1 --trust-remote-code \
    --attention-backend aiter
```

For large models (DeepSeek-R1, 8 GPUs):

```bash
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
sleep 2
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-R1-0528 \
    --tp-size 8 --trust-remote-code \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3
```

## Test Your Deployment

```bash
curl localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-0.6B",
         "messages": [{"role": "user", "content": "Hello!"}],
         "max_tokens": 50}'
```

## Disaggregated Serving (Multi-Node)

Disaggregated prefill-decode serving with MoRI RDMA KV cache transfer.

### Prerequisites

1. **Fix libionic ABI** in each container (bind-mount is unreliable; use `docker cp`):
```bash
# From the HOST, after docker run:
HOST_LIB=$(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* | head -1)
docker cp "$HOST_LIB" CONTAINER:/usr/lib/x86_64-linux-gnu/libionic.so.1

# Verify inside container:
ibv_devinfo 2>&1 | grep hca_id | wc -l   # must be 8

# Or use the helper script from inside the container:
bash scripts/benchmark/setup_network.sh --verify
# Or from the host:
bash scripts/benchmark/setup_network.sh --fix-abi CONTAINER_NAME
```

2. **Configure ionic IPv4** on all nodes:
```bash
bash scripts/benchmark/setup_network.sh        # auto-detect node ID
bash scripts/benchmark/setup_network.sh --verify  # check config
```

3. **Start etcd + NATS** on the head node:
```bash
etcd &
nats-server -p 4222 -js &
# Use ip route (not hostname -I, which may return ionic IP on MI355X nodes)
MY_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
export ETCD_ENDPOINTS=http://${MY_IP}:2379
export NATS_SERVER=nats://${MY_IP}:4222
```

### 1P1D (1 Prefill + 1 Decode, 2 nodes)

**Head node (prefill + frontend):**
```bash
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-R1-0528 \
    --tp-size 8 --trust-remote-code \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 \
    --mem-fraction-static 0.8 --disable-radix-cache
```

**Decode node:**
```bash
export ETCD_ENDPOINTS=http://<HEAD_IP>:2379
export NATS_SERVER=nats://<HEAD_IP>:4222
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-R1-0528 \
    --tp-size 8 --trust-remote-code \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 \
    --mem-fraction-static 0.85 --prefill-round-robin-balance
```

### EP/DP-Attn (DEP8, High Throughput)

For maximum throughput with MoE expert parallelism:

```bash
# Required environment variables
export MORI_SHMEM_MODE=ISOLATION
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=4

# Decode worker with EP/DP-Attn
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-R1-0528 \
    --tp-size 8 --ep-size 8 --dp-size 8 \
    --moe-a2a-backend mori --deepep-mode normal \
    --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ionic_0,...,ionic_7
```

## Transfer Backends

| Backend | Use Case | Config |
|---------|----------|--------|
| **MoRI** | Production (best performance) | `--disaggregation-transfer-backend mori` |
| **Mooncake** | Alternative RDMA (requires Dynamo frontend) | `--disaggregation-transfer-backend mooncake` |
| **RIXL/nixl** | Cross-node with DRAM staging | `--disaggregation-transfer-backend nixl` |

## Benchmarking (InferenceX-Aligned)

Two paths: **manual** (SSH into nodes) or **InferenceX CI** (Slurm).
See [`scripts/benchmark/README.md`](../scripts/benchmark/README.md) for full step-by-step.

### Quick Start (Manual, Single-Node)

```bash
# Pre-flight: clean VRAM
docker rm -f $(docker ps -aq)

# Start container
docker run -d --name bench --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v $(pwd):/workspace -v /path/to/models:/models \
    amdprimus/dynamo-rocm-sglang:latest tail -f /dev/null

# Fix ionic ABI (from HOST)
bash scripts/benchmark/setup_network.sh --fix-abi bench

# Run benchmark (inside container)
docker exec -it bench bash
cd /workspace/dynamo/scripts/benchmark
export MODEL_DIR=/models MODEL_NAME=DeepSeek-R1-0528
export FRONTEND_TYPE=dynamo
export BENCH_MAX_CONCURRENCY="4 8 16 32 64"
bash server.sh
```

### Multi-Node Disaggregated (Manual)

```bash
# On EACH node: start container + fix ABI (from HOST)
docker run -d --name bench ... amdprimus/dynamo-rocm-sglang:latest tail -f /dev/null
bash scripts/benchmark/setup_network.sh --fix-abi bench

# Inside container on each node: configure ionic IPv4
docker exec bench bash -c 'bash /workspace/dynamo/scripts/benchmark/setup_network.sh'

# Head node (NODE_RANK=0): runs prefill + frontend + benchmarks
docker exec -it bench bash -c '
    cd /workspace/dynamo/scripts/benchmark
    export MODEL_DIR=/models MODEL_NAME=DeepSeek-R1-0528
    export FRONTEND_TYPE=dynamo IPADDRS="10.0.0.1,10.0.0.2" xP=1 yD=1 NODE_RANK=0
    export PREFILL_TP_SIZE=8 DECODE_TP_SIZE=8
    bash server.sh'

# Decode node (NODE_RANK=1): runs decode worker
docker exec -it bench bash -c '
    cd /workspace/dynamo/scripts/benchmark
    export MODEL_DIR=/models MODEL_NAME=DeepSeek-R1-0528
    export FRONTEND_TYPE=dynamo IPADDRS="10.0.0.1,10.0.0.2" xP=1 yD=1 NODE_RANK=1
    export PREFILL_TP_SIZE=8 DECODE_TP_SIZE=8
    bash server.sh'
```

### InferenceX CI (Slurm)

```bash
cd InferenceX/benchmarks/multi_node/amd_utils

export SLURM_ACCOUNT=myaccount SLURM_PARTITION=compute TIME_LIMIT=08:00:00
export MODEL_PATH=/nfsdata MODEL_NAME=deepseek-ai/DeepSeek-R1-0528
export CONTAINER_IMAGE=amdprimus/dynamo-rocm-sglang:latest
export RUNNER_NAME=dsr1-fp8-mi355x
export FRONTEND_TYPE=dynamo   # optional, default: sglang

# 1 prefill node + 2 decode nodes, ISL/OSL=1024, concurrencies 4-256
bash submit.sh  1 1  2 2  1024 1024  "4x8x16x32x64x128x256"  inf
```

Slurm automatically handles: container creation, ionic ABI fix, IPv4 config, server launch, and benchmarking.

## AMD-Specific Environment Variables

```bash
# Required
export SGLANG_USE_AITER=1              # Enable aiter kernels
export RCCL_MSCCL_ENABLE=0             # Disable MSCCL
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Performance
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_MLA_PERSIST=False  # Fix 11x TTFT on DeepSeek

# Disagg timeouts
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ibv_devinfo` shows ABI warnings | `docker cp` host libionic into container (see Prerequisites). Bind-mount (`-v`) is unreliable with symlinks. |
| `std::bad_cast` on disagg startup | Run `setup_network.sh` to configure ionic IPv4 |
| `ibv_modify_qp` timeout | Check ionic subnet matching: `setup_network.sh --match <REMOTE>` |
| Very slow first inference (~5 min) | Normal: aiter JIT + CUDA graph capture. Send warmup request. |
| Port 8000 conflict | Use `--http-port 9000` for frontend when co-located with worker |

## Next Steps

- [Ionic RDMA Fix Guide](ionic-rdma-fixes.md) — all ionic RDMA issues, cross-backend fixes, lessons learned
- [Feature Test Runbook](amd-feature-test-runbook.md) — full test suite, ionic debugging
- [Deployment Guide](amd-rocm-guide.md) — containers, Kubernetes, production config
- [System Design](amd-system-design.md) — architecture and data flow diagrams
- [Performance Dashboard](mi355x-benchmark-dashboard.html) — benchmark results
