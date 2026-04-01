# Dynamo on AMD ROCm — Deployment & Development Guide

> NVIDIA Dynamo ported to AMD Instinct MI355X GPUs with all four core features validated.

## Quick Start

```bash
# Pull the SGLang MoRI container (includes ROCm 7.2 + aiter + MoRI)
docker pull rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2

# Run standalone DSV3 inference (single node, 8 GPUs)
docker run --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v /path/to/DeepSeek-V3:/models/DeepSeek-V3 \
    rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2 \
    python3 -m sglang.launch_server \
        --model-path /models/DeepSeek-V3 \
        --tp-size 8 --attention-backend aiter \
        --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 \
        --chunked-prefill-size 32768 --mem-fraction-static 0.80
```

**Critical env var**: `export SGLANG_AITER_MLA_PERSIST=False` (11x TTFT improvement for DSV3).

---

## 1. Hardware & Network Setup

### Requirements

| Component | Specification |
|-----------|--------------|
| GPU | AMD Instinct MI355X (288GB HBM3e), 8 per node |
| Network | Pensando Pollara 400 AI NIC (400Gb/s RoCE v2) |
| OS | Ubuntu 22.04/24.04 |
| ROCm | 7.1.1+ |
| Container | `rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2` |

### AI NIC (Ionic) Configuration

Each MI355X node has 8 ionic IB devices (`ionic_0` through `ionic_7`). For cross-node RDMA:

**1. Find matching subnets between nodes:**
```bash
# On each node, check ionic GID subnets
for i in 0 1 2 3 4 5 6 7; do
    gid=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1)
    subnet=$(echo $gid | cut -d: -f1-4)
    echo "ionic_$i: $subnet"
done
```

Match ionic devices that share the same subnet between nodes. For example:
- Node A `ionic_0` subnet `:0148` ↔ Node B `ionic_1` subnet `:0148` ✓
- Node A `ionic_0` subnet `:014e` ↔ Node B `ionic_0` subnet `:0147` ✗

**2. Assign IPv4 addresses on matching interfaces:**
```bash
# Find the network interface for the ionic device
ls /sys/class/infiniband/ionic_0/device/net/  # e.g., enp233s0

# Assign IP (same /24 subnet on both nodes)
ip addr add 192.168.48.10/24 dev enp233s0  # Node A
ip addr add 192.168.48.11/24 dev enp9s0    # Node B

# Verify
ping -c 1 192.168.48.11  # Should respond in <1ms
```

**3. Configure QoS/DCQCN (per [ROCm docs](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/sglang-mori-distributed.html)):**
```bash
sudo nicctl update qos dscp-to-priority --dscp 0-63 --priority 0
sudo nicctl update qos dscp-to-priority --dscp 24 --priority 3
sudo nicctl update qos dscp-to-priority --dscp 46 --priority 6
sudo nicctl update qos pfc --priority 3 --no-drop enable
sudo nicctl update qos scheduling --priority 3,0,6 --dwrr 99,1,0 --rate-limit 0,0,10

for dev in ionic_0 ionic_1 ionic_2 ionic_3 ionic_4 ionic_5 ionic_6 ionic_7; do
    sudo nicctl update dcqcn -r $dev -i 1 --cnp-dscp 46
done
```

### Ionic Driver Fix (Container)

The container's `libionic` may have an ABI mismatch with the host kernel. Fix by mounting the host's driver:

```bash
docker run ... \
    -v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185:/host-ionic/libionic.so.1.1.54.0-185:ro \
    -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/host-ionic/libionic-rdmav34.so:ro \
    -v /etc/libibverbs.d/ionic.driver:/host-ionic/ionic.driver:ro

# Inside container:
cp /host-ionic/libionic.so.1.1.54.0-185 /usr/lib/x86_64-linux-gnu/
ln -sf libionic.so.1.1.54.0-185 /usr/lib/x86_64-linux-gnu/libionic.so.1
ldconfig
```

---

## 2. Docker Deployment

### Container Variants

| Image | Use Case |
|-------|----------|
| `rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2` | SGLang + MoRI (recommended) |
| `rocm/vllm:latest` | vLLM on ROCm |
| `rocm/pytorch:latest` | Base PyTorch + ROCm |

### Standard Launch

```bash
docker run -d --name dynamo-worker \
    --network=host --privileged --ulimit memlock=-1:-1 \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v /path/to/models:/workspace/models:ro \
    -v /path/to/dynamo:/workspace/dynamo:ro \
    rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2 \
    bash -c "
export SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 SGLANG_AITER_MLA_PERSIST=False
python3 -m sglang.launch_server --model-path /workspace/models/DeepSeek-V3 \
    --tp-size 8 --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --cuda-graph-max-bs 16 --host 0.0.0.0 --port 9000
"
```

### Building Dynamo in Container

```bash
# Copy source (avoid concurrent builds on shared mount)
cp -r /workspace/dynamo /tmp/dynamo

# Install build tools (rocm/sgl-dev image has Rust/maturin pre-installed)
apt-get install -y build-essential pkg-config libclang-dev curl protobuf-compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.93.1
export PATH=/root/.cargo/bin:$PATH

# Critical: set LIBCLANG_PATH and BINDGEN fix for stdbool.h
export LIBCLANG_PATH=/opt/rocm/lib/llvm/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I$(find /usr/lib/gcc -name stdbool.h | head -1 | xargs dirname)"
export VIRTUAL_ENV=/opt/venv

# Build Rust bindings (dynamo.llm)
cd /tmp/dynamo/lib/bindings/python && maturin develop --release

# Install Python package
cd /tmp/dynamo && pip install -e .

# Install test dependencies
pip install pytest pytest-benchmark pytest-httpserver pytest-asyncio \
    pytest-timeout nats-py kr8s prometheus_api_client filterpy pmdarima \
    prophet boto3 kubernetes_asyncio
```

**Verified**: 190+ tests pass, 42 skipped (expected), on MI355X with this recipe.

---

## 3. Dynamo Features on AMD

### 3.1 KV-Cache-Aware Routing

```bash
# Start infrastructure
etcd &
nats-server -p 4222 -js &
export ETCD_ENDPOINTS="http://localhost:2379" NATS_SERVER="nats://localhost:4222"

# Start frontend with KV Router
python3 -m dynamo.frontend --http-port 8000 --router-mode kv --kv-cache-block-size 16

# Start worker (use --page-size 16 for KV Router compatibility)
python3 -m dynamo.sglang --model /models/DeepSeek-V3 --tp-size 8 \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 \
    --page-size 16
```

**Result**: 4.35x TTFT improvement at c=32 vs Round-Robin (100% success vs 50%).

### 3.2 KVBM Multi-turn

```bash
# Enable KVBM with CPU cache offloading
export DYN_KVBM_CPU_CACHE_GB=20

python3 -m dynamo.sglang --model /models/DeepSeek-V3 --tp-size 8 ...
```

**Result**: 2.17x–3.34x TTFT improvement for multi-turn conversations.

### 3.3 Disaggregated Serving

Three RDMA backends are supported on AMD:

| Backend | Setup | Performance | Best For |
|---------|-------|-------------|----------|
| **MoRI** | Out of the box | 7.4 req/s DSV3, 106.6 req/s Qwen | Default choice |
| **RIXL + DRAM staging** | `export SGLANG_NIXL_ROCM_STAGING=1` | RDMA via pinned host bounce | When MoRI unavailable |
| **Mooncake RDMA** | `bash scripts/patch_mooncake_rocm.sh` | Requires rebuild | When Mooncake is required |

#### MoRI (recommended)

```bash
# PREFILL NODE (Node A, ionic_0)
python3 -m dynamo.sglang --model /models/DeepSeek-V3 --tp-size 8 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ionic_0

# DECODE NODE (Node B, ionic_1 — must match prefill's ionic subnet)
python3 -m dynamo.sglang --model /models/DeepSeek-V3 --tp-size 8 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ionic_1
```

#### RIXL + DRAM Staging

```bash
export SGLANG_NIXL_ROCM_STAGING=1  # auto-detected on ROCm
# Then use --disaggregation-transfer-backend nixl (same args as above)
```

#### Mooncake RDMA (requires patch + rebuild)

```bash
# One-time setup: patch and rebuild Mooncake in container
bash scripts/patch_mooncake_rocm.sh

# Then use --disaggregation-transfer-backend mooncake
```

**Result**: MoRI achieves 7.4 req/s, 475 tok/s with DSV3 671B, 100% success rate.

#### Multi-Node End-to-End Walkthrough

Complete steps to run disaggregated serving across 2 MI355X nodes.

**Prerequisites**: Two nodes (e.g. `<prefill-node>`, `<decode-node>`) with matching ionic subnets, model on shared storage.

**Step 1 — Start containers on both nodes:**

```bash
# Same command on both nodes
docker run -d --name dynamo-worker \
    --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v /path/to/dynamo:/workspace \
    -v /path/to/models:/models:ro \
    rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2 \
    sleep 86400
```

**Step 2 — Find matching ionic devices** (run inside each container):

```bash
for i in 0 1 2 3 4 5 6 7; do
    gid=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1 2>/dev/null)
    [ -n "$gid" ] && echo "ionic_$i  $(echo $gid | cut -d: -f1-4)"
done
# Match devices with the same subnet prefix between nodes
# Example: <prefill-node> ionic_2 (:0148) <-> <decode-node> ionic_0 (:0148)
```

**Step 3 — Prefill node** (e.g. `<prefill-node>`): start etcd, NATS, frontend, prefill worker:

```bash
docker exec -it dynamo-worker bash

export SGLANG_AITER_MLA_PERSIST=False
export RCCL_MSCCL_ENABLE=0
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MY_IP=$(hostname -I | awk '{print $1}')

# Install etcd + NATS (if not present)
which etcd  || { wget -q https://github.com/etcd-io/etcd/releases/download/v3.5.21/etcd-v3.5.21-linux-amd64.tar.gz -O /tmp/etcd.tar.gz && mkdir -p /usr/local/bin/etcd-dir && tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd-dir --strip-components=1 && ln -sf /usr/local/bin/etcd-dir/etcd /usr/local/bin/etcd; }
which nats-server || { wget -q https://github.com/nats-io/nats-server/releases/download/v2.10.28/nats-server-v2.10.28-amd64.deb -O /tmp/nats.deb && dpkg -i /tmp/nats.deb >/dev/null 2>&1; }

# Start infrastructure
rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://${MY_IP}:2379 &
nats-server -p 4222 -js --client_advertise ${MY_IP}:4222 &
sleep 3

export ETCD_ENDPOINTS="http://${MY_IP}:2379"
export NATS_SERVER="nats://${MY_IP}:4222"

# Build Dynamo (first time only)
cp -r /workspace/dynamo /tmp/dynamo && cd /tmp/dynamo
MODE=develop bash scripts/build_dynamo_wheel.sh

# Start Frontend
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
sleep 3

# Start Prefill Worker
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 \
    --tp-size 8 --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 \
    --chunked-prefill-size 32768 --mem-fraction-static 0.80 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ionic_2  # your matched device
```

**Step 4 — Decode node** (e.g. `<decode-node>`): start decode worker pointing to prefill's etcd/NATS:

```bash
docker exec -it dynamo-worker bash

export SGLANG_AITER_MLA_PERSIST=False
export RCCL_MSCCL_ENABLE=0
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Point to prefill node's infrastructure
PREFILL_IP=`<prefill-node-ip>`
export ETCD_ENDPOINTS="http://${PREFILL_IP}:2379"
export NATS_SERVER="nats://${PREFILL_IP}:4222"

# Build Dynamo (first time only)
cp -r /workspace/dynamo /tmp/dynamo && cd /tmp/dynamo
MODE=develop bash scripts/build_dynamo_wheel.sh

# Start Decode Worker
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 \
    --tp-size 8 --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 \
    --mem-fraction-static 0.80 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ionic_0  # your matched device
```

**Step 5 — Send requests** (from any machine):

```bash
# Single request
curl http://`<prefill-ip>`:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V3","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'

# Benchmark
python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url http://`<prefill-ip>`:8000 \
    --model DeepSeek-V3 \
    --num-prompts 100 --request-rate 8
```

#### Quick Test with Small Model

For fast validation (starts in seconds), replace the model args:

```bash
# On both nodes, use Qwen instead of DSV3:
--model-path Qwen/Qwen2.5-0.5B-Instruct --tp-size 1
# Remove: --kv-cache-dtype, --cuda-graph-max-bs, --chunked-prefill-size
```

---

## 4. Deployment Examples

### Example A — Single-Node Standalone (simplest)

One node, one worker, no disaggregation. Good for dev/test.

```bash
docker exec -it dynamo-worker bash

export SGLANG_AITER_MLA_PERSIST=False
export RCCL_MSCCL_ENABLE=0
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MY_IP=$(hostname -I | awk '{print $1}')

# etcd + NATS
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://${MY_IP}:2379 &
nats-server -p 4222 -js &
sleep 2
export ETCD_ENDPOINTS="http://${MY_IP}:2379"
export NATS_SERVER="nats://${MY_IP}:4222"

# Build dynamo (first time)
cp -r /workspace/dynamo /tmp/dynamo && cd /tmp/dynamo
MODE=develop bash scripts/build_dynamo_wheel.sh

# Frontend + Worker
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 \
    --tp-size 8 --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 \
    --chunked-prefill-size 32768 --mem-fraction-static 0.80

# Test: curl http://localhost:8000/v1/chat/completions -d '...'
```

### Example B — Single-Node with KV Router (shared-prefix caching)

Same as A but with KV-cache-aware routing. Best for workloads with repeated system prompts.

```bash
# Frontend with KV Router (not round-robin)
python3 -m dynamo.frontend --http-port 8000 \
    --router-mode kv --kv-cache-block-size 16 &

# Worker must use --page-size 16 to match
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 \
    --tp-size 8 --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 \
    --page-size 16 \
    --mem-fraction-static 0.80
```

**Result**: 4.35x TTFT improvement at c=32.

### Example C — 2-Node KV Router (scale-out)

Two worker nodes behind one frontend. KV Router directs requests based on prefix cache affinity.

```bash
# Node A: Frontend + etcd/NATS + Worker-1
python3 -m dynamo.frontend --http-port 8000 \
    --router-mode kv --kv-cache-block-size 16 &
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 \
    --tp-size 8 --page-size 16 \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --cuda-graph-max-bs 16 --mem-fraction-static 0.80

# Node B: Worker-2 (points to Node A's etcd/NATS)
export ETCD_ENDPOINTS="http://<nodeA-ip>:2379"
export NATS_SERVER="nats://<nodeA-ip>:4222"
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 \
    --tp-size 8 --page-size 16 \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --cuda-graph-max-bs 16 --mem-fraction-static 0.80
```

**Result**: 20 req/s, 1,279 tok/s at c=32 (100% success).

### Example D — KVBM Multi-Turn (conversation caching)

Single node with KV Block Manager for multi-turn conversation acceleration.

```bash
export DYN_KVBM_CPU_CACHE_GB=20  # offload KV to 20GB host memory

python3 -m dynamo.frontend --http-port 8000 --router-mode kv --kv-cache-block-size 16 &
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 \
    --tp-size 8 --page-size 16 \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --cuda-graph-max-bs 16 --mem-fraction-static 0.80
```

**Result**: 2.17-3.34x TTFT improvement for 15-user x 20-turn conversations.

### Example E — 2-Node Disagg with RIXL DRAM Staging

Same as the MoRI disagg walkthrough above, but using RIXL + DRAM staging:

```bash
# On BOTH nodes:
export SGLANG_NIXL_ROCM_STAGING=1

# Prefill node:
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 --tp-size 8 \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl

# Decode node:
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 --tp-size 8 \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl
```

### Example F — 2-Node Disagg with Mooncake RDMA

Apply the ROCm patch first, then run:

```bash
# One-time (in each container):
bash scripts/patch_mooncake_rocm.sh

# Prefill node:
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 --tp-size 8 \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake

# Decode node:
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 --tp-size 8 \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake
```

### Example G — Dynamic Planner (auto-scaling)

Single node, planner in virtual mode (no K8s needed):

```bash
export DYN_FORWARDPASS_METRIC_PORT=20380

# Frontend + Worker (same as Example A)
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
python3 -m dynamo.sglang \
    --model-path /models/DeepSeek-V3 --tp-size 8 \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 &

# Planner (separate terminal)
python3 -m dynamo.planner \
    --environment virtual --backend sglang --mode agg \
    --enable-load-scaling --ttft 1000 --itl 100 \
    --min-endpoint 1 --max-gpu-budget 16
```

### Example H — Quick Dev Loop with Qwen-0.5B

For rapid iteration — model loads in seconds, runs on 1 GPU:

```bash
export HIP_VISIBLE_DEVICES=0

# Standalone (no disagg)
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
python3 -m dynamo.sglang \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --tp-size 1 --trust-remote-code

# Disagg (single-node, 2 GPUs, one prefill + one decode)
export HIP_VISIBLE_DEVICES=0
python3 -m dynamo.sglang \
    --model-path Qwen/Qwen2.5-0.5B-Instruct --tp-size 1 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori &

export HIP_VISIBLE_DEVICES=1
python3 -m dynamo.sglang \
    --model-path Qwen/Qwen2.5-0.5B-Instruct --tp-size 1 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori
```

### Scenario Summary

| Example | Nodes | Feature | Backend | Model |
|---------|-------|---------|---------|-------|
| A | 1 | Standalone | — | DSV3 |
| B | 1 | KV Router | — | DSV3 |
| C | 2 | KV Router scale-out | — | DSV3 |
| D | 1 | KVBM multi-turn | — | DSV3 |
| E | 2 | Disagg P/D | RIXL DRAM | DSV3 |
| F | 2 | Disagg P/D | Mooncake | DSV3 |
| G | 1 | Dynamic Planner | — | DSV3 |
| H | 1 | Quick dev/test | MoRI | Qwen-0.5B |
| I | 1 | Planner + FPM relay | — | DSV3 |

> The MoRI disagg walkthrough (§3.3) covers Example E with MoRI backend — the most common production scenario.

---

### Example I — Dynamic Planner with FPM Relay

```bash
# Enable FPM relay for SGLang
export DYN_FORWARDPASS_METRIC_PORT=20380

# Planner in virtual mode (no K8s required)
python3 -m dynamo.planner \
    --environment virtual --backend sglang --mode agg \
    --enable-load-scaling --ttft 1000 --itl 100 \
    --min-endpoint 1 --max-gpu-budget 16
```

---

## 5. Slurm Deployment

Pre-built Slurm scripts in `scripts/slurm/`:

```bash
# Single node
sbatch scripts/slurm/dynamo_standalone.slurm

# 2-node KV Router
sbatch scripts/slurm/dynamo_kv_router.slurm

# 2-node disagg (MoRI RDMA)
sbatch scripts/slurm/dynamo_disagg.slurm

# 2-node disagg (RIXL DRAM staging)
BACKEND=nixl sbatch scripts/slurm/dynamo_disagg.slurm

# Quick test with small model
MODEL=/models/Qwen2.5-0.5B-Instruct TP=1 sbatch scripts/slurm/dynamo_standalone.slurm
```

Each script auto-installs etcd/NATS, builds Dynamo, and starts the workers inside Docker containers launched by `srun`. See [scripts/slurm/README.md](../scripts/slurm/README.md) for configuration options.

---

## 6. Kubernetes Deployment

### Prerequisites

- K3s/K8s cluster with AMD GPU Operator (`kube-amd-gpu`)
- `amd.com/gpu` resource available on nodes

### Install Dynamo CRDs

```bash
kubectl create namespace dynamo

# Apply CRDs (use --server-side for large schemas)
for crd in deploy/operator/config/crd/bases/*.yaml; do
    kubectl apply --server-side -f $crd
done

# Deploy etcd + NATS
kubectl apply -n dynamo -f deploy/helm/charts/platform/
```

### Deploy Workers

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-worker
  namespace: dynamo
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: worker
        image: rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2
        resources:
          limits:
            amd.com/gpu: 8
        env:
        - name: SGLANG_AITER_MLA_PERSIST
          value: "False"
```

---

## 7. Bug Fixes Applied

| Bug | Fix | Impact |
|-----|-----|--------|
| CUDA graph conflict | `SGLANG_AITER_MLA_PERSIST=False` | 11x TTFT improvement |
| KV Router block_size panic | Patched `multi_worker.rs` assertion | KV Router works with SGLang |
| Mooncake RDMA on AMD | Use MoRI backend instead | Disagg serving works |
| Ionic ABI mismatch | Install `libionic1 54.0-185` from host | MoRI RDMA works |
| Python 3.10 `typing.Self` | Conditional import | Import crash fixed |
| vLLM `OmniConfig` | Lazy import | Worker startup fixed |

---

## 8. Performance Summary

| Benchmark | Result |
|-----------|--------|
| Standalone DSV3 peak | **17.3 req/s, 1,108 tok/s** (c=16) |
| KV Router c=32 | **4.35x TTFT, 1,427 tok/s** (vs RR) |
| KVBM multi-turn | **2.17x–3.34x** TTFT improvement |
| MoRI disagg DSV3 | **7.4 req/s, 475 tok/s** (100% ok) |
| MoRI disagg Qwen | **106.6 req/s, P50=68ms** |
| RIXL DRAM transfer | **39.4 GB/s** (79% of 400Gb/s) |
| RCCL all_reduce | **406 GB/s** |

---

## 9. Repository Structure

```
dynamo/
├── lib/kv-router/src/sequences/multi_worker.rs  # block_size fix
├── components/src/dynamo/
│   ├── sglang/
│   │   ├── fpm_relay.py          # SGLang FPM for Dynamic Planner
│   │   └── publisher.py          # FPM integration
│   └── common/
│       └── forward_pass_metrics.py
├── docs/
│   └── amd-rocm-guide.md         # This file
└── deploy/
    └── operator/config/crd/       # K8s CRDs
```

**Branch**: `amd-dynamo` — 99.4% additive, upstream-rebaseable via `git rebase`.

---

## 10. Python Version Compatibility

Dynamo supports **Python 3.10, 3.11, and 3.12** via the PyO3 stable ABI (`abi3-py310`). A single Rust wheel works on all versions >= 3.10.

| Container | Python | vLLM | Dynamo | Notes |
|-----------|--------|------|--------|-------|
| `rocm/sgl-dev:...-mori-...` | 3.10 | ❌ | ✅ | Has Rust pre-installed |
| `rocm/vllm:latest` | 3.12 | ✅ | ✅ | Run `build_dynamo_wheel.sh` |
| `rocm/pytorch:latest` | 3.10 | ❌ | ✅ | General purpose |

**Verified**: `maturin develop --release` builds and passes 175 tests on Python 3.12 (`rocm/vllm:latest` on MI355X).

```bash
# Build Dynamo in any Python >= 3.10 container:
MODE=develop bash scripts/build_dynamo_wheel.sh
```

---

## References

- [ROCm SGLang + MoRI distributed](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/sglang-mori-distributed.html)
- [ROCm SGLang + Mooncake distributed](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/sglang-distributed.html)
- [ROCm vLLM inference](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html)
- [vLLM ROCm Docker image](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags)
- [AMD Instinct MI355X System Acceptance](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi355x.html)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)
