# Dynamo on AMD ROCm — Feature Test Runbook

> Reproducible, end-to-end validation of every Dynamo feature on AMD Instinct GPUs.  
> **Tested on**: MI355X with Pensando Pollara 400 ionic NICs, ROCm 7.2, SGLang 0.5.9  
> **Last updated**: 2026-04-01

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Single-Node Feature Tests](#single-node-feature-tests)
4. [Multi-Node Disaggregated Serving](#multi-node-disaggregated-serving)
5. [Automated Test Suite](#automated-test-suite)
6. [Benchmarking](#benchmarking)
7. [Troubleshooting](#troubleshooting)
8. [Results Summary](#results-summary)

---

## Prerequisites

### Hardware

| Component | Requirement |
|-----------|-------------|
| GPU | AMD Instinct MI300X, MI325X, or MI355X |
| Host driver | ROCm 7.1+ installed (`/opt/rocm` present) |
| Networking | For disagg tests: 2+ nodes with Pensando ionic 400Gb NICs on matching subnets |

### Docker Image

Three pre-built images are available:

| Image | Backend | Includes |
|-------|---------|----------|
| `Dockerfile.rocm-sglang` | SGLang | Dynamo, SGLang, MoRI, RIXL, Mooncake, etcd, NATS, kv-indexer |
| `Dockerfile.rocm-vllm` | vLLM | Dynamo, vLLM, RIXL, etcd, NATS, kv-indexer |
| `Dockerfile.rocm-atom` | Atom (MI355X) | Dynamo, Atom, RIXL, etcd, NATS, kv-indexer, KVBM HIP (gfx950) |

Build from source:

```bash
cd /path/to/repo
docker build -f dynamo/container/Dockerfile.rocm-sglang -t dynamo-rocm-sglang:latest .
# Atom (MI355X only):
docker build -f dynamo/container/Dockerfile.rocm-atom -t dynamo-rocm-atom:latest .
# Atom with MTP support:
docker build -f dynamo/container/Dockerfile.rocm-atom \
    --build-arg ATOM_IMAGE=rocm/atom:rocm7.2.0-ubuntu24.04-pytorch2.9-atom0.1.1 \
    -t dynamo-rocm-atom:mtp .
```

### Launch Container

```bash
docker run -d --name my-dynamo \
    --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v /path/to/hf_cache:/root/.cache/huggingface \
    dynamo-rocm-sglang:latest bash
```

### Ionic Driver ABI Fix (REQUIRED for MoRI RDMA)

Container and host `libionic` versions almost always differ, causing `ibv_devinfo` to show
0 devices (all ionic ports invisible). **Use `docker cp` to fix** — bind-mount (`-v`) is
unreliable because Docker does not correctly override symlink targets inside the container.

```bash
# From the HOST — copy host libionic into the running container:
HOST_LIB=$(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* | head -1)
docker cp "$HOST_LIB" my-dynamo:/usr/lib/x86_64-linux-gnu/libionic.so.1

# Or use the helper script:
# bash scripts/benchmark/setup_network.sh --fix-abi my-dynamo

# Verify INSIDE container — must show 8 devices with no ABI warnings:
docker exec my-dynamo ibv_devinfo 2>&1 | grep hca_id | wc -l   # expect: 8
```

> **Why not bind-mount?** Docker's `-v host.so:/container/libionic.so.1:ro` should work in
> theory, but in practice the container's `libionic.so.1` is a symlink to
> `libionic.so.1.0.54.0-149.gXXXXXXXX`. Docker follows the symlink and mounts at the target
> path, leaving the symlink itself unchanged. The result: the old library is still loaded via
> the symlink. `docker cp` directly overwrites the symlink with the correct file.

---

## Quick Start

Start infrastructure and a worker, then send a request — all in one block:

```bash
# Environment
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_AITER_MLA_PERSIST=False
export RCCL_MSCCL_ENABLE=0
MY_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')  # NOT hostname -I (may return ionic IP)

# Infrastructure
rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://${MY_IP}:2379 &
nats-server -p 4222 -js --client_advertise ${MY_IP}:4222 &
sleep 3

export ETCD_ENDPOINTS=http://${MY_IP}:2379
export NATS_SERVER=nats://${MY_IP}:4222

# Frontend + Worker
cd /opt/dynamo
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
sleep 2
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code &

# Wait for worker readiness (~50s for model load + warmup)
echo "Waiting for worker..."
for i in $(seq 1 60); do
    curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && break
    sleep 2
done

# Test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
```

---

## Single-Node Feature Tests

> All tests below assume the Quick Start infrastructure is running.

### Test 1 — Chat Completion

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Count to 5"}],"max_tokens":50}'
```

**Expected**: JSON response with generated text.

### Test 2 — Tool Calling

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role":"user","content":"What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {"type":"object","properties":{"city":{"type":"string"}}}
      }
    }],
    "max_tokens": 300
  }'
```

**Expected**: Response references `get_weather` / `Paris`.

### Test 3 — Streaming (SSE)

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20,"stream":true}' \
  | grep -c "data:"
```

**Expected**: ≥ 2 SSE chunks.

### Test 4 — Multi-Turn Conversation

```bash
for turn in 1 2 3 4 5; do
    curl -sf http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"Turn $turn\"}],\"max_tokens\":20}" \
      > /dev/null && echo "Turn $turn: OK" || echo "Turn $turn: FAIL"
done
```

**Expected**: All 5 turns complete successfully.

### Test 5 — Speculative Decoding CLI

```bash
python3 -m sglang.launch_server --help 2>&1 | grep -c "speculative"
```

**Expected**: ≥ 50 speculative-decoding related arguments (EAGLE, EAGLE3, NGRAM, NEXTN).

### Test 6 — Request Migration (worker failover)

```bash
pkill -f "dynamo.sglang" 2>/dev/null; sleep 2

# Start two workers on separate GPUs
HIP_VISIBLE_DEVICES=0 python3 -m dynamo.sglang \
    --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code &
W1=$!
HIP_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \
    --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code &

# Wait for both workers
for i in $(seq 1 60); do
    models=$(curl -sf http://localhost:8000/v1/models 2>/dev/null)
    [ "$(echo "$models" | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("data",[])))' 2>/dev/null)" -ge 1 ] && break
    sleep 2
done
sleep 5

# Start long stream, kill first worker mid-stream
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Write a long essay about AI"}],"max_tokens":300,"stream":true}' \
  > /tmp/stream.txt &
sleep 3
kill $W1 2>/dev/null; wait $W1 2>/dev/null
wait  # wait for curl to finish

echo "Chunks received: $(grep -c 'data:' /tmp/stream.txt)"
```

**Expected**: ≥ 100 chunks — request migrated to surviving worker.

### Test 7 — Multimodal (Vision-Language)

```bash
pkill -f "dynamo.sglang" 2>/dev/null; sleep 2

python3 -m dynamo.sglang \
    --model-path Qwen/Qwen3-VL-2B-Instruct --tp-size 1 --trust-remote-code &

# Wait for model (VL models need ~4 min for aiter JIT + graph capture on first load)
for i in $(seq 1 150); do
    curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && break
    sleep 2
done

# Create a red test image and encode as base64
B64=$(python3 -c "
import base64, io; from PIL import Image
img = Image.new('RGB', (100, 100), color=(255, 0, 0))
buf = io.BytesIO(); img.save(buf, format='PNG')
print(base64.b64encode(buf.getvalue()).decode())
")

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen/Qwen3-VL-2B-Instruct\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this image.\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${B64}\"}}
      ]
    }],
    \"max_tokens\": 100
  }"
```

**Expected**: Model describes a red image.

> **Note**: First inference after model load takes ~5 minutes on ROCm due to aiter JIT kernel compilation (~135s) and CUDA graph capture (~120s). Subsequent requests are fast.

---

## Multi-Node Disaggregated Serving

> Requires 2+ nodes with ionic NICs on the same switch fabric. Complete Prerequisites on ALL nodes.
>
> **Run preflight first:** `bash dynamo/scripts/preflight_check.sh <node1> <node2> [node3]`
> This validates SSH, Docker, GPU, ionic (devices + IPv4 + subnet match), NFS mount, and management IP in one pass. See [preflight docs](ionic-rdma-fixes.md#preflight-check-script).

**Important notes:**
- `--ep-dispatch-algorithm fake` (in InferenceX base_flags) **crashes in TP-only mode**. Only use it when EP is enabled.
- Router `--decode` requires **separate flags per URL**: `--decode http://d1:8000 --decode http://d2:8000` (not space-separated).
- `setup_ionic_network.sh` can exit successfully without assigning all 8 IPv4 addresses. Always verify 8/8 after running.
- Known excluded nodes: chi2811 (no ionic), chi2881 (no NFS/vast mount).

### Ionic Network Configuration (CRITICAL)

MoRI RDMA requires **every ionic port to have an IPv4 address**. Without IPv4, the RDMA GID table lacks IPv4-mapped entries and `ibv_modify_qp` fails with `EINVAL`, causing silent process death. This is the most common cause of MoRI disagg failures.

**Step 1: Fix ionic driver ABI** (from HOST, after `docker run`):

```bash
# Use docker cp — bind-mount (-v) is unreliable with symlinks (see "Launch Container" above)
HOST_LIB=$(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* | head -1)
docker cp "$HOST_LIB" <container>:/usr/lib/x86_64-linux-gnu/libionic.so.1

# Or use the helper:
bash scripts/benchmark/setup_network.sh --fix-abi <container>

# Verify INSIDE container:
ibv_devinfo 2>&1 | grep hca_id   # must show 8 devices, no ABI warnings
```

**Step 2: Assign IPv4 addresses to ALL ionic ports on EVERY node**:

The setup script auto-detects ionic interfaces and assigns subnets based on GID prefixes,
so it works on any MI355X hardware regardless of interface naming:

```bash
# Run on EACH node — auto-detects node ID from hostname
bash dynamo/scripts/setup_ionic_network.sh

# Or specify node ID explicitly (must be unique per node, 1-254)
bash dynamo/scripts/setup_ionic_network.sh --node-id 99
```

The script reads each ionic device's GID subnet prefix from sysfs and derives a unique IP subnet,
ensuring devices on the same physical switch get the same IP subnet across nodes.

**Step 3: Verify cross-node connectivity** on all 8 paths:

```bash
# From node A, ping node B on each subnet
for subnet in 20 21 22 23 24 25 26 27; do
    ping -c 1 -W 1 192.168.${subnet}.<NODE_B_ID> && echo "subnet $subnet: OK" || echo "subnet $subnet: FAIL"
done
```

**Step 4: Verify RDMA GIDs** show IPv4-mapped entries:

```bash
ibv_devinfo -d ionic_0 -v 2>&1 | grep GID
# Should show: GID[2]: ::ffff:192.168.20.<NODE_ID>, RoCE v2
# If only link-local (fe80::) and subnet (fd93::) GIDs appear, IPv4 is NOT configured
```

> **Automated setup**: Use the provided script to configure all steps automatically:
> ```bash
> # On each node (auto-detects node ID from hostname):
> bash dynamo/scripts/setup_ionic_network.sh
>
> # Or specify node ID explicitly:
> bash dynamo/scripts/setup_ionic_network.sh --node-id 99
>
> # Verify configuration:
> bash dynamo/scripts/setup_ionic_network.sh --verify
>
> # Test cross-node connectivity:
> bash dynamo/scripts/setup_ionic_network.sh --verify --remote-id 100
> ```

> **Troubleshooting**: If `ibv_modify_qp` fails with `Invalid argument` or processes die with
> `RuntimeError: std::bad_cast`, the most likely cause is missing IPv4 addresses on ionic ports.
> See [Troubleshooting](#troubleshooting) for details.

### Test 9 — MoRI RDMA (Recommended)

MoRI is AMD's native RDMA library for Pensando ionic NICs.

**Prefill node** (runs etcd + NATS + frontend):

```bash
# Start infrastructure (see Quick Start), then:
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device <PREFILL_IONIC_DEV>
```

**Decode node** (points to prefill's etcd/NATS):

```bash
export ETCD_ENDPOINTS=http://<PREFILL_IP>:2379
export NATS_SERVER=nats://<PREFILL_IP>:4222

python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device <DECODE_IONIC_DEV>
```

> **Critical flags**:
> - `--host 0.0.0.0` — required; SGLang defaults to `127.0.0.1` which blocks cross-node bootstrap
> - `--disaggregation-ib-device` — must be a subnet-matched ionic device (see above)

**Verified results** (Qwen3-0.6B, 2× MI355X):

| Concurrency | P50 Latency | Throughput | Success |
|:-----------:|:-----------:|:----------:|:-------:|
| 1 | 73 ms | 0.4 req/s | 100% |
| 4 | 90 ms | 39.9 req/s | 100% |
| 8 | 95 ms | 81.7 req/s | 100% |

### Test 10 — Mooncake RDMA + DRAM Staging

Ionic NICs cannot register GPU VRAM for RDMA (`ibv_reg_mr ENOMEM`). The DRAM staging monkey-patch bounces data through pinned host memory automatically.

**Both nodes** — set staging env vars:

```bash
export SGLANG_MOONCAKE_ROCM_STAGING=1
export MC_MAX_SGE=2
```

**Prefill node**:

```bash
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake
```

**Decode node**:

```bash
export ETCD_ENDPOINTS=http://<PREFILL_IP>:2379
export NATS_SERVER=nats://<PREFILL_IP>:4222

python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake
```

> **Key points**:
> - `SGLANG_MOONCAKE_ROCM_STAGING=1` activates DRAM staging (auto-applied in the Docker image)
> - `MC_MAX_SGE=2` is required for ionic NICs (set in Docker image by default)
> - No `--disaggregation-ib-device` needed — Mooncake auto-discovers NICs

**Verified results**:

| Concurrency | P50 Latency | Throughput | Success |
|:-----------:|:-----------:|:----------:|:-------:|
| 1 | 85 ms | 10.8 req/s | 100% |
| 4 | 113 ms | 25.2 req/s | 100% |
| 8 | 109 ms | 61.1 req/s | 100% |

### Test 11 — RIXL + C++ DRAM Staging

RIXL (AMD's port of NIXL) includes native C++ DRAM staging in its UCX plugin. On NICs without GPUDirect RDMA, VRAM registration automatically falls back to pinned DRAM — no Python monkey-patch or env vars needed.

**Prefill node**:

```bash
export SGLANG_USE_AITER=0
export UCX_TLS=rc_v,tcp

python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
HIP_VISIBLE_DEVICES=0 python3 -m dynamo.sglang \
    --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --mem-fraction-static 0.015 \
    --attention-backend triton --disable-cuda-graph
```

**Decode node**:

```bash
export SGLANG_USE_AITER=0
export UCX_TLS=rc_v,tcp
export ETCD_ENDPOINTS=http://<PREFILL_IP>:2379
export NATS_SERVER=nats://<PREFILL_IP>:4222

HIP_VISIBLE_DEVICES=0 python3 -m dynamo.sglang \
    --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --mem-fraction-static 0.016 \
    --attention-backend triton --disable-cuda-graph
```

> **Key points**:
> - C++ DRAM staging is automatic — no env var needed
> - `UCX_TLS=rc_v,tcp` enables RDMA data path + TCP control plane
> - `SGLANG_USE_AITER=0` + `--attention-backend triton` avoids aiter JIT segfaults on gfx950
> - Data path: GPU → hipMemcpy D2H → DRAM → RDMA WRITE → DRAM → hipMemcpy H2D → GPU

---

## Automated Test Suite

### Quick Smoke Test (~30s, no GPU needed)

```bash
python3 -m pytest --override-ini=filterwarnings=default \
    tests/basic/test_bug_fixes.py \
    tests/disagg/test_nixl_rocm_staging.py \
    tests/basic/test_rocm_version_consistency.py \
    --no-header -q --tb=no
```

**Expected**: 52 passed.

### ROCm GPU Tests (~2 min)

```bash
python3 -m pytest --override-ini=filterwarnings=default \
    tests/basic/test_rocm_gpu_detection.py \
    tests/disagg/test_ionic_validation.py \
    -v --tb=short --timeout=120
```

**Expected**: 20 passed.

### Full Suite (~12 min)

```bash
python3 -m pytest --override-ini=filterwarnings=default \
    tests/router/test_router_e2e_with_mockers.py \
    tests/router/test_router_block_size_regression.py \
    tests/frontend/test_completion_mocker_engine.py \
    tests/mocker/test_config.py \
    tests/planner/unit/ \
    tests/planner/test_fpm_relay_sglang.py \
    tests/planner/test_planner_virtual_sglang.py \
    tests/planner/test_replica_calculation.py \
    tests/planner/test_load_generator.py \
    tests/global_planner/unit/ \
    tests/serve/test_prometheus_exposition_format_injection.py \
    tests/test_predownload_models.py \
    -v --tb=short --timeout=300
```

> **Note**: Use `--override-ini=filterwarnings=default` to prevent `PytestAssertRewriteWarning` crash caused by `anyio` pre-import in the container.

### Test Results (MI355X, latest run)

| Suite | Passed | Failed | Skipped | Notes |
|-------|-------:|-------:|--------:|-------|
| Router E2E | **27** | 1 | 0 | `kv-aic` requires NVIDIA `aiconfigurator` — expected |
| Bug-Fix Unit Tests | **39** | 0 | 0 | Validates BUG-1/2/3/4, H-1/2/3/5/6/7/9/10/11/13/14 fixes |
| Frontend Mocker | **4** | 0 | 0 | |
| Mocker Config | **7** | 0 | 0 | |
| Prometheus Metrics | **10** | 0 | 0 | |
| Block Size Regression | **2** | 0 | 1 | `block_size=1` xfail (upstream KV Router limitation) |
| Planner Config | **8** | 0 | 0 | |
| Load Predictors | **26** | 0 | 0 | |
| Load Based Scaling | **21** | 0 | 0 | |
| Replica Calculation | **10** | 0 | 0 | |
| Global Planner | **10** | 0 | 0 | |
| FPM Relay | **6** | 0 | 0 | |
| NIXL ROCm Staging | **12** | 0 | 0 | |
| ROCm GPU Detection | **13** | 0 | 0 | |
| ROCm Version Consistency | **1** | 0 | 0 | |
| FT etcd HA | **2** | 0 | 0 | |
| K8s CRD dry-run | **5** | 0 | 0 | |
| Ionic Validation | **7** | 0 | 0 | |
| Process Teardown | **6** | 1 | 0 | Docker PID namespace limitation |
| Predownload Models | **2** | 0 | 4 | |
| **Total** | **~217** | **~2** | **~50** | |

**Remaining 2 failures** (both non-code):

| Test | Root Cause | Impact |
|------|-----------|--------|
| `test_mocker_router[tcp-kv-aic]` | Requires `aiconfigurator` module (`--aic-system h200_sxm`) — NVIDIA-specific | None on AMD |
| `test_child_in_own_pgid_killed` | `os.fork()` + `setpgid` child not visible via `psutil` in Docker | Docker limitation, not a code bug |

---

## Benchmarking

Run after any serving test is working:

```bash
python3 -c "
import requests, time, concurrent.futures

URL = 'http://localhost:8000/v1/chat/completions'
MODEL = 'Qwen/Qwen3-0.6B'

for conc in [1, 4, 8]:
    N = max(conc * 3, 8)

    def send(i):
        t0 = time.time()
        try:
            r = requests.post(URL, json={
                'model': MODEL,
                'messages': [{'role': 'user', 'content': f'Question {i}'}],
                'max_tokens': 32, 'temperature': 0.7
            }, timeout=30)
            return {'ms': (time.time() - t0) * 1000, 'ok': r.status_code == 200}
        except Exception:
            return {'ms': (time.time() - t0) * 1000, 'ok': False}

    # Warmup
    for w in range(2):
        send(w)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(conc) as pool:
        results = list(pool.map(send, range(N)))
    wall = time.time() - t0

    ok = [r for r in results if r['ok']]
    p50 = sorted([r['ms'] for r in ok])[len(ok) // 2] if ok else 0
    print(f'  c={conc:2d}  P50={p50:6.0f}ms  {len(ok)/wall:5.1f} req/s  ({len(ok)}/{N} ok)')
"
```

### Backend Comparison (Qwen3-0.6B, 2× MI355X, ionic 400Gb/s)

| Backend | P50 (c=1) | req/s (c=8) | Configuration |
|---------|----------:|------------:|---------------|
| **MoRI** | 73 ms | 81.7 | `--disaggregation-transfer-backend mori --disaggregation-ib-device <dev>` |
| **Mooncake** | 85 ms | 61.1 | `SGLANG_MOONCAKE_ROCM_STAGING=1 --disaggregation-transfer-backend mooncake` |
| **RIXL** | — | — | `--disaggregation-transfer-backend nixl` (C++ DRAM staging, automatic) |

---

## Troubleshooting

> For a comprehensive catalog of all ionic RDMA issues, cross-backend
> applicability, and lessons learned, see
> [Ionic RDMA Fix Guide](ionic-rdma-fixes.md).

### Container & Driver Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `libibverbs: Warning: Driver ionic does not support the kernel ABI` | Container/host libionic version mismatch | `docker cp $(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* \| head -1) CONTAINER:/usr/lib/x86_64-linux-gnu/libionic.so.1` — bind-mount (`-v`) is unreliable with symlinks |
| `stdbool.h not found` during build | bindgen can't find GCC headers | `export BINDGEN_EXTRA_CLANG_ARGS="-I$(find /usr/lib/gcc -name stdbool.h \| head -1 \| xargs dirname)"` |
| `RouterConfig.__new__() got unexpected keyword` | Stale Rust wheel in container | `maturin build --release --out /tmp/w && pip install /tmp/w/*.whl --force-reinstall` |
| `HIP_VISIBLE_DEVICES != CUDA_VISIBLE_DEVICES` | vLLM 0.18.1 validates env var consistency | Don't set `HIP_VISIBLE_DEVICES` when running vLLM tests |

### ionic RDMA & Subnet Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| **`RuntimeError: std::bad_cast`** with MoRI disagg TP>1 | **ionic ports missing IPv4 addresses** → `ibv_modify_qp EINVAL` → `exit(-1)` → misleading Python error | **Assign IPv4 to ALL 8 ionic ports** on every node (see [Ionic Network Configuration](#ionic-network-configuration-critical)). MoRI requires IPv4-mapped GIDs for RoCE v2 QP establishment. |
| `ibv_modify_qp INIT->RTR failed: Invalid argument` | ionic RDMA QP cannot connect — no matching IPv4 GID between local and remote devices | Assign matching IPv4 subnets per interface (same interface name = same subnet). Verify with `ibv_devinfo -v \| grep GID` — must show `::ffff:192.168.X.Y` entries |
| MoRI hangs during init | Ionic subnet mismatch between nodes | Match subnets per Ionic Subnet Matching section |
| `no active messages transport ... local IPv6 remote IPv4` | UCX transport mismatch on ionic | `UCX_TLS=rc_v,tcp` |
| **Mooncake `Failed to modify QP to RTR`** | Mooncake auto-pairs ionic devices by index (ionic_0↔ionic_0), but devices at the same index may be on **different subnets** across nodes | Specify `--mooncake-ib-device ionic_N` where `ionic_N` has matching GID subnet on both nodes. Verify with: `cat /sys/class/infiniband/ionic_N/ports/1/gids/1` — the 4th hex group must match |
| **Mooncake `received packet mismatch` / `Cannot make connection`** | Same as above — auto-discovery tries all 8 ionic devices and fails on mismatched ones | Use `--mooncake-ib-device` to restrict to matching devices only |
| **`hostname -I` returns ionic IP, not management IP** | ionic interfaces with IPv4 configured appear first in `hostname -I` output | Use `ip route get 1.1.1.1 \| awk '/src/ {print $7}'` for management IP instead of `hostname -I` |

### ionic RDMA Memory Registration Limits

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ibv_reg_mr ENOMEM` with Mooncake large model (e.g. DeepSeek-R1) | ionic `ibv_reg_mr` has **~199 MB max single MR** and **~250 MB total per device**. DeepSeek-R1 KV staging is ~1887 MB per TP worker. | Use chunked register→transfer→unregister (see [Mooncake RDMA Chunked MR](#mooncake-rdma-chunked-mr-for-large-models) below) |
| `ibv_reg_mr EINVAL` with `hipHostMalloc` memory | ionic rejects `ibv_reg_mr` on HIP-pinned memory entirely | Use `mmap+mlock` instead of `torch.pin_memory()` / `hipHostMalloc`. The `rocm_dram_staging_common.py` already does this. |
| `Transfer Engine does not support overlapped memory region` | Multiple `register_memory` calls overlap in address range | Register one contiguous region per layer buffer (min→max offset) instead of individual pages |

### EP/DP-Attn (DEP8) Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| DEP8 decode crashes with `NCCL error: unhandled cuda error` | `MORI_DISABLE_AUTO_XGMI=1` prevents MoRI from using XGMI for intra-node EP A2A communication | **Do NOT set `MORI_DISABLE_AUTO_XGMI=1`** for EP/DP configs. InferenceX does not set it. |
| **DEP8 decode OOM during CUDA graph capture** | Stale docker containers hold GPU VRAM. DEP8 with `dp-size=8` needs ~216 GB/GPU; residual VRAM from old containers causes OOM at graph capture. | **Always `docker rm -f $(docker ps -aq)` on ALL nodes before DEP8 benchmarks.** Verify with `amd-smi monitor --gpu all` — VRAM should be < 5 GB before launching. |
| **DEP8 decode `RuntimeError: unknown parameter type`** | Secondary crash after OOM corrupts DP-attention state. The `all_gather` in `scheduler_dp_attn_mixin.py:93` fails because some DP workers died from OOM. | Fix the OOM first (clean stale containers). If OOM is resolved, this error disappears. |
| DEP8 decode crashes at MoRI RDMA init | Missing InferenceX MoRI environment variables for EP mode | Set all required env vars from InferenceX `env.sh`: `MORI_SHMEM_MODE=ISOLATION`, `MORI_EP_LAUNCH_CONFIG_MODE=AUTO`, `MORI_IO_QP_MAX_SEND_WR=16384`, `MORI_IO_QP_MAX_CQE=32768`, `MORI_IO_QP_MAX_SGE=4` |
| EP/DP disagg KV transfer fails (`ibv_modify_qp` timeout) | Same ionic subnet mismatch as above, but via `--disaggregation-ib-device` | Specify only matching ionic devices in `--disaggregation-ib-device`. Auto-detect matching devices using GID subnet prefix (4th hex group in `/sys/class/infiniband/ionic_N/ports/1/gids/1`). |

### Dynamo Frontend Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Dynamo frontend `--router-mode kv` not used by NVIDIA in production | KV-aware routing is experimental. All InferenceX DSR1 benchmarks use `--router-mode round-robin`. | Use `--router-mode round-robin` for fair comparison. `--router-mode kv` with `--router-kv-events --router-track-active-blocks` gives unfair advantage in multi-decode configs. |
| Mooncake disagg with `sglang_router` returns 0 output tokens | `sglang_router --pd-disaggregation` does not pass Mooncake's `bootstrap_room_id` | Use **Dynamo frontend** (`python3 -m dynamo.frontend`) instead of `sglang_router` for Mooncake disagg. MoRI disagg works with both. |
| Dynamo frontend `port 8000 already in use` when co-located with worker | Both `dynamo.frontend --http-port 8000` and the `dynamo.sglang` worker try to bind the same port | Use different ports: `dynamo.frontend --http-port 9000` or let the worker auto-assign. |

### Other Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ibv_reg_mr ENOMEM` with Mooncake (small model) | Ionic can't register GPU VRAM for RDMA | `SGLANG_MOONCAKE_ROCM_STAGING=1` (auto-set in Docker) |
| `NIXL_ERR_BACKEND` with RIXL VRAM | Ionic can't GPUDirect RDMA | Use RIXL with C++ DRAM staging (automatic with patched RIXL) |
| aiter segfault on MI355X (gfx950) | aiter JIT kernels crash on gfx950 | `SGLANG_USE_AITER=0 --attention-backend triton` |
| Decode can't connect to prefill | SGLang binds to `127.0.0.1` by default | `--host 0.0.0.0` on both nodes |
| 11× slow TTFT on DeepSeek-V3 | aiter MLA persistent kernel conflict | `SGLANG_AITER_MLA_PERSIST=False` |
| `gpu_gb_to_total_fraction: failed to query GPU` | `gpu_utils.sh` only calls `nvidia-smi` | Updated with `amd-smi` fallback + GB/MiB unit conversion |
| `No module named 'nixl'` in vLLM container | `vllm-openai-rocm` doesn't include nixl | Use `Dockerfile.rocm-vllm` or install RIXL manually |
| `/v1/embeddings` returns 404 | Embedding endpoint registers ~15s after `/health` | Poll `/v1/embeddings` with retry, not just `/health` |
| First inference takes 5+ minutes | aiter JIT (~135s) + CUDA graph capture (~120s) on ROCm | Send warmup request with 15-min timeout; subsequent requests are fast |
| `PytestAssertRewriteWarning: anyio` crashes pytest | Container pre-imports `anyio` | `--override-ini=filterwarnings=default` |
| `dynamo.indexer is not available in this build` | Wheel not built with `--features kv-indexer` | Rebuild: `NIXL_PREFIX=/opt/rocm/rixl maturin build --features "kv-indexer,kv-indexer-runtime"` (now built into Docker) |
| `No module named 'aiconfigurator'` | NVIDIA-specific AIC perf model | Not applicable on AMD — expected skip |

---

## Mooncake RDMA Chunked MR for Large Models

ionic NICs have strict `ibv_reg_mr` limits:
- **Max single MR**: ~199 MB
- **Max total per device**: ~250 MB
- **8 devices total**: ~2000 MB (but `register_memory` registers on ALL devices simultaneously, so effective limit is still ~250 MB)

For large models like DeepSeek-R1 (KV staging ~1887 MB per TP worker), the patched `mooncake_rocm_staging.py` implements **chunked register→transfer→unregister**:

```
Prefill GPU KV ─hipMemcpy D2H─▶ DRAM staging
    ──── for each 190MB chunk: ────
    ibv_reg_mr(chunk) → RDMA WRITE → ibv_dereg_mr(chunk)
    ──── end loop ────
Decode DRAM staging ─hipMemcpy H2D─▶ Decode GPU KV
```

### Ionic MR Limit Probe Script

Run inside a container to verify ionic MR limits on your hardware:

```bash
docker exec <container> python3 -c '
from mooncake.engine import TransferEngine
import mmap, ctypes

e = TransferEngine()
e.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "ionic_0")
libc = ctypes.CDLL("libc.so.6")

for size_mb in [1, 10, 50, 100, 150, 199, 200, 250]:
    s = size_mb * 1024 * 1024
    m = mmap.mmap(-1, s)
    buf = (ctypes.c_char * s).from_buffer(m)
    ptr = ctypes.addressof(buf)
    libc.mlock(ctypes.c_void_p(ptr), ctypes.c_size_t(s))
    ret = e.register_memory(ptr, s)
    status = "OK" if ret == 0 else "FAIL"
    print(f"  {size_mb}MB: {status}")
    if ret == 0:
        e.unregister_memory(ptr)
'
```

### Ionic Subnet Matching Script

Before running cross-node disagg, verify ionic GID subnet matching:

```bash
# Run on each node to get ionic GID subnets:
for i in 0 1 2 3 4 5 6 7; do
    gid=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1 2>/dev/null)
    subnet=$(echo $gid | cut -d: -f4)
    echo "ionic_$i: subnet=$subnet gid=$gid"
done

# Matching rule: two nodes can communicate via ionic_X and ionic_Y
# if they have the SAME 4th hex group (subnet prefix).
# Example: ionic_4 on both nodes has subnet 0142 → use --disaggregation-ib-device ionic_4
```

---

## EP/DP-Attn (DEP8) Configuration Reference

InferenceX uses EP/DP-Attn for high-throughput disaggregated serving of MoE models. Key configuration:

### Required Environment Variables (from InferenceX `env.sh`)

```bash
# MoRI EP configuration
export MORI_SHMEM_MODE=ISOLATION
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=4
export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384
export MORI_MAX_DISPATCH_TOKENS_DECODE=160

# SGLang disagg timeouts
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

# DO NOT set MORI_DISABLE_AUTO_XGMI=1 — EP/DP needs XGMI for intra-node A2A
```

### DEP8 Decode Launch (1 node, 8 GPUs)

```bash
python3 -m sglang.launch_server --model-path /models/DeepSeek-R1-0528 \
    --tp-size 8 --ep-size 8 --dp-size 8 \
    --moe-a2a-backend mori --deepep-mode normal \
    --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head \
    --disaggregation-mode decode \
    --disaggregation-ib-device ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 \
    --disaggregation-transfer-backend mori \
    --kv-cache-dtype fp8_e4m3 --attention-backend aiter \
    --mem-fraction-static 0.85 --max-running-requests 4096 \
    --prefill-round-robin-balance --cuda-graph-max-bs 160
```

### Dynamo Frontend (NVIDIA does NOT use KV router in production)

```bash
# NVIDIA InferenceX uses round-robin for ALL disagg benchmarks:
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin

# NOT: --router-mode kv --router-kv-events --router-track-active-blocks
# KV-aware routing is experimental (only used for Qwen3-32B test configs)
```

---

## Results Summary

### Manual Feature Tests

| # | Feature | Status | Key Metric |
|:-:|---------|:------:|------------|
| 1 | Chat Completion | **PASS** | JSON response with generated text |
| 2 | Tool Calling | **PASS** | Model identifies `get_weather` function |
| 3 | Streaming (SSE) | **PASS** | 21 SSE chunks received |
| 4 | Multi-Turn Conversation | **PASS** | 5/5 turns, ~50ms each |
| 5 | Speculative Decoding | **PASS** | EAGLE / EAGLE3 / NGRAM / NEXTN supported |
| 6 | Request Migration | **PASS** | 301 chunks after worker kill |
| 7 | Multimodal (VL) | **PASS** | VL model correctly describes image |
| 8 | Pytest Suite | **PASS** | 217+ passed, 2 failed (non-code), ~50 skipped |
| 9 | MoRI RDMA (2-node) | **PASS** | 81.7 req/s at c=8, 100% success |
| 10 | Mooncake + DRAM staging (small model) | **PASS** | 61.1 req/s at c=8 (Qwen3-0.6B, TCP transport) |
| 11 | RIXL cross-node | **PASS** | C++ DRAM staging, sub-1s latency |
| 12 | **Mooncake RDMA (large model)** | **PASS** | DeepSeek-R1 FP8, chunked MR, 97.7 tok/s at c=1, TPOT 7.11ms |
| 13 | **MoRI Disagg 1P2D** | **PASS** | DeepSeek-R1 FP8, 8715 tok/s at c=256 (3 nodes, 24 GPUs) |
| 14 | **Dynamo Disagg 1P2D** | **PASS** | DeepSeek-R1 FP8, 8658 tok/s at c=256 (round-robin, same as InferenceX) |
| 15 | **EP/DP-Attn DEP8 Disagg** | **PASS** | DeepSeek-R1 FP8, 16011 tok/s at c=1024 (1334 tput/GPU) |

### Known Issues & Fixes

| Issue | Root Cause | Fix | Status |
|-------|-----------|-----|:------:|
| `HipDeviceProperties` struct offset | Missing `uuid`/`luid` fields before `total_global_mem` | Added 3 intermediate fields + alignment | **Fixed** |
| hipMemcpy return values unchecked | `_hip.hipMemcpy()` return value discarded | Added `_check_hip()`, raises `RuntimeError` | **Fixed** |
| `gpu_utils.sh` GB→MiB confusion | `"192 GB"` stripped to `192`, treated as MiB | Unit-aware conversion: GB `*1024`, MB pass-through | **Fixed** |
| `staging_tensors` list empty | Pinned memory freed during RDMA | `staging_tensors.append(tensor)` after allocation | **Fixed** |
| `copy_d2h` silently skips | Unregistered GPU buffers pass through | Returns `bool`; `batch_transfer_sync` raises on `False` | **Fixed** |
| Hardcoded `gfx942` in test | Fails on MI355X (gfx950) | Auto-detect from `rocminfo`, env var override | **Fixed** |
| `sglang` marker on version test | Test skipped unnecessarily | Marker removed | **Fixed** |
| kv-indexer not in Docker | 5 router tests failed | `Dockerfile.rocm-sglang` now builds with `--features kv-indexer` | **Fixed** |
| No vLLM + Dynamo image | vLLM container missing RIXL/Dynamo | Created `Dockerfile.rocm-vllm` | **Fixed** |
| **Mooncake RDMA fails on DeepSeek-R1** | ionic `ibv_reg_mr` limit (~250MB/device) vs ~1887MB KV staging | Chunked register→transfer→unregister in `mooncake_rocm_staging.py` (190MB chunks) | **Fixed** |
| **Mooncake + sglang_router = 0 output** | `sglang_router` doesn't pass `bootstrap_room_id` for Mooncake | Use Dynamo frontend instead of `sglang_router` for Mooncake disagg | **Fixed** |
| **Mooncake RDMA QP mismatch across nodes** | Auto-discovery pairs ionic by index, but subnets differ per-node | Specify `--mooncake-ib-device ionic_N` with matching subnet | **Fixed** |
| **EP/DP DEP8 crashes with `MORI_DISABLE_AUTO_XGMI=1`** | XGMI disabled prevents intra-node MoRI A2A for expert parallel | Remove `MORI_DISABLE_AUTO_XGMI=1` for EP/DP configs | **Fixed** |
| **EP/DP DEP8 missing MoRI env vars** | `MORI_SHMEM_MODE`, `MORI_EP_LAUNCH_CONFIG_MODE`, QP limits not set | Copy all env vars from InferenceX `env.sh` | **Fixed** |
| **`hostname -I` returns ionic IP** | ionic interface with IPv4 appears first | Use `ip route get 1.1.1.1 \| awk '/src/ {print $7}'` | **Fixed** |
| **Dynamo frontend port conflict** | Frontend and worker on same node both bind port 8000 | Use `--http-port 9000` for frontend when co-located with worker | **Fixed** |
| aiter JIT ~135s | First inference triggers HIP kernel JIT | Warmup request; allow 10+ min | **Expected** |
| CUDA graph capture ~120s | SGLang graph capture for batch sizes 1–512 | Expected on ROCm; just wait | **Expected** |
| **Mooncake RDMA throughput < MoRI** | Chunked MR overhead (~2.8s TTFT) + single ionic device | Known limitation; MoRI is recommended for MI355X production | **Known** |
| `aiconfigurator` missing | NVIDIA-specific AIC perf model | Skip on AMD | **N/A** |
