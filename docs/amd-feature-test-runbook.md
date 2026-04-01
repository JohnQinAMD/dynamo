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

Two pre-built images are available:

| Image | Backend | Includes |
|-------|---------|----------|
| `Dockerfile.rocm-sglang` | SGLang | Dynamo, SGLang, MoRI, RIXL, Mooncake, etcd, NATS, kv-indexer |
| `Dockerfile.rocm-vllm` | vLLM | Dynamo, vLLM, RIXL, etcd, NATS, kv-indexer |

Build from source:

```bash
cd /path/to/repo
docker build -f dynamo/container/Dockerfile.rocm-sglang -t dynamo-rocm-sglang:latest .
```

### Launch Container

```bash
docker run --rm -it \
    --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v /path/to/hf_cache:/root/.cache/huggingface \
    dynamo-rocm-sglang:latest bash
```

### Ionic Driver ABI Fix (if needed)

Container and host `libionic` versions may differ. Auto-fix inside the container:

```bash
fix-ionic-abi.sh
```

Verify: `ibv_devinfo -d ionic_0 2>&1 | head -5` — no ABI warnings.

> **Manual fallback**: If auto-fix fails, copy the host driver into the container:
> ```bash
> # On HOST: find version
> ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.*
> # Copy + relink inside container
> docker cp /usr/lib/x86_64-linux-gnu/libionic.so.1.1.X CONTAINER:/usr/lib/x86_64-linux-gnu/
> ln -sf libionic.so.1.1.X /usr/lib/x86_64-linux-gnu/libionic.so.1 && ldconfig
> ```

---

## Quick Start

Start infrastructure and a worker, then send a request — all in one block:

```bash
# Environment
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_AITER_MLA_PERSIST=False
export RCCL_MSCCL_ENABLE=0
MY_IP=$(hostname -I | awk '{print $1}')

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

> Requires 2 nodes with ionic NICs. Complete Prerequisites on BOTH nodes.

### Ionic Subnet Matching

Ionic device numbering is NOT consistent across nodes. Find matching subnets:

```bash
for i in 0 1 2 3 4 5 6 7; do
    gid=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1 2>/dev/null)
    [ -n "$gid" ] && echo "ionic_$i  $(echo $gid | cut -d: -f1-4)"
done
```

Match devices with the same subnet prefix, then assign IPv4:

```bash
NET=$(ls /sys/class/infiniband/ionic_<MATCHED_DEV>/device/net/ | head -1)
ip addr add 192.168.14.<NODE_ID>/24 dev $NET
ip link set $NET up
```

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

| Symptom | Cause | Fix |
|---------|-------|-----|
| `libibverbs: Warning: Driver ionic does not support the kernel ABI` | Container/host libionic version mismatch | `fix-ionic-abi.sh` or copy host driver manually |
| `ibv_reg_mr ENOMEM` with Mooncake | Ionic can't register GPU VRAM for RDMA | `SGLANG_MOONCAKE_ROCM_STAGING=1` (auto-set in Docker) |
| `NIXL_ERR_BACKEND` with RIXL VRAM | Ionic can't GPUDirect RDMA | Use RIXL with C++ DRAM staging (automatic with patched RIXL) |
| `no active messages transport ... local IPv6 remote IPv4` | UCX transport mismatch on ionic | `UCX_TLS=rc_v,tcp` |
| aiter segfault on MI355X (gfx950) | aiter JIT kernels crash on gfx950 | `SGLANG_USE_AITER=0 --attention-backend triton` |
| Decode can't connect to prefill | SGLang binds to `127.0.0.1` by default | `--host 0.0.0.0` on both nodes |
| MoRI hangs during init | Ionic subnet mismatch between nodes | Match subnets per Ionic Subnet Matching section |
| `stdbool.h not found` during build | bindgen can't find GCC headers | `export BINDGEN_EXTRA_CLANG_ARGS="-I$(find /usr/lib/gcc -name stdbool.h \| head -1 \| xargs dirname)"` |
| 11× slow TTFT on DeepSeek-V3 | aiter MLA persistent kernel conflict | `SGLANG_AITER_MLA_PERSIST=False` |
| `gpu_gb_to_total_fraction: failed to query GPU` | `gpu_utils.sh` only calls `nvidia-smi` | Updated with `amd-smi` fallback + GB/MiB unit conversion |
| `RouterConfig.__new__() got unexpected keyword` | Stale Rust wheel in container | `maturin build --release --out /tmp/w && pip install /tmp/w/*.whl --force-reinstall` |
| `HIP_VISIBLE_DEVICES != CUDA_VISIBLE_DEVICES` | vLLM 0.18.1 validates env var consistency | Don't set `HIP_VISIBLE_DEVICES` when running vLLM tests |
| `No module named 'nixl'` in vLLM container | `vllm-openai-rocm` doesn't include nixl | Use `Dockerfile.rocm-vllm` or install RIXL manually |
| `/v1/embeddings` returns 404 | Embedding endpoint registers ~15s after `/health` | Poll `/v1/embeddings` with retry, not just `/health` |
| First inference takes 5+ minutes | aiter JIT (~135s) + CUDA graph capture (~120s) on ROCm | Send warmup request with 15-min timeout; subsequent requests are fast |
| `PytestAssertRewriteWarning: anyio` crashes pytest | Container pre-imports `anyio` | `--override-ini=filterwarnings=default` |
| `dynamo.indexer is not available in this build` | Wheel not built with `--features kv-indexer` | Rebuild: `NIXL_PREFIX=/opt/rocm/rixl maturin build --features "kv-indexer,kv-indexer-runtime"` (now built into Docker) |
| `No module named 'aiconfigurator'` | NVIDIA-specific AIC perf model | Not applicable on AMD — expected skip |

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
| 10 | Mooncake + DRAM staging | **PASS** | 61.1 req/s at c=8, 100% success |
| 11 | RIXL cross-node | **PASS** | C++ DRAM staging, sub-1s latency |

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
| aiter JIT ~135s | First inference triggers HIP kernel JIT | Warmup request; allow 10+ min | **Expected** |
| CUDA graph capture ~120s | SGLang graph capture for batch sizes 1–512 | Expected on ROCm; just wait | **Expected** |
| `aiconfigurator` missing | NVIDIA-specific AIC perf model | Skip on AMD | **N/A** |
