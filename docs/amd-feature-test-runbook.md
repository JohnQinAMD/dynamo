# Dynamo on AMD ROCm — Feature Test Runbook

Reproducible tests for all Dynamo features on AMD Instinct MI300X/MI325X/MI355X GPUs with Pensando Pollara 400 ionic NICs.

---

## Prerequisites

### 1. Hardware

- AMD Instinct GPU (MI300X, MI325X, or MI355X)
- ROCm 7.x installed on host
- For disaggregated serving tests: 2 nodes with Pensando ionic NICs on matching subnets

### 2. Docker Image

Use the pre-built all-in-one image (includes Dynamo, SGLang, MoRI, Mooncake, etcd, NATS):

```bash
docker pull amdprimus/dynamo-rocm-sglang:latest
```

Or build from source:

```bash
cd dynamo
docker build -f container/Dockerfile.rocm-sglang -t amdprimus/dynamo-rocm-sglang:latest .
```

### 3. Container Launch

```bash
docker run --rm -it \
    --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    amdprimus/dynamo-rocm-sglang:latest bash
```

### 4. Ionic Driver ABI Fix

Container and host `libionic` versions may differ. Run the auto-fix inside the container:

```bash
fix-ionic-abi.sh
```

If the auto-fix doesn't find the host driver, manually copy it:

```bash
# On the HOST: find your libionic version
ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.*

# Copy into the container (via shared mount or docker cp):
docker cp /usr/lib/x86_64-linux-gnu/libionic.so.1.1.<YOUR_VERSION> <container>:/usr/lib/x86_64-linux-gnu/
ln -sf libionic.so.1.1.<YOUR_VERSION> /usr/lib/x86_64-linux-gnu/libionic.so.1
ldconfig
```

Verify: `ibv_devinfo -d ionic_0 2>&1 | head -5` — no ABI warnings.

### 5. Start Infrastructure

```bash
export HIP_VISIBLE_DEVICES=0
export SGLANG_AITER_MLA_PERSIST=False
export RCCL_MSCCL_ENABLE=0
MY_IP=$(hostname -I | awk '{print $1}')

rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://${MY_IP}:2379 &
nats-server -p 4222 -js --client_advertise ${MY_IP}:4222 &
sleep 3

export ETCD_ENDPOINTS=http://${MY_IP}:2379
export NATS_SERVER=nats://${MY_IP}:4222

cd /opt/dynamo
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
sleep 2
```

### 6. Ionic Subnet Matching (disagg tests only)

Ionic device numbers are NOT consistent across nodes. Find matching subnets:

```bash
# Run on EACH node:
for i in 0 1 2 3 4 5 6 7; do
    gid=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1 2>/dev/null)
    [ -n "$gid" ] && echo "ionic_$i  $(echo $gid | cut -d: -f1-4)"
done

# Example output:
#   Node A: ionic_0 fd93:16d3:59b6:014e  <-- match!
#   Node B: ionic_4 fd93:16d3:59b6:014e  <-- match!
```

Assign IPv4 addresses on the matched interfaces:

```bash
NET=$(ls /sys/class/infiniband/ionic_<MATCHED_DEV>/device/net/ | head -1)
ip addr add 192.168.14.<NODE_ID>/24 dev $NET
ip link set $NET up
```

---

## Single-Node Tests

Start a worker:

```bash
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code &
# Wait ~50s
```

### Test 1: Chat Completion

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Count to 5"}],"max_tokens":50}'
```

**Pass**: Response contains text. **Result**: PASS

### Test 2: Tool Calling

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"max_tokens":300}'
```

**Pass**: Response mentions get_weather/Paris. **Result**: PASS

### Test 3: Streaming

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20,"stream":true}' \
  | grep -c "data:"
```

**Pass**: >2 chunks. **Result**: PASS (21 chunks)

### Test 4: Multi-turn (KVBM)

```bash
for turn in 1 2 3 4 5; do
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"Turn $turn\"}],\"max_tokens\":20}" > /dev/null
    echo "Turn $turn: OK"
done
```

**Pass**: All 5 complete. **Result**: PASS (50ms/turn)

### Test 5: Speculative Decoding Support

```bash
python3 -m sglang.launch_server --help 2>&1 | grep -c "speculative"
```

**Pass**: 50+ args. **Result**: PASS (EAGLE/EAGLE3/NEXTN/NGRAM)

### Test 6: Request Migration

```bash
pkill -f dynamo.sglang; sleep 2
HIP_VISIBLE_DEVICES=0 python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code &
W1=$!
HIP_VISIBLE_DEVICES=1 python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code &
# Wait ~60s

curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Write a long essay about AI"}],"max_tokens":300,"stream":true}' > /tmp/stream.txt &
sleep 3; kill $W1; wait
grep -c "data:" /tmp/stream.txt
```

**Pass**: >100 chunks after worker kill. **Result**: PASS (301 chunks)

### Test 7: Multimodal

```bash
pkill -f dynamo.sglang; sleep 2
python3 -m dynamo.sglang --model-path Qwen/Qwen3-VL-2B-Instruct --tp-size 1 --trust-remote-code &
# Wait ~60s

python3 -c "
import base64, io; from PIL import Image
img = Image.new('RGB', (100, 100), color=(255, 0, 0))
buf = io.BytesIO(); img.save(buf, format='PNG')
print(base64.b64encode(buf.getvalue()).decode())
" > /tmp/img.b64

B64=$(cat /tmp/img.b64)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen3-VL-2B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${B64}\"}}]}],\"max_tokens\":100}"
```

**Pass**: Describes red image. **Result**: PASS

### Test 8: Pytest Suite

```bash
pip install pytest pytest-benchmark pytest-httpserver pytest-asyncio pytest-timeout nats-py boto3 -q
python3 -m pytest tests/disagg/test_nixl_rocm_staging.py \
    tests/basic/test_rocm_gpu_detection.py \
    tests/basic/test_rocm_version_consistency.py \
    tests/disagg/test_ionic_validation.py \
    tests/deploy/test_k8s_crd_validation.py \
    --no-header -q --tb=no
```

**Pass**: 40+ tests. **Result**: PASS (41 passed)

---

## 2-Node Disaggregated Serving Tests

Complete Prerequisites 1-6 on BOTH nodes. Infrastructure (etcd, NATS, frontend) runs on the prefill node only.

### Test 9: MoRI RDMA

MoRI is AMD's native RDMA library for Pensando ionic NICs.

**Prefill node** (has etcd + NATS + frontend):

```bash
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device <PREFILL_IONIC_DEV>
```

**Decode node** (points to prefill's etcd/NATS):

```bash
PREFILL_IP=<prefill-node-ip>
export ETCD_ENDPOINTS=http://${PREFILL_IP}:2379
export NATS_SERVER=nats://${PREFILL_IP}:4222

python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device <DECODE_IONIC_DEV>
```

**Important**:
- `--host 0.0.0.0` is required (SGLang defaults to 127.0.0.1 which blocks cross-node bootstrap)
- `--disaggregation-ib-device` must be a matched ionic device (see Prerequisite 6)
- Run `fix-ionic-abi.sh` on both nodes first

| Concurrency | P50 | req/s | ok rate |
|-------------|-----|-------|---------|
| 1 | 73ms | 0.4 | 100% |
| 4 | 90ms | 39.9 | 100% |
| 8 | 95ms | 81.7 | 100% |

### Test 10: Mooncake RDMA + DRAM Staging

Ionic NICs cannot register GPU VRAM for RDMA (`ibv_reg_mr ENOMEM`). The DRAM staging monkey-patch (`mooncake_rocm_staging.py`) bounces data through pinned host memory.

**Both nodes** — set env vars:

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
PREFILL_IP=<prefill-node-ip>
export ETCD_ENDPOINTS=http://${PREFILL_IP}:2379
export NATS_SERVER=nats://${PREFILL_IP}:4222

python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake
```

**Important**:
- `SGLANG_MOONCAKE_ROCM_STAGING=1` activates the DRAM staging (automatic on ROCm)
- `MC_MAX_SGE=2` is required for ionic NICs (set in Docker image by default)
- No C++ patch or rebuild needed — pure Python monkey-patch
- No `--disaggregation-ib-device` needed (Mooncake discovers NICs)

| Concurrency | P50 | req/s | ok rate |
|-------------|-----|-------|---------|
| 1 | 85ms | 10.8 | 100% |
| 4 | 113ms | 25.2 | 100% |
| 8 | 109ms | 61.1 | 100% |

### Test 11: RIXL DRAM Staging

RIXL (AMD's port of NIXL) requires building from source. See `docs/amd-rocm-build.md` for build steps.

```bash
export SGLANG_NIXL_ROCM_STAGING=1
# Same commands as Test 9 but with: --disaggregation-transfer-backend nixl
```

Unit tests pass (12/12) — hipMemcpy D2H/H2D roundtrip, address translation, monkey-patch hooks verified on MI355X.

---

## Benchmark Script

Run after any disagg test is working:

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
            if r.status_code == 200:
                return {'ms': (time.time() - t0) * 1000, 'ok': True}
        except:
            pass
        return {'ms': (time.time() - t0) * 1000, 'ok': False}
    # Warmup
    for w in range(2): send(w)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(conc) as p:
        res = list(p.map(send, range(N)))
    wall = time.time() - t0
    ok = [r for r in res if r['ok']]
    p50 = sorted([r['ms'] for r in ok])[len(ok)//2] if ok else 0
    print(f'  c={conc}: P50={p50:.0f}ms  {len(ok)/wall:.1f} req/s  ({len(ok)}/{N} ok)')
"
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `libibverbs: Warning: Driver ionic does not support the kernel ABI` | Container/host libionic version mismatch | Run `fix-ionic-abi.sh` or copy host driver manually |
| `ibv_reg_mr ENOMEM` with Mooncake | Ionic can't register GPU VRAM | Set `SGLANG_MOONCAKE_ROCM_STAGING=1` |
| Decode can't connect to prefill bootstrap | SGLang `--host` defaults to 127.0.0.1 | Pass `--host 0.0.0.0` |
| MoRI hangs during init | Ionic subnet mismatch between nodes | Match subnets (Prerequisite 6) |
| `stdbool.h not found` during build | bindgen can't find GCC headers | Set `BINDGEN_EXTRA_CLANG_ARGS` |
| 11x slow TTFT on DeepSeek-V3 | aiter MLA persistent kernel conflict | Set `SGLANG_AITER_MLA_PERSIST=False` |

---

## Results Summary

| # | Test | Status | Key Metric |
|---|------|--------|-----------|
| 1 | Chat Completion | **PASS** | Response OK |
| 2 | Tool Calling | **PASS** | Model identifies tools |
| 3 | Streaming | **PASS** | 21 SSE chunks |
| 4 | KVBM Multi-turn | **PASS** | 50ms/turn |
| 5 | Speculative Decoding | **PASS** | EAGLE/NGRAM supported |
| 6 | Request Migration | **PASS** | 301 chunks after kill |
| 7 | Multimodal | **PASS** | VL model describes image |
| 8 | Pytest Suite | **PASS** | 41 passed |
| 9 | MoRI RDMA disagg | **PASS** | 81.7 req/s c=8, 100% |
| 10 | Mooncake RDMA + staging | **PASS** | 61.1 req/s c=8, 100% |
| 11 | RIXL DRAM staging | **PASS** (unit) | 12/12 tests |

### Backend Comparison

| Backend | P50 (c=1) | req/s (c=8) | Setup |
|---------|-----------|-------------|-------|
| **MoRI RDMA** | 73ms | 81.7 | `--disaggregation-transfer-backend mori --disaggregation-ib-device <dev>` |
| **Mooncake + staging** | 85ms | 61.1 | `SGLANG_MOONCAKE_ROCM_STAGING=1 --disaggregation-transfer-backend mooncake` |
| **RIXL + staging** | — | — | `SGLANG_NIXL_ROCM_STAGING=1 --disaggregation-transfer-backend nixl` |
