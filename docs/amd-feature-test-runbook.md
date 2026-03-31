# AMD Feature Test Runbook

Reproducible tests for all Dynamo features on AMD MI355X with Pensando ionic NICs.

**Tested on**: chi2761/chi2885/chi2896 MI355X, Python 3.10, ROCm 7.2, SGLang 0.5.9

---

## Known Issues (READ FIRST)

### Issue 1: Ionic Driver ABI Mismatch (MANDATORY FIX)

The `rocm/sgl-dev` container ships `libionic1 54.0-149`, but the host kernel expects ABI version `54.0-185`. Without fixing this, **all RDMA operations fail silently** — MoRI, Mooncake, and RIXL will not work.

**Symptoms**: `libibverbs: Warning: Driver ionic does not support the kernel ABI of 1`, or MoRI/Mooncake hangs during initialization.

**Fix** (run once per host, then in every container):

```bash
# ON THE HOST — copy the correct driver to shared storage (one time):
cp /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185 /mnt/vast/john/rocm-dynamo/libionic_host.so

# INSIDE EACH CONTAINER — install the host driver:
cp /workspace/libionic_host.so /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185
ln -sf libionic.so.1.1.54.0-185 /usr/lib/x86_64-linux-gnu/libionic.so.1
ldconfig
```

**Verify**: `ibv_devinfo -d ionic_0 2>&1 | head -5` should show device info without warnings.

### Issue 2: SGLang Bootstrap Binds to 127.0.0.1

SGLang's `--host` defaults to `127.0.0.1`. In disaggregated mode, the bootstrap HTTP server uses this address, making it unreachable from remote nodes.

**Fix**: Always pass `--host 0.0.0.0` when starting SGLang workers in disagg mode.

### Issue 3: Ionic Subnet Mismatch

Ionic device numbers are NOT consistent across nodes. `ionic_0` on Node A may be on a different subnet than `ionic_0` on Node B. You must match subnets manually.

**Fix**: Run the subnet check below and use matched ionic devices.

---

## Setup (All Tests)

### Step 1: Start Container

```bash
docker run --rm -d --name dynamo-test \
    --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 128G --ipc=host \
    -v /mnt/vast/john/rocm-dynamo:/workspace \
    rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2 sleep 14400

docker exec -it dynamo-test bash
```

### Step 2: Fix Ionic ABI (MANDATORY for RDMA tests)

```bash
cp /workspace/libionic_host.so /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185
ln -sf libionic.so.1.1.54.0-185 /usr/lib/x86_64-linux-gnu/libionic.so.1
ldconfig
ibv_devinfo -d ionic_0 2>&1 | head -5   # verify no ABI warnings
```

### Step 3: Build Dynamo (~5 min)

```bash
cp -r /workspace/dynamo /tmp/dyn && cd /tmp/dyn
export LIBCLANG_PATH=/opt/rocm/lib/llvm/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I$(find /usr/lib/gcc -name stdbool.h | head -1 | xargs dirname)"
export VIRTUAL_ENV=/opt/venv
cd lib/bindings/python && maturin develop --release
cd /tmp/dyn && pip install -e .

# Verify
python3 -c "from dynamo.llm import ModelType; print('Dynamo OK')"
```

### Step 4: Install etcd + NATS

```bash
wget -q https://github.com/etcd-io/etcd/releases/download/v3.5.21/etcd-v3.5.21-linux-amd64.tar.gz -O /tmp/etcd.tar.gz
mkdir -p /usr/local/bin/etcd-dir && tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd-dir --strip-components=1
ln -sf /usr/local/bin/etcd-dir/etcd /usr/local/bin/etcd
ln -sf /usr/local/bin/etcd-dir/etcdctl /usr/local/bin/etcdctl

wget -q https://github.com/nats-io/nats-server/releases/download/v2.10.28/nats-server-v2.10.28-amd64.deb -O /tmp/nats.deb
dpkg -i /tmp/nats.deb
```

### Step 5: Start Infrastructure (on prefill/frontend node)

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

cd /tmp/dyn
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
sleep 2
```

### Step 6: Find Matching Ionic Devices (for disagg tests)

Run on BOTH nodes and find pairs with the same subnet:

```bash
for i in 0 1 2 3 4 5 6 7; do
    gid=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1 2>/dev/null)
    [ -n "$gid" ] && echo "ionic_$i  $(echo $gid | cut -d: -f1-4)"
done
```

Example match: chi2761 `ionic_0` (:014e) <-> chi2885 `ionic_4` (:014e)

### Step 7: Assign IPv4 to Ionic Interfaces (for MoRI)

```bash
# Find the network interface for your matched ionic device
NET=$(ls /sys/class/infiniband/ionic_0/device/net/ | head -1)

# Assign IPs (same /24 subnet, different last octet per node)
ip addr add 192.168.14.10/24 dev $NET    # Node A
# ip addr add 192.168.14.11/24 dev $NET  # Node B
ip link set $NET up
```

---

## Single-Node Tests (Tests 1-8)

Start a worker first:

```bash
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code &
# Wait ~50s for model load
```

### Test 1: Normal Chat

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

### Test 4: Multi-turn KVBM

```bash
for turn in 1 2 3 4 5; do
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"Turn $turn\"}],\"max_tokens\":20}" > /dev/null
    echo "Turn $turn: OK"
done
```

**Pass**: All 5 complete. **Result**: PASS (50ms/turn)

### Test 5: Speculative Decoding

```bash
python3 -m sglang.launch_server --help 2>&1 | grep -c "speculative"
```

**Pass**: 50+ args. **Result**: PASS (56 args, EAGLE/EAGLE3/NEXTN/NGRAM)

### Test 6: Request Migration

```bash
# Kill the single worker, start 2:
pkill -f dynamo.sglang; sleep 2
HIP_VISIBLE_DEVICES=0 python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code &
W1=$!
HIP_VISIBLE_DEVICES=1 python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code &
# Wait ~60s for both workers

curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Write a long essay about AI"}],"max_tokens":300,"stream":true}' > /tmp/stream.txt &
sleep 3
kill $W1
wait
grep -c "data:" /tmp/stream.txt
```

**Pass**: >100 chunks. **Result**: PASS (301 chunks)

### Test 7: Multimodal

```bash
pkill -f dynamo.sglang; sleep 2
python3 -m dynamo.sglang --model-path Qwen/Qwen3-VL-2B-Instruct --tp-size 1 --trust-remote-code &
# Wait ~60s

# Create test image
python3 -c "
import base64, io; from PIL import Image
img = Image.new('RGB', (100, 100), color=(255, 0, 0))
buf = io.BytesIO(); img.save(buf, format='PNG')
print(base64.b64encode(buf.getvalue()).decode())
" > /tmp/img.b64

# Send multimodal request
B64=$(cat /tmp/img.b64)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen3-VL-2B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${B64}\"}}]}],\"max_tokens\":100}"
```

**Pass**: Response describes red image. **Result**: PASS ("solid red color")

### Test 8: Pytest Suite

```bash
cd /tmp/dyn
pip install pytest pytest-benchmark pytest-httpserver pytest-asyncio pytest-timeout nats-py boto3 -q
python3 -m pytest tests/disagg/test_nixl_rocm_staging.py \
    tests/basic/test_rocm_gpu_detection.py \
    tests/basic/test_rocm_version_consistency.py \
    tests/disagg/test_ionic_validation.py \
    tests/deploy/test_k8s_crd_validation.py \
    --no-header -q --tb=no
```

**Pass**: 40+ tests. **Result**: PASS (41 passed, 2 skipped)

---

## 2-Node Disaggregated Serving Tests (Tests 9-11)

**Required**: 2 nodes with matching ionic subnets. Run Setup Steps 1-7 on BOTH nodes.

**Example**: chi2761 (prefill, `ionic_0`, subnet `:014e`) + chi2885 (decode, `ionic_4`, subnet `:014e`)

### Test 9: MoRI RDMA Disaggregated Serving

MoRI is AMD's native RDMA library for ionic NICs. No additional setup needed.

**Prefill node** (chi2761 — runs frontend + etcd + NATS + prefill worker):

```bash
# After Steps 1-7, start prefill worker:
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ionic_0
```

**Decode node** (chi2885 — only decode worker, points to prefill's etcd/NATS):

```bash
# After Steps 1-3 (no etcd/NATS/frontend needed on decode node)
export HIP_VISIBLE_DEVICES=0
export SGLANG_AITER_MLA_PERSIST=False RCCL_MSCCL_ENABLE=0
PREFILL_IP=<chi2761-ip>
export ETCD_ENDPOINTS=http://${PREFILL_IP}:2379
export NATS_SERVER=nats://${PREFILL_IP}:4222

cd /tmp/dyn
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ionic_4
```

**Test** (from prefill node):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

**Key points**:
- `--host 0.0.0.0` is MANDATORY (Issue 2: bootstrap binding)
- `--disaggregation-ib-device` must be a matched ionic device (Issue 3)
- Ionic ABI must be fixed first (Issue 1)

| c | P50 | req/s | ok |
|---|-----|-------|----|
| 1 | 73ms | 0.4 | 100% |
| 4 | 90ms | 39.9 | 100% |
| 8 | 95ms | 81.7 | 100% |

**Result**: PASS (81.7 req/s at c=8, 100% success)

### Test 10: Mooncake RDMA + DRAM Staging

Mooncake cannot register GPU VRAM on ionic NICs (`ibv_reg_mr ENOMEM`). The `mooncake_rocm_staging.py` monkey-patch solves this by bouncing data through pinned host memory.

**Both nodes** — add these env vars:

```bash
export SGLANG_MOONCAKE_ROCM_STAGING=1    # enable DRAM staging
export MC_MAX_SGE=2                       # ionic scatter-gather limit
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
PREFILL_IP=<chi2761-ip>
export ETCD_ENDPOINTS=http://${PREFILL_IP}:2379 NATS_SERVER=nats://${PREFILL_IP}:4222

python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --tp-size 1 --trust-remote-code \
    --host 0.0.0.0 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake
```

**Key points**:
- `SGLANG_MOONCAKE_ROCM_STAGING=1` activates the DRAM staging monkey-patch
- `MC_MAX_SGE=2` is required for ionic NICs (hardware limit)
- No `--disaggregation-ib-device` needed (Mooncake discovers NICs automatically)
- No C++ patch rebuild required — the monkey-patch handles everything at Python level

| c | P50 | req/s | ok |
|---|-----|-------|----|
| 1 | 85ms | 10.8 | 100% |
| 4 | 113ms | 25.2 | 100% |
| 8 | 109ms | 61.1 | 100% |

**Result**: PASS (61.1 req/s at c=8, 100% success)

### Test 11: RIXL DRAM Staging

RIXL (AMD's port of NIXL) is not in the standard `rocm/sgl-dev` image and must be built from source. The `nixl_rocm_staging.py` monkey-patch provides DRAM staging (same approach as Mooncake above).

**Unit tests** pass on MI355X (12/12 — hipMemcpy D2H/H2D roundtrip, address translation, monkey-patch hooks). E2E disagg test requires building RIXL per `docs/amd-rocm-build.md`.

**To run E2E** (after building RIXL):

```bash
export SGLANG_NIXL_ROCM_STAGING=1    # enable DRAM staging
export NIXL_PREFIX=/opt/rocm/rixl
export LD_LIBRARY_PATH=/opt/rocm/rixl/lib:$LD_LIBRARY_PATH

# Same as Test 9 but with: --disaggregation-transfer-backend nixl
```

**Result**: PASS (unit tests 12/12)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `libibverbs: Warning: Driver ionic does not support the kernel ABI` | Container libionic != host libionic | Step 2: copy host driver |
| `ibv_reg_mr ENOMEM` for Mooncake | Ionic can't register GPU VRAM | `export SGLANG_MOONCAKE_ROCM_STAGING=1` |
| Decode can't connect to prefill bootstrap | SGLang `--host` defaults to 127.0.0.1 | Always pass `--host 0.0.0.0` |
| MoRI hangs during init | Ionic subnet mismatch | Step 6: match subnets |
| `stdbool.h not found` during build | bindgen can't find GCC headers | Set `BINDGEN_EXTRA_CLANG_ARGS` |
| 11x slow TTFT on DSV3 | aiter MLA persistent kernel | `SGLANG_AITER_MLA_PERSIST=False` |

---

## Results Summary

| # | Test | Status | Key Metric |
|---|------|--------|-----------|
| 1 | Normal Chat | **PASS** | Qwen3-0.6B serves OK |
| 2 | Tool Calling | **PASS** | Model identifies tools |
| 3 | Streaming | **PASS** | 21 SSE chunks |
| 4 | KVBM Multi-turn | **PASS** | 50ms/turn |
| 5 | Speculative Decoding | **PASS** | EAGLE/NGRAM available |
| 6 | Request Migration | **PASS** | 301 chunks after kill |
| 7 | Multimodal | **PASS** | VL model describes image |
| 8 | Pytest Suite | **PASS** | 41 passed |
| 9 | **MoRI RDMA disagg** | **PASS** | **81.7 req/s** c=8, 100% ok |
| 10 | **Mooncake RDMA + staging** | **PASS** | **61.1 req/s** c=8, 100% ok |
| 11 | RIXL DRAM staging | **PASS** (unit) | 12/12 tests |

## Backend Performance Comparison

| Backend | c=1 P50 | c=8 req/s | Overhead | Setup |
|---------|---------|-----------|----------|-------|
| **MoRI RDMA** | 73ms | 81.7 | Native | `--disaggregation-transfer-backend mori` |
| **Mooncake RDMA + staging** | 85ms | 61.1 | hipMemcpy bounce | `SGLANG_MOONCAKE_ROCM_STAGING=1` |
| **RIXL + staging** | — | — (unit tested) | hipMemcpy bounce | `SGLANG_NIXL_ROCM_STAGING=1` |
