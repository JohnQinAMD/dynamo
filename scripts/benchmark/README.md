# AMD MI355X Benchmark Scripts

InferenceX-aligned benchmark scripts for Dynamo on AMD Instinct GPUs.
Produces result JSONs compatible with InferenceX's `process_result.py`.

## Directory Layout

```
scripts/benchmark/
├── env.sh              # Environment (MoRI, RCCL, ionic ABI check, VRAM preflight)
├── server.sh           # Unified launcher (Dynamo frontend or sglang_router)
├── bench.sh            # Benchmark runner with InferenceX parameters
├── setup_network.sh    # Ionic: IPv4 assignment + subnet match + ABI fix
├── models.yaml         # Per-model configs (parallelism, memory, CUDA graph)
└── README.md           # This file
```

## Two Ways to Run

| | **Path A: Manual (SSH)** | **Path B: InferenceX CI (Slurm)** |
|---|---|---|
| Use case | Developer quick-test, debugging | Automated CI/CD baselines |
| Container mgmt | Manual `docker run` + `docker cp` | Slurm `srun` + `job.slurm` auto |
| Ionic fix | Manual `setup_network.sh --fix-abi` | Automatic in `env.sh` from `/host_libs` |
| Node orchestration | SSH into each node | Slurm schedules all nodes |
| Scripts | `dynamo/scripts/benchmark/` | `InferenceX/benchmarks/multi_node/amd_utils/` |

---

## Path A: Manual Testing (Step by Step)

### Step 0: Pre-flight (run on EVERY node)

```bash
# Kill stale containers — they hold GPU VRAM even when idle
docker rm -f $(docker ps -aq)

# Verify VRAM is clean (all GPUs < 1 GB)
amd-smi monitor --gpu all | awk 'NR>1{print $1, $NF}'
```

### Step 1: Start containers on each node

```bash
# Run on EACH node (e.g., via SSH)
docker run -d --name bench --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v /path/to/workspace:/workspace \
    -v /path/to/models:/models \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    amdprimus/dynamo-rocm-sglang:latest tail -f /dev/null
```

### Step 2: Fix ionic ABI (run from HOST on each node)

```bash
# Option A: helper script
bash /path/to/dynamo/scripts/benchmark/setup_network.sh --fix-abi bench

# Option B: manual
docker cp $(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* | head -1) \
    bench:/usr/lib/x86_64-linux-gnu/libionic.so.1

# Verify (must show 8)
docker exec bench ibv_devinfo 2>&1 | grep hca_id | wc -l
```

> **Why not bind-mount?** Docker `-v host.so:/container/libionic.so.1:ro` does not
> correctly override symlinks inside the container. `docker cp` is the only reliable method.

### Step 3: Configure ionic IPv4 (inside container, each node)

```bash
docker exec bench bash -c '
    cd /workspace/dynamo/scripts/benchmark
    bash setup_network.sh              # auto-assign IPv4 to ionic ports
    bash setup_network.sh --verify     # show devices + IPs
'
```

### Step 4: Verify cross-node connectivity

```bash
# From prefill node, check subnet match with decode node:
docker exec bench bash -c '
    bash /workspace/dynamo/scripts/benchmark/setup_network.sh --match <DECODE_NODE_HOSTNAME>
'
# Output shows which ionic devices have matching subnets
```

### Step 5: Launch servers

**On the head node (NODE_RANK=0, runs prefill + frontend + benchmarks):**

```bash
docker exec -it bench bash
cd /workspace/dynamo/scripts/benchmark

export MODEL_DIR=/models
export MODEL_NAME=DeepSeek-R1-0528
export FRONTEND_TYPE=dynamo           # "dynamo" or "sglang"
export IPADDRS="10.0.0.1,10.0.0.2"   # head_ip,decode_ip (comma-separated)
export xP=1 yD=1                      # 1 prefill + 1 decode
export NODE_RANK=0
export PREFILL_TP_SIZE=8 DECODE_TP_SIZE=8
export BENCH_MAX_CONCURRENCY="4 8 16 32 64 128 256"

bash server.sh   # launches prefill + frontend, waits for decode, runs benchmarks
```

**On each decode node (NODE_RANK=1, 2, ...):**

```bash
docker exec -it bench bash
cd /workspace/dynamo/scripts/benchmark

export MODEL_DIR=/models
export MODEL_NAME=DeepSeek-R1-0528
export FRONTEND_TYPE=dynamo
export IPADDRS="10.0.0.1,10.0.0.2"
export xP=1 yD=1
export NODE_RANK=1                     # 1 for first decode, 2 for second, etc.
export PREFILL_TP_SIZE=8 DECODE_TP_SIZE=8

bash server.sh   # launches decode worker, blocks until head finishes
```

### Step 6: Collect results

Results are saved to `$RESULTS_DIR` (default: `infx_bench_results/`) as JSON files:

```
infx_bench_results/
├── dynamo_1p1d_c4.json
├── dynamo_1p1d_c8.json
├── dynamo_1p1d_c16.json
└── ...
```

---

## Path B: InferenceX CI (Slurm)

Scripts: `InferenceX/benchmarks/multi_node/amd_utils/`

### How It Works

1. `submit.sh` → submits a Slurm job via `job.slurm`
2. `job.slurm` → `srun` launches Docker on each node with:
   - `/host_libs` bind-mount (for ionic ABI fix)
   - All env vars passed via `-e`
3. `server.sh` → sources `env.sh` (auto-fixes ionic ABI + IPv4) → launches workers
4. `bench.sh` → runs benchmarks → saves results

### Submit a Job

```bash
cd InferenceX/benchmarks/multi_node/amd_utils

# Required environment
export SLURM_ACCOUNT=myaccount
export SLURM_PARTITION=compute
export TIME_LIMIT=08:00:00
export MODEL_PATH=/nfsdata
export MODEL_NAME=deepseek-ai/DeepSeek-R1-0528
export CONTAINER_IMAGE=amdprimus/dynamo-rocm-sglang:latest
export RUNNER_NAME=dsr1-fp8-mi355x

# Optional: use Dynamo frontend (default: sglang_router)
export FRONTEND_TYPE=dynamo

# Submit: prefill_nodes prefill_workers decode_nodes decode_workers ISL OSL concurrencies rate
bash submit.sh  1 1  2 2  1024 1024  "4x8x16x32x64x128x256"  inf
```

### What Slurm Does Automatically

- Allocates 3 nodes (1 prefill + 2 decode)
- Starts Docker containers with all required mounts
- Fixes ionic ABI from `/host_libs` (no manual `docker cp` needed)
- Assigns ionic IPv4 addresses
- Starts etcd + NATS (for Dynamo frontend)
- Launches prefill/decode workers
- Runs benchmarks at all concurrency levels
- Saves results to `benchmark_logs/`

### CI Config (amd-master.yaml)

Framework configs in `.github/configs/amd-master.yaml` control which tests run:

```yaml
dsr1-fp8-mi355x-dynamo-sglang-disagg:
  image: amdprimus/dynamo-rocm-sglang:latest
  model: deepseek-ai/DeepSeek-R1-0528
  framework: dynamo-sglang
  additional-settings:
    - "FRONTEND_TYPE=dynamo"
```

---

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRONTEND_TYPE` | `sglang` | `dynamo` = Dynamo frontend + etcd/nats, `sglang` = sglang_router |
| `MODEL_DIR` | `/models` | Base model directory |
| `MODEL_NAME` | (required) | Model subdirectory (e.g., `DeepSeek-R1-0528`) |
| `IPADDRS` | (required) | Comma-separated node IPs (head first) |
| `NODE_RANK` | `0` | 0 = head/prefill, 1+ = decode |
| `xP` | `1` | Number of prefill workers |
| `yD` | `1` | Number of decode workers |
| `PREFILL_TP_SIZE` | `8` | Tensor parallelism for prefill |
| `DECODE_TP_SIZE` | `8` | Tensor parallelism for decode |
| `BENCH_INPUT_LEN` | `1024` | Input sequence length |
| `BENCH_OUTPUT_LEN` | `1024` | Output sequence length |
| `BENCH_MAX_CONCURRENCY` | `"4 8 16 32 64 128 256"` | Space-separated concurrency levels |
| `RESULTS_DIR` | `infx_bench_results` | Where result JSONs are saved |
| `IBDEVICES` | auto-detect | RDMA devices (e.g., `ionic_0,...,ionic_7`) |
| `MORI_RDMA_TC` | auto-detect | RDMA traffic class for QoS |

## Ionic RDMA Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ibv_devinfo` shows 0 devices | libionic ABI mismatch | `docker cp` host libionic (NOT bind-mount) |
| `ibv_modify_qp: Invalid argument` | ionic ports lack IPv4 | `setup_network.sh` to assign IPs |
| `Connection timed out` on RDMA | ionic subnets don't match | `setup_network.sh --match REMOTE` to check |
| `availDevices.size() > 0` assert | Same as ABI mismatch | Fix libionic first |
| OOM during CUDA graph capture | Stale containers hold VRAM | `docker rm -f $(docker ps -aq)` on all nodes |
| Very slow first request (~5 min) | aiter JIT + CUDA graph warmup | Normal; send warmup requests first |

## Benchmark Results (DeepSeek-R1 FP8, ISL/OSL=1024/1024)

| Config | Frontend | Best Throughput | TPOT P50 | Notes |
|--------|----------|----------------|----------|-------|
| TP8 1P2D | sglang | 8,715 tok/s (c=256) | 25.7ms | MoRI KV transfer |
| TP8 1P2D | dynamo | 8,658 tok/s (c=256) | 25.9ms | +6% at c=4-64 |
| DEP8 1P2D | sglang | 10,555 tok/s (c=512) | 45.5ms | EP/DP-Attn |
| DEP8 1P2D | dynamo | 14,216 tok/s (c=512) | 32.8ms | **+35% vs sglang** |
| Mooncake RDMA | dynamo | 3,088 tok/s (c=64) | 18.9ms | Chunked MR slab |
