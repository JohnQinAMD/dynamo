# AMD MI355X Benchmark Scripts

InferenceX-aligned benchmark scripts for manual testing on AMD Instinct GPUs.
Produces result JSONs compatible with InferenceX's `process_result.py`.

## Pre-flight

```bash
# 1. Kill ALL stale containers (they hold GPU VRAM)
docker rm -f $(docker ps -aq)

# 2. Verify GPU VRAM is clean
amd-smi monitor --gpu all | awk 'NR>1{print $1, $NF}'
# All GPUs should show < 1 GB used

# 3. Verify ionic network (for disagg)
bash setup_network.sh --verify
bash setup_network.sh --match <REMOTE_NODE>
```

## Quick Start

```bash
# Single-node aggregated
export MODEL_DIR=/models MODEL_NAME=DeepSeek-R1-0528
export FRONTEND_TYPE=dynamo  # or "sglang" for sglang_router
export BENCH_MAX_CONCURRENCY="4 8 16 32 64"
bash server.sh

# Multi-node disaggregated (run on each node via Slurm)
export FRONTEND_TYPE=dynamo
export PREFILL_TP_SIZE=8 DECODE_TP_SIZE=8
export xP=1 yD=2  # 1 prefill + 2 decode workers
bash server.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `env.sh` | Environment setup (MoRI, RCCL, ionic, VRAM check) |
| `server.sh` | Unified launcher (Dynamo frontend or sglang_router) |
| `bench.sh` | Benchmark runner with InferenceX parameters |
| `setup_network.sh` | Ionic IPv4 assignment + subnet matching |
| `models.yaml` | Model-specific configs (from InferenceX amd_utils) |

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRONTEND_TYPE` | `sglang` | `dynamo` = Dynamo frontend, `sglang` = sglang_router |
| `MODEL_DIR` | `/models` | Base model directory |
| `MODEL_NAME` | - | Model subdirectory name |
| `BENCH_INPUT_LEN` | `1024` | Input sequence length |
| `BENCH_OUTPUT_LEN` | `1024` | Output sequence length |
| `BENCH_MAX_CONCURRENCY` | `"4 8 16 32 64 128 256"` | Space-separated concurrency levels |
| `RESULTS_DIR` | `infx_bench_results` | Where result JSONs are saved |

## InferenceX Integration

These scripts produce the same JSON format as InferenceX's `benchmark_serving.py --save-result`.
To upload results to the InferenceX Neon DB:

```bash
# Process results into InferenceX format
python3 InferenceX/utils/process_result.py results/

# Compare with DB baselines
DATABASE_URL=<neon_url> python3 InferenceX/utils/compare_results.py results/
```

## Benchmark Results

| Config | Frontend | Best Throughput | TPOT P50 | Notes |
|--------|----------|----------------|----------|-------|
| TP8 1P2D | sglang | 8,715 tok/s (c=256) | 25.7ms | MoRI KV transfer |
| TP8 1P2D | dynamo | 8,658 tok/s (c=256) | 25.9ms | +6% at c=4-64 |
| DEP8 1P2D | sglang | 10,555 tok/s (c=512) | 45.5ms | EP/DP-Attn |
| DEP8 1P2D | dynamo | 14,216 tok/s (c=512) | 32.8ms | **+35% vs sglang** |
| Mooncake RDMA | dynamo | 3,088 tok/s (c=64) | 18.9ms | Slab optimization |
