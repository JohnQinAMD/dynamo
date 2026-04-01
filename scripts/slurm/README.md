# Dynamo Slurm Scripts for AMD MI355X

Launch Dynamo on Slurm-managed AMD GPU clusters.

## Scripts

| Script | Nodes | Description |
|--------|-------|-------------|
| `dynamo_standalone.slurm` | 1 | Single node, standalone serving |
| `dynamo_kv_router.slurm` | 2 | KV-cache-aware routing across 2 nodes |
| `dynamo_disagg.slurm` | 2 | Disaggregated prefill/decode with MoRI RDMA |

## Quick Start

```bash
# Single node standalone
sbatch dynamo_standalone.slurm

# 2-node KV Router
sbatch dynamo_kv_router.slurm

# 2-node disagg with MoRI (default)
sbatch dynamo_disagg.slurm

# 2-node disagg with RIXL DRAM staging
BACKEND=nixl sbatch dynamo_disagg.slurm

# 2-node disagg with Mooncake
BACKEND=mooncake sbatch dynamo_disagg.slurm

# Quick test with small model
MODEL=/models/Qwen2.5-0.5B-Instruct TP=1 sbatch dynamo_standalone.slurm
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `/models/DeepSeek-V3` | Model path (must be on shared storage) |
| `TP` | `8` | Tensor parallelism size |
| `BACKEND` | `mori` | Transfer backend: `mori`, `nixl`, `mooncake` |
| `DOCKER_IMAGE` | `rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2` | Container image |
| `DYNAMO_SRC` | `/path/to/workspace` | Dynamo source (shared mount) |
| `PORT` | `8000` | Frontend HTTP port |

## Prerequisites

- Slurm cluster with `--gres=gpu:8` on MI355X nodes
- Model on shared filesystem (NFS/Lustre/VAST) accessible from all nodes
- Container image pulled on all nodes
- For disagg: ionic NICs with matched subnets between nodes

## Logs

Logs go to `logs/dynamo-*-<jobid>.log`. Check with:

```bash
tail -f logs/dynamo-standalone-12345.log
```

## Sending Requests

After the job starts (check logs for "Server ready"):

```bash
# Get the node IP
NODE_IP=$(scontrol show job $SLURM_JOB_ID | grep NodeList | head -1 | awk -F= '{print $2}')

# Send a request
curl http://${NODE_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V3","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'

# Benchmark
python3 -m sglang.bench_serving \
    --backend sglang --base-url http://${NODE_IP}:8000 \
    --model DeepSeek-V3 --num-prompts 100 --request-rate 8
```
