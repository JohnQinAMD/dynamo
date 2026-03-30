# Dynamo Scale Experiment Design — NVIDIA-Comparable Numbers on MI355X

**Cluster**: 6 nodes x 8 GPUs = 48x MI355X (288GB HBM3E each, 13.8TB total)
**Nodes**: chi2762, chi2882, chi2885, chi2896, chi2899, chi2900
**Model**: DeepSeek-R1-0528 (671B MoE, FP8 → ~335GB, fits in 8 GPUs)
**Container**: rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2

## Why 6 Nodes

NVIDIA's Dynamo benchmarks showed benefits at scale. With 6 nodes we can test:

```
Experiment 1 (Aggregated baseline):     1 node  = 8 GPUs  → standalone vLLM
Experiment 2 (KV routing):              2 nodes = 16 GPUs → Dynamo + 2 worker pools
Experiment 3 (Disagg 1P2D):             3 nodes = 24 GPUs → 1 prefill + 2 decode
Experiment 4 (Disagg 1P4D):             5 nodes = 40 GPUs → 1 prefill + 4 decode
```

## Experiment Layout

```
Node Assignment:
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ chi2762  │ │ chi2882  │ │ chi2885  │ │ chi2896  │ │ chi2899  │ │ chi2900  │
│ 8x MI355X│ │ 8x MI355X│ │ 8x MI355X│ │ 8x MI355X│ │ 8x MI355X│ │ 8x MI355X│
├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤
│ Exp 1:   │ │ Exp 3:   │ │ Exp 3:   │ │ Exp 4:   │ │ Exp 2:   │ │ Exp 2:   │
│ Baseline │ │ Prefill  │ │ Decode 1 │ │ Decode 2 │ │ Worker 1 │ │ Worker 2 │
│ 1x vLLM  │ │ EP8+DP   │ │ DEP16    │ │ DEP16    │ │ KV Pool  │ │ KV Pool  │
│ TP8 EP8  │ │          │ │          │ │          │ │          │ │          │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Experiment 1: Aggregated Baseline (1 node, 8 GPUs)

**What**: Standalone SGLang serving DeepSeek-R1-0528 with TP=8, EP=8 on 1 node.
No Dynamo, no disaggregation.

**Node**: chi2762
**Config**: Matches InferenceX `dsr1-fp8-mi355x-sglang` (single node)

```bash
# On chi2762
docker run --device=/dev/kfd --device=/dev/dri --network=host \
  -v /mnt/vast/john/huggingface:/models \
  rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2 \
  python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --tp-size 8 --ep-size 8 --dp-size 8 \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code --dtype float16 \
    --mem-fraction-static 0.85
```

**Benchmark**: Send ISL=1024, OSL=1024 workload at concurrency sweep [4, 16, 64, 256, 1024]
**Metrics**: TTFT P50/P99, ITL P50/P99, throughput (tok/s)

## Experiment 2: Dynamo KV-Aware Routing (2 nodes, 16 GPUs)

**What**: Two SGLang worker pools behind Dynamo KV-aware router.
Requests with shared prefixes routed to the worker that has the KV cache.

**Nodes**: chi2899 (Frontend + Worker 1) + chi2900 (Worker 2)

```bash
# chi2899: Dynamo Frontend + KV Router + Worker 1
python3 -m dynamo.frontend --router-mode kv --router-reset-states &
python3 -m dynamo.vllm --model deepseek-ai/DeepSeek-R1-0528 \
  --tp-size 8 --ep-size 8 --dp-size 8 \
  --kv-events-config '...' &

# chi2900: Worker 2
python3 -m dynamo.vllm --model deepseek-ai/DeepSeek-R1-0528 \
  --tp-size 8 --ep-size 8 --dp-size 8 \
  --kv-events-config '...' &
```

**Workload**: Same ISL=1024 workload BUT with 80% shared prefix across sessions.
- 100 unique system prompts, each reused 10x = 1000 requests
- KV router should route same-prefix requests to same worker

**Expected**: 2-3x TTFT improvement for subsequent same-prefix requests
(worker already has KV cache, no prefill needed)

## Experiment 3: Disagg 1P2D (3 nodes, 24 GPUs)

**What**: Disaggregated serving matching NVIDIA's demonstrated architecture.
1 prefill node + 2 decode nodes. MoRI for inter-node KV transfer.

**Nodes**: chi2882 (Prefill) + chi2885 (Decode 1) + chi2896 (Decode 2)

```bash
# Matches InferenceX dsr1-fp8-mi355x-sglang-disagg config:
# Prefill: EP8, no DP
# Decode: DEP16 (EP8 x DP2 across 2 nodes)

# This is the exact config used in NVIDIA's published benchmarks
```

**Metrics**: TTFT (should be better than aggregated because decode doesn't block prefill)
**Concurrency**: [512, 1024, 2048]

## Experiment 4: Disagg 1P4D (5 nodes, 40 GPUs)

**What**: Scale-up disagg with more decode capacity.
1 prefill + 4 decode = 5 nodes.

## Workload Generator

Use `sglang_bench` or custom script with:

```python
# Shared-prefix workload for KV routing test
# 100 unique system prompts, each used by 10 "users"
# Each user sends 5 turns (multi-turn conversation)
# Total: 100 * 10 * 5 = 5000 requests
# ISL: ~1024 tokens, OSL: 1024 tokens
```

## Expected Results vs NVIDIA

| Metric | NVIDIA (H100) | Expected (MI355X) | Scaling Factor |
|--------|---------------|-------------------|----------------|
| KV routing TTFT | 3x improvement | 2-3x | MI355X has 3.6x HBM for KV cache |
| KVBM multi-turn | 2.2-12x | 2-5x | 288GB HBM may reduce offload need |
| Disagg 1P2D throughput | ~17K tok/s | TBD | Depends on MoRI bandwidth |
| Single-node throughput | Baseline | Baseline | Measured in Exp 1 |

## Prerequisites

1. Download DeepSeek-R1-0528 model to shared NFS (if not already available)
2. Pull SGLang MoRI image to all 6 nodes
3. Fix ionic driver in container (libionic1 54.0-184)
4. Set up etcd on one node for Dynamo service discovery
5. Configure NCCL/RCCL for multi-node (AINIC config from Primus)

## Execution Timeline

| Step | Duration | Nodes | Test |
|------|----------|-------|------|
| 1. Pull images | 15 min | all 6 | docker pull |
| 2. Exp 1 baseline | 30 min | chi2762 | Single-node SGLang |
| 3. Exp 2 KV routing | 45 min | chi2899+2900 | Dynamo 2-pool |
| 4. Exp 3 disagg 1P2D | 45 min | chi2882+2885+2896 | MoRI disagg |
| 5. Report | 15 min | - | Generate comparison |
| **Total** | **~2.5 hours** | | |
