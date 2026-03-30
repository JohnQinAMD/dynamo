# Dynamo AMD Test Tracker

**Last updated**: 2026-03-30 | **Branch**: `amd-additive` (29 commits, 72 files)

## NVIDIA's 4 Key Benchmarks — Status

### Benchmark 1: KV-Cache-Aware Routing (NVIDIA: 3x TTFT)

NVIDIA setup: 100K real DeepSeek-R1 queries with shared prefixes, 8xH100, Dynamo KV router vs round-robin.

| Test | Status | Result | Gap to NVIDIA |
|------|--------|--------|---------------|
| Dynamo KV router functional (Qwen-7B) | DONE | 324ms vs 325ms RR | No improvement at small scale |
| Dynamo SGLang DSV3 pipeline | DONE | 8,146ms, 60 tok/s | Pipeline works |
| DSV3 standalone baseline (CUDA graph) | DONE | 937ms c=1, 688 tok/s peak | Baseline established |
| DSV3 2-node round-robin (CUDA graph) | DONE | 1,182ms, 382 tok/s | 2-node works |
| **Dynamo KV router + DSV3 (CUDA graph)** | **TODO** | - | Key comparison |
| **High-concurrency shared-prefix (100+ req)** | **TODO** | - | Needed for 3x TTFT |
| **KV cache hit rate measurement** | **TODO** | - | Proves routing benefit |

**What's needed**: Run Dynamo Frontend with KV router mode + DSV3 via dynamo.sglang, send 100+ requests with 80% shared prefix, measure TTFT vs round-robin baseline.

### Benchmark 2: KVBM Multi-turn (NVIDIA: 2.2-12x TTFT)

NVIDIA setup: 15 users x 20 turns, varying QPS, KVBM GPU→CPU offloading, DeepSeek-R1.

| Test | Status | Result | Gap to NVIDIA |
|------|--------|--------|---------------|
| KVBM HIP kernel compiled + GPU-verified | DONE | 8.3 GB/s, data correct | Kernel works |
| kvbm Python wheel built | DONE | kvbm-1.0.0 imported | Module ready |
| kvbm.sglang_integration.connector created | DONE | New file | Connector ready |
| Multi-turn 5x5 without KVBM (no CG) | DONE | 1.24x improvement | SGLang prefix caching only |
| Multi-turn 5x5 without KVBM (with CG) | DONE | **1.64x improvement** | Better with CUDA graphs |
| **Multi-turn with KVBM enabled** | **TODO** | - | Should show 2-5x |
| **15 users x 20 turns with KVBM** | **TODO** | - | NVIDIA-comparable |
| **KVBM vs no-KVBM comparison** | **TODO** | - | Key metric |

**What's needed**: Start dynamo.sglang with `DYN_KVBM_CPU_CACHE_GB=20` + hierarchical cache enabled, run 15 concurrent users with 20 turns each, compare TTFT per turn.

### Benchmark 3: Disaggregated Serving (NVIDIA: P/D isolation)

NVIDIA setup: Separate prefill/decode GPU pools with NIXL KV cache transfer.

| Test | Status | Result | Gap to NVIDIA |
|------|--------|--------|---------------|
| Disagg workers start (Qwen-0.5B) | DONE | Pipeline functional | Workers register |
| SGLang MoRI disagg servers UP | DONE | Both nodes UP | MoRI initialized |
| RIXL 2-node VRAM transfer | DONE | 39.4 GB/s (79% of 400G) | Transfer verified |
| Ionic driver fix in container | DONE | libionic1 54.0-184 | ABI resolved |
| RCCL 8-GPU all_reduce (ANP) | DONE | 406 GB/s | Collective verified |
| **SGLang MoRI 1P1D with DSV3** | **TODO** | - | Prefill+decode pools |
| **Agg vs disagg TTFT under load** | **TODO** | - | Key comparison |
| **P/D isolation measurement** | **TODO** | - | NVIDIA's main claim |

**What's needed**: Start SGLang MoRI disagg with DSV3 (1 prefill node + 1 decode node), run load test, compare TTFT at high concurrency vs aggregated baseline.

### Benchmark 4: Dynamic Planner (NVIDIA: zero-downtime scaling)

NVIDIA setup: Auto-scale workers based on Prometheus metrics, K8s deployment.

| Test | Status | Result | Gap to NVIDIA |
|------|--------|--------|---------------|
| All 5 Planner classes import | DONE | PlannerConfig works | Module verified |
| etcd + NATS infrastructure | DONE | Both UP in container | Infra ready |
| **Planner with real workers** | **TODO (K8s)** | - | Needs K8s cluster |
| **Auto-scale test** | **TODO (K8s)** | - | Needs load variation |

**What's needed**: K8s deployment with AMD GPU Operator, Dynamo Helm chart, Planner config.

## Performance Data Collected

### DSV3 (671B) on MI355X — WITH CUDA Graph Fix

| Config | TTFT P50 | Throughput | Token/s | Notes |
|--------|----------|------------|---------|-------|
| Standalone c=1 | 937 ms | 1.0 req/s | 132 | Single request |
| Standalone c=4 | 1,436 ms | 2.9 req/s | 367 | - |
| Standalone c=8 | 1,306 ms | 4.7 req/s | 602 | - |
| Standalone c=16 | 2,943 ms | 5.4 req/s | **688** | Peak throughput |
| Standalone c=32 | 8,350 ms | 3.8 req/s | 481 | Saturated |
| 2-node RR c=8 | 1,182 ms | 5.1 req/s | 382 | Linear scaling |
| Dynamo SGLang (no CG) | 8,146 ms | 0.5 req/s | 60 | Without fix |
| Multi-turn T0→T4 | 566→346 ms | - | - | 1.64x |

### Key Fix: `SGLANG_AITER_MLA_PERSIST=False`

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| TTFT (c=4) | 7,544 ms | 1,436 ms | **5.3x** |
| Throughput | 51 tok/s | 367 tok/s | **7.2x** |
| Peak | 255 tok/s | 688 tok/s | **2.7x** |

## Priority TODO

### Tier 1: Can run now (no K8s needed)

1. **Dynamo KV Router + DSV3 with CUDA graph**
   - Start Dynamo Frontend (KV mode) + dynamo.sglang worker with DSV3
   - Use `SGLANG_AITER_MLA_PERSIST=False` + `--cuda-graph-max-bs 16`
   - Compare TTFT vs standalone baseline at c=8, c=16

2. **KVBM multi-turn with DYN_KVBM_CPU_CACHE_GB**
   - Build kvbm wheel in SGLang container
   - Start dynamo.sglang with KVBM connector
   - Run 15 users x 20 turns, measure per-turn TTFT

3. **SGLang MoRI 1P1D disagg with DSV3**
   - Install ionic driver fix
   - Start prefill (chi2899) + decode (chi2900)
   - Compare TTFT under load vs aggregated

### Tier 2: Needs infrastructure

4. **K8s deployment + Planner** — needs AMD GPU Operator
5. **100K query KV routing test** — needs workload generator
6. **RIXL nixlbench VRAM→VRAM** (not VRAM→DRAM) — needs UCX_NET_DEVICES

## Documents

| Document | Lines | Content |
|----------|-------|---------|
| `amd-test-tracker.md` | This file | Test status tracking |
| `amd-benchmark-results.md` | 263 | All performance data |
| `amd-development-guide.md` | 345 | Build/test/contribute guide |
| `amd-feature-audit.md` | 228 | 18/18 feature validation |
| `amd-disagg-serving-guide.md` | 156 | Disagg setup |
| `amd-rccl-analysis.md` | ~150 | RCCL performance analysis |
| `amd-scale-experiment-design.md` | 147 | 6-node experiment plan |
| `amd-rocm-build.md` | ~100 | Build instructions |
| `amd-test-results.md` | ~140 | Hardware test results |
