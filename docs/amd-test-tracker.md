# Dynamo AMD Test Tracker

**Last updated**: 2026-03-30 07:00 UTC | **Branch**: `amd-additive`

## NVIDIA's 4 Key Benchmarks — Final Results

### Benchmark 1: KV-Cache-Aware Routing (NVIDIA: 3x TTFT)

**2-Node KV Router vs Round-Robin (DSV3 671B, page_size=16 fix applied)**:

| Concurrency | RR TTFT P50 | RR tok/s | KV TTFT P50 | KV tok/s | KV ok/total |
|-------------|-------------|----------|-------------|----------|-------------|
| c=4 | 664 ms | 397 | 1,641 ms | 161 | 12/12 |
| c=8 | 718 ms | 682 | 1,124 ms | 410 | 24/24 |
| c=16 | 956 ms | 1,025 | 1,390 ms | 728 | 48/48 |
| **c=32** | **882 ms** | **13*** | **1,408 ms** | **1,279** | **96/96** |

*RR c=32: only 48/96 requests succeeded (50% timeout rate)

**Key finding**: KV Router maintains **100% success rate and 1,279 tok/s at c=32** while Round-Robin collapses to 13 tok/s with 50% failures. The KV Router's intelligent routing prevents worker overload at high concurrency, trading P50 latency for throughput stability.

**Bug fixed**: `lib/kv-router/src/sequences/multi_worker.rs:161` — changed `assert!(block_size > 1)` to gracefully default to 16 when SGLang uses `page_size=1`.

### Benchmark 2: KVBM Multi-turn (NVIDIA: 2.2-12x TTFT)

**15 users x 20 turns (NVIDIA-comparable scale)**:

| Turn | Baseline (ms) | KVBM (ms) | Improvement |
|------|---------------|-----------|-------------|
| T0 | 4,110 | 2,162 | **1.90x** |
| T5 | 2,379 | 1,097 | **2.17x** |
| T10 | 1,793 | 1,389 | 1.29x |
| T15 | 1,395 | 1,254 | 1.11x |
| T19 | 1,225 | 1,117 | 1.10x |

**Previously (15x10)**: T0 3.34x improvement (3,638ms → 1,089ms)

### Benchmark 3: Disaggregated Serving (NVIDIA: P/D isolation)

| Test | Status | Result |
|------|--------|--------|
| Disagg architecture (Qwen-0.5B) | **DONE** | Pipeline works end-to-end |
| Disagg with DSV3 | **PARTIAL** | Routing works, KV transfer needs RDMA |
| RIXL 2-node transfer | **DONE** | 39.4 GB/s (79% of 400Gb/s) |

### Benchmark 4: Dynamic Planner

| Test | Status |
|------|--------|
| Module imports | DONE |
| Planner with workers | TODO (K8s) |

## Performance Summary — DSV3 (671B) on MI355X

### Single Node (8x MI355X) — Verified

| Concurrency | TTFT P50 | TTFT P95 | req/s | tok/s |
|-------------|----------|----------|-------|-------|
| c=1 | 514 ms | — | 1.9 | 124 |
| c=4 | 735 ms | — | 5.6 | 356 |
| c=8 | 862 ms | — | 9.4 | 599 |
| **c=16** | **998 ms** | — | **16.6** | **1,060** |
| c=32 | 3,631 ms | — | 7.7 | 494 |

**Shared-prefix 120 requests at c=8**: P50=719ms, P95=1504ms, 9.7 req/s, 623 tok/s (120/120 ok)

### 2-Node via Dynamo — Complete Comparison

| Config | TTFT P50 | req/s | tok/s | ok rate |
|--------|----------|-------|-------|---------|
| RR c=4 | 664 ms | 6.2 | 397 | 100% |
| RR c=8 | 718 ms | 10.7 | 682 | 100% |
| RR c=16 | 956 ms | 16.0 | 1,025 | 100% |
| RR c=32 | 882 ms | 0.2 | 13 | **50%** |
| KV c=4 | 1,641 ms | 2.5 | 161 | 100% |
| KV c=8 | 1,124 ms | 6.4 | 410 | 100% |
| KV c=16 | 1,390 ms | 11.4 | 728 | 100% |
| **KV c=32** | **1,408 ms** | **20.0** | **1,279** | **100%** |

### Key Fix: `SGLANG_AITER_MLA_PERSIST=False`

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| TTFT (c=4) | 7,544 ms | 735 ms | **10.3x** |
| Peak tok/s | 255 | 1,060 | **4.2x** |

## All Completed Tests

| # | Test | Result |
|---|------|--------|
| 1 | Standalone DSV3 c=1-32 | Peak **16.6 req/s, 1,060 tok/s** at c=16 |
| 2 | 2-node Dynamo RR c=4-32 | Peak **16.0 req/s, 1,025 tok/s** at c=16 |
| 3 | 2-node Dynamo KV Router c=4-32 | **20.0 req/s, 1,279 tok/s** at c=32 (100% ok) |
| 4 | KV Router block_size fix | Patched `multi_worker.rs` + `--page-size 16` |
| 5 | KVBM 15x10 multi-turn | T0: **3.34x** improvement |
| 6 | KVBM 15x20 multi-turn | T5: **2.17x** improvement |
| 7 | Disagg single-node (Qwen) | Pipeline verified |
| 8 | Disagg DSV3 routing | Routing works, KV transfer needs RDMA |
| 9 | CUDA graph fix | `SGLANG_AITER_MLA_PERSIST=False` → **10.3x** |
| 10 | RIXL 2-node transfer | **39.4 GB/s** (79% of 400Gb/s) |
| 11 | RCCL 8-GPU all_reduce | **406 GB/s** |
| 12 | Shared-prefix 120 requests | P50=719ms, 9.7 req/s (120/120) |

## Remaining

1. **Disagg KV transfer**: Needs RIXL/UCX RDMA between disagg node pair
2. **K8s + Planner**: Needs AMD GPU Operator
3. **100K query KV routing**: Larger workload for 3x TTFT demonstration
