# Dynamo AMD Test Tracker

**Last updated**: 2026-03-30 07:30 UTC | **Branch**: `amd-additive`

## NVIDIA's 4 Key Benchmarks — Final Results

### Benchmark 1: KV-Cache-Aware Routing (NVIDIA: 3x TTFT)

**2-Node KV Router vs Round-Robin (DSV3 671B)**:

| Concurrency | RR TTFT P50 | RR tok/s | KV TTFT P50 | KV tok/s | KV ok/total |
|-------------|-------------|----------|-------------|----------|-------------|
| c=4 | 664–819 ms | 213–397 | 1,641 ms | 161 | 12/12 |
| c=8 | 718–1,313 ms | 425–682 | 1,124 ms | 410 | 24/24 |
| c=16 | 851–956 ms | 936–1,025 | 1,390 ms | 728 | 48/48 |
| **c=32** | **882–962 ms** | **13*** | **1,408 ms** | **1,279** | **96/96** |

*RR c=32: only 48/96 succeeded across multiple runs (50% timeout rate)

**Single-Node KV Router vs Standalone (120 shared-prefix requests, c=8)**:

| Config | P50 | req/s | tok/s | ok |
|--------|-----|-------|-------|----|
| Standalone | 719 ms | 8.2 | 527 | 120/120 |
| **KV Router** | **718 ms** | **9.6** | **613** | **120/120** |
| **Improvement** | — | **+17%** | **+16%** | — |

**Key findings**:
1. KV Router provides **17% higher throughput** on shared-prefix workloads (single-node 120 requests)
2. KV Router maintains **100% success at c=32** while RR collapses to 50%
3. At high concurrency (c=32), KV Router delivers **1,279 tok/s vs 13 tok/s** for RR

**Bug fixed**: `lib/kv-router/src/sequences/multi_worker.rs:161` — `assert!(block_size > 1)` panic when SGLang uses `page_size=1`. Changed to gracefully default to 16.

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
| **Disagg cross-node (Qwen, TCP)** | **DONE** | **76.2 req/s at c=8, P50=91ms** |
| Disagg DSV3 routing | **DONE** | Both workers register, requests route |
| Disagg DSV3 TCP (256mb segment) | **BLOCKED** | Decode crashes on large KV transfer |
| Disagg DSV3 nixl/RIXL backend | **BLOCKED** | VRAM registration fails (no GDR for ionic+AMD) |
| Disagg DSV3 1-node (2xTP4) | **BLOCKED** | OOM: 2x DSV3 TP=4 exceeds node memory |
| RIXL 2-node DRAM transfer | **DONE** | 39.4 GB/s (79% of 400Gb/s) |

### Benchmark 4: Dynamic Planner

| Test | Status |
|------|--------|
| Module imports | DONE |
| Planner with workers | TODO (K8s) |

## Performance Summary — DSV3 (671B) on MI355X

### Single Node (8x MI355X) — Verified (multiple runs)

| Concurrency | TTFT P50 | req/s | tok/s |
|-------------|----------|-------|-------|
| c=1 | 514 ms | 1.9 | 124 |
| c=4 | 687–735 ms | 5.5–5.6 | 351–356 |
| c=8 | 725–862 ms | 9.4–11.0 | 599–704 |
| **c=16** | **881–998 ms** | **16.6–17.3** | **1,060–1,108** |
| c=32 | 3,631–3,688 ms | 7.6–7.7 | 489–494 |

### 2-Node Dynamo (RR c=32 comparison, multiple runs)

| Config | TTFT P50 | req/s | tok/s | ok rate |
|--------|----------|-------|-------|---------|
| RR c=16 | 851–956 ms | 14.1–16.0 | 903–1,025 | 100% |
| RR c=32 | 882–962 ms | 0.2 | 13 | **50%** |
| **KV c=32** | **1,408 ms** | **20.0** | **1,279** | **100%** |

### Key Fix: `SGLANG_AITER_MLA_PERSIST=False`

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| TTFT (c=4) | 7,544 ms | 687 ms | **11.0x** |
| Peak tok/s | 255 | 1,108 | **4.3x** |

## All Completed Tests (14 total)

| # | Test | Result |
|---|------|--------|
| 1 | Standalone DSV3 c=1-32 | Peak **17.3 req/s, 1,108 tok/s** at c=16 |
| 2 | 2-node Dynamo RR c=4-32 | Peak **16.0 req/s** at c=16; c=32 collapses |
| 3 | 2-node KV Router c=4-32 | **20.0 req/s, 1,279 tok/s** at c=32 (100% ok) |
| 4 | **KV Router 120 shared-prefix** | **+17% throughput** vs standalone (9.6 vs 8.2 req/s) |
| 5 | KV Router block_size fix | Patched `multi_worker.rs` assertion |
| 6 | KVBM 15x10 multi-turn | T0: **3.34x** improvement |
| 7 | KVBM 15x20 multi-turn | T5: **2.17x** improvement |
| 8 | Disagg single-node (Qwen) | Pipeline verified |
| 9 | **Disagg cross-node (Qwen, TCP)** | **76.2 req/s at c=8, P50=91ms** |
| 10 | Disagg DSV3 routing | Routing works, KV cache too large for TCP |
| 11 | CUDA graph fix | `SGLANG_AITER_MLA_PERSIST=False` → **11.0x** |
| 12 | RIXL 2-node transfer | **39.4 GB/s** (79% of 400Gb/s) |
| 13 | RCCL 8-GPU all_reduce | **406 GB/s** |
| 14 | Standalone verify (2 runs) | Consistent peak 17+ req/s at c=16 |

## Remaining

1. **Disagg KV transfer**: Tested 7 approaches across 3 backends:
   - **Mooncake RDMA**: ibv_reg_mr ENOMEM (ionic lacks AMD GPU Direct RDMA)
   - **Mooncake TCP** (patched `"rdma"→"tcp"`): Init OK, KV data transfer still fails internally
   - **RIXL/nixl backend**: `register_memory("VRAM")` → NIXL_ERR_BACKEND
   - **Single-node 2xTP4**: GPU OOM even at mem_fraction=0.30 (DSV3+mooncake+CUDA graphs exceed 4-GPU capacity)
   - **Sequential load**: Same OOM — DSV3 TP=4 leaves <30% GPU memory, below KV cache minimum
   - **Cross-node Qwen TCP disagg WORKS**: 76.2 req/s, P50=91ms (proven on 2 node pairs)
   - **DSV3 TCP disagg (256mb segment)**: Prefill processes, decode crashes during KV receive
   - **MoRI backend** (`--disaggregation-transfer-backend mori`): EOFError — ionic NICs need backend IPv4 network config (per ROCm docs: benic* interfaces + 192.168.x.x subnets + QoS/DCQCN)
   - **RIXL DRAM staging** (patched "VRAM"→"DRAM"): Registration succeeds but transfer fails — GPU addresses registered as DRAM are invalid; needs actual host-pinned buffers + GPU↔CPU copies
   - **Cross-node Qwen TCP disagg WORKS**: 76.2 req/s, P50=91ms (confirmed again)
   - **Recommended fix for DSV3**: Configure ionic backend network (IPv4 + MTU 9000 + QoS), then use `--disaggregation-transfer-backend mori`
   - DRAM-to-DRAM RIXL transfer verified at 39.4 GB/s
2. **K8s + Planner**: Needs AMD GPU Operator deployment
