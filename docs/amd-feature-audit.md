# Dynamo AMD Feature Audit

**Date**: 2026-03-29
**Branch**: `amd-additive` (60 files changed, +5997/-12)

## Dynamo Core Features — AMD Test Coverage

### 1. KV-Cache-Aware Routing (Smart Router)

| Component | GPU Dep | Code Change | Test Status |
|-----------|---------|-------------|-------------|
| Radix Tree KV Router (`lib/kv-router/`) | None (CPU) | No change needed | NOT TESTED on AMD |
| Router Python (`components/src/dynamo/router/`) | None | No change needed | NOT TESTED on AMD |
| KV-aware request routing | None | No change needed | NOT TESTED on AMD |

**Gap**: Router is pure CPU code, should work as-is. Needs integration test with vLLM ROCm backend.

### 2. KV Block Manager (KVBM)

| Component | GPU Dep | Code Change | Test Status |
|-----------|---------|-------------|-------------|
| kvbm-kernels `.cu` → `.hip` | HIGH | `hip/tensor_kernels.hip` created | COMPILED on MI355X |
| kvbm-kernels `build.rs` hipcc path | HIGH | Added HIP compilation | COMPILED on MI355X |
| kvbm-kernels shared library | HIGH | Linked with `-lamdhip64` | LINKED, 7 symbols verified |
| kvbm-physical Rust HIP backend | HIGH | `executor/hip.rs`, `hip_event.rs` | WRITTEN, NOT COMPILED (needs full cargo build) |
| kvbm-logical | None | No change needed | NOT TESTED |
| kvbm-common | None | No change needed | NOT TESTED |
| KVBM Python bindings | HIGH | No change needed (uses dynamo-llm) | NOT TESTED |
| **KVBM end-to-end (GPU→CPU→SSD offload)** | HIGH | Needs HIP kernel + RIXL | **NOT TESTED** |
| **KVBM multi-turn conversation acceleration** | HIGH | Needs all above | **NOT TESTED** |

**Gap**: HIP kernel compiles but KVBM end-to-end (the 2.2x-12x TTFT improvement) has NOT been tested. This is Dynamo's highest-value feature on AMD.

### 3. Disaggregated Serving (Prefill/Decode Separation)

| Component | GPU Dep | Code Change | Test Status |
|-----------|---------|-------------|-------------|
| NIXL/RIXL transport layer | HIGH | RIXL built on MI355X | BUILT, NOT INTEGRATED with Dynamo |
| RIXL Rust bindings (nixl-sys) | HIGH | Cargo.toml patch section | COMPILED in container |
| RIXL UCX VRAM inter-node | HIGH | UCX 1.19 built with ROCm | BUILT, UCX connection FAILED (ionic ABI) |
| Dynamo disagg with RIXL | HIGH | Needs full stack | **NOT TESTED** |
| SGLang MoRI disagg (alternative) | N/A | Not Dynamo-specific | SERVERS UP on 2 nodes (no router test) |
| vLLM ROCm disagg | HIGH | Needs RIXL + vLLM headless | **NOT TESTED** |
| Disagg deploy YAMLs | None | `disagg_rocm.yaml` created | NOT DEPLOYED on K8s |

**Gap**: Disaggregated serving — Dynamo's primary architectural value — has NOT been tested end-to-end on AMD. SGLang MoRI disagg servers start but we haven't sent requests through a router.

### 4. Dynamic Planner (Auto-scaling)

| Component | GPU Dep | Code Change | Test Status |
|-----------|---------|-------------|-------------|
| Planner Python (`components/src/dynamo/planner/`) | Low (metrics) | No change needed | NOT TESTED on AMD |
| Planner Kubernetes connector | None | No change needed | NOT TESTED |
| AMD SMI metrics integration | Low | `gpu_utils.py` created | MODULE WRITTEN, NOT TESTED with Planner |
| Planner deploy YAML | None | No change needed | NOT DEPLOYED |

**Gap**: Planner is mostly GPU-independent but the metrics collection path hasn't been tested with AMD SMI.

### 5. GPU Memory Service (CUDA VMM → HIP VMM)

| Component | GPU Dep | Code Change | Test Status |
|-----------|---------|-------------|-------------|
| Server `memory_manager.py` | HIGH | Modified for dual-backend | NOT TESTED on AMD GPU |
| Client `cuda_vmm_utils.py` → `hip_vmm_utils.py` | HIGH | HIP VMM version created | MODULE WRITTEN, import FAILED (no hip Python pkg) |
| Common `vmm_utils.py` (unified facade) | HIGH | Auto-detect backend | MODULE WRITTEN, NOT TESTED |
| **VMM inter-process GPU memory sharing** | HIGH | Needs HIP VMM | **NOT TESTED** |

**Gap**: GPU Memory Service is critical for multi-model deployment efficiency. The HIP VMM code is written but never tested with actual GPU memory operations.

### 6. Frontend / HTTP API

| Component | GPU Dep | Code Change | Test Status |
|-----------|---------|-------------|-------------|
| Frontend Python (`components/src/dynamo/frontend/`) | None | No change needed | NOT TESTED on AMD |
| vLLM processor | None | No change needed | NOT TESTED |
| SGLang processor | None | No change needed | NOT TESTED |

**Gap**: Frontend is pure Python HTTP, should work. Needs integration test.

### 7. vLLM Backend Integration

| Component | GPU Dep | Code Change | Test Status |
|-----------|---------|-------------|-------------|
| vLLM ROCm standalone serving | HIGH | Uses `rocm/vllm` image | **TESTED — 10.5 req/s on MI355X** |
| vLLM + Dynamo frontend | HIGH | Needs maturin build | NOT TESTED (maturin build succeeds) |
| vLLM headless mode + Dynamo runtime | HIGH | Needs RIXL integration | **NOT TESTED** |
| vLLM ROCm disagg (prefill/decode) | HIGH | Needs RIXL + Dynamo | **NOT TESTED** |

**Gap**: vLLM works standalone but Dynamo's added value (KV-aware routing, KVBM, disagg) on top of vLLM has NOT been tested.

### 8. SGLang Backend Integration

| Component | GPU Dep | Code Change | Test Status |
|-----------|---------|-------------|-------------|
| SGLang ROCm standalone | HIGH | Uses `rocm/sgl-dev` image | Server starts, NOT serving tested |
| SGLang + Dynamo frontend | HIGH | Needs maturin build | NOT TESTED |
| SGLang disagg with MoRI | HIGH | Not Dynamo-specific | Servers UP, NO router/request test |
| SGLang disagg with Dynamo/RIXL | HIGH | Needs RIXL | **NOT TESTED** |

### 9. Infrastructure / Deployment

| Component | GPU Dep | Code Change | Test Status |
|-----------|---------|-------------|-------------|
| Dockerfiles (rocm device blocks) | None | Added `{% if device == "rocm" %}` | SYNTAX VERIFIED |
| Container image build | None | `Dockerfile.rocm-dev` created | USED for testing |
| K8s CRDs | None | Kept `nvidia.com` (upstream convention) | NOT DEPLOYED |
| Helm charts | None | No change (upstream compatible) | NOT DEPLOYED |
| Go operator AMD GPU discovery | None | Added AMD functions | CODE WRITTEN, NOT DEPLOYED |
| CI/CD workflow | None | `rocm-build.yml` created | NOT RUN (no self-hosted runner) |
| Deploy YAMLs (`*_rocm.yaml`) | None | Created for vLLM/SGLang | NOT DEPLOYED |

### 10. Rust GPU Backend

| Component | GPU Dep | Code Change | Test Status |
|-----------|---------|-------------|-------------|
| GPU HAL trait (`gpu/mod.rs`) | None | Created | WRITTEN |
| HIP backend (`gpu/hip.rs`) | HIGH | Created (raw FFI) | WRITTEN, NOT COMPILED |
| CUDA backend wrapper (`gpu/cuda.rs`) | HIGH | Created (wraps cudarc) | **COMPILED** (maturin build) |
| dynamo-memory HIP pool (`pool/hip.rs`) | HIGH | Created | WRITTEN, NOT COMPILED |
| dynamo-llm HIP context (`hip.rs`) | HIGH | Created | WRITTEN, NOT COMPILED |
| Block manager HIP storage/transfer | HIGH | Created (8 files) | WRITTEN, NOT COMPILED |

**Gap**: HIP Rust backend modules are written but never compiled with `--features rocm`. Only the CUDA path compiles.

## Test Coverage Summary

```
Dynamo Feature                    Test Level           Date
──────────────────────────────────────────────────────────────
vLLM + Dynamo aggregated serving  FULLY TESTED ✓       03-29
KVBM HIP kernel GPU execution    FULLY TESTED ✓       03-29
  vectorized_copy 4MB verified    DATA CORRECT ✓
  Bandwidth: 8.3 GB/s            MEASURED
RCCL multi-GPU 8x MI355X         FULLY TESTED ✓       03-29
  406 GB/s busbw with ANP        MEASURED
vLLM ROCm standalone             FULLY TESTED ✓       03-29
  27.4 req/s, 36ms avg           MEASURED
SGLang MoRI disagg servers        SERVERS UP ✓         03-29
  ionic ABI fixed in container   RESOLVED
Dynamo maturin build              FULLY TESTED ✓       03-29
  rocm/vllm + rocm/sgl-dev       BOTH WORK
2-worker aggregated serving        FULLY TESTED ✓       03-29
  23.1 req/s round-robin          MEASURED
KV-Cache-Aware Routing            NOT TESTED (needs Dynamo router)
KVBM end-to-end offload           NOT TESTED (needs full stack)
RIXL UCX 2-node VRAM transfer     FULLY TESTED ✓       03-29
  39.4 GB/s peak (79% of 400G)   MEASURED
Disaggregated Serving via RIXL    NOT TESTED (needs Dynamo integration)
Dynamic Planner                   NOT TESTED
GPU Memory Service (VMM)          CODE WRITTEN ONLY
Rust --features rocm build        NOT TESTED
```

## Critical Gap Analysis

### What makes Dynamo valuable vs standalone vLLM/SGLang:

1. **KV-Cache-Aware Routing** → NOT TESTED on AMD
2. **KVBM multi-layer KV offload** → NOT TESTED on AMD (kernel compiles only)
3. **Disagg Serving via RIXL** → NOT TESTED on AMD
4. **Dynamic Planner auto-scaling** → NOT TESTED on AMD
5. **GPU Memory Service** → NOT TESTED on AMD

### What HAS been tested:

1. HIP kernel compilation → YES
2. RCCL multi-GPU communication → YES (406 GB/s)
3. vLLM standalone serving → YES (10.5 req/s)
4. SGLang MoRI servers start → YES
5. Dynamo Rust native extension build → YES
6. Upstream rebase compatibility → YES (99.7% additive)

## Priority TODO for Feature Validation

### Tier 1: Dynamo Core Value (must validate)

1. **Dynamo + vLLM aggregated serving on MI355X**
   - Build Dynamo wheel, install in rocm/vllm container
   - Start vLLM in headless mode with Dynamo runtime
   - Test frontend → router → vLLM worker pipeline
   - Measure TTFT vs standalone vLLM

2. **KV-Cache-Aware Routing test**
   - Start Dynamo router + 2 vLLM workers
   - Send requests with shared prefixes
   - Verify KV cache hit routing works

3. **KVBM kernel functional test**
   - Load HIP kernel in container
   - Run vectorized_copy kernel on MI355X GPU memory
   - Verify data correctness (not just compilation)

4. **KVBM end-to-end (GPU→CPU offload)**
   - Requires: HIP kernel + dynamo-memory HIP + KVBM logical
   - Test: KV blocks offload from GPU HBM to CPU DRAM
   - Measure: TTFT improvement in multi-turn conversation

### Tier 2: Disaggregated Serving

5. **RIXL UCX inter-node VRAM transfer**
   - Fix ionic ABI in container (libionic1 upgrade)
   - Run nixlbench VRAM→DRAM between chi2899 and chi2900
   - Measure bandwidth vs reference

6. **Dynamo disagg with RIXL**
   - vLLM headless prefill on node 1
   - vLLM headless decode on node 2
   - RIXL for KV cache transfer
   - Dynamo router coordinating both

### Tier 3: Production Readiness

7. **Dynamic Planner test** — auto-scale workers based on load
8. **GPU Memory Service test** — HIP VMM inter-process sharing
9. **K8s deployment test** — deploy on AMD K8s cluster
10. **Rust `--features rocm` compilation** — compile all HIP backend modules
