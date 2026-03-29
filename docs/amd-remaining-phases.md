# Dynamo AMD Adaptation — Remaining Phases

**Date**: 2026-03-29
**Branch**: `amd-additive` (upstream-compatible, 99.6% additive)

## Completed Work Summary

| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | Dockerfile `{% if device == "rocm" %}` blocks | Done |
| Phase 1 | `container/context.yaml` ROCm configs | Done |
| Phase 1 | ROCm deploy YAMLs (`*_rocm.yaml`) | Done |
| Phase 1 | ROCm launch scripts (`launch/rocm/*.sh`) | Done |
| Phase 1 | `pyproject.toml` vllm-rocm/sglang-rocm optional deps | Done |
| Phase 1 | ROCm dev container (`Dockerfile.rocm-dev`) | Done |
| Phase 2 | HIP kernel port (`tensor_kernels.hip`) — compiled on MI355X | Done |
| Phase 2 | `build.rs` hipcc compilation path | Done |
| Phase 2 | GPU Memory Service HIP VMM utils + unified facade | Done |
| Phase 2 | GPU HAL abstraction (`lib/memory/src/gpu/`) | Done |
| Phase 2 | Cargo.toml `rocm` feature flags (6 crates) | Done |
| Phase 2 | RIXL built and installed on MI355X | Done |
| Phase 2 | RIXL Rust bindings (`nixl-sys`) compiled | Done |
| Phase 3 | Go operator AMD GPU discovery functions | Done |
| Phase 3 | Snapshot AMD GPU UUID via `amd-smi` | Done |
| Phase 3 | Unified `gpu_utils.py` (amd/nvidia auto-detect) | Done |
| Phase 4 | Upstream sync script (`scripts/sync-upstream.sh`) | Done |
| Phase 4 | MI355X integration test suite (9/11 passed) | Done |

## Remaining Work

### Phase 5: Rust Block Manager HIP Backend (Agent-suitable)

The block manager has CUDA-specific Rust modules that need HIP counterparts.
Each `cuda.rs` file needs a parallel `hip.rs` with the same API surface.

| File | Lines | What it does | HIP Equivalent |
|------|-------|-------------|----------------|
| `lib/llm/src/cuda.rs` | ~208 | CudaContext/CudaStream wrappers | `hip.rs` with HipContext/HipStream |
| `lib/memory/src/pool/cuda.rs` | ~358 | CUDA memory pool (cuMemPool*) | `pool/hip.rs` with HIP memory pool |
| `lib/llm/src/block_manager/storage/cuda.rs` | ~612 | Device/pinned storage allocators | `storage/hip.rs` |
| `lib/llm/src/block_manager/block/transfer/cuda.rs` | ~1163 | Async memcpy + vectorized kernel | `transfer/hip.rs` |
| `lib/llm/src/block_manager/v2/physical/transfer/executor/cuda.rs` | ~319 | Transfer executor | `executor/hip.rs` |
| `lib/llm/src/block_manager/v2/physical/transfer/notifications/cuda_event.rs` | ~89 | Event query | `notifications/hip_event.rs` |
| `lib/kvbm-physical/src/transfer/executor/cuda.rs` | ~328 | KVBM transfer executor | `executor/hip.rs` |
| `lib/kvbm-physical/src/transfer/notifications/cuda_event.rs` | ~88 | KVBM event query | `notifications/hip_event.rs` |

All HIP modules use the GPU HAL layer (`lib/memory/src/gpu/hip.rs`) already created.
Module declarations gated with `#[cfg(feature = "rocm")]`.

**Estimated effort**: 4-6 hours (Agent-suitable, mechanical API mapping)

### Phase 6: vLLM ROCm End-to-End Serving (Needs hardware)

Run actual vLLM inference serving with Dynamo on MI355X.

| Task | Details | Needs |
|------|---------|-------|
| Pull `rocm/vllm` image | vLLM with ROCm support | chi2899 docker |
| Install Dynamo in container | `pip install -e .[vllm-rocm]` | Build from source |
| Test aggregated serving | Single-node vLLM + Dynamo frontend | 1x MI355X |
| Test with actual model | Qwen3-0.6B or similar small model | HF model access |
| Benchmark TTFT/ITL | Compare with standalone vLLM | Performance data |

**Estimated effort**: 1-2 days (needs human for model selection and perf analysis)

### Phase 7: RIXL Disaggregated Serving (Needs 2 nodes)

Multi-node prefill/decode separation via RIXL UCX RDMA.

| Task | Details | Needs |
|------|---------|-------|
| Build UCX+RIXL in vLLM container | Full stack container | Container build |
| RIXL UCX VRAM transfer test | nixlbench between chi2899-chi2900 | 2x MI355X |
| Dynamo disagg with RIXL | Prefill on node1, decode on node2 | 2x MI355X + RDMA |
| KV cache transfer benchmark | Measure RIXL transfer bandwidth | Performance data |

**Estimated effort**: 1-2 weeks (needs RDMA networking verification)

### Phase 8: Planner AMD SMI Integration (Agent-suitable)

Replace pynvml/nvidia-smi usage in test utilities with amdsmi support.

| File | Current | AMD Path |
|------|---------|----------|
| `tests/utils/profile_pytest.py` | pynvml | amdsmi fallback |
| `tests/fault_tolerance/gpu_memory_service/utils/common.py` | pynvml | amdsmi fallback |
| `deploy/sanity_check.py` | nvidia-smi | amd-smi fallback |

Already partially done — `gpu_utils.py` provides unified detection.
Need to wire it into the test framework.

**Estimated effort**: 2-3 hours (Agent-suitable)

### Phase 9: CI/CD Pipeline (Agent-suitable)

Create GitHub Actions workflow for AMD GPU CI.

| Task | Details |
|------|---------|
| `.github/workflows/rocm-ci.yml` | ROCm build + test workflow |
| Container build stage | Build Dockerfile.rocm-dev |
| HIP kernel compile check | Compile tensor_kernels.hip |
| Rust feature check | `cargo check --features rocm` |
| Python import check | Verify gpu_utils, vmm_utils |
| Self-hosted runner config | AMD GPU runner documentation |

**Estimated effort**: 2-3 hours (Agent-suitable)

### Phase 10: Performance Benchmarking (Needs hardware)

| Benchmark | Tool | Baseline |
|-----------|------|----------|
| HIP kernel throughput | Custom benchmark | vs CUDA kernel on H100 |
| RIXL UCX bandwidth | nixlbench | vs NIXL on H100 |
| vLLM aggregated TTFT | Dynamo benchmarks | vs standalone vLLM |
| vLLM disagg TTFT | Dynamo benchmarks | vs aggregated |
| Multi-turn conversation | Dynamo benchmarks | KVBM impact |

**Estimated effort**: 1-2 weeks (needs controlled benchmark environment)

## Priority Order for Immediate Work

```
Phase 5 (Rust HIP backend)    ← Agent can do now, high value
Phase 8 (Planner AMD SMI)     ← Agent can do now, medium value
Phase 9 (CI/CD)               ← Agent can do now, medium value
Phase 6 (vLLM e2e)            ← Needs container + model, high value
Phase 7 (RIXL disagg)         ← Needs 2 nodes + RDMA, highest value
Phase 10 (Benchmarks)         ← Needs all above, final validation
```
