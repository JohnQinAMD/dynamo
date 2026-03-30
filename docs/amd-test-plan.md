# Dynamo on AMD ROCm — Test Plan

> Comprehensive plan to bring NVIDIA Dynamo's 68-file test suite to full ROCm coverage on MI300X / MI325X / MI355X.

**Status**: **164 tests pass on MI355X** (42 skipped, 1 infra error). Validated on `chi2896` using `rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2` with `maturin develop --release`.

---

## Current State

| Metric | Count | Notes |
|--------|-------|-------|
| Total test files (`dynamo/tests/`) | 68+ | Upstream + AMD-additive tests |
| Tests passing on MI355X | **164** | Full suite run on chi2896 |
| Tests skipped (expected) | **42** | vLLM not installed (34), RIXL not in image (2), ionic in Docker (1), env vars (5) |
| Tests blocked (NVIDIA-only) | **20** | TRT-LLM (6), vLLM Python 3.12 gap (14) |
| Manual ROCm tests (performance report) | 20 | Includes DRAM staging + mooncake patch |
| ROCm CI workflow | 1 | Build/lint + GPU test execution |
| Container | 1 | `rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2` |

### Build Recipe (required for `dynamo.llm`)

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
    --shm-size 64G -v /mnt/vast/john/rocm-dynamo:/workspace \
    rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2

# Inside container:
cp -r /workspace/dynamo /tmp/dynamo && cd /tmp/dynamo
export LIBCLANG_PATH=/opt/rocm/lib/llvm/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I$(find /usr/lib/gcc -name stdbool.h | head -1 | xargs dirname)"
export VIRTUAL_ENV=/opt/venv
cd lib/bindings/python && maturin develop --release
cd /tmp/dynamo && pip install -e .
pip install pytest pytest-benchmark pytest-httpserver pytest-asyncio pytest-timeout \
    nats-py kr8s prometheus_api_client filterpy pmdarima prophet boto3 kubernetes_asyncio

# Run all tests:
python3 -m pytest tests/ --no-header -q --tb=no  # 164 passed, 42 skipped
```

---

## Test Categories & ROCm Readiness

### Legend

- **Ready** — Can run on ROCm with zero or trivial changes
- **Adapt** — Needs config/fixture changes but logic is portable
- **Blocked** — Depends on CUDA-only component (TRT-LLM, NIXL, etc.)
- **New** — ROCm-specific test that doesn't exist upstream

---

### Phase 0: Infrastructure & Build Validation (Priority: P0)

These verify the build toolchain and basic GPU access before any Dynamo-specific tests.

| # | Test | File / Script | ROCm Status | Action |
|---|------|--------------|-------------|--------|
| 0.1 | GPU detection (rocm-smi) | `test_rocm_integration.sh` | Ready | Migrate to pytest: `tests/basic/test_rocm_gpu_detection.py` |
| 0.2 | HIP kernel compilation (gfx942/gfx950) | `test_rocm_integration.sh` | Ready | Migrate to pytest |
| 0.3 | Shared library linking (libkvbm_kernels) | `test_rocm_integration.sh` | Ready | Migrate to pytest |
| 0.4 | PyTorch GPU compute (HIP) | `test_rocm_integration.sh` | Ready | Migrate to pytest |
| 0.5 | CUDA version consistency | `tests/basic/test_cuda_version_consistency.py` | **Adapt** | Add ROCm version check path (rocm-smi / hipcc) |
| 0.6 | Wheel contents | `tests/basic/test_wheel_contents.py` | **Adapt** | Verify HIP .so files are included in ROCm wheel |
| 0.7 | Dynamo `gpu_utils` detection | `test_rocm_integration.sh` | Ready | Should detect `backend='amd'` |
| 0.8 | Rust bindings import | — | Ready | `import dynamo_llm` on ROCm container |
| 0.9 | RIXL/nixl library presence | `test_rocm_integration.sh` | Ready | Check `libnixl.so` under RIXL prefix |

**Target**: All P0 tests pass in CI on `rocm/pytorch` container.

---

### Phase 1: Unit Tests — No GPU Required (Priority: P0)

These tests run on CPU and are vendor-agnostic. They should pass on ROCm without changes.

| # | Test | File | ROCm Status | Action |
|---|------|------|-------------|--------|
| 1.1 | PlannerConfig validation | `tests/planner/unit/test_planner_config.py` | **Ready** | Add `@pytest.mark.rocm`, run with `backend: sglang` |
| 1.2 | Load predictors (Kalman) | `tests/planner/unit/test_load_predictors.py` | **Ready** | Pure math, no GPU |
| 1.3 | Load-based scaling logic | `tests/planner/unit/test_load_based_scaling.py` | **Ready** | Pure logic |
| 1.4 | SLA planner scaling | `tests/planner/unit/test_sla_planner_scaling.py` | **Ready** | Pure logic |
| 1.5 | Replica calculation | `tests/planner/test_replica_calculation.py` | **Ready** | Pure logic |
| 1.6 | Virtual connector | `tests/planner/unit/test_virtual_connector.py` | **Adapt** | May reference NVIDIA GPU types in fixtures |
| 1.7 | Remote planner | `tests/planner/unit/test_remote_planner.py` | **Ready** | Network-only |
| 1.8 | Prometheus helper | `tests/planner/unit/test_prometheus.py` | **Ready** | No GPU |
| 1.9 | Load generator | `tests/planner/test_load_generator.py` | **Ready** | Pure Python |
| 1.10 | Global planner scale request | `tests/global_planner/unit/test_scale_request_handler.py` | **Ready** | Logic only |
| 1.11 | KVBM imports | `tests/dependencies/test_kvbm_imports.py` | **Adapt** | Check HIP kernel imports instead of CUDA |
| 1.12 | Mocker config | `tests/mocker/test_config.py` | **Ready** | No GPU |
| 1.13 | Predownload models | `tests/test_predownload_models.py` | **Ready** | Network only |
| 1.14 | Disagg logprobs serialization | `tests/serve/test_disagg_logprobs_serialization.py` | **Ready** | Pure data test |
| 1.15 | Prometheus exposition format | `tests/serve/test_prometheus_exposition_format_injection.py` | **Ready** | String parsing |
| 1.16 | Output format test | `tests/utils/test_output.py` | **Ready** | No GPU |
| 1.17 | Mock GPU alloc | `tests/utils/test_mock_gpu_alloc.py` | **Adapt** | May reference `nvidia-smi`; switch to `rocm-smi` |
| 1.18 | Managed process teardown | `tests/utils/test_managed_process_teardown.py` | **Ready** | Process management |

**Target**: `pytest -m "rocm and unit" --co` should list 15+ tests.

---

### Phase 2: Profiler & Config Tests — No GPU Required (Priority: P1)

The profiler SLA tests validate YAML → Pydantic model parsing. They reference GPU SKUs that need AMD additions.

| # | Test | File | ROCm Status | Action |
|---|------|------|-------------|--------|
| 2.1 | DGDR rapid (supported) | `tests/profiler/test_profile_sla_dgdr.py` | **Adapt** | Add MI355X DGDR YAML configs (new GPU SKU type `mi355x` in `GPUSKUType` enum) |
| 2.2 | DGDR thorough helpers | `tests/profiler/test_helpers_thorough.py` | **Adapt** | Same — reference MI355X hardware |
| 2.3 | DGDR rapid helpers | `tests/profiler/test_helpers_rapid.py` | **Adapt** | Same |
| 2.4 | Profile SLA helpers | `tests/profiler/test_helpers_profile_sla.py` | **Adapt** | Same |
| 2.5 | Deploy YAML discovery | `tests/deploy/test_deploy.py` | **Adapt** | Add ROCm example deploy YAMLs |

**Action items**:
1. Add `mi300x`, `mi325x`, `mi355x` to `GPUSKUType` enum in `dgdr_v1beta1_types.py`
2. Create profiler YAML configs for AMD hardware under `tests/profiler/configs/`
3. Create example deploy YAMLs with `amd.com/gpu` resources

---

### Phase 3: Router Tests — 1+ GPU Required (Priority: P0)

KV-cache routing is the highest-value AMD feature (4.35x TTFT improvement validated). Router tests must work on ROCm.

| # | Test | File | ROCm Status | Action |
|---|------|------|-------------|--------|
| 3.1 | Router E2E with mockers | `tests/router/test_router_e2e_with_mockers.py` | **Ready** | Mocker backend is GPU-agnostic; add `@pytest.mark.rocm` |
| 3.2 | Router E2E with SGLang | `tests/router/test_router_e2e_with_sglang.py` | **Adapt** | Ensure `--attention-backend aiter`, `--kv-cache-dtype fp8_e4m3`, `--page-size 16` |
| 3.3 | Router E2E with vLLM | `tests/router/test_router_e2e_with_vllm.py` | **Blocked** | vLLM on ROCm needs Python 3.12; Dynamo needs 3.10 |
| 3.4 | Router E2E with TRT-LLM | `tests/router/test_router_e2e_with_trtllm.py` | **Blocked** | TRT-LLM is NVIDIA-only |
| 3.5 | **NEW**: KV Router block_size fix | — | **New** | Test that `block_size=1` (SGLang default) doesn't panic; regression test for `multi_worker.rs` fix |
| 3.6 | **NEW**: KV Router DSV3 2-node | — | **New** | Replicate the manual `test_kv_router_dsv3.sh` as pytest |
| 3.7 | MM Router E2E | `tests/mm_router/test_mm_router_e2e.py` | **Adapt** | Needs multimodal SGLang on ROCm |
| 3.8 | MM Router vLLM E2E | `tests/mm_router/test_vllm_mm_router_e2e.py` | **Blocked** | vLLM Python version gap |

**Critical**: Test 3.5 is a regression test for the `multi_worker.rs` block_size bug — must be in pre-merge CI.

---

### Phase 4: Frontend & Serve Tests — 1+ GPU Required (Priority: P1)

| # | Test | File | ROCm Status | Action |
|---|------|------|-------------|--------|
| 4.1 | Frontend completion (mocker) | `tests/frontend/test_completion_mocker_engine.py` | **Ready** | Mocker is GPU-agnostic |
| 4.2 | Frontend prepost | `tests/frontend/test_prepost.py` | **Ready** | Tokenizer only |
| 4.3 | Frontend prepost (Mistral) | `tests/frontend/test_prepost_mistral.py` | **Ready** | Tokenizer only |
| 4.4 | Frontend prompt embeds | `tests/frontend/test_prompt_embeds.py` | **Adapt** | Needs torch on ROCm |
| 4.5 | Frontend vLLM prepost | `tests/frontend/test_vllm_prepost_integration.py` | **Blocked** | vLLM import |
| 4.6 | Frontend vLLM | `tests/frontend/test_vllm.py` | **Blocked** | vLLM import |
| 4.7 | gRPC tensor params | `tests/frontend/grpc/test_tensor_parameters.py` | **Ready** | gRPC protocol |
| 4.8 | gRPC tensor mocker | `tests/frontend/grpc/test_tensor_mocker_engine.py` | **Ready** | Mocker engine |
| 4.9 | Serve SGLang | `tests/serve/test_sglang.py` | **Adapt** | Key test — SGLang serve on ROCm with aiter |
| 4.10 | Serve vLLM | `tests/serve/test_vllm.py` | **Blocked** | vLLM Python version gap |
| 4.11 | Serve TRT-LLM | `tests/serve/test_trtllm.py` | **Blocked** | TRT-LLM NVIDIA-only |
| 4.12 | Serve vLLM Omni | `tests/serve/test_vllm_omni.py` | **Blocked** | vLLM Omni NVIDIA-only |

**Priority**: Tests 4.1, 4.9 are the most important — they validate the SGLang serving path on ROCm.

---

### Phase 5: KVBM Tests — 1-8 GPU Required (Priority: P1)

KVBM provides 2.17-3.34x multi-turn TTFT improvement on AMD. These tests need HIP kernel support.

| # | Test | File | ROCm Status | Action |
|---|------|------|-------------|--------|
| 5.1 | KVBM basic | `tests/kvbm_integration/test_kvbm.py` | **Adapt** | Needs HIP KVBM kernels compiled; switch CUDA → HIP paths |
| 5.2 | KVBM CUDA graph | `tests/kvbm_integration/test_cuda_graph.py` | **Adapt** | ROCm CUDA graph (hipGraph) support; rename references |
| 5.3 | KVBM chunked prefill | `tests/kvbm_integration/test_chunked_prefill.py` | **Adapt** | Verify chunked prefill with `SGLANG_AITER_MLA_PERSIST=False` |
| 5.4 | KVBM vLLM integration | `tests/kvbm_integration/test_kvbm_vllm_integration.py` | **Blocked** | vLLM Python version gap |
| 5.5 | KVBM determinism (agg) | `tests/kvbm_integration/test_determinism_agg.py` | **Adapt** | Should work with SGLang backend |
| 5.6 | KVBM determinism (disagg) | `tests/kvbm_integration/test_determinism_disagg.py` | **Adapt** | Needs MoRI transfer backend |
| 5.7 | KVBM consolidator-router E2E | `tests/kvbm_integration/test_consolidator_router_e2e.py` | **Adapt** | Combines KVBM + router |
| 5.8 | **NEW**: KVBM multi-turn benchmark | — | **New** | Replicate 15×20 turn test from performance report |

**Prerequisite**: HIP KVBM kernels (`tensor_kernels.hip`) must compile and link.

---

### Phase 6: Planner E2E & Scaling Tests — Multi-GPU (Priority: P2)

| # | Test | File | ROCm Status | Action |
|---|------|------|-------------|--------|
| 6.1 | Planner scaling E2E | `tests/planner/test_scaling_e2e.py` | **Adapt** | Use SGLang backend + `environment: virtual` |
| 6.2 | **NEW**: FPM relay (SGLang) | — | **New** | Test `SglangFpmRelay` publishes ForwardPassMetrics to NATS |
| 6.3 | **NEW**: Planner virtual mode (SGLang) | — | **New** | End-to-end planner with SGLang worker on MI355X |
| 6.4 | **NEW**: Load-based scaling (SGLang) | — | **New** | Verify load scaling triggers with FPM metrics |

**Note**: Upstream planner tests require vLLM FPM — SGLang FPM relay is AMD-additive code that needs its own tests.

---

### Phase 7: Fault Tolerance Tests — Multi-GPU / K8s (Priority: P2)

| # | Test | File | ROCm Status | Action |
|---|------|------|-------------|--------|
| 7.1 | Cancellation (SGLang) | `tests/fault_tolerance/cancellation/test_sglang.py` | **Adapt** | Add ROCm SGLang launch args |
| 7.2 | Cancellation (vLLM) | `tests/fault_tolerance/cancellation/test_vllm.py` | **Blocked** | vLLM gap |
| 7.3 | Cancellation (TRT-LLM) | `tests/fault_tolerance/cancellation/test_trtllm.py` | **Blocked** | TRT-LLM |
| 7.4 | etcd HA (SGLang) | `tests/fault_tolerance/etcd_ha/test_sglang.py` | **Adapt** | Test etcd failover with SGLang on ROCm |
| 7.5 | etcd HA (vLLM) | `tests/fault_tolerance/etcd_ha/test_vllm.py` | **Blocked** | vLLM gap |
| 7.6 | etcd HA (TRT-LLM) | `tests/fault_tolerance/etcd_ha/test_trtllm.py` | **Blocked** | TRT-LLM |
| 7.7 | Migration (SGLang) | `tests/fault_tolerance/migration/test_sglang.py` | **Adapt** | SGLang migration on ROCm |
| 7.8 | Migration (vLLM) | `tests/fault_tolerance/migration/test_vllm.py` | **Blocked** | vLLM gap |
| 7.9 | Migration (TRT-LLM) | `tests/fault_tolerance/migration/test_trtllm.py` | **Blocked** | TRT-LLM |
| 7.10 | vLLM health check | `tests/fault_tolerance/test_vllm_health_check.py` | **Blocked** | vLLM |
| 7.11 | GMS sleep/wake (SGLang) | `tests/fault_tolerance/gpu_memory_service/test_gms_sleep_wake_sglang.py` | **Adapt** | Needs HIP VMM |
| 7.12 | GMS sleep/wake (vLLM) | `tests/fault_tolerance/gpu_memory_service/test_gms_sleep_wake_vllm.py` | **Blocked** | vLLM |
| 7.13 | GMS shadow failover (SGLang) | `tests/fault_tolerance/gpu_memory_service/test_gms_shadow_failover_sglang.py` | **Adapt** | Needs HIP VMM |
| 7.14 | GMS shadow failover (vLLM) | `tests/fault_tolerance/gpu_memory_service/test_gms_shadow_failover_vllm.py` | **Blocked** | vLLM |
| 7.15 | Failover lock | `tests/fault_tolerance/gpu_memory_service/test_failover_lock.py` | **Ready** | Lock logic is GPU-agnostic |
| 7.16 | Deploy test | `tests/fault_tolerance/deploy/test_deployment.py` | **Adapt** | Needs AMD DGD YAML templates |

---

### Phase 8: Disaggregated Serving Tests — Multi-Node (Priority: P1)

| # | Test | File / Script | ROCm Status | Action |
|---|------|--------------|-------------|--------|
| 8.1 | **NEW**: Disagg Qwen MoRI RDMA | — | **New** | 2-node test: prefill + decode via MoRI, small model |
| 8.2 | **NEW**: Disagg DSV3 MoRI RDMA | — | **New** | 2-node test: DSV3 671B with matched ionic subnets |
| 8.3 | **NEW**: Transfer backend selection | — | **New** | Verify `--disaggregation-transfer-backend mori` vs `tcp` |
| 8.4 | **NEW**: Ionic subnet validation | — | **New** | Pre-flight check: matched ionic GIDs between nodes |
| 8.5 | lmcache deploy scripts | `tests/lmcache/*.sh` | **Adapt** | Replace NVIDIA images with ROCm images |

---

### Phase 9: K8s & Deploy Tests — K8s Cluster Required (Priority: P3)

| # | Test | File | ROCm Status | Action |
|---|------|------|-------------|--------|
| 9.1 | Deploy test | `tests/deploy/test_deploy.py` | **Adapt** | Add AMD example DGD YAMLs with `amd.com/gpu` resources |
| 9.2 | FT deploy | `tests/fault_tolerance/deploy/test_deployment.py` | **Adapt** | MoE templates for AMD (if applicable) |
| 9.3 | **NEW**: K8s CRD validation | — | **New** | Verify 7 CRDs deploy correctly on AMD K8s |
| 9.4 | **NEW**: AMD GPU Operator integration | — | **New** | `amd.com/gpu` resource scheduling |
| 9.5 | Autodeploy backend | `tests/basic/test_autodeploy_backend.py` | **Blocked** | TRT-LLM autodeploy engine |

---

### Phase 10: InferenceX Benchmark Integration (Priority: P2)

| # | Test | Action |
|---|------|--------|
| 10.1 | Add `dynamo-sglang` entries to `amd-master.yaml` | MI355X disagg with MoRI RDMA |
| 10.2 | Add benchmark scripts for AMD Dynamo | `benchmarks/multi_node/dsr1_fp8_mi355x_dynamo-sglang.sh` |
| 10.3 | Validate sweep config generation | Extend `utils/matrix_logic/test_validation.py` for new entries |
| 10.4 | Performance regression tracking | Baseline: 7.4 req/s disagg DSV3, 17.3 req/s agg |

---

## Test Count Summary

| Category | Total | Ready | Adapt | Blocked | New |
|----------|-------|-------|-------|---------|-----|
| P0: Build & Infrastructure | 9 | 6 | 2 | 0 | 1 |
| P0: Unit Tests (no GPU) | 18 | 14 | 4 | 0 | 0 |
| P0: Router Tests | 8 | 1 | 2 | 3 | 2 |
| P1: Frontend & Serve | 12 | 5 | 2 | 5 | 0 |
| P1: KVBM | 8 | 0 | 6 | 1 | 1 |
| P1: Disagg Multi-Node | 5 | 0 | 1 | 0 | 4 |
| P2: Planner E2E | 4 | 0 | 1 | 0 | 3 |
| P2: InferenceX | 4 | 0 | 0 | 0 | 4 |
| P2: Fault Tolerance | 16 | 1 | 5 | 10 | 0 |
| P3: K8s & Deploy | 5 | 0 | 2 | 1 | 2 |
| **Total** | **89** | **27** | **25** | **20** | **17** |

---

## Blockers & Dependencies

| Blocker | Impact | Tests Blocked | Unblock Path |
|---------|--------|---------------|-------------|
| ~~vLLM Python 3.12 vs 3.10~~ | ~~All vLLM backend tests~~ | ~~14~~ | **RESOLVED** — abi3-py310 wheel works on 3.12; verified with `maturin develop` in `rocm/vllm:latest` (175 tests pass) |
| **TRT-LLM NVIDIA-only** | All TRT-LLM tests | 6 | N/A — skip permanently with `@pytest.mark.skip(reason="TRT-LLM NVIDIA-only")` |
| ~~NIXL/RIXL VRAM registration~~ | ~~RIXL-based disagg~~ | ~~2~~ | **FIXED** — DRAM staging monkey-patch (`nixl_rocm_staging.py`) |
| **HIP KVBM kernel linkage** | KVBM integration tests | 6 | Complete `cargo build --features block-manager-rocm` pipeline |

---

## Environment Setup

### Required Environment Variables

```bash
export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export SGLANG_AITER_MLA_PERSIST=False
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LIBCLANG_PATH=/opt/rocm/lib/llvm/lib
export NIXL_PREFIX=/opt/rocm/rixl
export LD_LIBRARY_PATH=/opt/rocm/rixl/lib:$LD_LIBRARY_PATH
```

### Containers

| Container | Python | Use For |
|-----------|--------|---------|
| `rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2` | 3.10 | SGLang + MoRI + Dynamo tests |
| `rocm/pytorch:latest` | 3.10 | Build & unit tests |
| `rocm/vllm:latest` | 3.12 | vLLM tests (once maturin 3.12 wheel available) |

### Running Tests

```bash
# P0: Unit tests (no GPU)
pytest tests/planner/unit/ tests/mocker/ tests/utils/ -m "not (vllm or trtllm)" -v

# P0: Build validation
pytest tests/basic/ -m "rocm" -v

# P0: Router with mocker (no real GPU needed)
pytest tests/router/test_router_e2e_with_mockers.py -v

# P1: SGLang serve (needs GPU)
pytest tests/serve/test_sglang.py tests/router/test_router_e2e_with_sglang.py -m "rocm" -v

# P1: KVBM (needs GPU + HIP kernels)
pytest tests/kvbm_integration/ -m "rocm and not vllm" -v

# Skip all NVIDIA-only tests
pytest -m "not (trtllm or h100)" -v
```

---

## CI Pipeline Design

### Tier 1: Pre-merge (every PR, ~5 min)

```yaml
runs-on: [self-hosted, rocm, mi355x]
tests:
  - pytest tests/basic/ tests/planner/unit/ tests/mocker/ -m "not (vllm or trtllm)" -v
  - pytest tests/frontend/test_completion_mocker_engine.py -v
  - pytest tests/router/test_router_e2e_with_mockers.py -v
  - cargo test -p dynamo-memory --features testing-all-rocm
```

### Tier 2: Post-merge (nightly, ~30 min)
```yaml
runs-on: [self-hosted, rocm, mi355x]
tests:
  - pytest tests/serve/test_sglang.py -m "rocm" -v
  - pytest tests/router/test_router_e2e_with_sglang.py -m "rocm" -v
  - pytest tests/kvbm_integration/ -m "rocm and not vllm" -v
  - pytest tests/fault_tolerance/cancellation/test_sglang.py -m "rocm" -v
  - pytest tests/fault_tolerance/etcd_ha/test_sglang.py -m "rocm" -v
```

### Tier 3: Weekly (multi-node, ~2 hr)

```yaml
runs-on: [self-hosted, rocm, mi355x-disagg]
tests:
  - bash tests/disagg/test_mori_qwen_2node.sh
  - bash tests/disagg/test_mori_dsv3_2node.sh
  - pytest tests/planner/test_scaling_e2e.py -m "rocm" -v
```

---

## Implementation Roadmap

### Sprint 1 (Week 1-2): Foundation

- [x] Add `@pytest.mark.rocm` to all Phase 1 unit tests
- [x] Create `tests/basic/test_rocm_gpu_detection.py` from `test_rocm_integration.sh`
- [x] Fix `test_cuda_version_consistency.py` for ROCm → `test_rocm_version_consistency.py`
- [ ] Verify all Phase 1 tests pass on `rocm/pytorch` container
- [x] Set up self-hosted ROCm runner in GitHub Actions → `.github/workflows/rocm-test.yml`

### Sprint 2 (Week 3-4): Core Features

- [x] Adapt `test_router_e2e_with_sglang.py` for ROCm (added `@pytest.mark.rocm`)
- [x] Write regression test for `multi_worker.rs` block_size fix → `test_router_block_size_regression.py`
- [x] Adapt `test_sglang.py` serve test for ROCm → `test_sglang_rocm.py`
- [x] Complete HIP KVBM kernel build pipeline → `test_kvbm_rocm.py` (compile + link tests)
- [x] Write KVBM SGLang integration tests → `test_kvbm_rocm.py::TestKvbmSglangIntegration`

### Sprint 3 (Week 5-6): Disagg & Planner

- [x] Write MoRI RDMA disagg pytest (2-node) → `tests/disagg/test_mori_rdma.py`
- [x] Write ionic NIC validation → `tests/disagg/test_ionic_validation.py`
- [x] Write FPM relay tests for SGLang → `tests/planner/test_fpm_relay_sglang.py`
- [x] Write planner virtual + SGLang tests → `tests/planner/test_planner_virtual_sglang.py`

### Sprint 4 (Week 7-8): CI & InferenceX

- [x] Implement Tier 1/2/3 CI pipelines → `.github/workflows/rocm-test.yml`
- [x] Add `dynamo-sglang` entries to InferenceX `amd-master.yaml`
- [x] Write benchmark script for AMD Dynamo → `benchmarks/multi_node/dsr1_fp8_mi355x_dynamo-sglang.sh`
- [x] Create test runner script → `scripts/run_rocm_tests.sh` (Tier 1/2/3 in-container runner)

### Sprint 5 (Week 9-10): Unblock vLLM & Harden

- [x] Create maturin Py3.12 build script → `scripts/build_maturin_py312.sh`
- [x] Create fault tolerance DGD templates for SGLang/ROCm → `templates/sglang/rocm_{agg,disagg}.yaml`
- [x] Write K8s CRD validation tests → `tests/deploy/test_k8s_crd_validation.py`
- [ ] *Run* all tests on MI355X hardware (requires GPU access)

---

## New Test Files to Create

| File | Phase | Status | Description |
|------|-------|--------|-------------|
| `tests/basic/test_rocm_gpu_detection.py` | P0 | **Done** | GPU detection, HIP compile, RIXL presence |
| `tests/basic/test_rocm_version_consistency.py` | P0 | **Done** | ROCm version consistency check |
| `tests/router/test_router_block_size_regression.py` | P0 | **Done** | Regression test for block_size=1 fix |
| `tests/serve/test_sglang_rocm.py` | P1 | **Done** | SGLang serve with aiter on ROCm |
| `tests/kvbm_integration/test_kvbm_rocm.py` | P1 | **Done** | KVBM with HIP kernels |
| `tests/disagg/test_mori_rdma.py` | P1 | **Done** | MoRI RDMA 2-node disagg |
| `tests/disagg/test_ionic_validation.py` | P1 | **Done** | Ionic subnet pre-flight checks |
| `tests/planner/test_fpm_relay_sglang.py` | P2 | **Done** | SGLang FPM → NATS relay |
| `tests/planner/test_planner_virtual_sglang.py` | P2 | **Done** | Planner virtual mode with SGLang |
| `tests/profiler/configs/rocm_mi355x_sglang_*.yaml` | P1 | **Done** | AMD DGDR configs (rapid + thorough) |
| `tests/fault_tolerance/deploy/templates/sglang/rocm_agg.yaml` | P2 | **Done** | AMD DGD agg template |
| `tests/fault_tolerance/deploy/templates/sglang/rocm_disagg.yaml` | P2 | **Done** | AMD DGD disagg template |

---

## Profiler Config for AMD (Example)

```yaml
# tests/profiler/configs/rocm_mi355x_sglang_rapid.yaml
model: deepseek-ai/DeepSeek-V3
backend: sglang
image: rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2
hardware:
  gpuSku: mi355x
  totalGpus: 8
  numGpusPerNode: 8
  vramMb: 294912   # 288 GB
workload:
  requestRate: 5.0
  isl: 1024
  osl: 1024
sla:
  ttft: 1000
  itl: 100
searchStrategy: rapid
features:
  planner:
    environment: virtual
    backend: sglang
    mode: agg
    enable_load_scaling: true
    ttft: 1000
    itl: 100
```

---

## Key Differences from NVIDIA Test Configs

| Aspect | NVIDIA | AMD |
|--------|--------|-----|
| GPU resource | `nvidia.com/gpu` | `amd.com/gpu` |
| GPU SKU | h100_sxm, h200_sxm, b200_sxm | mi300x, mi325x, mi355x |
| Backend | trtllm, vllm, sglang | sglang (primary), vllm (blocked) |
| Transfer backend | mooncake, nixl | **mori** |
| CUDA graph env | — | `SGLANG_AITER_MLA_PERSIST=False` |
| KV Router args | — | `--page-size 16` (block_size fix) |
| Network | InfiniBand (mlx5) | Pensando Pollara 400 (ionic) |
| Intra-node comm | NCCL | RCCL |
| FPM source | vLLM native | SGLang FPM relay (additive) |
| Container | `nvcr.io/nvidia/...` | `rocm/sgl-dev:...` |

---

## Success Criteria

| Milestone | Criteria | Target Date |
|-----------|----------|-------------|
| M1: Unit tests green | 18 Phase 1 tests pass on ROCm CI | Sprint 1 |
| M2: Core features tested | Router + SGLang serve + KVBM pass on MI355X | Sprint 2 |
| M3: Disagg automated | MoRI RDMA 2-node tests in nightly CI | Sprint 3 |
| M4: CI complete | Tier 1/2/3 pipelines operational | Sprint 4 |
| M5: InferenceX parity | `dynamo-sglang` in `amd-master.yaml` with regression tracking | Sprint 4 |
| M6: vLLM unblocked | maturin Py3.12 wheel enables vLLM test suite | Sprint 5 |
