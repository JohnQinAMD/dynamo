# Dynamo on AMD ROCm — Remaining Feature Implementation Plan

> Gap analysis comparing NVIDIA Dynamo's full feature set against the current AMD port. Each gap is a work item for implementation.

**Current state**: 4 core features (KV Router, KVBM, Disagg P/D, Planner) are working on MI355X. This plan covers everything else.

---

## Available Hardware

### Slurm Allocation

K8s 节点已通过 Slurm 占用，防止干扰：

```bash
# Job 11558 — 2 K8s nodes, 4 hours
# Nodes: chi2815, chi2878
salloc -p k8s -N 2 -w chi2815,chi2878 --job-name=dynamo-k8s-test -t 4:00:00 --no-shell
```

如果过期需要重新 salloc（从任意 chi 节点执行上面命令即可）。

### K8s 集群信息

| 项 | 值 |
|----|-----|
| K3s Master | chi2894 (216.128.149.131) |
| K3s Version | v1.34.5+k3s1 |
| Kubeconfig | `/mnt/vast/john/rocm-dynamo/dynamo/.k8s-kubeconfig.yaml` |
| Worker 节点 | chi2878 (Ready, MI355X 8 GPU, `amd.com/gpu.product-name=AMD_Instinct_MI355_OAM`) |
| GPU Operator | `kube-amd-gpu` namespace，device-plugin / metrics-exporter / node-labeller 已运行 |
| NIC Operator | `kube-amd-network` namespace，ionic NIC 已发现 (`amd.com/nic: 8`) |
| Dynamo CRDs | **未安装** — 需要先 `kubectl apply` CRD YAMLs |

**注意**: chi2815 不在 K8s 集群中（K3s agent 未运行）。chi2878 的 `amd.com/gpu: 0` 表示 GPU 当前被其他 Slurm 任务占用，需等 Slurm job 释放或 drain 该节点的 Slurm 任务后，GPU 设备插件才会报出可用 GPU。

### kubectl 使用方法

```bash
export KUBECONFIG=/mnt/vast/john/rocm-dynamo/dynamo/.k8s-kubeconfig.yaml
kubectl get nodes
kubectl get pods -A -o wide
```

### 安装 Dynamo CRDs（Task 4 前置步骤）

```bash
export KUBECONFIG=/mnt/vast/john/rocm-dynamo/dynamo/.k8s-kubeconfig.yaml
cd /mnt/vast/john/rocm-dynamo/dynamo

# Create namespace
kubectl create namespace dynamo 2>/dev/null || true

# Apply CRDs
for crd in deploy/operator/config/crd/bases/*.yaml; do
    kubectl apply --server-side -f $crd
done

# Verify
kubectl get crd | grep dynamo
```

### 计算节点（非 K8s 测试用）

| 节点 | 状态 | 用途 |
|------|------|------|
| chi2761, chi2885, chi2896 | idle (deepep-a66 partition) | 非 K8s 测试（FT, disagg, GPU serve） |

---

## Gap Analysis

### Feature Status Matrix

| # | Feature | NVIDIA Upstream | AMD Status | Gap | Priority |
|---|---------|----------------|------------|-----|----------|
| 1 | Frontend (OpenAI API) | ✅ | ✅ Working | — | — |
| 2 | KV-Cache Router | ✅ | ✅ Working (patched block_size) | — | — |
| 3 | KVBM (GPU↔CPU offload) | ✅ | ✅ Working | — | — |
| 4 | Disagg P/D (MoRI) | ✅ | ✅ Working | — | — |
| 5 | Local Planner | ✅ | ✅ Validated (virtual + K8s) | — | — |
| 6 | SGLang FPM Relay | N/A (AMD-new) | ✅ Working | — | — |
| 7 | **Global Router** | ✅ | ✅ Unit tests pass (10/10) | E2E with SGLang pending | P1 |
| 8 | **Global Planner** | ✅ | ✅ Unit tests pass (10/10) | E2E with SGLang pending | P1 |
| 9 | **Profiler / DGDR** | ✅ | ⚠️ Skipped (needs aiconfigurator) | NVIDIA internal dep | P1 |
| 10 | **Multimodal (Image)** | ✅ | ❌ Not tested | SGLang VL models on ROCm | P2 |
| 11 | **Multimodal E/P/D** | ✅ | ❌ Not tested | Encode/Prefill/Decode disagg | P2 |
| 12 | **Multimodal (Video)** | ✅ (preview) | ❌ Not tested | FastVideo / diffusion | P3 |
| 13 | **Fault Tolerance — Migration** | ✅ | ⚠️ Test runs, disagg timeout | Need longer timeout / pre-cached model | P1 |
| 14 | **Fault Tolerance — Cancellation** | ✅ | ⚠️ Agg skipped (expected), disagg timeout | Same timeout issue | P1 |
| 15 | **Fault Tolerance — etcd HA** | ✅ | ✅ Agg: 2/2 PASS, disagg: timeout | Agg path works | P1 |
| 16 | **GPU Memory Service** | ✅ | ❌ Not tested on ROCm | Sleep/wake, shadow failover | P2 |
| 17 | **Observability — Metrics** | ✅ | ✅ Prometheus exposition: 5 PASS | Validated | P2 |
| 18 | **Observability — Tracing** | ✅ | ❌ Not tested | OpenTelemetry tracing | P3 |
| 19 | **K8s Operator — DGD deploy** | ✅ | ✅ 7 CRDs + 5 DGD dry-run PASS | Need Operator for E2E | P1 |
| 20 | **K8s — DGDR auto-deploy** | ✅ | ❌ Not tested | DGDR → profiler → deploy | P2 |
| 21 | **K8s — Grove / Topology** | ✅ | ❌ Not tested | Topology-aware scheduling | P3 |
| 22 | **K8s — Inference Gateway** | ✅ | ❌ Not tested | GAIE KV-aware routing | P3 |
| 23 | **Agentic / nvext hints** | ✅ | ❌ Not tested | Priority, OSL, cache pin | P2 |
| 24 | **Tool Calling** | ✅ | ✅ Manual test passed (runbook Test 2) | Pytest pending | P2 |
| 25 | **Speculative Decoding** | ✅ | ⚠️ SGLang has it, not tested via Dynamo | EAGLE/MTP via Dynamo | P2 |
| 26 | **LoRA** | ✅ | ❌ Not tested | LoRA serving on ROCm | P3 |
| 27 | **Embedding Models** | ✅ | ❌ Not tested | Embedding API on ROCm | P2 |
| 28 | **vLLM backend E2E** | ✅ | ⚠️ Import verified, no serve test | Full vLLM serve E2E | P1 |
| 29 | **TensorRT-LLM backend** | ✅ | ❌ N/A — NVIDIA-only | Skip permanently | — |
| 30 | **KVBM SSD tier** | ✅ | ❌ Not tested | GPU→CPU→SSD offload | P3 |
| 31 | **Model Express** | ✅ | ❌ Not tested | Weight streaming via NIXL | P3 |

---

## Implementation Tasks

### P1 — Must Have (Sprint 1-2)

#### Task 1: Global Router + Global Planner on ROCm

**What**: The Global Router routes across multiple DGDs (e.g., different TP configs for the same model). The Global Planner coordinates GPU budgets across DGDs.

**Files to reference**:
- `components/src/dynamo/global_router/` — Global Router source
- `components/src/dynamo/global_planner/` — Global Planner source
- `examples/global_planner/global-planner-vllm-test.yaml` — example YAML
- `examples/global_planner/global-planner-mocker-test.yaml` — mocker-based test
- `tests/global_planner/unit/test_scale_request_handler.py` — existing unit test

**Steps**:
1. Run `test_scale_request_handler.py` on MI355X — already has `@pytest.mark.rocm`, should pass (pure logic)
2. Create `examples/global_planner/global-planner-sglang-test.yaml` — replace vLLM workers with SGLang workers, use `amd.com/gpu` resources
3. Run `global-planner-mocker-test.yaml` on MI355X — mocker is GPU-agnostic
4. Write E2E test: 2 prefill pools (TP1, TP2) + 1 decode pool, SGLang backend, verify GlobalRouter routes correctly
5. Test GPU budget enforcement: set `--max-total-gpus 16`, verify planner respects the limit

**Validation**: GlobalRouter routes to correct pool, GlobalPlanner enforces GPU budget

---

#### Task 2: Profiler / DGDR on ROCm

**What**: The Profiler automates TP/parallelism sweeps and feeds results to the Planner. DGDR (DynamoGraphDeploymentRequest) is a simplified SLA-driven YAML that auto-generates the full DGD.

**Files to reference**:
- `components/src/dynamo/profiler/` — Profiler source
- `components/src/dynamo/profiler/utils/dgdr_v1beta1_types.py` — DGDR Pydantic types
- `tests/profiler/test_profile_sla_dgdr.py` — DGDR YAML → config tests
- `tests/profiler/configs/` — existing NVIDIA DGDR YAMLs + our `rocm_mi355x_*.yaml`

**Steps**:
1. Add `mi355x` to `GPUSKUType` enum in `dgdr_v1beta1_types.py` (currently only has `h100_sxm`, `h200_sxm`, etc.)
2. Add `mi300x`, `mi325x` to the same enum
3. Run `test_profile_sla_dgdr.py` with our `rocm_mi355x_sglang_rapid.yaml` — verify YAML parses into valid `DynamoGraphDeploymentRequestSpec`
4. Run the rapid/thorough profiler helpers with MI355X hardware config
5. If profiler tries to spawn real workers, ensure SGLang launch args include `--page-size 16` and `SGLANG_AITER_MLA_PERSIST=False`

**Validation**: DGDR YAML with `gpuSku: mi355x` + `backend: sglang` produces valid profiling config

---

#### Task 3: Fault Tolerance — Migration & Cancellation

**What**: Request migration moves in-flight requests between workers when a worker is being drained. Cancellation allows clients to cancel streaming requests.

**Files to reference**:
- `tests/fault_tolerance/migration/test_sglang.py` — SGLang migration test
- `tests/fault_tolerance/cancellation/test_sglang.py` — SGLang cancel test
- `tests/fault_tolerance/etcd_ha/test_sglang.py` — etcd failover test

**Steps**:
1. Add `@pytest.mark.rocm` to `test_sglang.py` in migration, cancellation, and etcd_ha directories
2. Run each test on MI355X — these tests spawn SGLang workers internally
3. If SGLang launch fails, check for missing ROCm env vars (`SGLANG_AITER_MLA_PERSIST`, `HIP_VISIBLE_DEVICES`)
4. Verify migration: start streaming → kill worker → response completes on another worker
5. Verify cancellation: start streaming → send cancel → server stops generating

**Validation**: All 3 SGLang FT tests pass on MI355X

---

#### Task 4: K8s Operator — Full DGD Deploy E2E

**What**: Deploy a full `DynamoGraphDeployment` on a K8s cluster with AMD GPU Operator and verify end-to-end inference.

**Files to reference**:
- `tests/deploy/test_deploy.py` — existing deploy test
- `tests/deploy/conftest.py` — deployment target discovery
- `tests/fault_tolerance/deploy/templates/sglang/rocm_agg.yaml` — our AMD DGD template
- `tests/fault_tolerance/deploy/templates/sglang/rocm_disagg.yaml` — disagg template

**Steps**:
1. Verify AMD GPU Operator (`kube-amd-gpu`) is running and `amd.com/gpu` resources are schedulable
2. Create `examples/backends/sglang/deploy/agg_rocm.yaml` if not already there — DGD for aggregated SGLang on ROCm
3. Run `test_deploy.py` with `--framework sglang` targeting the ROCm deploy YAML
4. Verify: pods come up, frontend accepts requests, responses are correct
5. Test with disagg: deploy `rocm_disagg.yaml`, verify prefill and decode pods on separate nodes

**Validation**: DGD deploy → pods running → inference works → teardown clean

---

#### Task 5: vLLM Backend E2E on ROCm

**What**: Full vLLM serving test on ROCm, not just import verification.

**Files to reference**:
- `tests/serve/test_vllm.py` — upstream vLLM serve test
- `examples/backends/vllm/launch/rocm/` — ROCm vLLM launch scripts
- `scripts/build_maturin_py312.sh` — Py3.12 wheel builder

**Steps**:
1. Use `rocm/vllm:latest` container (Python 3.12)
2. Build Dynamo maturin wheel: `bash scripts/build_maturin_py312.sh`
3. Run `tests/serve/test_vllm.py` — specifically the `aggregated` config with Qwen model
4. If model load fails, ensure ROCm env vars are set
5. Add `@pytest.mark.rocm` to relevant vLLM serve test configs

**Validation**: vLLM serve test passes on MI355X with Py3.12

---

### P2 — Should Have (Sprint 3-4)

#### Task 6: Multimodal (Image) on ROCm

**What**: Serve vision-language models (e.g., Qwen-VL) through Dynamo with image inputs.

**Steps**:
1. Run `test_sglang.py::test_sglang_deployment[multimodal_agg_qwen]` on MI355X
2. If SGLang VL model loads successfully, verify image input processing
3. Create ROCm-specific multimodal test if upstream test needs adaptation
4. Test multimodal E/P/D if SGLang supports Encode/Prefill/Decode separation on ROCm

**Validation**: Image input → correct description output via Dynamo frontend

---

#### Task 7: Multimodal E/P/D Disagg on ROCm

**What**: Disaggregated multimodal serving with separate Encode, Prefill, and Decode workers.

**Steps**:
1. Check if `examples/backends/sglang/launch/multimodal_epd.sh` works on ROCm
2. If so, create `multimodal_epd_rocm.sh` with ROCm env vars
3. Run `test_sglang.py::test_sglang_deployment[multimodal_e_pd_qwen]` on MI355X
4. Verify all 3 workers (E, P, D) start and process image requests

**Validation**: Image → Encode worker → Prefill worker → Decode worker → response

---

#### Task 8: GPU Memory Service (Sleep/Wake, Shadow Failover)

**What**: GPU Memory Service manages GPU memory across workers — sleep/wake for memory reclamation, shadow failover for redundancy.

**Files to reference**:
- `tests/fault_tolerance/gpu_memory_service/test_gms_sleep_wake_sglang.py`
- `tests/fault_tolerance/gpu_memory_service/test_gms_shadow_failover_sglang.py`
- `lib/gpu_memory_service/` — includes `hip_vmm_utils.py` (AMD-additive)

**Steps**:
1. The HIP VMM utils already exist (`hip_vmm_utils.py`) — verify they load on MI355X
2. Run `test_gms_sleep_wake_sglang.py` — needs GPU + SGLang
3. Run `test_gms_shadow_failover_sglang.py`
4. If HIP VMM allocation fails, debug the `hipMemAddressReserve` / `hipMemMap` path

**Validation**: GMS sleep/wake and shadow failover work with HIP VMM on MI355X

---

#### Task 9: DGDR Auto-Deploy on K8s

**What**: Submit a DGDR YAML → Profiler runs sweeps → auto-generates DGD → deploys on K8s.

**Steps**:
1. Create DGDR YAML for MI355X + SGLang (already have `rocm_mi355x_sglang_rapid.yaml`)
2. Submit to Dynamo operator: `kubectl apply -f dgdr_mi355x.yaml`
3. Verify profiler starts TP sweep jobs
4. Verify final DGD is generated and applied
5. Verify inference works on the auto-deployed stack

**Validation**: DGDR submit → profiler sweep → DGD deploy → inference OK

---

#### Task 10: Agentic Workloads / nvext Hints

**What**: Support priority hints, output sequence length prediction, and cache pinning via `nvext` HTTP headers.

**Steps**:
1. Read `docs/features/agentic_workloads.md` for the hint format
2. Send requests with `nvext` headers to Dynamo frontend on MI355X
3. Verify priority routing (high-priority requests skip queue)
4. Verify cache pinning TTL (KV cache retained for specified duration)

**Validation**: Agentic hints affect routing/scheduling behavior

---

#### Task 11: Tool Calling via Dynamo

**What**: Tool calling (function calling) through the Dynamo frontend.

**Steps**:
1. Manual test already passed (see `amd-feature-test-runbook.md` Test 2)
2. Create pytest: send tool calling request via Dynamo → verify model returns tool call JSON
3. Add `@pytest.mark.rocm` marker

**Validation**: Tool calling pytest passes on MI355X

---

#### Task 12: Speculative Decoding via Dynamo

**What**: EAGLE / MTP / NGRAM speculative decoding through Dynamo pipeline.

**Steps**:
1. Verify SGLang on ROCm supports `--speculative-algorithm` arg
2. Launch worker with speculative decoding enabled
3. Send requests and verify speedup or at least correctness
4. Write pytest if not already covered

**Validation**: Speculative decoding works through Dynamo on MI355X

---

#### Task 13: Embedding Models

**What**: Serve embedding models through Dynamo's `/v1/embeddings` endpoint.

**Steps**:
1. Check if `test_sglang.py::test_sglang_deployment[embedding_agg]` works on ROCm
2. Run with `Qwen/Qwen3-Embedding-4B` on MI355X
3. Verify embedding vectors are returned

**Validation**: Embedding API returns valid vectors on MI355X

---

#### Task 14: Observability — Full Metrics Validation

**What**: Verify all Dynamo metrics (component, engine, system) are emitted on ROCm.

**Steps**:
1. Run `test_sglang.py::test_sglang_deployment[aggregated]` which validates metrics
2. Scrape Prometheus endpoint on MI355X worker
3. Verify `dynamo_component_*`, `sglang:*` metrics are present
4. Check `test_prometheus_exposition_format_injection.py` — already has `@pytest.mark.rocm`

**Validation**: All expected metrics present on MI355X

---

### P3 — Nice to Have (Sprint 5+)

#### Task 15: Multimodal Video / Diffusion

**Steps**: Test FastVideo / SGLang Diffusion pipeline on ROCm. Likely blocked on ROCm diffusion model support.

#### Task 16: K8s Grove / Topology-Aware Scheduling

**Steps**: Test Grove `PodCliqueSet` scheduling with `amd.com/gpu` topology annotations.

#### Task 17: K8s Inference Gateway (GAIE)

**Steps**: Test KV-aware routing via Kubernetes Gateway API on AMD.

#### Task 18: LoRA Serving

**Steps**: Test LoRA adapter loading on SGLang/vLLM with ROCm.

#### Task 19: KVBM SSD Tier

**Steps**: Test GPU→CPU→SSD KV offload path on MI355X.

#### Task 20: Model Express (Weight Streaming)

**Steps**: Test NIXL/RIXL-based weight streaming for large models.

#### Task 21: Observability — OpenTelemetry Tracing

**Steps**: Test OTLP tracing export from Dynamo workers on ROCm.

---

## Summary

| Priority | Count | Description |
|----------|-------|-------------|
| **P1** | 5 tasks | Global Router/Planner, Profiler/DGDR, Fault Tolerance, K8s Deploy, vLLM E2E |
| **P2** | 9 tasks | Multimodal, GMS, DGDR auto-deploy, Agentic, Tool Calling, Spec Decode, Embedding, Metrics |
| **P3** | 7 tasks | Video/Diffusion, Grove, GAIE, LoRA, SSD tier, Model Express, OTLP |
| **Total** | **21 tasks** | |

## How to Use This Plan

Each task is self-contained with:
- **What** the feature is
- **Files** to reference in the codebase
- **Steps** to implement/test
- **Validation** criteria for success

Start with P1 tasks in order. Most P1 tasks are "run existing upstream tests on MI355X" — the code exists, it just needs ROCm env vars and validation.
