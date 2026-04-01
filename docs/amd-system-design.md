# Dynamo on AMD ROCm — System Design

> NVIDIA Dynamo adapted for AMD Instinct MI355X. All changes are additive (🆕 new / 🔧 patched) — the upstream codebase is untouched.

---

## 1. System Architecture

The system follows Dynamo's standard Frontend → Router → Workers architecture, with AMD-specific additions highlighted.

```mermaid
flowchart TB
    Client(("Client"))
    Client -->|"HTTP /v1/chat/completions"| Frontend

    subgraph Frontend["Frontend – Rust / Axum"]
        direction LR
        HTTP["HTTP Server"] --> Pre["Preprocessor"] --> RT["Router"]
    end

    subgraph RouterBox[" "]
        direction TB
        RR["Round-Robin"]
        KV["🔧 KV Router<br>block_size fix"]
    end

    RT --> RouterBox
    RouterBox -->|"TCP"| NodeA
    RouterBox -->|"TCP"| NodeB

    subgraph Infra["  Discovery  "]
        ETCD[("etcd")] ~~~ NATS[("NATS<br>JetStream")]
    end
    RouterBox -.-|"lookup"| ETCD
    NodeA -.-|"KV events"| NATS
    NodeB -.-|"KV events"| NATS

    subgraph NodeA["Worker Node A – 8x MI355X"]
        WA["dynamo.sglang<br>SGLang + aiter TP8 FP8"]
        KVBM_A["KVBM<br>GPU → CPU offload"]
        FPM_A["🆕 FPM relay"]
    end

    subgraph NodeB["Worker Node B – 8x MI355X"]
        WB["dynamo.sglang<br>SGLang + aiter TP8 FP8"]
        KVBM_B["KVBM<br>GPU → CPU offload"]
        FPM_B["🆕 FPM relay"]
    end

    subgraph Planner["Dynamic Planner"]
        direction LR
        Prom["Prometheus<br>metrics"] ~~~ FPM_SUB["🆕 FPM<br>subscriber"]
    end

    FPM_A -.->|"NATS"| FPM_SUB
    FPM_B -.->|"NATS"| FPM_SUB
```

---

## 2. AMD Additive Changes

Every change lives on the `amd-dynamo` branch. Nothing is removed from upstream.

```mermaid
flowchart LR
    subgraph UP["Upstream Dynamo"]
        direction TB
        u1["Frontend"]
        u2["KV Router"]
        u3["KVBM CUDA"]
        u4["Disagg mooncake"]
        u5["FPM vLLM-only"]
        u6["Planner"]
    end

    subgraph AMD["AMD Additive – additive changes only"]
        direction TB
        a1["🔧 block_size fix<br>multi_worker.rs"]
        a2["🆕 HIP kernels<br>tensor_kernels.hip"]
        a3["🆕 MoRI backend<br>replaces mooncake"]
        a4["🆕 SGLang FPM relay<br>fpm_relay.py"]
        a5["🆕 GPU HAL + VMM<br>hip.rs modules"]
        a6["🆕 K8s + Docker<br>amd.com/gpu"]
    end

    u2 -.->|patch| a1
    u3 -.->|port| a2
    u4 -.->|replace| a3
    u5 -.->|extend| a4
```

| Layer | What Changed | Type |
|:------|:-------------|:----:|
| KV Router | `multi_worker.rs` — graceful default when `block_size ≤ 1` | 🔧 |
| KVBM Kernels | `tensor_kernels.hip` + HIP build path | 🆕 |
| Disagg Transfer | `--disaggregation-transfer-backend mori` (replaces mooncake) | 🆕 |
| RIXL DRAM Staging | `nixl_rocm_staging.py` — monkey-patch NixlKVManager for DRAM bounce | 🆕 |
| Mooncake ROCm Patch | `mooncake_rocm_rdma.patch` — GPU/CPU MR detection + ionic max_sge | 🆕 |
| Planner FPM | `fpm_relay.py` — SGLang KvMetrics → ForwardPassMetrics | 🆕 |
| GPU Memory | HIP VMM facade, `hip.rs` HAL modules (6 files) | 🆕 |
| Containers | ROCm Dockerfile blocks, `context.yaml` | 🆕 |
| K8s / Helm | AMD GPU discovery, `amd.com/gpu` resource | 🆕 |
| Python | Lazy imports for `nixl`, `OmniConfig`, `typing.Self` | 🔧 |
| CI | ROCm build workflow, pre-commit hook | 🆕 |

---

## 3. Disaggregated Serving with MoRI RDMA

Prefill and decode run on separate nodes. KV cache is transferred over RDMA via AMD's MoRI library through Pensando Pollara 400 ionic NICs.

```mermaid
flowchart LR
    C(("Client")) --> FE["Frontend"]

    subgraph P["Prefill Node – 8x MI355X"]
        PW["Prefill TP8"]
        P_NIC["🆕 ionic₀<br>subnet :0148"]
        PW --> P_NIC
    end

    subgraph D["Decode Node – 8x MI355X"]
        D_NIC["🆕 ionic₁<br>subnet :0148"]
        DW["Decode TP8"]
        D_NIC --> DW
    end

    FE -->|"①  route"| PW
    P_NIC ===|"② 🆕 MoRI RDMA 400Gb/s"| D_NIC
    DW -->|"③  stream tokens"| FE
    FE --> C
```

### Transfer Backend Matrix

| Backend | Status | Performance | Issue |
|:--------|:------:|:------------|:------|
| **🆕 MoRI RDMA** | ✅ | **106.6 req/s** (Qwen) · **7.4 req/s** (DSV3) | — |
| **🆕 RIXL + DRAM Staging** | ✅ | RDMA via pinned host bounce | Monkey-patch, zero SGLang changes |
| **🆕 Mooncake + ROCm patch** | ✅ | — | `bash scripts/patch_mooncake_rocm.sh` to enable |
| Mooncake RDMA (unpatched) | ❌ | — | `ibv_reg_mr ENOMEM` — no GDR on ionic |
| Mooncake TCP | ⚠️ | 76.2 req/s (Qwen only) | DSV3 crashes |
| RIXL / nixl (unpatched) | ❌ | — | VRAM registration fails |

### Ionic Subnet Matching

> **Critical**: ionic device numbers are **not** consistent across nodes. Always verify subnets via GID tables.

```mermaid
flowchart LR
    subgraph A["Node A"]
        direction TB
        a0["ionic_0 :014e"]
        a1["ionic_1 :0150"]
        a2["ionic_2 :0148 ✅"]
    end

    subgraph B["Node B"]
        direction TB
        b0["ionic_0 :0148 ✅"]
        b1["ionic_1 :0146"]
        b2["ionic_2 :0147"]
    end

    a0 -.-x|"❌ mismatch"| b0
    a2 ===|"✅ same subnet"| b0
```

Find matching pairs:

```bash
# On each node — check subnet of every ionic device
for i in 0 1 2 3 4 5 6 7; do
  gid=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1)
  echo "ionic_$i  $(echo $gid | cut -d: -f1-4)"
done
```

---

## 3b. RIXL DRAM Staging (Plan B)

When using RIXL/nixl instead of MoRI, GPU VRAM cannot be registered with ionic NICs. The `nixl_rocm_staging.py` monkey-patch solves this at runtime without modifying SGLang source.

```mermaid
flowchart LR
    subgraph Prefill["Prefill Node"]
        PG["GPU KV Cache"]
        PD["🆕 Pinned DRAM<br>staging buffer"]
        PN["ionic NIC"]
        PG -->|"hipMemcpy D2H"| PD
        PD -->|"ibv_reg_mr OK"| PN
    end

    subgraph Decode["Decode Node"]
        DN["ionic NIC"]
        DD["🆕 Pinned DRAM<br>staging buffer"]
        DG["GPU KV Cache"]
        DN -->|"RDMA WRITE"| DD
        DD -->|"hipMemcpy H2D"| DG
    end

    PN ===|"🆕 RIXL RDMA 400Gb/s"| DN
```

**Key design**: wrap `agent.get_xfer_descs` once at init → all 3 transfer methods (`_send_kvcache_generic`, `send_kvcache_slice`, `_send_mamba_state`) are automatically patched. SGLang `conn.py` stays pristine.

Enable: `export SGLANG_NIXL_ROCM_STAGING=1` or auto-detected on ROCm.

## 3c. Mooncake ROCm Patch (Plan A)

The C++ patch (`patches/mooncake_rocm_rdma.patch`) modifies `registerMemoryRegionInternal()`:

```mermaid
flowchart TB
    REG["registerMemoryRegionInternal()"]
    REG --> HIP{"#if USE_HIP"}
    HIP -->|GPU memory| TRY["ibv_reg_mr(GPU)"]
    TRY -->|ENOMEM| ERR["return ERR_CONTEXT<br>+ warning: use DRAM staging"]
    TRY -->|success| OK["GPU Direct RDMA ✅"]
    HIP -->|CPU memory| CPU["ibv_reg_mr(CPU) ✅"]
    REG --> CUDA{"#elif USE_CUDA"}
    CUDA --> DMABUF["ibv_reg_dmabuf_mr"]
```

Also patches `config.cpp`: auto-detects Pensando ionic (`vendor_id=0x1dd8`) → `max_sge=2`.

---

## 4. Component Stack

```mermaid
flowchart TB
    subgraph L1["Application"]
        direction LR
        A1["dynamo.frontend"]
        A2["dynamo.sglang"]
        A3["dynamo.planner"]
    end

    subgraph L2["Dynamo Runtime – Rust"]
        direction LR
        B1["Discovery<br>etcd"]
        B2["🔧 Router<br>KV-RR"]
        B3["Pipeline<br>TCP NATS"]
    end

    subgraph L3["Inference Engine"]
        direction LR
        C1["SGLang"]
        C2["aiter<br>MLA MoE"]
        C3["RCCL"]
        C4["🆕 MoRI<br>RDMA"]
    end

    subgraph L4["Hardware"]
        direction LR
        D1["🆕 MI355X<br>CDNA 4"]
        D2["HBM3e<br>288 GB"]
        D3["🆕 Pollara 400<br>ionic RDMA"]
    end

    L1 --> L2 --> L3 --> L4
```

---

## 5. SGLang FPM Relay

Upstream Dynamo only supports Forward Pass Metrics (FPM) for vLLM. The 🆕 `SglangFpmRelay` bridges SGLang scheduler metrics to the same event plane, enabling the Dynamic Planner to auto-scale SGLang workers.

```mermaid
sequenceDiagram
    participant S as SGLang Scheduler
    participant P as Publisher (parent)
    participant R as 🆕 SglangFpmRelay
    participant N as NATS
    participant PL as Planner

    S ->> P : KvMetrics  (ZMQ IPC)
    P ->> R : on_kv_metrics()
    R ->> R : Convert → ForwardPassMetrics
    R ->> N : ZMQ PUB → FpmEventRelay → NATS
    N ->> PL : FpmEventSubscriber
    PL ->> PL : Scale up / down decision
```

Enable: `export DYN_FORWARDPASS_METRIC_PORT=20380`

---

## 6. Bug Fixes & Performance Impact

| # | Bug | Layer | Fix | Before → After |
|:-:|:----|:------|:----|:---------------|
| 1 | CUDA graph conflict | aiter MLA kernel | `SGLANG_AITER_MLA_PERSIST=False` | 7,544 → **687 ms** TTFT  (**11×**) |
| 2 | KV Router panic | Rust runtime | Default `block_size` to 16 | Crash → **4.35× TTFT** at c=32 |
| 3 | Mooncake on AMD | Transfer backend | Switch to MoRI | Blocked → **7.4 req/s** DSV3 |
| 4 | Ionic ABI mismatch | Container driver | Mount host `libionic1` | No RDMA → **106.6 req/s** |
| 5 | `typing.Self` | Python 3.10 | Conditional import | Import crash → OK |
| 6 | `OmniConfig` | Python import | Lazy import | Startup crash → OK |

---

## 7. Kubernetes Deployment

```mermaid
flowchart TB
    subgraph K8s["K8s Cluster – K3s – 8 nodes – 64 GPUs"]
        subgraph ns_dynamo["🆕 namespace: dynamo"]
            E["etcd"] ~~~ N["NATS -js"] ~~~ PL["Planner"]
            CRD["7 CRDs"]
        end

        subgraph ns_gpu["namespace: kube-amd-gpu"]
            OP["AMD GPU Operator"]
        end

        W1["Worker – 8x MI355X"]
        W2["Worker – 8x MI355X"]
        W3["Worker – 8x MI355X"]
    end

    OP -.->|"amd.com/gpu: 8"| W1 & W2 & W3
```

---

## 8. Request Lifecycle

### Aggregated Mode

```mermaid
sequenceDiagram
    actor C as Client
    participant F as Frontend
    participant R as Router
    participant W as Worker

    C ->> F : POST /v1/chat/completions
    F ->> F : Template · Tokenize
    F ->> R : Route

    alt KV Router
        R ->> R : Prefix hash → best worker
    else Round-Robin
        R ->> R : Next worker
    end

    R ->> W : TCP plane
    W ->> W : Prefill → Decode
    W ->> F : Stream tokens
    F ->> C : SSE
```

### Disaggregated Mode

```mermaid
sequenceDiagram
    actor C as Client
    participant F as Frontend
    participant P as Prefill Node
    participant D as Decode Node

    C ->> F : Request
    F ->> P : ① Route to prefill
    P ->> P : Compute KV (TP=8)
    P ->> D : ② 🆕 MoRI RDMA transfer
    D ->> D : Decode tokens (TP=8)
    D ->> F : ③ Stream response
    F ->> C : SSE

    note over P, D : Requires matched ionic subnets + QoS / DCQCN
```
