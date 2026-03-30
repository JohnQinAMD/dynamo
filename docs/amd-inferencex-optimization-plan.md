# InferenceX AMD Benchmark Optimization Plan

> Deep analysis of all 53 benchmark scripts across 7 models x 6 GPUs.
> Identifies configuration gaps between AMD and NVIDIA, estimates performance impact,
> and provides concrete script-level patches.

## Executive Summary

After source-code-level analysis of SGLang's `aiter_backend.py` and all InferenceX
benchmark scripts, we identified **three tiers** of optimization opportunities:

| Tier | Model(s) | Issue | Est. Impact | Risk |
|------|----------|-------|-------------|------|
| **1** | Qwen 3.5 (all AMD) | Nearly zero tuning vs NV's 12+ params | **2-4x** | Low |
| **2** | GPT-OSS (all AMD) | Missing cudagraph + KV cache + max-num-seqs | **1.3-2x** | Low |
| **3** | DeepSeek R1 MI300X/MI325X | Minor prefill size gap vs MI355X | **1.1-1.3x** | Low |

**Corrected finding on SGLANG_AITER_MLA_PERSIST**: After reading the actual source code
in `aiter_backend.py` (lines 246-261), we confirmed that the persistent MLA + CUDA graph
crash has been **fixed in the code itself** for fp8 KV cache configurations. The current
InferenceX DeepSeek scripts are correctly configured — no `PERSIST=False` change is needed.

---

## Part 1: SGLANG_AITER_MLA_PERSIST — Source Code Analysis

### What the code actually does

```python
# aiter_backend.py lines 246-261
if (
    self.num_head == 16 or self.num_head == 128
) and self.kv_cache_dtype is not fp8_dtype:
    _use_mla_ps_kernel = False
    fast_mode = False
    intra_batch_mode = False
```

This means:
- **fp8 KV cache (all DeepSeek scripts)**: persistent MLA stays enabled → CUDA graph works
- **non-fp8 KV cache + TP8 (num_head=16)**: persistent MLA auto-disabled → avoids crash
- **num_head=128 (TP1 / DP attention)**: persistent MLA auto-disabled

### Why the earlier 7x regression existed

The `amd-benchmark-results.md` data (51→367 tok/s) was from SGLang v0.5.5 era, **before**
this conditional was added. The fix landed in v0.5.8 (PR #17327, referenced in
perf-changelog line 263). Since InferenceX now uses v0.5.8/v0.5.9, the persistent MLA
kernel coexists correctly with CUDA graph capture when using fp8 KV cache.

### CUDA Graph capture path verification

`init_cuda_graph_state()` (line 1176-1190) properly allocates persistent MLA metadata
buffers when `_use_mla_ps_kernel=True`:
```python
if self.use_mla and _use_mla_ps_kernel:
    # for persistent mla_decode_fwd
    (self.work_metadata, self.work_indptr, ...) = self.make_mla_decode_meta_data_buffer(...)
```

`init_forward_metadata_capture_cuda_graph()` (line 1310-1335) correctly populates
the metadata during graph capture.

**Conclusion**: No change needed for `SGLANG_AITER_MLA_PERSIST` on any DeepSeek script.
The current default (True) + fp8 KV cache is the optimal configuration.

---

## Part 2: Model-by-Model Gap Analysis

### 2.1 Qwen 3.5 — CRITICAL GAP (est. 2-4x)

This is the largest optimization opportunity. AMD Qwen scripts have **almost no tuning**.

#### Current AMD (MI355X) — 7 params total

```bash
python3 -m sglang.launch_server \
    --attention-backend triton \
    --model-path $MODEL \
    --host=0.0.0.0 --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --mem-fraction-static 0.8
```

#### NVIDIA B200 — 17+ params with deep tuning

```bash
python3 -m sglang.launch_server \
    --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
    --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 --ep-size $EP_SIZE \
    --quantization fp8 --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-dtype bfloat16 \
    --cuda-graph-max-bs $CUDA_GRAPH_MAX_BATCH_SIZE \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --mem-fraction-static $MEM_FRAC_STATIC \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --max-prefill-tokens $MAX_PREFILL_TOKENS \
    --context-length $CONTEXT_LENGTH --disable-radix-cache \
    --attention-backend trtllm_mha --moe-runner-backend flashinfer_trtllm \
    --enable-flashinfer-allreduce-fusion \
    --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
    --tokenizer-worker-num 6 --stream-interval 30
```

#### Parameter-level impact analysis

| Missing Parameter | What it does | Est. Impact | Applicable to AMD? |
|---|---|---|---|
| `--cuda-graph-max-bs` | Enables CUDA graph for decode | **1.5-2x** on decode throughput | Yes — triton backend supports cuda graph |
| `--kv-cache-dtype fp8_e4m3` | Halves KV cache memory | **1.3-1.5x** (more concurrent reqs) | Only for FP8 variant |
| `--disable-radix-cache` | Frees memory in benchmark | **1.1-1.2x** | Yes (already on MI325X/MI300X, missing MI355X) |
| `--chunked-prefill-size` | Controls prefill chunking | **1.1x** | Yes |
| `--max-prefill-tokens` | Limits prefill token budget | **1.05x** | Yes |
| `--num-continuous-decode-steps` | Batches decode steps | **1.05-1.1x** | Yes |
| `--max-running-requests` | Caps active requests | **avoids OOM** | Yes |
| `--context-length` | Limits context window | **avoids OOM** | Yes |
| `--stream-interval` | Reduces streaming overhead | **1.02x** | Yes |
| `--tokenizer-worker-num` | Parallel tokenization | **1.02x** on high conc | Yes |
| `--attention-backend aiter` | AMD's optimized attention | **1.2-1.5x** vs triton | Needs testing with Qwen |
| `--scheduler-recv-interval` | Reduce scheduler polling | **1.05x** at high conc | Yes |

**Compounded estimate**: Missing cuda-graph alone is 1.5-2x; combined with missing
KV cache optimization and other tuning, the gap is likely **2-4x**.

#### Recommended change for `qwen3.5_fp8_mi355x.sh`

```bash
export SGLANG_USE_AITER=1
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
CUDA_GRAPH_MAX_BATCH_SIZE=$CONC
MAX_RUNNING_REQUESTS=128
CONTEXT_LENGTH=$((ISL + OSL + 20))

python3 -m sglang.launch_server \
    --attention-backend triton \
    --model-path $MODEL \
    --host=0.0.0.0 --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --cuda-graph-max-bs $CUDA_GRAPH_MAX_BATCH_SIZE \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --max-prefill-tokens $MAX_PREFILL_TOKENS \
    --context-length $CONTEXT_LENGTH \
    --kv-cache-dtype fp8_e4m3 \
    --stream-interval 30 \
    $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &
```

Notes:
- Keep `triton` backend until `aiter` is verified with Qwen's Mamba-hybrid architecture
- `--kv-cache-dtype fp8_e4m3` only for FP8 variant; BF16 variant should omit this
- `--attention-backend aiter` could give additional 1.2-1.5x but needs validation
  (Qwen 3.5 has Mamba layers that may interact with aiter)

#### Same changes needed for:
- `qwen3.5_fp8_mi325x.sh` (already has --disable-radix-cache)
- `qwen3.5_fp8_mi300x.sh` (already has --disable-radix-cache)
- `qwen3.5_bf16_mi355x.sh` (omit --kv-cache-dtype)
- `qwen3.5_bf16_mi325x.sh` (omit --kv-cache-dtype)
- `qwen3.5_bf16_mi300x.sh` (omit --kv-cache-dtype)

---

### 2.2 GPT-OSS — MODERATE GAP (est. 1.3-2x)

#### Current AMD (MI355X vLLM)

```bash
vllm serve $MODEL --port $PORT \
  --attention-backend ROCM_AITER_UNIFIED_ATTN \
  -cc.pass_config.fuse_rope_kvcache=True -cc.use_inductor_graph_partition=True \
  --tensor-parallel-size=$TP \
  --gpu-memory-utilization 0.95 \
  --max-model-len $MAX_MODEL_LEN \
  --block-size=64 \
  --no-enable-prefix-caching
```

#### NVIDIA B200 (vLLM with config.yaml)

```yaml
kv-cache-dtype: fp8
compilation-config: '{"pass_config":{"fuse_allreduce_rms":true,"eliminate_noops":true}}'
no-enable-prefix-caching: true
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
```
Plus: `--max-num-seqs 512`, `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`

#### Gap analysis

| Missing on AMD | NV Setting | Est. Impact |
|---|---|---|
| `max-cudagraph-capture-size` | 2048 | **1.3-1.5x** — vLLM ROCm default is likely much smaller |
| `kv-cache-dtype` | fp8 | **1.2-1.3x** — more KV cache slots |
| `max-num-seqs` | 512 | **1.1x** at high concurrency |
| `max-num-batched-tokens` | 8192 | **1.05x** — controls decode batch efficiency |
| `fuse_allreduce_rms` | true | **1.05x** — already has `fuse_rope_kvcache` |

#### Recommended additions to `gptoss_fp4_mi355x.sh`

```bash
# Add to existing config
cat > config.yaml << EOF
no-enable-prefix-caching: true
max-num-batched-tokens: 8192
EOF

vllm serve $MODEL --port $PORT \
  $ATTN_BACKEND $FUSE_ROPE_KVCACHE \
  --config config.yaml \
  --tensor-parallel-size=$TP \
  --gpu-memory-utilization 0.95 \
  --max-model-len $MAX_MODEL_LEN \
  --block-size=64 \
  --max-num-seqs 512 \
  --kv-cache-dtype fp8 > $SERVER_LOG 2>&1 &
```

Note: `max-cudagraph-capture-size` requires testing on ROCm vLLM — the cudagraph
implementation may differ from CUDA. The AMD AITER unified attention + inductor graph
partition already provides some of the cudagraph benefit.

#### Same changes needed for:
- `gptoss_fp4_mi325x.sh`
- `gptoss_fp4_mi300x.sh`

---

### 2.3 DeepSeek R1 — MINOR GAPS (est. 1.1-1.3x on MI300X/MI325X)

DeepSeek scripts are the most well-tuned across all AMD models. The primary gap
is between MI300X/MI325X vs MI355X configurations.

#### MI355X (v0.5.9) vs MI300X/MI325X (v0.5.8) differences

| Parameter | MI355X | MI300X/MI325X | Better |
|---|---|---|---|
| SGLang version | v0.5.9 | v0.5.8 | MI355X |
| chunked-prefill-size | 196608 | 131072 | Depends on workload |
| max-prefill-tokens | 196608 | 131072 | Depends on workload |
| cuda-graph-max-bs | `$CONC` (dynamic) | 128 (fixed) | MI355X (adapts to workload) |
| RCCL_MSCCL_ENABLE | 0 | not set | MI355X |
| ROCM_QUICK_REDUCE_QUANTIZATION | INT4 | not set | MI355X |

#### Recommended changes for MI300X/MI325X

```bash
# Add to dsr1_fp8_mi300x.sh and dsr1_fp8_mi325x.sh:
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
# Change cuda-graph-max-bs from 128 to $CONC:
--cuda-graph-max-bs=$CONC
```

These are low-risk changes that align with the better-tuned MI355X config.

---

### 2.4 MiniMax M2.5 — MINIMAL GAP (<10%)

Both AMD and NVIDIA use vLLM with default attention backend and minimal tuning.
The scripts are reasonably equivalent. No urgent changes needed.

Only minor observation: NVIDIA B200 explicitly disables some MOE optimizations
(`VLLM_USE_FLASHINFER_MOE_FP8=0`, `VLLM_MOE_USE_DEEP_GEMM=0`) that don't work
well. AMD doesn't need these since it uses AITER MoE kernels.

---

### 2.5 Kimi K2.5 — SMALL GAP (est. 1.1-1.2x)

Recent updates (PRs #950, #957, #936) added AITER MLA support and triton fused MoE
tuning. The scripts are now reasonably well-optimized.

Remaining gap: NVIDIA B200 sets `VLLM_USE_FLASHINFER_MOE_INT4=1` for INT4 MoE
acceleration. AMD uses triton fused MoE which is already competitive.

---

### 2.6 GLM-5 — ON PAR

AMD MI355X uses tilelang NSA backends, NVIDIA B200 uses trtllm NSA backends.
Both are platform-specific optimal choices. Configuration is comparable.

---

## Part 3: Cross-Cutting Observations

### 3.1 CUDA Graph coverage

| Model | AMD CUDA Graph | NVIDIA CUDA Graph |
|---|---|---|
| DeepSeek R1 | `$CONC` or 128 | 128-512 |
| Qwen 3.5 | **NOT SET** | `$CONC` |
| GPT-OSS | **NOT SET** | 2048 |
| MiniMax | not set | not set (vLLM default) |
| Kimi K2.5 | not set | not set (vLLM default) |
| GLM-5 | not set | `$CONC` |

**Qwen and GPT-OSS are the biggest cuda-graph gaps on AMD.**

### 3.2 KV Cache dtype coverage

| Model | AMD KV Cache | NVIDIA KV Cache |
|---|---|---|
| DeepSeek R1 | fp8_e4m3 | fp8_e4m3 / fp8 |
| Qwen 3.5 FP8 | **NOT SET** | fp8_e4m3 |
| GPT-OSS | **NOT SET** | fp8 (B200) |
| MiniMax | not set | not set |
| Kimi K2.5 | not set | not set |
| GLM-5 | not set (MI355X) | fp8_e4m3 (B200) |

**Qwen FP8 and GPT-OSS are missing fp8 KV cache on AMD.**

### 3.3 Scheduler tuning

NVIDIA scripts frequently use `--scheduler-recv-interval` (10 for low conc, 30 for
high conc) and `--max-running-requests`. AMD scripts mostly omit these. This matters
at high concurrency (>16) where scheduler overhead becomes significant.

---

## Part 4: Implementation Roadmap

### Phase 1: Quick Wins (1-2 days, no validation needed)

| Script | Change | Risk |
|---|---|---|
| `qwen3.5_fp8_mi355x.sh` | Add `--disable-radix-cache` | None — already on MI325X/MI300X |
| `qwen3.5_bf16_mi355x.sh` | Add `--disable-radix-cache` | None |
| `dsr1_fp8_mi300x.sh` | Add `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | None — already on MI355X |
| `dsr1_fp8_mi325x.sh` | Add `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | None — already on MI355X |
| `dsr1_fp8_mi300x.sh` | Change `--cuda-graph-max-bs=128` to `$CONC` | Low — matches MI355X |
| `dsr1_fp8_mi325x.sh` | Change `--cuda-graph-max-bs=128` to `$CONC` | Low — matches MI355X |

### Phase 2: Medium-Risk Optimizations (needs A/B testing)

| Script | Change | Validation |
|---|---|---|
| All Qwen 3.5 AMD | Add `--cuda-graph-max-bs $CONC` | Run 1k1k c=16 before/after |
| All Qwen 3.5 FP8 AMD | Add `--kv-cache-dtype fp8_e4m3` | Check accuracy + throughput |
| All Qwen 3.5 AMD | Add `--chunked-prefill-size 32768 --max-prefill-tokens 32768` | Run 8k1k benchmark |
| All Qwen 3.5 AMD | Add `--max-running-requests 128 --context-length $((ISL+OSL+20))` | Memory stability test |
| All GPT-OSS AMD | Add `--kv-cache-dtype fp8 --max-num-seqs 512` | Check accuracy + throughput |

### Phase 3: Exploratory (needs engineering investigation)

| Script | Change | Notes |
|---|---|---|
| Qwen 3.5 AMD | Switch `--attention-backend triton` → `aiter` | Qwen's Mamba-hybrid may not be compatible with aiter |
| GPT-OSS AMD | Test `max-cudagraph-capture-size` in vLLM ROCm | ROCm vLLM cudagraph maturity unclear |
| All AMD SGLang | Add `--scheduler-recv-interval` tuning | Marginal gain but easy to add |
| Qwen 3.5 AMD | Add `--stream-interval 30` | Reduces SSE overhead |

---

## Part 5: Risk Assessment

### Why `SGLANG_AITER_MLA_PERSIST` should NOT be changed

The earlier recommendation to set `SGLANG_AITER_MLA_PERSIST=False` was based on
benchmarks from the v0.5.5 era. The source code (v0.5.9) now has an auto-detection
mechanism:

1. **fp8 KV cache + TP8 (num_head=16)**: auto-enables persistent MLA with `fast_mode=True`
2. **non-fp8 KV cache + TP8**: auto-disables persistent MLA (the old crash scenario)
3. **TP1 or DP attention (num_head=128)**: auto-disables persistent MLA

Since all InferenceX DeepSeek scripts use `--kv-cache-dtype fp8_e4m3`, persistent MLA
is correctly enabled. Setting it to False would **disable** the persistent kernel and
likely **reduce** decode throughput by losing the occupancy benefits of the persistent
design.

### Safe vs Risky changes

| Safety Level | Changes |
|---|---|
| **Safe** (no regression possible) | `--disable-radix-cache`, `ROCM_QUICK_REDUCE_QUANTIZATION`, `--stream-interval` |
| **Low risk** (matches proven config) | `--cuda-graph-max-bs $CONC` on MI300X/MI325X (matches MI355X) |
| **Medium risk** (needs validation) | Adding `--cuda-graph-max-bs` to Qwen, `--kv-cache-dtype` to Qwen/GPT-OSS |
| **High risk** (may break) | Switching Qwen from `triton` to `aiter` backend |

---

## Appendix: Complete Parameter Comparison Matrix

### DeepSeek R1 SGLang: AMD vs NVIDIA

| Parameter | MI355X FP8 | MI300X FP8 | MI325X FP8 | B200 FP8 | H200 FP8 |
|---|---|---|---|---|---|
| attention-backend | aiter | aiter | aiter | trtllm_mla | flashinfer |
| kv-cache-dtype | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 | (default) |
| cuda-graph-max-bs | $CONC | 128 | 128 | 128/32 | 256/512 |
| chunked-prefill-size | 196608 | 131072 | 131072 | 32768 | 32768 |
| max-prefill-tokens | 196608 | 131072 | 131072 | 32768 | 32768 |
| mem-fraction-static | 0.8 | 0.8 | 0.8 | 0.82 | 0.82 |
| disable-radix-cache | yes | yes | yes | yes | yes |
| decode-steps | 4 | 4 | 4 | (default) | (default) |
| quantization | (default) | (default) | (default) | fp8 | (default) |
| ep-size | (default) | (default) | (default) | $EP_SIZE | (default) |
| scheduler-recv | (default) | (default) | (default) | 10/30 | (default) |
| MLA_PERSIST | default(T) | explicit(1) | explicit(1) | N/A | N/A |
| QUICK_REDUCE | INT4 | not set | not set | N/A | N/A |

### Qwen 3.5 SGLang: AMD vs NVIDIA

| Parameter | MI355X FP8 | MI325X FP8 | B200 FP8 | H200 FP8 |
|---|---|---|---|---|
| attention-backend | triton | triton | trtllm_mha | flashinfer |
| kv-cache-dtype | **not set** | **not set** | fp8_e4m3 | fp8_e4m3 |
| cuda-graph-max-bs | **not set** | **not set** | $CONC | $CONC |
| quantization | **not set** | **not set** | fp8 | fp8 |
| chunked-prefill | **not set** | **not set** | 32768 | 16384 |
| max-prefill-tokens | **not set** | **not set** | 32768 | (default) |
| max-running-requests | **not set** | **not set** | 128 | 128 |
| disable-radix-cache | **no** (MI355X) | yes | yes | yes |
| scheduler-recv | **not set** | **not set** | 10/30 | (default) |
| context-length | **not set** | **not set** | ISL+OSL+20 | ISL+OSL+20 |
| stream-interval | **not set** | **not set** | 30 | 50 |
| tokenizer-workers | **not set** | **not set** | 6 | 6 |
| ep-size | **not set** | **not set** | $EP_SIZE | $EP_SIZE |

### GPT-OSS vLLM: AMD vs NVIDIA

| Parameter | MI355X | B200 | H200 |
|---|---|---|---|
| attention-backend | ROCM_AITER_UNIFIED_ATTN | (default) | (default) |
| kv-cache-dtype | **not set** | fp8 | **not set** |
| cudagraph-capture-size | **not set** | 2048 | 2048 |
| max-num-seqs | **not set** | 512 | $CONC |
| max-num-batched-tokens | **not set** | 8192 | 8192 |
| gpu-memory-util | 0.95 | 0.9 | 0.9 |
| block-size | 64 | (default) | (default) |
| compilation passes | fuse_rope_kvcache | fuse_allreduce_rms | (default) |
