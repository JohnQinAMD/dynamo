# AMD ROCm Integration Test Results

**Date**: 2026-03-29
**Hardware**: AMD Instinct MI355X (gfx950, CDNA 4)
**ROCm**: 7.2.1 (via rocm/pytorch:latest container)
**PyTorch**: 2.9.1+rocm7.2.1

## Test Summary


| #   | Test                               | Result | Notes                           |
| --- | ---------------------------------- | ------ | ------------------------------- |
| 1   | GPU Detection (MI355X)             | PASS   | 8x MI355X detected              |
| 2   | HIP Compiler (hipcc)               | PASS   | HIP 7.2, AMD clang 22.0         |
| 3   | HIP Kernel Compile (gfx950)        | PASS   | tensor_kernels.hip compiled     |
| 4   | HIP Kernel Compile (gfx942+gfx950) | PASS   | Dual-arch support               |
| 5   | Shared Library Link (-lamdhip64)   | PASS   | libkvbm_kernels.so linked       |
| 6   | Symbol Verification                | PASS   | 7 exported symbols              |
| 7   | RIXL Libraries Installed           | PASS   | libnixl.so + 3 plugins          |
| 8   | PyTorch GPU Compute                | PASS   | FP16 matmul on MI355X           |
| 9   | Dynamo gpu_utils                   | FAIL*  | dynamo.llm module not installed |
| 10  | GPU Memory Service VMM Utils       | FAIL*  | cuda.bindings not in container  |
| 11  | HIP Runtime FP16 GEMM              | PASS   | 1024x1024 FP16 GEMM verified    |


 Tests 9-10 fail due to missing pip packages (expected in base pytorch image, not a code bug)

## Additional Tests


| Test                           | Result | Notes                                         |
| ------------------------------ | ------ | --------------------------------------------- |
| RIXL Rust bindings (nixl-sys)  | PASS   | cargo check with NIXL_PREFIX pointing to RIXL |
| PyTorch FP16 Transformer Block | PASS   | 0.2ms/iteration on MI355X                     |
| Upstream Rebase Test           | PASS   | 99.6% additive, trivial conflicts only        |


## Build Artifacts Verified

- `lib/kvbm-kernels/hip/tensor_kernels.hip` → compiles to .o and .so
- RIXL install at `/opt/rocm/rixl/` → 10 .so libraries, 8 headers, 3 plugins
- UCX built with `--with-rocm=/opt/rocm` → GPU Direct RDMA ready

## vLLM ROCm Integration (Phase 6)

| Test | Result | Notes |
|------|--------|-------|
| vLLM 0.18.0 pip install in rocm/pytorch | PASS | Full install with all dependencies |
| vLLM import check | PASS | `import vllm` succeeds |
| Dynamo Python import | PARTIAL | Needs `maturin develop` for `dynamo._core` native extension |

## MI355X Performance Benchmarks (Phase 10)

Measured on AMD Instinct MI355X (gfx950), PyTorch 2.9.1+rocm7.2.1, single GPU.

| Benchmark | Size | Result |
|-----------|------|--------|
| FP16 GEMM | [4096x4096]*[4096x4096] | **1248.5 TFLOPS** |
| FP16 GEMM | [2048x4096]*[4096x4096] | 651.2 TFLOPS |
| FP16 GEMM | [512x4096]*[4096x4096] | 514.0 TFLOPS |
| BF16 GEMM | [2048x4096]*[4096x4096] | 852.6 TFLOPS |
| MHA Decode | seq=128, 32 heads | 0.034 ms |
| MHA Decode | seq=2048, 32 heads | 0.034 ms |
| MHA Decode | seq=8192, 32 heads | 0.044 ms |
| KV Block Copy | 100 blocks (839MB) | 0.485 ms |

## 2-Node Testing (Phase 7)

| Test | Result | Notes |
|------|--------|-------|
| Network connectivity (chi2899-chi2900) | PASS | 0.088ms RTT |
| RDMA devices (Pensando ionic) | PASS | PORT_ACTIVE on both nodes |
| GPU memory (per node) | PASS | 309.2GB HBM, 308GB free |
| RCCL all_reduce (2-node) | BLOCKED | Hangs on init — needs RCCL env config (NCCL_SOCKET_IFNAME, firewall rules) |
| RIXL nixlbench (2-node) | BLOCKED | nixlbench has HIP compile issues with CU_MEM_HANDLE_TYPE_FABRIC |

## Remaining Work

1. Configure RCCL for multi-node: set `NCCL_SOCKET_IFNAME`, open firewall ports
2. Fix nixlbench HIP compilation (CU_MEM_HANDLE_TYPE_FABRIC symbol)
3. Build Dynamo native extension (`maturin develop`) in ROCm container
4. End-to-end vLLM serving with Dynamo frontend
5. RIXL UCX RDMA KV cache transfer between nodes

