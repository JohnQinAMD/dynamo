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

## Remaining Work

1. Build Dynamo native extension (`maturin develop`) in ROCm container
2. Multi-node RIXL UCX RDMA testing between chi2899 and chi2900
3. End-to-end vLLM serving with Dynamo frontend on MI355X
4. KVBM end-to-end test with HIP kernels
5. Performance benchmarking vs CUDA baseline

