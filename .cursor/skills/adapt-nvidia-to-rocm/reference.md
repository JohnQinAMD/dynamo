# Reference — NVIDIA to ROCm Adaptation

## API Mapping

| NVIDIA | AMD ROCm | Notes |
|--------|----------|-------|
| `nvidia-smi` | `rocm-smi` / `amd-smi` | GPU monitoring |
| `nvcc` | `hipcc` | Kernel compiler |
| `cudaMalloc` | `hipMalloc` | GPU allocation |
| `cudaMemcpy` | `hipMemcpy` | Memory transfer |
| `cudaStream_t` | `hipStream_t` | Async stream |
| `NCCL` | `RCCL` | Collective comms |
| `cudarc` (Rust) | N/A (feature-gate) | No ROCm Rust crate yet |
| `nvidia.com/gpu` | `amd.com/gpu` | K8s resource |
| `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` | GPU selection |
| `ibv_reg_dmabuf_mr` | `ibv_reg_mr` (CPU only) | RDMA registration |
| NIXL | RIXL | KV transfer library |
| Mooncake RDMA | MoRI RDMA | Disagg transfer |

## Architecture Mapping

| NVIDIA | AMD |
|--------|-----|
| H100/H200/B200 | MI300X/MI325X/MI355X |
| SXM5 | OAM |
| HBM3 (80-192GB) | HBM3e (192-288GB) |
| NVLink | Infinity Fabric |
| InfiniBand (mlx5) | Pensando Pollara 400 (ionic) |
| GPU Direct RDMA | Not available on ionic — use DRAM staging |
| TensorRT-LLM | Not applicable — use SGLang + aiter |

## Container Environment Differences

| Variable | NVIDIA | AMD |
|----------|--------|-----|
| `CUDA_HOME` | `/usr/local/cuda` | `/opt/rocm` |
| `LD_LIBRARY_PATH` | `/usr/local/cuda/lib64` | `/opt/rocm/lib` |
| Compiler | `nvcc` | `hipcc` at `/opt/rocm/bin/hipcc` |
| Arch flag | `--arch=sm_90` | `--offload-arch=gfx942` (MI300X) / `gfx950` (MI355X) |
| Profiler | `nsys` / `ncu` | `rocprof` / `omniperf` |

## Dockerfile Pattern

```dockerfile
# Additive ROCm block in Jinja2 templates
{% if device == "cuda" %}
ENV CUDA_HOME=/usr/local/cuda
{% elif device == "rocm" %}
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm/hip
ENV HSA_FORCE_FINE_GRAIN_PCIE=1
{% endif %}
```

## Cargo Feature Gate Pattern

```rust
// lib.rs — conditional compilation
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "rocm")]
pub mod hip;

// Use the appropriate backend
#[cfg(feature = "cuda")]
use crate::cuda as gpu;
#[cfg(feature = "rocm")]
use crate::hip as gpu;
```

## Performance Baselines (MI355X, DSV3 671B)

| Metric | Value |
|--------|-------|
| Standalone peak | 17.3 req/s, 1,108 tok/s (c=16) |
| KV Router gain | 4.35x TTFT at c=32 |
| KVBM multi-turn | 2.17-3.34x TTFT |
| Disagg MoRI RDMA | 7.4 req/s, 475 tok/s |
| RCCL all_reduce | 406 GB/s |
| RIXL DRAM transfer | 39.4 GB/s |

## Node Inventory

| Node | GPUs | Image Available |
|------|------|----------------|
| chi2761 | 8x MI355X | rocm/pytorch |
| chi2885 | 8x MI355X | rocm/sgl-dev |
| chi2896 | 8x MI355X | rocm/sgl-dev |
| chi2899 | 8x MI355X | rocm/sgl-dev, rocm/vllm |
| chi2900 | 8x MI355X | rocm/sgl-dev |
