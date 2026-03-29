// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// HIP port of KVBM tensor kernels for AMD Instinct GPUs (CDNA 3/4).
// Converted from cuda/tensor_kernels.cu.

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#ifndef CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER __host__ __device__
#endif

namespace {

/**
 * There are three logical tensor views involved in these kernels:
 *
 * 1. Universal blocks: contiguous buffers whose logical shape is
 *    [nh, nl, no, nt, hd]. Every "block" is a separate pointer.
 * 2. NHD/HND block stacks: `nl * no` pointers per block, each pointing
 *    to a chunk shaped either [nt, nh, hd] (NHD) or [nh, nt, hd] (HND).
 *    Stacks are arranged as `[layer][outer]`.
 * 3. Operational blocks: contiguous buffers whose logical shape is
 *    [nl, no, inner], where inner = nt * nh * hd. These are used when
 *    the consumer does not care about the split between nh/nt/hd.
 *
 * Each kernel batch-processes `num_blocks` block pairs. All pointer
 * tables are flattened on the host:
 *   • universal_ptrs  : [num_blocks]
 *   • block_ptrs      : [num_blocks * nl * no]
 *   • operational_ptrs : [num_blocks]
 *
 * All pointer-list parameters (e.g. `universal_ptrs`, `src_ptrs`) must be
 * device-accessible: allocated via hipMalloc (device memory) or
 * hipHostMalloc / hipHostRegister (pinned/registered/page-locked host memory).
 */

enum class TensorDataType : int {
  F16 = 0,
  BF16 = 1,
  F32 = 2,
  F64 = 3,
};

enum class BlockLayout : int {
  NHD = 0,
  HND = 1,
};


template <TensorDataType>
struct DTypeTraits;

template <>
struct DTypeTraits<TensorDataType::F16> {
  using type = __half;
};

template <>
struct DTypeTraits<TensorDataType::BF16> {
  using type = __hip_bfloat16;
};

template <>
struct DTypeTraits<TensorDataType::F32> {
  using type = float;
};

template <>
struct DTypeTraits<TensorDataType::F64> {
  using type = double;
};

template <typename T>
CUDA_CALLABLE_MEMBER inline T*
ptr_offset(T* base, size_t index)
{
  return base + index;
}

template <typename T>
CUDA_CALLABLE_MEMBER inline const T*
ptr_offset(const T* base, size_t index)
{
  return base + index;
}

template <BlockLayout Layout>
CUDA_CALLABLE_MEMBER inline size_t
block_inner_offset(size_t nt_idx, size_t nh_idx, size_t hd_idx, size_t nt, size_t nh, size_t hd)
{
  if constexpr (Layout == BlockLayout::NHD) {
    return ((nt_idx * nh) + nh_idx) * hd + hd_idx;
  } else {
    return ((nh_idx * nt) + nt_idx) * hd + hd_idx;
  }
}

inline int
kvbm_kernels_compute_grid_dim(size_t total_elements, int block_dim)
{
  if (total_elements == 0) {
    return 0;
  }
  size_t blocks = (total_elements + static_cast<size_t>(block_dim) - 1) / static_cast<size_t>(block_dim);
  if (blocks == 0) {
    blocks = 1;
  }
  blocks = std::min<size_t>(blocks, 65535);
  return static_cast<int>(blocks);
}

inline int
kvbm_kernels_po2_shift(size_t x)
{
  if (x == 0 || (x & (x - 1)) != 0)
    return -1;
  return __builtin_ctzll(static_cast<unsigned long long>(x));
}

template <typename T, BlockLayout Layout>
__global__ void
kvbm_kernels_block_to_universal_kernel(
    const T* const* block_chunks, T* const* universal_blocks, size_t block_stride, size_t total_per_block,
    size_t num_blocks, size_t nh, size_t nl, size_t no, size_t nt, size_t hd)
{
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t total = total_per_block * num_blocks;

  while (thread_id < total) {
    size_t block_idx = thread_id / total_per_block;
    size_t residual = thread_id % total_per_block;

    size_t tmp = residual;
    size_t hd_idx = tmp % hd;
    tmp /= hd;

    size_t nt_idx = tmp % nt;
    tmp /= nt;

    size_t no_idx = tmp % no;
    tmp /= no;

    size_t nl_idx = tmp % nl;
    tmp /= nl;

    size_t nh_idx = tmp;

    const T* const* block_base = block_chunks + block_idx * block_stride;
    const T* chunk_base = block_base[nl_idx * no + no_idx];
    size_t chunk_offset = block_inner_offset<Layout>(nt_idx, nh_idx, hd_idx, nt, nh, hd);

    T* universal_base = universal_blocks[block_idx];
    universal_base[residual] = chunk_base[chunk_offset];
    thread_id += stride;
  }
}

template <typename T, BlockLayout Layout>
__global__ void
kvbm_kernels_universal_to_block_kernel(
    const T* const* universal_blocks, T* const* block_chunks, size_t block_stride, size_t total_per_block,
    size_t num_blocks, size_t nh, size_t nl, size_t no, size_t nt, size_t hd)
{
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t total = total_per_block * num_blocks;

  while (thread_id < total) {
    size_t block_idx = thread_id / total_per_block;
    size_t residual = thread_id % total_per_block;

    size_t tmp = residual;
    size_t hd_idx = tmp % hd;
    tmp /= hd;

    size_t nt_idx = tmp % nt;
    tmp /= nt;

    size_t no_idx = tmp % no;
    tmp /= no;

    size_t nl_idx = tmp % nl;
    tmp /= nl;

    size_t nh_idx = tmp;

    T* const* block_base = const_cast<T* const*>(block_chunks + block_idx * block_stride);
    T* chunk_base = block_base[nl_idx * no + no_idx];
    size_t chunk_offset = block_inner_offset<Layout>(nt_idx, nh_idx, hd_idx, nt, nh, hd);

    const T* universal_base = universal_blocks[block_idx];
    chunk_base[chunk_offset] = universal_base[residual];
    thread_id += stride;
  }
}

template <typename T>
hipError_t
kvbm_kernels_launch_block_to_universal_impl(
    void* const* universal_ptrs, const void* const* block_ptrs, size_t num_blocks, size_t nh, size_t nl, size_t no,
    size_t nt, size_t hd, BlockLayout layout, hipStream_t stream)
{
  size_t block_stride = nl * no;
  size_t total_per_block = nh * nl * no * nt * hd;
  size_t total = total_per_block * num_blocks;
  if (total == 0) {
    return hipSuccess;
  }

  if (!block_ptrs || !universal_ptrs) {
    return hipErrorInvalidValue;
  }

  constexpr int kBlockDim = 256;
  int grid_dim = kvbm_kernels_compute_grid_dim(total, kBlockDim);
  if (grid_dim == 0) {
    return hipSuccess;
  }

  const T* const* chunks = reinterpret_cast<const T* const*>(block_ptrs);
  T* const* universal_blocks = reinterpret_cast<T* const*>(const_cast<void* const*>(universal_ptrs));

  if (layout == BlockLayout::NHD) {
    kvbm_kernels_block_to_universal_kernel<T, BlockLayout::NHD><<<grid_dim, kBlockDim, 0, stream>>>(
        chunks, universal_blocks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  } else {
    kvbm_kernels_block_to_universal_kernel<T, BlockLayout::HND><<<grid_dim, kBlockDim, 0, stream>>>(
        chunks, universal_blocks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  }

  return hipGetLastError();
}

template <typename T>
hipError_t
kvbm_kernels_launch_block_from_universal_impl(
    const void* const* universal_ptrs, void* const* block_ptrs, size_t num_blocks, size_t nh, size_t nl, size_t no,
    size_t nt, size_t hd, BlockLayout layout, hipStream_t stream)
{
  size_t block_stride = nl * no;
  size_t total_per_block = nh * nl * no * nt * hd;
  size_t total = total_per_block * num_blocks;
  if (total == 0) {
    return hipSuccess;
  }

  if (!block_ptrs || !universal_ptrs) {
    return hipErrorInvalidValue;
  }

  constexpr int kBlockDim = 256;
  int grid_dim = kvbm_kernels_compute_grid_dim(total, kBlockDim);
  if (grid_dim == 0) {
    return hipSuccess;
  }

  const T* const* universal_blocks = reinterpret_cast<const T* const*>(universal_ptrs);
  T* const* chunks = reinterpret_cast<T* const*>(const_cast<void* const*>(block_ptrs));

  if (layout == BlockLayout::NHD) {
    kvbm_kernels_universal_to_block_kernel<T, BlockLayout::NHD><<<grid_dim, kBlockDim, 0, stream>>>(
        universal_blocks, chunks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  } else {
    kvbm_kernels_universal_to_block_kernel<T, BlockLayout::HND><<<grid_dim, kBlockDim, 0, stream>>>(
        universal_blocks, chunks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  }

  return hipGetLastError();
}

}  // namespace

extern "C" hipError_t
kvbm_kernels_launch_universal_from_block(
    void* const* universal_ptrs, const void* const* block_ptrs, size_t num_blocks, size_t nh, size_t nl, size_t no,
    size_t nt, size_t hd, int dtype_value, int layout_value, hipStream_t stream)
{
  auto dtype = static_cast<TensorDataType>(dtype_value);
  auto layout = static_cast<BlockLayout>(layout_value);

  switch (dtype) {
    case TensorDataType::F16:
      return kvbm_kernels_launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::F16>::type>(
          universal_ptrs, block_ptrs, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::BF16:
      return kvbm_kernels_launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::BF16>::type>(
          universal_ptrs, block_ptrs, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F32:
      return kvbm_kernels_launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::F32>::type>(
          universal_ptrs, block_ptrs, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F64:
      return kvbm_kernels_launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::F64>::type>(
          universal_ptrs, block_ptrs, num_blocks, nh, nl, no, nt, hd, layout, stream);
    default:
      return hipErrorInvalidValue;
  }
}

extern "C" hipError_t
kvbm_kernels_launch_block_from_universal(
    const void* const* universal_ptrs, void* const* block_ptrs, size_t num_blocks, size_t nh, size_t nl, size_t no,
    size_t nt, size_t hd, int dtype_value, int layout_value, hipStream_t stream)
{
  auto dtype = static_cast<TensorDataType>(dtype_value);
  auto layout = static_cast<BlockLayout>(layout_value);

  switch (dtype) {
    case TensorDataType::F16:
      return kvbm_kernels_launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::F16>::type>(
          universal_ptrs, block_ptrs, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::BF16:
      return kvbm_kernels_launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::BF16>::type>(
          universal_ptrs, block_ptrs, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F32:
      return kvbm_kernels_launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::F32>::type>(
          universal_ptrs, block_ptrs, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F64:
      return kvbm_kernels_launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::F64>::type>(
          universal_ptrs, block_ptrs, num_blocks, nh, nl, no, nt, hd, layout, stream);
    default:
      return hipErrorInvalidValue;
  }
}

/// HIP has no equivalent of cudaMemcpyBatchAsync; always returns false.
extern "C" bool
kvbm_kernels_has_memcpy_batch_async()
{
  return false;
}

/// Controls how kvbm_kernels_memcpy_batch dispatches copies.
/// On HIP, the batch API (cudaMemcpyBatchAsync) is unavailable, so we always
/// use individual hipMemcpyAsync calls unless the caller explicitly requests
/// batch-only mode (which returns hipErrorNotSupported).
enum class MemcpyBatchMode : int {
  BatchedWithFallback = 0,
  FallbackOnly = 1,
  BatchWithoutFallback = 2,
};

extern "C" hipError_t
kvbm_kernels_memcpy_batch(
    const void* const* src_ptrs, void* const* dst_ptrs, size_t size_per_copy, size_t num_copies, int mode_value,
    hipStream_t stream)
{
  auto mode = static_cast<MemcpyBatchMode>(mode_value);

  if (num_copies == 0 || size_per_copy == 0) {
    return hipSuccess;
  }

  if (!src_ptrs || !dst_ptrs) {
    return hipErrorInvalidValue;
  }

  if (mode == MemcpyBatchMode::BatchWithoutFallback) {
    return hipErrorNotSupported;
  }

  for (size_t i = 0; i < num_copies; ++i) {
    hipError_t copy_err = hipMemcpyAsync(dst_ptrs[i], src_ptrs[i], size_per_copy, hipMemcpyDefault, stream);
    if (copy_err != hipSuccess) {
      return copy_err;
    }
  }
  return hipSuccess;
}

/// Returns false - this is the real HIP implementation, not stubs.
extern "C" bool
kvbm_kernels_is_stub_build()
{
  return false;
}

/// Vectorized memory copy kernel for arbitrary device-visible pointer pairs.
///
/// Each block handles one or more (src, dst) pairs using a grid-strided loop.
/// Per-pair alignment detection selects the widest safe vector width:
///   - 16-byte (int4) if both pointers are 16-byte aligned
///   - 8-byte  (int2) if both pointers are 8-byte aligned
///   - 4-byte  (int)  if both pointers are 4-byte aligned
///   - 1-byte fallback for any remainder
__global__ void
kvbm_kernels_vectorized_copy_kernel(void** src_ptrs, void** dst_ptrs, size_t copy_size_in_bytes, int num_pairs)
{
  int pair_id = blockIdx.x;
  int block_stride = gridDim.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  for (; pair_id < num_pairs; pair_id += block_stride) {
    char* src = static_cast<char*>(src_ptrs[pair_id]);
    char* dst = static_cast<char*>(dst_ptrs[pair_id]);

    uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
    uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);

    size_t vectorized_bytes = 0;

    if (((src_addr & 0xF) == 0) && ((dst_addr & 0xF) == 0) && (copy_size_in_bytes >= 16)) {
      size_t num_int4 = copy_size_in_bytes >> 4;
      for (size_t i = tid; i < num_int4; i += block_size) {
        reinterpret_cast<int4*>(dst)[i] = reinterpret_cast<const int4*>(src)[i];
      }
      vectorized_bytes = num_int4 << 4;
    } else if (((src_addr & 0x7) == 0) && ((dst_addr & 0x7) == 0) && (copy_size_in_bytes >= 8)) {
      size_t num_int2 = copy_size_in_bytes >> 3;
      for (size_t i = tid; i < num_int2; i += block_size) {
        reinterpret_cast<int2*>(dst)[i] = reinterpret_cast<const int2*>(src)[i];
      }
      vectorized_bytes = num_int2 << 3;
    } else if (((src_addr & 0x3) == 0) && ((dst_addr & 0x3) == 0) && (copy_size_in_bytes >= 4)) {
      size_t num_int = copy_size_in_bytes >> 2;
      for (size_t i = tid; i < num_int; i += block_size) {
        reinterpret_cast<int*>(dst)[i] = reinterpret_cast<const int*>(src)[i];
      }
      vectorized_bytes = num_int << 2;
    }

    size_t remaining = copy_size_in_bytes - vectorized_bytes;
    for (size_t i = tid; i < remaining; i += block_size) {
      dst[vectorized_bytes + i] = src[vectorized_bytes + i];
    }
  }
}

extern "C" hipError_t
kvbm_kernels_launch_vectorized_copy(
    void** src_ptrs, void** dst_ptrs, size_t copy_size_bytes, int num_pairs, hipStream_t stream)
{
  if (num_pairs == 0 || copy_size_bytes == 0) {
    return hipSuccess;
  }

  if (!src_ptrs || !dst_ptrs) {
    return hipErrorInvalidValue;
  }

  constexpr int kBlockDim = 128;
  int grid_dim = std::min(num_pairs, 65535);

  kvbm_kernels_vectorized_copy_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
      src_ptrs, dst_ptrs, copy_size_bytes, num_pairs);

  return hipGetLastError();
}
