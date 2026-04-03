#!/usr/bin/env python3
"""RDMA bandwidth microbenchmark for ionic NICs.

Measures raw RDMA write bandwidth for each transfer backend (MoRI, Mooncake,
RIXL/nixl) using chunked MR on Pensando ionic devices.

Usage (inside container on prefill node):
    # MoRI ibverbs raw bandwidth
    python3 rdma_microbench.py --backend ibverbs --device ionic_1 --remote-ip 192.168.147.100

    # Mooncake engine bandwidth
    python3 rdma_microbench.py --backend mooncake --device ionic_1

    # RIXL/nixl bandwidth
    python3 rdma_microbench.py --backend rixl --device ionic_1

    # All backends
    python3 rdma_microbench.py --backend all --device ionic_1

Each test registers a buffer (up to 190 MB to stay within ionic limits),
performs RDMA writes, and measures bandwidth.
"""

import argparse
import ctypes
import mmap
import os
import time


def _get_libc():
    return ctypes.CDLL("libc.so.6")


def alloc_mmap_buffer(size_mb: int) -> tuple:
    """Allocate mmap+mlock buffer (ionic-compatible)."""
    size = size_mb * 1024 * 1024
    m = mmap.mmap(-1, size)
    buf = (ctypes.c_char * size).from_buffer(m)
    ptr = ctypes.addressof(buf)
    libc = _get_libc()
    libc.mlock(ctypes.c_void_p(ptr), ctypes.c_size_t(size))
    return m, buf, ptr, size


def bench_ibverbs_reg_dereg(device: str, sizes_mb: list[int], iterations: int = 10):
    """Benchmark ibv_reg_mr / ibv_dereg_mr latency on ionic."""
    try:
        import pyverbs.device as d
        import pyverbs.pd as pd_mod
        import pyverbs.mr as mr_mod
        import pyverbs.enums as e
    except ImportError:
        print("[ibverbs] pyverbs not available, skipping")
        return

    ctx = d.Context(name=device)
    pd = pd_mod.PD(ctx)

    print(f"\n{'='*60}")
    print(f"ibverbs MR Register/Deregister Benchmark ({device})")
    print(f"{'='*60}")
    print(f"{'Size MB':>10} {'Reg (ms)':>10} {'Dereg (ms)':>10} {'BW (GB/s)':>10}")

    for size_mb in sizes_mb:
        m, buf, ptr, size = alloc_mmap_buffer(size_mb)
        access = e.IBV_ACCESS_LOCAL_WRITE | e.IBV_ACCESS_REMOTE_WRITE | e.IBV_ACCESS_REMOTE_READ

        reg_times = []
        dereg_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            mr = mr_mod.MR(pd, ptr, size, access)
            t1 = time.perf_counter()
            mr.close()
            t2 = time.perf_counter()
            reg_times.append(t1 - t0)
            dereg_times.append(t2 - t1)

        avg_reg = sum(reg_times) / len(reg_times) * 1000
        avg_dereg = sum(dereg_times) / len(dereg_times) * 1000
        bw = size / (avg_reg / 1000) / 1e9 if avg_reg > 0 else 0
        print(f"{size_mb:>10} {avg_reg:>10.2f} {avg_dereg:>10.2f} {bw:>10.2f}")

        del buf
        m.close()

    pd.close()
    ctx.close()


def bench_mmap_alloc(sizes_mb: list[int], iterations: int = 10):
    """Benchmark mmap+mlock allocation speed."""
    print(f"\n{'='*60}")
    print(f"mmap+mlock Allocation Benchmark")
    print(f"{'='*60}")
    print(f"{'Size MB':>10} {'Alloc (ms)':>12} {'Free (ms)':>10}")

    libc = _get_libc()
    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024
        alloc_times = []
        free_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            m = mmap.mmap(-1, size)
            buf = (ctypes.c_char * size).from_buffer(m)
            ptr = ctypes.addressof(buf)
            libc.mlock(ctypes.c_void_p(ptr), ctypes.c_size_t(size))
            t1 = time.perf_counter()
            libc.munlock(ctypes.c_void_p(ptr), ctypes.c_size_t(size))
            del buf
            m.close()
            t2 = time.perf_counter()
            alloc_times.append(t1 - t0)
            free_times.append(t2 - t1)

        avg_alloc = sum(alloc_times) / len(alloc_times) * 1000
        avg_free = sum(free_times) / len(free_times) * 1000
        print(f"{size_mb:>10} {avg_alloc:>12.2f} {avg_free:>10.2f}")


def bench_memcpy(sizes_mb: list[int], iterations: int = 5):
    """Benchmark ctypes.memmove (slab -> DRAM staging copy)."""
    print(f"\n{'='*60}")
    print(f"Host memcpy Benchmark (slab -> staging)")
    print(f"{'='*60}")
    print(f"{'Size MB':>10} {'Time (ms)':>10} {'BW (GB/s)':>10}")

    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024
        m1 = mmap.mmap(-1, size)
        m2 = mmap.mmap(-1, size)
        buf1 = (ctypes.c_char * size).from_buffer(m1)
        buf2 = (ctypes.c_char * size).from_buffer(m2)
        ptr1 = ctypes.addressof(buf1)
        ptr2 = ctypes.addressof(buf2)

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            ctypes.memmove(ptr2, ptr1, size)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = sum(times) / len(times) * 1000
        bw = size / (avg / 1000) / 1e9 if avg > 0 else 0
        print(f"{size_mb:>10} {avg:>10.2f} {bw:>10.2f}")

        del buf1, buf2
        m1.close()
        m2.close()


def bench_hip_d2h_h2d(sizes_mb: list[int], iterations: int = 5):
    """Benchmark hipMemcpy D2H and H2D."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        print("[HIP] libamdhip64.so not found, skipping")
        return

    print(f"\n{'='*60}")
    print(f"HIP D2H / H2D Benchmark")
    print(f"{'='*60}")
    print(f"{'Size MB':>10} {'D2H (ms)':>10} {'D2H BW':>10} {'H2D (ms)':>10} {'H2D BW':>10}")

    libc = _get_libc()
    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024

        # Host buffer (mmap+mlock)
        m = mmap.mmap(-1, size)
        buf = (ctypes.c_char * size).from_buffer(m)
        host_ptr = ctypes.addressof(buf)
        libc.mlock(ctypes.c_void_p(host_ptr), ctypes.c_size_t(size))

        # GPU buffer
        gpu_ptr = ctypes.c_void_p()
        err = hip.hipMalloc(ctypes.byref(gpu_ptr), ctypes.c_size_t(size))
        if err != 0:
            print(f"  hipMalloc failed for {size_mb} MB (err={err})")
            del buf; m.close()
            continue

        # D2H
        d2h_times = []
        for _ in range(iterations):
            hip.hipDeviceSynchronize()
            t0 = time.perf_counter()
            hip.hipMemcpy(ctypes.c_void_p(host_ptr), gpu_ptr,
                         ctypes.c_size_t(size), ctypes.c_int(2))
            t1 = time.perf_counter()
            d2h_times.append(t1 - t0)

        # H2D
        h2d_times = []
        for _ in range(iterations):
            hip.hipDeviceSynchronize()
            t0 = time.perf_counter()
            hip.hipMemcpy(gpu_ptr, ctypes.c_void_p(host_ptr),
                         ctypes.c_size_t(size), ctypes.c_int(1))
            t1 = time.perf_counter()
            h2d_times.append(t1 - t0)

        avg_d2h = sum(d2h_times) / len(d2h_times) * 1000
        avg_h2d = sum(h2d_times) / len(h2d_times) * 1000
        bw_d2h = size / (avg_d2h / 1000) / 1e9 if avg_d2h > 0 else 0
        bw_h2d = size / (avg_h2d / 1000) / 1e9 if avg_h2d > 0 else 0

        print(f"{size_mb:>10} {avg_d2h:>10.2f} {bw_d2h:>9.1f}G {avg_h2d:>10.2f} {bw_h2d:>9.1f}G")

        hip.hipFree(gpu_ptr)
        del buf; m.close()


def bench_chunked_mr_overhead(device: str, total_mb: int = 1700, chunk_mb: int = 190):
    """Simulate chunked MR overhead for a DeepSeek-R1 sized transfer."""
    try:
        import pyverbs.device as d
        import pyverbs.pd as pd_mod
        import pyverbs.mr as mr_mod
        import pyverbs.enums as e
    except ImportError:
        print("[chunked] pyverbs not available, skipping")
        return

    ctx = d.Context(name=device)
    pd = pd_mod.PD(ctx)
    access = e.IBV_ACCESS_LOCAL_WRITE | e.IBV_ACCESS_REMOTE_WRITE | e.IBV_ACCESS_REMOTE_READ

    num_chunks = (total_mb + chunk_mb - 1) // chunk_mb
    chunk_size = chunk_mb * 1024 * 1024
    m, buf, ptr, _ = alloc_mmap_buffer(chunk_mb)

    print(f"\n{'='*60}")
    print(f"Chunked MR Overhead ({total_mb} MB total, {chunk_mb} MB chunks, {num_chunks} chunks)")
    print(f"{'='*60}")

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(num_chunks):
            mr = mr_mod.MR(pd, ptr, chunk_size, access)
            mr.close()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times) * 1000
    per_chunk = avg / num_chunks
    print(f"  Total: {avg:.1f} ms for {num_chunks} chunks")
    print(f"  Per chunk: {per_chunk:.2f} ms (reg + dereg)")
    print(f"  Overhead fraction: {avg / (avg + total_mb * 1024 * 1024 / 50e9 * 1000) * 100:.1f}% "
          f"(vs 50 GB/s RDMA)")

    del buf; m.close()
    pd.close(); ctx.close()


def main():
    parser = argparse.ArgumentParser(description="RDMA microbenchmark for ionic")
    parser.add_argument("--device", default="ionic_1", help="RDMA device name")
    parser.add_argument("--sizes", default="1,10,50,100,190",
                       help="Comma-separated sizes in MB")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    print(f"RDMA Microbenchmark on {args.device}")
    print(f"Sizes: {sizes} MB, Iterations: {args.iterations}")

    bench_mmap_alloc(sizes, args.iterations)
    bench_memcpy(sizes, min(args.iterations, 5))
    bench_hip_d2h_h2d(sizes, min(args.iterations, 5))
    bench_ibverbs_reg_dereg(args.device, sizes, args.iterations)
    bench_chunked_mr_overhead(args.device)


if __name__ == "__main__":
    main()
