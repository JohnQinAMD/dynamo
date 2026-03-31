---
name: adapt-nvidia-to-rocm
description: Adapt NVIDIA GPU frameworks (Dynamo, vLLM, SGLang, TensorRT-LLM) to AMD ROCm for MI300X/MI355X GPUs. Use when porting CUDA code to HIP, configuring ROCm containers, setting up Pensando ionic RDMA, adapting Kubernetes for amd.com/gpu, fixing Python import compatibility across 3.10-3.12, or debugging RCCL/MoRI/RIXL networking issues on AMD hardware.
---

# Adapt NVIDIA Frameworks to AMD ROCm

## Core Principles

1. **Additive-only changes** — never remove NVIDIA code; add `#elif`/`if rocm` paths
2. **Upstream-rebaseable** — all changes must survive `git rebase` against upstream `main`
3. **Monkey-patch over fork** — for third-party code (SGLang, vLLM), use runtime patches from your own repo instead of modifying their source
4. **Single wheel for all Python versions** — use PyO3 `abi3-py310` for Rust bindings

## Strategy Checklist

When adapting any NVIDIA framework component:

```
- [ ] Identify CUDA-specific code paths
- [ ] Add HIP/ROCm conditional paths (not replace)
- [ ] Handle Python version compatibility (3.10-3.12)
- [ ] Make imports lazy for optional NVIDIA deps (nixl, tensorrt, etc.)
- [ ] Test in ROCm container on MI355X
- [ ] Document in docs/amd-*.md
```

## 1. CUDA to HIP Porting

### C++ / Rust

```cpp
// Pattern: add #elif before #else, keep CUDA path intact
#if defined(USE_HIP)
    // ROCm path
    hipPointerAttribute_t attrs;
    hipPointerGetAttributes(&attrs, addr);
#elif !defined(WITH_NVIDIA_PEERMEM) && defined(USE_CUDA)
    // NVIDIA dmabuf path (unchanged)
    CUmemorytype memType;
    cuPointerGetAttribute(&memType, ...);
#else
    // Generic fallback (unchanged)
#endif
```

### Rust Cargo Features

```toml
# Add ROCm features alongside CUDA ones
[features]
default = ["cuda"]
cuda = ["cudarc"]
rocm = []
testing-rocm = ["rocm"]
```

### Build Environment

```bash
export LIBCLANG_PATH=/opt/rocm/lib/llvm/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I$(find /usr/lib/gcc -name stdbool.h | head -1 | xargs dirname)"
export VIRTUAL_ENV=/opt/venv
```

## 2. Python Compatibility

### Lazy Imports for NVIDIA-only Dependencies

```python
# Pattern: wrap optional NVIDIA imports in try/except
try:
    from nixl._api import nixl_agent
    _HAS_NIXL = True
except ImportError:
    _HAS_NIXL = False
    logger.warning("nixl not available")
```

### typing.Self Guard (Python 3.10 compat)

```python
import sys
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing import TypeVar
    Self = TypeVar("Self", bound="YourClass")
```

### PyO3 abi3 (single wheel for 3.10+)

The `abi3-py310` feature produces `_core.abi3.so` — one binary for all Python >= 3.10. Build with `maturin develop --release` in any 3.10+ container.

## 3. Monkey-Patching Third-Party Code

When adapting libraries you don't own (SGLang, vLLM):

```python
# Pattern: patch at runtime, zero changes to upstream source
def patch_for_rocm():
    from third_party.module import TargetClass

    _orig_method = TargetClass.method

    def _patched_method(self, *args, **kwargs):
        # ROCm-specific behavior
        return _orig_method(self, *args, **kwargs)

    TargetClass.method = _patched_method
```

Key example: `dynamo/components/src/dynamo/sglang/nixl_rocm_staging.py` patches `NixlKVManager` for DRAM staging without touching SGLang source.

## 4. Container & Docker

### Recommended Images

| Image | Python | Use |
|-------|--------|-----|
| `rocm/sgl-dev:sglang-*-mori-*` | 3.10 | SGLang + MoRI + Dynamo |
| `rocm/vllm:latest` | 3.12 | vLLM on ROCm |
| `rocm/pytorch:latest` | 3.10 | General purpose |

### Essential Docker Flags

```bash
docker run --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v /mnt/vast/john/rocm-dynamo:/workspace
```

### Critical Environment Variables

```bash
export SGLANG_AITER_MLA_PERSIST=False  # 11x TTFT fix for DSV3
export RCCL_MSCCL_ENABLE=0
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 5. RDMA / Networking (Pensando Ionic)

### Transfer Backend Selection

| Backend | When to Use |
|---------|------------|
| **MoRI** | Default for AMD disagg — validated, best performance |
| **RIXL + DRAM staging** | `SGLANG_NIXL_ROCM_STAGING=1`; monkey-patch, zero source changes |
| **Mooncake RDMA** | `bash scripts/patch_mooncake_rocm.sh` — patch + rebuild in container |
| **TCP** | Fallback; no RDMA hardware needed |

### Ionic Subnet Matching

```bash
# Find matching ionic devices across nodes
for i in 0 1 2 3 4 5 6 7; do
    gid=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1)
    echo "ionic_$i  $(echo $gid | cut -d: -f1-4)"
done
# Match devices with same subnet prefix between nodes
```

### Ionic Driver Fix

Container `libionic` ABI mismatch with host kernel — mount host driver:
```bash
-v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185:/host-ionic/libionic.so:ro
# Inside container: cp + ldconfig
```

## 6. Kubernetes Adaptation

### Resource Names

```yaml
# NVIDIA → AMD
resources:
  limits:
    amd.com/gpu: 8    # not nvidia.com/gpu
```

### GPU Discovery

Use `amd-smi` instead of `nvidia-smi`. The Go operator code needs `DiscoverAMDGPUs()` alongside the existing NVIDIA path.

## 7. Testing Strategy

### Build Dynamo

```bash
MODE=develop bash scripts/build_dynamo_wheel.sh
```

### Run Tests

```bash
# Full suite (164+ pass on MI355X)
python3 -m pytest tests/ --no-header -q --tb=no

# Specific categories
python3 -m pytest tests/disagg/test_nixl_rocm_staging.py  # DRAM staging
python3 -m pytest tests/basic/test_rocm_gpu_detection.py   # GPU detection
python3 -m pytest tests/planner/unit/                      # Planner (no GPU)
```

### Multi-Node Disagg Test

```bash
export DISAGG_PREFILL_HOST=chi2899
export DISAGG_DECODE_HOST=chi2900
python3 -m pytest tests/disagg/test_mori_rdma.py -v
```

## 8. Common Bugs & Fixes

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `ibv_reg_mr ENOMEM` | Ionic can't register GPU VRAM | Use DRAM staging or MoRI |
| 11x slow TTFT on DSV3 | aiter MLA persistent kernel conflict | `SGLANG_AITER_MLA_PERSIST=False` |
| `assert block_size > 1` panic | SGLang uses page_size=1 | Patch to default to 16 |
| `ModuleNotFoundError: nixl` | RIXL not installed | Make import lazy with try/except |
| `typing.Self` crash | Python 3.10 lacks Self | Guard with `sys.version_info` check |
| `stdbool.h not found` | bindgen can't find GCC headers | Set `BINDGEN_EXTRA_CLANG_ARGS` |
| ionic ABI mismatch | Container vs host driver version | Mount host `libionic1` |

## Additional Resources

- [Build guide](docs/amd-rocm-build.md) — step-by-step Rust + Python build
- [Deployment guide](docs/amd-rocm-guide.md) — containers, K8s, features
- [System design](docs/amd-system-design.md) — architecture diagrams
- [Performance report](docs/amd-performance-report.md) — benchmark results
- [Test plan](docs/amd-test-plan.md) — 89 tests across 10 phases
