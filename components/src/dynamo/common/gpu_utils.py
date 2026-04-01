# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified GPU utility module supporting both NVIDIA (pynvml) and AMD (amdsmi) GPUs.

Provides a common interface for querying GPU count, memory, utilization,
temperature, and power across both vendors. Auto-detects the available
backend at import time and falls back to CLI tools when native Python
bindings are unavailable.
"""

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_GPU_BACKEND: Optional[str] = None


@dataclass
class GpuInfo:
    """Vendor-agnostic GPU information."""

    index: int
    name: str
    uuid: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_gpu: float
    utilization_memory: float
    temperature: float
    power_draw_w: float
    driver_version: str


def detect_gpu_backend() -> str:
    """Detect available GPU backend: ``'amd'``, ``'nvidia'``, or ``'none'``.

    Override auto-detection by setting ``DYNAMO_GPU_BACKEND=amd|nvidia|none``.
    Otherwise checks for AMD first (``amd-smi`` or ROCm path), then NVIDIA
    (``nvidia-smi``). The result is cached module-wide after the first call.
    """
    global _GPU_BACKEND
    if _GPU_BACKEND is not None:
        return _GPU_BACKEND

    override = os.environ.get("DYNAMO_GPU_BACKEND", "").strip().lower()
    if override in ("amd", "nvidia", "none"):
        _GPU_BACKEND = override
        logger.info("GPU backend override via DYNAMO_GPU_BACKEND=%s", override)
        return _GPU_BACKEND

    if shutil.which("amd-smi") or os.path.exists("/opt/rocm/bin/amd-smi"):
        _GPU_BACKEND = "amd"
    elif shutil.which("nvidia-smi"):
        _GPU_BACKEND = "nvidia"
    else:
        _GPU_BACKEND = "none"
    return _GPU_BACKEND


# ---------------------------------------------------------------------------
# NVIDIA helpers
# ---------------------------------------------------------------------------

_nvml_available: Optional[bool] = None


def _check_nvml() -> bool:
    global _nvml_available
    if _nvml_available is None:
        try:
            import pynvml  # noqa: F401

            _nvml_available = True
        except ImportError:
            _nvml_available = False
    return _nvml_available


_nvml_initialized = False


def _nvml_ensure_init():
    global _nvml_initialized
    if not _nvml_initialized:
        import pynvml

        pynvml.nvmlInit()
        _nvml_initialized = True
        import atexit

        atexit.register(pynvml.nvmlShutdown)


def _nvidia_get_count_nvml() -> int:
    import pynvml

    _nvml_ensure_init()
    return pynvml.nvmlDeviceGetCount()


def _nvidia_get_count_cli() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return len(result.stdout.strip().splitlines())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def _nvidia_get_info_nvml(device_index: int) -> Optional[GpuInfo]:
    import pynvml

    _nvml_ensure_init()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode("utf-8")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            temp = 0.0
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError:
            power = 0.0
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8")

        return GpuInfo(
            index=device_index,
            name=name,
            uuid=uuid,
            memory_total_mb=int(mem.total) // (1024 * 1024),
            memory_used_mb=int(mem.used) // (1024 * 1024),
            memory_free_mb=int(mem.free) // (1024 * 1024),
            utilization_gpu=float(util.gpu),
            utilization_memory=float(util.memory),
            temperature=float(temp),
            power_draw_w=power,
            driver_version=driver,
        )
    except pynvml.NVMLError as exc:
        logger.warning("NVML query failed for device %d: %s", device_index, exc)
        return None


def _nvidia_get_info_cli(device_index: int) -> Optional[GpuInfo]:
    """Fallback: query nvidia-smi for a single GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=name,uuid,memory.total,memory.used,memory.free,"
                "utilization.gpu,utilization.memory,temperature.gpu,"
                "power.draw,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
        if len(parts) < 10:
            return None

        def _float(v: str) -> float:
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        def _int(v: str) -> int:
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return 0

        return GpuInfo(
            index=device_index,
            name=parts[0],
            uuid=parts[1],
            memory_total_mb=_int(parts[2]),
            memory_used_mb=_int(parts[3]),
            memory_free_mb=_int(parts[4]),
            utilization_gpu=_float(parts[5]),
            utilization_memory=_float(parts[6]),
            temperature=_float(parts[7]),
            power_draw_w=_float(parts[8]),
            driver_version=parts[9],
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# AMD helpers
# ---------------------------------------------------------------------------

_amdsmi_available: Optional[bool] = None


def _check_amdsmi() -> bool:
    global _amdsmi_available
    if _amdsmi_available is None:
        try:
            import amdsmi  # noqa: F401

            _amdsmi_available = True
        except ImportError:
            _amdsmi_available = False
    return _amdsmi_available


def _amd_smi_path() -> str:
    path = shutil.which("amd-smi")
    if path:
        return path
    rocm_path = "/opt/rocm/bin/amd-smi"
    if os.path.exists(rocm_path):
        return rocm_path
    return "amd-smi"


_amdsmi_initialized = False


def _amdsmi_ensure_init():
    global _amdsmi_initialized
    if not _amdsmi_initialized:
        import amdsmi

        amdsmi.amdsmi_init()
        _amdsmi_initialized = True
        import atexit

        atexit.register(amdsmi.amdsmi_shut_down)


def _amd_get_count_amdsmi() -> int:
    import amdsmi

    _amdsmi_ensure_init()
    handles = amdsmi.amdsmi_get_processor_handles()
    return len(handles)


def _amd_get_count_cli() -> int:
    try:
        result = subprocess.run(
            [_amd_smi_path(), "list", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                gpus = data.get("gpus", data.get("devices", []))
                return len(gpus) if isinstance(gpus, list) else 0
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return 0


def _amd_get_info_amdsmi(device_index: int) -> Optional[GpuInfo]:
    import amdsmi

    _amdsmi_ensure_init()
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        if device_index >= len(handles):
            return None
        handle = handles[device_index]

        name = "AMD GPU"
        try:
            asic_info = amdsmi.amdsmi_get_gpu_asic_info(handle)
            name = (
                asic_info.get("market_name", name)
                if isinstance(asic_info, dict)
                else name
            )
        except Exception:
            pass

        uuid = ""
        try:
            uuid = amdsmi.amdsmi_get_gpu_device_uuid(handle) or ""
        except Exception:
            pass

        mem_total = mem_used = mem_free = 0
        try:
            vram_info = amdsmi.amdsmi_get_gpu_memory_total(
                handle, amdsmi.AmdSmiMemoryType.VRAM
            )
            mem_total = int(vram_info) // (1024 * 1024) if vram_info else 0
        except Exception:
            pass
        try:
            vram_used = amdsmi.amdsmi_get_gpu_memory_usage(
                handle, amdsmi.AmdSmiMemoryType.VRAM
            )
            mem_used = int(vram_used) // (1024 * 1024) if vram_used else 0
        except Exception:
            pass
        mem_free = max(0, mem_total - mem_used)

        util_gpu = util_mem = 0.0
        try:
            engine_usage = amdsmi.amdsmi_get_gpu_activity(handle)
            if isinstance(engine_usage, dict):
                util_gpu = float(engine_usage.get("gfx_activity", 0))
                util_mem = float(engine_usage.get("umc_activity", 0))
        except Exception:
            pass

        temp = 0.0
        try:
            temp = float(
                amdsmi.amdsmi_get_temp_metric(
                    handle,
                    amdsmi.AmdSmiTemperatureType.EDGE,
                    amdsmi.AmdSmiTemperatureMetric.CURRENT,
                )
            )
        except Exception:
            try:
                temp = float(
                    amdsmi.amdsmi_get_temp_metric(
                        handle,
                        amdsmi.AmdSmiTemperatureType.JUNCTION,
                        amdsmi.AmdSmiTemperatureMetric.CURRENT,
                    )
                )
            except Exception:
                pass

        power = 0.0
        try:
            power_info = amdsmi.amdsmi_get_power_info(handle)
            if isinstance(power_info, dict):
                power = float(power_info.get("current_socket_power", 0))
            else:
                power = float(power_info) if power_info else 0.0
        except Exception:
            pass

        driver = ""
        try:
            ver = amdsmi.amdsmi_get_gpu_driver_info(handle)
            if isinstance(ver, dict):
                driver = ver.get("driver_version", "")
            elif isinstance(ver, str):
                driver = ver
        except Exception:
            pass

        return GpuInfo(
            index=device_index,
            name=name,
            uuid=uuid,
            memory_total_mb=mem_total,
            memory_used_mb=mem_used,
            memory_free_mb=mem_free,
            utilization_gpu=util_gpu,
            utilization_memory=util_mem,
            temperature=temp,
            power_draw_w=power,
            driver_version=driver,
        )
    except Exception as exc:
        logger.warning("amdsmi query failed for device %d: %s", device_index, exc)
        return None


def _parse_amd_smi_value(val: str) -> float:
    """Extract a numeric value from amd-smi output that may contain units."""
    if not val:
        return 0.0
    m = re.search(r"([\d.]+)", str(val))
    return float(m.group(1)) if m else 0.0


def _amd_get_info_cli(device_index: int) -> Optional[GpuInfo]:
    """Fallback: query amd-smi CLI for GPU info."""
    amd_smi = _amd_smi_path()
    name = "AMD GPU"
    uuid = ""
    driver = ""
    mem_total = mem_used = mem_free = 0
    util_gpu = util_mem = 0.0
    temp = 0.0
    power = 0.0

    try:
        result = subprocess.run(
            [amd_smi, "static", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            gpus = (
                data
                if isinstance(data, list)
                else data.get("gpus", data.get("devices", []))
            )
            if isinstance(gpus, list) and device_index < len(gpus):
                gpu = gpus[device_index]
                if isinstance(gpu, dict):
                    asic = gpu.get("asic", gpu)
                    name = (
                        asic.get("market_name")
                        or asic.get("name")
                        or gpu.get("name")
                        or name
                    )
                    uuid = gpu.get("uuid", uuid)
                    driver = gpu.get("driver_version", driver)
                    if isinstance(asic, dict):
                        driver = asic.get("driver_version", driver)
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        KeyError,
    ):
        pass

    try:
        result = subprocess.run(
            [amd_smi, "monitor", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            gpus = (
                data
                if isinstance(data, list)
                else data.get("gpus", data.get("devices", []))
            )
            if isinstance(gpus, list) and device_index < len(gpus):
                gpu = gpus[device_index]
                if isinstance(gpu, dict):
                    power = _parse_amd_smi_value(
                        str(gpu.get("power", gpu.get("power_usage", 0)))
                    )
                    temp = _parse_amd_smi_value(
                        str(gpu.get("temperature", gpu.get("temperature_edge", 0)))
                    )
                    util_gpu = _parse_amd_smi_value(
                        str(gpu.get("gfx_util", gpu.get("gpu_use", 0)))
                    )
                    util_mem = _parse_amd_smi_value(
                        str(gpu.get("mem_util", gpu.get("mem_use", 0)))
                    )
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        KeyError,
    ):
        pass

    try:
        result = subprocess.run(
            [amd_smi, "static", "--vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            gpus = (
                data
                if isinstance(data, list)
                else data.get("gpus", data.get("devices", []))
            )
            if isinstance(gpus, list) and device_index < len(gpus):
                gpu = gpus[device_index]
                if isinstance(gpu, dict):
                    vram = gpu.get("vram", gpu)
                    if isinstance(vram, dict):
                        mem_total = int(
                            _parse_amd_smi_value(
                                str(vram.get("total", vram.get("size", 0)))
                            )
                        )
                        mem_used = int(_parse_amd_smi_value(str(vram.get("used", 0))))
                        mem_free = max(0, mem_total - mem_used)
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        KeyError,
    ):
        pass

    if not name or name == "AMD GPU":
        return None

    return GpuInfo(
        index=device_index,
        name=name,
        uuid=uuid,
        memory_total_mb=mem_total,
        memory_used_mb=mem_used,
        memory_free_mb=mem_free,
        utilization_gpu=util_gpu,
        utilization_memory=util_mem,
        temperature=temp,
        power_draw_w=power,
        driver_version=driver,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_gpu_count() -> int:
    """Return the number of available GPUs."""
    backend = detect_gpu_backend()
    if backend == "nvidia":
        if _check_nvml():
            try:
                return _nvidia_get_count_nvml()
            except Exception:
                pass
        return _nvidia_get_count_cli()
    if backend == "amd":
        if _check_amdsmi():
            try:
                return _amd_get_count_amdsmi()
            except Exception:
                pass
        return _amd_get_count_cli()
    return 0


def get_gpu_info(device_index: int = 0) -> Optional[GpuInfo]:
    """Return :class:`GpuInfo` for *device_index*, or ``None`` on failure."""
    backend = detect_gpu_backend()
    if backend == "nvidia":
        if _check_nvml():
            try:
                return _nvidia_get_info_nvml(device_index)
            except Exception:
                pass
        return _nvidia_get_info_cli(device_index)
    if backend == "amd":
        if _check_amdsmi():
            try:
                return _amd_get_info_amdsmi(device_index)
            except Exception:
                pass
        return _amd_get_info_cli(device_index)
    return None


def get_all_gpu_info() -> List[GpuInfo]:
    """Return :class:`GpuInfo` for every GPU on the system."""
    count = get_gpu_count()
    results = []
    for i in range(count):
        info = get_gpu_info(i)
        if info is not None:
            results.append(info)
    return results


def get_gpu_memory_info(device_index: int = 0) -> Tuple[int, int, int]:
    """Return ``(total_mb, used_mb, free_mb)`` for *device_index*.

    Returns ``(0, 0, 0)`` if GPU info is unavailable.
    """
    info = get_gpu_info(device_index)
    if info is None:
        return (0, 0, 0)
    return (info.memory_total_mb, info.memory_used_mb, info.memory_free_mb)
