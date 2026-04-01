# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test to check ROCm version consistency across packages.

Counterpart to test_cuda_version_consistency.py for AMD ROCm environments.
Verifies that all ROCm components (hipcc, PyTorch, pip packages) use the
same major ROCm version to avoid ABI mismatches.

Skipped on NVIDIA/CUDA systems.
"""

import os
import re
import shutil
import subprocess

import pytest

if not os.path.exists("/opt/rocm") and not shutil.which("rocm-smi"):
    pytest.skip("ROCm version tests require /opt/rocm", allow_module_level=True)

pytestmark = [
    pytest.mark.rocm,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
]


def sh(cmd: str) -> str:
    p = subprocess.run(
        ["bash", "-lc", f"{cmd} 2>/dev/null"],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    return (p.stdout or "").strip()


def rocm_version_from_text(text: str) -> str | None:
    """Extract ROCm version (e.g. '7.1', '7.2') from arbitrary text."""
    if not text:
        return None
    pats = [
        r"rocm[- ]?([\d]+\.[\d]+)",
        r"ROCm[- ]?([\d]+\.[\d]+)",
        r"\+rocm([\d]+)",
        r"rocm([\d]+)",
    ]
    for pat in pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ver = m.group(1)
            if "." not in ver:
                ver = ver[0] + "." + ver[1:]
            return ver
    return None


def test_rocm_version_consistency():
    """Collect ROCm versions from multiple signals and assert consistency."""
    if not os.path.exists("/opt/rocm"):
        pytest.skip("ROCm not installed at /opt/rocm")

    signals = [
        (
            "rocm_path",
            "cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm/.info/version-utils 2>/dev/null",
        ),
        ("hipcc", "hipcc --version | head -3"),
        ("pip_torch", "python -m pip list --format=freeze 2>/dev/null | grep -i torch"),
        ("pip_rocm", "python -m pip list --format=freeze 2>/dev/null | grep -i rocm"),
        ("env_ROCM_VERSION", "echo $ROCM_VERSION"),
    ]

    rows: list[tuple[str, str | None, list[str]]] = []
    for label, cmd in signals:
        out = sh(cmd)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]

        if label == "rocm_path" and lines:
            m = re.match(r"([\d]+\.[\d]+)", lines[0])
            ver = m.group(1) if m else None
        else:
            ver = rocm_version_from_text(out)

        rows.append((label, ver, lines if lines else ["<no output>"]))

    detected = [ver for _, ver, _ in rows if ver is not None]
    if not detected:
        pytest.skip("No ROCm version detected from any signal")

    majors = sorted(set(v.split(".")[0] for v in detected))

    report = ["ROCm version signals:"]
    for label, ver, lines in rows:
        ver_s = ver if ver else "-"
        report.append(f"  {ver_s:>6}  {label}")
        for ln in lines[:10]:
            report.append(f"         {ln}")

    assert len(majors) == 1, (
        "\n".join(report) + f"\n\nInconsistent ROCm major versions: {majors}"
    )
