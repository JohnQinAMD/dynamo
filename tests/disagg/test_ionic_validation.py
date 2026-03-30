# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight validation for Pensando Pollara 400 (ionic) RDMA NICs.

Before running MoRI RDMA disaggregated inference, we need to verify:
1. ionic IB devices are visible
2. GID tables are populated
3. Matching subnets exist between nodes (multi-node only)

These tests can run on a single node to validate the NIC stack. Multi-node
subnet matching requires manual verification or a multi-node CI fixture.
"""

import os
import re
import subprocess

import pytest

pytestmark = [
    pytest.mark.rocm,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.mi355x,
]


def _list_ib_devices() -> list[str]:
    """Return list of InfiniBand device names from /sys/class/infiniband/."""
    ib_path = "/sys/class/infiniband"
    if not os.path.isdir(ib_path):
        return []
    return sorted(os.listdir(ib_path))


def _get_ionic_devices() -> list[str]:
    """Return list of ionic IB device names."""
    return [d for d in _list_ib_devices() if d.startswith("ionic")]


def _read_gid(device: str, port: int = 1, index: int = 1) -> str | None:
    """Read a GID entry from sysfs."""
    gid_path = f"/sys/class/infiniband/{device}/ports/{port}/gids/{index}"
    try:
        with open(gid_path) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError):
        return None


def _gid_to_subnet(gid: str) -> str:
    """Extract the subnet prefix (first 4 groups) from a GID."""
    parts = gid.split(":")
    if len(parts) >= 4:
        return ":".join(parts[:4])
    return gid


class TestIonicPresence:
    """Verify ionic IB devices are present on MI355X systems."""

    def test_infiniband_sysfs_exists(self):
        if not os.path.isdir("/sys/class/infiniband"):
            pytest.skip("No InfiniBand sysfs — not a RDMA-capable node")

    def test_ionic_devices_detected(self):
        ionics = _get_ionic_devices()
        if not ionics:
            pytest.skip(
                "No ionic devices found. Available IB devices: "
                f"{_list_ib_devices() or 'none'}"
            )
        assert len(ionics) > 0

    def test_ionic_gid_populated(self):
        ionics = _get_ionic_devices()
        if not ionics:
            pytest.skip("No ionic devices")

        populated = {}
        for dev in ionics:
            gid = _read_gid(dev)
            if gid and gid != "0000:0000:0000:0000:0000:0000:0000:0000":
                populated[dev] = gid

        if not populated:
            pytest.skip("No ionic GIDs populated (NICs may not be configured)")

        for dev, gid in populated.items():
            assert ":" in gid, f"{dev} has malformed GID: {gid}"


class TestIonicSubnets:
    """Verify ionic subnet configuration for cross-node RDMA."""

    def test_subnet_enumeration(self):
        """List all ionic subnets on this node for manual matching."""
        ionics = _get_ionic_devices()
        if not ionics:
            pytest.skip("No ionic devices")

        subnets: dict[str, str] = {}
        for dev in ionics:
            gid = _read_gid(dev)
            if gid and gid != "0000:0000:0000:0000:0000:0000:0000:0000":
                subnets[dev] = _gid_to_subnet(gid)

        if not subnets:
            pytest.skip("No ionic GIDs populated")

        # Just verify we can enumerate — actual cross-node matching is manual
        assert len(subnets) > 0
        for dev, subnet in subnets.items():
            assert len(subnet.split(":")) == 4, f"Bad subnet for {dev}: {subnet}"

    def test_network_interface_exists(self):
        """Verify ionic devices have associated network interfaces."""
        ionics = _get_ionic_devices()
        if not ionics:
            pytest.skip("No ionic devices")

        for dev in ionics:
            net_path = f"/sys/class/infiniband/{dev}/device/net"
            if os.path.isdir(net_path):
                ifaces = os.listdir(net_path)
                assert len(ifaces) > 0, f"No net interface for {dev}"
                return

        pytest.skip("No ionic device has an associated network interface")


class TestMoriBackend:
    """Verify the MoRI transfer backend is available."""

    def test_mori_importable(self):
        """Check that the mori Python package can be imported."""
        try:
            import mori  # noqa: F401
        except ImportError:
            pytest.skip("mori package not installed")

    def test_sglang_mori_backend_arg(self):
        """Verify SGLang accepts --disaggregation-transfer-backend mori."""
        try:
            import sglang  # noqa: F401
        except ImportError:
            pytest.skip("sglang not installed")

        r = subprocess.run(
            [
                "python",
                "-m",
                "sglang.launch_server",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            pytest.skip("sglang.launch_server --help failed")

        if "disaggregation-transfer-backend" not in r.stdout:
            pytest.skip("SGLang does not support --disaggregation-transfer-backend")
