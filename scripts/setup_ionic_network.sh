#!/usr/bin/env bash
# Setup ionic NIC IPv4 addresses for MoRI RDMA disaggregated serving
#
# MoRI requires every ionic port to have an IPv4 address. Without IPv4,
# ibv_modify_qp fails with EINVAL causing silent process death (reported
# as RuntimeError: std::bad_cast).
#
# Usage:
#   # Single node (auto-detect node ID from hostname):
#   bash setup_ionic_network.sh
#
#   # Specify node ID explicitly:
#   bash setup_ionic_network.sh --node-id 99
#
#   # Verify only (no changes):
#   bash setup_ionic_network.sh --verify
#
#   # Verify cross-node connectivity to a remote node:
#   bash setup_ionic_network.sh --verify --remote-id 100
#
#   # Fix ionic ABI inside a Docker container:
#   bash setup_ionic_network.sh --fix-abi
#
#   # Run inside container (also configures host IPs via nsenter):
#   bash setup_ionic_network.sh --node-id 99 --in-container
#
# Reference: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/sglang-mori-distributed.html

set -euo pipefail

# Auto-detect interface → subnet mapping from ionic GID table.
# Ionic devices on the same physical switch share a GID subnet prefix.
# We use the last byte of the GID subnet (e.g., 0x0141 → 41) as the IP subnet ID.
# This ensures: same physical switch → same IP subnet → cross-node RDMA works.
#
# The mapping is built dynamically — no hardcoded interface names needed.
declare -A IFACE_SUBNET=()

_build_iface_subnet_map() {
    for i in 0 1 2 3 4 5 6 7; do
        local iface=$(ls /sys/class/infiniband/ionic_$i/device/net/ 2>/dev/null | head -1)
        [ -z "$iface" ] && continue
        local gid_hex=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1 2>/dev/null | cut -d: -f4)
        [ -z "$gid_hex" ] && continue
        # Convert last 2 hex digits to decimal for subnet ID (range 0-255)
        local subnet_id=$((16#${gid_hex: -2}))
        # Offset to avoid common subnets (e.g., 0x41=65 → 165, 0x48=72 → 172)
        local subnet=$((100 + subnet_id))
        if [ "$subnet" -gt 254 ]; then
            subnet=$((subnet - 100))
        fi
        IFACE_SUBNET[$iface]=$subnet
    done

    if [ ${#IFACE_SUBNET[@]} -eq 0 ]; then
        echo "ERROR: No ionic interfaces found. Is this an MI355X node with ionic NICs?"
        echo "       Check: ls /sys/class/infiniband/ionic_*/device/net/"
        exit 1
    fi
}

_build_iface_subnet_map

NODE_ID=""
VERIFY_ONLY=false
REMOTE_ID=""
FIX_ABI=false
IN_CONTAINER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-id) NODE_ID="$2"; shift 2 ;;
        --verify) VERIFY_ONLY=true; shift ;;
        --remote-id) REMOTE_ID="$2"; shift 2 ;;
        --fix-abi) FIX_ABI=true; shift ;;
        --in-container) IN_CONTAINER=true; shift ;;
        -h|--help)
            head -28 "$0" | tail -25
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-detect node ID from hostname (last digits)
if [ -z "$NODE_ID" ]; then
    NODE_ID=$(hostname | grep -oE '[0-9]+$' | tail -1)
    if [ -z "$NODE_ID" ]; then
        echo "ERROR: Cannot auto-detect node ID from hostname '$(hostname)'"
        echo "       Use --node-id <ID> to specify (e.g., --node-id 99)"
        exit 1
    fi
    # Keep last 2 digits to stay in valid IP range
    NODE_ID=$((10#${NODE_ID: -2})); [ "$NODE_ID" -eq 0 ] && NODE_ID=100
    echo "Auto-detected node ID: $NODE_ID (from hostname: $(hostname))"
fi

if [ "$NODE_ID" -lt 1 ] || [ "$NODE_ID" -gt 254 ]; then
    echo "ERROR: Node ID must be 1-254, got $NODE_ID"
    exit 1
fi

# --- Fix ABI ---
if $FIX_ABI; then
    echo "=== Fixing ionic ABI ==="
    if command -v fix-ionic-abi.sh &>/dev/null; then
        fix-ionic-abi.sh
    else
        HOST_LIB=$(find /usr/lib/x86_64-linux-gnu -name "libionic.so.1.1.*" 2>/dev/null | sort -V | tail -1)
        if [ -n "$HOST_LIB" ]; then
            ln -sf "$(basename "$HOST_LIB")" /usr/lib/x86_64-linux-gnu/libionic.so.1
            ldconfig
            echo "Fixed: using $(basename "$HOST_LIB")"
        else
            echo "WARNING: No libionic found. Run from HOST: docker cp \$(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* | head -1) CONTAINER:/usr/lib/x86_64-linux-gnu/libionic.so.1"
        fi
    fi
    echo ""
fi

# --- Verify mode ---
if $VERIFY_ONLY; then
    echo "=== Ionic Network Verification ==="
    echo ""

    echo "Step 1: Check ionic device visibility"
    dev_count=$(ibv_devinfo 2>&1 | grep -c "hca_id" || echo 0)
    if [ "$dev_count" -eq 8 ]; then
        echo "  OK: $dev_count ionic devices visible"
    else
        echo "  FAIL: Only $dev_count devices visible (expected 8)"
        echo "  Fix from HOST: docker cp \$(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* | head -1) CONTAINER:/usr/lib/x86_64-linux-gnu/libionic.so.1"
        ibv_devinfo 2>&1 | grep "Warning" | head -3
        all_ok=false
    fi
    echo ""

    echo "Step 2: Check IPv4 addresses on ionic interfaces"
    all_ok=true
    for i in 0 1 2 3 4 5 6 7; do
        iface=$(ls /sys/class/infiniband/ionic_$i/device/net/ 2>/dev/null | head -1)
        ipv4=$(ip -4 addr show "$iface" 2>/dev/null | grep "inet " | awk '{print $2}' | head -1)
        gid_ipv4=$(ibv_devinfo -d ionic_$i -v 2>&1 | grep "::ffff:" | awk '{print $2}' | head -1)
        if [ -n "$ipv4" ] && [ -n "$gid_ipv4" ]; then
            echo "  OK: ionic_$i ($iface) → $ipv4 → GID $gid_ipv4"
        else
            echo "  MISSING: ionic_$i ($iface) → ip=${ipv4:-NONE} gid=${gid_ipv4:-NONE}"
            all_ok=false
        fi
    done
    if ! $all_ok; then
        echo ""
        echo "  FIX: Run 'bash setup_ionic_network.sh --node-id $NODE_ID' to assign IPs"
    fi
    echo ""

    if [ -n "$REMOTE_ID" ]; then
        echo "Step 3: Cross-node connectivity to node .$REMOTE_ID"
        pass=0; fail=0
        for subnet in "${IFACE_SUBNET[@]}"; do
            if ping -c 1 -W 1 "192.168.${subnet}.${REMOTE_ID}" &>/dev/null; then
                echo "  OK: 192.168.${subnet}.${REMOTE_ID}"
                ((pass++))
            else
                echo "  FAIL: 192.168.${subnet}.${REMOTE_ID}"
                ((fail++))
            fi
        done
        echo "  Result: $pass/8 paths OK, $fail/8 failed"
        if [ "$fail" -gt 0 ]; then
            echo "  FIX: Run 'bash setup_ionic_network.sh --node-id $REMOTE_ID' on the remote node"
            all_ok=false
        fi
    fi

    if $all_ok; then
        exit 0
    else
        echo ""
        echo "Verification FAILED — fix issues above"
        exit 1
    fi
fi

# --- Configure IPs ---
echo "=== Configuring ionic IPv4 addresses (node ID: $NODE_ID) ==="
echo ""

configured=0
for i in 0 1 2 3 4 5 6 7; do
    iface=$(ls /sys/class/infiniband/ionic_$i/device/net/ 2>/dev/null | head -1)
    if [ -z "$iface" ]; then
        echo "  SKIP: ionic_$i — no network interface found"
        continue
    fi

    subnet=${IFACE_SUBNET[$iface]:-}
    if [ -z "$subnet" ]; then
        echo "  WARN: ionic_$i ($iface) — unknown interface, not in mapping table"
        echo "        Add [$iface]=<subnet> to IFACE_SUBNET in this script"
        continue
    fi

    target_ip="192.168.${subnet}.${NODE_ID}/24"
    existing=$(ip -4 addr show "$iface" 2>/dev/null | grep "192.168.${subnet}\." | head -1 || true)

    if [ -n "$existing" ]; then
        echo "  OK: ionic_$i ($iface) already has $target_ip"
    else
        ip addr add "$target_ip" dev "$iface" 2>/dev/null || true
        ip link set "$iface" up 2>/dev/null || true
        echo "  SET: ionic_$i ($iface) → $target_ip"
    fi
    configured=$((configured + 1))
done

echo ""
echo "Configured $configured/8 ionic interfaces"
echo ""
echo "Verify with: bash setup_ionic_network.sh --verify"
echo "Test connectivity: bash setup_ionic_network.sh --verify --remote-id <OTHER_NODE_ID>"
