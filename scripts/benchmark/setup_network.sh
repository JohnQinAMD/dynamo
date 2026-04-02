#!/usr/bin/env bash
# Ionic network setup: IPv4 assignment + cross-node subnet matching.
#
# Usage:
#   bash setup_network.sh                     # Auto-detect node ID, assign IPs
#   bash setup_network.sh --verify            # Verify current config
#   bash setup_network.sh --match NODE_IP     # Find matching ionic devices with remote node
#   bash setup_network.sh --node-id 99        # Explicit node ID

set -uo pipefail

_get_mgmt_ip() { ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print $7}'; }

# --- Ionic subnet matching ---
match_subnets() {
    local remote=$1
    local SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5"
    echo "=== Ionic Subnet Matching: local vs $remote ==="
    echo ""
    printf "%-10s %-12s %-12s %s\n" "Device" "Local Sub" "Remote Sub" "Match"
    echo "──────────────────────────────────────────────"
    local matched=""
    for i in 0 1 2 3 4 5 6 7; do
        local lgid=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1 2>/dev/null)
        local rgid=$($SSH "$remote" "cat /sys/class/infiniband/ionic_$i/ports/1/gids/1 2>/dev/null" 2>/dev/null)
        local lsub=$(echo "$lgid" | cut -d: -f4)
        local rsub=$(echo "$rgid" | cut -d: -f4)
        local match="NO"
        if [[ -n "$lsub" ]] && [[ "$lsub" == "$rsub" ]]; then
            match="YES"
            matched="${matched:+$matched,}ionic_$i"
        fi
        printf "ionic_%-4s %-12s %-12s %s\n" "$i" "${lsub:-n/a}" "${rsub:-n/a}" "$match"
    done
    echo ""
    if [[ -n "$matched" ]]; then
        echo "Matching devices: $matched"
        echo "Use: --disaggregation-ib-device $matched"
    else
        echo "WARNING: No matching subnets found!"
        echo "Cross-node RDMA will fail. Check ionic IPv4 configuration."
    fi
}

# --- IPv4 assignment ---
assign_ips() {
    local node_id=$1
    echo "=== Assigning IPv4 to ionic ports (node_id=$node_id) ==="
    for i in 0 1 2 3 4 5 6 7; do
        local iface=$(ls /sys/class/infiniband/ionic_$i/device/net/ 2>/dev/null | head -1)
        [[ -z "$iface" ]] && continue
        local gid_hex=$(cat /sys/class/infiniband/ionic_$i/ports/1/gids/1 2>/dev/null | cut -d: -f4)
        [[ -z "$gid_hex" ]] && continue
        local subnet=$((16#${gid_hex: -2} + 100))
        [[ $subnet -gt 254 ]] && subnet=$((subnet - 100))
        local target_ip="192.168.${subnet}.${node_id}/24"
        local existing=$(ip -4 addr show "$iface" 2>/dev/null | grep "192.168.${subnet}\\." || true)
        if [[ -n "$existing" ]]; then
            echo "  OK: $iface already has $target_ip"
        else
            ip addr add "$target_ip" dev "$iface" 2>/dev/null || true
            ip link set "$iface" up 2>/dev/null || true
            echo "  SET: $iface → $target_ip"
        fi
    done
}

# --- Verify ---
verify() {
    echo "=== Ionic Verification ==="
    local dev_count=$(ibv_devinfo 2>&1 | grep -c "hca_id" || true)
    echo "  Devices visible: $dev_count"
    for i in 0 1 2 3 4 5 6 7; do
        local iface=$(ls /sys/class/infiniband/ionic_$i/device/net/ 2>/dev/null | head -1)
        [[ -z "$iface" ]] && continue
        local ipv4=$(ip -4 addr show "$iface" 2>/dev/null | grep "inet " | awk '{print $2}' | head -1)
        echo "  ionic_$i ($iface): ${ipv4:-NO IPv4}"
    done
}

# --- Parse args ---
NODE_ID=""
ACTION="assign"
REMOTE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-id) NODE_ID="$2"; shift 2 ;;
        --verify) ACTION="verify"; shift ;;
        --match) ACTION="match"; REMOTE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: setup_network.sh [--node-id N] [--verify] [--match REMOTE_HOST]"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

case "$ACTION" in
    verify) verify ;;
    match) match_subnets "$REMOTE" ;;
    assign)
        if [[ -z "$NODE_ID" ]]; then
            NODE_ID=$(hostname | grep -oE '[0-9]+$' | tail -1 || true)
            NODE_ID=$((10#${NODE_ID: -2}))
            [[ "$NODE_ID" -eq 0 ]] && NODE_ID=100
            echo "Auto-detected node ID: $NODE_ID"
        fi
        assign_ips "$NODE_ID"
        ;;
esac
