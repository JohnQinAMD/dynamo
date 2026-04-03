#!/usr/bin/env bash
# Reusable preflight checks for MoRI RDMA disaggregated benchmarks on MI355X.
#
# Validates: SSH, Docker, GPU count+memory, ionic (devices/ports/IPv4/libionic),
# NFS model mount, management IP, and cross-node reachability.
#
# Usage (standalone):
#   DOCKER_IMG=amdprimus/dynamo-rocm-sglang:latest \
#   MODEL_HOST=/mnt/vast/john/huggingface \
#   bash preflight_check.sh chi2863 chi2870 chi2900
#
# Usage (sourced by other scripts):
#   source "$(dirname "$0")/preflight_check.sh"
#   run_preflight chi2863 chi2870 chi2900
#
# All checks are ERROR (exit 1) except cross-node ping (WARN).
# Ionic IPv4 is auto-fixed via setup_ionic_network.sh; ERROR if still <8 after fix.

PREFLIGHT_SSH="${PREFLIGHT_SSH:-ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10}"
PREFLIGHT_DOCKER_IMG="${DOCKER_IMG:-amdprimus/dynamo-rocm-sglang:latest}"
PREFLIGHT_MODEL_HOST="${MODEL_HOST:-/mnt/vast/john/huggingface}"
PREFLIGHT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFLIGHT_DOCKER_USER="${DOCKER_USER:-}"
PREFLIGHT_DOCKER_PASS="${DOCKER_PASS:-}"

_pf_count_ionic_ipv4() {
    local node=$1
    $PREFLIGHT_SSH "$node" "
        c=0
        for i in 0 1 2 3 4 5 6 7; do
            f=\$(ls /sys/class/infiniband/ionic_\$i/device/net/ 2>/dev/null | head -1)
            [ -z \"\$f\" ] && continue
            ip addr show \$f 2>/dev/null | grep -q 'inet 192.168' && ((c++))
        done
        echo \$c
    " 2>/dev/null || echo 0
}

_pf_get_mgmt_ip() {
    $PREFLIGHT_SSH "$1" "ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print \$7}'" 2>/dev/null
}

run_preflight() {
    local nodes=("$@")
    if [[ ${#nodes[@]} -eq 0 ]]; then
        echo "ERROR: run_preflight requires node list as arguments"
        return 1
    fi

    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  PREFLIGHT CHECKS                                        ║"
    echo "╚═══════════════════════════════════════════════════════════╝"

    local PREFLIGHT_FAIL=0

    for node in "${nodes[@]}"; do
        echo ""
        echo "--- Checking $node ---"
        local FAIL=0

        # 1. SSH connectivity
        if ! $PREFLIGHT_SSH "$node" "echo ok" &>/dev/null; then
            echo "  [FAIL] SSH unreachable"
            PREFLIGHT_FAIL=1; continue
        fi
        echo "  [OK]   SSH"

        # 2. Docker daemon + image
        local DOCKER_OK
        DOCKER_OK=$($PREFLIGHT_SSH "$node" "docker info >/dev/null 2>&1 && echo yes || echo no" 2>/dev/null)
        if [[ "$DOCKER_OK" != "yes" ]]; then
            echo "  [FAIL] Docker daemon not running"
            FAIL=1
        else
            echo "  [OK]   Docker daemon"
        fi

        local IMG_EXISTS
        IMG_EXISTS=$($PREFLIGHT_SSH "$node" "docker images -q $PREFLIGHT_DOCKER_IMG 2>/dev/null | head -1" 2>/dev/null)
        if [[ -z "$IMG_EXISTS" ]]; then
            echo "  [FIX]  Image not found, pulling $PREFLIGHT_DOCKER_IMG..."
            local login_cmd=""
            if [[ -n "$PREFLIGHT_DOCKER_USER" && -n "$PREFLIGHT_DOCKER_PASS" ]]; then
                login_cmd="docker login -u $PREFLIGHT_DOCKER_USER -p $PREFLIGHT_DOCKER_PASS 2>/dev/null;"
            fi
            $PREFLIGHT_SSH "$node" "${login_cmd} docker pull $PREFLIGHT_DOCKER_IMG 2>/dev/null | tail -1" 2>/dev/null
            IMG_EXISTS=$($PREFLIGHT_SSH "$node" "docker images -q $PREFLIGHT_DOCKER_IMG 2>/dev/null | head -1" 2>/dev/null)
            if [[ -z "$IMG_EXISTS" ]]; then
                echo "  [FAIL] Failed to pull image"
                FAIL=1
            else
                echo "  [OK]   Image pulled"
            fi
        else
            echo "  [OK]   Docker image"
        fi

        # 3. GPU count
        local GPU_COUNT
        GPU_COUNT=$($PREFLIGHT_SSH "$node" "rocm-smi --showuse --csv 2>/dev/null | grep -c 'card'" 2>/dev/null || echo 0)
        if [[ "$GPU_COUNT" -lt 8 ]]; then
            echo "  [FAIL] Only $GPU_COUNT GPUs visible (need 8)"
            FAIL=1
        else
            echo "  [OK]   $GPU_COUNT GPUs"
        fi

        # 4. GPU VRAM — must be <8GB total, otherwise kill all containers+processes
        local GPU_MEM
        GPU_MEM=$($PREFLIGHT_SSH "$node" "rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR>1{u+=\$3} END{printf \"%.0f\", u/1073741824}'" 2>/dev/null || echo 0)
        if [[ "$GPU_MEM" -gt 8 ]]; then
            echo "  [FIX]  ${GPU_MEM}GB GPU VRAM in use — killing all containers and GPU processes..."
            $PREFLIGHT_SSH "$node" "docker ps -q | xargs -r docker rm -f 2>/dev/null; \
                rocm-smi --showpids 2>/dev/null | awk 'NR>3 && \$1+0>0{print \$1}' | xargs -r kill -9 2>/dev/null" 2>/dev/null
            sleep 5
            GPU_MEM=$($PREFLIGHT_SSH "$node" "rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR>1{u+=\$3} END{printf \"%.0f\", u/1073741824}'" 2>/dev/null || echo 0)
            if [[ "$GPU_MEM" -gt 8 ]]; then
                echo "  [FAIL] Still ${GPU_MEM}GB after cleanup"
                FAIL=1
            else
                echo "  [OK]   GPU VRAM freed (${GPU_MEM}GB)"
            fi
        else
            echo "  [OK]   GPU VRAM clean (${GPU_MEM}GB)"
        fi

        # 5. Ionic IB devices (8 required, all PORT_ACTIVE)
        local IB_COUNT
        IB_COUNT=$($PREFLIGHT_SSH "$node" "ls -d /sys/class/infiniband/ionic_* 2>/dev/null | wc -l" 2>/dev/null || echo 0)
        if [[ "$IB_COUNT" -lt 8 ]]; then
            echo "  [FAIL] Only $IB_COUNT ionic IB devices (need 8)"
            FAIL=1
        else
            local IB_ACTIVE
            IB_ACTIVE=$($PREFLIGHT_SSH "$node" "ibv_devinfo 2>/dev/null | grep -c 'PORT_ACTIVE'" 2>/dev/null || echo 0)
            if [[ "$IB_ACTIVE" -lt 8 ]]; then
                echo "  [FAIL] $IB_COUNT ionic devices but only $IB_ACTIVE PORT_ACTIVE"
                FAIL=1
            else
                echo "  [OK]   $IB_COUNT ionic ($IB_ACTIVE PORT_ACTIVE)"
            fi
        fi

        # 6. Ionic IPv4 — all 8 must have 192.168.x.x; auto-fix if missing
        local IONIC_IPS
        IONIC_IPS=$(_pf_count_ionic_ipv4 "$node")
        if [[ "$IONIC_IPS" -lt 8 ]]; then
            echo "  [FIX]  Only $IONIC_IPS/8 ionic IPv4, running setup_ionic_network.sh..."
            $PREFLIGHT_SSH "$node" "bash ${PREFLIGHT_SCRIPT_DIR}/setup_ionic_network.sh" 2>/dev/null
            sleep 2
            IONIC_IPS=$(_pf_count_ionic_ipv4 "$node")
            if [[ "$IONIC_IPS" -lt 8 ]]; then
                echo "  [FAIL] Still $IONIC_IPS/8 ionic IPv4 after auto-fix — MoRI RDMA will crash"
                FAIL=1
            else
                echo "  [OK]   Ionic IPv4 fixed ($IONIC_IPS/8)"
            fi
        else
            echo "  [OK]   Ionic IPv4 ($IONIC_IPS/8)"
        fi

        # 7. libionic.so on host
        local LIBIONIC
        LIBIONIC=$($PREFLIGHT_SSH "$node" "ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* 2>/dev/null | head -1" 2>/dev/null)
        if [[ -z "$LIBIONIC" ]]; then
            echo "  [FAIL] libionic.so not found on host"
            FAIL=1
        else
            echo "  [OK]   libionic $(basename "$LIBIONIC")"
        fi

        # 8. NFS/vast model mount
        local MODEL_OK
        MODEL_OK=$($PREFLIGHT_SSH "$node" "test -f ${PREFLIGHT_MODEL_HOST}/DeepSeek-R1-0528/config.json && echo Y || echo N" 2>/dev/null)
        if [[ "$MODEL_OK" != "Y" ]]; then
            echo "  [FAIL] NFS/vast mount missing — ${PREFLIGHT_MODEL_HOST}/DeepSeek-R1-0528 not accessible"
            FAIL=1
        else
            echo "  [OK]   NFS/vast model mount"
        fi

        # 9. Management IP — must not be ionic 192.168.x.x
        local MGMT_IP
        MGMT_IP=$(_pf_get_mgmt_ip "$node")
        if [[ -z "$MGMT_IP" ]]; then
            echo "  [FAIL] Cannot determine management IP"
            FAIL=1
        elif [[ "$MGMT_IP" == 192.168.* ]]; then
            echo "  [FAIL] Management IP is ionic address ($MGMT_IP) — MoRI bootstrap will fail"
            FAIL=1
        else
            echo "  [OK]   Management IP $MGMT_IP"
        fi

        # 10. Cross-node ping (WARN only — some clusters block ICMP)
        for peer in "${nodes[@]}"; do
            if [[ "$peer" != "$node" ]]; then
                local PING_OK
                PING_OK=$($PREFLIGHT_SSH "$node" "ping -c1 -W2 $peer >/dev/null 2>&1 && echo ok || echo fail" 2>/dev/null)
                if [[ "$PING_OK" != "ok" ]]; then
                    echo "  [WARN] Cannot ping $peer"
                fi
            fi
        done

        if [[ "$FAIL" -gt 0 ]]; then
            PREFLIGHT_FAIL=1
            echo "  >>> $node FAILED <<<"
        else
            echo "  >>> $node PASSED"
        fi
    done

    # 11. Cross-node ionic subnet match (same switch fabric required for RDMA)
    if [[ ${#nodes[@]} -gt 1 ]]; then
        echo ""
        echo "--- Cross-node ionic subnet check ---"
        local ref_node="${nodes[0]}"
        local ref_subnets
        ref_subnets=$($PREFLIGHT_SSH "$ref_node" "for i in 0 1 2 3 4 5 6 7; do cat /sys/class/infiniband/ionic_\$i/ports/1/gids/1 2>/dev/null | cut -d: -f4; done | sort" 2>/dev/null)
        local subnet_mismatch=0
        for peer in "${nodes[@]:1}"; do
            local peer_subnets
            peer_subnets=$($PREFLIGHT_SSH "$peer" "for i in 0 1 2 3 4 5 6 7; do cat /sys/class/infiniband/ionic_\$i/ports/1/gids/1 2>/dev/null | cut -d: -f4; done | sort" 2>/dev/null)
            if [[ "$ref_subnets" == "$peer_subnets" ]]; then
                echo "  [OK]   $ref_node <-> $peer: 8/8 GID subnets match (same switch fabric)"
            else
                local common
                common=$(comm -12 <(echo "$ref_subnets") <(echo "$peer_subnets") | wc -l)
                if [[ "$common" -ge 4 ]]; then
                    echo "  [WARN] $ref_node <-> $peer: only $common/8 GID subnets match (partial fabric overlap)"
                else
                    echo "  [FAIL] $ref_node <-> $peer: only $common/8 GID subnets match — nodes likely on different switches"
                    subnet_mismatch=1
                fi
            fi
        done
        if [[ "$subnet_mismatch" -gt 0 ]]; then
            PREFLIGHT_FAIL=1
        fi
    fi

    echo ""
    if [[ "$PREFLIGHT_FAIL" -gt 0 ]]; then
        echo "╔═══════════════════════════════════════════════════════════╗"
        echo "║  PREFLIGHT FAILED — fix issues above or use other nodes  ║"
        echo "╚═══════════════════════════════════════════════════════════╝"
        return 1
    fi

    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  ALL PREFLIGHT CHECKS PASSED                             ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    return 0
}

# Allow standalone execution: bash preflight_check.sh node1 node2 ...
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -eq 0 ]]; then
        echo "Usage: bash $0 <node1> [node2] [node3] ..."
        echo "  Env vars: DOCKER_IMG, MODEL_HOST, DOCKER_USER, DOCKER_PASS"
        exit 1
    fi
    run_preflight "$@"
    exit $?
fi
