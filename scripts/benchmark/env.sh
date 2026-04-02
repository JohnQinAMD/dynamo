#!/bin/bash
# Environment setup for MI355X disaggregated serving benchmarks.
#
# Sourced by server.sh and bench.sh. Sets MoRI, RCCL, and ionic config.
#
# REQUIRED (set by caller or auto-detected):
#   IBDEVICES      RDMA device names (default: auto-detect ionic_0-7)
#   MODEL_NAME     Model directory name under MODEL_DIR
#
# OPTIONAL:
#   MORI_RDMA_TC   RDMA traffic class (auto-detected from nicctl or hostname)
#   FRONTEND_TYPE  "dynamo" or "sglang" (default: sglang)

# --- Pre-flight: clean stale containers that hold GPU VRAM ---
if [[ "${SKIP_CLEANUP:-}" != "1" ]]; then
    stale=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -v "$(hostname)" || true)
    if [[ -n "$stale" ]]; then
        echo "[WARN] Found running containers that may hold GPU VRAM: $stale"
        echo "       Run 'docker rm -f <name>' to free VRAM before benchmarking."
        echo "       Set SKIP_CLEANUP=1 to suppress this warning."
    fi
    # Check GPU VRAM usage
    vram_used=$(amd-smi monitor --gpu 0 2>/dev/null | awk 'NR==2{print $NF}' | cut -d/ -f1)
    if [[ -n "$vram_used" ]]; then
        vram_free=$(echo "$vram_used" | awk '{printf "%.0f", 288 - $1}')
        echo "[INFO] GPU 0 VRAM: ${vram_used} GB used, ~${vram_free} GB free"
        if (( $(echo "$vram_used > 50" | bc -l 2>/dev/null || echo 0) )); then
            echo "[WARN] GPU VRAM not clean! Stale processes may cause OOM during CUDA graph capture."
            echo "       Kill all docker containers: docker rm -f \$(docker ps -aq)"
        fi
    fi
fi

# --- Management IP (NOT hostname -I which may return ionic IP) ---
export MGMT_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print $7}')

# --- IBDEVICES ---
if [[ -z "${IBDEVICES:-}" ]]; then
    if ls /sys/class/infiniband/ionic_0 &>/dev/null; then
        export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    elif ls /sys/class/infiniband/mlx5_0 &>/dev/null; then
        export IBDEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3
    else
        echo "[ERROR] No RDMA devices found and IBDEVICES not set" >&2
        exit 1
    fi
    echo "[INFO] Auto-detected IBDEVICES=$IBDEVICES"
fi

# --- Network interfaces ---
export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
export NCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export NCCL_IB_HCA=$IBDEVICES

# --- SGLang / aiter ---
export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH:-}

# --- MoRI ---
export MORI_SHMEM_MODE=ISOLATION
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=4
export MORI_APP_LOG_LEVEL=INFO
export SGLANG_MORI_FP8_DISP=True
export SGLANG_MORI_FP4_DISP=False
export SGLANG_MORI_FP8_COMB=False
export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384
export MORI_MAX_DISPATCH_TOKENS_DECODE=160
export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))

if [[ "${MODEL_NAME:-}" == *mxfp4* ]]; then
    export SGLANG_MORI_FP8_DISP=False
    export MORI_MAX_DISPATCH_TOKENS_PREFILL=12288
fi

# --- Disagg timeouts ---
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

# --- QoS (auto-detect) ---
if [[ -z "${MORI_RDMA_TC:-}" ]]; then
    if command -v nicctl &>/dev/null; then
        ND_PRIO=$(nicctl show qos 2>/dev/null | awk '/PFC no-drop priorities/ {print $NF; exit}')
        ND_DSCP=$(nicctl show qos 2>/dev/null | awk -v p="$ND_PRIO" '$1=="DSCP" && $2==":" && $NF==p {print $3; exit}')
        if [[ -n "$ND_DSCP" ]] && [[ -n "$ND_PRIO" ]]; then
            export MORI_RDMA_TC=$(( 4 * ND_DSCP ))
            export MORI_RDMA_SL=$ND_PRIO
        fi
    fi
    # Hostname fallback
    if [[ -z "${MORI_RDMA_TC:-}" ]]; then
        NODENAME=$(hostname -s)
        case "$NODENAME" in
            GPU*|smci355*) export MORI_RDMA_TC=96 ;;
            mia1*)         export MORI_RDMA_TC=104 ;;
            chi*)          export MORI_RDMA_TC=104 ;;
        esac
    fi
    [[ -n "${MORI_RDMA_TC:-}" ]] && echo "[INFO] MORI_RDMA_TC=$MORI_RDMA_TC"
fi

# --- Frontend type ---
export FRONTEND_TYPE="${FRONTEND_TYPE:-sglang}"
echo "[INFO] FRONTEND_TYPE=$FRONTEND_TYPE MGMT_IP=$MGMT_IP"
