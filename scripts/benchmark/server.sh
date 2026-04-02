#!/usr/bin/env bash
# Unified server launcher for MI355X disaggregated serving benchmarks.
#
# Supports both Dynamo frontend and sglang_router via FRONTEND_TYPE env var.
# Compatible with InferenceX amd_utils/ format.
#
# Usage:
#   # With sglang_router (InferenceX default):
#   FRONTEND_TYPE=sglang bash server.sh
#
#   # With Dynamo frontend (NVIDIA srt-slurm default):
#   FRONTEND_TYPE=dynamo bash server.sh
#
# Required env vars (set by env.sh or caller):
#   MODEL_DIR, MODEL_NAME, IBDEVICES, MGMT_IP
#   PREFILL_TP_SIZE, DECODE_TP_SIZE
#   PREFILL_ENABLE_EP, PREFILL_ENABLE_DP
#   DECODE_ENABLE_EP, DECODE_ENABLE_DP
#   xP (num prefill workers), yD (num decode workers)
#   IPADDRS (comma-separated node IPs)
#   NODE_RANK (0=head/prefill, others=decode)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Load model config
MODELS_YAML="${SCRIPT_DIR}/models.yaml"
if [[ ! -f "$MODELS_YAML" ]]; then
    echo "[ERROR] models.yaml not found at $MODELS_YAML"
    exit 1
fi

FRONTEND_TYPE="${FRONTEND_TYPE:-sglang}"
HEADNODE_PORT="${HEADNODE_PORT:-20000}"
DOCKER_IMG="${DOCKER_IMG:-amdprimus/dynamo-rocm-sglang:latest}"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"

# --- Build server configs from models.yaml ---
eval "$(python3 -c "
import yaml, os

with open('${MODELS_YAML}') as f:
    models = yaml.safe_load(f)

m = models.get('${MODEL_NAME}', {})
if not m:
    print('echo \"[WARN] Model ${MODEL_NAME} not in models.yaml, using defaults\"')
else:
    print(f'MODEL_BASE_FLAGS=\"{m.get(\"base_flags\", \"\")}\"')
    print(f'MODEL_MTP_FLAGS=\"{m.get(\"mtp_flags\", \"\")}\"')
    print(f'MODEL_DP_FLAGS=\"{m.get(\"dp_flags\", \"\")}\"')
    pf = m.get('prefill', {})
    dc = m.get('decode', {})
    nodp = pf.get('no_dp', {})
    print(f'PREFILL_MEM_FRACTION_STATIC=\"{pf.get(\"mem_fraction_static\", 0.8)}\"')
    print(f'PREFILL_MAX_RUNNING_REQUESTS=\"{nodp.get(\"max_running_requests\", 128)}\"')
    print(f'PREFILL_CHUNKED_PREFILL_SIZE=\"{nodp.get(\"chunked_prefill_size\", 262144)}\"')
    print(f'DECODE_MEM_FRACTION_STATIC=\"{dc.get(\"mem_fraction_static\", 0.85)}\"')
    dc_nodp = dc.get('no_dp', {})
    print(f'DECODE_MAX_RUNNING_REQUESTS=\"{dc_nodp.get(\"max_running_requests\", 128)}\"')
" 2>/dev/null)"

# --- Node topology ---
IFS=',' read -ra IP_ARRAY <<< "${IPADDRS:-$MGMT_IP}"
NODE_RANK="${NODE_RANK:-0}"
DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

# Build prefill/decode connection args
PREFILL_ARGS=""
DECODE_ARGS=""
for i in $(seq 0 $((xP - 1))); do
    PREFILL_ARGS="$PREFILL_ARGS --prefill http://${IP_ARRAY[$i]}:8000"
done
for i in $(seq 0 $((yD - 1))); do
    idx=$((i + xP))
    DECODE_ARGS="$DECODE_ARGS --decode http://${IP_ARRAY[$idx]}:8000"
done

# --- Build parallelism flags ---
build_parallel_flags() {
    local tp=$1 enable_ep=$2 enable_dp=$3
    local flags="--tp-size $tp"
    if [[ "$enable_ep" == "true" ]]; then
        flags="$flags --ep-size $tp"
    fi
    if [[ "$enable_dp" == "true" ]]; then
        flags="$flags --dp-size $tp"
        flags="$flags $MODEL_DP_FLAGS"
    fi
    echo "$flags"
}

PREFILL_PARALLEL=$(build_parallel_flags "$PREFILL_TP_SIZE" "${PREFILL_ENABLE_EP:-false}" "${PREFILL_ENABLE_DP:-false}")
DECODE_PARALLEL=$(build_parallel_flags "$DECODE_TP_SIZE" "${DECODE_ENABLE_EP:-false}" "${DECODE_ENABLE_DP:-false}")

# =========================================================================
# Launch based on node rank
# =========================================================================

if [[ "$NODE_RANK" -eq 0 ]]; then
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Node 0: Frontend + Prefill ($FRONTEND_TYPE)            ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo "Model:    $MODEL_NAME"
    echo "Frontend: $FRONTEND_TYPE"
    echo "Prefill:  $PREFILL_PARALLEL"
    echo "Decode:   $DECODE_PARALLEL"
    echo "Nodes:    ${IPADDRS:-$MGMT_IP}"

    if [[ "$FRONTEND_TYPE" == "dynamo" ]]; then
        # --- Dynamo frontend path ---
        MY_IP="$MGMT_IP"
        rm -rf /tmp/default.etcd
        etcd --listen-client-urls http://0.0.0.0:2379 \
             --advertise-client-urls http://${MY_IP}:2379 \
             > /tmp/etcd.log 2>&1 &
        nats-server -p 4222 -js --client_advertise ${MY_IP}:4222 \
             > /tmp/nats.log 2>&1 &
        sleep 3
        export ETCD_ENDPOINTS=http://${MY_IP}:2379
        export NATS_SERVER=nats://${MY_IP}:4222

        python3 -m dynamo.frontend --http-port 9000 \
            --router-mode round-robin --request-plane tcp \
            > /tmp/frontend.log 2>&1 &
        sleep 2

        SGLANG_CMD="python3 -m dynamo.sglang"
        BENCH_PORT=9000
    else
        # --- sglang_router path ---
        SGLANG_CMD="python3 -m sglang.launch_server"
        BENCH_PORT=30000
    fi

    # Start prefill server
    $SGLANG_CMD --model-path "$MODEL_PATH" \
        $PREFILL_PARALLEL \
        --disaggregation-mode prefill \
        --disaggregation-ib-device "$IBDEVICES" \
        ${MODEL_BASE_FLAGS:-} \
        --host 0.0.0.0 --port 8000 --trust-remote-code \
        --mem-fraction-static "${PREFILL_MEM_FRACTION_STATIC:-0.8}" \
        --disable-radix-cache \
        --max-running-requests "${PREFILL_MAX_RUNNING_REQUESTS:-128}" \
        --chunked-prefill-size "${PREFILL_CHUNKED_PREFILL_SIZE:-262144}" \
        > /tmp/prefill.log 2>&1 &

    echo "Prefill launched, waiting for all servers..."

    # Wait for all nodes
    sleep 10
    for ip in "${IP_ARRAY[@]}"; do
        for attempt in $(seq 1 600); do
            curl -s "http://${ip}:8000/v1/models" 2>/dev/null | grep -qi deepseek && break
            [[ $((attempt % 60)) -eq 0 ]] && echo "  Waiting for $ip... (${attempt}s)"
            sleep 1
        done
    done
    echo "All servers ready"

    if [[ "$FRONTEND_TYPE" == "sglang" ]]; then
        # Start sglang_router
        python3 -m sglang_router.launch_router \
            --pd-disaggregation --port 30000 \
            --policy random --prefill-policy random --decode-policy random \
            $PREFILL_ARGS $DECODE_ARGS \
            > /tmp/router.log 2>&1 &
        sleep 10
    fi

    echo "Frontend ready on port $BENCH_PORT"

    # Run benchmarks
    export BENCH_PORT
    bash "$SCRIPT_DIR/bench.sh"

else
    # --- Decode node ---
    echo "Node $NODE_RANK: Decode server"

    if [[ "$FRONTEND_TYPE" == "dynamo" ]]; then
        HEAD_IP="${IP_ARRAY[0]}"
        export ETCD_ENDPOINTS=http://${HEAD_IP}:2379
        export NATS_SERVER=nats://${HEAD_IP}:4222
        SGLANG_CMD="python3 -m dynamo.sglang"
    else
        SGLANG_CMD="python3 -m sglang.launch_server"
    fi

    $SGLANG_CMD --model-path "$MODEL_PATH" \
        $DECODE_PARALLEL \
        --disaggregation-mode decode \
        --disaggregation-ib-device "$IBDEVICES" \
        ${MODEL_BASE_FLAGS:-} \
        --host 0.0.0.0 --port 8000 --trust-remote-code \
        --mem-fraction-static "${DECODE_MEM_FRACTION_STATIC:-0.85}" \
        --max-running-requests "${DECODE_MAX_RUNNING_REQUESTS:-4096}" \
        --prefill-round-robin-balance \
        > /tmp/decode.log 2>&1 &

    echo "Decode server launched, waiting for head node router..."
    # Block until head node's router shuts down
    HEAD_IP="${IP_ARRAY[0]}"
    while curl -s "http://${HEAD_IP}:${BENCH_PORT:-30000}/readiness" &>/dev/null; do
        sleep 10
    done
fi
