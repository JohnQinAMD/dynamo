#!/usr/bin/env bash
# Shared function library for MoRI disaggregated benchmarks on MI355X.
#
# Config-driven: call bl_load_config() to resolve flags from InferenceX models.yaml,
# or set BL_PREFILL_FLAGS / BL_DECODE_FLAGS / BL_DOCKER_ENV_ARGS manually.
#
# Usage:
#   source "$(dirname "$0")/benchmark_lib.sh"
#   bl_load_config --model DeepSeek-R1-0528 --prefill-tp 8 --decode-tp 8
#   bl_start_container chi2863
#   bl_launch_sglang_prefill chi2863
#   bl_wait_server chi2863 8000
#   bl_run_bench chi2863 "tag" 64 30000

set -euo pipefail

BL_SSH="${BL_SSH:-ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10}"
BL_DOCKER_IMG="${BL_DOCKER_IMG:-amdprimus/dynamo-rocm-sglang:latest}"
BL_MODEL="${BL_MODEL:-/models/DeepSeek-R1-0528}"
BL_MODEL_HOST="${BL_MODEL_HOST:-/mnt/vast/john/huggingface}"
BL_WORKSPACE="${BL_WORKSPACE:-/mnt/vast/john/rocm-dynamo}"
BL_CONTAINER="${BL_CONTAINER:-bd}"
BL_IBDEVICES="${BL_IBDEVICES:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
BL_BENCH_SCRIPT="${BL_BENCH_SCRIPT:-/workspace/dynamo/InferenceX/utils/bench_serving/benchmark_serving.py}"
BL_RESULTS_DIR="${BL_RESULTS_DIR:-/workspace/dynamo/docs/infx_bench_results}"
BL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BL_RANDOM_RANGE_RATIO=0.8

# Config-driven variables (populated by bl_load_config or set externally)
BL_PREFILL_FLAGS="${BL_PREFILL_FLAGS:-}"
BL_DECODE_FLAGS="${BL_DECODE_FLAGS:-}"
BL_DOCKER_ENV_ARGS="${BL_DOCKER_ENV_ARGS:-}"
BL_MORI_DISPATCH_PREFILL="${BL_MORI_DISPATCH_PREFILL:-16384}"
BL_MORI_DISPATCH_DECODE="${BL_MORI_DISPATCH_DECODE:-160}"
BL_PREFILL_TP="${BL_PREFILL_TP:-8}"
BL_DECODE_TP="${BL_DECODE_TP:-8}"
BL_MTP="${BL_MTP:-0}"

# ── Config loader ──

bl_load_config() {
    local config_output
    config_output=$(python3 "$BL_SCRIPT_DIR/infx_config.py" "$@" --ibdevices "$BL_IBDEVICES") || {
        echo "ERROR: infx_config.py failed"
        return 1
    }
    eval "$config_output"
}

# ── Docker run args (built dynamically from config) ──

_bl_docker_run_args() {
    echo "--network=host --privileged \
--device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
--group-add video --shm-size 256G --ipc=host \
-v ${BL_WORKSPACE}:/workspace -v ${BL_MODEL_HOST}:/models \
-e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
${BL_DOCKER_ENV_ARGS}"
}

# ── Helpers ──

bl_get_ip() {
    $BL_SSH "$1" "ip route get 1.1.1.1 | awk '/src/ {print \$7}'" 2>/dev/null
}

# ── Container lifecycle ──

bl_start_container() {
    local node=$1 name=${2:-$BL_CONTAINER}
    local dra=$(_bl_docker_run_args)
    $BL_SSH "$node" "docker rm -f $name 2>/dev/null; docker run -d --name $name $dra $BL_DOCKER_IMG tail -f /dev/null" 2>/dev/null
    local host_lib=$($BL_SSH "$node" "ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* 2>/dev/null | head -1" 2>/dev/null)
    if [[ -n "$host_lib" ]]; then
        $BL_SSH "$node" "docker cp $host_lib $name:/usr/lib/x86_64-linux-gnu/libionic.so.1" 2>/dev/null
    fi
    echo "  Container $name on $node"
}

bl_cleanup() {
    local name=${1:-$BL_CONTAINER}
    shift || true
    for node in "$@"; do
        $BL_SSH "$node" "docker rm -f $name 2>/dev/null" 2>/dev/null &
    done
    wait
}

# ── Wait for server ──

bl_wait_server() {
    local node=$1 port=${2:-8000} timeout=${3:-900} path=${4:-"/v1/models"} pattern=${5:-"DeepSeek|model"}
    for i in $(seq 1 "$timeout"); do
        if $BL_SSH "$node" "docker exec $BL_CONTAINER curl -s -m 5 http://localhost:${port}${path} 2>/dev/null | grep -qiE '$pattern'" 2>/dev/null; then
            echo "  $node:$port READY (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: $node:$port not ready after ${timeout}s"
    return 1
}

# ── SGLang server launch (config-driven) ──

bl_launch_sglang_prefill() {
    local node=$1
    $BL_SSH "$node" "docker exec -d $BL_CONTAINER bash -c '
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$BL_MORI_DISPATCH_PREFILL
export GLOO_SOCKET_IFNAME=\$(ip route | grep \"^default\" | awk \"{print \\\$5}\" | head -1)
export NCCL_SOCKET_IFNAME=\$GLOO_SOCKET_IFNAME
python3 -m sglang.launch_server \
    --model-path $BL_MODEL --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --disaggregation-mode prefill --disaggregation-ib-device $BL_IBDEVICES \
    $BL_PREFILL_FLAGS \
    --log-level-http warning \
    > /tmp/prefill.log 2>&1
'" 2>/dev/null && echo "  SGLang prefill on $node"
}

bl_launch_sglang_decode() {
    local node=$1
    $BL_SSH "$node" "docker exec -d $BL_CONTAINER bash -c '
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$BL_MORI_DISPATCH_DECODE
export GLOO_SOCKET_IFNAME=\$(ip route | grep \"^default\" | awk \"{print \\\$5}\" | head -1)
export NCCL_SOCKET_IFNAME=\$GLOO_SOCKET_IFNAME
python3 -m sglang.launch_server \
    --model-path $BL_MODEL --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --disaggregation-mode decode --disaggregation-ib-device $BL_IBDEVICES \
    $BL_DECODE_FLAGS \
    --log-level-http warning \
    > /tmp/decode.log 2>&1
'" 2>/dev/null && echo "  SGLang decode on $node"
}

bl_launch_sglang_router() {
    local router_node=$1
    shift
    local prefill_args="" decode_args=""
    local mode="prefill"
    for arg in "$@"; do
        if [[ "$arg" == "--decode" ]]; then
            mode="decode"; continue
        fi
        if [[ "$mode" == "prefill" ]]; then
            prefill_args+=" --prefill http://${arg}:8000"
        else
            decode_args+=" --decode http://${arg}:8000"
        fi
    done
    $BL_SSH "$router_node" "docker exec -d $BL_CONTAINER python3 -m sglang_router.launch_router \
        --pd-disaggregation --port 30000 \
        --policy random --prefill-policy random --decode-policy random \
        $prefill_args $decode_args" 2>/dev/null
    sleep 10
    local rstat=$($BL_SSH "$router_node" "docker exec $BL_CONTAINER curl -s http://localhost:30000/readiness" 2>/dev/null)
    echo "  Router: $rstat"
    echo "$rstat" | grep -q '"ready"'
}

# ── Dynamo server launch (config-driven) ──

bl_launch_dynamo_prefill() {
    local node=$1 etcd_ip=$2
    local is_primary=${3:-true}
    local startup=""
    if [[ "$is_primary" == "true" ]]; then
        startup="
rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://\${MY_IP}:2379 > /tmp/etcd.log 2>&1 &
nats-server -p 4222 -js --client_advertise \${MY_IP}:4222 > /tmp/nats.log 2>&1 &
sleep 3
python3 -m dynamo.frontend \
    --http-port 8000 --router-mode kv --router-kv-events \
    --router-track-active-blocks --router-track-prefill-tokens \
    --router-queue-policy fcfs --enforce-disagg --request-plane tcp \
    > /tmp/frontend.log 2>&1 &
sleep 2"
    fi
    $BL_SSH "$node" "docker exec -d $BL_CONTAINER bash -c '
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$BL_MORI_DISPATCH_PREFILL
export GLOO_SOCKET_IFNAME=\$(ip route | grep \"^default\" | awk \"{print \\\$5}\" | head -1)
export NCCL_SOCKET_IFNAME=\$GLOO_SOCKET_IFNAME
export PYTHONHASHSEED=0
MY_IP=\$(ip route get 1.1.1.1 | awk \"/src/ {print \\\$7}\")
export ETCD_ENDPOINTS=http://${etcd_ip}:2379 NATS_SERVER=nats://${etcd_ip}:4222
${startup}
python3 -m dynamo.sglang \
    --model-path $BL_MODEL --trust-remote-code \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 --page-size 16 \
    --disaggregation-mode prefill --disaggregation-transfer-backend mori \
    --disaggregation-ib-device $BL_IBDEVICES --host 0.0.0.0 \
    $BL_PREFILL_FLAGS \
    --log-level-http warning \
    > /tmp/prefill.log 2>&1
'" 2>/dev/null && echo "  Dynamo prefill on $node${is_primary:+ (primary)}"
}

bl_launch_dynamo_decode() {
    local node=$1 etcd_ip=$2
    $BL_SSH "$node" "docker exec -d $BL_CONTAINER bash -c '
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$BL_MORI_DISPATCH_DECODE
export GLOO_SOCKET_IFNAME=\$(ip route | grep \"^default\" | awk \"{print \\\$5}\" | head -1)
export NCCL_SOCKET_IFNAME=\$GLOO_SOCKET_IFNAME
export PYTHONHASHSEED=0
export ETCD_ENDPOINTS=http://${etcd_ip}:2379 NATS_SERVER=nats://${etcd_ip}:4222
python3 -m dynamo.sglang \
    --model-path $BL_MODEL --trust-remote-code \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 --page-size 16 \
    --disaggregation-mode decode --disaggregation-transfer-backend mori \
    --disaggregation-ib-device $BL_IBDEVICES --host 0.0.0.0 \
    $BL_DECODE_FLAGS \
    --log-level-http warning \
    > /tmp/decode.log 2>&1
'" 2>/dev/null && echo "  Dynamo decode on $node"
}

# ── Benchmark runner ──

bl_run_bench() {
    local node=$1 tag=$2 conc=$3 port=${4:-8000} isl=${5:-1024} osl=${6:-1024}
    local num_prompts=$((conc * 10))
    local warmups=$((conc * 2))
    [[ $warmups -gt 512 ]] && warmups=512
    echo ">>> [$tag] c=$conc ISL=$isl OSL=$osl on $node:$port (${num_prompts} prompts, ${warmups} warmup)"
    $BL_SSH "$node" "docker exec $BL_CONTAINER python3 $BL_BENCH_SCRIPT \
        --backend openai \
        --base-url http://0.0.0.0:${port} \
        --model $BL_MODEL \
        --dataset-name random \
        --random-input-len $isl --random-output-len $osl \
        --random-range-ratio $BL_RANDOM_RANGE_RATIO \
        --num-prompts $num_prompts \
        --max-concurrency $conc \
        --request-rate inf \
        --ignore-eos --save-result \
        --num-warmups $warmups \
        --percentile-metrics ttft,tpot,itl,e2el \
        --result-dir $BL_RESULTS_DIR \
        --result-filename ${tag}" 2>/dev/null
}

# ── Sanity test ──

bl_sanity_test() {
    local node=$1 port=${2:-8000} max_retries=${3:-30}
    echo "Sanity test on $node:$port (up to ${max_retries} retries)..."
    for attempt in $(seq 1 "$max_retries"); do
        local resp=$($BL_SSH "$node" "docker exec $BL_CONTAINER curl -s -m 120 -X POST http://localhost:${port}/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -d '{\"model\":\"$BL_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5}'" 2>/dev/null)
        if echo "$resp" | grep -q '"content"'; then
            echo "  Inference OK! (attempt $attempt)"
            return 0
        fi
        echo "  Attempt $attempt: no response, workers may still be loading (retry in 30s)..."
        sleep 30
    done
    echo "  FAILED after $max_retries attempts"
    return 1
}
