#!/usr/bin/env bash
# InferenceX-aligned benchmarks: Dynamo vs non-Dynamo on MI355X
# Test 1: Aggregated single-node (Dynamo SGLang vs SGLang) — 2 nodes in parallel
# Test 2: Disaggregated 1P1D MoRI (Dynamo SGLang vs MoRI SGLang) — 2x2 nodes in parallel
#
# All parameters aligned with InferenceX official:
#   - ROCM_QUICK_REDUCE_QUANTIZATION=INT4
#   - --random-range-ratio 0.8
#   - --backend vllm --ignore-eos --num-warmups 2*CONC
#   - --cuda-graph-max-bs $CONC (dynamic per concurrency, matches InferenceX)
#   - --disable-radix-cache --mem-fraction-static 0.8/0.85
#   - DSR1 models.yaml base_flags for disagg
set -euo pipefail

DOCKER_IMG="amdprimus/dynamo-rocm-sglang:latest"
MODEL_FP8="/models/DeepSeek-R1-0528"
MODEL_HOST="/mnt/vast/john/huggingface"
WORKSPACE="/mnt/vast/john/rocm-dynamo"
HF_CACHE="/root/.cache/huggingface"
BENCH_SCRIPT="/workspace/InferenceX/utils/bench_serving/benchmark_serving.py"
RESULTS_DIR="/mnt/vast/john/rocm-dynamo/dynamo/docs/infx_bench_$(date +%Y%m%d_%H%M%S)"
RESULTS_SHARED="/mnt/vast/john/rocm-dynamo/dynamo/docs/infx_bench_results"

CONC_AGG="4 8 16 32 64"
CONC_DISAGG="1 2 4 8 16 32 64"
RANDOM_RANGE_RATIO=0.8
ISL=1024
OSL=1024

# InferenceX DSR1 models.yaml base_flags (disagg mode, no --disaggregation-transfer-backend here — set per-worker)
DISAGG_COMMON_FLAGS="--decode-log-interval 1000 --log-level warning --watchdog-timeout 3600 --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter"
IBDEVICES="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
MORI_RDMA_TC=104

# Node assignments (override via env vars)
NODE_AGG_SGLANG="${NODE_AGG_SGLANG:-chi2761}"
NODE_AGG_DYNAMO="${NODE_AGG_DYNAMO:-chi2863}"
NODE_DISAGG_SGLANG_P="${NODE_DISAGG_SGLANG_P:-chi2885}"
NODE_DISAGG_SGLANG_D="${NODE_DISAGG_SGLANG_D:-chi2896}"
NODE_DISAGG_DYNAMO_P="${NODE_DISAGG_DYNAMO_P:-chi2899}"
NODE_DISAGG_DYNAMO_D="${NODE_DISAGG_DYNAMO_D:-chi2900}"

mkdir -p "$RESULTS_DIR" "$RESULTS_SHARED"
echo "Results: $RESULTS_DIR"
echo "Shared results: $RESULTS_SHARED"
echo "Nodes: agg=$NODE_AGG_SGLANG/$NODE_AGG_DYNAMO  disagg=$NODE_DISAGG_SGLANG_P+$NODE_DISAGG_SGLANG_D / $NODE_DISAGG_DYNAMO_P+$NODE_DISAGG_DYNAMO_D"

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"

DOCKER_RUN_ARGS="--network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v ${WORKSPACE}:/workspace -v ${MODEL_HOST}:/models -v ${HF_CACHE}:${HF_CACHE} \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e SGLANG_USE_AITER=1 -e RCCL_MSCCL_ENABLE=0 \
    -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
    -e MORI_RDMA_TC=${MORI_RDMA_TC}"

start_container() {
    local node=$1 name=$2
    $SSH "$node" "docker rm -f $name 2>/dev/null; docker run -d --name $name $DOCKER_RUN_ARGS $DOCKER_IMG tail -f /dev/null" 2>/dev/null
    echo "  Container $name started on $node"
    fix_libionic "$node" "$name"
}

fix_libionic() {
    local node=$1 name=$2
    local host_lib=$($SSH "$node" "ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* 2>/dev/null | head -1" 2>/dev/null)
    if [[ -z "$host_lib" ]]; then return; fi
    $SSH "$node" "docker cp $host_lib $name:/usr/lib/x86_64-linux-gnu/libionic.so.1" 2>/dev/null
    local dev_count=$($SSH "$node" "docker exec $name ibv_devinfo 2>/dev/null | grep -c hca_id" 2>/dev/null)
    if [[ "$dev_count" -ge 8 ]]; then
        echo "  libionic ABI fixed on $node ($name): $dev_count devices"
    else
        echo "  [WARN] libionic fix on $node: only $dev_count devices visible (expected 8)"
    fi
}

wait_server() {
    local node=$1 port=$2 timeout=${3:-900} check_path=${4:-"/v1/models"}
    echo "  Waiting for server on $node:$port..."
    for i in $(seq 1 "$timeout"); do
        if $SSH "$node" "curl -s http://localhost:${port}${check_path} 2>/dev/null | grep -qiE 'DeepSeek|model|ok|healthy'" 2>/dev/null; then
            echo "  Server ready on $node:$port after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server not ready on $node:$port after ${timeout}s"
    return 1
}

run_bench() {
    local node=$1 container=$2 tag=$3 conc=$4 port=${5:-8000}
    echo ">>> [$tag] conc=$conc on $node:$port"
    $SSH "$node" "docker exec $container bash -c '
        python3 $BENCH_SCRIPT \
            --backend vllm \
            --base-url http://0.0.0.0:${port} \
            --model $MODEL_FP8 \
            --dataset-name random \
            --random-input-len $ISL --random-output-len $OSL \
            --random-range-ratio $RANDOM_RANGE_RATIO \
            --num-prompts $((conc * 10)) \
            --max-concurrency $conc \
            --request-rate inf \
            --ignore-eos \
            --save-result \
            --num-warmups $((2 * conc)) \
            --percentile-metrics ttft,tpot,itl,e2el \
            --result-dir /workspace/dynamo/docs/infx_bench_results \
            --result-filename ${tag}_c${conc}
    '" 2>&1 | tee "$RESULTS_DIR/${tag}_c${conc}.log"
}

MAX_CONC=$(echo $CONC_AGG | tr ' ' '\n' | sort -n | tail -1)

# ═══════════════════════════════════════════════════════════
#  PHASE 1: Aggregated (2 nodes in parallel)
#  InferenceX ref: dsr1_fp8_mi355x.sh + benchmark-tmpl.yml
# ═══════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  PHASE 1: Aggregated single-node (parallel)          ║"
echo "║  1a: SGLang on $NODE_AGG_SGLANG (InferenceX baseline) ║"
echo "║  1b: Dynamo on $NODE_AGG_DYNAMO (test group)          ║"
echo "╚═══════════════════════════════════════════════════════╝"

# Start containers
start_container "$NODE_AGG_SGLANG" "bench-sglang" &
start_container "$NODE_AGG_DYNAMO" "bench-dynamo" &
wait

# 1a: SGLang server — exact match of InferenceX dsr1_fp8_mi355x.sh
# --cuda-graph-max-bs=$MAX_CONC: InferenceX uses $CONC (per-run), but we launch once
# with max conc to avoid restarting server per concurrency level
$SSH "$NODE_AGG_SGLANG" "docker exec -d bench-sglang bash -c '
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4
python3 -m sglang.launch_server \
    --model-path $MODEL_FP8 --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 8 --trust-remote-code \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 196608 --mem-fraction-static 0.8 \
    --disable-radix-cache --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 --cuda-graph-max-bs $MAX_CONC \
    > /tmp/server.log 2>&1
'" 2>/dev/null && echo "  SGLang server launching on $NODE_AGG_SGLANG"

# 1b: Dynamo server — same SGLang engine params, Dynamo routing on top
$SSH "$NODE_AGG_DYNAMO" "docker exec -d bench-dynamo bash -c '
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4
rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://127.0.0.1:2379 > /tmp/etcd.log 2>&1 &
nats-server -p 4222 -js > /tmp/nats.log 2>&1 &
sleep 3
export ETCD_ENDPOINTS=http://127.0.0.1:2379 NATS_SERVER=nats://127.0.0.1:4222
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin > /tmp/frontend.log 2>&1 &
sleep 2
python3 -m dynamo.sglang \
    --model-path $MODEL_FP8 --tp-size 8 --trust-remote-code \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 --page-size 16 \
    --chunked-prefill-size 196608 --mem-fraction-static 0.8 \
    --disable-radix-cache --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 \
    > /tmp/worker.log 2>&1
'" 2>/dev/null && echo "  Dynamo server launching on $NODE_AGG_DYNAMO"

# Wait for both servers
wait_server "$NODE_AGG_SGLANG" 8000 900 &
wait_server "$NODE_AGG_DYNAMO" 8000 900 &
wait
echo "Both aggregated servers ready!"

# Run benchmarks in parallel (same conc, different nodes)
for conc in $CONC_AGG; do
    run_bench "$NODE_AGG_SGLANG" "bench-sglang" "sglang_agg" "$conc" &
    run_bench "$NODE_AGG_DYNAMO" "bench-dynamo" "dynamo_agg" "$conc" &
    wait
    echo "  Completed conc=$conc"
done

echo "Phase 1 complete!"
$SSH "$NODE_AGG_SGLANG" "docker stop bench-sglang" 2>/dev/null &
$SSH "$NODE_AGG_DYNAMO" "docker stop bench-dynamo" 2>/dev/null &
wait

# ═══════════════════════════════════════════════════════════
#  PHASE 2: Disaggregated 1P1D MoRI RDMA (2x2 nodes)
#  InferenceX ref: dsr1_fp8_mi355x_sglang-disagg.sh + amd_utils/server.sh + models.yaml
# ═══════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  PHASE 2: Disaggregated 1P1D MoRI RDMA (parallel)    ║"
echo "║  2a: MoRI SGLang on $NODE_DISAGG_SGLANG_P+$NODE_DISAGG_SGLANG_D  ║"
echo "║  2b: Dynamo KV on $NODE_DISAGG_DYNAMO_P+$NODE_DISAGG_DYNAMO_D    ║"
echo "╚═══════════════════════════════════════════════════════╝"

# Start containers on all 4 disagg nodes
for node in $NODE_DISAGG_SGLANG_P $NODE_DISAGG_SGLANG_D $NODE_DISAGG_DYNAMO_P $NODE_DISAGG_DYNAMO_D; do
    start_container "$node" "bench-disagg" &
done
wait

# Setup ionic networking on all disagg nodes (CRITICAL for MoRI RDMA)
# 1. Fix ABI: docker cp done in start_container() via fix_libionic()
# 2. Assign IPv4 to all 8 ionic ports (required for ibv_modify_qp)
echo "  Setting up ionic networking on disagg nodes..."
for node in $NODE_DISAGG_SGLANG_P $NODE_DISAGG_SGLANG_D $NODE_DISAGG_DYNAMO_P $NODE_DISAGG_DYNAMO_D; do
    $SSH "$node" "bash /mnt/vast/john/rocm-dynamo/dynamo/scripts/setup_ionic_network.sh 2>/dev/null && echo '$node: ionic configured'" 2>/dev/null &
done
wait
echo "  Ionic networking configured"

# Get IPs — must be from INSIDE containers (host network, so same as host IP)
get_ip() { $SSH "$1" "hostname -I | awk '{print \$1}'" 2>/dev/null; }
IP_SGLANG_P=$(get_ip "$NODE_DISAGG_SGLANG_P")
IP_SGLANG_D=$(get_ip "$NODE_DISAGG_SGLANG_D")
IP_DYNAMO_P=$(get_ip "$NODE_DISAGG_DYNAMO_P")
IP_DYNAMO_D=$(get_ip "$NODE_DISAGG_DYNAMO_D")
echo "  MoRI SGLang: P=$NODE_DISAGG_SGLANG_P($IP_SGLANG_P) D=$NODE_DISAGG_SGLANG_D($IP_SGLANG_D)"
echo "  Dynamo:      P=$NODE_DISAGG_DYNAMO_P($IP_DYNAMO_P) D=$NODE_DISAGG_DYNAMO_D($IP_DYNAMO_D)"

# Verify IPs are non-empty
for var_name in IP_SGLANG_P IP_SGLANG_D IP_DYNAMO_P IP_DYNAMO_D; do
    val="${!var_name}"
    if [ -z "$val" ]; then
        echo "ERROR: Failed to get IP for $var_name"
        exit 1
    fi
done

# --- 2a: MoRI SGLang disagg (InferenceX baseline, random router) ---
# Prefill node
$SSH "$NODE_DISAGG_SGLANG_P" "docker exec -d bench-disagg bash -c '
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export MORI_RDMA_TC=$MORI_RDMA_TC
python3 -m sglang.launch_server \
    --model-path $MODEL_FP8 --host 0.0.0.0 --port 8000 \
    --trust-remote-code --tp-size 8 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device $IBDEVICES \
    $DISAGG_COMMON_FLAGS \
    --mem-fraction-static 0.8 --disable-radix-cache \
    --max-running-requests 512 --chunked-prefill-size 262144 \
    --cuda-graph-max-bs 512 \
    > /tmp/prefill.log 2>&1
'" 2>/dev/null && echo "  MoRI prefill launching on $NODE_DISAGG_SGLANG_P"

# Decode node
$SSH "$NODE_DISAGG_SGLANG_D" "docker exec -d bench-disagg bash -c '
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export MORI_RDMA_TC=$MORI_RDMA_TC
python3 -m sglang.launch_server \
    --model-path $MODEL_FP8 --host 0.0.0.0 --port 8000 \
    --trust-remote-code --tp-size 8 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device $IBDEVICES \
    $DISAGG_COMMON_FLAGS \
    --mem-fraction-static 0.85 \
    --max-running-requests 512 --chunked-prefill-size 262144 \
    --prefill-round-robin-balance \
    --cuda-graph-max-bs 512 \
    > /tmp/decode.log 2>&1
'" 2>/dev/null && echo "  MoRI decode launching on $NODE_DISAGG_SGLANG_D"

# Wait for both P/D servers before starting router
echo "  Waiting for MoRI P/D servers (model load ~15 min)..."
wait_server "$NODE_DISAGG_SGLANG_P" 8000 900 &
wait_server "$NODE_DISAGG_SGLANG_D" 8000 900 &
wait

# SGLang PD router on prefill node (random policy — InferenceX baseline)
$SSH "$NODE_DISAGG_SGLANG_P" "docker exec -d bench-disagg bash -c '
python3 -m sglang_router.launch_router \
    --pd-disaggregation --port 30000 \
    --policy random --prefill-policy random --decode-policy random \
    --prefill http://${IP_SGLANG_P}:8000 \
    --decode http://${IP_SGLANG_D}:8000 \
    > /tmp/router.log 2>&1
'" 2>/dev/null && echo "  MoRI router launching on $NODE_DISAGG_SGLANG_P:30000"

# --- 2b: Dynamo SGLang disagg (KV-aware routing) ---
# Prefill node: etcd + NATS + Dynamo frontend + Dynamo prefill worker
$SSH "$NODE_DISAGG_DYNAMO_P" "docker exec -d bench-disagg bash -c '
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export MORI_RDMA_TC=$MORI_RDMA_TC
export PYTHONHASHSEED=0
MY_IP=\$(hostname -I | awk \"{print \\\$1}\")

rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://\${MY_IP}:2379 > /tmp/etcd.log 2>&1 &
nats-server -p 4222 -js --client_advertise \${MY_IP}:4222 > /tmp/nats.log 2>&1 &
sleep 3
export ETCD_ENDPOINTS=http://\${MY_IP}:2379 NATS_SERVER=nats://\${MY_IP}:4222

# Dynamo frontend: KV-aware router with load tracking
python3 -m dynamo.frontend \
    --http-port 8000 \
    --router-mode kv \
    --router-kv-events \
    --router-track-active-blocks \
    --router-track-prefill-tokens \
    --router-queue-policy fcfs \
    --enforce-disagg \
    --request-plane tcp \
    > /tmp/frontend.log 2>&1 &
sleep 2

# Dynamo prefill worker — same engine params as MoRI baseline
python3 -m dynamo.sglang \
    --model-path $MODEL_FP8 --tp-size 8 --trust-remote-code \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 --page-size 16 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device $IBDEVICES \
    --host 0.0.0.0 \
    --mem-fraction-static 0.8 --disable-radix-cache \
    --max-running-requests 512 --chunked-prefill-size 262144 \
    --decode-log-interval 1000 --watchdog-timeout 3600 \
    --load-balance-method round_robin \
    > /tmp/prefill.log 2>&1
'" 2>/dev/null && echo "  Dynamo prefill launching on $NODE_DISAGG_DYNAMO_P (KV router)"

# Decode node: connects to prefill node's etcd/NATS
$SSH "$NODE_DISAGG_DYNAMO_D" "docker exec -d bench-disagg bash -c '
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export MORI_RDMA_TC=$MORI_RDMA_TC
export PYTHONHASHSEED=0
export ETCD_ENDPOINTS=http://${IP_DYNAMO_P}:2379 NATS_SERVER=nats://${IP_DYNAMO_P}:4222

# Dynamo decode worker — same engine params as MoRI baseline
python3 -m dynamo.sglang \
    --model-path $MODEL_FP8 --tp-size 8 --trust-remote-code \
    --attention-backend aiter --kv-cache-dtype fp8_e4m3 --page-size 16 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device $IBDEVICES \
    --host 0.0.0.0 \
    --mem-fraction-static 0.85 \
    --max-running-requests 512 \
    --prefill-round-robin-balance \
    --decode-log-interval 1000 --watchdog-timeout 3600 \
    > /tmp/decode.log 2>&1
'" 2>/dev/null && echo "  Dynamo decode launching on $NODE_DISAGG_DYNAMO_D (KV router)"

# Wait for both disagg setups
echo "Waiting for disagg servers (model load ~10-15 min)..."
wait_server "$NODE_DISAGG_SGLANG_P" 30000 900 "/readiness" &
wait_server "$NODE_DISAGG_DYNAMO_P" 8000 900 &
wait
echo "Both disagg setups ready!"

# Run disagg benchmarks (parallel: MoRI vs Dynamo at each concurrency)
for conc in $CONC_DISAGG; do
    run_bench "$NODE_DISAGG_SGLANG_P" "bench-disagg" "mori_sglang_disagg" "$conc" 30000 &
    run_bench "$NODE_DISAGG_DYNAMO_P" "bench-disagg" "dynamo_disagg" "$conc" 8000 &
    wait
    echo "  Completed disagg conc=$conc"
done

echo "Phase 2 complete!"
for node in $NODE_DISAGG_SGLANG_P $NODE_DISAGG_SGLANG_D $NODE_DISAGG_DYNAMO_P $NODE_DISAGG_DYNAMO_D; do
    $SSH "$node" "docker stop bench-disagg" 2>/dev/null &
done
wait

# ═══════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  ALL TESTS COMPLETE                                   ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo "Results log: $RESULTS_DIR"
echo "Result JSONs: $RESULTS_SHARED"
echo ""
echo "Expected files:"
echo "  Agg:    sglang_agg_c{4,8,16,32,64}.json  dynamo_agg_c{4,8,16,32,64}.json"
echo "  Disagg: mori_sglang_disagg_c{1,2,4,8,16,32,64}.json  dynamo_disagg_c{1,2,4,8,16,32,64}.json"

# Release salloc jobs
echo ""
echo "Releasing salloc jobs..."
for jobid in $(squeue -u root --format="%i %j" --noheader | grep infx | awk '{print $1}'); do
    scancel "$jobid" 2>/dev/null
done
echo "Done."
