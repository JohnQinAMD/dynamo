#!/usr/bin/env bash
# Unified config-driven benchmark runner for Dynamo vs SGLang on MI355X.
# Reads model config from InferenceX models.yaml — no manual flag duplication.
#
# Usage:
#   # STP no_dp (default, matches our proven 1P2D runs):
#   ./run_benchmark.sh -f sglang -t 1p2d -c 64 -n chi2863,chi2870,chi2900
#
#   # Match InferenceX EP+DP+MTP config:
#   ./run_benchmark.sh -f dynamo -t 2p1d --ep --dp --mtp 1 -c 1024 --isl 8192 -n chi2863,chi2870,chi2900
#
#   # Run both frameworks:
#   ./run_benchmark.sh -f both -t 1p2d -c "4 8 16 32 64" -n chi2863,chi2870,chi2900
#
#   # Custom TP per role:
#   ./run_benchmark.sh -f sglang --prefill-tp 4 --decode-tp 8 -t 1p2d -c 64 -n chi2863,chi2870,chi2900
#
#   # Preflight only:
#   ./run_benchmark.sh --preflight-only -n chi2863,chi2870,chi2900
#
# Topology:
#   1p2d  — 1 prefill + 2 decode (node1=P, node2=D, node3=D)
#   2p1d  — 2 prefill + 1 decode (node1=P, node2=P, node3=D)
#   1p1d  — 1 prefill + 1 decode (node1=P, node2=D)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/benchmark_lib.sh"
source "$SCRIPT_DIR/preflight_check.sh"

# ── Defaults ──
FRAMEWORK="sglang"
TOPOLOGY="1p2d"
CONCURRENCIES="64"
ISL=1024
OSL=1024
NODES=""
PREFLIGHT_ONLY=false
SKIP_PREFLIGHT=false
TAG_PREFIX=""

# InferenceX config knobs (passed through to infx_config.py)
MODEL_NAME="DeepSeek-R1-0528"
PREFILL_TP=8
DECODE_TP=8
ENABLE_EP=false
ENABLE_DP=false
PREFILL_EP=""
DECODE_EP=""
PREFILL_DP=""
DECODE_DP=""
MTP=0

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --framework|-f)   FRAMEWORK="$2";       shift 2 ;;
        --topology|-t)    TOPOLOGY="$2";         shift 2 ;;
        -c|--concurrency) CONCURRENCIES="$2";    shift 2 ;;
        --isl)            ISL="$2";              shift 2 ;;
        --osl)            OSL="$2";              shift 2 ;;
        --nodes|-n)       NODES="$2";            shift 2 ;;
        --tag)            TAG_PREFIX="$2";       shift 2 ;;
        --model)          MODEL_NAME="$2";       shift 2 ;;
        --prefill-tp)     PREFILL_TP="$2";       shift 2 ;;
        --decode-tp)      DECODE_TP="$2";        shift 2 ;;
        --ep)             ENABLE_EP=true;        shift ;;
        --dp)             ENABLE_DP=true;        shift ;;
        --prefill-ep)     PREFILL_EP="$2";       shift 2 ;;
        --decode-ep)      DECODE_EP="$2";        shift 2 ;;
        --prefill-dp)     PREFILL_DP="$2";       shift 2 ;;
        --decode-dp)      DECODE_DP="$2";        shift 2 ;;
        --mtp)            MTP="$2";              shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=true;   shift ;;
        --skip-preflight) SKIP_PREFLIGHT=true;   shift ;;
        -h|--help)
            sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Resolve EP/DP: --ep/--dp shorthand overrides per-role flags
[[ "$ENABLE_EP" == "true" && -z "$PREFILL_EP" ]] && PREFILL_EP="true"
[[ "$ENABLE_EP" == "true" && -z "$DECODE_EP" ]] && DECODE_EP="true"
[[ "$ENABLE_DP" == "true" && -z "$PREFILL_DP" ]] && PREFILL_DP="true"
[[ "$ENABLE_DP" == "true" && -z "$DECODE_DP" ]] && DECODE_DP="true"
PREFILL_EP="${PREFILL_EP:-false}"
DECODE_EP="${DECODE_EP:-false}"
PREFILL_DP="${PREFILL_DP:-false}"
DECODE_DP="${DECODE_DP:-false}"

# ── Validate ──
if [[ -z "$NODES" ]]; then
    echo "ERROR: --nodes required (comma-separated, e.g. chi2863,chi2870,chi2900)"
    exit 1
fi

IFS=',' read -ra NODE_LIST <<< "$NODES"
NODE_COUNT=${#NODE_LIST[@]}

case "$TOPOLOGY" in
    1p2d) [[ $NODE_COUNT -ge 3 ]] || { echo "ERROR: 1p2d needs 3 nodes"; exit 1; } ;;
    2p1d) [[ $NODE_COUNT -ge 3 ]] || { echo "ERROR: 2p1d needs 3 nodes"; exit 1; } ;;
    1p1d) [[ $NODE_COUNT -ge 2 ]] || { echo "ERROR: 1p1d needs 2 nodes"; exit 1; } ;;
    *)    echo "ERROR: Unknown topology $TOPOLOGY (use 1p2d, 2p1d, 1p1d)"; exit 1 ;;
esac

[[ "$FRAMEWORK" =~ ^(sglang|dynamo|both)$ ]] || { echo "ERROR: --framework must be sglang, dynamo, or both"; exit 1; }

# ── Load config from InferenceX models.yaml ──
echo "Loading config from InferenceX models.yaml..."
bl_load_config \
    --model "$MODEL_NAME" \
    --prefill-tp "$PREFILL_TP" --decode-tp "$DECODE_TP" \
    --prefill-ep "$PREFILL_EP" --decode-ep "$DECODE_EP" \
    --prefill-dp "$PREFILL_DP" --decode-dp "$DECODE_DP" \
    --mtp "$MTP"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  MI355X Disaggregated Benchmark (config-driven)          ║"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  Framework:    $FRAMEWORK"
echo "║  Model:        $MODEL_NAME"
echo "║  Topology:     $TOPOLOGY"
echo "║  ISL/OSL:      ${ISL}/${OSL}"
echo "║  Concurrency:  $CONCURRENCIES"
echo "║  Prefill:      TP=$PREFILL_TP EP=$PREFILL_EP DP=$PREFILL_DP"
echo "║  Decode:       TP=$DECODE_TP EP=$DECODE_EP DP=$DECODE_DP MTP=$MTP"
echo "║  Nodes:        ${NODE_LIST[*]}"
echo "╚═══════════════════════════════════════════════════════════╝"

# ── Preflight ──
if ! $SKIP_PREFLIGHT; then
    run_preflight "${NODE_LIST[@]}" || exit 1
fi
$PREFLIGHT_ONLY && exit 0

# ── Assign roles ──
case "$TOPOLOGY" in
    1p2d)
        PREFILL_NODES=("${NODE_LIST[0]}")
        DECODE_NODES=("${NODE_LIST[1]}" "${NODE_LIST[2]}")
        TOPO_TAG="${TOPOLOGY}"
        ;;
    2p1d)
        PREFILL_NODES=("${NODE_LIST[0]}" "${NODE_LIST[1]}")
        DECODE_NODES=("${NODE_LIST[2]}")
        TOPO_TAG="${TOPOLOGY}"
        ;;
    1p1d)
        PREFILL_NODES=("${NODE_LIST[0]}")
        DECODE_NODES=("${NODE_LIST[1]}")
        TOPO_TAG="${TOPOLOGY}"
        ;;
esac

ALL_NODES=("${PREFILL_NODES[@]}" "${DECODE_NODES[@]}")
PRIMARY="${PREFILL_NODES[0]}"
PRIMARY_IP=$(bl_get_ip "$PRIMARY")

echo ""
echo "Prefill: ${PREFILL_NODES[*]}"
echo "Decode:  ${DECODE_NODES[*]}"
echo "Primary: $PRIMARY ($PRIMARY_IP)"

# ── Run one framework ──
_run_framework() {
    local fw=$1
    local bench_port=8000

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Setting up $fw ($TOPOLOGY)"
    echo "══════════════════════════════════════════════════"

    for node in "${ALL_NODES[@]}"; do
        bl_start_container "$node"
    done

    if [[ "$fw" == "sglang" ]]; then
        for pn in "${PREFILL_NODES[@]}"; do
            bl_launch_sglang_prefill "$pn"
        done
        for dn in "${DECODE_NODES[@]}"; do
            bl_launch_sglang_decode "$dn"
        done

        echo "Waiting for servers..."
        for node in "${ALL_NODES[@]}"; do
            bl_wait_server "$node" 8000 || exit 1
        done

        local prefill_ips=() decode_ips=()
        for pn in "${PREFILL_NODES[@]}"; do
            prefill_ips+=("$(bl_get_ip "$pn")")
        done
        for dn in "${DECODE_NODES[@]}"; do
            decode_ips+=("$(bl_get_ip "$dn")")
        done
        bl_launch_sglang_router "$PRIMARY" "${prefill_ips[@]}" "--decode" "${decode_ips[@]}" || { echo "Router failed"; exit 1; }
        bench_port=30000

    elif [[ "$fw" == "dynamo" ]]; then
        bl_launch_dynamo_prefill "$PRIMARY" "$PRIMARY_IP" "true"
        for pn in "${PREFILL_NODES[@]:1}"; do
            bl_launch_dynamo_prefill "$pn" "$PRIMARY_IP" "false"
        done
        for dn in "${DECODE_NODES[@]}"; do
            bl_launch_dynamo_decode "$dn" "$PRIMARY_IP"
        done

        echo "Waiting for frontend..."
        bl_wait_server "$PRIMARY" 8000 || exit 1
        local extra_workers=$(( ${#PREFILL_NODES[@]} - 1 + ${#DECODE_NODES[@]} ))
        if [[ $extra_workers -gt 0 ]]; then
            echo "  Waiting 60s for $extra_workers additional worker(s) to register..."
            sleep 60
        fi
    fi

    bl_sanity_test "$PRIMARY" "$bench_port" || { echo "Sanity failed for $fw"; bl_cleanup "$BL_CONTAINER" "${ALL_NODES[@]}"; return 1; }

    for conc in $CONCURRENCIES; do
        local tag="${TAG_PREFIX:+${TAG_PREFIX}_}aligned_${fw}_${TOPO_TAG}_isl${ISL}_osl${OSL}_c${conc}"
        bl_run_bench "$PRIMARY" "$tag" "$conc" "$bench_port" "$ISL" "$OSL"
    done

    bl_cleanup "$BL_CONTAINER" "${ALL_NODES[@]}"
}

# ── Execute ──
if [[ "$FRAMEWORK" == "both" ]]; then
    _run_framework "sglang"
    _run_framework "dynamo"
else
    _run_framework "$FRAMEWORK"
fi

# ── Print results ──
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  RESULTS                                                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"

TOTAL_GPU=$(( (${#PREFILL_NODES[@]} + ${#DECODE_NODES[@]}) * BL_PREFILL_TP ))

python3 -c "
import json, os

results_dir = '${BL_WORKSPACE}/dynamo/docs/infx_bench_results'
total_gpu = $TOTAL_GPU

frameworks = '${FRAMEWORK}'.replace('both','sglang,dynamo').split(',')
concurrencies = [int(c) for c in '${CONCURRENCIES}'.split()]

print(f'{\"Config\":<60} {\"tput/gpu\":>9} {\"TPOT\":>8} {\"TTFT\":>8} {\"Intvty\":>8}')
print('-' * 95)

for conc in concurrencies:
    for fw in frameworks:
        tag = '${TAG_PREFIX:+${TAG_PREFIX}_}aligned_' + fw + '_${TOPO_TAG}_isl${ISL}_osl${OSL}_c' + str(conc)
        path = os.path.join(results_dir, tag)
        try:
            d = json.load(open(path))
            tput = d['total_token_throughput']
            tpot = d['median_tpot_ms']
            ttft = d['median_ttft_ms']
            print(f'{tag:<60} {tput/total_gpu:>9.1f} {tpot:>8.2f} {ttft:>8.0f} {1000/tpot:>8.1f}')
        except Exception:
            print(f'{tag:<60} (not found)')
" 2>/dev/null || true

echo "Done."
