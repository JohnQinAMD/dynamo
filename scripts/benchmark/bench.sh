#!/usr/bin/env bash
# Benchmark runner aligned with InferenceX parameters.
#
# Usage:
#   BENCH_PORT=9000 MODEL_NAME=DeepSeek-R1-0528 bash bench.sh
#
# Env vars (set by server.sh or caller):
#   MODEL_DIR, MODEL_NAME, BENCH_PORT
#   BENCH_INPUT_LEN, BENCH_OUTPUT_LEN
#   BENCH_MAX_CONCURRENCY (space-separated or x-separated list)
#   BENCH_REQUEST_RATE (default: inf)
#   BENCH_RANDOM_RANGE_RATIO (default: 0.8)
#   BENCH_NUM_PROMPTS_MULTIPLIER (default: 10)
#   RESULTS_DIR (default: /workspace/dynamo/docs/infx_bench_results)
#   RESULT_PREFIX (default: benchmark)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODEL_DIR:-/models}/${MODEL_NAME:-DeepSeek-R1-0528}"
PORT="${BENCH_PORT:-30000}"
ISL="${BENCH_INPUT_LEN:-1024}"
OSL="${BENCH_OUTPUT_LEN:-1024}"
RATE="${BENCH_REQUEST_RATE:-inf}"
RATIO="${BENCH_RANDOM_RANGE_RATIO:-0.8}"
MULT="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
RESULTS="${RESULTS_DIR:-/workspace/dynamo/docs/infx_bench_results}"
PREFIX="${RESULT_PREFIX:-benchmark}"

# Parse concurrency list (supports "4 8 16" or "4x8x16")
CONC_STR="${BENCH_MAX_CONCURRENCY:-4 8 16 32 64 128 256}"
CONC_LIST=(${CONC_STR//x/ })

# Find benchmark_serving.py
BENCH_PY=""
for candidate in \
    "/workspace/InferenceX/utils/bench_serving/benchmark_serving.py" \
    "$(python3 -c 'import sglang,os; print(os.path.join(os.path.dirname(sglang.__file__),"..","benchmark","bench_serving.py"))' 2>/dev/null)" \
    "$(which benchmark_serving.py 2>/dev/null)"; do
    [[ -f "$candidate" ]] && { BENCH_PY="$candidate"; break; }
done

if [[ -z "$BENCH_PY" ]]; then
    echo "[ERROR] benchmark_serving.py not found"
    exit 1
fi

mkdir -p "$RESULTS"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Benchmark: ${PREFIX} ISL=${ISL} OSL=${OSL}              "
echo "║  Model: ${MODEL_NAME:-unknown}                            "
echo "║  Port: ${PORT}  Concurrencies: ${CONC_LIST[*]}           "
echo "╚══════════════════════════════════════════════════════════╝"

for conc in "${CONC_LIST[@]}"; do
    num_prompts=$((conc * MULT))
    warmups=$((conc * 2))
    tag="${PREFIX}_c${conc}"
    echo ">>> $tag (n=$num_prompts, warmup=$warmups)"

    python3 "$BENCH_PY" \
        --backend vllm \
        --base-url "http://0.0.0.0:${PORT}" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio "$RATIO" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$conc" \
        --request-rate "$RATE" \
        --ignore-eos \
        --save-result \
        --num-warmups "$warmups" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --result-dir "$RESULTS" \
        --result-filename "$tag" \
        2>&1 | grep -E "Successful|throughput|TPOT|TTFT"

    echo ""
done

echo "Results saved to: $RESULTS/${PREFIX}_c*"
