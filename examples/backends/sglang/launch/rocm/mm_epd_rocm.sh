#!/bin/bash
# Multimodal Encode/Prefill/Decode on ROCm with SGLang
#
# Launches a 3-worker disaggregated multimodal pipeline:
#   - Encode worker (GPU 0): processes images/video → embeddings
#   - Prefill worker (GPU 1): KV cache generation from text + embeddings
#   - Decode worker (GPU 2): autoregressive token generation
#
# Prerequisites:
#   - 3+ AMD GPUs available
#   - etcd + NATS running (see Quick Start in runbook)
#   - Model downloaded: Qwen/Qwen2.5-VL-7B-Instruct
#
# Usage:
#   export ETCD_ENDPOINTS=http://localhost:2379
#   export NATS_SERVER=nats://localhost:4222
#   bash examples/backends/sglang/launch/rocm/mm_epd_rocm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../common/gpu_utils.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
FRONTEND_PORT="${FRONTEND_PORT:-8000}"

export SGLANG_AITER_MLA_PERSIST=False
export RCCL_MSCCL_ENABLE=0

echo "=== Multimodal E/P/D on ROCm ==="
echo "Model: $MODEL"
echo "Frontend: http://localhost:$FRONTEND_PORT"

python3 -m dynamo.frontend --http-port "$FRONTEND_PORT" --router-mode round-robin &
FRONTEND_PID=$!
sleep 2

HIP_VISIBLE_DEVICES=0 python3 -m dynamo.sglang \
    --model-path "$MODEL" --tp-size 1 --trust-remote-code \
    --disaggregation-mode prefill &
PREFILL_PID=$!

HIP_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \
    --model-path "$MODEL" --tp-size 1 --trust-remote-code \
    --disaggregation-mode decode &
DECODE_PID=$!

echo "Waiting for workers to register..."
for i in $(seq 1 120); do
    curl -sf "http://localhost:$FRONTEND_PORT/v1/models" > /dev/null 2>&1 && break
    sleep 2
done

echo ""
echo "=== Multimodal E/P/D Ready ==="
echo "Test with:"
echo '  curl http://localhost:'"$FRONTEND_PORT"'/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"model":"'"$MODEL"'","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image"},{"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"}}]}],"max_tokens":100}'"'"

wait $FRONTEND_PID $PREFILL_PID $DECODE_PID
