#!/bin/bash
# Embedding model serving on ROCm with SGLang + Dynamo
#
# Uses --embedding-worker flag (not --is-embedding) for proper route registration.
#
# Prerequisites:
#   - AMD GPU available
#   - etcd + NATS running
#   - Model: Qwen/Qwen3-Embedding-4B
#
# Usage:
#   export ETCD_ENDPOINTS=http://localhost:2379
#   export NATS_SERVER=nats://localhost:4222
#   bash examples/backends/sglang/launch/rocm/agg_embed_rocm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../common/gpu_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-Embedding-4B}"
FRONTEND_PORT="${FRONTEND_PORT:-8000}"

export SGLANG_AITER_MLA_PERSIST=False
export RCCL_MSCCL_ENABLE=0

echo "=== Embedding Model on ROCm ==="
echo "Model: $MODEL"

python3 -m dynamo.frontend --http-port "$FRONTEND_PORT" --router-mode round-robin &
sleep 2

HIP_VISIBLE_DEVICES=0 python3 -m dynamo.sglang \
    --model-path "$MODEL" --tp-size 1 --trust-remote-code \
    --embedding-worker &

echo "Waiting for embedding endpoint..."
for i in $(seq 1 120); do
    status=$(curl -sf -o /dev/null -w '%{http_code}' \
        "http://localhost:$FRONTEND_PORT/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"input\":\"test\"}" 2>/dev/null || echo "000")
    [ "$status" != "404" ] && [ "$status" != "000" ] && break
    sleep 2
done

echo ""
echo "=== Embedding Endpoint Ready ==="
echo "Test: curl http://localhost:$FRONTEND_PORT/v1/embeddings -H 'Content-Type: application/json' -d '{\"model\":\"$MODEL\",\"input\":\"Hello world\"}'"

wait
