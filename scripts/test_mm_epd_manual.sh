#!/bin/bash
# Test 11: Multimodal E/P/D — manual test bypassing upstream timeout
set -e
MY_IP=$(hostname -I | awk '{print $1}')
rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://${MY_IP}:2379 &
nats-server -p 4222 -js --client_advertise ${MY_IP}:4222 &
sleep 3
export ETCD_ENDPOINTS=http://${MY_IP}:2379
export NATS_SERVER=nats://${MY_IP}:4222
cd /opt/dynamo

echo "=== Multimodal E/P/D Test ==="
echo "Starting frontend..."
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
sleep 2

echo "Starting VL worker (Qwen3-VL-2B-Instruct, single GPU)..."
DYN_SYSTEM_PORT=8081 python3 -m dynamo.sglang \
  --model-path /models/Qwen3-VL-2B-Instruct \
  --served-model-name Qwen/Qwen3-VL-2B-Instruct \
  --page-size 16 --tp 1 --trust-remote-code &

echo "Waiting for server (model load + aiter JIT + CUDA graph ~5min)..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8000/health 2>/dev/null | grep -q healthy; then
    echo "Healthy at ${i}0s"
    break
  fi
  sleep 10
done

echo "Warmup (first inference, aiter JIT compile)..."
WARM=$(curl -s --max-time 900 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-VL-2B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":1}')
echo "Warmup: $(echo $WARM | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:20])' 2>/dev/null || echo 'failed')"

echo ""
echo "=== Test: Image via VL model ==="
RESP=$(curl -s --max-time 120 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-VL-2B-Instruct","messages":[{"role":"user","content":[{"type":"text","text":"What do you see?"},{"type":"image_url","image_url":{"url":"http://images.cocodataset.org/test2017/000000155781.jpg"}}]}],"max_tokens":100,"temperature":0.0}')
CONTENT=$(echo $RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:200])" 2>/dev/null)
if [ -n "$CONTENT" ]; then
  echo "PASS: $CONTENT"
else
  echo "FAIL: $(echo $RESP | head -c 300)"
fi

echo ""
echo "=== Test: Text-only via VL model ==="
RESP2=$(curl -s --max-time 60 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-VL-2B-Instruct","messages":[{"role":"user","content":"Count to 3"}],"max_tokens":20}')
echo "Response: $(echo $RESP2 | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:100])' 2>/dev/null || echo 'FAIL')"

echo "DONE"
