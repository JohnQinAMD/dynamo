#!/bin/bash
set -e
MY_IP=$(hostname -I | awk '{print $1}')
rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://${MY_IP}:2379 &
nats-server -p 4222 -js --client_advertise ${MY_IP}:4222 &
sleep 3
export ETCD_ENDPOINTS=http://${MY_IP}:2379
export NATS_SERVER=nats://${MY_IP}:4222
cd /opt/dynamo

python3 -m dynamo.frontend --http-port 8000 &
sleep 2

DYN_SYSTEM_PORT=8081 python3 -m dynamo.sglang \
  --embedding-worker \
  --model-path /models/Qwen3-Embedding-4B \
  --served-model-name Qwen/Qwen3-Embedding-4B \
  --page-size 16 --tp 1 --trust-remote-code --use-sglang-tokenizer &

echo "Waiting for healthy..."
for i in $(seq 1 120); do
  curl -s http://localhost:8000/health 2>/dev/null | grep -q healthy && echo "Healthy at ${i}0s" && break
  sleep 10
done

echo "Warmup (may take 5+ min for aiter JIT + CUDA graph)..."
curl -s --max-time 900 http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":"warmup"}' > /tmp/warmup.json 2>&1
echo "Warmup done"
cat /tmp/warmup.json | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'warmup dim={len(r[\"data\"][0][\"embedding\"])}')" 2>/dev/null || echo "Warmup failed"

echo ""
echo "=== Embedding Test 1: Single input ==="
curl -s --max-time 120 http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":"Hello world"}' | \
  python3 -c "import sys,json; r=json.load(sys.stdin); print(f'PASS: dim={len(r[\"data\"][0][\"embedding\"])}, tokens={r[\"usage\"][\"total_tokens\"]}')" 2>/dev/null || echo "FAIL"

echo ""
echo "=== Embedding Test 2: Multiple inputs ==="
curl -s --max-time 120 http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":["Hello","World","Test"]}' | \
  python3 -c "import sys,json; r=json.load(sys.stdin); print(f'PASS: {len(r[\"data\"])} embeddings, dims={[len(d[\"embedding\"]) for d in r[\"data\"]]}')" 2>/dev/null || echo "FAIL"

echo ""
echo "DONE"
