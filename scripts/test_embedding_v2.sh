#!/bin/bash
# Test 27: Embedding — wait for endpoint registration, not just health
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

echo "Waiting for embedding endpoint (not just health)..."
for i in $(seq 1 180); do
  # Check if /v1/embeddings returns non-404
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-Embedding-4B","input":"test"}' 2>/dev/null)
  if [ "$STATUS" != "000" ] && [ "$STATUS" != "404" ] && [ "$STATUS" != "503" ]; then
    echo "Embedding endpoint ready at ${i}0s (status=$STATUS)"
    break
  fi
  [ $((i % 6)) -eq 0 ] && echo "  Still waiting... (${i}0s, status=$STATUS)"
  sleep 10
done

echo ""
echo "=== Test 1: Single embedding ==="
RESP=$(curl -s --max-time 300 http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":"Hello world"}')
echo "$RESP" | python3 -c "
import sys,json
r=json.load(sys.stdin)
d=r['data'][0]['embedding']
print(f'PASS: dim={len(d)}, tokens={r[\"usage\"][\"total_tokens\"]}')
" 2>/dev/null || echo "FAIL: $(echo $RESP | head -c 300)"

echo ""
echo "=== Test 2: Multiple embeddings ==="
RESP2=$(curl -s --max-time 120 http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":["Hello","World","Test"]}')
echo "$RESP2" | python3 -c "
import sys,json
r=json.load(sys.stdin)
print(f'PASS: {len(r[\"data\"])} embeddings, dims={[len(d[\"embedding\"]) for d in r[\"data\"]]}')
" 2>/dev/null || echo "FAIL: $(echo $RESP2 | head -c 300)"

echo "DONE"
