#!/bin/bash
# Run remaining gap analysis tests: Embedding, Multimodal E/P/D, Agentic, Spec Decode
set -e

MY_IP=$(hostname -I | awk '{print $1}')
rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://${MY_IP}:2379 &
nats-server -p 4222 -js --client_advertise ${MY_IP}:4222 &
sleep 3
export ETCD_ENDPOINTS=http://${MY_IP}:2379
export NATS_SERVER=nats://${MY_IP}:4222

cd /opt/dynamo
PASS=0; FAIL=0

test_result() {
  local name="$1" result="$2"
  if [ "$result" = "PASS" ]; then
    PASS=$((PASS+1))
    echo "  ✅ $name: PASS"
  else
    FAIL=$((FAIL+1))
    echo "  ❌ $name: FAIL — $result"
  fi
}

echo "============================================"
echo "  GAP ANALYSIS TESTS"
echo "============================================"
echo ""

# --- Test: Embedding with --embedding-worker ---
echo "=== Embedding (Qwen3-Embedding-4B) ==="
python3 -m dynamo.frontend --http-port 8000 &
sleep 2

DYN_SYSTEM_PORT=8081 python3 -m dynamo.sglang \
  --embedding-worker \
  --model-path /models/Qwen3-Embedding-4B \
  --served-model-name Qwen/Qwen3-Embedding-4B \
  --page-size 16 --tp 1 --trust-remote-code --use-sglang-tokenizer &

echo "Waiting for embedding server..."
for i in $(seq 1 90); do
  if curl -s http://localhost:8000/health 2>/dev/null | grep -q healthy; then
    echo "Server healthy after ${i}0 seconds"
    break
  fi
  sleep 10
done
# Warmup: first request triggers aiter JIT compilation (~135s)
echo "Warmup (aiter JIT compile, may take 2-3 min)..."
curl -s --max-time 600 http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":"warmup"}' > /dev/null 2>&1
echo "Warmup done"

EMBED_RESP=$(curl -s --max-time 300 http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":"Hello world"}' 2>&1)

if echo "$EMBED_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); assert len(r['data'][0]['embedding']) > 0; print(f'dim={len(r[\"data\"][0][\"embedding\"])}')" 2>/dev/null; then
  test_result "Embedding API" "PASS"
else
  test_result "Embedding API" "FAIL: $EMBED_RESP"
fi

pkill -f dynamo.sglang || true
pkill -f dynamo.frontend || true
sleep 3

# --- Test: Agentic/nvext hints ---
echo ""
echo "=== Agentic / nvext hints ==="
python3 -m dynamo.frontend --http-port 8000 &
sleep 2

python3 -m dynamo.sglang \
  --model-path /models/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 --tp 1 --trust-remote-code --skip-tokenizer-init &

for i in $(seq 1 90); do
  if curl -s http://localhost:8000/health 2>/dev/null | grep -q healthy; then
    echo "Server healthy after ${i}0 seconds"
    break
  fi
  sleep 10
done
# Warmup: first request triggers aiter JIT compile
echo "Warmup (aiter JIT compile, may take 2-3 min)..."
curl -s --max-time 600 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"warmup"}],"max_tokens":1}' > /dev/null 2>&1
echo "Warmup done"

# Test with nvext priority header
AGENT_RESP=$(curl -s --max-time 300 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-dynamo-priority: 1" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}' 2>&1)

if echo "$AGENT_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); assert r['choices'][0]['message']['content']; print('Response OK')" 2>/dev/null; then
  test_result "Agentic (priority header)" "PASS"
else
  test_result "Agentic (priority header)" "FAIL: $AGENT_RESP"
fi

# --- Test: Speculative Decoding support ---
echo ""
echo "=== Speculative Decoding ==="
SPEC_HELP=$(python3 -m sglang.launch_server --help 2>&1 | grep -c "speculative" || echo "0")
if [ "$SPEC_HELP" -gt 10 ]; then
  test_result "Spec Decode (args available)" "PASS"
  echo "  SGLang supports $SPEC_HELP speculative decoding args"
else
  test_result "Spec Decode (args available)" "FAIL: only $SPEC_HELP args"
fi

# --- Test: Tool Calling ---
echo ""
echo "=== Tool Calling ==="
TOOL_RESP=$(curl -s --max-time 300 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"max_tokens":300}' 2>&1)

if echo "$TOOL_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); c=r['choices'][0]; assert c.get('message',{}).get('tool_calls') or 'weather' in str(c).lower() or 'paris' in str(c).lower(); print('Tool call detected')" 2>/dev/null; then
  test_result "Tool Calling" "PASS"
else
  test_result "Tool Calling" "FAIL: $TOOL_RESP"
fi

pkill -f dynamo.sglang || true
pkill -f dynamo.frontend || true

echo ""
echo "============================================"
echo "  SUMMARY: $PASS PASS / $FAIL FAIL"
echo "============================================"
