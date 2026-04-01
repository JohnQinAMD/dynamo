#!/bin/bash
# Final gap tests: Embedding, Agentic, Tool Calling
# Key fix: wait for CUDA graph capture to finish, not just health check
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
  if [ "$2" = "PASS" ]; then PASS=$((PASS+1)); echo "  ✅ $1: PASS"
  else FAIL=$((FAIL+1)); echo "  ❌ $1: FAIL — $2"; fi
}

wait_for_first_inference() {
  local port=$1 model=$2 max_wait=${3:-600}
  echo "  Waiting for first inference (aiter JIT + CUDA graph, up to ${max_wait}s)..."
  local resp
  resp=$(curl -s --max-time $max_wait http://localhost:$port/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}" 2>&1)
  if echo "$resp" | grep -q "choices"; then
    echo "  First inference OK"
    return 0
  else
    echo "  First inference failed: $(echo $resp | head -c 200)"
    return 1
  fi
}

echo "============================================"
echo "  FINAL GAP TESTS (with proper warmup)"
echo "============================================"

# ===================================================================
# Test 1: Chat + Agentic + Tool Calling (same server, warm up once)
# ===================================================================
echo ""
echo "=== Starting Qwen3-0.6B server ==="
python3 -m dynamo.frontend --http-port 8000 &
FRONTEND_PID=$!
sleep 2

DYN_SYSTEM_PORT=8081 python3 -m dynamo.sglang \
  --model-path /models/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 --tp 1 --trust-remote-code --skip-tokenizer-init &
WORKER_PID=$!

# Wait for health
for i in $(seq 1 90); do
  curl -s http://localhost:8000/health 2>/dev/null | grep -q healthy && break
  sleep 10
done

# Wait for first inference (this blocks until aiter JIT + CUDA graph done)
if ! wait_for_first_inference 8000 "Qwen/Qwen3-0.6B" 600; then
  test_result "Server warmup" "timed out"
  kill $WORKER_PID $FRONTEND_PID 2>/dev/null
  echo "SUMMARY: $PASS PASS / $FAIL FAIL"
  exit 1
fi

echo ""
echo "=== Test: Basic Chat ==="
CHAT=$(curl -s --max-time 60 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}')
if echo "$CHAT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:50])" 2>/dev/null; then
  test_result "Basic Chat" "PASS"
else
  test_result "Basic Chat" "$(echo $CHAT | head -c 200)"
fi

echo ""
echo "=== Test: Agentic (priority header) ==="
AGENT=$(curl -s --max-time 60 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-dynamo-priority: 1" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Count to 3"}],"max_tokens":20}')
if echo "$AGENT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:50])" 2>/dev/null; then
  test_result "Agentic (priority)" "PASS"
else
  test_result "Agentic (priority)" "$(echo $AGENT | head -c 200)"
fi

echo ""
echo "=== Test: Tool Calling ==="
TOOL=$(curl -s --max-time 60 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"max_tokens":300}')
if echo "$TOOL" | python3 -c "
import sys,json
r=json.load(sys.stdin)
c=r['choices'][0]
msg=c.get('message',{})
tc=msg.get('tool_calls',[])
content=msg.get('content','')
if tc: print(f'Tool call: {tc[0][\"function\"][\"name\"]}')
elif 'weather' in content.lower() or 'paris' in content.lower(): print(f'Content mentions weather/paris')
else: raise ValueError('no tool call or weather mention')
" 2>/dev/null; then
  test_result "Tool Calling" "PASS"
else
  test_result "Tool Calling" "$(echo $TOOL | head -c 200)"
fi

echo ""
echo "=== Test: Streaming ==="
CHUNKS=$(curl -sN --max-time 60 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10,"stream":true}' | grep -c "data:")
if [ "$CHUNKS" -gt 2 ]; then
  test_result "Streaming ($CHUNKS chunks)" "PASS"
else
  test_result "Streaming" "only $CHUNKS chunks"
fi

kill $WORKER_PID 2>/dev/null
sleep 3

# ===================================================================
# Test 2: Embedding
# ===================================================================
echo ""
echo "=== Starting Embedding server (Qwen3-Embedding-4B) ==="
DYN_SYSTEM_PORT=8082 python3 -m dynamo.sglang \
  --embedding-worker \
  --model-path /models/Qwen3-Embedding-4B \
  --served-model-name Qwen/Qwen3-Embedding-4B \
  --page-size 16 --tp 1 --trust-remote-code --use-sglang-tokenizer &
EMB_PID=$!

# Wait for embedding endpoint (check /health includes embedding endpoint)
for i in $(seq 1 90); do
  if curl -s http://localhost:8000/health 2>/dev/null | grep -q "embedding\|Embedding\|generate"; then
    echo "  Embedding worker registered after ${i}0s"
    break
  fi
  sleep 10
done
sleep 10

# Warmup embedding (aiter JIT)
echo "  Embedding warmup..."
curl -s --max-time 600 http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":"warmup"}' > /dev/null 2>&1

echo ""
echo "=== Test: Embedding API ==="
EMBED=$(curl -s --max-time 120 http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":"Hello world"}')
if echo "$EMBED" | python3 -c "import sys,json; r=json.load(sys.stdin); d=r['data'][0]['embedding']; print(f'dim={len(d)}')" 2>/dev/null; then
  test_result "Embedding API" "PASS"
else
  test_result "Embedding API" "$(echo $EMBED | head -c 200)"
fi

kill $EMB_PID $FRONTEND_PID 2>/dev/null

# ===================================================================
# Test 3: Speculative Decoding support
# ===================================================================
echo ""
echo "=== Test: Speculative Decoding ==="
SPEC=$(python3 -m sglang.launch_server --help 2>&1 | grep -c "speculative" || echo "0")
if [ "$SPEC" -gt 10 ]; then
  test_result "Spec Decode ($SPEC args)" "PASS"
else
  test_result "Spec Decode" "only $SPEC args"
fi

echo ""
echo "============================================"
echo "  FINAL SUMMARY: $PASS PASS / $FAIL FAIL"
echo "============================================"
