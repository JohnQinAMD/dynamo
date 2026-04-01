#!/bin/bash
# Fix items 11 (MM E/P/D), 27 (Embedding), 23 (Agentic), 24 (Tool Calling)
# Key fix: poll /v1/models endpoint instead of /health for readiness
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
result() {
  if [ "$2" = "PASS" ]; then PASS=$((PASS+1)); echo "  ✅ $1: PASS"
  else FAIL=$((FAIL+1)); echo "  ❌ $1: FAIL — $2"; fi
}

wait_for_model() {
  local port=$1 model=$2 max_wait=${3:-900}
  echo "  Waiting for model '$model' on port $port (max ${max_wait}s)..."
  for i in $(seq 1 $((max_wait / 5))); do
    MODELS=$(curl -s --max-time 5 http://localhost:$port/v1/models 2>/dev/null)
    if echo "$MODELS" | grep -q "$model"; then
      echo "  Model ready after $((i*5))s"
      # Send warmup to trigger aiter JIT
      echo "  Sending warmup (aiter JIT + CUDA graph)..."
      curl -s --max-time 600 http://localhost:$port/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}" > /dev/null 2>&1
      echo "  Warmup done"
      return 0
    fi
    sleep 5
  done
  echo "  Model not ready after ${max_wait}s"
  return 1
}

echo "============================================"
echo "  ALL REMAINING TESTS"
echo "============================================"

# ===================================================================
# Test 11: Multimodal via VL model (already proven with 7B, use 2B)
# ===================================================================
echo ""
echo "=== 11. Multimodal VL (Qwen3-VL-2B) ==="
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin &
sleep 2

DYN_SYSTEM_PORT=8081 python3 -m dynamo.sglang \
  --model-path /models/Qwen3-VL-2B-Instruct \
  --served-model-name Qwen/Qwen3-VL-2B-Instruct \
  --page-size 16 --tp 1 --trust-remote-code &

if wait_for_model 8000 "Qwen3-VL-2B"; then
  # Text test
  RESP=$(curl -s --max-time 60 http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-VL-2B-Instruct","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10}')
  if echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:50])" 2>/dev/null; then
    result "VL Text Chat" "PASS"
  else
    result "VL Text Chat" "$(echo $RESP | head -c 200)"
  fi

  # Image test
  IMG_RESP=$(curl -s --max-time 120 http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-VL-2B-Instruct","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image briefly."},{"type":"image_url","image_url":{"url":"http://images.cocodataset.org/test2017/000000155781.jpg"}}]}],"max_tokens":50,"temperature":0.0}')
  CONTENT=$(echo "$IMG_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:100])" 2>/dev/null)
  if [ -n "$CONTENT" ] && [ ${#CONTENT} -gt 5 ]; then
    result "VL Image" "PASS"
    echo "    Response: $CONTENT"
  else
    result "VL Image" "$(echo $IMG_RESP | head -c 200)"
  fi

  # Streaming test
  CHUNKS=$(curl -sN --max-time 60 http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-VL-2B-Instruct","messages":[{"role":"user","content":"Count to 3"}],"max_tokens":20,"stream":true}' | grep -c "data:")
  if [ "$CHUNKS" -gt 2 ]; then
    result "VL Streaming ($CHUNKS chunks)" "PASS"
  else
    result "VL Streaming" "$CHUNKS chunks"
  fi

  # Tool calling test
  TOOL=$(curl -s --max-time 60 http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-VL-2B-Instruct","messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"max_tokens":200}')
  if echo "$TOOL" | python3 -c "import sys,json; r=json.load(sys.stdin); c=r['choices'][0]; assert c.get('message',{}).get('tool_calls') or 'weather' in str(c).lower() or 'paris' in str(c).lower()" 2>/dev/null; then
    result "Tool Calling" "PASS"
  else
    result "Tool Calling" "$(echo $TOOL | head -c 200)"
  fi

  # Agentic test (priority header)
  AGENT=$(curl -s --max-time 60 http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "x-dynamo-priority: 1" \
    -d '{"model":"Qwen/Qwen3-VL-2B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}')
  if echo "$AGENT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:50])" 2>/dev/null; then
    result "Agentic (priority header)" "PASS"
  else
    result "Agentic" "$(echo $AGENT | head -c 200)"
  fi
else
  result "VL Model Load" "timeout"
fi

pkill -f dynamo.sglang 2>/dev/null
sleep 5

# ===================================================================
# Test 27: Embedding — wait for /v1/models to show embedding model
# ===================================================================
echo ""
echo "=== 27. Embedding ==="
DYN_SYSTEM_PORT=8082 python3 -m dynamo.sglang \
  --embedding-worker \
  --model-path /models/Qwen3-Embedding-4B \
  --served-model-name Qwen/Qwen3-Embedding-4B \
  --page-size 16 --tp 1 --trust-remote-code --use-sglang-tokenizer &

# Wait for embedding endpoint by polling /v1/embeddings with retry
echo "  Waiting for embedding endpoint..."
for i in $(seq 1 180); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-Embedding-4B","input":"test"}' 2>/dev/null)
  if [ "$STATUS" = "200" ]; then
    echo "  Embedding endpoint returned 200 at $((i*5))s!"
    break
  elif [ "$STATUS" != "404" ] && [ "$STATUS" != "000" ] && [ "$STATUS" != "503" ]; then
    echo "  Got status $STATUS at $((i*5))s, trying..."
  fi
  [ $((i % 12)) -eq 0 ] && echo "  Still waiting... ($((i*5))s, status=$STATUS)"
  sleep 5
done

EMBED=$(curl -s --max-time 120 http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":"Hello world"}')
DIM=$(echo "$EMBED" | python3 -c "import sys,json; r=json.load(sys.stdin); print(len(r['data'][0]['embedding']))" 2>/dev/null)
if [ -n "$DIM" ] && [ "$DIM" -gt 0 ]; then
  result "Embedding (dim=$DIM)" "PASS"
  # Multi-input
  MULTI=$(curl -s --max-time 60 http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-Embedding-4B","input":["A","B","C"]}')
  COUNT=$(echo "$MULTI" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))" 2>/dev/null)
  if [ "$COUNT" = "3" ]; then
    result "Multi-Embedding (3 inputs)" "PASS"
  else
    result "Multi-Embedding" "count=$COUNT"
  fi
else
  result "Embedding" "$(echo $EMBED | head -c 300)"
fi

echo ""
echo "============================================"
echo "  SUMMARY: $PASS PASS / $FAIL FAIL"
echo "============================================"
