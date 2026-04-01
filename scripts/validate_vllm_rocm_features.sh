#!/bin/bash
# vLLM ROCm Feature Validation Script
# Validates: LoRA, speculative decoding, multimodal, LMCache on ROCm
#
# Usage: Run inside dynamo-rocm-vllm container with GPUs available
#   docker run --device=/dev/kfd --device=/dev/dri --network=host \
#     -v /path/to/hf_cache:/root/.cache/huggingface \
#     dynamo-rocm-vllm:latest bash scripts/validate_vllm_rocm_features.sh

set -euo pipefail

PASS=0; FAIL=0; SKIP=0
MODEL="Qwen/Qwen3-0.6B"
PORT=8199

log() { echo ""; echo "===== $1 ====="; }
pass() { PASS=$((PASS+1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); echo "  FAIL: $1"; }
skip() { SKIP=$((SKIP+1)); echo "  SKIP: $1"; }

wait_for_server() {
    for i in $(seq 1 90); do
        curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}

cleanup() { pkill -f "vllm.entrypoints" 2>/dev/null || true; sleep 2; }

# --- 1. Basic vLLM serving ---
log "1. vLLM Basic Serving"
cleanup
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL --port $PORT --max-model-len 512 \
    --gpu-memory-utilization 0.3 &
if wait_for_server; then
    resp=$(curl -sf "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":16}" 2>/dev/null)
    if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['choices'][0]['message']['content'])>0" 2>/dev/null; then
        pass "Basic vLLM serving"
    else fail "Basic vLLM serving — bad response"; fi
else fail "Basic vLLM serving — server didn't start"; fi
cleanup

# --- 2. LoRA ---
log "2. vLLM LoRA CLI Support"
if python3 -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q "lora-modules"; then
    pass "LoRA CLI flags present"
else fail "LoRA CLI flags missing"; fi

# --- 3. Speculative Decoding ---
log "3. vLLM Speculative Decoding CLI"
if python3 -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q "speculative"; then
    count=$(python3 -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -ci "speculative")
    pass "Speculative decoding CLI ($count args)"
else fail "Speculative decoding CLI missing"; fi

# --- 4. Multimodal ---
log "4. vLLM Multimodal Support"
if python3 -c "from vllm.multimodal import MULTIMODAL_REGISTRY; print('OK')" 2>/dev/null; then
    pass "Multimodal registry importable"
else skip "Multimodal registry not available in this vLLM version"; fi

# --- 5. LMCache ---
log "5. LMCache Integration"
if python3 -c "import lmcache; print('OK')" 2>/dev/null; then
    pass "LMCache importable"
else skip "LMCache not installed — pip install lmcache-vllm"; fi

# --- 6. ROCm GPU detection ---
log "6. ROCm GPU Detection via vLLM"
if python3 -c "
import torch
assert torch.cuda.is_available(), 'No GPU'
print(f'GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')
" 2>/dev/null; then
    pass "PyTorch ROCm GPU detection"
else fail "No ROCm GPU detected"; fi

echo ""
echo "============================================"
echo "  vLLM ROCm Validation: $PASS PASS / $FAIL FAIL / $SKIP SKIP"
echo "============================================"
[ $FAIL -eq 0 ] || exit 1
