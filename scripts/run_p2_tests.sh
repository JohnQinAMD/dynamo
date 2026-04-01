#!/bin/bash
# Run P2 remaining tests: Multimodal, vLLM, Embedding, Spec Decode, GMS, Agentic
set -e

DYNAMO_DIR="${DYNAMO_DIR:-/opt/dynamo}"
[ -d /workspace/dynamo/tests ] && DYNAMO_DIR=/workspace/dynamo
cd "$DYNAMO_DIR"
echo "Test dir: $DYNAMO_DIR"
echo "Python: $(python3 --version)"
python3 -c "import dynamo; print('dynamo OK')" 2>/dev/null || echo "dynamo: NOT INSTALLED"
python3 -c "import sglang; print(f'sglang {sglang.__version__}')" 2>/dev/null || echo "sglang: NOT INSTALLED"
python3 -c "import vllm; print(f'vllm {vllm.__version__}')" 2>/dev/null || echo "vllm: NOT INSTALLED"
echo ""

pip install -q pytest-benchmark pytest-httpserver 2>&1 | tail -1

PY='python3 -m pytest --override-ini=filterwarnings=default'
PASS=0; FAIL=0; SKIP=0

run() {
  local name="$1"; shift
  echo ""
  echo "===== $name ====="
  if $PY "$@" 2>&1; then
    PASS=$((PASS+1)); echo "  >> PASS"
  else
    rc=$?
    if [ $rc -eq 5 ]; then SKIP=$((SKIP+1)); echo "  >> SKIPPED"
    else FAIL=$((FAIL+1)); echo "  >> FAIL (rc=$rc)"; fi
  fi
}

echo "============================================"
echo "  P2 REMAINING TESTS"
echo "============================================"

# === Multimodal: SGLang serve with VL model ===
run "Serve SGLang (aggregated - Qwen-0.6B)" \
  tests/serve/test_sglang.py \
  -k "aggregated" -v --tb=short --timeout=300

# === Embedding ===
run "Serve SGLang (embedding_agg)" \
  tests/serve/test_sglang.py \
  -k "embedding_agg" -v --tb=short --timeout=300

# === Tool Calling (via mocker) ===
run "Frontend Prepost" \
  tests/frontend/test_prepost.py \
  -v --tb=short --timeout=120

# === GPU Memory Service ===
run "GMS Failover Lock" \
  tests/fault_tolerance/gpu_memory_service/test_failover_lock.py \
  -v --tb=short --timeout=120

# === vLLM serve (if vllm is installed) ===
run "Serve vLLM (aggregated)" \
  tests/serve/test_vllm.py \
  -k "aggregated" -v --tb=short --timeout=300

# === Router E2E with SGLang ===
run "Router E2E SGLang" \
  tests/router/test_router_e2e_with_sglang.py \
  -v --tb=short --timeout=300

# === Planner scaling (if deps available) ===
run "Planner Scaling E2E" \
  tests/planner/test_scaling_e2e.py \
  -v --tb=short --timeout=300

# === Disagg logprobs ===
run "Disagg Logprobs Serialization" \
  tests/serve/test_disagg_logprobs_serialization.py \
  -v --tb=short

# === KVBM imports ===
run "KVBM Imports" \
  tests/dependencies/test_kvbm_imports.py \
  -v --tb=short

# === ROCm-specific serve test ===
run "Serve SGLang ROCm" \
  tests/serve/test_sglang_rocm.py \
  -v --tb=short --timeout=300

# === Predownload models ===
run "Predownload Models" \
  tests/test_predownload_models.py \
  -v --tb=short --timeout=120

echo ""
echo "============================================"
echo "  SUMMARY"
echo "============================================"
echo "  PASS:    $PASS"
echo "  FAIL:    $FAIL"
echo "  SKIP:    $SKIP"
echo "  Total:   $((PASS+FAIL+SKIP))"
echo "============================================"
