#!/bin/bash
# Run remaining P1/P2 tests from amd-remaining-features-plan.md
# Container: amdprimus/dynamo-rocm-sglang:latest
set -e

DYNAMO_DIR="${DYNAMO_DIR:-/opt/dynamo}"
if [ -d /workspace/dynamo/tests ]; then
  DYNAMO_DIR=/workspace/dynamo
fi
cd "$DYNAMO_DIR"
echo "Test dir: $DYNAMO_DIR"

pip install -q pytest-benchmark 2>&1 | tail -1

# Pre-download small model to avoid timeout in FT tests
echo "Pre-downloading Qwen/Qwen3-0.6B..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B')" 2>&1 | tail -2
export HF_HUB_OFFLINE=1

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

echo ""
echo "============================================"
echo "  REMAINING FEATURE TESTS"
echo "============================================"

# === P1: FT with longer timeout + pre-cached model ===
run "FT Cancellation SGLang (agg)" \
  tests/fault_tolerance/cancellation/test_sglang.py \
  -k "aggregated" -v --tb=short --timeout=600

run "FT Cancellation SGLang (disagg)" \
  tests/fault_tolerance/cancellation/test_sglang.py \
  -k "decode_cancel" -v --tb=short --timeout=600

run "FT etcd HA SGLang (agg)" \
  tests/fault_tolerance/etcd_ha/test_sglang.py \
  -k "aggregated" -v --tb=short --timeout=600

run "FT etcd HA SGLang (disagg)" \
  tests/fault_tolerance/etcd_ha/test_sglang.py \
  -k "disaggregated" -v --tb=short --timeout=600

run "FT Migration SGLang" \
  tests/fault_tolerance/migration/test_sglang.py \
  -v --tb=short --timeout=600

# === P2: Multimodal ===
run "Serve SGLang (aggregated)" \
  tests/serve/test_sglang.py \
  -k "aggregated" -v --tb=short --timeout=300

# === P2: Frontend tests ===
run "Frontend Mocker Engine" \
  tests/frontend/test_completion_mocker_engine.py \
  -v --tb=short --timeout=120

run "Frontend gRPC Mocker" \
  tests/frontend/grpc/test_tensor_mocker_engine.py \
  -v --tb=short --timeout=120

# === P2: Router tests ===
run "Router E2E Mockers (all modes)" \
  tests/router/test_router_e2e_with_mockers.py \
  -v --tb=short --timeout=300

run "Router Block Size Regression" \
  tests/router/test_router_block_size_regression.py \
  -v --tb=short --timeout=120

# === P2: Other unit tests ===
run "Managed Process Teardown" \
  tests/utils/test_managed_process_teardown.py \
  -v --tb=short

run "Prometheus Exposition" \
  tests/serve/test_prometheus_exposition_format_injection.py \
  -v --tb=short

run "Mocker Config" \
  tests/mocker/test_config.py \
  -v --tb=short

run "Remote Planner" \
  tests/planner/unit/test_remote_planner.py \
  -v --tb=short

run "Prometheus Helper" \
  tests/planner/unit/test_prometheus.py \
  -v --tb=short

run "Virtual Connector" \
  tests/planner/unit/test_virtual_connector.py \
  -v --tb=short --timeout=120

run "Load Generator" \
  tests/planner/test_load_generator.py \
  -v --tb=short

run "Planner Virtual SGLang" \
  tests/planner/test_planner_virtual_sglang.py \
  -v --tb=short --timeout=120

run "Ionic Validation" \
  tests/disagg/test_ionic_validation.py \
  -v --tb=short

run "ROCm Version Consistency" \
  tests/basic/test_rocm_version_consistency.py \
  -v --tb=short

echo ""
echo "============================================"
echo "  SUMMARY"
echo "============================================"
echo "  PASS:    $PASS"
echo "  FAIL:    $FAIL"
echo "  SKIP:    $SKIP"
echo "  Total:   $((PASS+FAIL+SKIP))"
echo "============================================"
