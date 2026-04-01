#!/bin/bash
# Run tests inside amdprimus/dynamo-rocm-sglang container
set -e

# Use /workspace/dynamo for test files (latest code), /opt/dynamo for installed packages
DYNAMO_DIR="${DYNAMO_DIR:-/opt/dynamo}"
if [ -d /workspace/dynamo/tests ]; then
  DYNAMO_DIR=/workspace/dynamo
fi
cd "$DYNAMO_DIR"
echo "Using test dir: $DYNAMO_DIR"

pip install -q pytest-benchmark 2>&1 | tail -1

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
echo "  FULL IMAGE TESTS (amdprimus/dynamo-rocm-sglang)"
echo "============================================"

# Planner unit tests (previously blocked by prophet/pmdarima)
run "Planner Config" tests/planner/unit/test_planner_config.py -v --tb=short
run "Load Predictors" tests/planner/unit/test_load_predictors.py -v --tb=short
run "Load Based Scaling" tests/planner/unit/test_load_based_scaling.py -v --tb=short
run "SLA Planner Scaling" tests/planner/unit/test_sla_planner_scaling.py -v --tb=short
run "Replica Calculation" tests/planner/test_replica_calculation.py -v --tb=short
run "Global Planner" tests/global_planner/unit/test_scale_request_handler.py -v --tb=short

# Core tests
run "FPM Relay" tests/planner/test_fpm_relay_sglang.py -v --tb=short
run "NIXL ROCm Staging" tests/disagg/test_nixl_rocm_staging.py -v --tb=short
run "ROCm GPU Detection" tests/basic/test_rocm_gpu_detection.py -v --tb=short
run "Profiler DGDR" tests/profiler/ -v --tb=short --timeout=60

# FT tests (need etcd + nats in container)
run "FT Cancellation SGLang" tests/fault_tolerance/cancellation/test_sglang.py -v --tb=short --timeout=300
run "FT etcd HA SGLang" tests/fault_tolerance/etcd_ha/test_sglang.py -v --tb=short --timeout=300

echo ""
echo "============================================"
echo "  SUMMARY"
echo "============================================"
echo "  PASS:    $PASS"
echo "  FAIL:    $FAIL"
echo "  SKIP:    $SKIP"
echo "  Total:   $((PASS+FAIL+SKIP))"
echo "============================================"
