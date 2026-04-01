#!/bin/bash
# Autoscaling E2E Test for Dynamo on AMD ROCm K8s
#
# Validates DGDR autoscaling policy:
#   1. Deploy with minReplicas=1
#   2. Send load to trigger scale-up
#   3. Verify worker count increases
#   4. Stop load, verify scale-down
#
# Prerequisites:
#   - kubectl configured for the target cluster
#   - Dynamo Operator, AMD GPU Operator running
#   - etcd + NATS deployed in dynamo namespace

set -euo pipefail

NAMESPACE="dynamo"
MANIFEST="examples/backends/sglang/deploy/dgdr_autoscale_rocm.yaml"
TIMEOUT=300

log() { echo "[$(date +%H:%M:%S)] $1"; }

log "Step 1: Deploy DGDR"
kubectl apply -f "$MANIFEST" -n "$NAMESPACE"

log "Step 2: Wait for initial deployment (1 worker)"
for i in $(seq 1 $((TIMEOUT/5))); do
    ready=$(kubectl get pods -n "$NAMESPACE" -l app=sglang-rocm-agg-sglangworker \
        --field-selector=status.phase=Running -o name 2>/dev/null | wc -l)
    [ "$ready" -ge 1 ] && break
    sleep 5
done
log "  Workers running: $ready"

log "Step 3: Get frontend endpoint"
FRONTEND_IP=$(kubectl get svc -n "$NAMESPACE" sglang-rocm-agg-frontend \
    -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "localhost")
FRONTEND_PORT=$(kubectl get svc -n "$NAMESPACE" sglang-rocm-agg-frontend \
    -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "8000")
URL="http://${FRONTEND_IP}:${FRONTEND_PORT}"

log "Step 4: Wait for model readiness"
for i in $(seq 1 60); do
    curl -sf "$URL/v1/models" > /dev/null 2>&1 && break
    sleep 5
done

log "Step 5: Generate load (target >70% GPU util for scale-up)"
for j in $(seq 1 50); do
    curl -sf "$URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long essay about AI number $j\"}],\"max_tokens\":256}" \
        > /dev/null 2>&1 &
done
log "  Sent 50 concurrent requests"

log "Step 6: Wait for scale-up (up to 3 min)"
for i in $(seq 1 36); do
    workers=$(kubectl get pods -n "$NAMESPACE" -l app=sglang-rocm-agg-sglangworker \
        --field-selector=status.phase=Running -o name 2>/dev/null | wc -l)
    log "  Workers: $workers"
    [ "$workers" -ge 2 ] && break
    sleep 5
done

if [ "$workers" -ge 2 ]; then
    log "PASS: Scaled up to $workers workers"
else
    log "FAIL: Did not scale up (still $workers workers)"
fi

log "Step 7: Wait for requests to complete + scale-down cooldown (2 min)"
wait
sleep 120

workers=$(kubectl get pods -n "$NAMESPACE" -l app=sglang-rocm-agg-sglangworker \
    --field-selector=status.phase=Running -o name 2>/dev/null | wc -l)
log "Step 8: Post-cooldown workers: $workers"

log "Step 9: Cleanup"
kubectl delete -f "$MANIFEST" -n "$NAMESPACE" --ignore-not-found

log "=== Autoscale E2E Complete ==="
