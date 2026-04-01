#!/bin/bash
# Tests 20-22: K8s Operator, DGDR, Deploy
set -e
export KUBECONFIG=/mnt/vast/john/rocm-dynamo/dynamo/.k8s-kubeconfig.yaml

echo "============================================"
echo "  K8s Dynamo Tests"
echo "============================================"
PASS=0; FAIL=0; SKIP=0

result() {
  if [ "$2" = "PASS" ]; then PASS=$((PASS+1)); echo "  ✅ $1: PASS"
  elif [ "$2" = "SKIP" ]; then SKIP=$((SKIP+1)); echo "  ⏭️  $1: SKIP — $3"
  else FAIL=$((FAIL+1)); echo "  ❌ $1: FAIL — $2"; fi
}

echo ""
echo "=== Cluster status ==="
kubectl get nodes -o wide | head -15
echo ""

echo "=== GPU resources on nodes ==="
for node in $(kubectl get nodes -o jsonpath='{.items[*].metadata.name}'); do
  gpu=$(kubectl get node $node -o jsonpath='{.status.allocatable.amd\.com/gpu}' 2>/dev/null)
  [ -n "$gpu" ] && [ "$gpu" != "0" ] && echo "  $node: $gpu GPUs"
done
echo ""

echo "=== CRDs ==="
CRD_COUNT=$(kubectl get crd 2>/dev/null | grep -c dynamo)
if [ "$CRD_COUNT" -eq 7 ]; then
  result "Dynamo CRDs (7/7)" "PASS"
else
  result "Dynamo CRDs" "$CRD_COUNT/7 found"
fi

echo ""
echo "=== Dynamo namespace ==="
kubectl get pods,svc -n dynamo 2>/dev/null | head -15
ETCD_OK=$(kubectl exec -n dynamo etcd-0 -- etcdctl endpoint health 2>/dev/null | grep -c "is healthy")
if [ "$ETCD_OK" -gt 0 ]; then
  result "etcd in K8s" "PASS"
else
  result "etcd in K8s" "SKIP" "etcd pod not running"
fi

echo ""
echo "=== DGD dry-run (AMD templates) ==="
for f in /mnt/vast/john/rocm-dynamo/dynamo/tests/fault_tolerance/deploy/templates/sglang/*.yaml; do
  name=$(basename $f)
  if kubectl apply --dry-run=server -n dynamo -f $f 2>&1 | grep -q "created\|configured"; then
    result "DGD dry-run: $name" "PASS"
  else
    result "DGD dry-run: $name" "$(kubectl apply --dry-run=server -n dynamo -f $f 2>&1 | tail -1)"
  fi
done

echo ""
echo "=== Dynamo Operator ==="
OPERATOR=$(kubectl get deployments -n dynamo 2>/dev/null | grep -c operator)
if [ "$OPERATOR" -gt 0 ]; then
  result "Dynamo Operator running" "PASS"
else
  result "Dynamo Operator" "SKIP" "not deployed (need helm install)"
fi

echo ""
echo "=== AMD GPU Operator ==="
GPU_OP=$(kubectl get pods -n kube-amd-gpu 2>/dev/null | grep -c Running)
if [ "$GPU_OP" -gt 0 ]; then
  result "AMD GPU Operator ($GPU_OP pods)" "PASS"
else
  result "AMD GPU Operator" "SKIP" "not detected"
fi

echo ""
echo "============================================"
echo "  SUMMARY: $PASS PASS / $FAIL FAIL / $SKIP SKIP"
echo "============================================"
