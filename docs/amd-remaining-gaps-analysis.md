# AMD Dynamo — Remaining Gaps Analysis
## 1. K8s Deployment + Planner
### Current State
A **full K8s cluster is available** and ready for Dynamo deployment:
| Component | Status |
|-----------|--------|
| K3s cluster | v1.34.5, 8 nodes (7 Ready), control-plane on chi2894 |
| AMD GPU Operator | **Installed** (`kube-amd-gpu` namespace, running) |
| `amd.com/gpu` | 8 per node (64 total GPUs available) |
| cert-manager | Running |
| Existing workloads | Idle (completed JAX jobs only) |
| Dynamo CRDs | **NOT installed** (no `nvidia.com` or `dynamo.ai` CRDs) |
### Deployment Plan (non-disruptive)
1. **Install Dynamo CRDs** in a new `dynamo` namespace:
   ```bash
   kubectl create namespace dynamo
   kubectl apply -f deploy/operator/config/crd/bases/
   ```
2. **Deploy etcd + NATS** (Dynamo infrastructure):
   ```bash
   helm install dynamo-infra deploy/helm/charts/platform/ -n dynamo
   ```
3. **Deploy Dynamo frontend + SGLang worker** (single-node first):
   - Use `amd.com/gpu: 8` resource requests
   - Mount DeepSeek-V3 from shared NFS
4. **Deploy Planner** in virtual mode:
   ```yaml
   environment: virtual
   backend: sglang
   enable_load_scaling: false   # Skip FPM initially
   enable_throughput_scaling: true
   ```
5. **Test auto-scaling** by varying load
### Risk: The CRD API group needs adjustment
Dynamo CRDs use `nvidia.com` API group. Options:
- Use our `dynamo.ai` renamed CRDs from the `amd-additive` branch
- Or use the original `nvidia.com` CRDs (works regardless of GPU vendor)
---
## 2. SGLang FPM (Forward Pass Metrics) Relay
### Current State
FPM is **only implemented for vLLM**, not SGLang:
| Component | vLLM | SGLang |
|-----------|------|--------|
| `InstrumentedScheduler` | ✅ `dynamo/vllm/instrumented_scheduler.py` | ❌ Not implemented |
| `FpmEventRelay` | ✅ In vLLM worker factory | ❌ Not in SGLang |
| `FpmEventSubscriber` | ✅ (planner consumer) | ✅ (generic, works for any publisher) |
| Planner `enable_load_scaling` | ✅ Works | ❌ No data source |
The `forward_pass_metrics.py` explicitly has: `TODO: add metrics for TrtLLM/SGLang`.
### FPM Data Flow
```
EngineCore (child process):
  InstrumentedScheduler → _FpmPublisherThread → ZMQ PUB (localhost)
Dynamo parent process:
  FpmEventRelay (ZMQ SUB) → EventPublisher → Event Plane (NATS/ZMQ)
Consumer (planner):
  FpmEventSubscriber (auto-discovered) → decode() → ForwardPassMetrics
```
### How to Enable for SGLang
**Option A: Instrument SGLang's Scheduler** (Medium effort)
1. Wrap SGLang's scheduler step with FPM collection (similar to `InstrumentedScheduler`)
2. Collect per-iteration: `num_running_requests`, `num_waiting_requests`, `num_prefill_tokens`, `num_decode_tokens`, `cache_utilization`
3. Publish via ZMQ PUB to FpmEventRelay bridge
**Option B: Use Prometheus Metrics** (Low effort, less real-time)
1. SGLang already exports Prometheus metrics via `--enable-metrics`
2. The planner's `enable_throughput_scaling` already consumes Prometheus
3. Skip FPM entirely; use throughput-based scaling which is Prometheus-only
4. This works TODAY without any code changes
**Recommendation**: Start with **Option B** (Prometheus-only throughput scaling) for immediate testing, then implement Option A for production load-based scaling.
---
## 3. DSV3 Disagg Reliability via MoRI + QoS/DCQCN
### What Happened
MoRI RDMA backend created OK and **DSV3 generated a response** ("Hello! How can") — proving the full pipeline works. But reliability was only 3/6 at c=1 and c=4 failed entirely.
### Root Cause Analysis
**QoS misconfiguration is the likely cause**. Comparing with ROCm docs:
| Setting | ROCm Docs Recommendation | Slurm Nodes (chi2899) | K8s Nodes (chi2883) |
|---------|--------------------------|----------------------|---------------------|
| Data DSCP → Priority | DSCP 24 → Q3 | DSCP 26 → Q3 | DSCP 24,26,46 → Q0 |
| CNP DSCP → Priority | DSCP 46 → Q6 | DSCP 48 → Q7 | DSCP 48 → Q6 |
| PFC no-drop priority | Q3 | Not shown | Q0 |
| Scheduling | Q3:99%, Q0:1%, Q6:strict | Unknown | Unknown |
**Issues found**:
1. **DSCP mapping differs** from ROCm recommendation — data traffic may not get priority treatment
2. **PFC priority mismatch** — K8s nodes have PFC on Q0 (should be Q3 per docs)
3. **No backend IPv4 addresses** — ionic interfaces lack 192.168.x.x/24 subnet IPs
4. **GID subnet mismatch** — chi2899 ionic_0 is on `:141` subnet, chi2900 ionic_0 is on `:148` subnet (different physical links)
### Performance Assessment
The current 512ms P50 with 50% success rate is **NOT expected performance**. With proper QoS:
- ROCm docs show **17,560 tok/s** at 2048 concurrency for DSV3 disagg (1P2D)
- Our MoRI RDMA path is fundamentally correct (response was generated)
- The reliability issue is network QoS, not software
### Fix Plan
1. **Configure backend network** (per ROCm docs):
   ```bash
   # On each node, assign IPs to ionic interfaces
   ip addr add 192.168.{1..8}.{NODE_ID}/24 dev benic{1..8}p1
   ```
2. **Fix QoS/DCQCN** (match ROCm recommendation):
   ```bash
   # Set DSCP → Priority mapping
   sudo nicctl update qos dscp-to-priority --dscp 24 --priority 3
   sudo nicctl update qos dscp-to-priority --dscp 46 --priority 6
   # Enable PFC for Q3
   sudo nicctl update qos pfc --priority 3 --no-drop enable
   # Set scheduling
   sudo nicctl update qos scheduling --priority 3,0,6 --dwrr 99,1,0 --rate-limit 0,0,10
   ```
3. **Fix DCQCN** CNP DSCP:
   ```bash
   sudo nicctl update dcqcn -r ionic_0 -i 1 --cnp-dscp 46  # Currently 48
   ```
4. **Verify with RDMA bandwidth test**:
   ```bash
   ib_write_bw --use_rocm=0 -d ionic_0 --report_gbits -a
   ```
### Expected Outcome After Fix
- RDMA bandwidth: ~390 Gbps per NIC (verified at DRAM level: 39.4 GB/s = 315 Gbps)
- DSV3 disagg: reliable at c=1-32
- Performance comparable to ROCm docs benchmarks
---
## Summary: Priority Actions
| # | Action | Effort | Blocks |
|---|--------|--------|--------|
| 1 | Fix ionic QoS/DCQCN on Slurm nodes | Low | DSV3 disagg reliability |
| 2 | Assign backend IPs to matching ionic interfaces | Low | RDMA connectivity |
| 3 | Deploy Dynamo on K8s (new namespace) | Medium | Planner production test |
| 4 | Use Prometheus-only planner scaling | Low | Planner validation |
| 5 | Implement SGLang FPM relay | Medium | Load-based planner |
