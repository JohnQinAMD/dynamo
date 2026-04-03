# InferenceX-Aligned Benchmarks for MI355X

Benchmark scripts to reproduce and compare with [InferenceX](https://inferencex.com)
official results on MI355X with DeepSeek-R1-0528 FP8.

## Quick Start

```bash
# 1. Preflight check (validates nodes before anything runs)
bash preflight_check.sh chi2863 chi2870 chi2900

# 2. Run 1P2D benchmark (MoRI SGLang + Dynamo, parallel)
bash run_benchmark.sh

# 3. Run 1P1D MoRI-only benchmark
NODE_P=chi2863 NODE_D=chi2870 bash run_mori_1p1d.sh
```

## File Layout

| File | Description |
|------|-------------|
| `run_benchmark.sh` | Main benchmark: 1P2D MoRI SGLang vs Dynamo (6 nodes) |
| `run_mori_1p1d.sh` | MoRI SGLang 1P1D baseline (2 nodes) |
| `preflight_check.sh` | Pre-run validation (SSH, GPU, ionic, NFS, subnet match) |
| `setup_ionic_network.sh` | Assign IPv4 to ionic RDMA ports |
| `inferencex_reference_data.csv` | Official InferenceX results for comparison |
| `comparison_chart.html` | Visual comparison chart (open in browser) |
| `results/` | Benchmark output JSON files |

## Configuration Alignment with InferenceX

All parameters are aligned with InferenceX's official GitHub configuration
([models.yaml](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/multi_node/amd_utils/models.yaml),
[server.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/multi_node/amd_utils/server.sh),
[env.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/multi_node/amd_utils/env.sh),
[bench.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/multi_node/amd_utils/bench.sh)).

### Server Parameters (DeepSeek-R1-0528 no_dp)

| Parameter | Prefill | Decode | Source |
|-----------|---------|--------|--------|
| `--mem-fraction-static` | 0.80 | 0.85 | models.yaml |
| `--max-running-requests` | 128 | 128 | models.yaml |
| `--cuda-graph-bs` | 1-128 | 1-128 | models.yaml |
| `--chunked-prefill-size` | 262144 | (not set) | server.sh |
| `--disable-radix-cache` | yes | no | models.yaml |
| `--prefill-round-robin-balance` | no | yes | models.yaml |
| `--log-level-http` | warning | warning | server.sh |

### Environment Variables (from env.sh)

| Variable | Value | Purpose |
|----------|-------|---------|
| `MORI_SHMEM_MODE` | ISOLATION | GPU memory isolation -- required to avoid CUDA graph OOM |
| `SGLANG_MORI_FP8_DISP` | True | FP8 precision for KV dispatch (halves buffer size) |
| `SGLANG_MORI_FP4_DISP` | False | FP4 dispatch disabled |
| `SGLANG_MORI_FP8_COMB` | False | FP8 combine disabled |
| `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 16384 (P) / 160 (D) | MoRI buffer size per rank (decode=160 is critical for memory) |
| `SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD` | 320 | Kernel switch threshold (2x DECODE dispatch tokens) |
| `MORI_EP_LAUNCH_CONFIG_MODE` | AUTO | EP launch config auto-detection |
| `MORI_IO_QP_MAX_SEND_WR` | 16384 | RDMA QP send WR limit |
| `MORI_IO_QP_MAX_CQE` | 32768 | RDMA CQ entries limit |
| `MORI_IO_QP_MAX_SGE` | 4 | RDMA scatter/gather elements |
| `MORI_APP_LOG_LEVEL` | INFO | MoRI logging level |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | 1200 | Bootstrap connection timeout (seconds) |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | 1200 | Waiting timeout (seconds) |
| `NCCL_IB_HCA` | ionic_0,...,ionic_7 | NCCL IB device list |
| `MORI_MAX_DISPATCH_TOKENS_PREFILL` | 16384 | Prefill dispatch token limit |
| `MORI_MAX_DISPATCH_TOKENS_DECODE` | 160 | Decode dispatch token limit |
| `PYTHONDONTWRITEBYTECODE` | 1 | Prevent .pyc writes (minor perf) |

All 17 env vars must be set. Missing any can cause OOM, RDMA failures, or degraded performance.
They are set automatically via Docker `-e` flags in `run_benchmark.sh`.

### Benchmark Client

| Parameter | Value | Source |
|-----------|-------|--------|
| `--backend` | openai | bench.sh |
| `--random-range-ratio` | 0.8 | workflow env |
| `--num-prompts` | conc * 10 | bench.sh |
| `--request-rate` | inf | server.sh |

## Preflight Checks

The `preflight_check.sh` script validates 11 items before any benchmark runs:

| # | Check | Severity | Auto-fix |
|---|-------|----------|----------|
| 1 | SSH connectivity | ERROR | No |
| 2 | Docker daemon + image | ERROR | Auto-pull |
| 3 | GPU count >= 8 | ERROR | No |
| 4 | GPU VRAM < 8GB | ERROR | Kills containers/processes |
| 5 | Ionic IB devices = 8, PORT_ACTIVE | ERROR | No |
| 6 | Ionic IPv4 = 8/8 | ERROR | Runs setup_ionic_network.sh |
| 7 | libionic.so on host | ERROR | No |
| 8 | NFS/vast model mount | ERROR | No |
| 9 | Management IP not 192.168.x.x | ERROR | No |
| 10 | Cross-node ping | WARN | No |
| 11 | Cross-node ionic GID subnet match | ERROR | No |

## Node Requirements

- MI355X with 8x GPUs (288GB VRAM each)
- 8x Pensando ionic RDMA NICs (ionic_0 through ionic_7)
- All nodes on the same leaf switch fabric (GID subnets must match)
- NFS/vast mount at `/mnt/vast/john/huggingface` with DeepSeek-R1-0528 model
- Docker with `amdprimus/dynamo-rocm-sglang:latest` image

### Known Node Issues

| Node | Status | Issue |
|------|--------|-------|
| chi2811 | Excluded | No ionic IB devices |
| chi2881 | Excluded | No NFS/vast mount |

## Troubleshooting

### MoRI RDMA `std::bad_cast` crash
Missing ionic IPv4 addresses. Run `bash setup_ionic_network.sh` on the node.
See [ionic-rdma-fixes.md](../../docs/ionic-rdma-fixes.md) Layer 1a.

### Server OOM during CUDA graph capture
Missing `MORI_SHMEM_MODE=ISOLATION` or `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK`.
These env vars are set automatically by `run_benchmark.sh` via Docker `-e` flags.

### Router "No available decode workers"
Management IP confusion: `hostname -I` returns ionic IP instead of management IP.
Scripts use `ip route get 1.1.1.1` to get the correct management IP.

### Benchmark returns 0 successful requests
Check router log for `"unrecognized arguments"` -- the `--decode` flag requires
separate invocations per URL: `--decode http://d1:8000 --decode http://d2:8000`.

### `--ep-dispatch-algorithm fake` crashes in TP-only mode
This flag is in InferenceX `base_flags` but only works when EP is enabled.
For TP-only configs (our 1P2D no_dp), omit it. The scripts handle this automatically.

### `setup_ionic_network.sh` runs but ionic still has no IPv4
The script can exit 0 without assigning all 8 addresses if the GID-to-subnet
map misses an interface. Verify with: `ip addr show | grep 'inet 192.168'`
(should show 8 lines). Preflight auto-retries and ERRORs if still < 8.

### CSV data analysis: 8K/1K rows leaking into 1K/1K results
When grepping the CSV for `1024,1024` (ISL,OSL), rows with `osl=1024,conc=1024`
from 8K/1K configs also match. Use awk column filtering (`$7==1024 && $8==1024`)
instead of grep patterns.
