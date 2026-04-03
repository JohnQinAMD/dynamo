#!/usr/bin/env python3
"""Parse InferenceX models.yaml and emit shell variables for benchmark scripts.

Replicates the config resolution logic from InferenceX's server.sh (lines 75-320)
and env.sh (lines 92-129) so our run_benchmark.sh stays automatically aligned.

Usage:
    eval "$(python3 infx_config.py --model DeepSeek-R1-0528 --prefill-tp 8 --decode-tp 8)"
    eval "$(python3 infx_config.py --model DeepSeek-R1-0528 --prefill-tp 8 --decode-tp 8 --ep --dp --mtp 1)"
"""

import argparse
import os
import sys
import yaml


def parse_range(cuda_range, default_start=1, default_end=128):
    s = str(cuda_range)
    if "-" in s:
        parts = s.split("-")
        return int(parts[0]), int(parts[1])
    return default_start, default_end


def eval_formula(val, env_vars):
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val)
    ns = {}
    for k, v in env_vars.items():
        try:
            ns[k] = int(v)
        except (ValueError, TypeError):
            pass
    try:
        return int(eval(s, {"__builtins__": {}}, ns))
    except Exception:
        return int(val) if str(val).isdigit() else 262144


def main():
    p = argparse.ArgumentParser(description="Resolve InferenceX config to shell vars")
    p.add_argument("--model", default="DeepSeek-R1-0528")
    p.add_argument("--models-yaml", default=None,
                    help="Path to models.yaml (auto-detected from workspace)")
    p.add_argument("--prefill-tp", type=int, default=8)
    p.add_argument("--decode-tp", type=int, default=8)
    p.add_argument("--prefill-ep", type=str, default="false")
    p.add_argument("--decode-ep", type=str, default="false")
    p.add_argument("--prefill-dp", type=str, default="false")
    p.add_argument("--decode-dp", type=str, default="false")
    p.add_argument("--ep", action="store_true", help="Shorthand: enable EP on both prefill and decode")
    p.add_argument("--dp", action="store_true", help="Shorthand: enable DP on both prefill and decode")
    p.add_argument("--mtp", type=int, default=0, help="Decode MTP size (0=disabled)")
    p.add_argument("--ibdevices", default="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7")
    p.add_argument("--rdma-tc", default="104")
    args = p.parse_args()

    if args.ep:
        args.prefill_ep = "true"
        args.decode_ep = "true"
    if args.dp:
        args.prefill_dp = "true"
        args.decode_dp = "true"

    # Find models.yaml
    if args.models_yaml:
        yaml_path = args.models_yaml
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "..", "InferenceX", "benchmarks", "multi_node", "amd_utils", "models.yaml"),
            os.path.join(script_dir, "..", "..", "InferenceX", "benchmarks", "multi_node", "amd_utils", "models.yaml"),
            os.path.join(script_dir, "benchmark", "models.yaml"),
        ]
        yaml_path = None
        for c in candidates:
            if os.path.exists(c):
                yaml_path = os.path.realpath(c)
                break
        if not yaml_path:
            print(f'echo "ERROR: models.yaml not found. Searched: {candidates}"', file=sys.stderr)
            sys.exit(1)

    with open(yaml_path) as f:
        models = yaml.safe_load(f)

    if args.model not in models:
        print(f'echo "ERROR: Model {args.model} not found in {yaml_path}"')
        print(f'echo "Available: {", ".join(models.keys())}"')
        print("return 1 2>/dev/null || exit 1")
        sys.exit(0)

    m = models[args.model]

    # -- MoRI env vars (from env.sh lines 92-129) --
    is_mxfp4 = "mxfp4" in args.model.lower()
    mori_dispatch_prefill = 12288 if is_mxfp4 else 16384
    mori_dispatch_decode = 160
    if args.mtp > 0:
        mori_dispatch_decode = mori_dispatch_decode * (args.mtp + 1)
    inter_kernel_threshold = mori_dispatch_decode * 2

    env_vars = {
        "MORI_MAX_DISPATCH_TOKENS_PREFILL": mori_dispatch_prefill,
        "MORI_MAX_DISPATCH_TOKENS_DECODE": mori_dispatch_decode,
        "PREFILL_TP_SIZE": args.prefill_tp,
        "DECODE_TP_SIZE": args.decode_tp,
        "xP": 1,
    }

    # -- Parse model YAML (replicates server.sh lines 75-163) --
    prefill_cfg = m.get("prefill", {})
    decode_cfg = m.get("decode", {})

    prefill_mem = prefill_cfg.get("mem_fraction_static", 0.8)
    prefill_disable_radix = prefill_cfg.get("disable_radix_cache", True)
    decode_mem = decode_cfg.get("mem_fraction_static", 0.85)
    decode_round_robin = decode_cfg.get("prefill_round_robin_balance", True)

    # Prefill: dp vs no_dp
    if args.prefill_dp == "true":
        pf = prefill_cfg.get("dp", {})
        prefill_max_running = pf.get("max_running_requests", 24)
        prefill_chunked = eval_formula(pf.get("chunked_prefill_size", 262144), env_vars)
        prefill_cg_bs = pf.get("cuda_graph_bs", "1 2 3")
    else:
        pf = prefill_cfg.get("no_dp", {})
        prefill_max_running = pf.get("max_running_requests", 128)
        prefill_chunked = eval_formula(pf.get("chunked_prefill_size", 262144), env_vars)
        rng = pf.get("cuda_graph_bs_range", "1-128")
        s, e = parse_range(rng)
        prefill_cg_bs = " ".join(str(x) for x in range(s, e + 1))

    # Decode: dp > ep_only > no_dp (replicates server.sh lines 178-188)
    if args.decode_dp == "true":
        dc = decode_cfg.get("dp", {})
        rng = dc.get("cuda_graph_bs_range", "1-160")
        s, e = parse_range(rng)
        decode_cg_bs = " ".join(str(x) for x in range(s, e + 1))
        decode_max_running = e * args.decode_tp
        decode_chunked = eval_formula(dc.get("chunked_prefill_size", 262144), env_vars)
    elif args.decode_ep == "true":
        dc = decode_cfg.get("ep_only", {})
        rng = dc.get("cuda_graph_bs_range", "1-256")
        s, e = parse_range(rng)
        decode_cg_bs = " ".join(str(x) for x in range(s, e + 1))
        decode_max_running = dc.get("max_running_requests", 256)
        decode_chunked = eval_formula(dc.get("chunked_prefill_size", 262144), env_vars)
    else:
        dc = decode_cfg.get("no_dp", {})
        rng = dc.get("cuda_graph_bs_range", "1-128")
        s, e = parse_range(rng)
        decode_cg_bs = " ".join(str(x) for x in range(s, e + 1))
        decode_max_running = dc.get("max_running_requests", 128)
        decode_chunked = eval_formula(dc.get("chunked_prefill_size", 262144), env_vars)

    # -- Build parallelism args (replicates build_server_config lines 258-279) --
    def parallel_args(tp, enable_ep, enable_dp):
        parts = [f"--tp-size {tp}"]
        if enable_ep == "true":
            parts.append(f"--ep-size {tp}")
        if enable_dp == "true":
            parts.append(f"--dp-size {tp}")
        return " ".join(parts)

    prefill_parallel = parallel_args(args.prefill_tp, args.prefill_ep, args.prefill_dp)
    decode_parallel = parallel_args(args.decode_tp, args.decode_ep, args.decode_dp)

    # -- Cross-TP flags (server.sh lines 190-198) --
    cross_tp = ""
    if args.prefill_dp != args.decode_dp:
        if args.decode_dp == "true":
            cross_tp = f"--disaggregation-decode-tp {args.decode_tp} --disaggregation-decode-dp {args.decode_tp}"
        else:
            cross_tp = f"--disaggregation-decode-tp {args.decode_tp} --disaggregation-decode-dp 1"

    # -- Base flags from YAML --
    base_flags = m.get("base_flags", "")
    mtp_flags = ""
    if args.mtp > 0:
        mtp_flags = f'{m.get("mtp_flags", "")} --speculative-num-steps {args.mtp} --speculative-num-draft-tokens {args.mtp + 1}'
    dp_flags = m.get("dp_flags", "") if (args.prefill_dp == "true" or args.decode_dp == "true") else ""

    # -- Compose PREFILL_FLAGS (server.sh lines 200-204, 304-317) --
    prefill_mode = f"--mem-fraction-static {prefill_mem} --max-running-requests {prefill_max_running} --chunked-prefill-size {prefill_chunked} --cuda-graph-bs {prefill_cg_bs}"
    if cross_tp:
        prefill_mode += f" {cross_tp}"
    if prefill_disable_radix:
        prefill_mode += " --disable-radix-cache"

    prefill_flags = f"{prefill_parallel} {base_flags}"
    if dp_flags and args.prefill_dp == "true":
        prefill_flags += f" {dp_flags}"
    prefill_flags += f" {prefill_mode}"

    # -- Compose DECODE_FLAGS (server.sh lines 206-213, 304-317) --
    decode_mode = f"--mem-fraction-static {decode_mem} --max-running-requests {decode_max_running} --cuda-graph-bs {decode_cg_bs}"
    if decode_round_robin:
        decode_mode += " --prefill-round-robin-balance"

    decode_flags = f"{decode_parallel} {base_flags}"
    if args.mtp > 0:
        decode_flags += f" {mtp_flags}"
    if dp_flags and args.decode_dp == "true":
        decode_flags += f" {dp_flags}"
    decode_flags += f" {decode_mode}"

    # -- Docker env args (from env.sh) --
    docker_env = " ".join([
        "-e SGLANG_USE_AITER=1",
        "-e RCCL_MSCCL_ENABLE=0",
        "-e ROCM_QUICK_REDUCE_QUANTIZATION=INT4",
        f"-e MORI_RDMA_TC={args.rdma_tc}",
        "-e MORI_SHMEM_MODE=ISOLATION",
        f"-e SGLANG_MORI_FP8_DISP={'False' if is_mxfp4 else 'True'}",
        "-e SGLANG_MORI_FP4_DISP=False",
        "-e SGLANG_MORI_FP8_COMB=False",
        "-e MORI_EP_LAUNCH_CONFIG_MODE=AUTO",
        "-e MORI_IO_QP_MAX_SEND_WR=16384",
        "-e MORI_IO_QP_MAX_CQE=32768",
        "-e MORI_IO_QP_MAX_SGE=4",
        "-e MORI_APP_LOG_LEVEL=INFO",
        "-e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200",
        "-e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200",
        f"-e NCCL_IB_HCA={args.ibdevices}",
        f"-e MORI_MAX_DISPATCH_TOKENS_PREFILL={mori_dispatch_prefill}",
        f"-e MORI_MAX_DISPATCH_TOKENS_DECODE={mori_dispatch_decode}",
        f"-e SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD={inter_kernel_threshold}",
        "-e PYTHONDONTWRITEBYTECODE=1",
    ])

    # -- Output shell variables --
    print(f'BL_PREFILL_FLAGS="{prefill_flags}"')
    print(f'BL_DECODE_FLAGS="{decode_flags}"')
    print(f'BL_DOCKER_ENV_ARGS="{docker_env}"')
    print(f'BL_MORI_DISPATCH_PREFILL={mori_dispatch_prefill}')
    print(f'BL_MORI_DISPATCH_DECODE={mori_dispatch_decode}')
    print(f'BL_PREFILL_TP={args.prefill_tp}')
    print(f'BL_DECODE_TP={args.decode_tp}')
    print(f'BL_PREFILL_EP={args.prefill_ep}')
    print(f'BL_DECODE_EP={args.decode_ep}')
    print(f'BL_PREFILL_DP={args.prefill_dp}')
    print(f'BL_DECODE_DP={args.decode_dp}')
    print(f'BL_MTP={args.mtp}')
    print(f'BL_MODELS_YAML="{yaml_path}"')
    print(f'BL_MODEL_NAME="{args.model}"')

    # Debug: print what we resolved
    print(f'echo "  Config: model={args.model} P:TP{args.prefill_tp}/EP={args.prefill_ep}/DP={args.prefill_dp} D:TP{args.decode_tp}/EP={args.decode_ep}/DP={args.decode_dp} MTP={args.mtp}"')
    print(f'echo "  Prefill max_running={prefill_max_running} chunked={prefill_chunked} mem={prefill_mem}"')
    print(f'echo "  Decode  max_running={decode_max_running} mem={decode_mem}"')


if __name__ == "__main__":
    main()
