#!/bin/bash
# Quick ROCm patch health check
set -e
echo "=== ROCm Pre-commit Check ==="

# 1. Verify HIP kernel exists
test -f lib/kvbm-kernels/hip/tensor_kernels.hip && echo "PASS: HIP kernel exists" || echo "FAIL: HIP kernel missing"

# 2. Verify rocm features in Cargo
grep -q 'rocm' lib/memory/Cargo.toml && echo "PASS: Memory crate has rocm feature" || echo "FAIL"
grep -q 'rocm' lib/llm/Cargo.toml && echo "PASS: LLM crate has rocm feature" || echo "FAIL"

# 3. Verify Dockerfiles have rocm blocks
grep -q 'device == "rocm"' container/templates/vllm_runtime.Dockerfile && echo "PASS: vLLM Dockerfile has rocm" || echo "FAIL"

# 4. Check additiveness
if command -v git &> /dev/null && git rev-parse --is-inside-work-tree &> /dev/null; then
    UPSTREAM=$(git merge-base HEAD main 2>/dev/null || echo "")
    if [ -n "$UPSTREAM" ]; then
        DEL=$(git diff "$UPSTREAM"..HEAD | grep "^-[^-]" | wc -l)
        ADD=$(git diff "$UPSTREAM"..HEAD | grep "^+[^+]" | wc -l)
        echo "Additive ratio: +$ADD / -$DEL"
    fi
fi

echo "=== Check Complete ==="
