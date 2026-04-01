#!/bin/bash
# Run vLLM tests inside vllm/vllm-openai-rocm container
set -e
cd /workspace/dynamo

pip install -q -e . pytest pytest-benchmark pytest-httpserver pytest-asyncio pytest-timeout nats-py filelock 2>&1 | tail -2

# Install etcd + nats
which etcd >/dev/null 2>&1 || {
    wget -q https://github.com/etcd-io/etcd/releases/download/v3.5.21/etcd-v3.5.21-linux-amd64.tar.gz -O /tmp/etcd.tar.gz
    mkdir -p /usr/local/bin/etcd-dir && tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd-dir --strip-components=1
    ln -sf /usr/local/bin/etcd-dir/etcd /usr/local/bin/etcd
}
which nats-server >/dev/null 2>&1 || {
    wget -q https://github.com/nats-io/nats-server/releases/download/v2.10.28/nats-server-v2.10.28-amd64.deb -O /tmp/nats.deb
    dpkg -i /tmp/nats.deb >/dev/null 2>&1
}

echo "Environment:"
python3 --version
python3 -c "import vllm; print(f'vllm {vllm.__version__}')"
python3 -c "import dynamo; print('dynamo OK')"
echo ""

PY='python3 -m pytest --override-ini=filterwarnings=default'

echo "============================================"
echo "  vLLM E2E Tests on ROCm"
echo "============================================"
echo ""

echo "=== vLLM Serve (aggregated) ==="
$PY tests/serve/test_vllm.py \
    -k "aggregated and not logprobs and not lora and not multimodal and not toolcalling" \
    -v --tb=short --timeout=600 2>&1

echo ""
echo "=== vLLM Frontend ==="
$PY tests/frontend/test_vllm.py \
    tests/frontend/test_vllm_prepost_integration.py \
    -v --tb=short --timeout=300 2>&1

echo ""
echo "=== vLLM Frontend Prepost ==="
$PY tests/frontend/test_prepost.py \
    -v --tb=short --timeout=120 2>&1

echo "DONE"
