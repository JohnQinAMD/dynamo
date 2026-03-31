#!/bin/bash
#SBATCH --job-name=dynamo-disagg
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=logs/dynamo-disagg-%j.log
#SBATCH --error=logs/dynamo-disagg-%j.err
#
# Dynamo disaggregated P/D serving on 2x AMD MI355X nodes
# Node 0 = Prefill + Frontend + etcd + NATS
# Node 1 = Decode
#
# Usage: sbatch dynamo_disagg.slurm
#
# Override defaults:
#   MODEL=/models/Qwen2.5-0.5B-Instruct TP=1 BACKEND=mori sbatch dynamo_disagg.slurm

set -euo pipefail
mkdir -p logs

MODEL="${MODEL:-/models/DeepSeek-V3}"
TP="${TP:-8}"
BACKEND="${BACKEND:-mori}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2}"
DYNAMO_SRC="${DYNAMO_SRC:-/mnt/vast/john/rocm-dynamo}"
PORT="${PORT:-8000}"

ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
PREFILL_NODE=$(echo "$ALL_NODES" | head -1)
DECODE_NODE=$(echo "$ALL_NODES" | tail -1)
PREFILL_IP=$(srun --nodes=1 --ntasks=1 --nodelist="$PREFILL_NODE" hostname -I | awk '{print $1}')

echo "============================================"
echo "  Dynamo Disaggregated Serving"
echo "============================================"
echo "Prefill: $PREFILL_NODE ($PREFILL_IP)"
echo "Decode:  $DECODE_NODE"
echo "Model:   $MODEL | TP: $TP | Backend: $BACKEND"
echo "Job:     $SLURM_JOB_ID"
echo "============================================"

# Common Docker args
DOCKER_ARGS="--rm --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v ${DYNAMO_SRC}:/workspace \
    -v $(dirname $MODEL):/models:ro \
    -e SGLANG_AITER_MLA_PERSIST=False \
    -e RCCL_MSCCL_ENABLE=0 \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

# Extra env for RIXL DRAM staging
STAGING_ENV=""
if [ "$BACKEND" = "nixl" ]; then
    STAGING_ENV="-e SGLANG_NIXL_ROCM_STAGING=1"
fi

# Build script (shared between nodes)
BUILD_SCRIPT='
set -e
cp -r /workspace/dynamo /tmp/dynamo && cd /tmp/dynamo
MODE=develop bash scripts/build_dynamo_wheel.sh 2>&1 | tail -3
# Install etcd + NATS
which etcd >/dev/null 2>&1 || {
    wget -q https://github.com/etcd-io/etcd/releases/download/v3.5.21/etcd-v3.5.21-linux-amd64.tar.gz -O /tmp/etcd.tar.gz
    mkdir -p /usr/local/bin/etcd-dir && tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd-dir --strip-components=1
    ln -sf /usr/local/bin/etcd-dir/etcd /usr/local/bin/etcd
}
which nats-server >/dev/null 2>&1 || {
    wget -q https://github.com/nats-io/nats-server/releases/download/v2.10.28/nats-server-v2.10.28-amd64.deb -O /tmp/nats.deb
    dpkg -i /tmp/nats.deb >/dev/null 2>&1
}
'

MODEL_BASE=$(basename "$MODEL")

# --- Prefill Node ---
srun --nodes=1 --ntasks=1 --nodelist="$PREFILL_NODE" \
    docker run $DOCKER_ARGS $STAGING_ENV \
    --name "dynamo-prefill-${SLURM_JOB_ID}" \
    "$DOCKER_IMAGE" \
    bash -c "
$BUILD_SCRIPT

MY_IP=\$(hostname -I | awk '{print \$1}')
rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://\${MY_IP}:2379 &
nats-server -p 4222 -js --client_advertise \${MY_IP}:4222 &
sleep 3
export ETCD_ENDPOINTS=http://\${MY_IP}:2379
export NATS_SERVER=nats://\${MY_IP}:4222

cd /tmp/dynamo
python3 -m dynamo.frontend --http-port $PORT --router-mode round-robin &
sleep 2

# Auto-detect ionic device (pick first available)
IONIC_DEV=\$(ls /sys/class/infiniband/ 2>/dev/null | grep ionic | head -1 || echo '')

python3 -m dynamo.sglang \
    --model-path /models/$MODEL_BASE \
    --tp-size $TP --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 \
    --chunked-prefill-size 32768 --mem-fraction-static 0.80 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend $BACKEND \
    \$([ -n \"\$IONIC_DEV\" ] && echo \"--disaggregation-ib-device \$IONIC_DEV\")
" &

# --- Decode Node ---
srun --nodes=1 --ntasks=1 --nodelist="$DECODE_NODE" \
    docker run $DOCKER_ARGS $STAGING_ENV \
    --name "dynamo-decode-${SLURM_JOB_ID}" \
    "$DOCKER_IMAGE" \
    bash -c "
$BUILD_SCRIPT

export ETCD_ENDPOINTS=http://${PREFILL_IP}:2379
export NATS_SERVER=nats://${PREFILL_IP}:4222

cd /tmp/dynamo

IONIC_DEV=\$(ls /sys/class/infiniband/ 2>/dev/null | grep ionic | head -1 || echo '')

python3 -m dynamo.sglang \
    --model-path /models/$MODEL_BASE \
    --tp-size $TP --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 \
    --mem-fraction-static 0.80 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend $BACKEND \
    \$([ -n \"\$IONIC_DEV\" ] && echo \"--disaggregation-ib-device \$IONIC_DEV\")
" &

wait
