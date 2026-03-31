#!/bin/bash
#SBATCH --job-name=dynamo-kv-router
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=logs/dynamo-kv-router-%j.log
#SBATCH --error=logs/dynamo-kv-router-%j.err
#
# Dynamo 2-node KV Router scale-out on AMD MI355X
# Node 0 = Frontend + etcd + NATS + Worker-1
# Node 1 = Worker-2
#
# Usage: sbatch dynamo_kv_router.slurm

set -euo pipefail
mkdir -p logs

MODEL="${MODEL:-/models/DeepSeek-V3}"
TP="${TP:-8}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2}"
DYNAMO_SRC="${DYNAMO_SRC:-/mnt/vast/john/rocm-dynamo}"
PORT="${PORT:-8000}"

ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE0=$(echo "$ALL_NODES" | head -1)
NODE1=$(echo "$ALL_NODES" | tail -1)
NODE0_IP=$(srun --nodes=1 --ntasks=1 --nodelist="$NODE0" hostname -I | awk '{print $1}')

echo "Node0: $NODE0 ($NODE0_IP) | Node1: $NODE1 | Model: $MODEL"

DOCKER_ARGS="--rm --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --shm-size 256G --ipc=host \
    -v ${DYNAMO_SRC}:/workspace \
    -v $(dirname $MODEL):/models:ro \
    -e SGLANG_AITER_MLA_PERSIST=False \
    -e RCCL_MSCCL_ENABLE=0 \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

BUILD_SCRIPT='
set -e
cp -r /workspace/dynamo /tmp/dynamo && cd /tmp/dynamo
MODE=develop bash scripts/build_dynamo_wheel.sh 2>&1 | tail -3
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

# Node 0: Frontend + Worker-1
srun --nodes=1 --ntasks=1 --nodelist="$NODE0" \
    docker run $DOCKER_ARGS \
    --name "dynamo-kv0-${SLURM_JOB_ID}" \
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
python3 -m dynamo.frontend --http-port $PORT --router-mode kv --kv-cache-block-size 16 &
sleep 2
python3 -m dynamo.sglang --model-path /models/$MODEL_BASE \
    --tp-size $TP --page-size 16 --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 --mem-fraction-static 0.80
" &

# Node 1: Worker-2
srun --nodes=1 --ntasks=1 --nodelist="$NODE1" \
    docker run $DOCKER_ARGS \
    --name "dynamo-kv1-${SLURM_JOB_ID}" \
    "$DOCKER_IMAGE" \
    bash -c "
$BUILD_SCRIPT
export ETCD_ENDPOINTS=http://${NODE0_IP}:2379
export NATS_SERVER=nats://${NODE0_IP}:4222
cd /tmp/dynamo
python3 -m dynamo.sglang --model-path /models/$MODEL_BASE \
    --tp-size $TP --page-size 16 --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 --mem-fraction-static 0.80
" &

wait
