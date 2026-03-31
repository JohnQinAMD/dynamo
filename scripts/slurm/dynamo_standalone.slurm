#!/bin/bash
#SBATCH --job-name=dynamo-standalone
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=logs/dynamo-standalone-%j.log
#SBATCH --error=logs/dynamo-standalone-%j.err
#
# Dynamo single-node standalone on AMD MI355X
# Usage: sbatch dynamo_standalone.slurm
#
# Override defaults:
#   MODEL=/models/Qwen2.5-0.5B-Instruct TP=1 sbatch dynamo_standalone.slurm

set -euo pipefail
mkdir -p logs

MODEL="${MODEL:-/models/DeepSeek-V3}"
TP="${TP:-8}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2}"
DYNAMO_SRC="${DYNAMO_SRC:-/mnt/vast/john/rocm-dynamo}"
PORT="${PORT:-8000}"

MY_IP=$(hostname -I | awk '{print $1}')
echo "Node: $(hostname) ($MY_IP) | Model: $MODEL | TP: $TP | Job: $SLURM_JOB_ID"

srun docker run --rm \
    --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --shm-size 256G --ipc=host \
    -v "${DYNAMO_SRC}:/workspace" \
    -v "$(dirname $MODEL):/models:ro" \
    -e SGLANG_AITER_MLA_PERSIST=False \
    -e RCCL_MSCCL_ENABLE=0 \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "$DOCKER_IMAGE" \
    bash -c "
set -e
cp -r /workspace/dynamo /tmp/dynamo && cd /tmp/dynamo
MODE=develop bash scripts/build_dynamo_wheel.sh 2>&1 | tail -3

MY_IP=\$(hostname -I | awk '{print \$1}')

# etcd + NATS
wget -q https://github.com/etcd-io/etcd/releases/download/v3.5.21/etcd-v3.5.21-linux-amd64.tar.gz -O /tmp/etcd.tar.gz
mkdir -p /usr/local/bin/etcd-dir && tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd-dir --strip-components=1
ln -sf /usr/local/bin/etcd-dir/etcd /usr/local/bin/etcd
wget -q https://github.com/nats-io/nats-server/releases/download/v2.10.28/nats-server-v2.10.28-amd64.deb -O /tmp/nats.deb
dpkg -i /tmp/nats.deb >/dev/null 2>&1

rm -rf /tmp/default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://\${MY_IP}:2379 &
nats-server -p 4222 -js &
sleep 3
export ETCD_ENDPOINTS=http://\${MY_IP}:2379
export NATS_SERVER=nats://\${MY_IP}:4222

python3 -m dynamo.frontend --http-port $PORT --router-mode round-robin &
sleep 2

python3 -m dynamo.sglang \
    --model-path /models/$(basename $MODEL) \
    --tp-size $TP --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 16 \
    --chunked-prefill-size 32768 --mem-fraction-static 0.80 \
    --host 0.0.0.0
"
