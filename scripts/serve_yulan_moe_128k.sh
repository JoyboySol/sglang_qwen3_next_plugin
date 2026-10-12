#!/usr/bin/env bash
set -euo pipefail

cd /mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin

export CUDA_VISIBLE_DEVICES=0,1
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin
export PATH=/mnt/yulan/lvzhihao/PostTrain/slime/.venv/bin:$PATH

/mnt/yulan/lvzhihao/PostTrain/slime/.venv/bin/python \
  -m sglang.launch_server \
  --model-path /mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k \
  --host 127.0.0.1 \
  --port 30110 \
  --context-length 131072 \
  --tp-size 2 \
  --ep-size 2 \
  --dtype bfloat16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --mem-fraction-static 0.85
