#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin
RUN_DIR="${ROOT}/eval_runs/yulan_moe_128k_aime2024/$(date +%Y%m%d_%H%M%S)"
PYTHON=/mnt/yulan/lvzhihao/PostTrain/slime/.venv/bin/python

curl -fsS --max-time 5 http://127.0.0.1:30110/health >/dev/null
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

nohup env PYTHONUNBUFFERED=1 "${PYTHON}" \
  "${ROOT}/scripts/eval_yulan_moe_128k_aime2024.py" \
  >eval.log 2>&1 </dev/null &

echo $! >eval.pid
echo "PID: $(cat "${RUN_DIR}/eval.pid")"
echo "Log: ${RUN_DIR}/eval.log"
echo "Result: ${RUN_DIR}/aime2024_yulan_moe_128k_results.jsonl"
