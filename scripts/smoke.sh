#!/usr/bin/env bash
# CPU smoke test: tiny fine-tune, merge, serve with the HF baseline, benchmark it.
# Proves the pipeline runs. The numbers it prints say nothing about GPU serving.
set -euo pipefail
export MLFLOW_DISABLE_AGENT_HINT=1
CFG=configs/smoke.yaml
PORT=${PORT:-8011}

python -m lfs.finetune "$CFG"
python -m lfs.merge "$CFG"

python -m lfs.hf_server --model outputs/smoke/merged --dtype float32 --port "$PORT" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf "localhost:$PORT/v1/models" >/dev/null && break
  sleep 1
done

python -m lfs.bench.cli --base-url "http://localhost:$PORT" --server hf-generate \
  --model outputs/smoke/merged --dtype float32 --concurrency 1,2 --requests-per-level 4 \
  --warmup 1 --input-tokens 64 --max-tokens 16 --out results/smoke/hf-generate-cpu
