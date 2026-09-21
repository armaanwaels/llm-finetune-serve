.PHONY: install install-gpu test lint smoke finetune merge serve serve-hf bench modal

CONFIG ?= configs/mistral7b.yaml
MODEL ?= outputs/mistral7b/merged
URL ?= http://localhost:8000
SERVER ?= vllm
OUT ?= results/$(SERVER)
CONCURRENCY ?= 1,4,16,32

install:            ## CPU: training stack + tests (no vLLM)
	uv sync --extra train

install-gpu:        ## Linux + NVIDIA GPU
	uv sync --extra train --extra gpu

test:
	uv run pytest -q

lint:
	uv run ruff check .

smoke:              ## CPU end to end on Qwen2.5-0.5B: fine-tune, merge, HF server, bench
	uv run ./scripts/smoke.sh

finetune:
	uv run python -m lfs.finetune $(CONFIG)
	uv run python -m lfs.merge $(CONFIG)

serve:              ## vLLM, OpenAI-compatible, continuous batching
	uv run python -m lfs.serve_vllm --model $(MODEL)

serve-hf:           ## naive HF generate baseline, one request at a time
	uv run python -m lfs.hf_server --model $(MODEL) --dtype bfloat16 --port 8000

bench:              ## benchmark whatever is listening on $(URL)
	uv run python -m lfs.bench.cli --base-url $(URL) --server $(SERVER) --model $(MODEL) \
		--concurrency $(CONCURRENCY) --out $(OUT)

modal:              ## full GPU run on Modal (bills your Modal account)
	modal run modal_app.py
