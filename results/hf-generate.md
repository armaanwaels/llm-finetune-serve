| server | concurrency | output_tok_per_s | req_per_s | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | e2e_p50_ms | e2e_p95_ms | e2e_p99_ms | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| hf-generate | 1 | 15.7 | 0.12 | 203.3 | 273.3 | 274.5 | 8113.7 | 8264.7 | 8381.2 | 0 |
| hf-generate | 4 | 15.9 | 0.12 | 24430.8 | 24479.8 | 24493.8 | 32253.4 | 32346.2 | 32349.0 | 0 |
| hf-generate | 16 | 15.6 | 0.12 | 61294.8 | 116895.7 | 121825.9 | 69340.2 | 124934.8 | 129917.0 | 0 |

Run environment: cpu=x86_64, dtype=bfloat16, gpu=NVIDIA L4, gpu_mem_total_mib=23034, gpu_mem_used_mib=14304, ignore_eos=True, input_tokens=256, max_tokens=128, model=/vol/outputs/mistral7b/merged, timestamp_utc=2026-09-21T22:33:12Z, torch=2.11.0, transformers=4.57.6, vllm=0.20.2, warmup=4
