| server | concurrency | output_tok_per_s | req_per_s | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | e2e_p50_ms | e2e_p95_ms | e2e_p99_ms | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| vllm | 1 | 17.7 | 0.14 | 137.9 | 143.9 | 146.5 | 7265.5 | 7274.1 | 7275.9 | 0 |
| vllm | 4 | 67.3 | 0.53 | 170.9 | 173.2 | 173.4 | 7619.0 | 7622.0 | 7622.3 | 0 |
| vllm | 16 | 231.6 | 1.81 | 746.9 | 1347.9 | 1354.3 | 9018.2 | 9215.7 | 9221.6 | 0 |
| vllm | 32 | 404.0 | 3.16 | 307.4 | 2396.6 | 2400.2 | 9817.8 | 11764.5 | 11770.3 | 0 |

Run environment: cpu=x86_64, dtype=bfloat16, gpu=NVIDIA L4, gpu_mem_total_mib=23034, gpu_mem_used_mib=20618, ignore_eos=True, input_tokens=256, max_tokens=128, model=/vol/outputs/mistral7b/merged, timestamp_utc=2026-09-21T22:24:26Z, torch=2.11.0, transformers=4.57.6, vllm=0.20.2, warmup=4
