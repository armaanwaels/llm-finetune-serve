"""Launch vLLM's OpenAI-compatible server on the merged model.

    python -m lfs.serve_vllm --model outputs/mistral7b/merged

Continuous batching is vLLM's default scheduler; the flags below only bound it.
"""
import argparse
import os
import shlex


def vllm_command(model: str, port: int, dtype: str, max_model_len: int, max_num_seqs: int, gpu_mem: float) -> list[str]:
    return [
        "vllm", "serve", model,
        "--port", str(port),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", str(gpu_mem),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    a = ap.parse_args()
    cmd = vllm_command(a.model, a.port, a.dtype, a.max_model_len, a.max_num_seqs, a.gpu_memory_utilization)
    print("+", shlex.join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
