"""Run the Mistral-7B fine-tune and the serving benchmark on Modal.

    modal run modal_app.py                 # download, fine-tune + merge, bench vLLM, bench HF baseline
    modal run modal_app.py --step bench    # only rerun the two benchmarks

GPU: one L4 (24 GB). It is the cheapest Modal GPU that holds Mistral-7B in bf16
(about 14.5 GB of weights) with room for LoRA training activations under gradient
checkpointing and for vLLM's KV cache. An A10 has about twice the memory bandwidth,
so decode would be faster, at about 1.4x the price. Set LFS_GPU=A10 to use it.

Results are written to the `llm-finetune-serve` volume and copied into results/.
"""
import os
import subprocess
import time
from pathlib import Path

import modal

GPU = os.environ.get("LFS_GPU", "L4")
CONFIG = "configs/mistral7b.yaml"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MERGED = "/vol/outputs/mistral7b/merged"
RESULTS = "/vol/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .uv_sync(extras=["train", "gpu"])  # exact versions from uv.lock, same as the CPU smoke test
    .env({"HF_HOME": "/vol/hf", "PYTHONPATH": "/root", "MLFLOW_TRACKING_URI": "sqlite:////vol/mlflow.db",
          "MLFLOW_DISABLE_AGENT_HINT": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/lfs", "/root/lfs")
    .add_local_dir("configs", "/root/configs")
)
vol = modal.Volume.from_name("llm-finetune-serve", create_if_missing=True)
app = modal.App("llm-finetune-serve", image=image)


def sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root")


@app.function(volumes={"/vol": vol}, timeout=30 * 60, cpu=4)
def download() -> None:
    """Pull the base weights onto the volume on a CPU container, so no GPU time is spent downloading."""
    from huggingface_hub import snapshot_download

    # The repo also ships consolidated.safetensors (Mistral format); skip it, the sharded files are enough.
    snapshot_download(
        MODEL_ID, allow_patterns=["*.json", "*.safetensors", "tokenizer*"], ignore_patterns=["consolidated*"]
    )
    vol.commit()


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=90 * 60, cpu=4, memory=48 * 1024)
def finetune() -> None:
    sh("python", "-m", "lfs.finetune", CONFIG)
    sh("python", "-m", "lfs.merge", CONFIG)
    vol.commit()


def wait_for(url: str, proc: subprocess.Popen, limit_s: int = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        if subprocess.run(["curl", "-sf", url], capture_output=True).returncode == 0:
            return
        time.sleep(3)
    raise TimeoutError(f"{url} not up after {limit_s}s")


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=45 * 60, cpu=4, memory=32 * 1024)
def bench(server: str) -> None:
    """Start one server on the GPU, benchmark it from the same container, stop it.

    The client runs next to the server so network latency does not enter TTFT.
    """
    if server == "vllm":
        cmd = ["python", "-m", "lfs.serve_vllm", "--model", MERGED, "--port", "8000", "--max-model-len", "4096"]
        levels, per_level = "1,4,16,32", "0"          # 0 = 4 x concurrency, at least 16
    elif server == "hf-generate":
        cmd = ["python", "-m", "lfs.hf_server", "--model", MERGED, "--dtype", "bfloat16", "--port", "8000"]
        levels, per_level = "1,4,16", "16"           # serial server: 32-way adds only queueing time
    else:
        raise ValueError(server)
    proc = subprocess.Popen(cmd, cwd="/root")
    try:
        wait_for("http://localhost:8000/v1/models", proc)
        Path(RESULTS).mkdir(parents=True, exist_ok=True)
        sh("python", "-m", "lfs.bench.cli", "--base-url", "http://localhost:8000", "--server", server,
           "--model", MERGED, "--dtype", "bfloat16", "--concurrency", levels, "--requests-per-level", per_level,
           "--warmup", "4", "--input-tokens", "256", "--max-tokens", "128", "--out", f"{RESULTS}/{server}")
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    vol.commit()


@app.local_entrypoint()
def main(step: str = "all") -> None:
    if step in ("all", "finetune"):
        download.remote()
        finetune.remote()
    if step in ("all", "bench"):
        bench.remote("vllm")
        bench.remote("hf-generate")
    out = Path("results")
    out.mkdir(exist_ok=True)
    for server in ("vllm", "hf-generate"):
        for ext in ("csv", "md"):
            name = f"{server}.{ext}"
            try:
                data = b"".join(vol.read_file(f"results/{name}"))
            except FileNotFoundError:
                print(f"missing on volume: results/{name}")
                continue
            (out / name).write_bytes(data)
            print(f"wrote results/{name}")
