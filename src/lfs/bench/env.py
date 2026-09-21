"""Record what the numbers were measured on."""
import datetime as dt
import importlib.metadata as md
import platform
import shutil
import subprocess


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def gpu_info() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"gpu": "none (CPU)", "gpu_mem_total_mib": "", "gpu_mem_used_mib": ""}
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    name, total, used = [x.strip() for x in out.split(",")]
    return {"gpu": name, "gpu_mem_total_mib": total, "gpu_mem_used_mib": used}


def capture(model: str, dtype: str) -> dict:
    return {
        "model": model,
        "dtype": dtype,
        **gpu_info(),
        "cpu": platform.processor() or platform.machine(),
        "torch": _version("torch"),
        "vllm": _version("vllm"),
        "transformers": _version("transformers"),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
