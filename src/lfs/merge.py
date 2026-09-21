"""Fold the LoRA adapter into the base weights so vLLM serves a plain model.

    python -m lfs.merge configs/smoke.yaml
"""
import argparse
from pathlib import Path

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from lfs.config import load_config
from lfs.finetune import DTYPES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    out = Path(cfg["output_dir"])
    model = AutoPeftModelForCausalLM.from_pretrained(out / "adapter", dtype=DTYPES[cfg["model"]["dtype"]])
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    AutoTokenizer.from_pretrained(out / "adapter").save_pretrained(out / "merged")
    print(f"merged model saved to {out / 'merged'}")


if __name__ == "__main__":
    main()
