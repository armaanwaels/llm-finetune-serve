"""Load a YAML run config. One file describes model, data, LoRA and training."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("model", "data", "lora", "train", "output_dir"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required key '{key}'")
    return cfg
