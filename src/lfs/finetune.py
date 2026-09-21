"""LoRA fine-tune with PEFT + TRL. Everything comes from the YAML config.

    python -m lfs.finetune configs/smoke.yaml
"""
import argparse
import os
from pathlib import Path

import mlflow
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from lfs.config import load_config
from lfs.data import load_splits

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    adapter_dir = out / "adapter"

    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"], dtype=DTYPES[cfg["model"]["dtype"]], device_map=cfg["model"].get("device_map")
    )

    d = cfg["data"]
    train_ds, eval_ds = load_splits(d["id"], d["n_train"], d["n_eval"], d["seed"])

    t = cfg["train"]
    sft = SFTConfig(
        output_dir=str(out / "checkpoints"),
        max_length=t["max_length"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"],
        num_train_epochs=t.get("epochs", 1),
        max_steps=t.get("max_steps", -1),
        warmup_ratio=t.get("warmup_ratio", 0.0),
        lr_scheduler_type="cosine",
        logging_steps=t["logging_steps"],
        eval_strategy="steps" if t.get("eval_steps") else "no",
        eval_steps=t.get("eval_steps"),
        save_strategy="no",
        bf16=cfg["model"]["dtype"] == "bfloat16",
        gradient_checkpointing=t.get("gradient_checkpointing", False),
        use_cpu=not torch.cuda.is_available(),
        report_to=["mlflow"],
        run_name=cfg.get("run_name", out.name),
        seed=d["seed"],
    )
    lora = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg.get("mlflow_uri", "sqlite:///mlflow.db")))
    mlflow.set_experiment(cfg.get("experiment", "llm-finetune-serve"))

    trainer = SFTTrainer(
        model=model, args=sft, train_dataset=train_ds, eval_dataset=eval_ds if sft.eval_strategy != "no" else None,
        processing_class=tok, peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    # One explicit run: the Trainer's MLflow callback logs into it instead of opening and closing its own.
    with mlflow.start_run(run_name=sft.run_name):
        mlflow.log_params({"config_file": args.config, "model_id": cfg["model"]["id"], "dataset_id": d["id"]})
        mlflow.log_artifact(args.config)
        result = trainer.train()
        metrics = {"train_loss": result.training_loss, "train_runtime_s": result.metrics["train_runtime"]}
        if len(eval_ds):
            metrics.update(trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix="final_eval"))
        trainer.model.save_pretrained(adapter_dir)
        tok.save_pretrained(adapter_dir)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    print(f"adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
