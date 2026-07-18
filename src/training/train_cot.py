"""E2: Chain-of-Thought training.

Trains a LoRA adapter on Llama-3.2-1B to produce explicit reasoning
in <reasoning> tags followed by <answer>LABEL</answer>.

Target text includes full reasoning trace + answer. Loss is computed
on both reasoning tokens and the answer token.

Entry points:
  - Called from notebooks/04_train_pilot.ipynb (Colab)
  - Called from scripts/train_cot.sh (A100)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from latent_watch.training.base_trainer import PromptRiskDataset, train
from latent_watch.training.formatters import format_cot
from latent_watch.training.lora_utils import attach_lora, load_base_model


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_hparams(hparams: dict | None, config_path: str | None) -> dict:
    base = {}
    if config_path is not None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        base.update(cfg.get("training", {}))
        base.update(cfg.get("model", {}))
        base.update(cfg.get("lora", {}))
    if hparams:
        base.update(hparams)
    return base


def train_cot(
    dataset_dir: str | Path,
    output_dir: str | Path,
    hparams: dict | None = None,
    config_path: str | None = None,
) -> Path:
    """Train the CoT model (E2).

    Args:
        dataset_dir: Path to cot/ split directory containing
                     train.jsonl, validation.jsonl, test.jsonl.
        output_dir: Root checkpoint directory (e.g. checkpoints/cot).
        hparams: Inline hyperparameter dict. Overrides config_path values.
        config_path: Optional path to configs/training/llama_1b_cot.yaml.

    Returns:
        Path to the best checkpoint directory.
    """
    cfg = _resolve_hparams(hparams, config_path)

    model_name   = cfg.get("model_name_or_path", "meta-llama/Llama-3.2-1B")
    load_in_4bit = cfg.get("load_in_4bit", True)
    fp16         = cfg.get("fp16", True)
    max_seq_len  = cfg.get("max_seq_length", 512)
    num_epochs   = cfg.get("num_train_epochs", 3)
    batch_size   = cfg.get("per_device_train_batch_size", 8)
    grad_accum   = cfg.get("gradient_accumulation_steps", 4)
    lr           = cfg.get("learning_rate", 2e-4)
    warmup_ratio = cfg.get("warmup_ratio", 0.05)
    lora_r       = cfg.get("lora_r", 16)
    lora_alpha   = cfg.get("lora_alpha", 32)
    lora_dropout = cfg.get("lora_dropout", 0.05)
    target_mods  = cfg.get("target_modules", ["q_proj", "v_proj"])

    dataset_dir = Path(dataset_dir)

    print("=" * 60)
    print("E2: Chain-of-Thought")
    print(f"  dataset_dir : {dataset_dir}")
    print(f"  output_dir  : {output_dir}")
    print(f"  model       : {model_name}")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_base_model(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        fp16=fp16,
    )

    # Attach LoRA
    model = attach_lora(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_mods,
    )

    # Build datasets
    train_rows = _load_jsonl(dataset_dir / "train.jsonl")
    val_rows   = _load_jsonl(dataset_dir / "validation.jsonl")

    train_ds = PromptRiskDataset(
        train_rows, tokenizer, format_cot, max_seq_length=max_seq_len
    )
    val_ds = PromptRiskDataset(
        val_rows, tokenizer, format_cot, max_seq_length=max_seq_len
    )

    print(f"Train examples: {len(train_ds):,} | Val examples: {len(val_ds):,}")

    # Train
    best_ckpt = train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=Path(output_dir),
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        save_tokenizer=False,
        experiment="cot",
    )

    # Copy best checkpoint to best_adapter/
    best_adapter_dir = Path(output_dir) / "best_adapter"
    if best_adapter_dir.exists():
        shutil.rmtree(best_adapter_dir)
    shutil.copytree(best_ckpt, best_adapter_dir)
    print(f"Best adapter → {best_adapter_dir}")

    return best_adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: CoT training")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="checkpoints/cot")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    hparams = {}
    if args.epochs:
        hparams["num_train_epochs"] = args.epochs
    if args.batch_size:
        hparams["per_device_train_batch_size"] = args.batch_size
    if args.lr:
        hparams["learning_rate"] = args.lr

    train_cot(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        hparams=hparams or None,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
