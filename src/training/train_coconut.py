"""E3: COCONUT (Continuous Chain-of-Thought) training.

Implements the staged curriculum from Hao et al. 2024:
  - Stage 0: standard CoT (identical to E2)
  - Stage k>0: first k reasoning sentences replaced by k latent tokens
  - Stage C: all reasoning latent, only <answer> token supervised

LoRA + COCONUT ordering (enforced here):
  1. Load base model
  2. Add <bot>/<eot> tokens + resize embeddings  ← BEFORE LoRA
  3. Attach LoRA to base model
  4. Wrap in CoconutWrapper
  5. Run curriculum training

Entry points:
  - Called from notebooks/04_train_pilot.ipynb (Colab)
  - Called from scripts/train_coconut.sh (A100)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score

from training.coconut import (
    BOT_TOKEN,
    CoconutCurriculumDataset,
    CoconutWrapper,
    add_coconut_tokens,
)
from training.lora_utils import (
    attach_lora,
    load_base_model,
    save_adapter_with_tokenizer,
)


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
        base.update(cfg.get("coconut", {}))
    if hparams:
        base.update(hparams)
    return base


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _evaluate_f1(
    model: CoconutWrapper,
    val_rows: list[dict],
    tokenizer,
    stage: int,
    num_stages: int,
    bot_token_id: int,
    max_seq_length: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Evaluate weighted F1 on validation set for the current curriculum stage."""
    val_ds = CoconutCurriculumDataset(
        val_rows, tokenizer, stage=stage, num_stages=num_stages,
        bot_token_id=bot_token_id, max_seq_length=max_seq_length,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    high_risk_id = tokenizer.encode("HIGH_RISK", add_special_tokens=False)[0]
    low_risk_id  = tokenizer.encode("LOW_RISK",  add_special_tokens=False)[0]

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels        = batch["labels"].to(device)

            for label_row in labels:
                valid = label_row[label_row != -100]
                if len(valid) == 0:
                    y_true.append(-1)
                    continue
                decoded_target = tokenizer.decode(valid, skip_special_tokens=True).strip() 
                if "HIGH_RISK" in decoded_target:
                    y_true.append(1)
                elif "LOW_RISK" in decoded_target:
                    y_true.append(0)
                else:
                    y_true.append(-1)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.convert_tokens_to_ids("<eot>"),
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = outputs[:, input_ids.shape[1]:]
            if len(y_pred) == 0:
                print("DEBUG raw token ids:", new_tokens[0][:20].tolist())
                token_row = new_tokens[0]
                valid_tokens = token_row[token_row != tokenizer.pad_token_id]
                print("DEBUG decoded:", repr(tokenizer.decode(valid_tokens, skip_special_tokens=False)[:300]))
            for i in range(new_tokens.shape[0]):
                token_row = new_tokens[i]
                valid_tokens = token_row[token_row != tokenizer.pad_token_id]
                text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                y_pred.append(1 if "HIGH_RISK" in text else (0 if "LOW_RISK" in text else -1))

    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != -1]
    if not pairs:
        return 0.0
    yt, yp = zip(*pairs)
    return f1_score(list(yt), list(yp), average="weighted", zero_division=0)


def train_coconut(
    dataset_dir: str | Path,
    output_dir: str | Path,
    hparams: dict | None = None,
    config_path: str | None = None,
) -> Path:
    """Train the COCONUT model (E3).

    Args:
        dataset_dir: Path to cot/ split directory (same data as E2).
        output_dir: Root checkpoint directory (e.g. checkpoints/coconut).
        hparams: Inline hyperparameter dict. Overrides config_path values.
        config_path: Optional path to configs/training/llama_1b_coconut.yaml.

    Returns:
        Path to the best checkpoint directory.
    """
    cfg = _resolve_hparams(hparams, config_path)

    model_name    = cfg.get("model_name_or_path", "meta-llama/Llama-3.2-1B")
    load_in_4bit  = cfg.get("load_in_4bit", True)
    fp16          = cfg.get("fp16", True)
    max_seq_len   = cfg.get("max_seq_length", 512)
    batch_size    = cfg.get("per_device_train_batch_size", 4)
    grad_accum    = cfg.get("gradient_accumulation_steps", 8)
    lr            = cfg.get("learning_rate", 1e-4)
    warmup_ratio  = cfg.get("warmup_ratio", 0.05)
    lora_r        = cfg.get("lora_r", 16)
    lora_alpha    = cfg.get("lora_alpha", 32)
    lora_dropout  = cfg.get("lora_dropout", 0.05)
    target_mods   = cfg.get("target_modules", ["q_proj", "v_proj"])
    num_stages    = cfg.get("num_coconut_stages", 3)
    epochs_per_stage = cfg.get("epochs_per_stage", 1)

    dataset_dir = Path(dataset_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.float16 if fp16 else torch.bfloat16

    print("=" * 60)
    print("E3: COCONUT (Latent Reasoning)")
    print(f"  dataset_dir  : {dataset_dir}")
    print(f"  output_dir   : {output_dir}")
    print(f"  model        : {model_name}")
    print(f"  num_stages   : {num_stages}")
    print(f"  epochs/stage : {epochs_per_stage}")
    print("=" * 60)

    # ── Step 1: Load base model ───────────────────────────────────────────
    model, tokenizer = load_base_model(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        fp16=fp16,
    )

    # ── Step 2: Add special tokens + resize BEFORE LoRA ──────────────────
    bot_token_id, eot_token_id = add_coconut_tokens(tokenizer, model)

    # ── Step 3: Attach LoRA to base model ────────────────────────────────
    model = attach_lora(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_mods,
    )

    # ── Step 4: Wrap in COCONUT ───────────────────────────────────────────
    coconut_model = CoconutWrapper(model, bot_token_id=bot_token_id, eot_token_id=eot_token_id)
    from transformers import GenerationConfig
    coconut_model.model.generation_config = GenerationConfig(
      bos_token_id=128000,
      eos_token_id=eot_token_id,
      pad_token_id=eot_token_id,
      do_sample=False,
    )
    # ── Load data ─────────────────────────────────────────────────────────
    train_rows = _load_jsonl(dataset_dir / "train.jsonl")
    val_rows   = _load_jsonl(dataset_dir / "validation.jsonl")

    # ── Step 5: Curriculum training ───────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    best_f1 = -1.0
    best_ckpt_dir: Path | None = None
    history = []

    optimizer = AdamW(
        [p for p in coconut_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    total_batches_per_stage = (
        (len(train_rows) // batch_size) // grad_accum
    ) * epochs_per_stage
    total_steps = total_batches_per_stage * (num_stages + 1)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for stage in range(num_stages + 1):
        print(f"\n{'─' * 50}")
        print(f"Curriculum Stage {stage}/{num_stages}")
        if stage == 0:
            print("  (Full CoT — identical to E2)")
        elif stage == num_stages:
            print(f"  (Fully latent — all {num_stages} reasoning steps replaced)")
        else:
            print(f"  (First {stage} reasoning step(s) replaced by <bot>)")
        print(f"{'─' * 50}")

        train_ds = CoconutCurriculumDataset(
            train_rows, tokenizer,
            stage=stage, num_stages=num_stages,
            bot_token_id=bot_token_id,
            max_seq_length=max_seq_len,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=0
        )

        for epoch in range(1, epochs_per_stage + 1):
            coconut_model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader, start=1):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=fp16, dtype=compute_dtype):
                    outputs = coconut_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / grad_accum

                scaler.scale(loss).backward()
                total_loss += loss.item() * grad_accum

                if step % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in coconut_model.parameters() if p.requires_grad], 1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)

            val_f1 = _evaluate_f1(
                coconut_model, val_rows, tokenizer,
                stage=stage, num_stages=num_stages,
                bot_token_id=bot_token_id,
                max_seq_length=max_seq_len,
                batch_size=batch_size,
                device=device,
            )

            print(
                f"  [Stage {stage} | Epoch {epoch}] "
                f"train_loss={avg_loss:.4f} | val_weighted_f1={val_f1:.4f}"
            )

            ckpt_dir = out / f"stage_{stage}_epoch_{epoch}"
            save_adapter_with_tokenizer(coconut_model, tokenizer, ckpt_dir)
            history.append({
                "stage": stage, "epoch": epoch,
                "train_loss": avg_loss, "val_f1": val_f1,
            })

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_ckpt_dir = ckpt_dir
                print(f"  ↑ New best (val F1={best_f1:.4f}) → {best_ckpt_dir}")

    # Save training history
    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Copy best to best_adapter/
    best_adapter_dir = out / "best_adapter"
    if best_adapter_dir.exists():
        shutil.rmtree(best_adapter_dir)
    shutil.copytree(best_ckpt_dir, best_adapter_dir)
    print(f"\nBest adapter → {best_adapter_dir} (val F1={best_f1:.4f})")

    return best_adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: COCONUT training")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="checkpoints/coconut")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    parser.add_argument("--epochs-per-stage", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    hparams = {}
    if args.num_stages:
        hparams["num_coconut_stages"] = args.num_stages
    if args.epochs_per_stage:
        hparams["epochs_per_stage"] = args.epochs_per_stage
    if args.batch_size:
        hparams["per_device_train_batch_size"] = args.batch_size
    if args.lr:
        hparams["learning_rate"] = args.lr

    train_coconut(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        hparams=hparams or None,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()