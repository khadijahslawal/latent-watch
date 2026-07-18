"""Shared training loop used by E1, E2, and E3.

Responsibilities:
- Tokenize (input, target) pairs with loss masking on prompt tokens
- Run the training loop with gradient accumulation
- Evaluate on validation set after every epoch (weighted F1)
- Save per-epoch checkpoints; track and return the best
- Load best checkpoint before returning

The caller (train_answer_only / train_cot / train_coconut) is responsible for:
- Loading and wrapping the model
- Building the dataset / dataloader
- Providing the formatter
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score

from training.lora_utils import save_adapter, save_adapter_with_tokenizer


# ── Dataset ───────────────────────────────────────────────────────────────

class PromptRiskDataset(Dataset):
    """Tokenizes (input, target) pairs with loss masking.

    Prompt tokens → labels = -100 (ignored by cross-entropy loss).
    Target tokens → labels = token ids (loss computed here).
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: AutoTokenizer,
        formatter: Callable,
        max_seq_length: int = 512,
        formatter_kwargs: dict | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []

        formatter_kwargs = formatter_kwargs or {}

        for row in rows:
            input_text, target_text = formatter(row, **formatter_kwargs)
            self.samples.append(
                self._tokenize(input_text, target_text)
            )

    def _tokenize(self, input_text: str, target_text: str) -> dict[str, torch.Tensor]:
        tok = self.tokenizer

        input_ids = tok(
            input_text,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        target_ids = tok(
            target_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Concatenate: full sequence = input + target
        full_ids = torch.cat([input_ids, target_ids], dim=0)

        # Loss mask: -100 for prompt tokens, actual ids for target tokens
        labels = torch.cat([
            torch.full((len(input_ids),), -100, dtype=torch.long),
            target_ids,
        ], dim=0)

        # Truncate to max_seq_length
        full_ids = full_ids[: self.max_seq_length]
        labels = labels[: self.max_seq_length]

        # Pad to max_seq_length
        pad_len = self.max_seq_length - len(full_ids)
        attention_mask = torch.ones(len(full_ids), dtype=torch.long)

        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id
            full_ids = torch.cat([
                torch.full((pad_len,), pad_id, dtype=torch.long),
                full_ids,
            ])
            labels = torch.cat([
                torch.full((pad_len,), -100, dtype=torch.long),
                labels,
            ])
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                attention_mask,
            ])

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── Evaluation helper ─────────────────────────────────────────────────────

def evaluate_weighted_f1(
    model,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    label_map: dict[str, int] | None = None,
) -> float:
    """Run greedy decoding on validation set; return weighted F1.

    Args:
        model: The model to evaluate.
        dataloader: Validation dataloader.
        tokenizer: For decoding outputs.
        device: Torch device.
        label_map: Maps label string → int id. Defaults to HIGH_RISK=1, LOW_RISK=0.

    Returns:
        Weighted F1 score.
    """
    if label_map is None:
        label_map = {"HIGH_RISK": 1, "LOW_RISK": 0}

    high_risk_id = tokenizer.encode("HIGH_RISK", add_special_tokens=False)[0]
    low_risk_id = tokenizer.encode("LOW_RISK", add_special_tokens=False)[0]

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Extract true labels from the last non-masked label token
            for label_row in labels:
                valid = label_row[label_row != -100]
                if len(valid) == 0:
                    y_true.append(-1)
                    continue
                # The last token of target is the label (HIGH_RISK/LOW_RISK)
                last_token = valid[-1].item()
                if last_token == high_risk_id:
                    y_true.append(1)
                elif last_token == low_risk_id:
                    y_true.append(0)
                else:
                    y_true.append(-1)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
                do_sample=False,
            )
            new_tokens = outputs[:, input_ids.shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for text in decoded:
                text = text.strip()
                if "HIGH_RISK" in text:
                    y_pred.append(1)
                elif "LOW_RISK" in text:
                    y_pred.append(0)
                else:
                    y_pred.append(-1)

    # Filter out rows where true label couldn't be extracted
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != -1]
    if not pairs:
        return 0.0
    yt, yp = zip(*pairs)
    return f1_score(list(yt), list(yp), average="weighted", zero_division=0)


# ── Main training loop ────────────────────────────────────────────────────

def train(
    model,
    tokenizer: AutoTokenizer,
    train_dataset: PromptRiskDataset,
    val_dataset: PromptRiskDataset,
    output_dir: str | Path,
    num_epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.05,
    fp16: bool = True,
    save_tokenizer: bool = False,
    experiment: str = "experiment",
) -> Path:
    """Run the training loop and return the path to the best checkpoint.

    Args:
        model: PEFT-wrapped model (or COCONUT wrapper containing one).
        tokenizer: Tokenizer.
        train_dataset: Training PromptRiskDataset.
        val_dataset: Validation PromptRiskDataset.
        output_dir: Root directory for checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Peak learning rate.
        warmup_ratio: Fraction of steps for LR warmup.
        fp16: Use fp16 mixed precision (T4). Set False for A100 bf16.
        save_tokenizer: Also save tokenizer with checkpoint (needed for E3).
        experiment: Name string used in logging.

    Returns:
        Path to the best checkpoint directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    best_f1 = -1.0
    best_ckpt_dir: Path | None = None
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * gradient_accumulation_steps

            if step % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_f1 = evaluate_weighted_f1(model, val_loader, tokenizer, device)
        print(
            f"[{experiment}] Epoch {epoch}/{num_epochs} | "
            f"train_loss={avg_loss:.4f} | val_weighted_f1={val_f1:.4f}"
        )

        # Save epoch checkpoint
        epoch_dir = out / f"epoch_{epoch}"
        if save_tokenizer:
            save_adapter_with_tokenizer(model, tokenizer, epoch_dir)
        else:
            save_adapter(model, epoch_dir)

        history.append({"epoch": epoch, "train_loss": avg_loss, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_ckpt_dir = epoch_dir
            print(f"  ↑ New best (val weighted F1={best_f1:.4f}) → {best_ckpt_dir}")

    # Save training history
    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest checkpoint: {best_ckpt_dir} (val F1={best_f1:.4f})")
    return best_ckpt_dir
