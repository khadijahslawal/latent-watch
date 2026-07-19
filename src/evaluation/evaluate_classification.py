"""Shared evaluation pipeline for E1, E2, and E3.

Runs a trained adapter on the test set, extracts predictions, and produces:
  - Console report (accuracy, weighted P/R/F1, HIGH_RISK recall, confusion matrix)
  - results/{experiment}.csv with one row per example

Weighted F1 is the primary metric (class-imbalanced dataset).
HIGH_RISK recall is highlighted separately — missing unsafe prompts is the costly error.

Usage:
  python evaluate_classification.py \
      --experiment answer_only \
      --adapter-dir checkpoints/answer_only/best_adapter \
      --dataset-dir data/processed/beavertails_risk_v1/answer_only \
      --output-file results/answer_only.csv

Or call evaluate() directly from the notebook.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer

from latent_watch.training.formatters import USER_PROMPT_TEMPLATE
from latent_watch.training.lora_utils import load_adapter, load_base_model


# ── Inference helpers ─────────────────────────────────────────────────────

def _build_input(prompt: str) -> str:
    return USER_PROMPT_TEMPLATE.format(prompt=prompt)


def _extract_label_answer_only(raw: str) -> tuple[str | None, bool]:
    """Extract label from answer-only output (raw text = label directly)."""
    text = raw.strip()
    if "HIGH_RISK" in text:
        return "HIGH_RISK", True
    if "LOW_RISK" in text:
        return "LOW_RISK", True
    return None, False


def _extract_label_cot(raw: str) -> tuple[str | None, bool, str | None]:
    """Extract label and reasoning from CoT output.

    Returns:
        (label, valid_output, generated_reasoning)
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    if answer_match:
        label_text = answer_match.group(1).strip()
        if label_text in ("HIGH_RISK", "LOW_RISK"):
            return label_text, True, reasoning
    return None, False, reasoning


def _extract_label_latent(raw: str) -> tuple[str | None, bool]:
    """Extract label from COCONUT output (no reasoning trace expected)."""
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if answer_match:
        label_text = answer_match.group(1).strip()
        if label_text in ("HIGH_RISK", "LOW_RISK"):
            return label_text, True
    # Fallback: scan raw text
    if "HIGH_RISK" in raw:
        return "HIGH_RISK", False
    if "LOW_RISK" in raw:
        return "LOW_RISK", False
    return None, False


def _get_label_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    high_risk_token_id: int,
    low_risk_token_id: int,
    device: torch.device,
) -> tuple[float, float]:
    """Get softmax scores for HIGH_RISK and LOW_RISK at the first generated position.

    Returns:
        (high_risk_score, low_risk_score) — probabilities summing to ~1.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        logits = outputs.logits[:, -1, :]  # logits at last input position
        probs = torch.softmax(logits, dim=-1)
        high = probs[0, high_risk_token_id].item()
        low  = probs[0, low_risk_token_id].item()
        # Renormalize to just these two classes
        total = high + low + 1e-12
        return high / total, low / total


# ── Main evaluation function ──────────────────────────────────────────────

def evaluate(
    experiment: str,
    adapter_dir: str | Path,
    dataset_dir: str | Path,
    output_file: str | Path,
    model_name: str = "meta-llama/Llama-3.2-1B",
    load_in_4bit: bool = True,
    fp16: bool = True,
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> pd.DataFrame:
    """Run evaluation on the test set and write results CSV.

    Args:
        experiment: One of 'answer_only', 'cot', 'latent'.
        adapter_dir: Path to best_adapter/ directory.
        dataset_dir: Path to the experiment's split directory
                     (answer_only/ or cot/ — test.jsonl is read from here).
        output_file: Path to write results CSV.
        model_name: Base model name (must match training).
        load_in_4bit: Load base model in 4-bit.
        fp16: Use fp16.
        batch_size: Inference batch size.
        max_new_tokens: Max tokens to generate.

    Returns:
        Results DataFrame.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(dataset_dir)
    adapter_dir = Path(adapter_dir)

    # Load tokenizer — use adapter dir in case vocab was extended (E3)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model + adapter
    base_model, _ = load_base_model(model_name, load_in_4bit=load_in_4bit, fp16=fp16)
    if len(tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))
    model = load_adapter(base_model, adapter_dir)
    model.eval()

    # Token ids for confidence scoring
    high_risk_token_id = tokenizer.encode("HIGH_RISK", add_special_tokens=False)[0]
    low_risk_token_id  = tokenizer.encode("LOW_RISK",  add_special_tokens=False)[0]

    # Load test set
    test_path = dataset_dir / "test.jsonl"
    rows: list[dict[str, Any]] = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Evaluating {len(rows):,} test examples [{experiment}]...")

    records = []
    for row in rows:
        input_text = _build_input(row["prompt"])
        enc = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Confidence scores at next-token distribution
        high_score, low_score = _get_label_scores(
            model, input_ids, attention_mask,
            high_risk_token_id, low_risk_token_id, device,
        )

        # Generate output
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # REPLACE WITH
        new_tokens = gen_ids[:, input_ids.shape[1]:]
        row = new_tokens[0]
        valid_tokens = row[row != tokenizer.pad_token_id]
        raw_output = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()

        # Extract prediction
        generated_reasoning = None
        if experiment == "answer_only":
            pred_label, valid = _extract_label_answer_only(raw_output)
        elif experiment == "cot":
            pred_label, valid, generated_reasoning = _extract_label_cot(raw_output)
        else:  # latent / coconut
            pred_label, valid = _extract_label_latent(raw_output)

        record = {
            "example_id":       row.get("id", ""),
            "prompt":           row["prompt"],
            "true_label":       row["label"],
            "predicted_label":  pred_label if pred_label else "INVALID",
            "HIGH_RISK_score":  round(high_score, 4),
            "LOW_RISK_score":   round(low_score, 4),
            "raw_output":       raw_output,
            "valid_output":     valid,
            "model":            model_name,
            "experiment":       experiment,
        }
        if experiment == "cot":
            record["generated_reasoning"] = generated_reasoning or ""

        records.append(record)

    results_df = pd.DataFrame(records)

    # Save CSV
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"Results → {out_path}")

    # ── Print report ──────────────────────────────────────────────────────
    valid_df = results_df[results_df["valid_output"]]
    invalid_count = (~results_df["valid_output"]).sum()
    if invalid_count > 0:
        print(f"WARNING: {invalid_count} examples had invalid/unparseable output.")

    y_true = valid_df["true_label"].tolist()
    y_pred = valid_df["predicted_label"].tolist()

    print("\n" + "=" * 60)
    print(f"Test Evaluation — {experiment}")
    print("=" * 60)
    print(f"Valid predictions: {len(valid_df):,} / {len(results_df):,}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    print(f"Weighted Precision : {p:.4f}")
    print(f"Weighted Recall    : {r:.4f}")
    print(f"Weighted F1        : {f:.4f}  ← PRIMARY METRIC")

    # HIGH_RISK recall specifically
    labels = ["HIGH_RISK", "LOW_RISK"]
    p_per, r_per, f_per, s_per = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    print(f"\nHIGH_RISK Recall   : {r_per[0]:.4f}  ← KEY (costly to miss unsafe prompts)")
    print(f"LOW_RISK  Recall   : {r_per[1]:.4f}")

    print("\nFull Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(f"              {'HIGH_RISK':>12} {'LOW_RISK':>12}")
    for i, label in enumerate(labels):
        print(f"  {label:>12}  {cm[i][0]:>12}  {cm[i][1]:>12}")

    return results_df


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained adapter on the test set")
    parser.add_argument(
        "--experiment", required=True,
        choices=["answer_only", "cot", "latent"],
        help="Which experiment to evaluate",
    )
    parser.add_argument("--adapter-dir", required=True, help="Path to best_adapter/")
    parser.add_argument("--dataset-dir", required=True, help="Path to split dir (answer_only/ or cot/)")
    parser.add_argument("--output-file", required=True, help="Output CSV path")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    evaluate(
        experiment=args.experiment,
        adapter_dir=args.adapter_dir,
        dataset_dir=args.dataset_dir,
        output_file=args.output_file,
        model_name=args.model_name,
        load_in_4bit=not args.no_4bit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
