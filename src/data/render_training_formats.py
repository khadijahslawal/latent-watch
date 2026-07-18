"""Steps 10–11: Export canonical JSONL and render training-specific formats.

Step 10: Write canonical/{train,validation,test}.jsonl
         Each row has the full schema including reasoning, harm_categories, stats.

Step 11: From canonical, render:
  - answer_only/{train,validation,test}.jsonl  — prompt + label only
  - cot/{train,validation,test}.jsonl          — prompt + reasoning + answer
  COCONUT reads from cot/ (no separate rendering needed).

Also writes:
  - manifest.json   — row counts and file hashes
  - label_schema.json
  - dataset_card.md
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


# ── Prompt templates ──────────────────────────────────────────────────────────

ANSWER_ONLY_USER_TMPL = """\
Assess the observed safety risk of the following prompt.

Prompt:
{prompt}

A prompt is HIGH_RISK if at least one observed response to it was unsafe.
A prompt is LOW_RISK if all observed responses were safe.\
"""

ANSWER_ONLY_ASSISTANT_TMPL = "{label}"

COT_USER_TMPL = ANSWER_ONLY_USER_TMPL  # same instruction

COT_ASSISTANT_TMPL = """\
<reasoning>
{reasoning}
</reasoning>
<answer>{label}</answer>\
"""


# ── Row formatters ────────────────────────────────────────────────────────────

def _canonical_row(row: pd.Series, idx: int) -> dict[str, Any]:
    return {
        "id": f"bt_prompt_{idx:06d}",
        "prompt": row["prompt"],
        "label": row["label"],
        "label_id": int(row["label_id"]),
        "reasoning": list(row.get("reasoning", [])),
        "harm_categories": list(row.get("harm_categories", [])),
        "observed_response_count": int(row.get("observed_response_count", 0)),
        "observed_safe_count": int(row.get("observed_safe_count", 0)),
        "observed_unsafe_count": int(row.get("observed_unsafe_count", 0)),
        "unsafe_response_rate": float(row.get("unsafe_response_rate", 0.0)),
        "any_unsafe_response": bool(row.get("any_unsafe_response", False)),
        "all_unsafe_responses": bool(row.get("all_unsafe_responses", False)),
        "source": str(row.get("source", "PKU-Alignment/BeaverTails")),
        "source_split": str(row.get("source_split", "330k_train")),
    }


def _answer_only_row(canonical: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": canonical["id"],
        "messages": [
            {
                "role": "user",
                "content": ANSWER_ONLY_USER_TMPL.format(prompt=canonical["prompt"]),
            },
            {
                "role": "assistant",
                "content": ANSWER_ONLY_ASSISTANT_TMPL.format(label=canonical["label"]),
            },
        ],
        "label": canonical["label"],
        "label_id": canonical["label_id"],
    }


def _cot_row(canonical: dict[str, Any]) -> dict[str, Any]:
    reasoning_text = "\n".join(canonical["reasoning"])
    return {
        "id": canonical["id"],
        "messages": [
            {
                "role": "user",
                "content": COT_USER_TMPL.format(prompt=canonical["prompt"]),
            },
            {
                "role": "assistant",
                "content": COT_ASSISTANT_TMPL.format(
                    reasoning=reasoning_text,
                    label=canonical["label"],
                ),
            },
        ],
        "label": canonical["label"],
        "label_id": canonical["label_id"],
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main rendering function ───────────────────────────────────────────────────

def render_all(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    rationale_model: str = "gpt-4o-mini",
) -> None:
    """Render all training formats from split DataFrames.

    Args:
        splits: Dict {'train': df, 'validation': df, 'test': df} with
                reasoning column populated (from generate/validate rationales).
        output_dir: Root output directory (e.g. data/processed/beavertails_risk_v1).
        rationale_model: Name of model used to generate rationales (for dataset card).
    """
    out = Path(output_dir)
    manifest: dict[str, Any] = {
        "version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rationale_model": rationale_model,
        "splits": {},
    }

    for split_name, df in splits.items():
        canonical_rows = []
        answer_only_rows = []
        cot_rows = []

        for i, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc=f"Rendering {split_name}")
        ):
            canonical = _canonical_row(row, i)
            canonical_rows.append(canonical)
            answer_only_rows.append(_answer_only_row(canonical))
            cot_rows.append(_cot_row(canonical))

        # Write files
        canon_path = out / "canonical" / f"{split_name}.jsonl"
        ao_path = out / "answer_only" / f"{split_name}.jsonl"
        cot_path = out / "cot" / f"{split_name}.jsonl"

        _write_jsonl(canonical_rows, canon_path)
        _write_jsonl(answer_only_rows, ao_path)
        _write_jsonl(cot_rows, cot_path)

        manifest["splits"][split_name] = {
            "n_examples": len(canonical_rows),
            "label_counts": df["label"].value_counts().to_dict(),
            "files": {
                "canonical": {"path": str(canon_path), "sha256": _sha256(canon_path)},
                "answer_only": {"path": str(ao_path), "sha256": _sha256(ao_path)},
                "cot": {"path": str(cot_path), "sha256": _sha256(cot_path)},
            },
        }

    # Write manifest
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest → {manifest_path}")

    # Write label schema
    label_schema = {
        "labels": [
            {"name": "HIGH_RISK", "id": 1, "definition": "At least one observed response was unsafe."},
            {"name": "LOW_RISK", "id": 0, "definition": "All observed responses were safe."},
        ]
    }
    with open(out / "label_schema.json", "w") as f:
        json.dump(label_schema, f, indent=2)

    # Write dataset card
    _write_dataset_card(out, manifest, rationale_model)
    print("All formats rendered.")


def _write_dataset_card(out: Path, manifest: dict, rationale_model: str) -> None:
    total = sum(v["n_examples"] for v in manifest["splits"].values())
    card = f"""# BeaverTails Risk Dataset — v1

## Overview

Binary prompt-risk classification derived from PKU-Alignment/BeaverTails (330k_train split).

**Task:** Given a prompt, predict HIGH_RISK or LOW_RISK.
- **HIGH_RISK**: at least one observed response to the prompt was unsafe.
- **LOW_RISK**: all observed responses were safe.

## Statistics

| Split | Examples |
|-------|----------|
"""
    for split_name, info in manifest["splits"].items():
        card += f"| {split_name} | {info['n_examples']:,} |\n"
    card += f"| **Total** | **{total:,}** |\n"

    card += f"""
## Reasoning Traces

Reasoning traces were **synthetically generated** by `{rationale_model}` (OpenAI),
conditioned on ground-truth labels and harm categories. They do **not** represent
human judgement.

The generator did not see paired responses. Rationales reference only the prompt,
label, and harm category annotations.

## Formats

- `canonical/` — full schema with all metadata fields
- `answer_only/` — formatted for label-only fine-tuning (no reasoning)
- `cot/` — formatted with `<reasoning>` + `<answer>` for CoT and COCONUT training

COCONUT training reads from `cot/` (no separate format needed).

## Provenance

- Source: `PKU-Alignment/BeaverTails`, split `330k_train`
- Created: {manifest['created_at']}
"""
    card_path = out / "dataset_card.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"Dataset card → {card_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Validated train JSONL")
    parser.add_argument("--validation", required=True, help="Validated validation JSONL")
    parser.add_argument("--test", required=True, help="Validated test JSONL")
    parser.add_argument("--output-dir", default="data/processed/beavertails_risk_v1")
    parser.add_argument("--rationale-model", default="gpt-4o-mini")
    args = parser.parse_args()

    splits = {
        "train": pd.read_json(args.train, lines=True),
        "validation": pd.read_json(args.validation, lines=True),
        "test": pd.read_json(args.test, lines=True),
    }
    render_all(splits, args.output_dir, rationale_model=args.rationale_model)
