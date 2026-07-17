"""Step 9: Validate generated reasoning traces and reject/flag bad examples.

Rejection criteria (from data strategy):
  - Rationale contradicts the label
  - Teacher changed the label
  - Reasoning mentions paired responses or dataset annotations
  - Rationale is empty or excessively long (>120 words across all three steps)
  - LOW_RISK rationale claims the prompt is harmful
  - HIGH_RISK rationale relies only on vague words like "dangerous"

Output: filtered DataFrame + rejection log.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


# Phrases that suggest the model referenced paired responses or dataset internals
DATASET_REFERENCE_PATTERNS = [
    r"\bone response\b",
    r"\bpaired response\b",
    r"\bobserved response\b",
    r"\bdataset\b",
    r"\bannotation\b",
    r"\blabeled (un)?safe\b",
    r"\bwas rated\b",
    r"\bwas marked\b",
]

# Vague-only words that should not be the sole justification for HIGH_RISK
VAGUE_HIGH_RISK_WORDS = {"dangerous", "harmful", "bad", "risky", "unsafe", "problematic"}

# Words that indicate a LOW_RISK rationale is claiming harm
LOW_RISK_HARM_CLAIMS = [
    r"\bthis prompt is (high.risk|dangerous|harmful)\b",
    r"\bunsafe\b",
    r"\bhigh.risk\b",
]

MAX_TOTAL_WORDS = 120


def _count_words(steps: list[str]) -> int:
    return sum(len(s.split()) for s in steps)


def _check_empty(steps: list[str]) -> bool:
    return all(s.strip() == "" for s in steps)


def _check_dataset_reference(steps: list[str]) -> bool:
    combined = " ".join(steps).lower()
    return any(re.search(p, combined) for p in DATASET_REFERENCE_PATTERNS)


def _check_low_risk_harm_claim(steps: list[str]) -> bool:
    combined = " ".join(steps).lower()
    return any(re.search(p, combined) for p in LOW_RISK_HARM_CLAIMS)


def _check_vague_only_high_risk(steps: list[str]) -> bool:
    """Return True if HIGH_RISK reasoning contains ONLY vague words and no specifics."""
    combined = " ".join(steps).lower()
    words = set(re.findall(r"\b\w+\b", combined))
    meaningful = words - VAGUE_HIGH_RISK_WORDS - {"the", "a", "an", "is", "this", "it", "of", "and", "or", "to", "in"}
    return len(meaningful) < 5  # fewer than 5 non-vague words = vague-only


def validate_row(row: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single example's reasoning trace.

    Args:
        row: Dict with keys: reasoning (list[str]), label (str).

    Returns:
        (is_valid, rejection_reason). rejection_reason is '' if valid.
    """
    steps: list[str] = row.get("reasoning", [])
    label: str = row.get("label", "")

    if not steps or _check_empty(steps):
        return False, "empty_rationale"

    if len(steps) != 3:
        return False, "wrong_step_count"

    if _count_words(steps) > MAX_TOTAL_WORDS:
        return False, "too_long"

    if _check_dataset_reference(steps):
        return False, "references_dataset_or_responses"

    if label == "LOW_RISK" and _check_low_risk_harm_claim(steps):
        return False, "low_risk_claims_harm"

    if label == "HIGH_RISK" and _check_vague_only_high_risk(steps):
        return False, "high_risk_too_vague"

    return True, ""


def validate_rationales(
    df: pd.DataFrame,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate all reasoning traces in df.

    Args:
        df: DataFrame with reasoning and label columns.
        verbose: If True, print each rejection reason.

    Returns:
        (valid_df, rejected_df) — both are subsets of the input DataFrame
        with an added 'rejection_reason' column in rejected_df.
    """
    valid_rows = []
    rejected_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating rationales"):
        is_valid, reason = validate_row(row.to_dict())
        if is_valid:
            valid_rows.append(row)
        else:
            row = row.copy()
            row["rejection_reason"] = reason
            rejected_rows.append(row)
            if verbose:
                print(f"[REJECT {reason}] {row.get('prompt', '')[:60]!r}")

    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    rejected_df = pd.DataFrame(rejected_rows).reset_index(drop=True)

    total = len(df)
    print(
        f"Validation complete: {len(valid_df):,}/{total:,} kept "
        f"({len(rejected_df):,} rejected)"
    )
    if len(rejected_df) > 0:
        print("Rejection breakdown:")
        print(rejected_df["rejection_reason"].value_counts().to_string())

    return valid_df, rejected_df


def save_rejection_log(rejected_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Rejection log → {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with reasoning traces")
    parser.add_argument("--output", required=True, help="Output JSONL (valid only)")
    parser.add_argument("--reject-log", default=None, help="Path to write rejection log")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    valid_df, rejected_df = validate_rationales(df, verbose=args.verbose)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_json(out, orient="records", lines=True, force_ascii=False)
    print(f"Valid examples → {out}")

    if args.reject_log:
        save_rejection_log(rejected_df, args.reject_log)
