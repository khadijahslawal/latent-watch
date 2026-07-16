"""Step 6: Split unique-prompt pool into train / validation / test BEFORE sampling.

Stratified split on binary label to prevent leakage.
Assignments are saved to data/interim/split_assignments.parquet so downstream
steps can reproduce the same split without recomputing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_splits(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    stratify_col: str = "label",
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Stratified split of aggregated prompt DataFrame.

    Args:
        df: Output of aggregate_prompts().
        train_frac: Fraction for train set.
        val_frac: Fraction for validation set.
        test_frac: Fraction for test set.
        stratify_col: Column to stratify on.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'train', 'validation', 'test' mapping to DataFrames.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    # First split off test
    test_rel = test_frac
    train_val, test = train_test_split(
        df,
        test_size=test_rel,
        stratify=df[stratify_col],
        random_state=random_seed,
    )

    # Then split val from train_val
    val_rel = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_rel,
        stratify=train_val[stratify_col],
        random_state=random_seed,
    )

    splits = {"train": train, "validation": val, "test": test}

    for name, split_df in splits.items():
        print(f"{name}: {len(split_df):,} prompts | label dist:\n{split_df[stratify_col].value_counts().to_dict()}")

    return splits


def save_split_assignments(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> Path:
    """Persist split assignments (prompt + split name) to parquet.

    Args:
        splits: Dict from build_splits().
        output_dir: Directory for split_assignments.parquet.

    Returns:
        Path to written file.
    """
    records = []
    for split_name, split_df in splits.items():
        tmp = split_df[["prompt", "label"]].copy()
        tmp["split"] = split_name
        records.append(tmp)

    assignments = pd.concat(records, ignore_index=True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "split_assignments.parquet"
    assignments.to_parquet(path, index=False)
    print(f"Saved split assignments → {path}")
    return path


if __name__ == "__main__":
    agg = pd.read_parquet("data/interim/prompt_aggregates.parquet")
    splits = build_splits(agg)
    save_split_assignments(splits, "data/interim")
