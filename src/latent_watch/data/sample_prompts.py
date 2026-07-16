"""Step 7: Downsample each split to pilot size with balanced labels.

Sampling logic (training set only — val and test use all available prompts
up to their target sizes):
  1. Within HIGH_RISK, apply per-category cap to improve category coverage.
  2. Downsample / upsample HIGH_RISK to match LOW_RISK count → ~50/50 balance.

Val and test are sampled to their target sizes with stratification only
(no per-category cap needed at those scales).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import pandas as pd


def _cap_high_risk_by_category(
    df: pd.DataFrame,
    per_category_cap: int,
    random_seed: int,
) -> pd.DataFrame:
    """Cap HIGH_RISK rows so no single harm category contributes more than per_category_cap.

    Rows with multiple harm categories can satisfy multiple caps simultaneously.
    We use a greedy inclusion approach: shuffle, then include a row if any of its
    categories still has remaining capacity.

    Args:
        df: HIGH_RISK subset of the train split.
        per_category_cap: Max rows per harm category.
        random_seed: For shuffling.

    Returns:
        Capped DataFrame (may be smaller than input).
    """
    rng = random.Random(random_seed)
    rows = df.to_dict("records")
    rng.shuffle(rows)

    category_counts: dict[str, int] = {}
    selected = []

    for row in rows:
        cats = row["harm_categories"]
        if not cats:
            # No category info — include freely (edge case)
            selected.append(row)
            continue
        if any(category_counts.get(c, 0) < per_category_cap for c in cats):
            selected.append(row)
            for c in cats:
                category_counts[c] = category_counts.get(c, 0) + 1

    return pd.DataFrame(selected)


def sample_split(
    df: pd.DataFrame,
    split: Literal["train", "validation", "test"],
    target_size: int,
    per_category_cap: int = 400,
    high_risk_fraction: float = 0.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sample a single split to the target size with balanced labels.

    Args:
        df: Full split DataFrame (output of build_splits for this split key).
        split: Which split this is (affects sampling strategy).
        target_size: Desired number of examples in output.
        per_category_cap: (train only) cap per harm category within HIGH_RISK.
        high_risk_fraction: Target fraction of HIGH_RISK examples.
        random_seed: Random seed.

    Returns:
        Sampled DataFrame.
    """
    high_risk_df = df[df["label"] == "HIGH_RISK"].copy()
    low_risk_df = df[df["label"] == "LOW_RISK"].copy()

    n_high = int(target_size * high_risk_fraction)
    n_low = target_size - n_high

    if split == "train":
        # Per-category cap before global balance
        high_risk_df = _cap_high_risk_by_category(high_risk_df, per_category_cap, random_seed)

    # Sample with replacement only if we need more than available
    high_sample = high_risk_df.sample(
        n=min(n_high, len(high_risk_df)),
        replace=n_high > len(high_risk_df),
        random_state=random_seed,
    )
    low_sample = low_risk_df.sample(
        n=min(n_low, len(low_risk_df)),
        replace=n_low > len(low_risk_df),
        random_state=random_seed,
    )

    result = pd.concat([high_sample, low_sample], ignore_index=True).sample(
        frac=1, random_state=random_seed
    )

    print(
        f"[{split}] sampled {len(result):,} / {target_size} "
        f"(HIGH_RISK={len(high_sample):,}, LOW_RISK={len(low_sample):,})"
    )
    return result


def sample_all_splits(
    splits: dict[str, pd.DataFrame],
    train_size: int = 5000,
    val_size: int = 500,
    test_size: int = 1000,
    per_category_cap: int = 400,
    high_risk_fraction: float = 0.5,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Sample all three splits to pilot sizes.

    Args:
        splits: Dict {'train': df, 'validation': df, 'test': df}.
        train_size: Target train examples.
        val_size: Target validation examples.
        test_size: Target test examples.
        per_category_cap: Per-category cap for HIGH_RISK in train.
        high_risk_fraction: Target HIGH_RISK balance.
        random_seed: Random seed.

    Returns:
        Dict with same keys, values are sampled DataFrames.
    """
    sizes = {"train": train_size, "validation": val_size, "test": test_size}
    return {
        split: sample_split(
            df=splits[split],
            split=split,  # type: ignore[arg-type]
            target_size=sizes[split],
            per_category_cap=per_category_cap,
            high_risk_fraction=high_risk_fraction,
            random_seed=random_seed,
        )
        for split in ("train", "validation", "test")
    }


if __name__ == "__main__":
    from latent_watch.data.build_splits import build_splits

    agg = pd.read_parquet("data/interim/prompt_aggregates.parquet")
    splits = build_splits(agg)
    pilot = sample_all_splits(splits)
    for name, df in pilot.items():
        print(f"{name}: {len(df)} | {df['label'].value_counts().to_dict()}")
