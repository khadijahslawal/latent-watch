"""Steps 3–5: Group by unique prompt, apply ANY-unsafe rule, aggregate harm categories.

Step 3: Group by unique normalized prompt; collect all responses and labels.
Step 4: Apply the ANY-unsafe rule → HIGH_RISK if any response was unsafe.
         Preserve per-prompt response statistics.
Step 5: Aggregate harm categories from UNSAFE rows only.
         LOW_RISK prompts get harm_categories = [].

Output: one row per unique prompt (pandas DataFrame and optionally parquet).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset
from tqdm import tqdm


LABEL_HIGH_RISK = "HIGH_RISK"
LABEL_LOW_RISK = "LOW_RISK"
LABEL_ID = {LABEL_HIGH_RISK: 1, LABEL_LOW_RISK: 0}


def aggregate_prompts(
    dataset: Dataset,
    source: str = "PKU-Alignment/BeaverTails",
    source_split: str = "330k_train",
) -> pd.DataFrame:
    """Group normalized rows by prompt and apply the ANY-unsafe labelling rule.

    Args:
        dataset: Normalized dataset (output of normalize_dataset).
        source: Dataset source string for provenance.
        source_split: Source split name for provenance.

    Returns:
        DataFrame with one row per unique prompt. Columns:
            prompt, label, label_id, reasoning (empty list — filled later),
            observed_response_count, observed_safe_count, observed_unsafe_count,
            unsafe_response_rate, any_unsafe_response, all_unsafe_responses,
            harm_categories, source, source_split
    """
    # Convert to pandas for groupby operations
    df = dataset.to_pandas()

    records = []
    for prompt, group in tqdm(df.groupby("prompt", sort=False), desc="Aggregating prompts"):
        n_total = len(group)
        n_unsafe = int((~group["is_safe"]).sum())
        n_safe = n_total - n_unsafe

        any_unsafe = n_unsafe > 0
        all_unsafe = n_unsafe == n_total

        label = LABEL_HIGH_RISK if any_unsafe else LABEL_LOW_RISK

        # Step 5: harm categories only from unsafe rows
        unsafe_rows = group[~group["is_safe"]]
        harm_cats: set[str] = set()
        for cats in unsafe_rows["harm_categories"]:
            harm_cats.update(cats)

        records.append(
            {
                "prompt": prompt,
                "label": label,
                "label_id": LABEL_ID[label],
                "reasoning": [],  # populated by generate_rationales.py
                "observed_response_count": n_total,
                "observed_safe_count": n_safe,
                "observed_unsafe_count": n_unsafe,
                "unsafe_response_rate": round(n_unsafe / n_total, 4),
                "any_unsafe_response": any_unsafe,
                "all_unsafe_responses": all_unsafe,
                "harm_categories": sorted(harm_cats),
                "source": source,
                "source_split": source_split,
            }
        )

    return pd.DataFrame(records)


def save_aggregates(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save aggregated DataFrame to parquet.

    Args:
        df: Aggregated prompts DataFrame.
        output_dir: Directory to write prompt_aggregates.parquet into.

    Returns:
        Path to the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "prompt_aggregates.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} unique prompts → {path}")
    return path


if __name__ == "__main__":
    from latent_watch.data.load_beavertails import load_beavertails
    from latent_watch.data.normalize import normalize_dataset

    raw = load_beavertails()
    norm = normalize_dataset(raw)
    agg = aggregate_prompts(norm)

    print(f"\nUnique prompts: {len(agg):,}")
    print(agg["label"].value_counts())

    save_aggregates(agg, "data/interim")
