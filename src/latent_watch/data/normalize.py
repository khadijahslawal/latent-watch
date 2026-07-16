"""Step 2: Normalize each BeaverTails source row into a consistent internal structure.

Normalization rules:
- Collapse runs of whitespace in prompt (strip leading/trailing, normalize internal spaces).
- Do NOT lowercase or remove punctuation — those carry semantic signal.
- Cast is_safe to bool explicitly (BeaverTails uses bool already, but be defensive).
- Flatten the harm category dict into a list of active category names.

Expected BeaverTails columns (330k_train):
    prompt, response, is_safe, category (dict of category_name -> bool)
"""

from __future__ import annotations

import re
from typing import Any

from datasets import Dataset


# BeaverTails stores harm categories as a dict {category_name: bool}.
# We unpack only the True entries.
HARM_CATEGORY_KEY = "category"


def _normalize_prompt(text: str) -> str:
    """Strip and collapse internal whitespace. Preserve case and punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_harm_categories(category_dict: dict[str, bool]) -> list[str]:
    """Return list of category names where value is True."""
    return [k for k, v in category_dict.items() if v]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single BeaverTails row.

    Args:
        row: Raw row dict from the BeaverTails dataset.

    Returns:
        Normalized dict with keys:
            prompt (str), response (str), is_safe (bool), harm_categories (list[str])
    """
    return {
        "prompt": _normalize_prompt(row["prompt"]),
        "response": row["response"],
        "is_safe": bool(row["is_safe"]),
        "harm_categories": _extract_harm_categories(row[HARM_CATEGORY_KEY]),
    }


def normalize_dataset(dataset: Dataset) -> Dataset:
    """Apply normalization to every row in a HuggingFace Dataset.

    Args:
        dataset: Raw BeaverTails dataset.

    Returns:
        Dataset with normalized columns. Original columns are replaced.
    """
    return dataset.map(
        normalize_row,
        remove_columns=dataset.column_names,
        desc="Normalizing rows",
    )


if __name__ == "__main__":
    from latent_watch.data.load_beavertails import load_beavertails

    ds = load_beavertails()
    norm = normalize_dataset(ds)
    print(f"Normalized {len(norm):,} rows")
    print("Columns:", norm.column_names)
    print("Sample:", norm[0])
