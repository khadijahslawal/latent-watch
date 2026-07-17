"""Step 1: Load the BeaverTails dataset from HuggingFace Hub.

Returns a HuggingFace Dataset object (330k_train split).
Data is streamed/cached by datasets; raw rows are never written to disk.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset


def load_beavertails(split: str = "330k_train") -> Dataset:
    """Load BeaverTails from HuggingFace Hub.

    Args:
        split: HuggingFace split name. Default is '330k_train'.

    Returns:
        datasets.Dataset with the raw BeaverTails rows.
    """
    dataset: Dataset = load_dataset(
        "PKU-Alignment/BeaverTails",
        split=split,
    )
    return dataset


if __name__ == "__main__":
    ds = load_beavertails()
    print(f"Loaded {len(ds):,} rows")
    print("Columns:", ds.column_names)
    print("Sample row:", ds[0])
