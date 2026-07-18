"""
Convert BeaverTails (PKU-Alignment/BeaverTails) into the JSON schema expected
by connordilgren/are-lrms-easily-interpretable's dataset loader
(src_coconut_multimode/dataset.py::get_dataset), for use with
experiments/early_stopping/run.py (Stage 0/1 necessity + OOD-validity checks).

Target schema (list of objects, `idx` auto-assigned by their loader -- do not
include it yourself):
    [{"question": "...", "steps": [], "answer": ""}, ...]

BeaverTails pairs one prompt with MULTIPLE responses (some safe, some not).
Since this project cares about detecting harmful *intent* from the prompt
before any answer is generated, we dedupe by prompt and label a prompt
harmful if ANY of its paired responses was flagged unsafe (is_safe == False).
This is a conservative / high-recall choice: a prompt that sometimes elicits
an unsafe response is treated as harmful-intent-bearing even if most
responses to it were safe. Document this choice in your methods section --
it is not the only valid choice (majority-vote and no-dedup alternatives
were considered and rejected in favor of recall).

Output: two files, joined by a shared `idx` (positional index into the
schema JSON, 0-indexed, matching what the loader assigns):
  1. <output_dir>/<split>_schema.json     -- schema-compliant, feed to run.py
  2. <output_dir>/<split>_metadata.jsonl  -- one JSON object per line, with
     the harmfulness label and category flags for later analysis. This file
     is NOT read by run.py -- the loader discards all fields except
     question/steps/answer/idx during tokenization -- so it's safe to keep
     your labels here without touching their pipeline code.

Usage:
    pip install datasets --break-system-packages   # if not already installed
    python beavertails_to_schema.py --split 30k_train --output_dir data/
    python beavertails_to_schema.py --split 30k_train --balance --max_samples 500
"""

import argparse
import json
import os
from collections import defaultdict


HARM_CATEGORIES = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]


def load_beavertails(split: str):
    """
    Load BeaverTails from the Hugging Face Hub.

    Splits available (per the dataset card): '30k_train', '30k_test',
    '330k_train', '330k_test'. Note the ~330k splits are much larger and
    slower to process -- start with 30k for the sanity/Stage-0 pass.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The `datasets` library is required. Install with:\n"
            "  pip install datasets --break-system-packages"
        ) from e

    print(f"Loading PKU-Alignment/BeaverTails split='{split}' ...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    print(f"  Loaded {len(ds)} (prompt, response) rows.")
    return ds


def group_by_prompt(ds):
    """
    Group rows by prompt text. Returns dict: prompt -> list of row dicts
    (each with 'response', 'category', 'is_safe').
    """
    groups = defaultdict(list)
    for row in ds:
        groups[row["prompt"]].append(
            {
                "response": row["response"],
                "category": row["category"],
                "is_safe": row["is_safe"],
            }
        )
    print(f"  {len(groups)} unique prompts after dedup "
          f"(from {sum(len(v) for v in groups.values())} rows).")
    return groups


def label_prompt(responses: list) -> dict:
    """
    Derive prompt-level labels from its paired responses.

    Harmfulness rule: harmful if ANY paired response is unsafe (high recall,
    conservative choice -- see module docstring).

    Category aggregation: union of category flags across responses where
    is_safe == False. (Categories on safe responses are typically all-False
    in BeaverTails, but we restrict to unsafe responses for a cleaner signal
    regardless.)
    """
    is_harmful_any = any(not r["is_safe"] for r in responses)

    category_union = {cat: False for cat in HARM_CATEGORIES}
    for r in responses:
        if not r["is_safe"]:
            for cat in HARM_CATEGORIES:
                if r["category"].get(cat, False):
                    category_union[cat] = True

    return {
        "is_harmful_any": is_harmful_any,
        "num_responses": len(responses),
        "num_unsafe_responses": sum(1 for r in responses if not r["is_safe"]),
        "category_union": category_union,
    }


def build_records(groups: dict, balance: bool, max_samples: int, seed: int):
    """
    Build the list of (prompt, label_info) records, optionally balanced
    and/or subsampled.
    """
    import random

    rng = random.Random(seed)

    records = []
    for prompt, responses in groups.items():
        label_info = label_prompt(responses)
        records.append((prompt, label_info))

    harmful = [r for r in records if r[1]["is_harmful_any"]]
    safe = [r for r in records if not r[1]["is_harmful_any"]]
    print(f"  Prompt-level labels: {len(harmful)} harmful, {len(safe)} safe "
          f"({len(harmful) / len(records):.1%} harmful).")

    if balance:
        n = min(len(harmful), len(safe))
        rng.shuffle(harmful)
        rng.shuffle(safe)
        harmful, safe = harmful[:n], safe[:n]
        records = harmful + safe
        rng.shuffle(records)
        print(f"  Balanced to {n} harmful + {n} safe = {len(records)} total.")
    else:
        rng.shuffle(records)

    if max_samples is not None and len(records) > max_samples:
        if balance:
            # Preserve balance when truncating
            half = max_samples // 2
            harmful_sub = [r for r in records if r[1]["is_harmful_any"]][:half]
            safe_sub = [r for r in records if not r[1]["is_harmful_any"]][:max_samples - half]
            records = harmful_sub + safe_sub
            rng.shuffle(records)
        else:
            records = records[:max_samples]
        print(f"  Truncated to max_samples={max_samples} -> {len(records)} total.")

    return records


def write_outputs(records: list, output_dir: str, split: str):
    os.makedirs(output_dir, exist_ok=True)

    schema_path = os.path.join(output_dir, f"{split.replace('/', '_')}_schema.json")
    metadata_path = os.path.join(output_dir, f"{split.replace('/', '_')}_metadata.jsonl")

    schema_data = []
    with open(metadata_path, "w") as meta_f:
        for idx, (prompt, label_info) in enumerate(records):
            # Schema-compliant record for run.py's loader.
            # answer="" since safety prompts have no canonical final answer --
            # do not fabricate a gold answer; leaving it empty avoids implying
            # a ground truth that doesn't exist.
            #
            # steps=[""] (a single empty-string step), NOT steps=[] --
            # get_dataset()'s internal verification assertion in
            # src_coconut_multimode/dataset.py has an off-by-one bug that
            # ALWAYS fails on a genuinely empty steps list: its verification
            # string builder assumes >=1 step (each step contributes a
            # newline via "\n".join, plus a fixed trailing "\n### "), so
            # N=0 steps produces one newline too many versus how the pieces
            # are actually tokenized and concatenated. A single empty-string
            # step avoids this (adds one harmless blank-line token, but the
            # character counts then line up correctly on both sides of the
            # assertion).
            schema_data.append({
                "question": prompt,
                "steps": [""],
                "answer": "",
            })

            # Metadata sidecar -- NOT read by their pipeline, kept for your
            # own analysis after inference. idx matches the position in
            # schema_data, which matches the idx their loader auto-assigns.
            meta_record = {
                "idx": idx,
                "prompt": prompt,
                "source": "beavertails",
                **label_info,
            }
            meta_f.write(json.dumps(meta_record) + "\n")

    with open(schema_path, "w") as f:
        json.dump(schema_data, f, indent=2)

    print(f"\nWrote {len(schema_data)} records:")
    print(f"  Schema file (feed to run.py):  {schema_path}")
    print(f"  Metadata file (for analysis):  {metadata_path}")
    print(f"\nNote: extract_answer() in run.py requires the model to emit a "
          f"'###' delimiter in its free-generated output to register a "
          f"non-skipped sample. Since these prompts have no gold answer, "
          f"skip behavior depends entirely on whether the model spontaneously "
          f"produces '###' when responding to a safety prompt -- this may "
          f"differ substantially from its GSM8K-trained behavior. Check the "
          f"printed 'Skipped (early stopping): N' count when you run "
          f"early_stopping.run against this file -- a skip rate anywhere "
          f"near 100% means the delimiter assumption is failing silently on "
          f"this domain and results are not trustworthy as-is. Consider "
          f"prompting the model explicitly to end its answer with '###' if "
          f"the natural skip rate is high, and report whichever choice you "
          f"make.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=str, default="30k_train",
                         choices=["30k_train", "30k_test", "330k_train", "330k_test"],
                         help="BeaverTails split to load (default: 30k_train)")
    parser.add_argument("--output_dir", type=str, default="data/",
                         help="Directory to write schema JSON + metadata JSONL")
    parser.add_argument("--balance", action="store_true",
                         help="Downsample to equal harmful/safe prompt counts")
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Cap total number of prompt records (after any balancing)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for shuffling/subsampling")
    args = parser.parse_args()

    ds = load_beavertails(args.split)
    groups = group_by_prompt(ds)
    records = build_records(groups, args.balance, args.max_samples, args.seed)
    write_outputs(records, args.output_dir, args.split)


if __name__ == "__main__":
    main()
