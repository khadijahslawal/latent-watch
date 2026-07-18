"""Step 8: Generate concise reasoning traces via the OpenAI API.

Rationales are produced by gpt-4o-mini (configurable via --rationale-model).
They are used only for CoT and COCONUT training — not the answer-only baseline.

IMPORTANT: The rationale generator sees only prompt, label, and harm_categories.
It must NOT see paired responses (information unavailable at inference time).

Teacher prompt follows the specification in the data strategy doc.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


TEACHER_PROMPT_TEMPLATE = """\
You are generating a concise training rationale for a prompt-risk classifier.
The classifier predicts whether a prompt has demonstrated a risk of eliciting
an unsafe response.

Label definition:
- HIGH_RISK: at least one observed response to the prompt was unsafe.
- LOW_RISK: all observed responses were safe.

Prompt:
{prompt}

Ground-truth label:
{label}

Relevant observed harm categories:
{categories}

Write exactly three concise reasoning steps:
1. Describe the action or information requested.
2. Explain why responding could or could not create safety risk.
3. Justify the supplied label.

Do not change the supplied label.
Do not mention dataset annotations, response counts, or that other responses were observed.\
"""


def _format_categories(cats: list[str]) -> str:
    if not cats:
        return "None"
    return ", ".join(cats)


def generate_rationale_for_row(
    client: OpenAI,
    prompt: str,
    label: str,
    harm_categories: list[str],
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> list[str] | None:
    """Call the OpenAI API to generate three reasoning steps for one example.

    Args:
        client: Authenticated OpenAI client.
        prompt: The prompt text.
        label: 'HIGH_RISK' or 'LOW_RISK'.
        harm_categories: List of harm category strings (empty for LOW_RISK).
        model: OpenAI model name.
        max_retries: Number of retry attempts on API errors.
        retry_delay: Seconds between retries.

    Returns:
        List of three reasoning strings, or None if generation failed.
    """
    teacher_input = TEACHER_PROMPT_TEMPLATE.format(
        prompt=prompt,
        label=label,
        categories=_format_categories(harm_categories),
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": teacher_input}],
                temperature=0.3,
                max_tokens=300,
            )
            raw = response.choices[0].message.content or ""
            steps = _parse_reasoning_steps(raw)
            return steps
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f"[WARN] Failed after {max_retries} attempts: {exc}")
                return None

    return None


def _parse_reasoning_steps(raw: str) -> list[str]:
    """Extract the three numbered reasoning steps from model output.

    Handles both '1. text' and '1) text' formats.
    Falls back to splitting by lines if numbered markers aren't found.
    """
    import re

    lines = raw.strip().split("\n")
    steps = []
    current = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[1-3][.)]\s", line):
            if current:
                steps.append(" ".join(current).strip())
                current = []
            current.append(re.sub(r"^[1-3][.)]\s+", "", line))
        else:
            current.append(line)

    if current:
        steps.append(" ".join(current).strip())

    # Ensure exactly 3 steps; pad or truncate
    while len(steps) < 3:
        steps.append("")
    return steps[:3]


def generate_rationales(
    df: pd.DataFrame,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    output_path: str | Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Generate reasoning traces for all rows in df that don't already have them.

    Args:
        df: DataFrame with columns: prompt, label, harm_categories, reasoning.
            Rows where reasoning is non-empty are skipped (resume behaviour).
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model: OpenAI model for rationale generation.
        output_path: If provided, write intermediate results here as JSONL after each batch.
        resume: If True, skip rows where reasoning is already populated.

    Returns:
        DataFrame with reasoning column populated.
    """
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    df = df.copy()

    # Ensure reasoning column exists and is a list
    if "reasoning" not in df.columns:
        df["reasoning"] = [[] for _ in range(len(df))]

    indices_to_generate = (
        df.index[df["reasoning"].apply(lambda r: len(r) == 0)]
        if resume
        else df.index
    )

    print(f"Generating rationales for {len(indices_to_generate):,} rows using {model}...")

    for idx in tqdm(indices_to_generate, desc="Generating rationales"):
        row = df.loc[idx]
        steps = generate_rationale_for_row(
            client=client,
            prompt=row["prompt"],
            label=row["label"],
            harm_categories=row["harm_categories"],
            model=model,
        )
        if steps is not None:
            df.at[idx, "reasoning"] = steps

    if output_path is not None:
        _write_jsonl(df, Path(output_path))

    return df


def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"Wrote {len(df):,} rows → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rationales via OpenAI API")
    parser.add_argument("--input", required=True, help="Input parquet or JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--rationale-model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--no-resume", action="store_true", help="Regenerate all rationales")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_json(input_path, lines=True)

    df = generate_rationales(
        df,
        api_key=args.api_key,
        model=args.rationale_model,
        output_path=args.output,
        resume=not args.no_resume,
    )
    print("Done.")


if __name__ == "__main__":
    main()
