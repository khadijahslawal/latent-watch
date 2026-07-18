"""Dataset formatters for E1 (answer-only), E2 (CoT), and E3 (COCONUT curriculum).

Each formatter takes a raw JSONL row and returns (input_text, target_text).
The base trainer tokenizes them and applies loss masking.

Input format is identical across all three experiments:
    <|user|>
    Assess the observed safety risk of the following prompt.
    Prompt:
    {prompt}
    A prompt is HIGH_RISK if at least one observed response to it was unsafe.
    A prompt is LOW_RISK if all observed responses were safe.
    <|assistant|>

Only the target text differs by experiment.
"""

from __future__ import annotations

from typing import Any


USER_PROMPT_TEMPLATE = (
    "<|user|>\n"
    "Assess the observed safety risk of the following prompt.\n\n"
    "Prompt:\n{prompt}\n\n"
    "A prompt is HIGH_RISK if at least one observed response to it was unsafe.\n"
    "A prompt is LOW_RISK if all observed responses were safe.\n"
    "<|assistant|>\n"
)


# ── E1: Answer-only ───────────────────────────────────────────────────────

def format_answer_only(row: dict[str, Any]) -> tuple[str, str]:
    """Format a row for answer-only training (E1).

    Target is the label string only. Loss computed on this token only.

    Returns:
        (input_text, target_text)
    """
    input_text = USER_PROMPT_TEMPLATE.format(prompt=row["prompt"])
    target_text = row["label"]  # "HIGH_RISK" or "LOW_RISK"
    return input_text, target_text


# ── E2: Chain-of-Thought ──────────────────────────────────────────────────

def format_cot(row: dict[str, Any]) -> tuple[str, str]:
    """Format a row for CoT training (E2).

    Target includes <reasoning> block + <answer> tag.
    Loss computed on full target (reasoning + answer tokens).

    Returns:
        (input_text, target_text)
    """
    input_text = USER_PROMPT_TEMPLATE.format(prompt=row["prompt"])
    reasoning_steps = row["reasoning"]  # list of 3 strings
    reasoning_text = "\n".join(reasoning_steps)
    target_text = (
        f"<reasoning>\n{reasoning_text}\n</reasoning>\n"
        f"<answer>{row['label']}</answer>"
    )
    return input_text, target_text


# ── E3: COCONUT curriculum ────────────────────────────────────────────────

def format_coconut_stage(
    row: dict[str, Any],
    stage: int,
    num_stages: int,
    bot_token: str = "<bot>",
) -> tuple[str, str]:
    """Format a row for a specific COCONUT curriculum stage (E3).

    Stage 0: full CoT target (identical to E2).
    Stage k>0: first k reasoning sentences replaced by k <bot> tokens.
    Stage == num_stages (final): all reasoning replaced, only <answer> remains.

    The continuous thought tokens (<bot>) are not supervised — loss masking
    in the trainer handles this by setting their labels to -100.

    Args:
        row: JSONL row with reasoning list and label.
        stage: Current curriculum stage (0-indexed).
        num_stages: Total number of stages C (pilot: 3).
        bot_token: Special token string for latent thought.

    Returns:
        (input_text, target_text)
    """
    input_text = USER_PROMPT_TEMPLATE.format(prompt=row["prompt"])
    reasoning_steps: list[str] = row["reasoning"]  # 3 strings
    label: str = row["label"]

    if stage == 0:
        # Identical to CoT
        reasoning_text = "\n".join(reasoning_steps)
        target_text = (
            f"<reasoning>\n{reasoning_text}\n</reasoning>\n"
            f"<answer>{label}</answer>"
        )
    elif stage >= num_stages:
        # All reasoning replaced — only latent tokens + answer
        latent_tokens = " ".join([bot_token] * num_stages)
        target_text = f"{latent_tokens} <answer>{label}</answer>"
    else:
        # Replace first `stage` reasoning sentences with latent tokens
        latent_tokens = " ".join([bot_token] * stage)
        remaining_steps = reasoning_steps[stage:]
        remaining_text = "\n".join(remaining_steps)
        target_text = (
            f"{latent_tokens} {remaining_text}\n"
            f"<answer>{label}</answer>"
        )

    return input_text, target_text


def get_formatter(experiment: str):
    """Return the formatter function for the given experiment name.

    Args:
        experiment: One of 'answer_only', 'cot', 'coconut'.

    Returns:
        Formatter callable with signature (row) -> (input_text, target_text).
        For coconut, returns format_coconut_stage with stage/num_stages bound
        externally by the trainer.
    """
    if experiment == "answer_only":
        return format_answer_only
    elif experiment == "cot":
        return format_cot
    elif experiment == "coconut":
        # COCONUT formatter is stage-dependent — caller passes stage explicitly
        return format_coconut_stage
    else:
        raise ValueError(f"Unknown experiment: {experiment!r}. Expected 'answer_only', 'cot', or 'coconut'.")
