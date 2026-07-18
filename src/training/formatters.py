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

    Reads from answer_only/ JSONL rows, which use a messages format:
      messages[0]["content"] = pre-formatted user prompt
      messages[1]["content"] = label string (HIGH_RISK or LOW_RISK)

    Returns:
        (input_text, target_text)
    """
    input_text  = row["messages"][0]["content"]
    target_text = row["messages"][1]["content"]  # "HIGH_RISK" or "LOW_RISK"
    return input_text, target_text


# ── E2: Chain-of-Thought ──────────────────────────────────────────────────

def format_cot(row: dict[str, Any]) -> tuple[str, str]:
    """Format a row for CoT training (E2).

    Reads from cot/ JSONL rows, which use a messages format:
      messages[0]["content"] = pre-formatted user prompt
      messages[1]["content"] = <reasoning>...</reasoning>\n<answer>LABEL</answer>

    Loss is computed on the full target (reasoning + answer tokens).

    Returns:
        (input_text, target_text)
    """
    input_text  = row["messages"][0]["content"]
    target_text = row["messages"][1]["content"]
    return input_text, target_text


# ── E3: COCONUT curriculum ────────────────────────────────────────────────

def format_coconut_stage(
    row: dict[str, Any],
    stage: int,
    num_stages: int,
    bot_token: str = "<bot>",
) -> tuple[str, str]:
    """Format a row for a specific COCONUT curriculum stage (E3).

    Reads from cot/ JSONL rows (same files as E2). Extracts the pre-formatted
    user prompt from messages[0]["content"] and reconstructs the target by
    progressively replacing reasoning sentences with <bot> tokens.

    Stage 0: full CoT target (identical to E2 — reads messages[1] directly).
    Stage k>0: first k reasoning sentences replaced by k <bot> tokens.
    Stage == num_stages (final): all reasoning replaced, only <answer> remains.

    Args:
        row: cot/ JSONL row with messages and label fields.
        stage: Current curriculum stage (0-indexed).
        num_stages: Total number of stages C (pilot: 3).
        bot_token: Special token string for latent thought.

    Returns:
        (input_text, target_text)
    """
    input_text = row["messages"][0]["content"]
    label: str = row["label"]

    # Extract reasoning steps from the pre-formatted assistant message
    # Format: "<reasoning>\nstep1\nstep2\nstep3\n</reasoning>\n<answer>LABEL</answer>"
    import re
    assistant_content = row["messages"][1]["content"]
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", assistant_content, re.DOTALL)
    if reasoning_match:
        reasoning_steps = [s.strip() for s in reasoning_match.group(1).strip().split("\n") if s.strip()]
    else:
        reasoning_steps = []

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