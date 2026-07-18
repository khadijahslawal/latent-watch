"""LoRA adapter utilities: attach, save, load, and inspect adapters.

Ordering rule (from training strategy):
  For E1/E2: load base model → attach LoRA
  For E3:    load base model → resize embeddings → attach LoRA → wrap in COCONUT

Do NOT call attach_lora on a COCONUT wrapper. Always attach to the inner base model.
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ── Default LoRA config (shared across all experiments) ───────────────────

LORA_DEFAULTS = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


def load_base_model(
    model_name: str = "meta-llama/Llama-3.2-1B",
    load_in_4bit: bool = True,
    fp16: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and tokenizer.

    Args:
        model_name: HuggingFace model name or local path.
        load_in_4bit: Load in NF4 quantization (required for T4).
        fp16: Use fp16 compute dtype (T4). Set False for A100 with bf16.

    Returns:
        (model, tokenizer) tuple. Model is on GPU if available.
    """
    compute_dtype = torch.float16 if fp16 else torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=compute_dtype if not load_in_4bit else None,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # causal LM: pad on left
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def attach_lora(
    model: AutoModelForCausalLM,
    r: int = LORA_DEFAULTS["r"],
    lora_alpha: int = LORA_DEFAULTS["lora_alpha"],
    lora_dropout: float = LORA_DEFAULTS["lora_dropout"],
    target_modules: list[str] = LORA_DEFAULTS["target_modules"],
    bias: str = LORA_DEFAULTS["bias"],
) -> AutoModelForCausalLM:
    """Attach LoRA adapters to the base model.

    Args:
        model: Base CausalLM model (not a COCONUT wrapper).
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout on LoRA layers.
        target_modules: Which attention projections to adapt.
        bias: Bias training mode.

    Returns:
        PEFT-wrapped model with LoRA adapters attached.
    """
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model


def print_trainable_parameters(model) -> None:
    """Print trainable vs total parameter count. Expected ~2–4M / ~1B."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )


def save_adapter(model, output_dir: str | Path) -> None:
    """Save only the LoRA adapter weights (not the base model).

    Args:
        model: PEFT model or COCONUT wrapper containing a PEFT inner model.
        output_dir: Directory to save adapter weights into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Handle COCONUT wrapper: save the inner peft model
    inner = getattr(model, "model", model)
    inner.save_pretrained(str(out))
    print(f"Adapter saved → {out}")


def save_adapter_with_tokenizer(
    model, tokenizer, output_dir: str | Path
) -> None:
    """Save adapter weights AND tokenizer (needed when vocab was extended, e.g. E3).

    Args:
        model: PEFT model or COCONUT wrapper.
        tokenizer: Tokenizer (may have added special tokens).
        output_dir: Directory to save into.
    """
    save_adapter(model, output_dir)
    tokenizer.save_pretrained(str(output_dir))
    print(f"Tokenizer saved → {output_dir}")


def load_adapter(
    base_model: AutoModelForCausalLM,
    adapter_dir: str | Path,
) -> PeftModel:
    """Load LoRA adapter weights onto a base model.

    Args:
        base_model: The same base model used during training.
        adapter_dir: Directory containing adapter_config.json and weights.

    Returns:
        PEFT model with adapter loaded.
    """
    return PeftModel.from_pretrained(base_model, str(adapter_dir))