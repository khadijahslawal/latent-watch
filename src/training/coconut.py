"""COCONUT: Continuous Chain-of-Thought wrapper and curriculum dataset.

Implements COCONUT from scratch based on Hao et al. 2024 (Section 3).
Reference: "Training Large Language Models to Reason in a Continuous Latent Space"

Key idea: Replace explicit reasoning tokens with recurrent hidden-state passes.
At curriculum stage k, the first k reasoning sentences are replaced by k continuous
thought tokens (<bot>). The model performs k recurrent forward passes through its
hidden states for the latent segment. Gradients flow back through the wrapper into
the LoRA adapters on the inner base model.

LoRA + COCONUT ordering (enforced here, not in wrapper):
  The caller must attach LoRA to the base model BEFORE passing it to CoconutWrapper.
  Do not apply LoRA to the wrapper itself.

Special tokens:
  <bot> = beginning of (latent) thought — inserted at tokenizer level
  <eot> = end of thought — reserved for future use; added for vocabulary completeness
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from latent_watch.training.formatters import format_coconut_stage


# ── COCONUT Wrapper ───────────────────────────────────────────────────────

class CoconutWrapper(nn.Module):
    """Wraps a LoRA-patched CausalLM model for COCONUT training.

    At each forward pass:
      1. Run the model on the prefix (input + any explicit tokens so far) to get
         hidden states at the last position.
      2. For each latent thought slot (<bot> position), feed the hidden state back
         as the input embedding for the next pass (recurrent latent reasoning).
      3. Concatenate the final hidden state with the remaining explicit tokens and
         compute the loss on supervised positions only.

    Args:
        model: LoRA-patched AutoModelForCausalLM. LoRA must already be attached.
        bot_token_id: Token id of the <bot> special token.
        eot_token_id: Token id of the <eot> special token.
    """

    def __init__(
        self,
        model: nn.Module,
        bot_token_id: int,
        eot_token_id: int,
    ):
        super().__init__()
        self.model = model
        self.bot_token_id = bot_token_id
        self.eot_token_id = eot_token_id
        self.config = model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        """Forward pass with latent thought recurrence for <bot> positions.

        For positions where input_ids == bot_token_id, instead of embedding the
        token, we feed the hidden state from the previous pass back as the input.
        This implements the recurrent latent thought mechanism from Hao et al.

        Args:
            input_ids: (batch, seq_len) — may contain bot_token_id positions.
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) — bot positions have label = -100.

        Returns:
            ModelOutput with .loss computed on non-(-100) label positions.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Find <bot> positions in the sequence (same for all items in batch)
        # We assume the curriculum ensures bot tokens are contiguous after the prompt
        bot_mask = (input_ids == self.bot_token_id)  # (batch, seq_len)
        has_latent = bot_mask.any()

        if not has_latent:
            # Stage 0 or no latent tokens: standard forward pass
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # ── Latent recurrence ─────────────────────────────────────────────
        # Split sequence at first <bot> token
        # Everything before first <bot> = prefix (input + explicit context)
        # <bot> positions = latent slots
        # Everything after last <bot> = explicit suffix (remaining reasoning + answer)

        # Get the position of the first <bot> in the sequence
        # (assumed same across batch items in a given training batch)
        bot_positions = bot_mask[0].nonzero(as_tuple=True)[0]
        first_bot = bot_positions[0].item()
        last_bot = bot_positions[-1].item()
        num_latent = len(bot_positions)

        prefix_ids = input_ids[:, :first_bot]
        prefix_mask = attention_mask[:, :first_bot]
        suffix_ids = input_ids[:, last_bot + 1:]
        suffix_labels = labels[:, last_bot + 1:]

        # Step 1: Run prefix to get hidden states
        embed_layer = self.model.get_input_embeddings()

        prefix_embeds = embed_layer(prefix_ids)  # (batch, prefix_len, hidden)

        # Iteratively feed hidden state back for each latent slot
        current_embeds = prefix_embeds
        current_mask = prefix_mask

        for _ in range(num_latent):
            outputs = self.model(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                output_hidden_states=True,
            )
            # Last hidden state at final position = latent thought vector
            latent_hidden = outputs.hidden_states[-1][:, -1:, :]  # (batch, 1, hidden)

            # Append latent hidden state as next "token embedding"
            current_embeds = torch.cat([current_embeds, latent_hidden], dim=1)
            current_mask = torch.cat([
                current_mask,
                torch.ones(batch_size, 1, dtype=torch.long, device=device),
            ], dim=1)

        # Step 2: Append suffix embeddings
        if suffix_ids.shape[1] > 0:
            suffix_embeds = embed_layer(suffix_ids)
            full_embeds = torch.cat([current_embeds, suffix_embeds], dim=1)
            full_mask = torch.cat([
                current_mask,
                attention_mask[:, last_bot + 1:],
            ], dim=1)
        else:
            full_embeds = current_embeds
            full_mask = current_mask

        # Step 3: Build labels — mask prefix and latent positions, supervise suffix
        prefix_labels = torch.full(
            (batch_size, current_embeds.shape[1]),
            -100,
            dtype=torch.long,
            device=device,
        )
        if suffix_ids.shape[1] > 0:
            full_labels = torch.cat([prefix_labels, suffix_labels], dim=1)
        else:
            full_labels = prefix_labels

        # Step 4: Final forward pass with full sequence
        return self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """Inference: run latent passes silently, then generate the answer token.

        The latent thought tokens do not produce readable output.
        The model generates the <answer>LABEL</answer> suffix.
        """
        # At inference we don't have <bot> tokens in input — we run the base
        # model directly and let it generate. The latent behaviour is only
        # meaningful when the model has been trained to produce latent states.
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    # Delegate parameter access to inner model for PEFT compatibility
    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()


# ── Special token setup ───────────────────────────────────────────────────

BOT_TOKEN = "<bot>"
EOT_TOKEN = "<eot>"


def add_coconut_tokens(
    tokenizer: AutoTokenizer,
    model: nn.Module,
) -> tuple[int, int]:
    """Add <bot> and <eot> special tokens and resize model embeddings.

    Must be called BEFORE attaching LoRA adapters so the new embedding
    rows are included in the parameter count.

    Args:
        tokenizer: Tokenizer to extend.
        model: Base model (pre-LoRA).

    Returns:
        (bot_token_id, eot_token_id)
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [BOT_TOKEN, EOT_TOKEN]}
    )
    print(f"Added {num_added} special tokens: {BOT_TOKEN}, {EOT_TOKEN}")

    model.resize_token_embeddings(len(tokenizer))
    print(f"Embedding matrix resized to {len(tokenizer)} tokens")

    bot_id = tokenizer.convert_tokens_to_ids(BOT_TOKEN)
    eot_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
    return bot_id, eot_id


# ── COCONUT curriculum dataset ────────────────────────────────────────────

class CoconutCurriculumDataset(Dataset):
    """Dataset for a single COCONUT curriculum stage.

    Formats each row using format_coconut_stage for the given stage,
    then tokenizes with loss masking. <bot> token positions always
    get label = -100 (they are not supervised).

    Args:
        rows: List of JSONL row dicts.
        tokenizer: Extended tokenizer with <bot>/<eot>.
        stage: Current curriculum stage (0 = full CoT, C = fully latent).
        num_stages: Total number of stages C.
        bot_token_id: Token id for <bot>.
        max_seq_length: Max token length.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: AutoTokenizer,
        stage: int,
        num_stages: int,
        bot_token_id: int,
        max_seq_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.bot_token_id = bot_token_id
        self.max_seq_length = max_seq_length
        self.samples = []

        formatter = partial(
            format_coconut_stage,
            stage=stage,
            num_stages=num_stages,
            bot_token=BOT_TOKEN,
        )

        for row in rows:
            input_text, target_text = formatter(row)
            self.samples.append(self._tokenize(input_text, target_text))

    def _tokenize(self, input_text: str, target_text: str) -> dict[str, torch.Tensor]:
        tok = self.tokenizer

        input_ids = tok(input_text, add_special_tokens=True, return_tensors="pt").input_ids.squeeze(0)
        target_ids = tok(target_text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

        full_ids = torch.cat([input_ids, target_ids], dim=0)

        # Build labels: mask prompt, supervise target but mask <bot> positions
        target_labels = target_ids.clone()
        target_labels[target_labels == self.bot_token_id] = -100

        labels = torch.cat([
            torch.full((len(input_ids),), -100, dtype=torch.long),
            target_labels,
        ], dim=0)

        # Truncate
        full_ids = full_ids[: self.max_seq_length]
        labels = labels[: self.max_seq_length]

        # Pad (left-pad for causal LM)
        pad_len = self.max_seq_length - len(full_ids)
        attention_mask = torch.ones(len(full_ids), dtype=torch.long)

        if pad_len > 0:
            pad_id = tok.pad_token_id
            full_ids = torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), full_ids])
            labels = torch.cat([torch.full((pad_len,), -100, dtype=torch.long), labels])
            attention_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), attention_mask])

        return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]
