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
from transformers.modeling_outputs import CausalLMOutput

from src.training.formatters import format_coconut_stage


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
        num_latent_steps: Number of recurrent latent passes used at inference.
    """

    def __init__(
        self,
        model: nn.Module,
        bot_token_id: int,
        eot_token_id: int,
        num_latent_steps: int,
    ):
        super().__init__()
        self.model = model
        self.bot_token_id = bot_token_id
        self.eot_token_id = eot_token_id
        self.num_latent_steps = num_latent_steps
        self.config = model.config

    def _forward_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one example using its own latent-token positions."""

        # Remove left padding.
        real_tokens = attention_mask.bool()
        input_ids = input_ids[real_tokens].unsqueeze(0)
        labels = labels[real_tokens].unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        bot_positions = (input_ids[0] == self.bot_token_id).nonzero(as_tuple=True)[0]

        # For examples or stages without latent tokens
        if bot_positions.numel() == 0:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            supervised_tokens = (labels != -100).sum().clamp_min(1)
            return output.loss, supervised_tokens

        first_bot = bot_positions[0].item()
        last_bot = bot_positions[-1].item()
        num_latent = bot_positions.numel()

        expected_positions = torch.arange(
            first_bot,
            last_bot + 1,
            device=bot_positions.device,
        )
        if not torch.equal(bot_positions, expected_positions):
            raise ValueError(
                "Expected contiguous <bot> tokens, but found positions "
                f"{bot_positions.tolist()}"
            )

        prefix_ids = input_ids[:, :first_bot]
        prefix_mask = attention_mask[:, :first_bot]

        suffix_ids = input_ids[:, last_bot + 1:]
        suffix_mask = attention_mask[:, last_bot + 1:]
        suffix_labels = labels[:, last_bot + 1:]

        embed_layer = self.model.get_input_embeddings()
        current_embeds = embed_layer(prefix_ids)
        current_mask = prefix_mask

        for _ in range(num_latent):
            output = self.model(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            latent_hidden = output.hidden_states[-1][:, -1:, :]

            current_embeds = torch.cat([current_embeds, latent_hidden], dim=1)
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(
                        (1, 1),
                        dtype=current_mask.dtype,
                        device=current_mask.device,
                    ),
                ],
                dim=1,
            )

        suffix_embeds = embed_layer(suffix_ids)

        full_embeds = torch.cat([current_embeds, suffix_embeds], dim=1)
        full_mask = torch.cat([current_mask, suffix_mask], dim=1)

        masked_prefix_labels = torch.full(
            (1, current_embeds.shape[1]),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        full_labels = torch.cat(
            [masked_prefix_labels, suffix_labels],
            dim=1,
        )

        output = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )

        supervised_tokens = (full_labels != -100).sum().clamp_min(1)
        return output.loss, supervised_tokens

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

        weighted_losses = []
        token_counts = []

        for row_index in range(input_ids.shape[0]):
            row_loss, supervised_tokens = self._forward_single(
                input_ids=input_ids[row_index],
                attention_mask=attention_mask[row_index],
                labels=labels[row_index],
            )

            weighted_losses.append(
                row_loss * supervised_tokens.to(row_loss.dtype)
            )
            token_counts.append(supervised_tokens)

        total_weighted_loss = torch.stack(weighted_losses).sum()
        total_supervised_tokens = torch.stack(token_counts).sum()

        batch_loss = (
            total_weighted_loss / total_supervised_tokens.to(total_weighted_loss.dtype)
        )

        return CausalLMOutput(
            loss=batch_loss,
            logits=None,
        )

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """Inference: Prompt → latent pass 1 → ... → latent pass N → answer.

        At inference the prompt contains no <bot> tokens, so the recurrent
        latent steps are driven explicitly here (mirroring forward()): the
        prompt is embedded, then each of num_latent_steps passes feeds the
        previous pass's final hidden state back in as the next input
        embedding. The accumulated embeddings then seed self.model.generate()
        to produce the <answer>LABEL</answer> suffix.
        """
        kwargs.setdefault("pad_token_id", self.eot_token_id)
        kwargs.setdefault("eos_token_id", self.model.config.eos_token_id)
        kwargs.setdefault("do_sample", False)

        embed_layer = self.model.get_input_embeddings()
        current_embeds = embed_layer(input_ids)
        current_mask = attention_mask

        with torch.no_grad():
            for _ in range(self.num_latent_steps):
                outputs = self.model(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                # Last hidden state at final position = latent thought vector
                latent_hidden = outputs.hidden_states[-1][:, -1:, :]

                current_embeds = torch.cat([current_embeds, latent_hidden], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    torch.ones(
                        current_mask.shape[0], 1,
                        dtype=torch.long, device=current_mask.device,
                    ),
                ], dim=1)

        return self.model.generate(
            inputs_embeds=current_embeds,
            attention_mask=current_mask,
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

        # Keep a separate prompt-only sequence for generation-based validation.
        # The supervised input below contains prompt + target and must never be
        # passed directly to generate(), or the ground-truth target leaks in.
        prompt_ids = input_ids[: self.max_seq_length]
        prompt_attention_mask = torch.ones(len(prompt_ids), dtype=torch.long)
        prompt_pad_len = self.max_seq_length - len(prompt_ids)
        if prompt_pad_len > 0:
            prompt_ids = torch.cat([
                torch.full(
                    (prompt_pad_len,), tok.pad_token_id, dtype=torch.long
                ),
                prompt_ids,
            ])
            prompt_attention_mask = torch.cat([
                torch.zeros(prompt_pad_len, dtype=torch.long),
                prompt_attention_mask,
            ])

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

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "generation_input_ids": prompt_ids,
            "generation_attention_mask": prompt_attention_mask,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]
