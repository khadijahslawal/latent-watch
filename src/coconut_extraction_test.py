# coconut_extraction.py
# Utilities for extracting continuous thought vectors from COCONUT models.
# Generated from coconut_step1_validation.ipynb

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "coconut"))

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss
from collections import defaultdict
from coconut import Coconut, Outputs

from transformers import AutoTokenizer, GPT2LMHeadModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CoconutExtractor(Coconut):
    """
    Captures the continuous thought vector at each latent reasoning step.
    Access self.thought_vectors after forward(): list of {batch_idx: tensor(hidden_dim,)}
    """

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        self.thought_vectors = []
        logits = []
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists])

        latent_lists_max_len_at_step = []

        if max_n_latents > 0:
          for i in range(max_n_latents):
            max_len_at_i = max([l[i] for l in latent_lists if len(l) > i])
            latent_lists_max_len_at_step.append(max_len_at_i)


        next_compute_range = (0, input_ids.shape[1])

        if max_n_latents > 0:
          next_compute_range = (0, latent_lists_max_len_at_step[0] + 1)

        inputs_embeds = self.embedding(input_ids)


        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]
            logits.append(outputs.logits)
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1] if pass_idx + 1 >= max_n_latents
                    else latent_lists_max_len_at_step[pass_idx + 1] + 1
                ),
            )
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            step_thoughts = {}
            for batch_idx, token_idx in filling_indices:
                vec = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :].detach().cpu()
                step_thoughts[batch_idx] = vec
            self.thought_vectors.append(step_thoughts)
            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
            inputs_embeds = torch.stack([
                torch.stack(tensor_list[batch_idx])
                for batch_idx in range(inputs_embeds.shape[0])
            ])
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=(
                [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                 for k, v in kv_cache] if kv_cache else None
            ),
            output_hidden_states=True,
        )
        logits.append(outputs.logits)
        self.gen_forward_cnt += max_n_latents + 1
        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)


def tokenize_batch(texts, tokenizer, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels         = input_ids.clone()
    position_ids   = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    position_ids   = position_ids.expand(input_ids.shape[0], -1)
    return input_ids, attention_mask, labels, position_ids


def extract_thought_vectors(model, dataset, tokenizer, device, batch_size=4):
    model.eval()
    thought_vecs = defaultdict(list)
    for i in range(0, len(dataset), batch_size):
        batch_texts = dataset[i:i+batch_size]
        input_ids, attn_mask, labels, pos_ids = tokenize_batch(batch_texts, tokenizer, device)
        with torch.no_grad():
            model(input_ids, attn_mask, labels, pos_ids)
        for step_idx, step_dict in enumerate(model.thought_vectors):
            for batch_idx in range(len(batch_texts)):
                if batch_idx in step_dict:
                    thought_vecs[step_idx].append(step_dict[batch_idx])
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i+len(batch_texts)}/{len(dataset)} examples")
    return {k: torch.stack(v) for k, v in thought_vecs.items()}


def linear_CKA(X, Y):
    assert X.shape[0] == Y.shape[0]
    X = X.double() - X.double().mean(dim=0)
    Y = Y.double() - Y.double().mean(dim=0)
    return (torch.norm(Y.T @ X, "fro") ** 2 /
            (torch.norm(X.T @ X, "fro") * torch.norm(Y.T @ Y, "fro"))).item()


def mean_cosine_similarity(X, Y):
    assert X.shape == Y.shape
    return F.cosine_similarity(X.float(), Y.float(), dim=-1).mean().item()


def compute_alignment_metrics(coconut_vecs, cot_vecs):
    common_steps = sorted(set(coconut_vecs.keys()) & set(cot_vecs.keys()))
    per_step_cka, per_step_cosine = {}, {}
    for step in common_steps:
        X, Y = coconut_vecs[step], cot_vecs[step]
        if X.shape[1] != Y.shape[1]:
            proj = torch.nn.Linear(Y.shape[1], X.shape[1], bias=False)
            torch.nn.init.orthogonal_(proj.weight)
            Y = proj(Y.float()).detach()


        per_step_cosine[step] = mean_cosine_similarity(X, Y)

        if X.shape[0] < 2:
            print(f"  Step {step}: skipping CKA (only {X.shape[0]} example)")
            continue
        per_step_cka[step] = linear_CKA(X, Y)

    mean_cka = np.mean(list(per_step_cka.values())) if per_step_cka else float("nan")
    mean_cosine = np.mean(list(per_step_cosine.values()))
    return {
        "per_step_cka": per_step_cka,
        "per_step_cosine": per_step_cosine,
        "mean_cka": mean_cka,
        "mean_cosine": mean_cosine,
    }

def make_coconut_input(question: str, n_latent_steps: int, answer: str = "") -> str:
    """
    Format a question with COCONUT latent placeholders.

    n_latent_steps corresponds to the number of reasoning steps.
    Each step gets one <lat> token (K=1 per step).
    """
    latent_block = " <bot> <lat> <eot>" * n_latent_steps
    return f"Q: {question}{latent_block} A: {answer}"


def main():
    MODEL_NAME = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Add the three special tokens COCONUT requires
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>", "<lat>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")

    START_LATENT_ID = tokenizer.convert_tokens_to_ids("<bot>")
    END_LATENT_ID   = tokenizer.convert_tokens_to_ids("<eot>")
    LATENT_TOKEN_ID = tokenizer.convert_tokens_to_ids("<lat>")

    base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    base_model.resize_token_embeddings(len(tokenizer))  # accommodate the 3 new tokens
    base_model = base_model.to(DEVICE)

    # Example math problem with 3 reasoning steps
    example = make_coconut_input(
        question="Janet has 3 apples. She buys 5 more then gives 2 away. How many does she have?",
        n_latent_steps=3,
        answer="6",
    )

    print(example)

    # Tokenize batch
    input_ids, attention_mask, labels, position_ids = tokenize_batch(example, tokenizer, DEVICE)
    
    print(f"input_ids shape:      {input_ids.shape}   # [batch=2, seq_len]")
    print(f"attention_mask shape: {attention_mask.shape}")

    # Verify latent tokens are present in the encoded input
    lat_count = (input_ids == LATENT_TOKEN_ID).sum().item()
    print(f"<lat> tokens in batch: {lat_count}  (expected: 2+4 = 6)")

    ## Run forward pass and verify extraction
    extractor = CoconutExtractor(base_causallm=base_model,
                                latent_token_id=LATENT_TOKEN_ID,
                                start_latent_id=START_LATENT_ID,
                                end_latent_id=END_LATENT_ID,
                                    eos_token_id=tokenizer.eos_token_id,
    )
    extractor.eval()
    
    with torch.no_grad():
        outputs = extractor(input_ids, attention_mask, labels, position_ids)
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}   # [batch, seq_len, vocab_size]")
    print(f"\nCaptured thought vectors at {len(extractor.thought_vectors)} latent steps")

if __name__ == "__main__":
    main()
