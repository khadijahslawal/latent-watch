# Latent Watch
# Can Latent Reasoning Be Safety-Monitored? Necessity, Validity, and the Case for Custom Training

**BlueDot Impact -  Technical AI Safety Project Sprint**

**Author:** Khadija Shuaib

**Project Status:** Active; Stage 0 in progress

**Full working log:** [link]

---

## The problem

As reasoning migrates from token-level chain-of-thought into continuous latent space (COCONUT-style architectures), CoT monitoring doesn't just get less reliable. It becomes structurally impossible, since there's no longer a natural-language trace to read. This project asks whether that closing window matters in practice: **is safety-relevant signal (harmful intent) present and extractable from latent representations at all**, before it becomes unreadable by construction?

This builds directly on Chang et al. (arXiv:2606.01243), which showed latent vectors in COCONUT models are semantically meaningful and causally upstream of outputs, but did not test safety-relevant content. We ask whether that structure extends to safety.

## Research question

Reframed (with reviewer input) from a single probing question into a staged, falsifiable structure that protects against an uninformative null result:

1. **Do latent reasoning mechanisms learned on reasoning tasks generalize sufficiently out of distribution to support safety-relevant activation monitoring?**
2. *Conditional on (1):* When latent computation is causally involved in safety-relevant decisions, can its intermediate states predict harmful outputs beyond what's available from the input alone?

| Stage | Question |
|---|---|
| 0  | Is the latent pathway causally load-bearing on safety prompts, or unused? |
| 1  | Does the latent trajectory stay stable/non-degenerate off-distribution? |
| 2 - Monitorability | Can linear probes detect safety-relevant signal in the latents? |
| 3 - Temporal | Where in the trajectory does predictive signal emerge? |

Stage 0 is a requirement: The checkpoint matrix we use comes from a paper (Dilgren & Wiegreffe, arXiv:2604.04902) that independently found latent tokens are often causally inert on logical-reasoning tasks. Skipping straight to probing would risk detecting signal in latents that aren't actually doing anything.

## Why we pivoted away from custom training (and why that pivot itself produced a finding)

- The original design targeted a custom-trained Llama-3.1-8B COCONUT model, replicating Chang et al.'s training curriculum before extending it with a safety-probing layer. This failed during the CoT-replication stage - single-epoch training exceeded 14 hours, and a GPU disk failure during epoch 2 lost the run after ~$45 of compute.

- We pivoted to Dilgren & Wiegreffe's released checkpoint matrix (`connordilgren/are-lrms-easily-interpretable`, MIT-licensed):
  -  GPT-2 and Llama-3.2-1B, across three datasets (GSM8K, ProntoQA, ProsQA) and six reasoning-format conditions (No-CoT, CoT, COCONUT, CODI, multimode variants).
  -  This let us run a controlled, cross-condition Stage 0 investigation at near-zero marginal compute cost via Colab: a methodological upgrade over the original single-checkpoint plan, not just a fallback.

## What we've found so far (preliminary - see full log for caveats and sample sizes)

Running `early_stopping.run` (a necessity-ablation tool from the checkpoint repo) on BeaverTails safety prompts, across two COCONUT checkpoints:

- **GPT-2 / ProntoQA / COCONUT** (n=150):
  - 96.7% of prompts produced the identical answer (`'False'`), regardless of content: a violent threat and a benign small-talk question got the same response.
  -  99.3% committed to that answer using **zero** latent reasoning steps.
    
- **Llama-3.2-1B / ProntoQA / COCONUT** (n=150): same phenomenon, mirrored polarity (98.7% `'True'`). Confirms the collapse is **not specific to small models**. It persists across an 8× parameter-scale increase.

**Preliminary contrast with CoT** (same base model, same prompts): 

Where COCONUT preserves output format but appears to ignore prompt content, the CoT checkpoint shows the opposite surface symptom. It frequently breaks the expected output format entirely (95% of generations didn't produce a parseable delimiter) while what *does* get through looks more content-engaged. This is not yet a confirmed finding, the non-skipped CoT sample was small and non-random, and a full-coverage rerun (logging all outputs regardless of format compliance) is in progress. **Do not cite this specific contrast externally without an update from the running log.**

## Why this matters for the case to fund the original (8B, custom-trained) question

The pretrained-checkpoint pivot was meant to be a low-cost way to make progress on the general research question while compute constraints were resolved. It has done more than that: it has surfaced a **concrete, load-bearing obstacle** where generalization/necessity failure under distribution shift that any safety-monitoring approach built on these architectures will need to address before Stage 2 (probing) can produce interpretable results.

This is useful evidence for a funding request in two ways:

1. It shows the staged research design is working as intended by catching a real validity problem before wasted probe-training investment, rather than assuming it away.
2. It suggests the available pretrained checkpoints failure may be that they're trained on narrow, single-domain tasks (GSM8K/ProntoQA/ProsQA) with small, closed answer spaces which is precisely the kind of limitation that a **custom-trained model on a broader, safety-adjacent curriculum** (the original 8B proposal) would be positioned to test directly. Compute to train (or fine-tune) a model with broader distributional coverage is now a better-justified ask than it was at the project's outset, because we can point to a specific, documented failure mode it would let us investigate.

## Methodology notes worth knowing before reusing this pipeline

The checkpoint repo's tooling was built exclusively around GSM8K/ProntoQA/ProsQA (all of which have gold reasoning traces and canonical short answers). Adapting it for safety prompts (no gold trace, no canonical answer) required several fixes, documented in full in the research log — most notably a schema workaround for an off-by-one bug in the dataset loader's internal consistency check, and recognizing that the `###`-delimiter answer-extraction method behaves very differently (and non-comparably) across reasoning formats. Worth reading before extending this work.

## Links

- Checkpoint repo: `connordilgren/are-lrms-easily-interpretable` 
- Checkpoint paper: Dilgren & Wiegreffe, arXiv:2604.04902
- Foundational paper: Chang et al., arXiv:2606.01243
- Full research log (methodology, all experiments, change history): **[TODO: paste google drive link]**

## Current next steps

1. Finish full-coverage CoT raw-output capture and resolve whether the CoT/COCONUT contrast is real or a small-sample artifact.
2. Rule out generic small-model decoding degeneration (repetition loops) as a confound in the CoT results.
3. Complete the three-way format comparison (COCONUT / CoT / No-CoT) on matched base model + dataset.
4. Test a GSM8K-trained COCONUT checkpoint to check whether collapse is specific to narrow binary-answer domains like ProntoQA.

## Data & Licensing

Uses BeaverTails (CC BY-NC 4.0) for prompt data, and checkpoints/tooling from Dilgren & Wiegreffe's `are-lrms-easily-interpretable` (MIT) for the underlying experiments. Non-commercial research use only. Full citations and license terms in [`ATTRIBUTION.md`](./ATTRIBUTION.md).