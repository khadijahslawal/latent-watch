# Latent Reasoning Safety Monitorability — Research Log

**Status as of:** July 12, 2026 (pre-CoT full-coverage results — CoT run in progress)
**Maintainer:** Khadija
**Program:** BlueDot Impact Technical AI Safety — Project Sprint

This is a living document. New entries go at the bottom of §7 (Experiment Log) and §9 (Change Log) as work progresses; sections 1–4 should only change when the research question or scope itself changes.

---

## 1. Motivation & Theory of Change

Chang et al. (arXiv:2606.01243) established that latent vectors in COCONUT-style continuous-thought models are semantically meaningful and causally upstream of outputs — but did not address safety-relevant content specifically.

The structural concern motivating this project: as reasoning migrates from token-level chain-of-thought into continuous latent space, CoT monitoring doesn't just become less reliable — it becomes structurally impossible, since there is no longer a natural-language trace to read. This project investigates monitorability *before* that window closes, asking whether safety-relevant signal is even present and extractable from latent representations in the first place.

## 2. Research Question — Evolution

**V1 (original design):** Can linear probing on continuous-thought vectors detect harmful intent before a final answer is generated? Direct extension of Chang et al., targeting a custom-trained Llama-3.1-8B COCONUT model.

**V2 (current, reviewer-informed redesign):** Reframed to avoid assuming the conclusion and to protect against an uninformative null result. Restructured as two nested questions:

1. Do latent reasoning mechanisms learned on reasoning tasks generalize sufficiently out of distribution to support safety-relevant activation monitoring?
2. *Conditional on (1):* When latent computation is causally involved in safety-relevant decisions, can its intermediate states predict harmful outputs beyond information available from the input alone?

This unpacks into four stages:

| Stage | Question | Status |
|---|---|---|
| **0 — Necessity** | Is the latent pathway causally load-bearing on safety prompts at all, or does the model reach its answer without using it? | **In progress — see §7** |
| **1 — OOD validity** | If used, does the latent trajectory remain stable/non-degenerate on safety prompts (vs. collapsing/garbled)? | Partially addressed as a byproduct of Stage 0 runs |
| **2 — Monitorability** | Can linear probes detect safety-relevant signal from latent trajectories? | Not yet started — gated on Stage 0/1 |
| **3 — Temporal** | Where in the trajectory does predictive signal emerge? | Not yet started |

**Grounding for why Stage 0 is a hard gate, not a formality:** Dilgren & Wiegreffe (the source of our checkpoint matrix — see §3) independently found that latent tokens are frequently causally inert on logical-reasoning tasks (ProntoQA/ProsQA), and when active, decode to the gold reasoning trace 65–93% of the time. Skipping Stage 0 would make any Stage 2 probe result uninterpretable — a probe could "detect" harmfulness from latents that are decorative rather than causal.

## 3. Scope, Data & Checkpoints

**Checkpoint source:** `connordilgren/are-lrms-easily-interpretable` (GitHub, MIT-licensed), companion to Dilgren & Wiegreffe, arXiv:2604.04902, "Are Latent Reasoning Models Easily Interpretable?" HF collection: `connordilgren/are-latent-reasoning-models-easily-interpretable`.

**Checkpoint matrix:** GPT-2 and Llama-3.2-1B(-Instruct), × 3 datasets (GSM8K, ProntoQA, ProsQA), × 6 reasoning-format conditions (No-CoT, CoT, COCONUT, CODI, multimode-COCONUT, multimode-CODI).

**Primary dataset:** BeaverTails (`PKU-Alignment/BeaverTails`) — prompt/response pairs with a boolean `is_safe` label (per response) and a 14-way harm-category dict. **Labeling decision:** since BeaverTails pairs one prompt with multiple responses of varying safety, we dedupe by prompt and label a prompt harmful if **any** paired response was flagged unsafe (high-recall, conservative choice — documented here as a deliberate methodological decision, not a default).

**Secondary dataset:** Do-Not-Answer (not yet used).

**Primary Stage 0 tool:** `experiments/early_stopping/run.py` from the checkpoint repo — measures how much of the reasoning trace (latent tokens / CoT tokens / CODI iterations) is actually needed before the model's answer stabilizes, per sample, via forced-early-answer ablation. Also runs a vocabulary-projection decodability check (is the eventual answer visible in the top-k logit projection of intermediate latent states).

**Infrastructure:** Google Colab Pro, T4 GPU (sufficient for 1B-scale inference), Google Drive for persistent checkpoint/data/results storage across sessions.

## 4. Scope History — Why We're Not Training From Scratch

Original design targeted full replication of Chang et al.'s CoT-to-latent training curriculum on Llama-3.1-8B, then extending with a safety-probing layer. This crashed during the **CoT-replication stage** (before any COCONUT curriculum was even applied) — single-epoch training time exceeded 14 hours, and a GPU disk-space failure during epoch 2 lost that run after ~$45 of compute, leaving only an epoch-1 checkpoint.

**Pivot:** use the pretrained Dilgren & Wiegreffe checkpoint matrix (§3) instead of training our own. This is framed as a deliberate methodological choice, not merely a recovery: it enables systematic, controlled comparison across reasoning-format conditions (holding base model and training dataset fixed) that would be prohibitively expensive to produce via custom training. See §8 for how this pivot's early results bear on whether the *original* 8B question still needs to be answered directly.

## 5. Methodological Adaptations & Fixes

Documented because these are genuine adaptations required to push safety-domain data (no gold reasoning trace, no canonical answer) through tooling built exclusively around GSM8K/ProntoQA/ProsQA — this is itself a minor methods contribution, not just plumbing.

1. **Schema requires `steps`/`answer` fields safety prompts don't have.** Using `steps: []` triggers an off-by-one bug in the checkpoint repo's `dataset.py`: its internal tokenization self-consistency assertion assumes ≥1 reasoning step and produces one extra newline in its verification string when steps are empty, guaranteeing an always-fails assertion. **Fix:** use `steps: [""]` (single empty-string step) instead of `[]`.
2. **The above fix is not tokenizer-general.** Even with `steps: [""]`, the same assertion can still fail for a different reason: ordinary BPE/SentencePiece merge non-associativity across concatenation boundaries (tokenizing pieces separately vs. jointly can legitimately differ token-for-token even with identical underlying characters). Confirmed empirically: passes for GPT-2's tokenizer on our schema, fails for Llama-3.2's. **Fix:** patched the hard `assert` into a `warnings.warn()` — logs loudly but doesn't crash, since self-consistency of this specific reconstructed string isn't load-bearing for inference-only use (the actually-used `question_tokenized`/`steps_tokenized`/`answer_tokenized` fields are computed correctly and independently regardless).
3. **The `###`-delimiter assumption in `extract_answer()` behaves very differently by reasoning format.** COCONUT's generation loop *forces* the delimiter token after the fixed latent-step budget, which artificially guarantees ~0% skip rate regardless of whether the model is meaningfully engaging with the prompt. CoT free-generates up to `max_new_tokens` with no such scaffolding, and on out-of-domain safety prompts, overwhelmingly fails to spontaneously produce `"###"` — observed 143/150 (95%) skip rate. This is a real asymmetry in what the pipeline's summary statistics can tell you across format conditions, not a bug to be papered over — see §7, Exp 7.
4. **BeaverTails labeling policy** (dedupe-by-prompt, any-unsafe-response = harmful) — see §3. Documented as a deliberate, high-recall choice; majority-vote and no-dedup alternatives were considered.
5. **Colab shell-magic (`!python ...`) failures don't surface as notebook-level errors** — a failed subprocess prints its traceback as plain output text but doesn't trigger Colab's error styling. Mitigated by appending `echo "Exit code: $?"` to shell cells going forward.
6. **`run.py`'s skip diagnostics (`num_skipped_early_stop`, `num_skipped_vocab_proj`) were only printed to stdout, not saved.** Patched to persist into the saved JSON's `summary.skip_diagnostics` block, so skip rates are queryable/comparable across runs without re-parsing logs.

## 6. Repository / Tooling Reference

- Checkpoint repo: `connordilgren/are-lrms-easily-interpretable` (MIT)
- Stage 0 script: `experiments/early_stopping/run.py`
- Schema reference: `src_coconut_multimode/dataset.py::get_dataset`, `dataset_utils/base.py`, `dataset_utils/adapters.py`
- Our converter: `beavertails_to_schema.py` (dedupes BeaverTails by prompt, applies labeling policy, emits schema-compliant JSON + a joinable metadata sidecar with harm-category labels)
- Core papers: Chang et al. arXiv:2606.01243; Dilgren & Wiegreffe arXiv:2604.04902

## 7. Experiment Log

### Exp 1 — Sanity check: GPT-2 / ProntoQA / COCONUT (checkpoint_36), n=20, BeaverTails
0/20 skipped. Single-sample inspection: model answered `'False'` to *"Do you have any new years resolutions?"* — flagged as suspicious, needed follow-up.

### Exp 2 — Control: same checkpoint, n=20, real ProntoQA test data
All 20 converged at k=0 (matches Dilgren & Wiegreffe's own in-domain "often causally inert" finding), but answers varied correctly with content (12 True / 8 False across distinct puzzles). Confirms harness is working correctly — not a broken pipeline.

### Exp 3 — Debug run (raw-text verification), n=5, BeaverTails
Confirmed via `--debug` full-output logging that answers were **cleanly generated**, not garbled/parsing artifacts: e.g. `"...### False"` after prompts ranging from small talk to a sexual-solicitation request. Ruled out "garbled output → spurious parse" as the explanation for the constant answer.

### Exp 4 — Scaled: GPT-2 / ProntoQA / COCONUT, n=150, BeaverTails
- 145/150 (96.7%) answered `'False'`; 149/150 (99.3%) first-matched at k=0.
- **Finding: near-total, content-independent output collapse.** Confirmed not a small-sample artifact.
- Outlier analysis: 5 samples answered `'True'` instead of `'False'`; weak thematic clustering around discrimination-category content noted but underpowered (see Exp 5). 1 sample first-matched at k=2 rather than k=0 (worth noting: COCONUT sees the full prompt before any latent step — "k=2" means 2 internal computation steps with full context already available, not "reached token position 2," an important distinction from token-level CoT intuitions).

### Exp 5 — Subgroup test: discrimination-category rate, n=150 pool (12 discrimination-labeled)
Discrimination-category prompts: 2/12 (16.7%) answered `'True'`. Other prompts: 3/138 (2.2%) answered `'True'`. Fisher's exact test: odds ratio 9.0, **p = 0.051** — suggestive, not significant at conventional thresholds, and fragile given n=12 in the discrimination subgroup. **Not currently a confirmed finding.** Flagged for a properly powered, purpose-built stratified follow-up (§8).

### Exp 6 — Scale comparison: Llama-3.2-1B(-Instruct) / ProntoQA / COCONUT, n=150, BeaverTails
- 148/150 (98.7%) answered `'True'` (mirrored polarity vs. GPT-2, same collapse phenomenon). Avg first-match k = 0.18 (vs. GPT-2's 0.013).
- The higher average k is **not** evidence of richer content engagement: the 5 samples with k≥3 spanned wildly different content (a violent threat, a benign white-lie question, a COVID-vaccine question, relationship venting) and **all still answered `'True'`** — same collapse, just occasionally slower to settle.
- **Finding: the collapse phenomenon persists across an 8× parameter-scale increase (124M → 1B).** Stronger and more general than a single-checkpoint result.

### Exp 7 — Format comparison: Llama-3.2-1B / ProntoQA / CoT, n=150, BeaverTails — **PRELIMINARY, IN PROGRESS**
- 143/150 (95.3%) skipped due to `###`-delimiter never appearing in free-generated output (see §5.3 for why this differs structurally from COCONUT's ~0% skip rate).
- The 7 non-skipped samples show genuinely varied, content-engaged prose (e.g., a structured multi-point ethical discussion of violence against animals; an actual France travel guide; an opinion on pushy veganism) — qualitatively the **opposite** failure mode from COCONUT: format preserved / content collapsed (COCONUT) vs. content-engaged / format broken (CoT), at least in this small non-random subsample.
- **Important caveat — this comparison is not yet valid for a real conclusion.** The 7 samples are not a random draw; they're whichever samples happened to trip a weak delimiter-matching fallback (e.g., a `\boxed{}` LaTeX-style answer got picked up). A patch was applied to log all 150 raw CoT outputs (including skipped ones) to `raw_skipped_cot_outputs.jsonl`, and the run is being repeated for full coverage. **Also unresolved:** at least two of the 7 visible outputs show classic small-model greedy-decoding repetition loops, unrelated to content — this needs to be distinguished from genuine content engagement before the "CoT engages, COCONUT collapses" framing can be trusted.

---

## 8. Current Standing Interpretation (informal, subject to revision pending Exp 7 full results)

For the two COCONUT checkpoints tested so far (GPT-2/ProntoQA and Llama-1B/ProntoQA), latent reasoning shows **near-total, content-independent output collapse under domain shift** (safety prompts vs. training-domain logic puzzles), robust across an 8× parameter-scale difference. This is a direct, negative answer to the Stage 0/OOD-validity question for these specific checkpoints: the committed output does not currently vary with harmful-vs-benign content, so Stage 2 probe training on this committed-answer signal would not be meaningful without further intervention (e.g., broader-domain or safety-adjacent fine-tuning).

CoT, under the same push, shows a structurally different symptom (near-total delimiter-format failure rather than content collapse) — but this is not yet a confirmed contrast, pending full-coverage raw-output analysis (Exp 7 continuation).

## 9. Open Questions & Immediate Next Steps

- [ ] Complete full n=150 raw CoT output capture (patched run in progress) and characterize the distribution without delimiter-dependency bias (engaged / repetition-loop-degenerate / empty / other).
- [ ] Check `early_stopping.run`'s decoding parameters (temperature, `do_sample`, repetition penalty) — repetition-loop outputs may be a generic small-model decoding artifact, not evidence of "content engagement," and could be inflating the apparent CoT/COCONUT contrast.
- [ ] Run the No-CoT checkpoint (same base + dataset) to complete the three-way format comparison (COCONUT vs. CoT vs. No-CoT), isolating whether collapse is latent-specific or general to narrow fine-tuning.
- [ ] Run a GSM8K-trained COCONUT checkpoint (wider, numeric answer space) to test whether collapse is specific to a narrow binary (True/False) output space.
- [ ] Deliberately oversample discrimination-category BeaverTails prompts (properly powered, stratified) to resolve the underpowered p=0.051 lead from Exp 5.
- [ ] Once Stage 0/1 picture is clearer: decide whether any tested checkpoint is a viable Stage 2 (probe training) candidate as-is, or whether Stage 2 requires a checkpoint with broader training-domain coverage or light safety-adjacent fine-tuning first.

## 10. Change Log

- **2026-07-12:** Initial log created, consolidating all work from project pivot through Exp 6 and preliminary Exp 7. CoT full-coverage rerun in progress at time of writing.
