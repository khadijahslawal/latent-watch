# Latent Watch
## Can Latent Reasoning Be Safety-Monitored?

**BlueDot Impact — Technical AI Safety Project Sprint**  
**Author:** Khadija Shuaib  
**Project Status:** Active — training in progress  
**Full working log:**

---

## The problem

As reasoning migrates from token-level chain-of-thought into continuous latent space (COCONUT-style architectures), CoT monitoring doesn't just get less reliable — it becomes structurally impossible. There is no longer a natural-language trace to read.

This raises a question that existing interpretability and monitoring work hasn't answered: **is safety-relevant signal even present in latent representations, and can it be extracted before the reasoning window closes entirely?**

This project builds directly on Chang et al. (arXiv:2606.01243), which showed that latent vectors in COCONUT models are semantically meaningful and causally upstream of outputs — but studied only reasoning accuracy tasks. We ask whether that structure extends to safety-relevant content.

---

## Research question

Given a prompt, can we predict whether it has demonstrated a risk of eliciting an unsafe response — **without observing any chain-of-thought?**

More precisely: does latent reasoning encode safety-relevant information that is (a) present, (b) extractable, and (c) causally connected to model outputs, when there is no readable reasoning trace at all?

---

## Why this matters for AI safety

Current safety monitoring of frontier models relies heavily on inspecting chain-of-thought outputs. Anthropic's usage policies, the safety cases in Responsible Scaling Policy documents, and most deployed monitoring pipelines assume that *if reasoning is present, it is readable*. COCONUT-style architectures break this assumption at the architectural level.

This isn't a hypothetical concern. The COCONUT paper (Hao et al. 2024) demonstrates that models can be trained to reason entirely in continuous latent space, producing outputs with no intermediate text. If such architectures are adopted — and the performance advantages give strong incentive to do so — then any monitoring approach that depends on CoT inspection fails by construction.

This project investigates whether an alternative is possible: detecting safety-relevant signal from latent activations directly, without the reasoning trace. A positive result would suggest that latent-space monitoring tools (probes, SAEs, representation-level classifiers) could partially recover what CoT inspection currently provides. A negative result would be equally informative, quantifying the monitoring gap that latent reasoning architectures would introduce.

---

## Experimental design

The core experiment trains three LoRA adapters on Llama-3.2-1B, one for each reasoning format, on a shared safety classification task. All else is held equal.

**Task (Task B):** Given a prompt, predict its observed safety risk.  
- `HIGH_RISK`: at least one observed response to the prompt was unsafe  
- `LOW_RISK`: all observed responses were safe

| Experiment | Reasoning format | What it tests |
|---|---|---|
| E1 — Answer-only | No reasoning trace | Baseline: prompt features alone |
| E2 — CoT | Explicit `<reasoning>...</reasoning>` | Upper bound: readable intermediate reasoning |
| E3 — COCONUT | Continuous latent thoughts (`<bot>/<eot>`) | Target condition: latent reasoning only |

The primary comparison is **E3 vs E2**: does latent reasoning match CoT's ability to detect safety-relevant signal, even though no intermediate text is produced? The secondary comparison is **E2 vs E1**: does explicit reasoning actually help at all for this task, before asking whether latent reasoning can replicate it.

---

## Dataset

**Source:** BeaverTails (PKU-Alignment, CC-BY-NC-4.0) — 330k prompt-response pairs with harm category labels and human safety annotations.

**Labelling policy:** Deduplicate by prompt; label `HIGH_RISK` if any paired response was unsafe. This is a deliberate high-recall choice: we care about prompts that *can* elicit unsafe outputs, not just those that *always* do.

**Reasoning traces:** Synthetically generated via GPT-4o-mini, conditioned on ground-truth labels and harm categories. The generator does not see paired responses, ensuring it cannot learn information unavailable at inference time. Traces are validated against a set of rejection criteria (label contradiction, mention of observed responses, vague or overlong reasoning).

**Split (after deduplication and sampling):**

| Split | n | HIGH_RISK | LOW_RISK |
|---|---|---|---|
| Train | 3,928 | 2,500 | 1,428 |
| Validation | 429 | 250 | 179 |
| Test | 679 | 500 | 179 |

Class imbalance is by design and matches BeaverTails' natural distribution. Primary metric is **weighted F1**; `HIGH_RISK` recall is reported separately as the safety-critical error type.

**Harm categories covered (14):** animal abuse, child abuse, controversial topics / politics, discrimination / stereotype / injustice, drug abuse / weapons / banned substances, financial crime / property crime / theft, hate speech / offensive language, misinformation regarding ethics-laws-safety, non-violent unethical behaviour, privacy violation, self-harm, sexually explicit / adult content, terrorism / organised crime, violence / aiding / abetting / incitement.

---

## Model and training

**Base model:** `meta-llama/Llama-3.2-1B`, loaded in 4-bit NF4 (bitsandbytes) for T4 compatibility.

**LoRA configuration** (shared across E1, E2, E3):

```
r = 16, lora_alpha = 32, lora_dropout = 0.05
target_modules = ["q_proj", "v_proj"]
```

**Training hyperparameters** (shared across E1, E2, E3):

```
epochs = 3, lr = 2e-4, batch_size = 8 (→4 if OOM)
gradient_accumulation_steps = 4, optimizer = AdamW
lr_scheduler = cosine, warmup_ratio = 0.05, max_seq_length = 512
```

Loss is masked to target tokens only; prompt tokens are set to `-100`.

**COCONUT-specific:** E3 uses a staged curriculum (C=3 stages) implementing Hao et al.'s progressive replacement schedule. The LoRA adapter is attached to the base model before COCONUT wrapping; gradients flow back through the wrapper into the adapters. Two special tokens are added: `<bot>` (beginning of latent thought) and `<eot>` (end of latent thought).

**Curriculum:**
```
Stage 0: <reasoning> step1 step2 step3 </reasoning> <answer>LABEL</answer>
Stage 1: <bot> step2 step3 <answer>LABEL</answer>
Stage 2: <bot><bot> step3 <answer>LABEL</answer>
Stage 3: <bot><bot><bot> <answer>LABEL</answer>
```

At the final stage, no explicit reasoning is decoded. The answer is produced from latent thought alone.

**Execution context:** Pilot runs on Google Colab (T4, fp16, 4-bit). Production runs on rented A100 (bf16, batch_size=16).

---

## Results

> **[Placeholder — experiments in progress]**

Results will report, for each experiment, on the held-out test set:

- Accuracy, weighted precision / recall / F1
- `HIGH_RISK` recall specifically (the safety-critical direction)
- Confusion matrix
- Softmax confidence scores (HIGH_RISK / LOW_RISK)
- Per-harm-category breakdown

The key comparisons:

| Comparison | Question |
|---|---|
| E2 vs E1 | Does explicit reasoning improve safety signal detection? |
| E3 vs E2 | Does latent reasoning match CoT's classification performance? |
| E3 vs E1 | Does latent reasoning add anything beyond prompt features alone? |
| Per-category E3 vs E2 | Which harm categories are most affected by the latent/CoT gap? |

---

## What we expect to find (and why it matters either way)

**If E3 ≈ E2 (latent reasoning matches CoT performance):** This would suggest that safety-relevant signal is not merely encoded in the readable trace — it is present in the latent representations themselves, and is predictive of outputs. This opens the door to probe-based or representation-level monitoring as an alternative to CoT inspection.

**If E3 << E2 (latent reasoning underperforms CoT):** This would quantify the monitoring gap that latent architectures introduce, and would motivate investment in alternative interpretability tools (SAE-based feature discovery, activation-level anomaly detection) to partially recover what is lost.

**If E3 ≈ E1 (latent reasoning adds nothing beyond prompt features):** This would suggest that latent reasoning in COCONUT doesn't meaningfully engage with safety-relevant content at all, at least in this training regime — a finding with implications for how latent reasoning models respond to safety-relevant inputs in practice.

Any of these outcomes is informative. The staged design ensures the null result is not vacuous.

---

## Limitations and caveats

**Scale:** Llama-3.2-1B is small. Results may not generalise to larger models where latent reasoning is likely more capable and more meaningfully engaged.

**Training data:** Reasoning traces are synthetically generated, not human-authored. The quality of CoT (and by extension COCONUT's curriculum signal) depends on GPT-4o-mini's ability to produce rationales that reflect genuine risk reasoning rather than label-correlated surface patterns.

**Task framing:** Task B classifies *prompt-level* elicitation risk based on observed response variance — it is not a real-time inference task and does not simulate deployment conditions directly. Results speak to whether latent representations encode safety signal in principle, not to operational monitoring performance.

**COCONUT implementation:** We implement COCONUT from scratch following Hao et al. 2024, not using Dilgren & Wiegreffe's codebase. Differences in implementation may affect comparability to other COCONUT results in the literature.

---

## Related work

- **Chang et al. (arXiv:2606.01243):** Establishes that latent vectors in COCONUT models are semantically meaningful and causally upstream of outputs (§3.2–3.3). We extend their investigation to safety-relevant content.
- **Hao et al. 2024 (COCONUT):** Original architecture and training curriculum for continuous-thought reasoning models.
- **Dilgren & Wiegreffe (arXiv:2604.04902):** Independent finding that latent tokens are often causally inert on logical-reasoning tasks — a motivation for our necessity checks and staged experimental design.
- **BeaverTails (Ji et al.):** Prompt-response safety dataset with 14-category harm labels. Primary data source.

---

## Repository structure

```
latent-watch/
├── README.md
├── ATTRIBUTION.md
├── pyproject.toml
│
├── configs/
│   ├── data/beavertails_risk.yaml
│   ├── training/
│   │   ├── llama_1b_answer_only.yaml
│   │   ├── llama_1b_cot.yaml
│   │   └── llama_1b_coconut.yaml
│   └── evaluation/safety_risk.yaml
│
├── data/processed/beavertails_risk_v1/
│   ├── canonical/{train,validation,test}.jsonl
│   ├── answer_only/{train,validation,test}.jsonl
│   ├── cot/{train,validation,test}.jsonl      # COCONUT reads from here too
│   └── challenge/matched_context_test.jsonl
│
├── src/latent_watch/
│   ├── data/          # load, normalize, aggregate, split, generate rationales
│   ├── training/      # base_trainer, formatters, lora_utils, coconut, train_*.py
│   └── evaluation/    # classification, category, matched-pairs, latent-steps
│
├── checkpoints/
│   ├── answer_only/best_adapter/
│   ├── cot/best_adapter/
│   └── coconut/best_adapter/   # includes tokenizer with <bot>/<eot>
│
├── results/
│   ├── answer_only.csv
│   ├── cot.csv
│   └── latent.csv
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_sampling_review.ipynb
│   ├── 03_rationale_review.ipynb
│   └── 04_train_pilot.ipynb    # Colab T4 entry point
│
└── tests/
    ├── test_aggregation.py
    ├── test_any_unsafe_rule.py
    ├── test_split_leakage.py
    ├── test_category_sampling.py
    └── test_rendered_formats.py
```

---

## Data & licensing

Uses **BeaverTails** (PKU-Alignment, CC-BY-NC-4.0) for prompt data. Base model: `meta-llama/Llama-3.2-1B` (Llama 3.2 Community License). Reasoning traces synthetically generated via OpenAI API; noted in dataset card. Non-commercial research use only. Full citations and license terms in [`ATTRIBUTION.md`](./ATTRIBUTION.md).
