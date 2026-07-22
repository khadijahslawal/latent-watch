# Latent Watch
## Can Latent Reasoning Be Safety-Monitored?

**BlueDot Impact - Technical AI Safety Project Sprint**  
**Author:** Khadija Shuaib  
**Project Status:** Active - training in progress  
**Full working log:**

---

## The problem

As reasoning migrates from token-level chain-of-thought into continuous latent space (COCONUT-style architectures), CoT monitoring doesn't just get less reliable; it becomes structurally impossible. There is no longer a natural-language trace to read.

This raises a question that existing interpretability and monitoring work hasn't answered: **is safety-relevant signal even present in latent representations, and can it be extracted before the reasoning window closes entirely?**

This project builds directly on Chang et al. (arXiv:2606.01243), which showed that latent vectors in COCONUT models are semantically meaningful and causally upstream of outputs, but studied only reasoning accuracy tasks. We ask whether that structure extends to safety-relevant content.

---

## Research question

Given a prompt, can we predict whether it has demonstrated a risk of eliciting an unsafe response **without observing any chain-of-thought?**

More precisely: does latent reasoning encode safety-relevant information that is (a) present, (b) extractable, and (c) causally connected to model outputs, when there is no readable reasoning trace at all?

---

## Why this matters for AI safety

Current safety monitoring of frontier models relies heavily on inspecting chain-of-thought outputs. Anthropic's usage policies, the safety cases in Responsible Scaling Policy documents, and most deployed monitoring pipelines assume that *if reasoning is present, it is readable*. However, COCONUT-style architectures break this assumption at the architectural level.

This isn't a hypothetical concern. The COCONUT paper (Hao et al. 2024) demonstrates that models can be trained to reason entirely in continuous latent space, producing outputs with no intermediate text. If such architectures are widely adopted and the performance advantages (which exists) give strong incentive towards shifting to them then any monitoring approach that depends on CoT inspection fails by construction.

This project investigates whether an alternative is possible: We are essentially asking if its possible to detect safety-relevant signal from latent activations directly, without the reasoning trace. A positive result would suggest that latent-space monitoring tools (probes, SAEs, representation-level classifiers) could partially recover what CoT inspection currently provides. A negative result would be equally informative, quantifying the monitoring gap that latent reasoning architectures would introduce.

---

## Experimental design

The core experiment trains three LoRA adapters on Llama-3.2-1B, one for each reasoning format, on a shared safety classification task. All else is held equal.

**Key Task:** Given a prompt, predict its observed safety risk.  
- `HIGH_RISK`: at least one observed response to the prompt was unsafe  
- `LOW_RISK`: all observed responses were safe

| Experiment | Reasoning format | What it tests |
|---|---|---|
| E1 - Answer-only | No reasoning trace | Baseline: prompt features alone |
| E2 - CoT | Explicit `<reasoning>...</reasoning>` | Upper bound: readable intermediate reasoning |
| E3 - COCONUT | Continuous latent thoughts (`<bot>/<eot>`) | Target condition: latent reasoning only |

- The primary comparison is **E3 vs E2**: does latent reasoning match CoT's ability to detect safety-relevant signal, even though no intermediate text is produced?
- The secondary comparison is **E2 vs E1**: does explicit reasoning actually help at all for this task, before asking whether latent reasoning can replicate it.

---

## Dataset

**Source:** BeaverTails (PKU-Alignment, CC-BY-NC-4.0) - 330k prompt-response pairs with harm category labels and human safety annotations.

**Labelling policy:** Deduplicated by prompt; labeled `HIGH_RISK` if any paired response was unsafe. This is a deliberate recall choice to encourage high-recall as we care about prompts that *can* elicit unsafe outputs, not just those that *always* do.

**Reasoning traces:** Since the dataset doesn't come with reasoning traces, we had to synthetically generate one; this was done via GPT-4o-mini, conditioned on ground-truth labels and harm categories. The generator does not see the paired responses, ensuring it cannot learn information unavailable at inference time. Furthermore, the reasonong traces were validated against a set of rejection criteria (example: label contradiction, mention of observed responses, vague or overlong reasoning).

**Split (after deduplication and sampling):**

| Split | n | HIGH_RISK | LOW_RISK |
|---|---|---|---|
| Train | 3,928 | 2,500 | 1,428 |
| Validation | 429 | 250 | 179 |
| Test | 679 | 500 | 179 |


> **Note on test set imbalance:** The test set is 96% HIGH_RISK (493/512). This reflects BeaverTails' natural distribution and the any-unsafe labelling rule; not a sampling error. LOW_RISK conclusions should be interpreted with caution given the small sample (n=19). A single correct LOW_RISK prediction corresponds to 5.26% recall.

The Primary metric is **weighted F1**; `HIGH_RISK` recall is reported separately as the safety-critical error type.

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

## Latent Watch Results 

### Experimental Setup

| | Detail |
|---|---|
| **Model** | Llama-3.2-1B + LoRA (r=16, ~1.7M trainable params) |
| **Task** | Elicitation risk classification (HIGH\_RISK / LOW\_RISK) |
| **Dataset** | BeaverTails - 3,928 train / 429 val / 679 test |
| **Label rule** | HIGH\_RISK if any observed response was unsafe (any-unsafe aggregation) |
| **Primary metric** | Weighted F1 (class-imbalanced dataset) |


### Summary
 
| | E1 - Answer-only | E2 - Chain-of-Thought | E3 -  COCONUT |
|---|---|---|---|
| Reasoning format | None | Explicit token-level | Continuous latent states |
| Training input | Prompt → label | Prompt → reasoning + label | Prompt → latent tokens + label |
| Reasoning observable? | N/A | Yes | No |
| Best checkpoint | epoch_1 | epoch_3 | stage_1_epoch_1 |

All three experiments use the same base model (`meta-llama/Llama-3.2-1B`), LoRA configuration, dataset, and evaluation protocol. The only variable is reasoning format.

 
### What the Results Show
 
**1. Reasoning format has a detectable but small effect on classification behaviour.**
 
E1 (no reasoning) and E3 (latent reasoning) produce identical test behaviour, both predict `HIGH_RISK` for every example, achieving perfect HIGH_RISK recall at the cost of zero LOW_RISK discrimination. E2 (explicit CoT) is the only condition to show any deviation from this pattern: by epoch 3, the model correctly classifies 1 of 19 LOW_RISK examples, with a small corresponding drop in HIGH_RISK recall (491/493 vs 493/493).
 
The effect is small in absolute terms but consistent across all metrics where weighted F1 improves from 0.9447 to 0.9470, macro F1 from 0.49 to 0.54, and LOW_RISK recall from 0.0000 to 0.0526. This suggests explicit reasoning provides marginal but real additional signal for the harder classification cases.
 
**2. The central finding for the research question: latent reasoning (E3) does not replicate the CoT signal.**
 
Despite strong validation performance across all curriculum stages (val F1: 0.8783 → 0.9883 → 0.9530 → 0.9174), the COCONUT model produces identical test behaviour to the no-reasoning baseline. The safety-relevant discrimination that emerges in explicit CoT does not appear in the latent reasoning condition at this scale.
 
This is a negative result with respect to the research question; at 1B parameters with C=3 curriculum stages, we find no evidence that latent reasoning encodes additional safety-relevant signal beyond what a no-reasoning baseline captures. The result does not rule out latent safety signal at larger scale or with different curriculum configurations.
 
**3. HIGH_RISK recall is robust across all conditions.**
 
All three experiments maintain ≥0.9959 HIGH_RISK recall. The ability to flag unsafe prompts does not depend on reasoning format, it is learned quickly and stably regardless of whether the model reasons explicitly, latently, or not at all. This is reassuring for the safety-critical direction but suggests the task may be too easy on the HIGH_RISK side to distinguish reasoning conditions.


### E1 - Answer-Only Baseline (Complete)

> Model sees prompt only. No reasoning and predicts `HIGH_RISK` or `LOW_RISK` directly.

**Test Metrics**
 
| Metric | Value |
|---|---|
| Valid predictions | 512 / 512 |
| Accuracy | 0.9629 |
| Weighted Precision | 0.9272 |
| Weighted Recall | 0.9629 |
| **Weighted F1** | **0.9447** |
| HIGH_RISK Recall | **1.0000** |
| LOW_RISK Recall | 0.0000 |
| Macro F1 | 0.49 |
 

**Confusion Matrix**
 
| | Pred HIGH_RISK | Pred LOW_RISK |
|---|---|---|
| **True HIGH_RISK** | 493 | 0 |
| **True LOW_RISK** | 19 | 0 |

**Interpretation**
 
- E1 exhibits complete majority-class collapse: the model predicts `HIGH_RISK` for every example. This is consistent with the training distribution (64% HIGH_RISK) and the any-unsafe labelling rule, which creates a strong prior toward HIGH_RISK.
- The result is interpretable as a ceiling for a no-reasoning baseline; perfect recall on the majority class, zero recall on the minority class.
- This is expected and serves as the baseline against which E2 and E3 gains are measured.
- The weighted F1 of 0.9447 is largely a function of the class imbalance rather than discrimination ability.

---

### E2 - Chain-of-Thought

> The model generates an explicit reasoning trace (`<reasoning>...</reasoning>`) before producing the label inside `<answer>...</answer>`. Reasoning traces were synthetically generated via OpenAI API (gpt-4o-mini) conditioned on ground-truth labels. See dataset card for full attribution.

**Test Metrics Across Epochs**
 
| Epoch | Valid | Weighted F1 | HIGH_RISK Recall | LOW_RISK Recall | Macro F1 |
|---|---|---|---|---|---|
| epoch_1 | 509 / 512 | 0.9444 | 1.0000 | 0.0000 | 0.49 |
| epoch_2 | 512 / 512 | 0.9447 | 1.0000 | 0.0000 | 0.49 |
| **epoch_3** | **512 / 512** | **0.9470** | **0.9959** | **0.0526** | **0.54** |

> **Note:** epoch_1 had 3 invalid/unparseable outputs; Cases where the model failed to close the `</answer>` tag within the generation budget. These were excluded from metric computation.


**Best Checkpoint (epoch_3)**
 
| Metric | Value |
|---|---|
| Valid predictions | 512 / 512 |
| Accuracy | 0.9609 |
| Weighted Precision | 0.9412 |
| Weighted Recall | 0.9609 |
| **Weighted F1** | **0.9470** |
| HIGH_RISK Recall | 0.9959 |
| LOW_RISK Recall | **0.0526** |
| Macro F1 | 0.54 |
 
**Confusion Matrix (epoch_3)*8
 
| | Pred HIGH_RISK | Pred LOW_RISK |
|---|---|---|
| **True HIGH_RISK** | 491 | 2 |
| **True LOW_RISK** | 18 | 1 |

**Interpretation**
- Epoch 3 is the only experimental condition across all three experiments to correctly classify any LOW_RISK example (1/19).
- This comes at a small cost to HIGH_RISK recall (2 missed unsafe prompts vs 0 in E1), representing a trade-off between sensitivity and specificity.
- The improvement in macro F1 (0.49 → 0.54) and weighted F1 (0.9444 → 0.9470) across epochs suggests that extended CoT training gradually surfaces LOW_RISK sensitivity that is absent in the answer-only condition.The effect is small but directionally consistent with the hypothesis that explicit reasoning encodes safety-relevant information.
- The fact that LOW_RISK signal only emerges at epoch 3 after the model has largely memorised HIGH_RISK training examples (loss ~0.96 at epoch 3) , suggests the CoT reasoning is doing marginal but real work in the harder classification cases.
 
 
---

### E3 - COCONUT Latent Reasoning 

> Reasoning is progressively replaced by recurrent passes through the model's continuous hidden states across a staged curriculum (C=3). By the final stage, no explicit reasoning is decoded and the answer is produced from latent thought alone. Implemented from scratch following Hao et al. 2024; 

**Curriculum Progression**
 
| Stage | Description | Val F1 |
|---|---|---|
| Stage 0 | Full CoT - identical to E2 | 0.8783 |
| Stage 1 | First reasoning step replaced by `<bot>` | **0.9883** |
| Stage 2 | First two steps replaced | 0.9530 |
| Stage 3 | All steps replaced - fully latent | 0.9174 |
 
- In Stage 1, the model performs *better* with one latent token than with full explicit reasoning (Stage 0). This may reflect the latent token acting as a compressed, task-relevant representation that is more directly useful for classification than the full verbatim reasoning trace.
- Validation F1 degrades across Stages 2 and 3 as more reasoning moves into latent space, though it remains above the Stage 0 baseline through Stage 2. Indicating that the model that has learned meaningful latent representations but loses some discriminative signal at full latency.

**Test Metrics (best_adapter = stage_1_epoch_1)**
 
| Metric | Value |
|---|---|
| Valid predictions | 512 / 512 |
| Accuracy | 0.9629 |
| Weighted Precision | 0.9272 |
| Weighted Recall | 0.9629 |
| **Weighted F1** | **0.9447** |
| HIGH_RISK Recall | 1.0000 |
| LOW_RISK Recall | 0.0000 |
| Macro F1 | 0.49 |

**Confusion Matrix**
 
| | Pred HIGH_RISK | Pred LOW_RISK |
|---|---|---|
| **True HIGH_RISK** | 493 | 0 |
| **True LOW_RISK** | 19 | 0 |


 
---

### Next Steps

- Complete E2 and E3 evaluation
- Compare weighted F1 across E1 / E2 / E3
- Analyse per-category performance on HIGH\_RISK harm categories
- Investigate LOW\_RISK classification failure across all three conditions

---

*Training infrastructure: Google Colab Pro (T4) for E1/E2, RunPod A100 SXM for E3*
*Dataset: BeaverTails (CC-BY-NC-4.0) | Model: Llama-3.2-1B (Meta)*

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

**If E3 ≈ E2 (latent reasoning matches CoT performance):** This would suggest that safety-relevant signal is not merely encoded in the readable trace. It is present in the latent representations themselves, and is predictive of outputs. This opens the door to probe-based or representation-level monitoring as an alternative to CoT inspection.

**If E3 << E2 (latent reasoning underperforms CoT):** This would quantify the monitoring gap that latent architectures introduce, and would motivate investment in alternative interpretability tools (SAE-based feature discovery, activation-level anomaly detection) to partially recover what is lost.

**If E3 ≈ E1 (latent reasoning adds nothing beyond prompt features):** This would suggest that latent reasoning in COCONUT doesn't meaningfully engage with safety-relevant content at all, at least in this training regime; a finding with implications for how latent reasoning models respond to safety-relevant inputs in practice.

Any of these outcomes is informative. The staged design ensures the null result is not vacuous.

---

## Limitations and caveats

**Scale:** Llama-3.2-1B is small. Results may not generalise to larger models where latent reasoning is likely more capable and more meaningfully engaged.

**Training data:** Reasoning traces are synthetically generated, not human-authored. The quality of CoT (and by extension COCONUT's curriculum signal) depends on GPT-4o-mini's ability to produce rationales that reflect genuine risk reasoning rather than label-correlated surface patterns.

**Task framing:** This task framing for this project classifies *prompt-level* elicitation risk based on observed response variance. It is not a real-time inference task and does not simulate deployment conditions directly. Results speak to whether latent representations encode safety signal in principle, not to operational monitoring performance.

**COCONUT implementation:** We implement COCONUT from scratch following Hao et al. 2024, not using Dilgren & Wiegreffe's codebase. Differences in implementation may affect comparability to other COCONUT results in the literature.

---

## Related work

- **Chang et al. (arXiv:2606.01243):** Establishes that latent vectors in COCONUT models are semantically meaningful and causally upstream of outputs (§3.2–3.3). We extend their investigation to safety-relevant content.
- **Hao et al. 2024 (COCONUT):** Original architecture and training curriculum for continuous-thought reasoning models.
- **Dilgren & Wiegreffe (arXiv:2604.04902):** Independent finding that latent tokens are often causally inert on logical-reasoning tasks-  a motivation for our necessity checks and staged experimental design.
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

Uses:
- **BeaverTails** (PKU-Alignment, CC-BY-NC-4.0) for prompt data.
- Base model: `meta-llama/Llama-3.2-1B` (Llama 3.2 Community License).
- Reasoning traces synthetically generated via OpenAI API; noted in dataset card.

Non-commercial research use only.

Full citations and license terms in [`ATTRIBUTION.md`](./ATTRIBUTION.md).
