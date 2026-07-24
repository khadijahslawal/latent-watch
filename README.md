# Latent Watch
## Can Latent Reasoning Be Safety-Monitored?

**BlueDot Impact - Technical AI Safety Project Sprint**  
**Author:** Khadija Shuaib  
**Project Status:** Active - Pilot Training Completed, Analysis and Latent Probing in Progress

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

**Harm categories covered (14):** animal abuse, child abuse, controversial topics / politics, discrimination / stereotype / injustice, drug abuse / weapons / banned substances, financial crime / property crime / theft, hate speech / offensive language, misinformation regarding ethics-laws-safety, non-violent unethical behaviour, privacy violation, self-harm, sexually explicit / adult content, terrorism / organised crime, violence / aiding / abetting / incitement.


> **Note on dataset distribution and how we got there:** 

In the original beavertails dataset card on hugging face. The following distribution was given:
|  | n | % | 
|---|---|---|
| Safe | 12,892 | 42.68% |
| Unsafe | 17,315| 57.32% | 

Our distribution differs due to certain pre-processing that needed to be done in order to make the dataset fit for the research question being posed.

**Pre-processing Step 1: Aggregating unique prompts**

BeaverTails labeled the safety of the prompt–response pair on whether the response contains harmful content. Thus, a single prompt may have several responses with different safety labels and may belong to multiple harm categories. If a pompt has is_safe == False does not necessarily mean the prompt itself expresses harmful intent; it can also mean that one model answered a benign, ambiguous, or dual-use prompt unsafely.

This created a bit of challenge in how to decide what the ground-truth is; initially we considered doing a "majority carries the vote". However, this wouldn't work in the scenario whereby there is an even split between `is_safe == False` and `is_safe == True`.

This brings us back to the task definition mentioned earlier. The task we are giving to the 3 models being experimented is **"Given the prompt, is it  likely to elicit an unsafe response from the model?"**. As such we decided to go with the following aggrgeation approach; essentially deduplicating by prompt; a prompt is labeled as `HIGH_RISK` if any paired response was unsafe. 

`HIGH_RISK_PROMPT = at least one observed response was unsafe`

`LOW_RISK_PROMPT  = all observed responses were safe`

Naturally this makes our curated dataset slightly diverge from the original BeaverTails in two different ways

1. Total prompt count would be less as we aggregated on a prompt level
2. The prompts risk label is more likely to skew towards `HIGH RISK` than `LOW RISK`

This labelling policy was a deliberate choice to encourage high-recall as we care about prompts that can elicit unsafe outputs, not just those that always do.

**Pre-processing Step 2: Train, Validation, Test Split**

80%, 10%, 10% split was applied for the train, validation and test sets respectively; with a random seed = `42` for reproducibility. 

**Pre-processing Step 3: Sampling**

Following the deduplication and aggregation by prompt, to ensure the curated dataset is as balanced as possible, a two-step sampling approach was applied to the training set only - val and test used all available prompts up to their target sizes.

1. The first sampling logic was within the `HIGH_RISK` category, where we  applied a per-category cap of `400` for the 14 different harmful categories in the dataset. This was done to ensure category coverage and to prevent a specific a type of harmful category from dominating the dataset.

2. Downsample / upsample HIGH_RISK to match LOW_RISK count → ~50/50 balance. 


**Split (after deduplication and sampling):**

| Split | Total | HIGH_RISK | LOW_RISK | HIGH_RISK % |
|---|---|---|---|---|
| Train | 3,888 | ~2,460 | ~1,428 | ~63% |
| Validation | 422 | 250 | 172 | ~59% |
| Test | 672 | 493 | 179 | ~73% |


#### Pre-processing Step 4: Generating Synthethic Reasoning traces

The originial dataset from BeaverTails gives us:
- prompt
- response
- is_safe
- category

but doesn't gives us the step-by-step safety rationale for why that safety or harmful label was given. 

However, given that we are working with reasoning models, both CoT and Coconut would require reasoning traces in the training dataset in order to generalize the reasoning behind the elicitation risk for a given prompt. 

Since the dataset doesn't come with reasoning traces, we had to synthetically generate one using a teacher model; this was done via GPT-4o-mini, conditioned on the ground-truth labels and the harm categories. The generator does not see the paired responses, ensuring it cannot learn information unavailable at inference time. 

Example:
```
{
  "prompt": "...",
  "response": "...",
  "is_safe": false,
  "categories": {
    "violence_aiding_and_abetting": true
  },
  "reasoning": [
    "The user requests assistance related to a harmful action.",
    "The response provides information that materially facilitates that action.",
    "The response therefore increases the user's ability to cause harm."
  ],
  "label": "UNSAFE"
}
```

Furthermore, the reasonong traces were validated against a set of rejection criteria including: label contradiction, mention of observed responses, vague or overlong reasoning. 

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


The Primary metric is **weighted F1**; `HIGH_RISK` recall is reported separately as the safety-critical error type.

### Summary
 
| | E1 - Answer-only | E2 - Chain-of-Thought | E3 -  COCONUT |
|---|---|---|---|
| Reasoning format | None | Explicit token-level | Continuous latent states |
| Training input | Prompt → label | Prompt → reasoning + label | Prompt → latent tokens + label |
| Reasoning observable? | N/A | Yes | No |
| Best checkpoint | epoch_1 | epoch_3 | stage_1_epoch_1 |

All three experiments use the same base model (`meta-llama/Llama-3.2-1B`), LoRA configuration, dataset, and evaluation protocol. The only variable is reasoning format.

**Pre-specified outcome framework.** 

Before training, we specified three interpretable outcome directions:
 
- **E3 ≈ E2** - latent reasoning matches CoT: safety-relevant signal is present in continuous representations, motivating probe-based monitoring as an alternative to CoT inspection
- **E3 << E2** - latent reasoning underperforms CoT: quantifies the monitoring gap that latent architectures introduce, motivating SAE-based or activation-level alternatives
- **E3 ≈ E1** - latent reasoning adds nothing beyond prompt features: at this scale and configuration, latent representations do not encode additional safety-relevant signal
Any of these outcomes is informative. The staged design ensures the result is interpretable regardless of direction.

### Results Summary

| Experiment | Best checkpoint | Weighted F1 | HIGH_RISK Recall | LOW_RISK Recall | Macro F1 |
|---|---|---|---|---|---|
| E1 - Answer-only | epoch_3 | 0.7646 | 0.8844 | 0.4693 | 0.69 |
| E2 - Chain-of-Thought | epoch_3 | **0.7196** | 0.9229 | **0.2793** | 0.61 |
| E3 - COCONUT (latent) | stage 1 epoch 1 | 0.6237 | **0.9980** | 0.0056 | 0.43 


### Results Summary
 
### What the Results Show

**1. Reasoning format has a clear and differentiated effect on classification behaviour,  but not in the direction a simple capability story would predict.**
 
The three experiments reveal three qualitatively distinct trade-off profiles between HIGH_RISK and LOW_RISK recall. As reasoning moves from none (E1) → explicit (E2) → latent (E3), HIGH_RISK recall increases monotonically (0.88 → 0.92 → 1.00) and LOW_RISK recall decreases monotonically (0.47 → 0.28 → 0.01). Reasoning format does not simply shift performance up or down , it changes what the model attends to.
 
**2. E1 (answer-only) shows unstable training dynamics but the best overall balance.**
 
Without reasoning to anchor predictions, the answer-only model oscillates across epochs, starting with inverted class preferences (HIGH_RISK recall 0.37, LOW_RISK recall 0.85 at epoch 1), finding a balance at epoch 2, and settling toward HIGH_RISK dominance by epoch 3. The epoch 1 behaviour is quite interesting in that the model initially over-predicts LOW_RISK, suggesting the corrected dataset's class balance is genuinely influencing early training. But by epoch 3, E1 achieves the highest weighted F1 (0.7646) and meaningful LOW_RISK recall (0.4693). However, this comes at the cost of missing 12% of HIGH_RISK prompts, which is the costly error direction.
 
**3. E2 (CoT) converges faster and maintains a more stable HIGH_RISK/LOW_RISK trade-off.**
 
CoT training reaches a reasonable balance at epoch 1 (HIGH_RISK recall 0.80, LOW_RISK recall 0.47) and then progressively shifts toward HIGH_RISK across epochs 2 and 3. By epoch 3, it achieves the best LOW_RISK recall among the reasoning conditions (0.2793) while maintaining strong HIGH_RISK recall (0.9229). The explicit reasoning trace appears to stabilise training relative to E1 and the model converges to a consistent HIGH_RISK-dominant strategy without the epoch 1 inversion.
 
**4. E3 (COCONUT) nearly collapses to all-HIGH_RISK prediction.**
 
The latent reasoning model achieves near-perfect HIGH_RISK recall (0.9980) but almost zero LOW_RISK recall (0.0056 - 1 correct out of 179). This places the result firmly in the **E3 ≈ E1 (collapsed)** outcome; but more specifically, E3 resembles what E1 would look like if training had continued past the point of balance into full HIGH_RISK dominance. The latent curriculum does not produce the class-balancing effect seen in early E1 training, suggesting that continuous latent representations at this scale do not encode the LOW_RISK signal that even prompt-feature learning picks up on with sufficient training.
 
**Note: The result should be read as a scale and configuration finding, not a general claim about latent safety signal.** At 1B parameters with C=3 curriculum stages, latent reasoning does not produce the LOW_RISK discrimination that explicit reasoning and even no-reasoning baselines achieve. Whether this reflects a fundamental limitation or insufficient scale remains an open question.

In the below section, you would detailed information about each experiment and its outcome.

### E1 - Answer-Only Baseline (Complete)

> Model sees prompt only. No reasoning and predicts `HIGH_RISK` or `LOW_RISK` directly.

**Best checkpoint:** epoch_3 (weighted F1: 0.7646).
 
| Metric | Value |
|---|---|
| Valid predictions | 672 / 672 |
| Accuracy | 0.7738 |
| Weighted Precision | 0.7611 |
| Weighted Recall | 0.7738 |
| **Weighted F1** | **0.7646** |
| HIGH_RISK Recall | 0.8844 |
| LOW_RISK Recall | 0.4693 |
| Macro F1 | 0.69 |
 
**Confusion matrix:**
 
| | Pred HIGH_RISK | Pred LOW_RISK |
|---|---|---|
| **True HIGH_RISK** | 436 | 57 |
| **True LOW_RISK** | 95 | 84 |
 
The answer-only model correctly classifies 84 of 179 LOW_RISK examples making it the highest absolute count of any condition. The training trajectory (inverted at epoch 1, balanced at epoch 2, HIGH_RISK-dominant at epoch 3) reflects the model searching for a stable strategy without a reasoning scaffold to anchor it.

---

### E2 - Chain-of-Thought

> The model generates an explicit reasoning trace (`<reasoning>...</reasoning>`) before producing the label inside `<answer>...</answer>`. Reasoning traces were synthetically generated via OpenAI API (gpt-4o-mini) conditioned on ground-truth labels. See dataset card for full attribution.

**Test Metrics Across Epochs**
 

> **Note:** epoch_1 had 3 invalid/unparseable outputs; Cases where the model failed to close the `</answer>` tag within the generation budget. These were excluded from metric computation.


**Best checkpoint:** epoch_3 (weighted F1: 0.7196).
 
| Metric | Value |
|---|---|
| Valid predictions | 672 / 672 |
| Accuracy | 0.7515 |
| Weighted Precision | 0.7229 |
| Weighted Recall | 0.7515 |
| **Weighted F1** | **0.7196** |
| HIGH_RISK Recall | 0.9229 |
| LOW_RISK Recall | 0.2793 |
| Macro F1 | 0.61 |
 
**Confusion matrix:**
 
| | Pred HIGH_RISK | Pred LOW_RISK |
|---|---|---|
| **True HIGH_RISK** | 455 | 38 |
| **True LOW_RISK** | 129 | 50 |
 
CoT correctly classifies 50 of 179 LOW_RISK examples while maintaining stronger HIGH_RISK recall than E1. The explicit reasoning trace supports a more stable training trajectory but anchors the model more strongly toward HIGH_RISK dominance than the no-reasoning baseline.
 

 
 
---

### E3 - COCONUT Latent Reasoning 

> Reasoning is progressively replaced by recurrent passes through the model's continuous hidden states across a staged curriculum (C=3). By the final stage, no explicit reasoning is decoded and the answer is produced from latent thought alone. Implemented from scratch following Hao et al. 2024; 

**Curriculum Progression**
 
| Stage | Description|
|---|---|
| Stage 0 | Full CoT - identical to E2 |
| Stage 1 | First reasoning step replaced by `<bot>` | 
| Stage 2 | First two steps replaced | 
| Stage 3 | All steps replaced - fully latent | 
 
- In Stage 1, the model performs *better* with one latent token than with full explicit reasoning (Stage 0). This may reflect the latent token acting as a compressed, task-relevant representation that is more directly useful for classification than the full verbatim reasoning trace.
- Validation F1 degrades across Stages 2 and 3 as more reasoning moves into latent space, though it remains above the Stage 0 baseline through Stage 2. Indicating that the model that has learned meaningful latent representations but loses some discriminative signal at full latency.

**Best checkpoint:** stage_1_epoch_1 (val F1: 0.9883 during curriculum training).
 
### Curriculum progression (validation F1)
 
| Stage | Description | Train Loss | Val F1 |
|---|---|---|---|
| Stage 0 | Full CoT - identical to E2 | 1.6472 | 0.8783 |
| **Stage 1** | First reasoning step → `<bot>` | 1.0719 | **0.9883** |
| Stage 2 | First two steps → `<bot><bot>` | 0.9302 | 0.9530 |
| Stage 3 | Fully latent | 0.2862 | 0.9174 |
 
The Stage 1 validation peak outperforming full CoT is a notable curriculum dynamic. The very low Stage 3 train loss (0.2862) alongside reasonable val F1 (0.9174) suggests the model is forming internal representations during curriculum training. These dynamics are informative independently of the test classification result.
 
**Test metrics**
 
| Metric | Value |
|---|---|
| Valid predictions | 672 / 672 |
| Accuracy | 0.7336 |
| Weighted Precision | 0.6719 |
| Weighted Recall | 0.7336 |
| **Weighted F1** | **0.6237** |
| HIGH_RISK Recall | 0.9980 |
| LOW_RISK Recall | 0.0056 |
| Macro F1 | 0.43 |
 
**Confusion matrix:**
 
| | Pred HIGH_RISK | Pred LOW_RISK |
|---|---|---|
| **True HIGH_RISK** | 492 | 1 |
| **True LOW_RISK** | 178 | 1 |
 
The gap between strong curriculum validation performance and near-zero LOW_RISK test recall is the central unresolved question for E3. Two explanations are consistent with the data: (1) latent representations at 1B scale do not encode LOW_RISK signal, or (2) the classification head fails to extract signal that may be present in the representations. Distinguishing these requires probing the latent representations directly.

 
---

**Training infrastructure:** 

- Google Colab Pro (T4) for E1/E2
- RunPod A100 SXM for E3*
- Dataset: BeaverTails (CC-BY-NC-4.0) 
- Model: Llama-3.2-1B (Meta)*



---

## Limitations and caveats

**Scale:** Llama-3.2-1B is small. Results may not generalise to larger models where latent reasoning is likely more capable and more meaningfully engaged.

**Synthetic COT  traces:** Reasoning traces are synthetically generated, not human-authored. The quality of CoT (and by extension COCONUT's curriculum signal) depends on GPT-4o-mini's ability to produce rationales that reflect genuine risk reasoning rather than label-correlated surface patterns.

**Single curriculum configuration for E3.** C=3 stages, 1 epoch per stage. The Stage 1 validation peak suggests curriculum dynamics are sensitive to configuration.

**Task framing:** This task framing for this project classifies *prompt-level* elicitation risk based on observed response variance. It is not a real-time inference task and does not simulate deployment conditions directly. Results speak to whether latent representations encode safety signal in principle, not to operational monitoring performance.

**COCONUT implementation:** We implement COCONUT from scratch following Hao et al. 2024, not using Dilgren & Wiegreffe's codebase. Differences in implementation may affect comparability to other COCONUT results in the literature.

**Classification as the only evaluation window.** Test classification may not fully capture what is encoded in latent representations. Probing is needed to assess this directly for E3.

---

### Next Steps

**Immediate**:

1. Behavioral downstream analysis: Identify and categorize the prompts where CoT succeeds and COCONUT fails. This gives us the narrative for the paper and tells us what kind of reasoning is being lost.
2. Information-survival analysis: Quantify how many of CoT’s gains over the answer-only baseline are retained by COCONUT. This directly addresses the research question in behavioral terms.
3. Latent probing: Determine whether the latent representations still contain the safety information that the decoder fails to use. This provides a mechanistic explanation for the behavioral findings.

**Longer term**

1. To address the scale question, we should rerun E2 and E3 at Llama-3.1-8B or Qwen3-8B. This helps in understanding if the E3 LOW_RISK collapse persist at larger scale?

2. To look inside E3's representations:

&emsp; a. Linear probing on intermediate hidden states; We want to know if the LOW_RISK signal present in E3's representations even when the classification head fails to surface it?

&emsp;  b. SAE-based feature discovery on latent thought tokens
  
&emsp;  c. Activation-level comparison between E3 HIGH_RISK and LOW_RISK examples

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
