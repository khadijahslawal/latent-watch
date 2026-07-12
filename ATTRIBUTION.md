# Attribution

This project builds on, and includes data derived from, the following third-party sources. Please retain this file if you fork or redistribute this repository.

## Dataset: BeaverTails

Prompts and harm-category labels in `data/schema_files/` are derived from:

> Ji, J., Liu, M., Dai, J., Pan, X., Zhang, C., Bian, C., Chen, B., Sun, R., Wang, Y., & Yang, Y. (2023). *BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset.* arXiv:2307.04657.

Source: [`PKU-Alignment/BeaverTails`](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) on Hugging Face.

**License: CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International). This means:
- Attribution to the original authors is required (this file + citations in `research_log.md`/`README.md` serve that purpose).
- **Non-commercial use only.** This repository and its derived data are for academic/research purposes (BlueDot Impact Technical AI Safety project sprint). Do not use `data/schema_files/` or any BeaverTails-derived content for commercial purposes.
- Adaptation and redistribution of derived works is permitted under the same non-commercial terms.

`data/schema_files/` contains a *filtered and reformatted* subset of BeaverTails (deduplicated by prompt, relabeled at the prompt level per the policy documented in `research_log.md` §3), not the full original dataset. See `data/beavertails_to_schema.py` for the exact transformation applied.

## Checkpoints & Interpretability Tooling

Model checkpoints (referenced by name/commit, not redistributed in this repo) and the `early_stopping` necessity-ablation tooling originate from:

> Dilgren, C., & Wiegreffe, S. (2026). *Are Latent Reasoning Models Easily Interpretable?* arXiv:2604.04902.

Source: [`connordilgren/are-lrms-easily-interpretable`](https://github.com/connordilgren/are-lrms-easily-interpretable) (GitHub, MIT License).

Checkpoint collection: [`connordilgren/are-latent-reasoning-models-easily-interpretable`](https://huggingface.co/collections/connordilgren/are-latent-reasoning-models-easily-interpretable) on Hugging Face.

This repository uses a **fork** of the above (`external/are-lrms-easily-interpretable/`, added as a git submodule) with three additional commits applying fixes/adaptations necessary to run their tooling on out-of-distribution safety-prompt data. See `patches/` for the individual diffs and `research_log.md` §5 for the rationale behind each change. The original MIT license is preserved in the fork.

## Foundational Work

The research question builds on, but does not redistribute any code or data from:

> Chang, et al. (2026). arXiv:2606.01243. (See `research_log.md` §1 for how this project's question relates to theirs.)

## Base Models

Checkpoints used are fine-tuned from:
- `openai-community/gpt2` (OpenAI, MIT License)
- `meta-llama/Llama-3.2-1B-Instruct` (Meta, Llama 3.2 Community License — gated access required; note this license has its own use-based restrictions separate from BeaverTails' CC-BY-NC-4.0 terms, and is more restrictive than a standard open-source license. Anyone reusing this repo's Llama-based results should independently confirm they meet Meta's license terms.)
