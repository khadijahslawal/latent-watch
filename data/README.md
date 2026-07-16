# Data Directory

```
data/
├── interim/                         # Intermediate parquet files (not committed)
│   ├── prompt_aggregates.parquet    # One row per unique prompt with response stats
│   └── split_assignments.parquet   # Prompt → {train, validation, test} assignment
│
└── processed/
    └── beavertails_risk_v1/
        ├── canonical/               # One row per unique prompt, full fields
        │   ├── train.jsonl
        │   ├── validation.jsonl
        │   └── test.jsonl
        ├── answer_only/             # Formatted for answer-only fine-tuning
        │   ├── train.jsonl
        │   ├── validation.jsonl
        │   └── test.jsonl
        ├── cot/                     # Formatted for CoT and COCONUT fine-tuning
        │   ├── train.jsonl
        │   ├── validation.jsonl
        │   └── test.jsonl
        ├── dataset_card.md
        ├── label_schema.json
        └── manifest.json
```

## Regeneration

The pipeline runs once and outputs persist on Google Drive. To re-run:

```bash
bash scripts/prepare_dataset.sh
```

or step-by-step via the Colab notebook `notebooks/04_train_pilot.ipynb` (cells 1–11).

## Notes

- Rationale generation calls the OpenAI API (`gpt-4o-mini` by default). Set `OPENAI_API_KEY`.
- `cot/` is shared by CoT and COCONUT training — do not maintain separate copies.
- Raw BeaverTails data is streamed from HuggingFace Hub and never saved locally.
