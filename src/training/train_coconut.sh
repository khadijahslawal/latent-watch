#!/bin/bash
# E3: COCONUT training — entry point for rented GPU (A100)
# Usage: bash scripts/train_coconut.sh
# Assumes repo is installed and data is available at DATA_DIR
 
set -euo pipefail
 
# ── Paths (edit these) ────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/processed/beavertails_risk_v1/cot}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/coconut}"

# ── Run ───────────────────────────────────────────────────────────────────
python -m latent_watch.training.train_coconut \
    --dataset-dir "$DATA_DIR" \
    --output-dir  "$OUTPUT_DIR" \
    --config-path configs/training/llama_1b_coconut.yaml
 