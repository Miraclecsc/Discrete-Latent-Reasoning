#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT/deepseek_codebook"
export PYTHONUNBUFFERED=1
export DLR_DATA_ROOT="${DLR_DATA_ROOT:-$PROJECT_ROOT/data}"
export DLR_OUTPUT_ROOT="${DLR_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
export DLR_CODEBOOK_TRAIN_DATASET="${DLR_CODEBOOK_TRAIN_DATASET:-$DLR_DATA_ROOT/rendered_cot_hf}"
export CODEBOOK_INIT_PACKAGE_PATH="${CODEBOOK_INIT_PACKAGE_PATH:-$DLR_DATA_ROOT/codebook_init_10k_vmf.pt}"

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE:-8}" train_codebook.py
