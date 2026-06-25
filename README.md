# Discrete Latent Reasoning 

This repository contains the open training code for **Discrete Latent Reasoning (DLR)**, organized from the submission *Why Struggle with Continuous Latents? Interpretable Discrete Latent Reasoning via Rendered Compression*.

DLR turns rendered chain-of-thought traces into discrete latent tokens, then trains a language model over a joint text + latent vocabulary. The code is organized around the paper pipeline:

1. Render CoT traces and train a DeepSeek-OCR2 based latent codebook.
2. Export codebook ids as latent-token sequences.
3. Train an augmented latent language model with latent-text alignment, latent SFT, and latent GRPO.
4. Evaluate and optionally decode latent ids back into interpretable traces.

## Repository Layout

```text
deepseek_codebook/        DeepSeek-OCR2 codebook training, id export, and latent decoding tools
latent_llm/scripts/       Latent pretraining, SFT, GRPO, dataset conversion, and evaluation entrypoints
latent_llm/trl/           Minimal local TRL fork with DLR latent-token extensions
latent_llm/configs/       DeepSpeed config used by the launch scripts
examples/                 8-GPU launch scripts with environment-variable overrides
```

The repository intentionally excludes model weights, generated datasets, checkpoints, evaluation runs, and cache files.

## Installation

```bash
cd Discrete_Latent_Reasoning
conda create -n DLR python=3.12 -y
conda activate DLR
pip install -r requirements.txt
pip install -e latent_llm/trl
```

Install a CUDA-specific PyTorch wheel first if your environment requires one.

## Data Format

The codebook stage expects a Hugging Face dataset saved to disk with rendered CoT images and target text fields:

```text
data/rendered_cot_hf/
```

The exported codebook JSONL records should contain:

```json
{
  "id": "...",
  "image_codebook_input_ids": [12, 345, 678],
  "text": "reasoning trace",
  "original": {"question": "...", "answer": "..."}
}
```

The latent LLM training scripts consume processed JSONL files with `input_ids` and `labels`.

## Training

### 1. Train the DeepSeek-OCR2 codebook

```bash
DLR_CODEBOOK_TRAIN_DATASET=/path/to/rendered_cot_hf \
CODEBOOK_INIT_PACKAGE_PATH=/path/to/codebook_init.pt \
DLR_OUTPUT_ROOT=$PWD/outputs \
bash examples/run_deepseek_codebook_8gpu.sh
```

This stage corresponds to the paper's stochastic latent codebook construction: rendered CoT images are encoded, quantized through a learned codebook, and reconstructed with the OCR decoder.

### 2. Export latent ids

```bash
python deepseek_codebook/export_codebook_ids.py \
  --input-jsonl data/train.jsonl \
  --checkpoint outputs/deepseek_codebook/checkpoint-last \
  --output-jsonl outputs/source2_codebook_input_ids.jsonl \
  --overwrite
```

### 3. Prepare latent LLM datasets

```bash
python latent_llm/scripts/prepare_source2_latent_pretrain_dataset.py \
  --model-name-or-path models/Qwen3-VL-4B-Instruct \
  --input-jsonl outputs/source2_codebook_input_ids.jsonl \
  --output-jsonl data/latent_pretrain.jsonl

python latent_llm/scripts/prepare_source2_latent_sft_dataset.py \
  --model-name-or-path models/Qwen3-VL-4B-Instruct \
  --input-jsonl outputs/source2_codebook_input_ids.jsonl \
  --output-jsonl data/latent_sft.jsonl
```

### 4. Latent-text alignment / pretraining

```bash
MODEL_PATH=/path/to/base_model \
CODEBOOK_PATH=outputs/deepseek_codebook/latent/codebook.pt \
TRAIN_DATA_PATH=data/latent_pretrain.jsonl \
OUTPUT_DIR=outputs/latent_pretrain \
bash examples/run_latent_pretrain_8gpu.sh
```

### 5. Latent SFT

```bash
MODEL_PATH=/path/to/base_model \
CODEBOOK_PATH=outputs/deepseek_codebook/latent/codebook.pt \
PROJECTOR_PATH=outputs/latent_pretrain/latent/projector.pt \
TRAIN_DATA_PATH=data/latent_sft.jsonl \
OUTPUT_DIR=outputs/latent_sft \
bash examples/run_latent_sft_8gpu.sh
```

### 6. Latent GRPO

```bash
MODEL_PATH=outputs/latent_sft \
TRAIN_DATA_PATH=data/latent_grpo.jsonl \
OUTPUT_DIR=outputs/latent_grpo \
bash examples/run_latent_grpo_8gpu.sh
```

The GRPO script implements final-answer rewards for arithmetic-style tasks. The paper's process-alignment reward depends on the auxiliary decoder and evaluator LLM; the decoder-side tooling is included in `deepseek_codebook/visualize_latent_ids.py`.

## Latent Checkpoint Format

Latent checkpoints are saved as a standard base model plus a `latent/` directory:

```text
checkpoint/
  config.json
  model.safetensors
  tokenizer.*
  latent/
    codebook.safetensors or codebook.pt
    projector.safetensors or projector.pt
    config.json
```

Latent token ids occupy `[base_vocab_size, base_vocab_size + num_latent_tokens - 1]`.

## Notes

- `latent_llm/trl/` is a local TRL fork containing the DLR latent embedding, LM head, SFT trainer, and GRPO trainer extensions.
- `deepseek_codebook/model_source_10k/` contains the codebook-augmented DeepSeek-OCR2 model source. It does not include model weights.
- Paths in scripts are defaults only; override them with environment variables or command-line arguments for your environment.
