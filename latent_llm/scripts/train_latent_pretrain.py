#!/usr/bin/env python
# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Latent pretraining entrypoint for processed datasets with full-sequence supervision.

Expected dataset format:
    {
        "input_ids": [...],
        "labels": [...],
        ...
    }

The dataset is assumed to be pre-tokenized already:
    [latent token ids] + [text token ids]

Loss is computed on the whole sequence via `labels=input_ids`.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset

from trl.models.latent_utils import align_model_embeddings_with_tokenizer
from trl.models.local_model_utils import is_deepseek_decoder_model
from trl.models.local_model_utils import is_vlm_model as is_vlm_model_path
from trl.models.local_model_utils import load_model_for_training, load_tokenizer_for_model


DEFAULT_MODEL_PATH = "models/Qwen3-VL-4B-Instruct"
DEFAULT_CODEBOOK_PATH = "outputs/deepseek_codebook/latent/codebook.pt"
DEFAULT_TRAIN_DATA_PATH = "data/latent_pretrain.jsonl"
DEFAULT_OUTPUT_DIR = "./latent_pretrain_output_2"


def is_main_process(local_rank: int) -> bool:
    """
    Check whether the current process is the main logging process.

    Args:
        local_rank (`int`):
            Rank from the launcher environment.

    Returns:
        `bool`:
            Whether this process should print user-facing logs.
    """
    return local_rank in (-1, 0)


def log(message: str, local_rank: int) -> None:
    """
    Print a log line only from the main process.

    Args:
        message (`str`):
            Message to print.
        local_rank (`int`):
            Rank from the launcher environment.
    """
    if is_main_process(local_rank):
        print(message)


def is_vlm_model(model_name_or_path: str) -> bool:
    """
    Detect whether a model path corresponds to a vision-language model.

    Args:
        model_name_or_path (`str`):
            Local model path or model identifier.

    Returns:
        `bool`:
            Whether the model should be loaded through `AutoModelForImageTextToText`.
    """
    return is_vlm_model_path(model_name_or_path)


def load_model(model_name_or_path: str, torch_dtype: torch.dtype):
    """
    Load a local model using the appropriate auto class.

    Args:
        model_name_or_path (`str`):
            Local model path or model identifier.
        torch_dtype (`torch.dtype`):
            Target loading dtype.

    Returns:
        `PreTrainedModel`:
            Loaded model instance.
    """
    return load_model_for_training(
        model_name_or_path,
        torch_dtype=torch_dtype,
    )


def read_first_jsonl_record(path: Path) -> dict:
    """
    Read the first non-empty JSON object from a JSONL file.

    Args:
        path (`Path`):
            Input JSONL path.

    Returns:
        `dict`:
            First record.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSON records found in {path}")


def inspect_codebook(codebook_path: str) -> tuple[int, int, torch.dtype]:
    """
    Inspect the standalone codebook tensor file.

    Args:
        codebook_path (`str`):
            Local path to the extracted codebook tensor.

    Returns:
        `tuple[int, int, torch.dtype]`:
            Number of latent tokens, codebook dimension, and tensor dtype.
    """
    tensor = torch.load(codebook_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {codebook_path}, got {type(tensor)}")
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D codebook tensor, got shape {tuple(tensor.shape)}")
    return tensor.shape[0], tensor.shape[1], tensor.dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--codebook_path", type=str, default=DEFAULT_CODEBOOK_PATH)
    parser.add_argument("--train_data_path", type=str, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_latent_tokens", type=int, default=10000)
    parser.add_argument("--codebook_dim", type=int, default=1280)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--optim_args", type=str, default="foreach=False")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp_find_unused_parameters", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze_base_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze_codebook", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deepseek_attn_implementation",
        type=str,
        choices=("eager", "flash_attention_2"),
        default=None,
    )
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    log("=" * 80, local_rank)
    log("Latent Pretraining", local_rank)
    log("=" * 80, local_rank)

    log("\n[1/5] Inspecting inputs...", local_rank)
    train_data_path = Path(args.train_data_path)
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")
    if not Path(args.codebook_path).exists():
        raise FileNotFoundError(f"Codebook not found: {args.codebook_path}")

    codebook_tokens, codebook_dim, codebook_dtype = inspect_codebook(args.codebook_path)
    if codebook_tokens != args.num_latent_tokens:
        raise ValueError(
            f"--num_latent_tokens={args.num_latent_tokens} does not match codebook rows {codebook_tokens}"
        )
    if codebook_dim != args.codebook_dim:
        raise ValueError(f"--codebook_dim={args.codebook_dim} does not match codebook dim {codebook_dim}")

    first_record = read_first_jsonl_record(train_data_path)
    for required_key in ("input_ids", "labels"):
        if required_key not in first_record:
            raise KeyError(f"Processed dataset must contain `{required_key}`: missing in first record")

    log(f"  train_data_path: {train_data_path}", local_rank)
    log(f"  codebook_path: {args.codebook_path}", local_rank)
    log(f"  codebook shape: ({codebook_tokens}, {codebook_dim}) dtype={codebook_dtype}", local_rank)
    log(f"  first sample length: {len(first_record['input_ids'])}", local_rank)

    log("\n[2/5] Loading tokenizer and model...", local_rank)
    tokenizer = load_tokenizer_for_model(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_for_training(
        args.model_name_or_path,
        torch_dtype=model_dtype,
        deepseek_attn_implementation=args.deepseek_attn_implementation,
    )
    vocab_layout = align_model_embeddings_with_tokenizer(model, tokenizer)
    base_vocab_size = int(vocab_layout["embedding_vocab_size"])
    latent_min_id = base_vocab_size
    latent_max_id = base_vocab_size + args.num_latent_tokens - 1

    log(f"  model_path: {args.model_name_or_path}", local_rank)
    log(f"  model_type: {'VLM' if is_vlm_model(args.model_name_or_path) else 'LLM'}", local_rank)
    log(f"  tokenizer_vocab_size: {vocab_layout['tokenizer_vocab_size']}", local_rank)
    log(f"  config_vocab_size: {vocab_layout['config_vocab_size']}", local_rank)
    log(f"  base_vocab_size: {base_vocab_size}", local_rank)
    log(f"  reserved_token_count: {vocab_layout['reserved_token_count']}", local_rank)
    log(f"  resized_to_tokenizer: {vocab_layout['resized_to_tokenizer']}", local_rank)
    if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        log(f"  attn_implementation: {model.config._attn_implementation}", local_rank)
    log(f"  latent token range: [{latent_min_id}, {latent_max_id}]", local_rank)

    log("\n[3/5] Loading dataset...", local_rank)
    train_dataset = load_dataset("json", data_files=str(train_data_path), split="train")
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")

    log(f"  dataset size: {len(train_dataset)}", local_rank)
    sample = train_dataset[0]
    log(f"  sample sequence_length: {len(sample['input_ids'])}", local_rank)
    log(f"  sample labels_equal_input_ids: {sample['labels'] == sample['input_ids']}", local_rank)

    log("\n[4/5] Building trainer...", local_rank)
    from trl import LatentSFTConfig, LatentSFTTrainer

    ddp_find_unused_parameters = args.ddp_find_unused_parameters
    if ddp_find_unused_parameters is None:
        ddp_find_unused_parameters = True

    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "optim": args.optim,
        "optim_args": args.optim_args,
        "lr_scheduler_type": args.lr_scheduler_type,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "report_to": [],
        "disable_tqdm": False,
        "ddp_find_unused_parameters": ddp_find_unused_parameters,
        "num_latent_tokens": args.num_latent_tokens,
        "codebook_dim": args.codebook_dim,
        "codebook_path": args.codebook_path,
        "freeze_base_model": args.freeze_base_model,
        "freeze_codebook": args.freeze_codebook,
    }
    if args.deepspeed:
        training_args_dict["deepspeed"] = args.deepspeed
    if args.max_steps > 0:
        training_args_dict["max_steps"] = args.max_steps

    training_args = LatentSFTConfig(**training_args_dict)
    trainer = LatentSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    log(f"  freeze_base_model: {training_args.freeze_base_model}", local_rank)
    log(f"  freeze_codebook: {training_args.freeze_codebook}", local_rank)
    log(f"  deepspeed: {training_args.deepspeed}", local_rank)
    log(f"  gradient_checkpointing: {training_args.gradient_checkpointing}", local_rank)
    log(f"  ddp_find_unused_parameters: {training_args.ddp_find_unused_parameters}", local_rank)
    log(f"  expanded_vocab_size: {trainer.model.get_input_embeddings().num_embeddings}", local_rank)

    if is_main_process(local_rank):
        device = next(model.parameters()).device
        attn_implementation = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if (
            is_deepseek_decoder_model(args.model_name_or_path)
            and device.type != "cuda"
            and isinstance(attn_implementation, str)
            and attn_implementation.startswith("flash_attention_")
        ):
            print(
                "  skipping sanity forward: model is still on CPU before trainer launch, "
                f"and {attn_implementation} requires CUDA"
            )
        else:
            sanity_ids = sample["input_ids"][: min(len(sample["input_ids"]), 16)]
            sanity_input = torch.tensor([sanity_ids], device=device)
            with torch.no_grad():
                outputs = model(sanity_input)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            print(f"  sanity logits shape: {tuple(logits.shape)}")

    log("\n[5/5] Starting training...", local_rank)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    log("\nSaving final model...", local_rank)
    trainer.save_model(args.output_dir)

    log("\nTraining complete.", local_rank)
    log(f"  output_dir: {args.output_dir}", local_rank)
    log(f"  latent_dir: {args.output_dir}/latent", local_rank)


if __name__ == "__main__":
    main()
