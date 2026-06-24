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
Latent SFT entrypoint for processed datasets with question-only masked supervision.

Expected dataset format:
    {
        "input_ids": [...],
        "labels": [...],
        ...
    }

Fixed training format:
    input_ids = [question ids] + [Think prefix ids] + [latent ids] + [answer prefix ids] + [answer ids]
    labels    = [-100 ...] + [Think prefix ids] + [latent ids] + [answer prefix ids] + [answer ids]
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from trl.models.latent_utils import align_model_embeddings_with_tokenizer
from trl.models.local_model_utils import is_deepseek_decoder_model
from trl.models.local_model_utils import is_vlm_model as is_vlm_model_path
from trl.models.local_model_utils import load_model_for_training, load_tokenizer_for_model


DEFAULT_MODEL_PATH = "models/Qwen3-VL-4B-Instruct"
DEFAULT_CODEBOOK_PATH = "outputs/deepseek_codebook/latent/codebook.pt"
DEFAULT_PROJECTOR_PATH = "outputs/latent_pretrain/latent/projector.pt"
DEFAULT_TRAIN_DATA_PATH = "data/latent_sft.jsonl"
DEFAULT_OUTPUT_DIR = "./latent_sft_processed_output"
THINK_PREFIX = "\nThink:"
ANSWER_PREFIX = "\nSo the answer is: "


def is_main_process(local_rank: int) -> bool:
    """
    Check whether the current process should emit logs.

    Args:
        local_rank (`int`):
            Rank from launcher environment.

    Returns:
        `bool`:
            Whether this process is the main process.
    """
    return local_rank in (-1, 0)


def log(message: str, local_rank: int) -> None:
    """
    Print from the main process only.

    Args:
        message (`str`):
            Message to print.
        local_rank (`int`):
            Rank from launcher environment.
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
            Whether to use `AutoModelForImageTextToText`.
    """
    return is_vlm_model_path(model_name_or_path)


def load_model(model_name_or_path: str, torch_dtype: torch.dtype):
    """
    Load a local model using the appropriate auto class.

    Args:
        model_name_or_path (`str`):
            Local model path or model identifier.
        torch_dtype (`torch.dtype`):
            Loading dtype.

    Returns:
        `PreTrainedModel`:
            Loaded model instance.
    """
    return load_model_for_training(
        model_name_or_path,
        torch_dtype=torch_dtype,
    )


def inspect_tensor(path: str) -> tuple[int, ...]:
    """
    Load a tensor and return its shape.

    Args:
        path (`str`):
            Tensor path.

    Returns:
        `tuple[int, ...]`:
            Tensor shape.
    """
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, got {type(tensor)}")
    return tuple(tensor.shape)


def read_first_jsonl_record(path: Path) -> dict:
    """
    Read the first JSON object from a JSONL file.

    Args:
        path (`Path`):
            Input JSONL path.

    Returns:
        `dict`:
            First parsed record.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSON records found in {path}")


def build_latent_init_dir(codebook_path: str, projector_path: str, output_dir: str) -> str:
    """
    Materialize a `latent/` directory consumable by `LatentSFTTrainer`.

    Args:
        codebook_path (`str`):
            Standalone codebook `.pt`.
        projector_path (`str`):
            Standalone projector `.pt`.
        output_dir (`str`):
            Training output directory.

    Returns:
        `str`:
            Parent directory that contains a `latent/` subdirectory.
    """
    base_dir = Path(output_dir).resolve() / "latent_init"
    latent_dir = base_dir / "latent"
    latent_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(codebook_path, latent_dir / "codebook.pt")
    shutil.copy2(projector_path, latent_dir / "projector.pt")
    return str(base_dir)


def _extract_question_latent_answer(
    example: dict[str, Any],
    tokenizer,
    base_vocab_size: int,
) -> tuple[list[int], list[int], str, bool]:
    """
    Split an existing processed SFT sample into question tokens, latent tokens, and answer text.

    The input format is assumed to be the current processed layout:
        question_ids + latent_ids + answer_ids (+ eos)
    with labels:
        -100 for question, then supervised latent/answer tokens.
    """
    input_ids = example.get("input_ids")
    labels = example.get("labels")
    if not isinstance(input_ids, list) or not isinstance(labels, list):
        raise TypeError("Expected `input_ids` and `labels` to be lists.")
    if len(input_ids) != len(labels):
        raise ValueError("`input_ids` and `labels` must have the same length.")

    question_ids = [int(token_id) for token_id, label in zip(input_ids, labels, strict=True) if int(label) == -100]
    latent_ids = [
        int(token_id)
        for token_id, label in zip(input_ids, labels, strict=True)
        if int(label) != -100 and int(token_id) >= base_vocab_size
    ]
    answer_text_ids = [
        int(token_id)
        for token_id, label in zip(input_ids, labels, strict=True)
        if int(label) != -100
        and 0 <= int(token_id) < base_vocab_size
        and int(token_id) != tokenizer.eos_token_id
    ]

    original = example.get("original")
    if isinstance(original, dict) and original.get("answer") is not None:
        answer_text = str(original["answer"]).strip()
    else:
        answer_text = tokenizer.decode(answer_text_ids, skip_special_tokens=True).strip()

    has_eos = (
        tokenizer.eos_token_id is not None
        and len(input_ids) > 0
        and int(input_ids[-1]) == tokenizer.eos_token_id
        and int(labels[-1]) != -100
    )

    if not question_ids:
        raise ValueError("Could not recover question tokens from processed sample.")
    if not latent_ids:
        raise ValueError("Could not recover latent tokens from processed sample.")
    if not answer_text:
        raise ValueError("Could not recover answer text from processed sample.")

    return question_ids, latent_ids, answer_text, has_eos


def reformat_processed_sft_example(
    example: dict[str, Any],
    tokenizer,
    base_vocab_size: int,
) -> dict[str, Any]:
    """
    Reformat a processed SFT sample online without modifying the source dataset on disk.

    Target format:
        question + THINK_PREFIX + latent_ids + ANSWER_PREFIX + answer_text (+ eos)

    Loss mask:
        -100 for question only
        supervised for THINK_PREFIX + latent_ids + ANSWER_PREFIX + answer_text (+ eos)
    """
    question_ids, latent_ids, answer_text, has_eos = _extract_question_latent_answer(
        example,
        tokenizer=tokenizer,
        base_vocab_size=base_vocab_size,
    )

    think_prefix_ids = tokenizer.encode(THINK_PREFIX, add_special_tokens=False)
    answer_prefix_ids = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    input_ids = question_ids + think_prefix_ids + latent_ids + answer_prefix_ids + answer_ids
    labels = ([-100] * len(question_ids)) + think_prefix_ids + latent_ids + answer_prefix_ids + answer_ids

    if has_eos and tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    updated = dict(example)
    updated["input_ids"] = input_ids
    updated["labels"] = labels
    updated["sequence_length"] = len(input_ids)
    updated["prompt_token_count"] = len(question_ids)
    updated["latent_token_count"] = len(latent_ids)
    updated["completion_token_count"] = len(think_prefix_ids) + len(latent_ids) + len(answer_prefix_ids) + len(answer_ids)
    updated["answer_token_count"] = len(answer_prefix_ids) + len(answer_ids)
    updated["think_prefix"] = THINK_PREFIX
    updated["answer_prefix"] = ANSWER_PREFIX
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--codebook_path", type=str, default=DEFAULT_CODEBOOK_PATH)
    parser.add_argument("--projector_path", type=str, default=DEFAULT_PROJECTOR_PATH)
    parser.add_argument("--train_data_path", type=str, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_train_samples", type=int, default=None)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--optim_args", type=str, default="foreach=False")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp_find_unused_parameters", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze_base_model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze_codebook", action=argparse.BooleanOptionalAction, default=False)
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
    log("Latent SFT (Processed Dataset)", local_rank)
    log("=" * 80, local_rank)

    log("\n[1/6] Inspecting initialization tensors...", local_rank)
    if not Path(args.codebook_path).exists():
        raise FileNotFoundError(f"Codebook not found: {args.codebook_path}")
    if not Path(args.projector_path).exists():
        raise FileNotFoundError(f"Projector not found: {args.projector_path}")
    codebook_shape = inspect_tensor(args.codebook_path)
    projector_shape = inspect_tensor(args.projector_path)
    if len(codebook_shape) != 2:
        raise ValueError(f"Codebook must be 2D, got shape {codebook_shape}")
    if len(projector_shape) != 2:
        raise ValueError(f"Projector must be 2D, got shape {projector_shape}")

    num_latent_tokens, codebook_dim = codebook_shape
    if projector_shape[0] != codebook_dim:
        raise ValueError(
            f"Projector input dim {projector_shape[0]} does not match codebook dim {codebook_dim}"
        )

    log(f"  codebook_path: {args.codebook_path}", local_rank)
    log(f"  codebook shape: {codebook_shape}", local_rank)
    log(f"  projector_path: {args.projector_path}", local_rank)
    log(f"  projector shape: {projector_shape}", local_rank)

    log("\n[2/6] Inspecting dataset...", local_rank)
    train_data_path = Path(args.train_data_path)
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")
    first_record = read_first_jsonl_record(train_data_path)
    for required_key in ("input_ids", "labels"):
        if required_key not in first_record:
            raise KeyError(f"Processed dataset must contain `{required_key}`")
    log(f"  train_data_path: {train_data_path}", local_rank)
    log(f"  first sample length: {len(first_record['input_ids'])}", local_rank)
    log(f"  first sample masked question tokens: {sum(int(x == -100) for x in first_record['labels'])}", local_rank)

    log("\n[3/6] Loading tokenizer and model...", local_rank)
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
    if projector_shape[1] != model.get_input_embeddings().embedding_dim:
        raise ValueError(
            f"Projector output dim {projector_shape[1]} does not match model hidden size "
            f"{model.get_input_embeddings().embedding_dim}"
        )
    log(f"  model_path: {args.model_name_or_path}", local_rank)
    log(f"  model_type: {'VLM' if is_vlm_model(args.model_name_or_path) else 'LLM'}", local_rank)
    log(f"  tokenizer_vocab_size: {vocab_layout['tokenizer_vocab_size']}", local_rank)
    log(f"  config_vocab_size: {vocab_layout['config_vocab_size']}", local_rank)
    log(f"  base_vocab_size: {base_vocab_size}", local_rank)
    log(f"  reserved_token_count: {vocab_layout['reserved_token_count']}", local_rank)
    log(f"  resized_to_tokenizer: {vocab_layout['resized_to_tokenizer']}", local_rank)
    if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        log(f"  attn_implementation: {model.config._attn_implementation}", local_rank)
    log(f"  latent token range: [{base_vocab_size}, {base_vocab_size + num_latent_tokens - 1}]", local_rank)

    log("\n[4/6] Loading processed dataset...", local_rank)
    train_dataset = load_dataset("json", data_files=str(train_data_path), split="train")
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    train_dataset = train_dataset.map(
        reformat_processed_sft_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "base_vocab_size": base_vocab_size,
        },
        desc="Injecting Think/Answer format into processed SFT data",
    )
    sample = train_dataset[0]
    log(f"  dataset size: {len(train_dataset)}", local_rank)
    log(f"  sample sequence_length: {len(sample['input_ids'])}", local_rank)
    log(f"  think_prefix: {THINK_PREFIX!r}", local_rank)
    log(f"  answer_prefix: {ANSWER_PREFIX!r}", local_rank)
    sample_prompt_ids = [
        int(token_id)
        for token_id, label in zip(sample["input_ids"], sample["labels"], strict=True)
        if int(label) == -100 and 0 <= int(token_id) < base_vocab_size
    ]
    sample_completion_ids = [
        int(token_id)
        for token_id, label in zip(sample["input_ids"], sample["labels"], strict=True)
        if int(label) != -100 and 0 <= int(token_id) < base_vocab_size
    ]
    log(f"  sample prompt text: {tokenizer.decode(sample_prompt_ids, skip_special_tokens=True)!r}", local_rank)
    log(
        f"  sample completion text: {tokenizer.decode(sample_completion_ids, skip_special_tokens=True)!r}",
        local_rank,
    )

    log("\n[5/6] Building latent init directory and trainer...", local_rank)
    latent_init_root = build_latent_init_dir(args.codebook_path, args.projector_path, args.output_dir)
    log(f"  latent_init_root: {latent_init_root}", local_rank)

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
        "num_latent_tokens": num_latent_tokens,
        "codebook_dim": codebook_dim,
        "codebook_path": latent_init_root,
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

    log("\n[6/6] Starting training...", local_rank)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    log("\nSaving final model...", local_rank)
    trainer.save_model(args.output_dir)

    log("\nTraining complete.", local_rank)
    log(f"  output_dir: {args.output_dir}", local_rank)
    log(f"  latent_dir: {args.output_dir}/latent", local_rank)


if __name__ == "__main__":
    main()
