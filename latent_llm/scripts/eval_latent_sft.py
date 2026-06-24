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
Evaluate a latent SFT checkpoint on a parquet test set.

Evaluation protocol:
1. Load the saved base model from the checkpoint directory.
2. Re-attach latent parameters from `checkpoint/latent/` using
   `extend_model_for_latent_tokens`.
3. Feed only the plain-text question as input.
4. Let the model autoregressively generate a sequence that may contain both
   latent token ids and normal text token ids.
5. Save raw generated ids and mixed decoded output for downstream analysis.

The script supports both single-GPU execution and multi-GPU execution through
`torchrun --nproc_per_node=N ...`. In distributed mode, each rank processes a
disjoint shard of the test set independently, one sample at a time, and rank 0
merges the shard files at the end.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from tqdm.auto import tqdm

from trl.models.latent_utils import align_model_embeddings_with_tokenizer, extend_model_for_latent_tokens


DEFAULT_CKPT_PATH = "outputs/latent_grpo/checkpoint-last"
DEFAULT_TEST_PATH = "data/eval.parquet"
DEFAULT_OUTPUT_PATH = "outputs/latent_eval_outputs.jsonl"
DEFAULT_MAX_EXAMPLES = None
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 42
DEFAULT_BF16 = True


def is_main_process(rank: int) -> bool:
    return rank in (-1, 0)


def log(message: str, rank: int) -> None:
    if is_main_process(rank):
        print(message)


def maybe_init_distributed() -> tuple[int, int, torch.device]:
    """
    Initialize distributed execution when launched with `torchrun`.

    Returns:
        `tuple[int, int, torch.device]`:
            `(rank, world_size, device)`
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    if world_size == 1 and local_rank == -1:
        rank = 0

    return rank, world_size, device


def barrier_if_needed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


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
    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    architecture = config.architectures[0] if getattr(config, "architectures", None) else ""
    vlm_arches = {
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "MllamaForConditionalGeneration",
    }
    vlm_model_types = {"qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "mllama"}
    return architecture in vlm_arches or getattr(config, "model_type", "") in vlm_model_types


def load_model(model_name_or_path: str, torch_dtype: torch.dtype):
    """
    Load a model checkpoint through the appropriate auto class.

    Args:
        model_name_or_path (`str`):
            Checkpoint directory.
        torch_dtype (`torch.dtype`):
            Loading dtype.

    Returns:
        `PreTrainedModel`:
            Loaded model.
    """
    model_cls = AutoModelForImageTextToText if is_vlm_model(model_name_or_path) else AutoModelForCausalLM
    return model_cls.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )


def load_latent_config(ckpt_path: Path) -> dict[str, Any]:
    """
    Read latent config metadata from a checkpoint directory.

    Args:
        ckpt_path (`Path`):
            Checkpoint directory containing `latent/config.json`.

    Returns:
        `dict[str, Any]`:
            Parsed latent config.
    """
    config_path = ckpt_path / "latent" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing latent config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_mixed_generation(token_ids: list[int], tokenizer, base_vocab_size: int) -> tuple[list[Any], str]:
    """
    Decode a mixed sequence containing both latent ids and text ids.

    Text spans are merged into strings, while latent ids are kept as integers.

    Args:
        token_ids (`list[int]`):
            Generated token ids.
        tokenizer:
            Tokenizer used for decoding text ids.
        base_vocab_size (`int`):
            All ids >= this threshold are treated as latent ids.

    Returns:
        `tuple[list[Any], str]`:
            `(mixed_items, text_only)` where:
            - `mixed_items` preserves order using strings for text spans and ints for latent ids
            - `text_only` decodes only base-vocab ids
    """
    mixed_items = []
    text_buffer = []
    text_only_ids = []

    def flush_text_buffer():
        if text_buffer:
            mixed_items.append(tokenizer.decode(text_buffer, skip_special_tokens=False))
            text_buffer.clear()

    for token_id in token_ids:
        if token_id < base_vocab_size:
            text_buffer.append(token_id)
            text_only_ids.append(token_id)
        else:
            flush_text_buffer()
            mixed_items.append(token_id)

    flush_text_buffer()
    text_only = tokenizer.decode(text_only_ids, skip_special_tokens=True)
    return mixed_items, text_only


def make_serializable(value: Any) -> Any:
    """
    Convert pandas / numpy values into JSON-serializable Python objects.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict, str, int, float, bool)):
        return value
    if hasattr(value, "tolist") and not isinstance(value, str):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def shard_output_path(output_path: Path, rank: int) -> Path:
    return output_path.parent / f"{output_path.stem}.rank{rank:02d}{output_path.suffix}"


def merge_rank_outputs(output_path: Path, world_size: int) -> None:
    """
    Merge per-rank JSONL shard files into a single output file ordered by index.
    """
    records = []
    shard_paths = [shard_output_path(output_path, rank) for rank in range(world_size)]

    for path in shard_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    records.sort(key=lambda record: record["index"])

    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    for path in shard_paths:
        if path.exists():
            path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max_examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=DEFAULT_BF16)
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("This script runs in independent per-rank mode. Please set --batch_size 1.")

    rank, world_size, device = maybe_init_distributed()
    torch.manual_seed(args.seed + max(rank, 0))

    ckpt_path = Path(args.ckpt_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rank_output_path = shard_output_path(output_path, rank)

    log("=" * 80, rank)
    log("Latent SFT Evaluation", rank)
    log("=" * 80, rank)
    log(f"Checkpoint: {ckpt_path}", rank)
    log(f"Test data: {args.test_path}", rank)
    log(f"Output: {output_path}", rank)
    log(f"Model type: {'VLM' if is_vlm_model(str(ckpt_path)) else 'LLM'}", rank)
    log("Prompt format: question only", rank)
    log(f"World size: {world_size}", rank)
    log("Per-rank mode: single-sample decoding without padding", rank)
    log(f"Device: {device}", rank)

    print(f"[rank {rank}] Loading tokenizer from: {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[rank {rank}] Loading base model from: {ckpt_path}")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model(str(ckpt_path), torch_dtype=torch_dtype)
    vocab_layout = align_model_embeddings_with_tokenizer(model, tokenizer)
    log(
        f"[rank {rank}] tokenizer_vocab_size={vocab_layout['tokenizer_vocab_size']} "
        f"embedding_vocab_size={vocab_layout['embedding_vocab_size']} "
        f"reserved_token_count={vocab_layout['reserved_token_count']} "
        f"resized_to_tokenizer={vocab_layout['resized_to_tokenizer']}",
        rank,
    )
    model = model.to(device)

    latent_config = load_latent_config(ckpt_path)
    base_vocab_size = latent_config["base_vocab_size"]
    num_latent_tokens = latent_config["num_latent_tokens"]
    codebook_dim = latent_config["codebook_dim"]

    print(f"[rank {rank}] Re-attaching latent weights from: {ckpt_path / 'latent'}")
    model, _, _ = extend_model_for_latent_tokens(
        model=model,
        num_latent_tokens=num_latent_tokens,
        codebook_dim=codebook_dim,
        codebook_path=str(ckpt_path),
    )
    model.eval()

    print(f"[rank {rank}] Reading test set: {args.test_path}")
    df = pd.read_parquet(args.test_path)
    if args.max_examples is not None:
        df = df.iloc[: args.max_examples]
    all_items = list(df.iterrows())
    sharded_items = all_items[rank::world_size]

    print(f"[rank {rank}] Assigned examples: {len(sharded_items)} / {len(all_items)}")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    else:
        generation_kwargs["do_sample"] = False

    progress = tqdm(
        total=len(sharded_items),
        desc=f"rank{rank}",
        position=max(rank, 0),
        dynamic_ncols=True,
        leave=True,
    )

    with rank_output_path.open("w", encoding="utf-8") as fout:
        for idx, row in sharded_items:
            question = str(row["question"])
            prompt_text = question
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            input_ids = torch.tensor([prompt_ids], device=device)

            with torch.no_grad():
                output_ids = model.generate(input_ids=input_ids, **generation_kwargs)

            generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
            mixed_items, text_only = split_mixed_generation(generated_ids, tokenizer, base_vocab_size)
            record = {
                "index": int(idx),
                "question": make_serializable(row.get("question")),
                "answer": make_serializable(row.get("answer")),
                "steps": make_serializable(row.get("steps")),
                "prompt_ids": prompt_ids,
                "generated_ids": generated_ids,
                "generated_mixed": mixed_items,
                "generated_text_only": text_only,
                "base_vocab_size": base_vocab_size,
                "num_latent_tokens": num_latent_tokens,
                "ckpt_path": str(ckpt_path),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.update(1)
    progress.close()

    print(f"[rank {rank}] Finished shard: {rank_output_path}")

    barrier_if_needed(world_size)
    if rank == 0:
        print("Merging shard outputs...")
        merge_rank_outputs(output_path, world_size)
        print("=" * 60)
        print(f"Saved generation outputs to: {output_path}")
        print("=" * 60)
    barrier_if_needed(world_size)
    cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
