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
Convert Source2 codebook-id data into processed latent-pretraining samples.

This script produces a processed JSONL dataset that can be consumed directly by
TRL/Transformers trainers expecting tokenized examples with `input_ids` and
`labels`.

Sequence format:
    [latent token ids] + [CoT text token ids] + [answer text token ids] (+ eos)

Loss format:
    labels = input_ids

The codebook ids from the source file are treated as latent vocabulary indices
and shifted by the base vocabulary size of the target language model.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from trl.models.latent_utils import infer_base_vocab_size_from_config_and_tokenizer
from trl.models.local_model_utils import load_model_config, load_tokenizer_for_model


DEFAULT_MODEL_PATH = "models/Qwen3-VL-4B-Instruct"
DEFAULT_INPUT_JSONL = "outputs/source2_codebook_input_ids.jsonl"
DEFAULT_OUTPUT_JSONL = "data/latent_pretrain.jsonl"


def iter_jsonl(path: Path):
    """
    Yield decoded JSON objects from a JSONL file.

    Args:
        path (`Path`):
            Input JSONL path.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_text_segment(example: dict[str, Any], answer_prefix: str) -> str:
    """
    Build the text continuation appended after latent token ids.

    Args:
        example (`dict[str, Any]`):
            Source example.
        answer_prefix (`str`):
            Prefix inserted before the answer text when an answer is present.

    Returns:
        `str`:
            Text segment to tokenize after the latent token prefix.
    """
    cot_text = example.get("text")
    if cot_text is None:
        cot_text = example.get("original", {}).get("cot", "")
    answer = example.get("original", {}).get("answer", "")

    text = cot_text or ""
    if answer:
        text = f"{text}{answer_prefix}{answer}" if text else answer
    return text


def maybe_shift_latent_ids(latent_ids: list[int], base_vocab_size: int, num_latent_tokens: int) -> list[int]:
    """
    Convert raw codebook indices into expanded-vocabulary token ids.

    If the ids already appear to be in token-id space, they are returned as-is.

    Args:
        latent_ids (`list[int]`):
            Raw ids from the source example.
        base_vocab_size (`int`):
            Base language-model vocabulary size.
        num_latent_tokens (`int`):
            Number of latent tokens in the codebook.

    Returns:
        `list[int]`:
            Latent token ids in the expanded vocabulary range.
    """
    if not latent_ids:
        return []

    min_id = min(latent_ids)
    max_id = max(latent_ids)

    if 0 <= min_id and max_id < num_latent_tokens:
        return [base_vocab_size + token_id for token_id in latent_ids]

    valid_min = base_vocab_size
    valid_max = base_vocab_size + num_latent_tokens - 1
    if valid_min <= min_id and max_id <= valid_max:
        return latent_ids

    raise ValueError(
        f"Latent ids are neither raw codebook indices nor shifted token ids: "
        f"min={min_id}, max={max_id}, expected raw < {num_latent_tokens} or "
        f"shifted in [{valid_min}, {valid_max}]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input_jsonl", type=str, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output_jsonl", type=str, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--latent_field", type=str, default="image_codebook_input_ids")
    parser.add_argument("--answer_prefix", type=str, default="\n")
    parser.add_argument("--num_latent_tokens", type=int, default=10000)
    parser.add_argument("--append_eos", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--drop_metadata", action="store_true", default=False)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)

    tokenizer = load_tokenizer_for_model(args.model_name_or_path)
    config = load_model_config(args.model_name_or_path)
    base_vocab_size = infer_base_vocab_size_from_config_and_tokenizer(config, tokenizer)
    eos_token_id = tokenizer.eos_token_id

    print(f"Model path: {args.model_name_or_path}")
    print(f"Base vocab size: {base_vocab_size}")
    print(f"Input: {input_jsonl}")
    print(f"Output: {output_jsonl}")

    num_examples = 0
    truncated_examples = 0
    total_length = 0
    min_length = None
    max_length_seen = None
    max_latent_id = None

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with input_jsonl.open("r", encoding="utf-8") as fin, output_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            example = json.loads(line)
            raw_latent_ids = [int(x) for x in example.get(args.latent_field, [])]
            latent_token_ids = maybe_shift_latent_ids(raw_latent_ids, base_vocab_size, args.num_latent_tokens)

            text_segment = build_text_segment(example, args.answer_prefix)
            text_token_ids = tokenizer.encode(text_segment, add_special_tokens=False)

            input_ids = latent_token_ids + text_token_ids
            if args.append_eos and eos_token_id is not None:
                input_ids.append(eos_token_id)

            if args.max_length is not None and len(input_ids) > args.max_length:
                input_ids = input_ids[: args.max_length]
                truncated_examples += 1

            labels = input_ids.copy()

            output = {
                "id": example.get("id"),
                "input_ids": input_ids,
                "labels": labels,
                "latent_token_count": len(latent_token_ids),
                "text_token_count": len(text_token_ids),
                "sequence_length": len(input_ids),
            }
            if not args.drop_metadata:
                output["source"] = example.get("source")
                output["checkpoint"] = example.get("checkpoint")
                output["codebook_placeholder_id"] = example.get("codebook_placeholder_id")
                output["original"] = example.get("original")
                output["text"] = example.get("text")
                output["raw_latent_ids"] = raw_latent_ids
                output["shifted_latent_token_ids"] = latent_token_ids

            fout.write(json.dumps(output, ensure_ascii=False) + "\n")

            num_examples += 1
            seq_len = len(input_ids)
            total_length += seq_len
            min_length = seq_len if min_length is None else min(min_length, seq_len)
            max_length_seen = seq_len if max_length_seen is None else max(max_length_seen, seq_len)
            if latent_token_ids:
                local_max_latent = max(latent_token_ids)
                max_latent_id = local_max_latent if max_latent_id is None else max(max_latent_id, local_max_latent)

            if args.max_samples is not None and num_examples >= args.max_samples:
                break

            if num_examples % 5000 == 0:
                print(f"Processed {num_examples} examples...")

    avg_length = total_length / num_examples if num_examples else 0.0
    print("=" * 60)
    print(f"Saved examples: {num_examples}")
    print(f"Truncated examples: {truncated_examples}")
    print(f"Sequence length min/avg/max: {min_length} / {avg_length:.2f} / {max_length_seen}")
    print(f"Max shifted latent token id: {max_latent_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
