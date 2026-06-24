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
Convert Source2 latent data into processed SFT examples.

Target training objective:
    input_ids = [question text ids] + [latent token ids] + [answer text ids] (+ eos)
    labels    = [-100 for question] + [latent token ids] + [answer text ids] (+ eos)

This format is designed to be consumed directly by `LatentSFTTrainer` through a
processed dataset containing `input_ids` and `labels`.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from trl.models.latent_utils import infer_base_vocab_size_from_config_and_tokenizer
from trl.models.local_model_utils import load_model_config, load_tokenizer_for_model


DEFAULT_MODEL_PATH = "models/Qwen3-VL-4B-Instruct"
DEFAULT_INPUT_JSONL = "outputs/source2_codebook_input_ids.jsonl"
DEFAULT_OUTPUT_JSONL = "data/latent_sft.jsonl"


def iter_jsonl(path: Path):
    """
    Yield JSON objects from a JSONL file.

    Args:
        path (`Path`):
            Input JSONL path.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_latent_ids(example: dict[str, Any]) -> list[int]:
    """
    Read latent ids from any of the supported source formats.

    Args:
        example (`dict[str, Any]`):
            Input example.

    Returns:
        `list[int]`:
            Latent ids, either raw codebook indices or already-shifted token ids.
    """
    for key in ("shifted_latent_token_ids", "image_codebook_input_ids", "raw_latent_ids"):
        if key in example:
            return [int(x) for x in example[key]]
    raise KeyError("Expected one of: shifted_latent_token_ids/image_codebook_input_ids/raw_latent_ids")


def maybe_shift_latent_ids(latent_ids: list[int], base_vocab_size: int, num_latent_tokens: int) -> list[int]:
    """
    Convert raw codebook indices into shifted latent token ids when needed.

    Args:
        latent_ids (`list[int]`):
            Raw codebook indices or already-shifted token ids.
        base_vocab_size (`int`):
            Base language-model vocabulary size.
        num_latent_tokens (`int`):
            Number of latent tokens.

    Returns:
        `list[int]`:
            Shifted latent token ids.
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
        f"Latent ids are outside both raw and shifted ranges: min={min_id}, max={max_id}, "
        f"raw<[0,{num_latent_tokens - 1}] shifted=[{valid_min},{valid_max}]"
    )


def get_question_and_answer(example: dict[str, Any]) -> tuple[str, str]:
    """
    Extract question and answer strings from a source example.

    Args:
        example (`dict[str, Any]`):
            Input example.

    Returns:
        `tuple[str, str]`:
            `(question, answer)`
    """
    original = example.get("original", {})
    question = example.get("question", original.get("question", ""))
    answer = example.get("answer", original.get("answer", ""))
    return question or "", answer or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input_jsonl", type=str, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output_jsonl", type=str, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--num_latent_tokens", type=int, default=10000)
    parser.add_argument("--append_eos", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--answer_prefix", type=str, default="\n")
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
    skipped_examples = 0
    total_length = 0
    min_length = None
    max_length_seen = None

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for example in iter_jsonl(input_jsonl):
            question, answer = get_question_and_answer(example)
            if not question or not answer:
                skipped_examples += 1
                continue

            latent_ids = maybe_shift_latent_ids(
                get_latent_ids(example),
                base_vocab_size=base_vocab_size,
                num_latent_tokens=args.num_latent_tokens,
            )

            question_ids = tokenizer.encode(question, add_special_tokens=True)
            answer_text = f"{args.answer_prefix}{answer}" if args.answer_prefix else answer
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

            input_ids = question_ids + latent_ids + answer_ids
            labels = ([-100] * len(question_ids)) + latent_ids + answer_ids

            if args.append_eos and eos_token_id is not None:
                input_ids.append(eos_token_id)
                labels.append(eos_token_id)

            if args.max_length is not None and len(input_ids) > args.max_length:
                input_ids = input_ids[: args.max_length]
                labels = labels[: args.max_length]
                truncated_examples += 1

            if all(label == -100 for label in labels):
                skipped_examples += 1
                continue

            output = {
                "id": example.get("id"),
                "input_ids": input_ids,
                "labels": labels,
                "question_token_count": len(question_ids),
                "latent_token_count": len(latent_ids),
                "answer_token_count": len(answer_ids),
                "sequence_length": len(input_ids),
            }
            if not args.drop_metadata:
                output["source"] = example.get("source")
                output["checkpoint"] = example.get("checkpoint")
                output["codebook_placeholder_id"] = example.get("codebook_placeholder_id")
                output["original"] = example.get("original")
                output["shifted_latent_token_ids"] = latent_ids

            fout.write(json.dumps(output, ensure_ascii=False) + "\n")

            num_examples += 1
            seq_len = len(input_ids)
            total_length += seq_len
            min_length = seq_len if min_length is None else min(min_length, seq_len)
            max_length_seen = seq_len if max_length_seen is None else max(max_length_seen, seq_len)

            if num_examples % 5000 == 0:
                print(f"Processed {num_examples} examples...")
            if args.max_samples is not None and num_examples >= args.max_samples:
                break

    avg_length = total_length / num_examples if num_examples else 0.0
    print("=" * 60)
    print(f"Saved examples: {num_examples}")
    print(f"Skipped examples: {skipped_examples}")
    print(f"Truncated examples: {truncated_examples}")
    print(f"Sequence length min/avg/max: {min_length} / {avg_length:.2f} / {max_length_seen}")
    print("=" * 60)


if __name__ == "__main__":
    main()
