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
Retokenize an existing processed latent-SFT dataset for a different target model.

Expected source layout:
    input_ids = [question ids] + [latent ids] + [answer ids] (+ eos)
    labels    = [-100 for question] + [latent ids] + [answer ids] (+ eos)

Target layout keeps the same semantics, but re-encodes question/answer text with
the target tokenizer and remaps latent token ids to the target model's base vocab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trl.models.latent_utils import infer_base_vocab_size_from_config_and_tokenizer
from trl.models.local_model_utils import load_model_config, load_tokenizer_for_model


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def infer_source_has_eos(example: dict[str, Any]) -> bool:
    sequence_length = example.get("sequence_length")
    question_token_count = example.get("question_token_count")
    latent_token_count = example.get("latent_token_count")
    answer_token_count = example.get("answer_token_count")
    if all(isinstance(x, int) for x in [sequence_length, question_token_count, latent_token_count, answer_token_count]):
        return sequence_length == question_token_count + latent_token_count + answer_token_count + 1
    return False


def recover_raw_latent_ids(example: dict[str, Any], source_base_vocab_size: int) -> list[int]:
    shifted_latent_ids = example.get("shifted_latent_token_ids")
    if not isinstance(shifted_latent_ids, list):
        raise KeyError("Expected `shifted_latent_token_ids` in each processed SFT example.")
    raw_latent_ids = [int(token_id) - source_base_vocab_size for token_id in shifted_latent_ids]
    if raw_latent_ids and min(raw_latent_ids) < 0:
        raise ValueError(
            f"Recovered negative latent ids from source_base_vocab_size={source_base_vocab_size}: "
            f"{raw_latent_ids[:10]}"
        )
    return raw_latent_ids


def get_question_and_answer(example: dict[str, Any]) -> tuple[str, str]:
    original = example.get("original")
    if isinstance(original, dict):
        question = original.get("question")
        answer = original.get("answer")
        if question is not None and answer is not None:
            return str(question), str(answer)
    raise KeyError("Expected `original.question` and `original.answer` in the source dataset.")


def build_output_record(
    example: dict[str, Any],
    *,
    tokenizer,
    target_base_vocab_size: int,
    source_base_vocab_size: int,
    num_latent_tokens: int,
    append_eos: bool,
    answer_prefix: str,
) -> dict[str, Any]:
    question, answer = get_question_and_answer(example)
    raw_latent_ids = recover_raw_latent_ids(example, source_base_vocab_size)
    if raw_latent_ids and max(raw_latent_ids) >= num_latent_tokens:
        raise ValueError(
            f"Raw latent ids exceed num_latent_tokens={num_latent_tokens}: max={max(raw_latent_ids)}"
        )

    question_ids = tokenizer.encode(question, add_special_tokens=True)
    answer_text = f"{answer_prefix}{answer}" if answer_prefix else answer
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    shifted_latent_token_ids = [target_base_vocab_size + token_id for token_id in raw_latent_ids]

    input_ids = question_ids + shifted_latent_token_ids + answer_ids
    labels = ([-100] * len(question_ids)) + shifted_latent_token_ids + answer_ids

    if append_eos and tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    output = dict(example)
    output["input_ids"] = input_ids
    output["labels"] = labels
    output["shifted_latent_token_ids"] = shifted_latent_token_ids
    output["question_token_count"] = len(question_ids)
    output["latent_token_count"] = len(shifted_latent_token_ids)
    output["answer_token_count"] = len(answer_ids)
    output["sequence_length"] = len(input_ids)
    output["source_base_vocab_size"] = source_base_vocab_size
    output["target_base_vocab_size"] = target_base_vocab_size
    output["target_model_name_or_path"] = tokenizer.name_or_path
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_jsonl", type=str, required=True)
    parser.add_argument("--target_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--source_base_vocab_size", type=int, default=151936)
    parser.add_argument("--num_latent_tokens", type=int, default=10000)
    parser.add_argument("--answer_prefix", type=str, default="\n")
    parser.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    source_jsonl = Path(args.source_jsonl)
    output_jsonl = Path(args.output_jsonl)
    if not source_jsonl.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_jsonl}")

    tokenizer = load_tokenizer_for_model(args.target_model_name_or_path)
    config = load_model_config(args.target_model_name_or_path)
    target_base_vocab_size = infer_base_vocab_size_from_config_and_tokenizer(config, tokenizer)

    print(f"Source dataset: {source_jsonl}")
    print(f"Target model: {args.target_model_name_or_path}")
    print(f"Source base vocab size: {args.source_base_vocab_size}")
    print(f"Target base vocab size: {target_base_vocab_size}")
    print(f"Target latent range: [{target_base_vocab_size}, {target_base_vocab_size + args.num_latent_tokens - 1}]")
    print(f"Output dataset: {output_jsonl}")

    inferred_append_eos = None
    num_examples = 0
    total_length = 0
    min_length = None
    max_length = None

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for example in iter_jsonl(source_jsonl):
            if args.append_eos is None:
                if inferred_append_eos is None:
                    inferred_append_eos = infer_source_has_eos(example)
                    print(f"Inferred append_eos from source dataset: {inferred_append_eos}")
                append_eos = inferred_append_eos
            else:
                append_eos = args.append_eos

            output = build_output_record(
                example,
                tokenizer=tokenizer,
                target_base_vocab_size=target_base_vocab_size,
                source_base_vocab_size=args.source_base_vocab_size,
                num_latent_tokens=args.num_latent_tokens,
                append_eos=append_eos,
                answer_prefix=args.answer_prefix,
            )
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")

            seq_len = output["sequence_length"]
            num_examples += 1
            total_length += seq_len
            min_length = seq_len if min_length is None else min(min_length, seq_len)
            max_length = seq_len if max_length is None else max(max_length, seq_len)

            if num_examples % 5000 == 0:
                print(f"Processed {num_examples} examples...")
            if args.max_samples is not None and num_examples >= args.max_samples:
                break

    avg_length = total_length / num_examples if num_examples else 0.0
    print("=" * 60)
    print(f"Saved examples: {num_examples}")
    print(f"Sequence length min/avg/max: {min_length} / {avg_length:.2f} / {max_length}")
    print("=" * 60)


if __name__ == "__main__":
    main()
