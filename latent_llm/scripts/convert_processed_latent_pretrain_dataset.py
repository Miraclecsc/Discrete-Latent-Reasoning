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
Retokenize an existing processed latent-pretrain dataset for a different model.

Expected source format:
    {
        "input_ids": [...],
        "labels": [...],
        "raw_latent_ids": [...],          # preferred
        "shifted_latent_token_ids": [...],# accepted fallback
        "text": "...",                    # preferred
        ...
    }

Target format:
    [shifted latent token ids for target model] + [target-model text token ids] (+ eos)
"""

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


def get_raw_latent_ids(example: dict[str, Any], source_base_vocab_size: int | None) -> list[int]:
    if "raw_latent_ids" in example and example["raw_latent_ids"] is not None:
        return [int(x) for x in example["raw_latent_ids"]]

    shifted = example.get("shifted_latent_token_ids")
    if shifted is not None:
        shifted = [int(x) for x in shifted]
        if source_base_vocab_size is None:
            raise ValueError("Source base vocab size is required to recover raw latent ids from shifted ids.")
        return [token_id - source_base_vocab_size for token_id in shifted]

    raise KeyError("Expected `raw_latent_ids` or `shifted_latent_token_ids` in each example.")


def infer_source_has_eos(example: dict[str, Any]) -> bool:
    sequence_length = example.get("sequence_length")
    latent_token_count = example.get("latent_token_count")
    text_token_count = example.get("text_token_count")
    if isinstance(sequence_length, int) and isinstance(latent_token_count, int) and isinstance(text_token_count, int):
        return sequence_length == latent_token_count + text_token_count + 1
    return False


def build_output_record(
    example: dict[str, Any],
    *,
    tokenizer,
    target_base_vocab_size: int,
    num_latent_tokens: int,
    append_eos: bool,
):
    text = example.get("text")
    if text is None:
        raise KeyError("Expected `text` in source dataset for pretrain retokenization.")

    raw_latent_ids = get_raw_latent_ids(example, source_base_vocab_size=example.get("source_base_vocab_size"))
    if raw_latent_ids:
        min_id = min(raw_latent_ids)
        max_id = max(raw_latent_ids)
        if min_id < 0 or max_id >= num_latent_tokens:
            raise ValueError(
                f"Raw latent ids out of range: min={min_id}, max={max_id}, expected [0, {num_latent_tokens - 1}]"
            )

    latent_token_ids = [target_base_vocab_size + token_id for token_id in raw_latent_ids]
    text_token_ids = tokenizer.encode(str(text), add_special_tokens=False)

    input_ids = latent_token_ids + text_token_ids
    if append_eos and tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)

    output = dict(example)
    output["input_ids"] = input_ids
    output["labels"] = input_ids.copy()
    output["raw_latent_ids"] = raw_latent_ids
    output["shifted_latent_token_ids"] = latent_token_ids
    output["latent_token_count"] = len(latent_token_ids)
    output["text_token_count"] = len(text_token_ids)
    output["sequence_length"] = len(input_ids)
    output["target_model_name_or_path"] = tokenizer.name_or_path
    output["target_base_vocab_size"] = target_base_vocab_size
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_jsonl", type=str, required=True)
    parser.add_argument("--target_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--num_latent_tokens", type=int, default=10000)
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
    print(f"Target base vocab size: {target_base_vocab_size}")
    print(f"Target latent range: [{target_base_vocab_size}, {target_base_vocab_size + args.num_latent_tokens - 1}]")
    print(f"Output dataset: {output_jsonl}")

    num_examples = 0
    total_length = 0
    min_length = None
    max_length = None
    inferred_append_eos = None

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

            if "source_base_vocab_size" not in example:
                source_base_vocab_size = example.get("target_base_vocab_size")
                if source_base_vocab_size is None:
                    shifted = example.get("shifted_latent_token_ids")
                    raw = example.get("raw_latent_ids")
                    if shifted is not None and raw is not None and len(shifted) == len(raw) and len(raw) > 0:
                        source_base_vocab_size = int(shifted[0]) - int(raw[0])
                if source_base_vocab_size is not None:
                    example = dict(example)
                    example["source_base_vocab_size"] = int(source_base_vocab_size)

            output = build_output_record(
                example,
                tokenizer=tokenizer,
                target_base_vocab_size=target_base_vocab_size,
                num_latent_tokens=args.num_latent_tokens,
                append_eos=append_eos,
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
