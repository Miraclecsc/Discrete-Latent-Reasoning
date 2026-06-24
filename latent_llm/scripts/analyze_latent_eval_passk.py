#!/usr/bin/env python
"""
Compute sample-level accuracy and pass@k accuracy for multi-sample latent-eval outputs.

Each record is expected to contain:
    - index
    - sample_index
    - generated_text_only
    - answer

The same answer-matching logic as `analyze_latent_eval_accuracy.py` is reused.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_latent_eval_accuracy import (
    DEFAULT_INPUT_PATH,
    extract_numeric_answer,
    extract_text_answer,
    normalize_answer,
    normalize_format,
    normalize_prediction,
)


DEFAULT_PASS_AT_K = 64
DEFAULT_SHOW_FAILURES = 2


def is_record_correct(record: dict) -> tuple[bool, dict]:
    pred_raw = record.get("generated_text_only", "")
    ref_raw = record.get("answer", "")
    pred_legacy = normalize_prediction(pred_raw)
    ref_legacy = normalize_answer(ref_raw)

    pred = extract_text_answer(pred_raw)
    ref = extract_text_answer(ref_raw)
    pred_format = normalize_format(pred)
    ref_format = normalize_format(ref)
    pred_decimal = extract_numeric_answer(pred_raw)
    ref_decimal = extract_numeric_answer(ref_raw)

    is_correct = False
    if pred_decimal is not None and ref_decimal is not None:
        is_correct = pred_decimal == ref_decimal
    elif pred == ref:
        is_correct = True

    details = {
        "prediction_raw": pred_raw,
        "prediction_legacy": pred_legacy,
        "prediction_norm": pred,
        "prediction_format": pred_format,
        "prediction_decimal": None if pred_decimal is None else str(pred_decimal),
        "answer_raw": ref_raw,
        "answer_legacy": ref_legacy,
        "answer_norm": ref,
        "answer_format": ref_format,
        "answer_decimal": None if ref_decimal is None else str(ref_decimal),
    }
    return is_correct, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--pass_at_k", type=int, default=DEFAULT_PASS_AT_K)
    parser.add_argument("--show_failures", type=int, default=DEFAULT_SHOW_FAILURES)
    args = parser.parse_args()

    if args.pass_at_k < 1:
        raise ValueError("--pass_at_k must be >= 1")

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    groups = defaultdict(list)
    sample_total = 0
    sample_correct = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            is_correct, details = is_record_correct(record)
            enriched = {
                **record,
                **details,
                "is_correct": is_correct,
                "sample_index": int(record.get("sample_index", 0)),
            }
            groups[int(record["index"])].append(enriched)
            sample_total += 1
            if is_correct:
                sample_correct += 1

    if not groups:
        raise ValueError(f"No valid records found in {input_path}")

    total_examples = len(groups)
    pass_correct = 0
    failure_cases = []
    sample_counts = set()

    for index in sorted(groups):
        records = sorted(groups[index], key=lambda item: item["sample_index"])
        sample_counts.add(len(records))
        effective_k = min(args.pass_at_k, len(records))
        hit = any(record["is_correct"] for record in records[:effective_k])
        if hit:
            pass_correct += 1
            continue

        if len(failure_cases) < args.show_failures:
            failure_cases.append(
                {
                    "index": index,
                    "question": records[0].get("question"),
                    "answer_norm": records[0]["answer_norm"],
                    "effective_k": effective_k,
                    "sample_count": len(records),
                    "attempts": [
                        {
                            "sample_index": record["sample_index"],
                            "prediction_norm": record["prediction_norm"],
                            "prediction_decimal": record["prediction_decimal"],
                        }
                        for record in records[:effective_k]
                    ],
                }
            )

    sample_accuracy = sample_correct / sample_total
    pass_accuracy = pass_correct / total_examples

    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Total examples: {total_examples}")
    print(f"Total samples: {sample_total}")
    print(f"Sample counts observed: {sorted(sample_counts)}")
    print(f"Per-sample accuracy: {sample_accuracy:.6f} ({sample_accuracy * 100:.2f}%)")
    print(
        f"Pass@{args.pass_at_k} accuracy: {pass_accuracy:.6f} ({pass_accuracy * 100:.2f}%), "
        f"correct={pass_correct}/{total_examples}"
    )
    print("=" * 80)

    if failure_cases:
        print(f"Showing first {len(failure_cases)} pass@{args.pass_at_k} failures:")
        for case in failure_cases:
            print("-" * 80)
            print(f"index: {case['index']}")
            print(f"question: {case['question']}")
            print(f"answer_norm: {case['answer_norm']!r}")
            print(f"effective_k: {case['effective_k']}")
            print(f"sample_count: {case['sample_count']}")
            for attempt in case["attempts"]:
                print(
                    f"sample_index={attempt['sample_index']}: "
                    f"prediction_norm={attempt['prediction_norm']!r}, "
                    f"prediction_decimal={attempt['prediction_decimal']}"
                )


if __name__ == "__main__":
    main()
