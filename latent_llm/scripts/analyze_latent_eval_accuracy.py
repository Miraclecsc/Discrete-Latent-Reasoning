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
Compute exact-match accuracy for latent-eval JSONL outputs.

The default comparison is answer-aware:
1. Try to extract a final numeric answer from the generated text.
2. If both prediction and reference are numeric, compare numerically.
3. Otherwise, extract the final text answer span and compare exact text.

This supports the current latent output format:
    Think: ... So the answer is: ...
"""

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path


DEFAULT_INPUT_PATH = "outputs/latent_eval_outputs.jsonl"
DEFAULT_SHOW_MISMATCHES = 2
DEFAULT_SHOW_NEAR_MISSES = 2
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?")
FINAL_ANSWER_PATTERNS = [
    re.compile(r"####\s*([^\n\r]+)"),
    re.compile(r"(?:^|\n)\s*answer\s*[:：]\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"(?:final answer|final response)\s*(?:is|:)\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"(?:the answer is|answer is)\s*([^\n\r]+)", re.IGNORECASE),
]


def normalize_prediction(text: str) -> str:
    """
    Normalize generated text before exact-match comparison.

    Args:
        text (`str`):
            Raw `generated_text_only` field.

    Returns:
        `str`:
            Stripped prediction text.
    """
    return text.strip()


def normalize_answer(answer) -> str:
    """
    Normalize reference answer before exact-match comparison.

    Args:
        answer:
            Raw `answer` field from the JSONL record.

    Returns:
        `str`:
            Stripped reference answer.
    """
    return str(answer).strip()


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def extract_last_number(text: str) -> str:
    matches = NUMBER_PATTERN.findall(str(text or ""))
    if not matches:
        return ""
    return matches[-1].replace(",", "")


def extract_last_boxed_answer(text: str) -> str:
    marker = r"\boxed{"
    start = str(text or "").rfind(marker)
    if start == -1:
        return ""

    cursor = start + len(marker)
    depth = 1
    collected = []
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        collected.append(char)
        cursor += 1

    if depth != 0:
        return ""
    return "".join(collected).strip()


def extract_answer_candidates(text: str) -> list[str]:
    text = str(text or "")
    candidates = []

    boxed = extract_last_boxed_answer(text)
    if boxed:
        candidates.append(boxed)

    for pattern in FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            candidates.append(matches[-1].strip())

    last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    if last_line:
        candidates.append(last_line)

    normalized_full = normalize_whitespace(text)
    if normalized_full:
        candidates.append(normalized_full)

    deduped = []
    seen = set()
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def normalize_answer_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"^[#:=：\-\s]+", "", text)
    text = text.rstrip(" \t\r\n.。")
    return text


def parse_number_candidate(text: str):
    cleaned = normalize_answer_text(text)
    cleaned = cleaned.replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    if not cleaned:
        return None

    if re.fullmatch(r"[-+]?\d+/\d+", cleaned):
        numerator, denominator = cleaned.split("/", 1)
        try:
            return Fraction(int(numerator), int(denominator))
        except ZeroDivisionError:
            return None

    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", cleaned):
        try:
            return Fraction(Decimal(cleaned))
        except InvalidOperation:
            return None

    return None


def extract_numeric_answer(text: str):
    for candidate in extract_answer_candidates(text):
        value = parse_number_candidate(candidate)
        if value is not None:
            return value

        last_number = extract_last_number(candidate)
        if last_number:
            value = parse_number_candidate(last_number)
            if value is not None:
                return value

    last_number = extract_last_number(text)
    if last_number:
        return parse_number_candidate(last_number)
    return None


def extract_text_answer(text: str) -> str:
    candidates = extract_answer_candidates(text)
    if not candidates:
        return normalize_answer_text(text)
    return normalize_answer_text(candidates[0])


def normalize_format(text: str) -> str:
    """
    Apply lightweight format normalization for diagnostic accuracy.

    Args:
        text (`str`):
            Prediction or answer text.

    Returns:
        `str`:
            Lightly normalized text.
    """
    text = str(text).strip()
    if text.endswith("."):
        text = text[:-1].strip()
    text = text.replace(",", "")
    if text.startswith("$"):
        text = text[1:].strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    return text


def parse_decimal(text: str) -> Decimal | None:
    """
    Parse normalized numeric text into a decimal value.

    Args:
        text (`str`):
            Prediction or answer text.

    Returns:
        `Decimal` or `None`:
            Parsed numeric value when available.
    """
    try:
        return Decimal(normalize_format(text))
    except InvalidOperation:
        return None


def is_integer_value(value: Decimal) -> bool:
    """
    Check whether a decimal value is integer-valued.

    Args:
        value (`Decimal`):
            Numeric value.

    Returns:
        `bool`:
            Whether the value is an integer.
    """
    return value == value.to_integral_value()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--show_mismatches", type=int, default=DEFAULT_SHOW_MISMATCHES)
    parser.add_argument("--show_near_misses", type=int, default=DEFAULT_SHOW_NEAR_MISSES)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    total = 0
    correct = 0
    raw_exact = 0
    format_correct = 0
    integer_tolerance_correct = 0
    mismatches = []
    format_gain_cases = []
    integer_tolerance_gain_cases = []
    extraction_gain_cases = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total += 1

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

            if pred_raw == str(ref_raw):
                raw_exact += 1

            is_correct = False
            if pred_decimal is not None and ref_decimal is not None:
                is_correct = pred_decimal == ref_decimal
            elif pred == ref:
                is_correct = True

            if is_correct:
                correct += 1
            if pred_format == ref_format:
                format_correct += 1
            if is_correct:
                integer_tolerance_correct += 1
            elif (
                pred_decimal is not None
                and ref_decimal is not None
                and ref_decimal.denominator == 1
                and abs(pred_decimal - ref_decimal) <= Fraction(1, 2)
            ):
                integer_tolerance_correct += 1
                if len(integer_tolerance_gain_cases) < args.show_near_misses:
                    integer_tolerance_gain_cases.append(
                        {
                            "index": record.get("index"),
                            "prediction_norm": pred,
                            "answer_norm": ref,
                            "prediction_decimal": str(pred_decimal),
                            "answer_decimal": str(ref_decimal),
                        }
                    )

            if pred_legacy != ref_legacy and is_correct and len(extraction_gain_cases) < args.show_near_misses:
                extraction_gain_cases.append(
                    {
                        "index": record.get("index"),
                        "prediction_raw": pred_raw,
                        "prediction_legacy": pred_legacy,
                        "prediction_extracted": pred,
                        "answer_raw": ref_raw,
                        "answer_legacy": ref_legacy,
                        "answer_extracted": ref,
                    }
                )

            if not is_correct:
                mismatches.append(
                    {
                        "index": record.get("index"),
                        "question": record.get("question"),
                        "prediction_raw": pred_raw,
                        "prediction_legacy": pred_legacy,
                        "prediction_norm": pred,
                        "prediction_decimal": None if pred_decimal is None else str(pred_decimal),
                        "answer_raw": ref_raw,
                        "answer_legacy": ref_legacy,
                        "answer_norm": ref,
                        "answer_decimal": None if ref_decimal is None else str(ref_decimal),
                    }
                )
                if pred_format == ref_format and len(format_gain_cases) < args.show_near_misses:
                    format_gain_cases.append(
                        {
                            "index": record.get("index"),
                            "prediction_norm": pred,
                            "answer_norm": ref,
                            "prediction_format": pred_format,
                            "answer_format": ref_format,
                        }
                    )

    if total == 0:
        raise ValueError(f"No valid records found in {input_path}")

    accuracy = correct / total
    raw_accuracy = raw_exact / total
    format_accuracy = format_correct / total
    integer_tolerance_accuracy = integer_tolerance_correct / total

    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Total samples: {total}")
    print(f"Correct after answer extraction: {correct}")
    print(f"Answer-aware accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print(f"Correct after lightweight format normalization: {format_correct}")
    print(
        "Format-normalized accuracy: "
        f"{format_accuracy:.6f} ({format_accuracy * 100:.2f}%), "
        f"gain={format_correct - correct}"
    )
    print(f"Correct with integer-answer tolerance (|pred-ref| <= 0.5): {integer_tolerance_correct}")
    print(
        "Integer-tolerance diagnostic accuracy: "
        f"{integer_tolerance_accuracy:.6f} ({integer_tolerance_accuracy * 100:.2f}%), "
        f"gain={integer_tolerance_correct - correct}"
    )
    print(f"Raw exact matches without strip(): {raw_exact}")
    print(f"Raw exact-match accuracy: {raw_accuracy:.6f} ({raw_accuracy * 100:.2f}%)")
    print("=" * 80)

    if args.show_mismatches > 0 and mismatches:
        print(f"Showing first {min(args.show_mismatches, len(mismatches))} mismatches:")
        for mismatch in mismatches[: args.show_mismatches]:
            print("-" * 80)
            print(f"index: {mismatch['index']}")
            print(f"question: {mismatch['question']}")
            print(f"prediction_raw: {mismatch['prediction_raw']!r}")
            print(f"prediction_legacy: {mismatch['prediction_legacy']!r}")
            print(f"prediction_norm: {mismatch['prediction_norm']!r}")
            print(f"prediction_decimal: {mismatch['prediction_decimal']}")
            print(f"answer_raw: {mismatch['answer_raw']!r}")
            print(f"answer_legacy: {mismatch['answer_legacy']!r}")
            print(f"answer_norm: {mismatch['answer_norm']!r}")
            print(f"answer_decimal: {mismatch['answer_decimal']}")

    if extraction_gain_cases:
        print("=" * 80)
        print(f"Examples fixed by answer extraction ({len(extraction_gain_cases)} shown):")
        for case in extraction_gain_cases:
            print("-" * 80)
            print(f"index: {case['index']}")
            print(f"prediction_raw: {case['prediction_raw']!r}")
            print(f"prediction_legacy: {case['prediction_legacy']!r}")
            print(f"prediction_extracted: {case['prediction_extracted']!r}")
            print(f"answer_raw: {case['answer_raw']!r}")
            print(f"answer_legacy: {case['answer_legacy']!r}")
            print(f"answer_extracted: {case['answer_extracted']!r}")

    if format_gain_cases:
        print("=" * 80)
        print(f"Examples fixed only by format normalization ({len(format_gain_cases)} shown):")
        for case in format_gain_cases:
            print("-" * 80)
            print(f"index: {case['index']}")
            print(f"prediction_norm: {case['prediction_norm']!r}")
            print(f"answer_norm: {case['answer_norm']!r}")
            print(f"prediction_format: {case['prediction_format']!r}")
            print(f"answer_format: {case['answer_format']!r}")

    if integer_tolerance_gain_cases:
        print("=" * 80)
        print(f"Examples fixed only by integer-answer tolerance ({len(integer_tolerance_gain_cases)} shown):")
        for case in integer_tolerance_gain_cases:
            print("-" * 80)
            print(f"index: {case['index']}")
            print(f"prediction_norm: {case['prediction_norm']!r}")
            print(f"answer_norm: {case['answer_norm']!r}")
            print(f"prediction_decimal: {case['prediction_decimal']}")
            print(f"answer_decimal: {case['answer_decimal']}")


if __name__ == "__main__":
    main()
