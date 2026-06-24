#!/usr/bin/env python
"""
Normalize supported math benchmarks into the standard parquet format expected by
`eval_latent_sft.py`.

The output parquet always contains at least:
    - question
    - answer
    - steps

Extra metadata columns are included for traceability but are ignored by the
current evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_BENCHMARKS = ("auto", "parquet_qa", "deepseek_parquet", "gsmhard", "multiarith", "svamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="auto", choices=SUPPORTED_BENCHMARKS)
    parser.add_argument("--max_examples", type=int, default=None)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, "g")
    return str(value).strip()


def load_json_like(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Failed to read JSONL from {path}. The file is not valid UTF-8 text, "
            "which usually means the benchmark type and input path do not match."
        ) from exc
    return records


def infer_benchmark(path: Path) -> str:
    lower_name = path.name.lower()
    if path.suffix == ".parquet":
        return "parquet_qa"
    if "gsmhard" in lower_name:
        return "gsmhard"
    if "multiarith" in lower_name:
        return "multiarith"
    if "svamp" in lower_name:
        return "svamp"

    if path.suffix == ".jsonl":
        sample_records = load_jsonl(path)[:1]
        if sample_records and {"input", "target"}.issubset(sample_records[0].keys()):
            return "gsmhard"

    if path.suffix == ".json":
        payload = load_json_like(path)
        sample = payload[0] if isinstance(payload, list) and payload else payload
        if isinstance(sample, dict):
            if {"question", "final_ans"}.issubset(sample.keys()):
                return "multiarith"
            if {"Body", "Question", "Answer"}.issubset(sample.keys()):
                return "svamp"

    raise ValueError(f"Unable to infer benchmark type for: {path}")


def convert_parquet_qa(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet benchmark is missing required columns: {sorted(missing)}")

    result = pd.DataFrame(
        {
            "question": df["question"].map(normalize_text),
            "answer": df["answer"].map(normalize_answer),
            "steps": df["steps"] if "steps" in df.columns else [[] for _ in range(len(df))],
        }
    )
    return result


def convert_gsmhard(path: Path) -> pd.DataFrame:
    records = load_jsonl(path)
    normalized = []
    for idx, record in enumerate(records):
        normalized.append(
            {
                "question": normalize_text(record.get("input")),
                "answer": normalize_answer(record.get("target")),
                "steps": [],
                "source_index": idx,
            }
        )
    return pd.DataFrame(normalized)


def convert_multiarith(path: Path) -> pd.DataFrame:
    payload = load_json_like(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list for MultiArith, got: {type(payload).__name__}")

    normalized = []
    for idx, record in enumerate(payload):
        normalized.append(
            {
                "question": normalize_text(record.get("question")),
                "answer": normalize_answer(record.get("final_ans")),
                "steps": [],
                "source_index": idx,
            }
        )
    return pd.DataFrame(normalized)


def convert_svamp(path: Path) -> pd.DataFrame:
    payload = load_json_like(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list for SVAMP, got: {type(payload).__name__}")

    normalized = []
    for idx, record in enumerate(payload):
        question = normalize_text(
            " ".join(part for part in [str(record.get("Body", "")).strip(), str(record.get("Question", "")).strip()] if part)
        )
        normalized.append(
            {
                "question": question,
                "answer": normalize_answer(record.get("Answer")),
                "steps": [],
                "source_index": idx,
                "source_id": record.get("ID"),
                "type": record.get("Type"),
            }
        )
    return pd.DataFrame(normalized)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input benchmark not found: {input_path}")

    benchmark = args.benchmark
    if benchmark == "auto":
        benchmark = infer_benchmark(input_path)

    benchmark_label = benchmark
    benchmark_type = "parquet_qa" if benchmark == "deepseek_parquet" else benchmark

    if benchmark_type == "parquet_qa":
        df = convert_parquet_qa(input_path)
    elif benchmark_type == "gsmhard":
        df = convert_gsmhard(input_path)
    elif benchmark_type == "multiarith":
        df = convert_multiarith(input_path)
    elif benchmark_type == "svamp":
        df = convert_svamp(input_path)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    if args.max_examples is not None:
        df = df.iloc[: args.max_examples].copy()

    df.insert(0, "benchmark", benchmark_label)
    df.insert(1, "source_path", str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print("=" * 80)
    print("Prepared latent eval benchmark")
    print(f"Benchmark: {benchmark_label}")
    print(f"Benchmark type: {benchmark_type}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total examples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
