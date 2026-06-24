#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge JSONL shards and sort by index.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--glob", default="*.jsonl")
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows = []
    for path in sorted(input_dir.glob(args.glob)):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

    rows.sort(key=lambda row: (int(row.get("index", -1)), int(row.get("sample_index", -1))))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
