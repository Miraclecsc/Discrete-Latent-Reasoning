#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert JSONL OCR training data to a Hugging Face dataset saved with save_to_disk()."
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the converted Hugging Face dataset.",
    )
    parser.add_argument(
        "--image-field",
        type=str,
        default="rendered_image_path",
        help="Field name in JSONL that stores the image path.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name in JSONL that stores the target text.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory used to resolve relative image paths. Defaults to the JSONL parent directory.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of worker processes for dataset.map/filter.",
    )
    parser.add_argument(
        "--keep-extra-columns",
        action="store_true",
        help="Keep original columns in addition to image_path/text.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Drop samples whose resolved image_path does not exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_jsonl = Path(args.input_jsonl).resolve()
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_jsonl}")

    output_dir = Path(args.output_dir).resolve()
    base_dir = Path(args.base_dir).resolve() if args.base_dir else input_jsonl.parent

    normalized_jsonl = None
    total_lines = 0
    missing_image_field = 0
    missing_text_field = 0

    print(f"Input JSONL: {input_jsonl}")
    print(f"Output dir: {output_dir}")
    print(f"Base dir for relative image paths: {base_dir}")
    print("Phase 1/3: normalizing JSONL to image_path/text only ...")

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", delete=False
    ) as tmp_file:
        normalized_jsonl = Path(tmp_file.name)

        with input_jsonl.open("r", encoding="utf-8") as src:
            for line_no, line in enumerate(
                tqdm(src, desc="Normalizing JSONL", unit="lines"),
                start=1,
            ):
                line = line.strip()
                if not line:
                    continue

                total_lines += 1
                example = json.loads(line)

                if args.image_field not in example:
                    missing_image_field += 1
                if args.text_field not in example:
                    missing_text_field += 1

                raw_path = example.get(args.image_field)
                if raw_path is None:
                    resolved_path = None
                else:
                    raw_path = str(raw_path)
                    resolved_path = raw_path if os.path.isabs(raw_path) else str((base_dir / raw_path).resolve())

                normalized = {
                    "image_path": resolved_path,
                    "text": example.get(args.text_field),
                }
                tmp_file.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    print(f"Phase 1/3 done. Temporary normalized JSONL: {normalized_jsonl}")
    print("Phase 2/3: loading normalized JSONL with datasets ...")
    dataset = load_dataset("json", data_files=str(normalized_jsonl), split="train")
    print("Phase 2/3 done.")

    if args.skip_missing_images:
        dataset = dataset.filter(
            lambda example: example["image_path"] is not None and os.path.exists(example["image_path"]),
            num_proc=args.num_proc,
            desc="Filtering missing images",
        )

    remove_columns = [col for col in dataset.column_names if col not in {"image_path", "text"}]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print("Phase 3/3: saving Hugging Face dataset to disk ...")
    dataset.save_to_disk(str(output_dir))
    print("Phase 3/3 done.")

    if normalized_jsonl is not None and normalized_jsonl.exists():
        normalized_jsonl.unlink()

    print(f"Saved dataset to: {output_dir}")
    print(f"Num samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"Total input lines: {total_lines}")
    print(f"Missing '{args.image_field}' field: {missing_image_field}")
    print(f"Missing '{args.text_field}' field: {missing_text_field}")
    print("First sample:")
    print(dataset[0])


if __name__ == "__main__":
    main()
