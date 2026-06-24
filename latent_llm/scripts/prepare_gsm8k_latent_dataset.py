#!/usr/bin/env python
"""
Prepare latent-token SFT data from GSM8K rendered dataset.

This script reads:
1) metadata jsonl (question/answer/image path)
2) per-image feature tensors (.pt)
3) codebook tensor (.pt)

For each feature token, it finds the nearest codebook row by L2 distance and converts
it to latent token id: `base_vocab_size + codebook_index` (0-based index).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_JSONL = "data/gsm8k_train_processed.jsonl"
DEFAULT_FEATURE_DIR = "data/gsm8k_features"
DEFAULT_CODEBOOK = "outputs/deepseek_codebook/latent/codebook.pt"
DEFAULT_OUTPUT = "./gsm8k_latent_train_1k.json"


def _load_tensor_from_pt(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        priority_keys = ["features", "feature", "token_features", "tokens", "x", "hidden_states", "embeddings"]
        for key in priority_keys:
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in obj.values():
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"Cannot find tensor in {path}")


def _normalize_feature_shape(feature: torch.Tensor, codebook_dim: int) -> torch.Tensor:
    if feature.ndim == 1:
        feature = feature.unsqueeze(0)
    elif feature.ndim > 2:
        feature = feature.reshape(-1, feature.shape[-1])

    # Expected shape: [num_tokens, dim]
    if feature.shape[-1] == codebook_dim:
        return feature.float().contiguous()
    if feature.shape[0] == codebook_dim:
        return feature.T.float().contiguous()
    raise ValueError(f"Feature shape {tuple(feature.shape)} incompatible with codebook_dim={codebook_dim}")


def _nearest_codebook_indices(
    feature_tokens: torch.Tensor,
    codebook: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Returns nearest codebook row index for each feature token by squared L2 distance.
    """
    codebook_norm = (codebook * codebook).sum(dim=1)  # [K]
    codebook_t = codebook.T.contiguous()  # [D, K]

    outputs = []
    for start in range(0, feature_tokens.shape[0], chunk_size):
        end = min(start + chunk_size, feature_tokens.shape[0])
        x = feature_tokens[start:end]  # [B, D]
        x_norm = (x * x).sum(dim=1, keepdim=True)  # [B, 1]
        dist = x_norm + codebook_norm.unsqueeze(0) - 2.0 * (x @ codebook_t)  # [B, K]
        outputs.append(dist.argmin(dim=1).cpu())
    return torch.cat(outputs, dim=0)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _feature_path_from_record(record: dict[str, Any], feature_dir: Path) -> Path:
    image_path = Path(record["rendered_image_path"])
    return feature_dir / f"{image_path.stem}.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default=DEFAULT_JSONL)
    parser.add_argument("--feature_dir", type=str, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--codebook_path", type=str, default=DEFAULT_CODEBOOK)
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--base_vocab_size", type=int, default=152064)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk_size", type=int, default=2048)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    feature_dir = Path(args.feature_dir)
    codebook_path = Path(args.codebook_path)
    output_json = Path(args.output_json)

    print(f"Loading codebook from: {codebook_path}")
    codebook = _load_tensor_from_pt(codebook_path).float()
    if codebook.ndim != 2:
        raise ValueError(f"Codebook must be 2D, got shape={tuple(codebook.shape)}")
    num_codebook_tokens, codebook_dim = codebook.shape
    print(f"Codebook shape: {tuple(codebook.shape)}")
    print(f"Base vocab size: {args.base_vocab_size}")
    print(f"Latent id range: [{args.base_vocab_size}, {args.base_vocab_size + num_codebook_tokens - 1}]")

    device = torch.device(args.device)
    codebook = codebook.to(device)

    outputs = []
    missing_features = 0
    processed = 0

    for record in _iter_jsonl(input_jsonl):
        if processed >= args.max_samples:
            break

        feature_path = _feature_path_from_record(record, feature_dir)
        if not feature_path.exists():
            missing_features += 1
            continue

        feature = _load_tensor_from_pt(feature_path)
        feature = _normalize_feature_shape(feature, codebook_dim)
        feature = feature.to(device)

        with torch.no_grad():
            nearest_idx = _nearest_codebook_indices(feature, codebook, chunk_size=args.chunk_size)  # [T], 0-based
        latent_token_ids = (nearest_idx + args.base_vocab_size).tolist()  # 0-based offset

        outputs.append(
            {
                "id": record.get("id"),
                "question": record["question"],
                "answer": record["answer"],
                "rendered_image_path": record["rendered_image_path"],
                "feature_path": str(feature_path),
                "cot_codebook_indices": nearest_idx.tolist(),
                "cot_token_ids": latent_token_ids,
                "num_cot_tokens": len(latent_token_ids),
            }
        )
        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} samples...")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Saved {len(outputs)} samples to: {output_json}")
    print(f"Missing feature files skipped: {missing_features}")
    if outputs:
        lengths = [x["num_cot_tokens"] for x in outputs]
        print(f"Average COT token length: {sum(lengths) / len(lengths):.2f}")
        print(f"Min/Max COT token length: {min(lengths)} / {max(lengths)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
