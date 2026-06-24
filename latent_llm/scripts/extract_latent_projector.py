#!/usr/bin/env python
"""
Export a standalone latent projector tensor from a latent checkpoint.

This script is meant for the workflow:
1. Run latent pretraining from a base model + codebook.
2. Take the final checkpoint produced by the latent trainer.
3. Export `projector.pt` for downstream SFT initialization.

By default it only saves the projector. If needed, it can also export the
trained codebook, which is useful because the current pretrain defaults train
both `codebook` and `projector`.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch


def resolve_latent_dir(ckpt_path: Path) -> Path:
    """
    Resolve the latent weights directory from a checkpoint path.

    Supported inputs:
    - checkpoint root containing `latent/`
    - direct `latent/` directory
    """
    if ckpt_path.is_dir():
        if ckpt_path.name == "latent":
            return ckpt_path
        latent_dir = ckpt_path / "latent"
        if latent_dir.is_dir():
            return latent_dir
    raise FileNotFoundError(
        f"Could not resolve a latent directory from: {ckpt_path}. "
        f"Expected either `<ckpt>/latent/` or a direct `latent/` path."
    )


def load_tensor_from_safetensors(path: Path, key: str) -> torch.Tensor:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required to read `.safetensors` latent weights. "
            "Please install it or use a checkpoint saved with `.pt` latent files."
        ) from exc

    payload = load_file(path)
    if key not in payload:
        raise KeyError(f"Missing key `{key}` in {path}. Available keys: {list(payload.keys())}")
    return payload[key]


def load_tensor_from_pt(path: Path, key: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        if len(payload) == 1:
            return next(iter(payload.values()))
    raise TypeError(f"Unsupported payload in {path}: expected Tensor or dict containing `{key}`.")


def load_named_tensor(latent_dir: Path, stem: str) -> tuple[torch.Tensor, Path]:
    """
    Load a named latent tensor from `latent_dir`.

    Prefers safetensors, then falls back to `.pt`.
    """
    safetensors_path = latent_dir / f"{stem}.safetensors"
    if safetensors_path.exists():
        return load_tensor_from_safetensors(safetensors_path, stem), safetensors_path

    pt_path = latent_dir / f"{stem}.pt"
    if pt_path.exists():
        return load_tensor_from_pt(pt_path, stem), pt_path

    raise FileNotFoundError(
        f"Could not find `{stem}.safetensors` or `{stem}.pt` under {latent_dir}."
    )


def validate_matrix(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} is not a torch.Tensor: {type(tensor)}")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")
    return tensor.detach().cpu()


def save_tensor(tensor: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="outputs/latent_pretrain",
        help="Checkpoint root containing `latent/`, or a direct `latent/` directory.",
    )
    parser.add_argument(
        "--projector_output_path",
        type=str,
        default="outputs/latent_pretrain/latent/projector.pt",
        help="Output path for standalone `projector.pt`.",
    )
    parser.add_argument(
        "--codebook_output_path",
        type=str,
        default=None,
        help="Optional output path for standalone `codebook.pt`.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    latent_dir = resolve_latent_dir(ckpt_path)

    projector, projector_src = load_named_tensor(latent_dir, "projector")
    projector = validate_matrix("projector", projector)
    projector_output_path = Path(args.projector_output_path).expanduser().resolve()
    save_tensor(projector, projector_output_path)

    print("=" * 80)
    print("Latent Projector Export")
    print("=" * 80)
    print(f"checkpoint: {ckpt_path}")
    print(f"latent_dir: {latent_dir}")
    print(f"loaded_projector_from: {projector_src}")
    print(f"projector_shape: {tuple(projector.shape)}")
    print(f"saved_projector_to: {projector_output_path}")

    if args.codebook_output_path:
        codebook, codebook_src = load_named_tensor(latent_dir, "codebook")
        codebook = validate_matrix("codebook", codebook)
        codebook_output_path = Path(args.codebook_output_path).expanduser().resolve()
        save_tensor(codebook, codebook_output_path)
        print(f"loaded_codebook_from: {codebook_src}")
        print(f"codebook_shape: {tuple(codebook.shape)}")
        print(f"saved_codebook_to: {codebook_output_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
