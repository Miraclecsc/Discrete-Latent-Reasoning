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
Shared helpers for trainers that operate on latent-extended vocabularies.
"""

import json
from pathlib import Path
from typing import Optional

import torch

from ..models.latent_utils import infer_tokenizer_vocab_size


class LatentTrainerMixin:
    """
    Shared save/load helpers for trainers using latent token extensions.

    This mixin assumes the subclass sets the following attributes during init:
    - `self.codebook`
    - `self.projector`
    - `self._latent_embed`
    - `self._latent_lm_head`
    """

    def _configure_parameter_freezing(self, model, args) -> None:
        """
        Configure which parameters are trainable.
        """
        if args.freeze_base_model:
            for name, param in model.named_parameters():
                if name not in ["codebook", "projector"] and "codebook" not in name and "projector" not in name:
                    param.requires_grad = False

            self.codebook.requires_grad = not args.freeze_codebook
            self.projector.requires_grad = True
            print("Base model frozen. Only training codebook and projector.")
        elif args.freeze_codebook:
            self.codebook.requires_grad = False
            self.projector.requires_grad = True
            print("Codebook frozen. Training projector and base model.")
        else:
            print("Full fine-tuning: all parameters (base model, codebook, projector) are trainable.")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model in a production-ready latent checkpoint layout.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_vocab_size = self._latent_embed.base_embed.num_embeddings
        num_latent_tokens = self.codebook.shape[0]

        latent_dir = output_dir / "latent"
        latent_dir.mkdir(exist_ok=True)

        self._save_latent_weights(latent_dir, base_vocab_size, num_latent_tokens)
        self._save_base_model(output_dir, base_vocab_size, num_latent_tokens, _internal_call)
        self._save_readme(output_dir, base_vocab_size, num_latent_tokens)

        if self.args.should_save:
            self._log_save_summary(output_dir, latent_dir)

    def _save_latent_weights(self, latent_dir: Path, base_vocab_size: int, num_latent_tokens: int) -> None:
        try:
            from safetensors.torch import save_file

            save_file({"codebook": self.codebook.detach().cpu()}, latent_dir / "codebook.safetensors")
            save_file({"projector": self.projector.detach().cpu()}, latent_dir / "projector.safetensors")
        except ImportError:
            torch.save(self.codebook.detach().cpu(), latent_dir / "codebook.pt")
            torch.save(self.projector.detach().cpu(), latent_dir / "projector.pt")

        tokenizer_vocab_size = infer_tokenizer_vocab_size(getattr(self, "processing_class", None))
        latent_config = {
            "num_latent_tokens": num_latent_tokens,
            "codebook_dim": self.codebook.shape[1],
            "hidden_size": self.projector.shape[1],
            "base_vocab_size": base_vocab_size,
            "embedding_vocab_size": base_vocab_size,
            "latent_token_offset": base_vocab_size,
            "total_vocab_size": base_vocab_size + num_latent_tokens,
            "version": "1.1.0",
        }
        if tokenizer_vocab_size is not None:
            latent_config["tokenizer_vocab_size"] = tokenizer_vocab_size
            latent_config["reserved_token_count"] = base_vocab_size - tokenizer_vocab_size

        with open(latent_dir / "config.json", "w") as f:
            json.dump(latent_config, f, indent=2)

        readme_content = f"""# Latent Weights

This directory contains the latent token weights for the extended vocabulary model.

## Files

- `codebook.safetensors` / `codebook.pt`: Codebook tensor of shape [{num_latent_tokens}, {self.codebook.shape[1]}]
- `projector.safetensors` / `projector.pt`: Projector matrix of shape [{self.codebook.shape[1]}, {self.projector.shape[1]}]
- `config.json`: Latent configuration

## Usage

Latent embeddings are computed as:
```python
latent_embeddings = torch.matmul(codebook, projector)
```
"""
        with open(latent_dir / "README.md", "w") as f:
            f.write(readme_content)

    def _save_base_model(
        self, output_dir: Path, base_vocab_size: int, num_latent_tokens: int, _internal_call: bool
    ) -> None:
        model = self.model
        extended_vocab_size = base_vocab_size + num_latent_tokens

        trained_base_embed = self._latent_embed.base_embed
        trained_base_lm_head = self._latent_lm_head.base_lm_head

        original_embed = model.get_input_embeddings()
        original_lm_head = model.lm_head
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.model.embed_tokens = trained_base_embed
        elif hasattr(model, "embed_tokens"):
            model.embed_tokens = trained_base_embed
        elif hasattr(model, "set_input_embeddings"):
            model.set_input_embeddings(trained_base_embed)

        model.lm_head = trained_base_lm_head

        if hasattr(model, "config"):
            if hasattr(model.config, "vocab_size"):
                model.config.vocab_size = base_vocab_size
            if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "vocab_size"):
                model.config.text_config.vocab_size = base_vocab_size
        if hasattr(model, "vocab_size"):
            model.vocab_size = base_vocab_size

        try:
            super().save_model(str(output_dir), _internal_call)
        finally:
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                model.model.embed_tokens = original_embed
            elif hasattr(model, "embed_tokens"):
                model.embed_tokens = original_embed
            elif hasattr(model, "set_input_embeddings"):
                model.set_input_embeddings(original_embed)

            model.lm_head = original_lm_head

            if hasattr(model, "config"):
                if hasattr(model.config, "vocab_size"):
                    model.config.vocab_size = extended_vocab_size
                if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "vocab_size"):
                    model.config.text_config.vocab_size = extended_vocab_size
            if hasattr(model, "vocab_size"):
                model.vocab_size = extended_vocab_size

    def _save_readme(self, output_dir: Path, base_vocab_size: int, num_latent_tokens: int) -> None:
        readme_content = f"""# Latent Model

This checkpoint contains a base model plus latent token weights.

## Total Vocabulary

- Base tokens: `[0, {base_vocab_size - 1}]`
- Latent tokens: `[{base_vocab_size}, {base_vocab_size + num_latent_tokens - 1}]`
- Total size: `{base_vocab_size + num_latent_tokens}`
"""
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)

    def _log_save_summary(self, output_dir: Path, latent_dir: Path) -> None:
        print(f"\n{'=' * 60}")
        print("Model saved successfully (Latent Format)")
        print(f"{'=' * 60}")
        print(f"\nOutput Directory: {output_dir}")
        print(f"Latent Directory: {latent_dir}")
        print(f"Codebook shape: {tuple(self.codebook.shape)}")
        print(f"Projector shape: {tuple(self.projector.shape)}")
        print(f"{'=' * 60}\n")
