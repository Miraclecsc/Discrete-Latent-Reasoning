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
Latent-aware GRPO trainer.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import torch.nn as nn
from transformers import PreTrainedModel

from ..models.latent_utils import align_model_embeddings_with_tokenizer, extend_model_for_latent_tokens
from ..models.local_model_utils import is_deepseek_decoder_model, load_model_for_training
from . import grpo_trainer as grpo_trainer_module
from .grpo_trainer import GRPOTrainer
from .latent_grpo_config import LatentGRPOConfig
from .latent_trainer_mixin import LatentTrainerMixin


def _find_latent_dir(path_like: Optional[str]) -> Optional[Path]:
    if path_like is None:
        return None

    path = Path(path_like)
    candidates = []
    if path.is_file():
        candidates.append(path.parent / "latent")
    elif path.name == "latent":
        candidates.append(path)
    else:
        candidates.append(path / "latent")
        candidates.append(path)

    for candidate in candidates:
        config_path = candidate / "config.json"
        if candidate.is_dir() and config_path.exists():
            return candidate
    return None


def _load_latent_config(path_like: Optional[str]) -> Optional[dict]:
    latent_dir = _find_latent_dir(path_like)
    if latent_dir is None:
        return None
    with open(latent_dir / "config.json", encoding="utf-8") as f:
        return json.load(f)


class LatentGRPOTrainer(LatentTrainerMixin, GRPOTrainer):
    """
    GRPO trainer that transparently re-attaches latent vocabulary weights.
    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        reward_funcs=None,
        args: Optional[LatentGRPOConfig] = None,
        peft_config=None,
        **kwargs,
    ):
        if args is None:
            args = LatentGRPOConfig()
        if not isinstance(args, LatentGRPOConfig):
            raise ValueError("args must be an instance of LatentGRPOConfig")

        if peft_config is not None:
            raise NotImplementedError(
                "LatentGRPOTrainer does not support PEFT yet. "
                "The latent codebook/projector are raw parameters, and PEFT would silently leave them out of training."
            )

        resolved = self._resolve_latent_setup(model, args)
        args.num_latent_tokens = resolved["num_latent_tokens"]
        args.codebook_dim = resolved["codebook_dim"]
        args.codebook_path = resolved["codebook_path"]

        if isinstance(model, (PreTrainedModel, nn.Module)):
            align_model_embeddings_with_tokenizer(model, kwargs.get("processing_class"))
            model, codebook, projector = extend_model_for_latent_tokens(
                model=model,
                num_latent_tokens=resolved["num_latent_tokens"],
                codebook_dim=resolved["codebook_dim"],
                codebook_path=resolved["codebook_path"],
            )
            self.codebook = codebook
            self.projector = projector
            self._latent_embed = model.get_input_embeddings()
            self._latent_lm_head = model.lm_head
            self._configure_parameter_freezing(model, args)

        with self._patch_create_model_from_path(resolved, kwargs.get("processing_class")):
            super().__init__(
                model=model,
                reward_funcs=reward_funcs,
                args=args,
                peft_config=peft_config,
                **kwargs,
            )

        if not hasattr(self, "codebook"):
            self.codebook = self.model.codebook
            self.projector = self.model.projector
            self._latent_embed = self.model.get_input_embeddings()
            self._latent_lm_head = self.model.lm_head
            self._configure_parameter_freezing(self.model, args)

        self._latent_setup = resolved

    @staticmethod
    def _resolve_latent_setup(model, args: LatentGRPOConfig) -> dict:
        model_path = model if isinstance(model, str) else getattr(model.config, "_name_or_path", None)
        config_sources = [args.codebook_path, model_path]

        latent_config = None
        for source in config_sources:
            latent_config = _load_latent_config(source)
            if latent_config is not None:
                break

        num_latent_tokens = args.num_latent_tokens
        codebook_dim = args.codebook_dim
        if latent_config is not None:
            num_latent_tokens = num_latent_tokens or latent_config["num_latent_tokens"]
            codebook_dim = codebook_dim or latent_config["codebook_dim"]

        if num_latent_tokens is None or codebook_dim is None:
            raise ValueError(
                "Could not infer latent setup. Pass `num_latent_tokens` and `codebook_dim`, "
                "or point `model` / `codebook_path` to a checkpoint containing latent/config.json."
            )

        codebook_path = args.codebook_path or model_path

        return {
            "model_path": model_path,
            "codebook_path": codebook_path,
            "num_latent_tokens": num_latent_tokens,
            "codebook_dim": codebook_dim,
        }

    @contextmanager
    def _patch_create_model_from_path(self, latent_setup: dict, processing_class=None):
        original = grpo_trainer_module.create_model_from_path

        def create_model_from_path_with_latent(model_id: str, architecture=None, **kwargs):
            if is_deepseek_decoder_model(model_id):
                model = load_model_for_training(model_id, **kwargs)
            else:
                model = original(model_id, architecture=architecture, **kwargs)
            align_model_embeddings_with_tokenizer(model, processing_class)
            model, _, _ = extend_model_for_latent_tokens(
                model=model,
                num_latent_tokens=latent_setup["num_latent_tokens"],
                codebook_dim=latent_setup["codebook_dim"],
                codebook_path=latent_setup["codebook_path"] or model_id,
            )
            return model

        grpo_trainer_module.create_model_from_path = create_model_from_path_with_latent
        try:
            yield
        finally:
            grpo_trainer_module.create_model_from_path = original
