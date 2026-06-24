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
Configuration class for Latent SFT Trainer.
"""

from dataclasses import dataclass, field
from typing import Optional

from .sft_config import SFTConfig


@dataclass
class LatentSFTConfig(SFTConfig):
    """
    Configuration class for [`LatentSFTTrainer`].
    
    Extends [`SFTConfig`] with parameters specific to latent reasoning training.
    
    Args:
        num_latent_tokens (`int`, *optional*, defaults to `10000`):
            Number of latent tokens to add to the vocabulary.
        codebook_dim (`int`, *optional*, defaults to `1024`):
            Dimension of the codebook (e.g., deepseek encoder output dimension).
        codebook_path (`str`, *optional*):
            Path to pretrained codebook file (.pt, .pth, or .safetensors).
            If None, codebook will be initialized randomly.
        freeze_base_model (`bool`, *optional*, defaults to `False`):
            Whether to freeze the base model parameters. If False, all parameters
            including codebook, projector, and base model will be trained (full fine-tuning).
        freeze_codebook (`bool`, *optional*, defaults to `False`):
            Whether to freeze the codebook. If True, only projector will be trained.
    """
    
    num_latent_tokens: int = field(
        default=10000,
        metadata={"help": "Number of latent tokens to add to the vocabulary."}
    )
    codebook_dim: int = field(
        default=1024,
        metadata={"help": "Dimension of the codebook (e.g., deepseek encoder output dimension)."}
    )
    codebook_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained codebook file (.pt, .pth, or .safetensors)."}
    )
    freeze_base_model: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the base model parameters."}
    )
    freeze_codebook: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the codebook. If True, only projector will be trained."}
    )
