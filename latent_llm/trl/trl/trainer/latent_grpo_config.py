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
Configuration class for Latent GRPO training.
"""

from dataclasses import dataclass, field
from typing import Optional

from .grpo_config import GRPOConfig


@dataclass
class LatentGRPOConfig(GRPOConfig):
    """
    GRPO configuration extended with latent-token metadata.
    """

    num_latent_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Number of latent tokens. If omitted, it is inferred from latent/config.json."},
    )
    codebook_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Codebook dimension. If omitted, it is inferred from latent/config.json."},
    )
    codebook_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to latent weights root or latent/ directory. Defaults to the model checkpoint path."},
    )
    freeze_base_model: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the base model parameters and train only latent parameters."},
    )
    freeze_codebook: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the codebook while training the projector/base model."},
    )
