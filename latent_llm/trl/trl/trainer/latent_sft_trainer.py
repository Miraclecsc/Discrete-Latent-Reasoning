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
Latent SFT Trainer for training language models with expanded vocabulary for latent reasoning.
"""

import os
import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from .sft_trainer import SFTTrainer
from .latent_sft_config import LatentSFTConfig
from ..models.latent_utils import (
    LatentEmbedding,
    LatentLMHead,
    align_model_embeddings_with_tokenizer,
    extend_model_for_latent_tokens,
    infer_tokenizer_vocab_size,
)


class LatentSFTTrainer(SFTTrainer):
    """
    Trainer for Supervised Fine-Tuning with latent tokens.
    
    This trainer extends the standard SFTTrainer to support:
    1. Expanded vocabulary with latent tokens
    2. Weight binding between embedding and lm_head for latent tokens
    3. Dynamic computation of latent embeddings via codebook + projector
    
    The key innovation is that latent tokens are treated as regular vocabulary tokens,
    allowing efficient teacher-forcing training with standard cross-entropy loss.
    
    Args:
        model (`Union[PreTrainedModel, nn.Module, str]`):
            The model to train or path to a pretrained model.
        args (`LatentSFTConfig`, *optional*):
            Configuration for the trainer.
        tokenizer (`PreTrainedTokenizer`, *optional*):
            The tokenizer to use.
        **kwargs:
            Additional arguments passed to [`SFTTrainer`].
    """
    
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[LatentSFTConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs
    ):
        if args is None:
            args = LatentSFTConfig()
        
        if not isinstance(args, LatentSFTConfig):
            raise ValueError("args must be an instance of LatentSFTConfig")

        processing_class = kwargs.get("processing_class", tokenizer)
        
        # Extend model for latent tokens before calling parent __init__
        # This ensures the model is ready before any data processing
        if isinstance(model, (PreTrainedModel, nn.Module)):
            print("\n" + "="*50)
            print("Extending model for latent tokens...")
            print("="*50)
            if processing_class is not None:
                align_model_embeddings_with_tokenizer(model, processing_class)
            
            model, codebook, projector = extend_model_for_latent_tokens(
                model=model,
                num_latent_tokens=args.num_latent_tokens,
                codebook_dim=args.codebook_dim,
                codebook_path=args.codebook_path,
            )
            
            self.codebook = codebook
            self.projector = projector
            
            # Store references for later use
            self._latent_embed = model.get_input_embeddings()
            self._latent_lm_head = model.lm_head
            
            # Configure parameter freezing
            self._configure_parameter_freezing(model, args)
            
            print("="*50)
            print("Model extension complete!")
            print("="*50 + "\n")
        
        # Call parent __init__
        # SFTTrainer uses processing_class instead of tokenizer
        if 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(model=model, args=args, **kwargs)
    
    def _configure_parameter_freezing(self, model, args: LatentSFTConfig):
        """
        Configure which parameters to freeze/train.
        
        By default (freeze_base_model=False, freeze_codebook=False):
        - All parameters are trainable (full fine-tuning)
        
        If freeze_base_model=True:
        - Only codebook and projector are trainable
        
        If freeze_codebook=True:
        - Codebook is frozen, projector is trainable
        """
        if args.freeze_base_model:
            # Freeze all base model parameters
            for name, param in model.named_parameters():
                if name not in ['codebook', 'projector'] and 'codebook' not in name and 'projector' not in name:
                    param.requires_grad = False
            
            # Ensure codebook and projector are trainable
            self.codebook.requires_grad = not args.freeze_codebook
            self.projector.requires_grad = True
            
            print("Base model frozen. Only training codebook and projector.")
        elif args.freeze_codebook:
            # Freeze only codebook
            self.codebook.requires_grad = False
            self.projector.requires_grad = True
            print("Codebook frozen. Training projector and base model.")
        else:
            # Full fine-tuning: all parameters trainable
            print("Full fine-tuning: all parameters (base model, codebook, projector) are trainable.")
        
        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model in a production-ready format.
        
        Directory structure:
            output_dir/
            ├── config.json                  # Model config (base vocab size)
            ├── model.safetensors           # Base model weights (trained)
            ├── tokenizer.*                 # Tokenizer files
            ├── generation_config.json      # Generation config
            ├── training_args.bin           # Training arguments
            ├── trainer_state.json          # Trainer state
            ├── latent/                     # Latent weights directory
            │   ├── codebook.safetensors   # Codebook weights
            │   ├── projector.safetensors  # Projector weights
            │   ├── config.json            # Latent config
            │   └── README.md              # Documentation
            └── checkpoints/               # Checkpoints (optional)
        
        Args:
            output_dir: Directory to save the model
            _internal_call: Whether this is an internal call
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base vocab size
        base_vocab_size = self._latent_embed.base_embed.num_embeddings
        num_latent_tokens = self.codebook.shape[0]
        
        # Save latent weights to dedicated directory
        latent_dir = output_dir / "latent"
        latent_dir.mkdir(exist_ok=True)
        
        self._save_latent_weights(latent_dir, base_vocab_size, num_latent_tokens)
        
        # Save base model with standard save_pretrained
        self._save_base_model(output_dir, base_vocab_size, num_latent_tokens, _internal_call)
        
        # Save README
        self._save_readme(output_dir, base_vocab_size, num_latent_tokens)
        
        # Log summary
        if self.args.should_save:
            self._log_save_summary(output_dir, latent_dir)
    
    def _save_latent_weights(self, latent_dir: Path, base_vocab_size: int, num_latent_tokens: int):
        """Save codebook and projector weights in safetensors format."""
        try:
            from safetensors.torch import save_file
            
            # Save codebook
            codebook_path = latent_dir / "codebook.safetensors"
            save_file({"codebook": self.codebook.detach().cpu()}, codebook_path)
            
            # Save projector
            projector_path = latent_dir / "projector.safetensors"
            save_file({"projector": self.projector.detach().cpu()}, projector_path)
            
        except ImportError:
            # Fallback to torch.save if safetensors not available
            codebook_path = latent_dir / "codebook.pt"
            torch.save(self.codebook.detach().cpu(), codebook_path)
            
            projector_path = latent_dir / "projector.pt"
            torch.save(self.projector.detach().cpu(), projector_path)
        
        # Save latent config
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
        
        # Save README
        readme_content = f"""# Latent Weights

This directory contains the latent token weights for the extended vocabulary model.

## Files

- `codebook.safetensors` / `codebook.pt`: Codebook tensor of shape [{num_latent_tokens}, {self.codebook.shape[1]}]
- `projector.safetensors` / `projector.pt`: Projector matrix of shape [{self.codebook.shape[1]}, {self.projector.shape[1]}]
- `config.json`: Latent configuration

## Configuration

```json
{json.dumps(latent_config, indent=2)}
```

## Usage

Latent embeddings are computed as:
```python
latent_embeddings = torch.matmul(codebook, projector)  # [{num_latent_tokens}, {self.projector.shape[1]}]
```

These embeddings extend the base vocabulary [{base_vocab_size}] to [{base_vocab_size + num_latent_tokens}].

## Loading

```python
from trl.models.latent_utils import extend_model_for_latent_tokens

# Load base model
model = AutoModel.from_pretrained("path/to/model")

# Extend with latent tokens
model, codebook, projector = extend_model_for_latent_tokens(
    model=model,
    num_latent_tokens={num_latent_tokens},
    codebook_dim={self.codebook.shape[1]},
    codebook_path="path/to/model/latent",
)
```
"""
        with open(latent_dir / "README.md", "w") as f:
            f.write(readme_content)
    
    def _save_base_model(self, output_dir: Path, base_vocab_size: int, num_latent_tokens: int, _internal_call: bool):
        """Save base model with original vocab size."""
        model = self.model
        extended_vocab_size = base_vocab_size + num_latent_tokens
        
        # Get trained base weights
        trained_base_embed = self._latent_embed.base_embed
        trained_base_lm_head = self._latent_lm_head.base_lm_head
        
        # Store original modules
        original_embed = model.get_input_embeddings()
        original_lm_head = model.lm_head
        
        # Temporarily replace with base weights for saving
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens = trained_base_embed
        elif hasattr(model, 'embed_tokens'):
            model.embed_tokens = trained_base_embed
        elif hasattr(model, 'set_input_embeddings'):
            model.set_input_embeddings(trained_base_embed)
        
        model.lm_head = trained_base_lm_head
        
        # Temporarily restore original vocab size in config
        if hasattr(model, 'config'):
            if hasattr(model.config, 'vocab_size'):
                model.config.vocab_size = base_vocab_size
            if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'vocab_size'):
                model.config.text_config.vocab_size = base_vocab_size
        if hasattr(model, 'vocab_size'):
            model.vocab_size = base_vocab_size
        
        # Save base model using parent's save_model
        try:
            super().save_model(str(output_dir), _internal_call)
        finally:
            # Restore latent modules
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                model.model.embed_tokens = original_embed
            elif hasattr(model, 'embed_tokens'):
                model.embed_tokens = original_embed
            elif hasattr(model, 'set_input_embeddings'):
                model.set_input_embeddings(original_embed)
            
            model.lm_head = original_lm_head
            
            # Restore vocab size to EXTENDED training value.
            # This guarantees training continues with latent vocab after checkpoint save.
            if hasattr(model, 'config'):
                if hasattr(model.config, 'vocab_size'):
                    model.config.vocab_size = extended_vocab_size
                if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'vocab_size'):
                    model.config.text_config.vocab_size = extended_vocab_size
            if hasattr(model, 'vocab_size'):
                model.vocab_size = extended_vocab_size
    
    def _save_readme(self, output_dir: Path, base_vocab_size: int, num_latent_tokens: int):
        """Save main README.md."""
        readme_content = f"""# Latent SFT Model

This model has been trained with latent tokens for visual reasoning.

## Model Structure

### Base Model
- **Vocabulary Size**: {base_vocab_size}
- **Hidden Size**: {self.projector.shape[1]}
- **Files**: Standard HuggingFace format (`config.json`, `model.safetensors`, etc.)

### Latent Tokens
- **Number of Latent Tokens**: {num_latent_tokens}
- **Codebook Dimension**: {self.codebook.shape[1]}
- **Location**: `latent/` directory

## Total Vocabulary

The complete vocabulary consists of:
- Base tokens: `[0, {base_vocab_size-1}]` ({base_vocab_size} tokens)
- Latent tokens: `[{base_vocab_size}, {base_vocab_size + num_latent_tokens - 1}]` ({num_latent_tokens} tokens)
- **Total**: {base_vocab_size + num_latent_tokens} tokens

## Usage

### Loading the Model

```python
from transformers import AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from trl.models.latent_utils import extend_model_for_latent_tokens

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Load base model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "path/to/model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Extend with latent tokens
model, codebook, projector = extend_model_for_latent_tokens(
    model=model,
    num_latent_tokens={num_latent_tokens},
    codebook_dim={self.codebook.shape[1]},
    codebook_path="path/to/model/latent",
)
```

### Training Configuration

See `training_args.bin` for full training configuration.

## File Structure

```
.
├── config.json                  # Model configuration
├── model.safetensors           # Model weights (base vocab)
├── tokenizer.json              # Tokenizer
├── tokenizer_config.json       # Tokenizer config
├── generation_config.json      # Generation config
├── latent/                     # Latent token weights
│   ├── codebook.safetensors   # Codebook
│   ├── projector.safetensors  # Projector
│   ├── config.json            # Latent config
│   └── README.md              # Latent weights documentation
└── checkpoints/               # Training checkpoints (if any)
```

## Citation

If you use this model, please cite the original Qwen2.5-VL paper and the TRL library.
"""
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)
    
    def _log_save_summary(self, output_dir: Path, latent_dir: Path):
        """Log save summary."""
        print(f"\n{'='*60}")
        print("Model saved successfully (Production Format)")
        print(f"{'='*60}")
        print(f"\n📁 Output Directory: {output_dir}")
        print(f"\n📦 Base Model:")
        print(f"   - config.json, model.safetensors, tokenizer.*")
        print(f"   - Vocab size: {self._latent_embed.base_embed.num_embeddings}")
        print(f"\n🔮 Latent Weights:")
        print(f"   - Directory: {latent_dir}")
        print(f"   - codebook.safetensors: {tuple(self.codebook.shape)}")
        print(f"   - projector.safetensors: {tuple(self.projector.shape)}")
        print(f"   - config.json, README.md")
        print(f"\n💡 To load this model:")
        print(f"   model = AutoModel.from_pretrained('{output_dir}')")
        print(f"   model, cb, proj = extend_model_for_latent_tokens(")
        print(f"       model, codebook_path='{output_dir}/latent'")
        print(f"   )")
        print(f"{'='*60}\n")
    
    def create_optimizer(self):
        """
        Create optimizer.
        
        Ensures codebook and projector are included in the optimizer.
        """
        # The model parameters already include codebook and projector
        # as they are registered as nn.Parameter in the model
        return super().create_optimizer()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with DDP compatibility fix.
        
        This override ensures that codebook and projector always receive gradients
        in distributed training, even when a batch doesn't contain any latent tokens.
        This prevents DDP sync errors like:
        "Expected to have finished reduction in the prior iteration"
        
        The fix adds a dummy term to the loss that forces gradients to flow through
        codebook and projector while having zero effect on the actual loss value.
        """
        # Call parent compute_loss
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        
        # Extract loss from result (parent returns (loss, outputs) if return_outputs=True)
        if return_outputs:
            loss, outputs = result
        else:
            loss = result
        
        # DDP fix: Ensure codebook and projector always receive gradients
        # This is critical when gradient_accumulation_steps > 1 and some batches
        # don't contain any latent tokens. The dummy term has zero effect on
        # the loss value but ensures all DDP ranks have the same gradient flow.
        if model.training and hasattr(self, 'codebook') and hasattr(self, 'projector'):
            dummy_term = self.codebook.sum() * 0.0 + self.projector.sum() * 0.0
            loss = loss + dummy_term
        
        if return_outputs:
            return (loss, outputs)
        return loss
