#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModel, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding


DEFAULT_INPUT_JSONL = "outputs/latent_eval_outputs.jsonl"
DEFAULT_CHECKPOINT = "outputs/deepseek_codebook/checkpoint-last"
DEFAULT_CODEBOOK = "outputs/deepseek_codebook/latent/codebook.pt"
DEFAULT_OUTPUT_JSONL = "outputs/latent_decoder_visualization.jsonl"
DEFAULT_PROMPT = "<image>\nFree OCR. "
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
STOP_STR = "<｜end▁of▁sentence｜>"
ALT_STOP_STR = "<｜end of sentence｜>"
MODEL_SOURCE_DIR = SCRIPT_DIR = Path(__file__).resolve().parent / "model_source_10k"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode latent ids with the trained dpsk decoder by replacing image-token embeddings with codebook vectors."
    )
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--codebook", default=DEFAULT_CODEBOOK)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def patch_cache_compat() -> None:
    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, new_seq_length=None, layer_idx=None):
            return self.get_seq_length()
        DynamicCache.get_usable_length = get_usable_length

    if not hasattr(DynamicCache, "get_max_length"):
        def get_max_length(self):
            return None
        DynamicCache.get_max_length = get_max_length

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())


def patch_llama_attention_compat() -> None:
    if getattr(LlamaAttention.forward, "_dpsk_compat", False):
        return

    original_forward = LlamaAttention.forward

    def compat_forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        past_key_value = kwargs.pop("past_key_value", None)
        output_attentions = kwargs.pop("output_attentions", False)
        use_cache = kwargs.pop("use_cache", False)

        if position_embeddings is None:
            if position_ids is None:
                position_ids = torch.arange(
                    hidden_states.shape[1], device=hidden_states.device, dtype=torch.long
                ).unsqueeze(0)
            if not hasattr(self, "_dpsk_compat_rotary_emb"):
                self._dpsk_compat_rotary_emb = LlamaRotaryEmbedding(config=self.config).to(hidden_states.device)
            position_embeddings = self._dpsk_compat_rotary_emb(hidden_states, position_ids)

        if past_key_values is None and past_key_value is not None:
            past_key_values = past_key_value

        attn_output, attn_weights = original_forward(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

        present_key_value = past_key_values if use_cache or past_key_values is not None else None
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, present_key_value

    compat_forward._dpsk_compat = True
    LlamaAttention.forward = compat_forward


def create_patched_checkpoint_dir(checkpoint_dir: str) -> str:
    src = Path(checkpoint_dir)
    patched_dir = Path(tempfile.mkdtemp(prefix="dpsk_ckpt_patch_", dir="/tmp"))

    for path in src.iterdir():
        target = patched_dir / path.name
        if path.name == "config.json":
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg.setdefault("ep_size", 1)
            cfg.setdefault("moe_layer_freq", 1)
            cfg.setdefault("scoring_func", "softmax")
            cfg.setdefault("aux_loss_alpha", 0.001)
            cfg.setdefault("seq_aux", True)
            cfg.setdefault("rope_parameters", {"rope_theta": 10000.0, "rope_type": "default"})
            auto_map = cfg.setdefault("auto_map", {})
            auto_map["AutoConfig"] = "modeling_deepseekocr2.DeepseekOCR2Config"
            auto_map["AutoModel"] = "modeling_deepseekocr2.DeepseekOCR2ForCausalLM"
            if isinstance(cfg.get("language_config"), dict):
                cfg["language_config"].setdefault("ep_size", 1)
                cfg["language_config"].setdefault("moe_layer_freq", 1)
                cfg["language_config"].setdefault("scoring_func", "softmax")
                cfg["language_config"].setdefault("aux_loss_alpha", 0.001)
                cfg["language_config"].setdefault("seq_aux", True)
                cfg["language_config"].setdefault(
                    "rope_parameters", {"rope_theta": 10000.0, "rope_type": "default"}
                )
            with target.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False)
        elif path.name == "modeling_deepseekocr2.py":
            with path.open("r", encoding="utf-8") as f:
                text = f.read()
            text = text.replace(
                "from transformers import DeepseekV2Model, DeepseekV2ForCausalLM",
                "from .modeling_deepseekv2 import DeepseekV2Model, DeepseekV2ForCausalLM",
            )
            text = text.replace(
                "from transformers import DeepseekV2Config",
                "from .configuration_deepseek_v2 import DeepseekV2Config",
            )
            text = text.replace(
                "from transformers.models.deepseek_v2.modeling_deepseek_v2 import (\n"
                "    DeepseekV2Attention,\n"
                "    DeepseekV2MLP,\n"
                "    DeepseekV2MoE,\n"
                "    DeepseekV2RMSNorm,\n"
                "    DeepseekV2DecoderLayer,\n"
                ")",
                "from .modeling_deepseekv2 import (\n"
                "    DeepseekV2Attention,\n"
                "    DeepseekV2MLP,\n"
                "    DeepseekV2MoE,\n"
                "    DeepseekV2RMSNorm,\n"
                "    DeepseekV2DecoderLayer,\n"
                ")",
            )
            with target.open("w", encoding="utf-8") as f:
                f.write(text)
        else:
            os.symlink(path, target)

    for name in ["configuration_deepseek_v2.py", "__init__.py"]:
        target = patched_dir / name
        if not target.exists():
            os.symlink(MODEL_SOURCE_DIR / name, target)

    deepseekv2_src = MODEL_SOURCE_DIR / "modeling_deepseekv2.py"
    deepseekv2_target = patched_dir / "modeling_deepseekv2.py"
    with deepseekv2_src.open("r", encoding="utf-8") as f:
        text = f.read()
    text = text.replace(
        "from transformers.utils.import_utils import is_torch_fx_available",
        "def is_torch_fx_available():\n    return False",
    )
    with deepseekv2_target.open("w", encoding="utf-8") as f:
        f.write(text)
    return str(patched_dir)


def load_model_and_tokenizer(checkpoint_dir: str, device: str, attn_impl: str):
    patch_cache_compat()
    patch_llama_attention_compat()
    patched_ckpt_dir = create_patched_checkpoint_dir(checkpoint_dir)

    torch_dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.startswith("cuda") else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(patched_ckpt_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        patched_ckpt_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        _attn_implementation=attn_impl,
    )
    model.eval()
    model.to(device)
    return model, tokenizer, patched_ckpt_dir


def format_messages(conversations: list[dict[str, str]], sft_format: str = "plain", system_prompt: str = "") -> str:
    remote_module = __import__(model_module_name(), fromlist=["get_conv_template"])
    conv = remote_module.get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    return conv.get_prompt().strip()


_MODEL_MODULE_NAME: str | None = None


def model_module_name() -> str:
    if _MODEL_MODULE_NAME is None:
        raise RuntimeError("Model module name is not initialized")
    return _MODEL_MODULE_NAME


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False) -> list[int]:
    t = tokenizer.encode(text, add_special_tokens=False)
    if bos:
        t = [0] + t
    if eos:
        t = t + [1]
    return t


def build_prompt_and_mask(tokenizer, prompt: str, image_token_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    text_splits = prompt.split(IMAGE_TOKEN)
    if len(text_splits) != 2:
        raise ValueError(f"Prompt must contain exactly one {IMAGE_TOKEN!r}: {prompt!r}")

    tokenized_str: list[int] = []
    images_seq_mask: list[bool] = []

    tokenized_sep = text_encode(tokenizer, text_splits[0], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    tokenized_image = [IMAGE_TOKEN_ID] * image_token_count
    tokenized_str += tokenized_image
    images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = text_encode(tokenizer, text_splits[1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    tokenized_str = [0] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    return torch.tensor(tokenized_str, dtype=torch.long), torch.tensor(images_seq_mask, dtype=torch.bool)


def extract_latent_segment(record: dict[str, Any]) -> dict[str, Any]:
    base_vocab_size = int(record["base_vocab_size"])
    generated_ids = list(record["generated_ids"])

    latent_extended_ids = [token_id for token_id in generated_ids if token_id >= base_vocab_size]
    if not latent_extended_ids:
        raise ValueError(f"No latent ids found in record index={record.get('index')}")

    latent_codebook_ids = [token_id - base_vocab_size for token_id in latent_extended_ids]
    placeholder_terminated = latent_codebook_ids[-1] == 0
    content_codebook_ids = latent_codebook_ids[:-1] if placeholder_terminated else latent_codebook_ids

    return {
        "latent_extended_ids": latent_extended_ids,
        "latent_codebook_ids": latent_codebook_ids,
        "content_codebook_ids": content_codebook_ids,
        "placeholder_terminated": placeholder_terminated,
        "base_vocab_size": base_vocab_size,
    }


def decode_with_codebook_features(
    model,
    tokenizer,
    codebook: torch.Tensor,
    codebook_ids: list[int],
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    if not codebook_ids:
        raise ValueError("No content codebook ids to decode")

    input_ids, images_seq_mask = build_prompt_and_mask(tokenizer, prompt, len(codebook_ids))
    base_input_ids = input_ids.unsqueeze(0).to(device)
    base_images_seq_mask = images_seq_mask.unsqueeze(0).to(device)

    codebook_features = codebook.index_select(
        0, torch.tensor(codebook_ids, dtype=torch.long, device=codebook.device)
    )
    codebook_features = codebook_features.to(device=device, dtype=model.dtype)

    current_input_ids = base_input_ids
    generated_ids: list[int] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            attention_mask = torch.ones_like(current_input_ids, device=device)
            inputs_embeds = model.get_input_embeddings()(current_input_ids).clone()
            prefix_mask = torch.zeros(
                (1, current_input_ids.shape[1]), dtype=torch.bool, device=device
            )
            prefix_mask[:, : base_images_seq_mask.shape[1]] = base_images_seq_mask
            inputs_embeds[0] = inputs_embeds[0].masked_scatter(
                prefix_mask[0].unsqueeze(-1), codebook_features
            )

            outputs = model(
                input_ids=current_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )
            next_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            generated_ids.append(next_token_id)

            next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)

            if next_token_id == tokenizer.eos_token_id:
                break

    decoded_text = tokenizer.decode(generated_ids)
    if decoded_text.endswith(STOP_STR):
        decoded_text = decoded_text[: -len(STOP_STR)]
    if decoded_text.endswith(ALT_STOP_STR):
        decoded_text = decoded_text[: -len(ALT_STOP_STR)]
    decoded_text = decoded_text.replace("Ġ", " ")
    decoded_text = decoded_text.replace("Ċ", "\n")
    decoded_text = decoded_text.strip()

    return {
        "decoder_generated_ids": generated_ids,
        "decoder_output_text": decoded_text,
        "decoder_prompt": prompt,
    }


def load_codebook(codebook_path: str, model, device: str) -> torch.Tensor:
    codebook = torch.load(codebook_path, map_location="cpu")
    if not isinstance(codebook, torch.Tensor):
        raise TypeError(f"Expected tensor codebook, got {type(codebook)}")
    codebook = codebook.to(dtype=model.dtype, device=device)

    base_model = model.model if hasattr(model, "model") else model
    if not hasattr(base_model, "codebook"):
        raise AttributeError("Loaded model does not have a codebook parameter")
    if tuple(base_model.codebook.shape) != tuple(codebook.shape):
        raise ValueError(
            f"Codebook shape mismatch: model {tuple(base_model.codebook.shape)} vs external {tuple(codebook.shape)}"
        )
    with torch.no_grad():
        base_model.codebook.copy_(codebook)
    return codebook


def read_first_n_records(input_jsonl: str, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def write_jsonl(output_jsonl: str, rows: list[dict[str, Any]]) -> None:
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    output_path = Path(args.output_jsonl)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    model, tokenizer, patched_ckpt_dir = load_model_and_tokenizer(
        args.checkpoint,
        args.device,
        args.attn_impl,
    )
    global _MODEL_MODULE_NAME
    _MODEL_MODULE_NAME = model.__class__.__module__

    codebook = load_codebook(args.codebook, model, args.device)
    records = read_first_n_records(args.input_jsonl, args.limit)
    formatted_prompt = format_messages(
        [
            {"role": "<|User|>", "content": args.prompt},
            {"role": "<|Assistant|>", "content": ""},
        ],
        sft_format="plain",
        system_prompt="",
    )

    rows: list[dict[str, Any]] = []
    total_records = len(records)
    for row_idx, record in enumerate(records, start=1):
        print(
            f"[decode] {row_idx}/{total_records} "
            f"index={record.get('index')} sample_index={record.get('sample_index')}",
            flush=True,
        )
        latent_info = extract_latent_segment(record)
        decoded = decode_with_codebook_features(
            model=model,
            tokenizer=tokenizer,
            codebook=codebook,
            codebook_ids=latent_info["content_codebook_ids"],
            prompt=formatted_prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )

        rows.append(
            {
                "index": record.get("index"),
                "sample_index": record.get("sample_index"),
                "sample_seed": record.get("sample_seed"),
                "num_samples": record.get("num_samples"),
                "selected_pass_label": record.get("selected_pass_label"),
                "selection_reason": record.get("selection_reason"),
                "prediction_text": record.get("prediction_text"),
                "prediction_is_correct": record.get("prediction_is_correct"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "generated_text_only": record.get("generated_text_only"),
                "generated_mixed": record.get("generated_mixed"),
                "latent_extended_ids": latent_info["latent_extended_ids"],
                "latent_codebook_ids": latent_info["latent_codebook_ids"],
                "content_codebook_ids": latent_info["content_codebook_ids"],
                "placeholder_terminated": latent_info["placeholder_terminated"],
                "decoder_prompt": decoded["decoder_prompt"],
                "decoder_generated_ids": decoded["decoder_generated_ids"],
                "decoder_output_text": decoded["decoder_output_text"],
                "dpsk_checkpoint": args.checkpoint,
                "external_codebook": args.codebook,
                "patched_checkpoint_dir": patched_ckpt_dir,
            }
        )
        print(
            f"[decoded] {row_idx}/{total_records} "
            f"index={record.get('index')} sample_index={record.get('sample_index')} "
            f"latent_len={len(latent_info['content_codebook_ids'])}",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), rows)
    print(f"[done] wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
