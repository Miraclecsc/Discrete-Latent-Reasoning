#!/usr/bin/env python
"""
GRPO training entrypoint for latent-extended checkpoints.

This script reuses the processed latent SFT dataset only as a source of prompts and
reward targets. No token-level supervision from SFT is used during RL training.
"""

import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import torch
from datasets import load_dataset

from trl.models.local_model_utils import load_tokenizer_for_model


DEFAULT_MODEL_PATH = "outputs/latent_sft"
DEFAULT_TRAIN_DATA_PATH = "data/latent_grpo.jsonl"
DEFAULT_OUTPUT_DIR = "./latent_grpo_output"
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?")
FINAL_ANSWER_PATTERNS = [
    re.compile(r"####\s*([^\n\r]+)"),
    re.compile(r"(?:^|\n)\s*answer\s*[:：]\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"(?:final answer|final response)\s*(?:is|:)\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"(?:the answer is|answer is)\s*([^\n\r]+)", re.IGNORECASE),
]


def is_main_process(local_rank: int) -> bool:
    return local_rank in (-1, 0)


def log(message: str, local_rank: int) -> None:
    if is_main_process(local_rank):
        print(message)


def decode_text_only(token_ids: list[int], tokenizer, base_vocab_size: int) -> str:
    text_ids = [int(token_id) for token_id in token_ids if 0 <= int(token_id) < base_vocab_size]
    return tokenizer.decode(text_ids, skip_special_tokens=True)


def get_question_and_answer_from_processed(example: dict, tokenizer, base_vocab_size: int) -> tuple[str, str]:
    input_ids = example.get("input_ids")
    labels = example.get("labels")
    if not isinstance(input_ids, list) or not isinstance(labels, list):
        raise KeyError("Processed example is missing `input_ids`/`labels` lists.")
    if len(input_ids) != len(labels):
        raise ValueError("Processed example has mismatched `input_ids` and `labels` lengths.")

    question_ids = [
        int(token_id)
        for token_id, label in zip(input_ids, labels, strict=True)
        if int(label) == -100 and 0 <= int(token_id) < base_vocab_size
    ]
    answer_ids = [
        int(token_id)
        for token_id, label in zip(input_ids, labels, strict=True)
        if int(label) != -100 and 0 <= int(token_id) < base_vocab_size
    ]

    question = tokenizer.decode(question_ids, skip_special_tokens=True).strip()
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return question, answer


def get_question_and_answer(example: dict, tokenizer, base_vocab_size: int) -> tuple[str, str]:
    if "prompt" in example and "answer" in example:
        prompt = example["prompt"]
        if isinstance(prompt, str):
            return prompt.strip(), str(example["answer"]).strip()

    if "question" in example and "answer" in example:
        return str(example["question"]).strip(), str(example["answer"]).strip()

    original = example.get("original", {})
    if isinstance(original, dict) and "question" in original and "answer" in original:
        return str(original["question"]).strip(), str(original["answer"]).strip()

    if "input_ids" in example and "labels" in example:
        return get_question_and_answer_from_processed(example, tokenizer, base_vocab_size)

    raise KeyError("Could not extract question/answer from example.")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def extract_last_number(text: str) -> str:
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return ""
    return matches[-1].replace(",", "")


def extract_last_boxed_answer(text: str) -> str:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return ""

    cursor = start + len(marker)
    depth = 1
    collected = []
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        collected.append(char)
        cursor += 1

    if depth != 0:
        return ""
    return "".join(collected).strip()


def extract_answer_candidates(text: str) -> list[str]:
    text = str(text or "")
    candidates = []

    boxed = extract_last_boxed_answer(text)
    if boxed:
        candidates.append(boxed)

    for pattern in FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            candidates.append(matches[-1].strip())

    last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    if last_line:
        candidates.append(last_line)

    normalized_full = normalize_text(text)
    if normalized_full:
        candidates.append(normalized_full)

    deduped = []
    seen = set()
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def normalize_answer_text(text: str) -> str:
    text = normalize_text(str(text or ""))
    text = re.sub(r"^[#:=：\-\s]+", "", text)
    text = text.rstrip(" \t\r\n.。")
    return text


def parse_number_candidate(text: str):
    cleaned = normalize_answer_text(text)
    cleaned = cleaned.replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    if not cleaned:
        return None

    if re.fullmatch(r"[-+]?\d+/\d+", cleaned):
        numerator, denominator = cleaned.split("/", 1)
        try:
            return Fraction(int(numerator), int(denominator))
        except ZeroDivisionError:
            return None

    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", cleaned):
        try:
            return Fraction(Decimal(cleaned))
        except InvalidOperation:
            return None

    return None


def extract_numeric_answer(text: str):
    for candidate in extract_answer_candidates(text):
        value = parse_number_candidate(candidate)
        if value is not None:
            return value

        last_number = extract_last_number(candidate)
        if last_number:
            value = parse_number_candidate(last_number)
            if value is not None:
                return value

    last_number = extract_last_number(text)
    if last_number:
        return parse_number_candidate(last_number)
    return None


def extract_text_answer(text: str) -> str:
    candidates = extract_answer_candidates(text)
    if not candidates:
        return normalize_answer_text(text)
    return normalize_answer_text(candidates[0])


def make_reward_fn(tokenizer, base_vocab_size: int, reward_mode: str):
    def reward_fn(prompts, completions, completion_ids, answer, **kwargs):
        rewards = []
        for token_ids, target in zip(completion_ids, answer, strict=True):
            predicted_text = decode_text_only(token_ids, tokenizer, base_vocab_size)
            if reward_mode == "number":
                predicted = extract_numeric_answer(predicted_text)
                expected = extract_numeric_answer(str(target))
            else:
                predicted = extract_text_answer(predicted_text)
                expected = extract_text_answer(str(target))
            rewards.append(1.0 if predicted is not None and expected is not None and predicted == expected else 0.0)
        return rewards

    reward_fn.__name__ = f"latent_{reward_mode}_reward"
    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train_data_path", type=str, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--reward_mode", type=str, choices=["number", "exact"], default="number")

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deepseek_attn_implementation",
        type=str,
        choices=("eager", "flash_attention_2"),
        default=None,
    )
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--steps_per_generation", type=int, default=None)
    parser.add_argument("--generation_batch_size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    log("=" * 80, local_rank)
    log("Latent GRPO Training", local_rank)
    log("=" * 80, local_rank)

    ckpt_path = Path(args.model_name_or_path)
    latent_config_path = ckpt_path / "latent" / "config.json"
    if not latent_config_path.exists():
        raise FileNotFoundError(f"Missing latent config: {latent_config_path}")
    if not Path(args.train_data_path).exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data_path}")

    log("\n[1/4] Loading tokenizer and latent metadata...", local_rank)
    tokenizer = load_tokenizer_for_model(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    from trl import LatentGRPOConfig, LatentGRPOTrainer

    with open(latent_config_path, encoding="utf-8") as f:
        latent_config = json.load(f)
    base_vocab_size = latent_config["base_vocab_size"]
    num_latent_tokens = latent_config["num_latent_tokens"]
    codebook_dim = latent_config["codebook_dim"]

    log(f"  checkpoint: {args.model_name_or_path}", local_rank)
    log(f"  base_vocab_size: {base_vocab_size}", local_rank)
    log(f"  num_latent_tokens: {num_latent_tokens}", local_rank)
    log(f"  codebook_dim: {codebook_dim}", local_rank)
    if args.deepseek_attn_implementation is not None:
        log(f"  deepseek_attn_implementation: {args.deepseek_attn_implementation}", local_rank)

    log("\n[2/4] Loading and converting RL dataset...", local_rank)
    train_dataset = load_dataset("json", data_files=str(args.train_data_path), split="train")
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))

    def to_rl_record(example):
        question, answer = get_question_and_answer(example, tokenizer, base_vocab_size)
        return {
            "prompt": question,
            "answer": answer,
            "id": example.get("id"),
        }

    train_dataset = train_dataset.map(to_rl_record, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.filter(lambda example: bool(example["prompt"]) and bool(example["answer"]))
    log(f"  dataset size: {len(train_dataset)}", local_rank)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after conversion.")

    sample = train_dataset[0]
    log(f"  sample prompt: {sample['prompt'][:120]}", local_rank)
    log(f"  sample answer: {sample['answer'][:120]}", local_rank)

    log("\n[3/4] Building LatentGRPOTrainer...", local_rank)
    reward_fn = make_reward_fn(tokenizer, base_vocab_size, args.reward_mode)

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    training_args_dict = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "report_to": [],
        "disable_tqdm": False,
        "remove_unused_columns": False,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "use_vllm": False,
        "model_init_kwargs": {
            "dtype": model_dtype,
            "local_files_only": True,
        },
        "num_latent_tokens": num_latent_tokens,
        "codebook_dim": codebook_dim,
        "codebook_path": args.model_name_or_path,
    }
    if args.steps_per_generation is not None:
        training_args_dict["steps_per_generation"] = args.steps_per_generation
    if args.generation_batch_size is not None:
        training_args_dict["generation_batch_size"] = args.generation_batch_size
    if args.deepseek_attn_implementation is not None:
        training_args_dict["model_init_kwargs"]["deepseek_attn_implementation"] = args.deepseek_attn_implementation
    if args.deepspeed:
        training_args_dict["deepspeed"] = args.deepspeed
    if args.max_steps > 0:
        training_args_dict["max_steps"] = args.max_steps

    training_args = LatentGRPOConfig(**training_args_dict)
    trainer = LatentGRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    log(f"  reward_mode: {args.reward_mode}", local_rank)
    log("  prompt format: question only", local_rank)
    log(f"  beta: {training_args.beta}", local_rank)
    log(f"  num_generations: {training_args.num_generations}", local_rank)
    log(f"  max_completion_length: {training_args.max_completion_length}", local_rank)

    log("\n[4/4] Starting training...", local_rank)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    log("\nSaving final model...", local_rank)
    trainer.save_model(args.output_dir)

    log("\nTraining complete.", local_rank)
    log(f"  output_dir: {args.output_dir}", local_rank)
    log(f"  latent_dir: {args.output_dir}/latent", local_rank)


if __name__ == "__main__":
    main()
