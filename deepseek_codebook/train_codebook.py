# from huggingface_hub import snapshot_download
# snapshot_download("unsloth/DeepSeek-OCR-2", local_dir = "deepseek_ocr2")

from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from transformers import AutoModel
import torch.nn.functional as F
import os
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SOURCE_PATH = os.path.join(SCRIPT_DIR, "model_source_10k")
DEFAULT_DATA_ROOT = os.environ.get("DLR_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data"))
DEFAULT_OUTPUT_ROOT = os.environ.get("DLR_OUTPUT_ROOT", os.path.join(SCRIPT_DIR, "..", "outputs"))
CODEBOOK_INIT_PACKAGE_PATH = os.environ.get(
    "CODEBOOK_INIT_PACKAGE_PATH",
    os.path.join(DEFAULT_DATA_ROOT, "codebook_init_10k_vmf.pt"),
)
TRAIN_DATASET_PATH = os.environ.get(
    "DLR_CODEBOOK_TRAIN_DATASET",
    os.path.join(DEFAULT_DATA_ROOT, "rendered_cot_hf"),
)
TRAIN_SPLIT_PATH = os.environ.get(
    "DLR_CODEBOOK_TRAIN_SPLIT",
    os.path.join(DEFAULT_OUTPUT_ROOT, "codebook_train_split"),
)
EVAL_SPLIT_PATH = os.environ.get(
    "DLR_CODEBOOK_EVAL_SPLIT",
    os.path.join(DEFAULT_OUTPUT_ROOT, "codebook_eval_split"),
)
BALANCED_OUTPUT_DIR = os.environ.get(
    "DLR_CODEBOOK_OUTPUT_DIR",
    os.path.join(DEFAULT_OUTPUT_ROOT, "deepseek_codebook"),
)
FINAL_OUTPUT_DIR = os.environ.get(
    "DLR_CODEBOOK_FINAL_DIR",
    os.path.join(BALANCED_OUTPUT_DIR, "final"),
)

os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit", # Qwen 3 vision support
    "unsloth/Qwen3-VL-8B-Thinking-bnb-4bit",
    "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit",
    "unsloth/Qwen3-VL-32B-Thinking-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_SOURCE_PATH,
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    auto_model = AutoModel,
    trust_remote_code = True,
    unsloth_force_compile = True,
    # use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    use_gradient_checkpointing = False,
    full_finetuning = True,
)


from datasets import load_from_disk
dataset = load_from_disk(TRAIN_DATASET_PATH)
print("*"*50)
print(len(dataset))


''' # eval
dataset[1523]['image_path'].save("your_image.jpg")

dataset[1523]['image_path']


# prompt = "<image>\nFree OCR. "
prompt = "<image>\nFree OCR. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'
# infer(self, tokenizer, prompt = '', image_file = '', output_path = ' ', base_size = 1024, image_size = 768, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 768, image_size = 768, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 768, crop_mode = True

res = model.infer(tokenizer, prompt = prompt, image_file = image_file, output_path = output_path, base_size = 1024, image_size = 768, crop_mode = True, save_results = True, test_compress = False)

dataset[1523]["text"]
'''

# model = FastVisionModel.get_peft_model(
#     model,
#     target_modules = [
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ],

#     r = 16,           # The larger, the higher the accuracy, but might overfit
#     lora_alpha = 16,  # Recommended alpha == r at least
#     lora_dropout = 0,
#     bias = "none",
#     random_state = 3407,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
#     # target_modules = "all-linear", # Optional now! Can specify a list if needed
# )

''' # data format
[
{ "role": "<|User|>",
  "content": "",
  "images": []
},
{ "role": "<|Assistant|>",
  "content": ""
},
]
'''

instruction = "<image>\nFree OCR. "

def convert_to_conversation(sample):
    """Convert dataset sample to conversation format"""
    conversation = [
        {
            "role": "<|User|>",
            "content": instruction,
            "images": [sample['image']]
        },
        {
            "role": "<|Assistant|>",
            "content": sample["text"]
        },
    ]
    return {"messages": conversation}

# Load dataset
# Note: The new dataset has image_path as file paths, need to load PIL images
from PIL import Image

# Don't use dataset.map() - it's too slow for large datasets
# Instead, we'll load images on-the-fly in the data collator

# Split dataset into train and test (95% train, 5% test)
from datasets import Dataset as HFDataset
import numpy as np

# Check if split already exists
train_split_path = TRAIN_SPLIT_PATH
test_split_path = EVAL_SPLIT_PATH

if os.path.exists(train_split_path) and os.path.exists(test_split_path):
    print(f"Loading existing train/test splits from {train_split_path} and {test_split_path}")
    train_dataset = HFDataset.load_from_disk(train_split_path)
    test_dataset = HFDataset.load_from_disk(test_split_path)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
else:
    print("Creating new train/test split...")
    total_size = len(dataset)
    test_size = int(total_size * 0.01)
    train_size = total_size - test_size

    # Create indices for splitting
    indices = np.random.RandomState(seed=3407).permutation(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Split dataset (keep image_path, don't load images yet)
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)

    print(f"Total dataset size: {total_size}")
    print(f"Train dataset size: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"Test dataset size: {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")

    # Save train/test split for reproducibility
    train_dataset.save_to_disk(train_split_path)
    test_dataset.save_to_disk(test_split_path)
    print(f"Saved train/test splits to {train_split_path} and {test_split_path}")

# Convert to conversation format (images will be loaded on-the-fly)
def convert_to_conversation_lazy(sample):
    """Convert dataset sample to conversation format (lazy loading)"""
    conversation = [
        {
            "role": "<|User|>",
            "content": instruction,
            "images": [sample['image_path']]  # Keep path, not PIL Image
        },
        {
            "role": "<|Assistant|>",
            "content": sample["text"]
        },
    ]
    return {"messages": conversation}

converted_train_dataset = [convert_to_conversation_lazy(sample) for sample in train_dataset]
converted_test_dataset = [convert_to_conversation_lazy(sample) for sample in test_dataset]

print(f"Dataset conversion completed. Ready to start training!")
converted_train_dataset[0]

# @title Create datacollator

import torch
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
import io

import importlib

remote_module_name = model.model.__class__.__module__
remote_module = importlib.import_module(remote_module_name)

format_messages = remote_module.format_messages
text_encode = remote_module.text_encode
BasicImageTransform = remote_module.BasicImageTransform
dynamic_preprocess = remote_module.dynamic_preprocess


def get_base_model(model):
    root_model = model.model if hasattr(model, "model") else model
    return root_model.model if hasattr(root_model, "model") else root_model

def load_codebook_init_package_into_model(model):
    base_model = get_base_model(model)
    if not os.path.exists(CODEBOOK_INIT_PACKAGE_PATH):
        raise FileNotFoundError(
            f"Codebook init package not found: {CODEBOOK_INIT_PACKAGE_PATH}"
        )

    package = torch.load(CODEBOOK_INIT_PACKAGE_PATH, map_location="cpu")
    if "codebook" not in package:
        raise KeyError(f"Invalid codebook package: missing `codebook` key in {CODEBOOK_INIT_PACKAGE_PATH}")

    codebook_tensor = package["codebook"]
    if tuple(codebook_tensor.shape) != tuple(base_model.codebook.shape):
        raise ValueError(
            f"Codebook shape mismatch: package={tuple(codebook_tensor.shape)} "
            f"model={tuple(base_model.codebook.shape)}"
        )

    device = base_model.codebook.device
    dtype = base_model.codebook.dtype
    base_model.codebook = torch.nn.Parameter(
        codebook_tensor.to(device=device, dtype=dtype).contiguous()
    )

    base_model.codebook_usage_ema.zero_()
    base_model.codebook_idle_steps.zero_()
    base_model.codebook_step.zero_()
    base_model.codebook_last_unique_codes.zero_()
    base_model.codebook_last_top1_fraction.zero_()
    base_model.codebook_last_top5_fraction.zero_()
    base_model.codebook_last_effective_codes.zero_()
    base_model.codebook_last_refresh_count.zero_()
    base_model.codebook_recent_bank.zero_()
    base_model.codebook_recent_bank_ptr.zero_()
    base_model.codebook_recent_bank_count.zero_()
    base_model.codebook_target_norm.fill_(float(package["target_norm"]))
    base_model.codebook_noise_std.fill_(float(package["noise_std"]))
    base_model.codebook_vmf_kappa.fill_(float(package["vmf_kappa"]))
    base_model.codebook_last_top1_distance.fill_(float(package["top1_distance"]))
    base_model.codebook_last_top10_distance.fill_(float(package["top10_distance"]))
    top128_distance = float(package.get("top128_distance", package.get("top64_distance", 0.0)))
    top128_angle = float(package.get("top128_angle", package.get("top64_angle", 0.0)))
    base_model.codebook_last_top128_distance.fill_(top128_distance)
    base_model.codebook_last_top128_angle.fill_(top128_angle)
    base_model.project_codebook_to_sphere_()

    print(
        f"[INIT] codebook package loaded. "
        f"path={CODEBOOK_INIT_PACKAGE_PATH} "
        f"shape={tuple(base_model.codebook.shape)} "
        f"target_norm={float(base_model.codebook_target_norm.item()):.6f} "
        f"noise_std={float(base_model.codebook_noise_std.item()):.6f} "
        f"vmf_kappa={float(base_model.codebook_vmf_kappa.item()):.6f}"
    )
    if package.get("metadata") is not None:
        print(f"[INIT] codebook metadata: {package['metadata']}")


load_codebook_init_package_into_model(model)
base_model = get_base_model(model)
base_model.codebook_st_mode = "strict"
base_model.codebook_assign_topk = 128
base_model.codebook_assign_capacity = 1
base_model.codebook_soft_update_topk = 1
base_model.codebook_usage_penalty_alpha = 1.5
base_model.codebook_idle_reward_alpha = 0.25
base_model.codebook_enable_refresh = True
base_model.codebook_dead_steps = 50
base_model.codebook_refresh_interval = 10
base_model.codebook_refresh_max_codes = 128
base_model.codebook_refresh_noise_scale = 0.02
print(
    "[INIT] balanced-assignment mode enabled. "
    "st_mode=strict assign_topk=128 capacity=1 soft_update_topk=1 "
    "usage_penalty_alpha=1.5 idle_reward_alpha=0.25 "
    "refresh=(enabled dead_steps=50 interval=10 max_codes=128)"
)

@dataclass
class DeepSeekOCR2DataCollator:
    """
    Args:
        tokenizer: Tokenizer
        model: Model
        image_size: Size for image patches (default: 768)
        base_size: Size for global view (default: 1024)
        crop_mode: Whether to use dynamic cropping for large images
        train_on_responses_only: If True, only train on assistant responses (mask user prompts)
    """
    tokenizer: Any
    model: Any
    image_size: int = 768
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True

    def __init__(
        self,
        tokenizer,
        model,
        image_size: int = 768,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype  # Get dtype from model
        self.train_on_responses_only = train_on_responses_only

        self.image_transform = BasicImageTransform(
            mean = (0.5, 0.5, 0.5),
            std = (0.5, 0.5, 0.5),
            normalize = True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        # Get BOS token ID from tokenizer
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0
            print(f"Warning: tokenizer has no bos_token_id, using default: {self.bos_id}")

    def deserialize_image(self, image_data) -> Image.Image:
        """Convert image data (bytes dict, PIL Image, or file path) to PIL Image in RGB mode"""
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, str):
            # It's a file path - load on-the-fly
            return Image.open(image_data).convert("RGB")
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def calculate_image_token_count(self, image: Image.Image, crop_ratio: Tuple[int, int]) -> int:
        """Calculate the number of tokens this image will generate"""
        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

        width_crop_num, height_crop_num = crop_ratio

        if self.crop_mode:
            img_tokens = num_queries_base * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                img_tokens += (num_queries * width_crop_num) * (num_queries * height_crop_num)
        else:
            target_size = min(max(image.size[0], image.size[1]), self.image_size)
            num_queries = math.ceil((target_size // self.patch_size) / self.downsample_ratio)
            img_tokens = num_queries * num_queries + 1

        return img_tokens

    def process_image(self, image: Image.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
        """
        Process a single image based on crop_mode and size thresholds

        Returns:
            Tuple of (images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio)
        """
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            # Determine crop ratio based on image size
            if image.size[0] <= 768 and image.size[1] <= 768:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num = 2, max_num = 6,
                    image_size = self.image_size, use_thumbnail = False
                )

            # Process global view with padding
            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color = tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # Process local views (crops) if applicable
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(
                        self.image_transform(crop_img).to(self.dtype)
                    )

            # Calculate image tokens
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = ([self.image_token_id] * num_queries_base) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num)) * (
                    num_queries * height_crop_num)

        else:  # crop_mode = False
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            # Match modeling_deepseekocr2.py infer(crop_mode=False):
            # adaptively resize the image to a square whose side equals the long edge,
            # capped by image_size (768 in current training config).
            orig_w, orig_h = image.size
            target_size = min(max(orig_w, orig_h), self.image_size)
            resized_image = image.resize((target_size, target_size), Image.LANCZOS)
            images_list.append(self.image_transform(resized_image).to(self.dtype))

            num_queries = math.ceil((target_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Process a single conversation into model inputs.
        """

        # --- 1. Setup ---
        images = []
        for message in messages:
            if "images" in message and message["images"]:
                for img_data in message["images"]:
                    if img_data is not None:
                        pil_image = self.deserialize_image(img_data)
                        images.append(pil_image)

        if not images:
            raise ValueError("No images found in sample. Please ensure all samples contain images.")

        tokenized_str = []
        images_seq_mask = []
        images_list, images_crop_list, images_spatial_crop = [], [], []

        prompt_token_count = -1 # Index to start training
        assistant_started = False
        image_idx = 0

        # Add BOS token at the very beginning
        tokenized_str.append(self.bos_id)
        images_seq_mask.append(False)

        for message in messages:
            role = message["role"]
            content = message["content"]

            # Check if this is the assistant's turn
            if role == "<|Assistant|>":
                if not assistant_started:
                    # This is the split point. All tokens added *so far*
                    # are part of the prompt.
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True

                # Append the EOS token string to the *end* of assistant content
                content = f"{content.strip()} {self.tokenizer.eos_token}"

            # Split this message's content by the image token
            text_splits = content.split('<image>')

            for i, text_sep in enumerate(text_splits):
                # Tokenize the text part
                tokenized_sep = text_encode(self.tokenizer, text_sep, bos = False, eos = False)
                tokenized_str.extend(tokenized_sep)
                images_seq_mask.extend([False] * len(tokenized_sep))

                # If this text is followed by an <image> tag
                if i < len(text_splits) - 1:
                    if image_idx >= len(images):
                        raise ValueError(
                            f"Data mismatch: Found '<image>' token but no corresponding image."
                        )

                    # Process the image
                    image = images[image_idx]
                    img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial_crop)

                    # Add image placeholder tokens
                    tokenized_str.extend(tok_img)
                    images_seq_mask.extend([True] * len(tok_img))

                    image_idx += 1 # Move to the next image

        # --- 3. Validation and Final Prep ---
        if image_idx != len(images):
            raise ValueError(
                f"Data mismatch: Found {len(images)} images but only {image_idx} '<image>' tokens were used."
            )

        # If we never found an assistant message, we're in a weird state
        # (e.g., user-only prompt). We mask everything.
        if not assistant_started:
            print("Warning: No assistant message found in sample. Masking all tokens.")
            prompt_token_count = len(tokenized_str)

        # Prepare image tensors
        images_ori = torch.stack(images_list, dim = 0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype = torch.long)

        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim = 0)
        else:
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype = self.dtype)

        return {
            "input_ids": torch.tensor(tokenized_str, dtype = torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype = torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
            "prompt_token_count": prompt_token_count, # This is now accurate
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples"""
        batch_data = []

        # Process each sample
        for feature in features:
            try:
                processed = self.process_single_sample(feature['messages'])
                batch_data.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        # Extract lists
        input_ids_list = [item['input_ids'] for item in batch_data]
        images_seq_mask_list = [item['images_seq_mask'] for item in batch_data]
        prompt_token_counts = [item['prompt_token_count'] for item in batch_data]

        # Pad sequences
        input_ids = pad_sequence(input_ids_list, batch_first = True, padding_value = self.tokenizer.pad_token_id)
        images_seq_mask = pad_sequence(images_seq_mask_list, batch_first = True, padding_value = False)

        # Create labels
        labels = input_ids.clone()

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask image tokens (model shouldn't predict these)
        labels[images_seq_mask] = -100

        # Mask user prompt tokens when train_on_responses_only = True (only train on assistant responses)
        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if prompt_count > 0:
                    labels[idx, :prompt_count] = -100

        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Prepare images batch (list of tuples)
        images_batch = []
        for item in batch_data:
            images_batch.append((item['images_crop'], item['images_ori']))

        # Stack spatial crop info
        images_spatial_crop = torch.cat([item['images_spatial_crop'] for item in batch_data], dim = 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from unsloth import is_bf16_supported

# IMPORTANT: Set trainable parameters BEFORE creating Trainer
# This ensures all DDP processes have the same configuration
model.train()

for param in model.parameters():
    param.requires_grad = True


def print_trainable_parameters(model):
    total = 0
    trainable = 0
    trainable_names = []
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            if len(trainable_names) < 20:
                trainable_names.append(name)
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    print("trainable parameter examples:")
    for name in trainable_names:
        print(f"  {name}")

print_trainable_parameters(model)

# Custom callback to reduce learning rate when loss plateaus
from transformers import TrainerCallback



class ReduceLROnPlateauCallback(TrainerCallback):
    def __init__(self, patience=1, factor=0.5, min_lr=1e-7, verbose=True):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = float('inf')
        self.wait = 0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        current_loss = metrics.get('eval_loss', float('inf'))
        
        # Check if loss improved
        if current_loss < self.best_loss - 1e-6:  # Small threshold for numerical stability
            self.best_loss = current_loss
            self.wait = 0
            if self.verbose and state.is_local_process_zero:
                print(f"\n[LR Scheduler] Loss improved to {current_loss:.6f}, resetting patience counter")
        else:
            self.wait += 1
            if self.verbose and state.is_local_process_zero:
                print(f"\n[LR Scheduler] No improvement for {self.wait} epoch(s)")
            
            # Reduce LR if no improvement for 'patience' epochs
            if self.wait >= self.patience:
                old_lr = state.optimizer.param_groups[0]['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                
                if new_lr != old_lr:
                    for param_group in state.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    if self.verbose and state.is_local_process_zero:
                        print(f"\n[LR Scheduler] Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
                    
                    self.wait = 0  # Reset counter after reducing LR
                else:
                    if self.verbose and state.is_local_process_zero:
                        print(f"\n[LR Scheduler] Already at minimum LR ({self.min_lr:.2e})")


class LossBreakdownTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        wrapped_model = model.module if hasattr(model, "module") else model
        if hasattr(wrapped_model, "model") and hasattr(wrapped_model.model, "set_curriculum_progress"):
            epoch_progress = float(self.state.epoch) if self.state.epoch is not None else 0.0
            wrapped_model.model.set_curriculum_progress(epoch_progress)
            if hasattr(wrapped_model.model, "project_codebook_to_sphere_"):
                wrapped_model.model.project_codebook_to_sphere_()

        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        self._last_loss_breakdown = getattr(wrapped_model, "_last_loss_breakdown", None)

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs, start_time=None):
        if hasattr(self, "_last_loss_breakdown") and "loss" in logs:
            logs = dict(logs)
            logs.update(self._last_loss_breakdown)
        return super().log(logs, start_time)


class CodebookSphereProjectionCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        wrapped_model = model.module if hasattr(model, "module") else model
        if hasattr(wrapped_model, "model") and hasattr(wrapped_model.model, "project_codebook_to_sphere_"):
            wrapped_model.model.project_codebook_to_sphere_()

data_collator = DeepSeekOCR2DataCollator(
    tokenizer = tokenizer,
    model = model,
    image_size = 768,
    base_size = 1024,
    crop_mode = False,
    train_on_responses_only = True,
)
trainer = LossBreakdownTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = data_collator, # Must use!
    train_dataset = converted_train_dataset,  # Use train split
    eval_dataset = converted_test_dataset,    # Use test split for evaluation
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 16,
        warmup_ratio = 0.03,  # ~3% of total steps; enough for a long run without over-delaying peak LR
        # max_steps = 60,  # Removed - using early stopping instead
        num_train_epochs = 3,  # 3 epochs should be enough for codebook training
        learning_rate = 1e-4,
        logging_steps = 1,
        optim = "adamw_bnb_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        max_grad_norm = 1.0,  # Add gradient clipping for stability
        seed = 3407,
        fp16 = not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16 = is_bf16_supported(),  # Use bf16 if supported
        output_dir = BALANCED_OUTPUT_DIR,
        report_to = "none",     # For Weights and Biases
        dataloader_num_workers = 4,
        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        # Distributed training settings for 8 GPUs
        ddp_find_unused_parameters = True,   # MoE decoder does not use every expert on every rank/step
        ddp_backend = "nccl",
        local_rank = -1,  # Will be set automatically by torchrun
        # Evaluation and early stopping settings
        eval_strategy = "steps",             # Evaluate every N steps
        eval_steps = 2000,                   # Evaluate every 2000 steps
        save_strategy = "steps",             # Save checkpoint every N steps
        save_steps = 2000,                   # Save every 2000 steps
        load_best_model_at_end = True,       # Load best model at the end
        metric_for_best_model = "eval_loss", # Use validation loss as metric
        greater_is_better = False,           # Lower loss is better
        save_total_limit = 3,                # Keep only 3 best checkpoints
    ),
    callbacks = [
        CodebookSphereProjectionCallback(),
        EarlyStoppingCallback(
            early_stopping_patience = 3,  # Stop if no improvement for 3 epochs
            early_stopping_threshold = 0.0  # Any improvement counts
        ),
    ]
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


''' # inference
prompt = "<image>\nFree OCR. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 768, image_size = 768, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 768, crop_mode = True

res = model.infer(tokenizer, prompt = prompt, image_file = image_file,
    output_path = output_path,
    image_size = 768,
    base_size = 1024,
    crop_mode = True,
    save_results = True,
    test_compress = False)
'''


# Save the trained model.
model.save_pretrained(FINAL_OUTPUT_DIR)
tokenizer.save_pretrained(FINAL_OUTPUT_DIR)
