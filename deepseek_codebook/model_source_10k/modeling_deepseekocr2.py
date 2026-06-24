import os
import math
import re
from tqdm import tqdm
from abc import ABC
from typing import List, Optional, Tuple, Union

from addict import Dict
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torchvision import transforms

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import DeepseekV2Model, DeepseekV2ForCausalLM
from transformers import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2MLP,
    DeepseekV2MoE,
    DeepseekV2RMSNorm,
    DeepseekV2DecoderLayer,
)
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers import TextStreamer
from .deepencoderv2 import build_sam_vit_b, build_qwen2_decoder_as_encoder, MlpProjector
from .conversation import get_conv_template
import torch.nn.functional as F

torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def load_image(image_path):

    try:
        image = Image.open(image_path)
        
        corrected_image = ImageOps.exif_transpose(image)
        
        return corrected_image
        
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    # pattern1 = r'<\|ref\|>.*?<\|/ref\|>\n'
    # new_text1 = re.sub(pattern1, '', text, flags=re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):

    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, ouput_path):

    image_width, image_height = image.size
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    # try:
    # except IOError:
    #     try:
    #         font = ImageFont.truetype("DejaVuSans.ttf", 20) 
    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{ouput_path}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path):

    result_image = draw_bounding_boxes(image, ref_texts, output_path)
    
    return result_image





def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=6, image_size=768, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # print(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio



def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform



def format_messages(
        conversations: List[Dict[str, str]],
        sft_format: str = "deepseek",
        system_prompt: str = "",
):
    """
    Applies the SFT template to conversation.

    Args:
        conversations (List[Dict]): A List of messages.
        sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
        system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

    Returns:
        sft_prompt (str): The formatted text.
    """

    conv = get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    sft_prompt = conv.get_prompt().strip()

    return sft_prompt


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]

    return t

def load_pil_images(conversations: List[Dict[str, str]]) -> List[Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            # print('----------------')
            # print(image_path)
            # print('----------------')
            # exit()
            
            # pil_img = Image.open(image_path)
            pil_img = load_image(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images


class BaseTransform(ABC):

    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


class BasicImageTransform(BaseTransform):
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        x = self.transform(x)
        return x

class NoEOSTextStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):

        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")

def decoder_layer_init(self, config: DeepseekV2Config, layer_idx: int):
    nn.Module.__init__(self)
    self.hidden_size = config.hidden_size

    if config.use_mla:
        self.self_attn = DeepseekV2Attention(config=config, layer_idx=layer_idx)
    else:
        config.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = LlamaAttention(config, layer_idx)
    self.mlp = DeepseekV2MoE(config) if layer_idx >= config.first_k_dense_replace else DeepseekV2MLP(config)

    self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


DeepseekV2DecoderLayer.__init__ = decoder_layer_init

class DeepseekOCR2Config(DeepseekV2Config):
    model_type = "DeepseekOCR2"

class DeepseekOCR2Model(DeepseekV2Model):
    config_class = DeepseekOCR2Config

    def __init__(self, config: DeepseekV2Config):
        super(DeepseekOCR2Model, self).__init__(config)

        self.sam_model = build_sam_vit_b()
        self.qwen2_model = build_qwen2_decoder_as_encoder()
        # self.conv_2 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=2, stride=2)
        n_embed = 1280
        codebook_size = 10000
        self.codebook_placeholder_slots = 1
        self.projector =  MlpProjector(Dict(projector_type="linear", input_dim=896, n_embed=n_embed))
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        # self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        codebook = torch.randn(codebook_size, 1280, dtype=torch.float32) * embed_std
        self.codebook = nn.Parameter(codebook)
        self.codebook_assign_topk = 128
        self.codebook_assign_capacity = 1
        self.codebook_usage_penalty_alpha = 0.0
        self.codebook_idle_reward_alpha = 0.0
        self.codebook_soft_update_topk = 128
        self.codebook_soft_update_temperature = 1.0
        self.codebook_soft_update_chunk = 32
        self.codebook_st_mode = "strict"
        self.codebook_enable_refresh = False
        self.codebook_usage_decay = 0.995
        self.codebook_dead_steps = 1000
        self.codebook_refresh_interval = 100
        self.codebook_refresh_max_codes = 512
        self.codebook_refresh_noise_scale = 0.01
        self.codebook_feature_bank_sample_size = 128
        self.register_buffer("codebook_usage_ema", torch.zeros(codebook_size, dtype=torch.float32))
        self.register_buffer("codebook_idle_steps", torch.zeros(codebook_size, dtype=torch.long))
        self.register_buffer("codebook_recent_bank", torch.zeros(32768, n_embed, dtype=torch.float32))
        self.register_buffer("codebook_recent_bank_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("codebook_recent_bank_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("codebook_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("codebook_last_unique_codes", torch.zeros((), dtype=torch.long))
        self.register_buffer("codebook_last_top1_fraction", torch.zeros((), dtype=torch.float32))
        self.register_buffer("codebook_last_top5_fraction", torch.zeros((), dtype=torch.float32))
        self.register_buffer("codebook_last_effective_codes", torch.zeros((), dtype=torch.float32))
        self.register_buffer("codebook_last_refresh_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("codebook_target_norm", torch.ones((), dtype=torch.float32))
        self.register_buffer("codebook_noise_std", torch.zeros((), dtype=torch.float32))
        self.register_buffer("codebook_vmf_kappa", torch.ones((), dtype=torch.float32))
        self.register_buffer("codebook_last_top1_distance", torch.zeros((), dtype=torch.float32))
        self.register_buffer("codebook_last_top10_distance", torch.zeros((), dtype=torch.float32))
        self.register_buffer("codebook_last_top128_distance", torch.zeros((), dtype=torch.float32))
        self.register_buffer("codebook_last_top128_angle", torch.zeros((), dtype=torch.float32))
        self.register_buffer("curriculum_progress", torch.zeros((), dtype=torch.float32))
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def _codebook_main_slice(self):
        return slice(self.codebook_placeholder_slots, self.codebook.shape[0])

    def _codebook_world_size(self):
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    @torch.no_grad()
    def _gather_feature_samples(self, x_fp32: torch.Tensor) -> Optional[torch.Tensor]:
        if x_fp32.numel() == 0:
            return None

        sample_cap = min(self.codebook_feature_bank_sample_size, x_fp32.shape[0])
        if x_fp32.shape[0] == sample_cap:
            sample = x_fp32.detach()
        else:
            sample_idx = torch.linspace(
                0, x_fp32.shape[0] - 1, steps=sample_cap, device=x_fp32.device
            ).round().long()
            sample = x_fp32.index_select(0, sample_idx).detach()

        padded = torch.zeros(
            self.codebook_feature_bank_sample_size,
            x_fp32.shape[-1],
            device=x_fp32.device,
            dtype=torch.float32,
        )
        padded[:sample_cap] = sample.to(dtype=torch.float32)

        if self._codebook_world_size() == 1:
            return padded[:sample_cap]

        local_count = torch.tensor([sample_cap], device=x_fp32.device, dtype=torch.long)
        gathered_counts = [torch.zeros_like(local_count) for _ in range(self._codebook_world_size())]
        dist.all_gather(gathered_counts, local_count)

        gathered_samples = [torch.zeros_like(padded) for _ in range(self._codebook_world_size())]
        dist.all_gather(gathered_samples, padded)

        merged = []
        for gathered, count in zip(gathered_samples, gathered_counts):
            count_int = int(count.item())
            if count_int > 0:
                merged.append(gathered[:count_int])
        if not merged:
            return None
        return torch.cat(merged, dim=0)

    @torch.no_grad()
    def _append_recent_bank(self, features: Optional[torch.Tensor]) -> None:
        if features is None or features.numel() == 0:
            return

        bank_size = int(self.codebook_recent_bank.shape[0])
        features = features.to(device=self.codebook_recent_bank.device, dtype=self.codebook_recent_bank.dtype)
        if features.shape[0] > bank_size:
            features = features[-bank_size:]

        num_new = int(features.shape[0])
        ptr = int(self.codebook_recent_bank_ptr.item())
        first = min(bank_size - ptr, num_new)
        second = num_new - first

        self.codebook_recent_bank[ptr:ptr + first] = features[:first]
        if second > 0:
            self.codebook_recent_bank[:second] = features[first:first + second]

        self.codebook_recent_bank_ptr.fill_((ptr + num_new) % bank_size)
        current_count = int(self.codebook_recent_bank_count.item())
        self.codebook_recent_bank_count.fill_(min(bank_size, current_count + num_new))

    @torch.no_grad()
    def _maybe_refresh_dead_codes(self) -> None:
        self.codebook_last_refresh_count.zero_()
        if not self.codebook_enable_refresh:
            return
        step = int(self.codebook_step.item())
        bank_count = int(self.codebook_recent_bank_count.item())
        if step < self.codebook_dead_steps:
            return
        if step % self.codebook_refresh_interval != 0:
            return
        if bank_count == 0:
            return

        main_slice = self._codebook_main_slice()
        idle_steps_main = self.codebook_idle_steps[main_slice]
        dead_indices = torch.nonzero(
            idle_steps_main >= self.codebook_dead_steps, as_tuple=False
        ).flatten()
        if dead_indices.numel() == 0:
            return
        dead_indices = dead_indices + self.codebook_placeholder_slots

        refresh_count = min(
            int(dead_indices.numel()),
            self.codebook_refresh_max_codes,
            bank_count,
        )
        if refresh_count <= 0:
            return

        idle_scores = self.codebook_idle_steps.index_select(0, dead_indices)
        chosen_dead = dead_indices[torch.topk(idle_scores, k=refresh_count).indices]

        bank = self.codebook_recent_bank[:bank_count]
        generator = torch.Generator(device=bank.device)
        generator.manual_seed(step)
        sample_ids = torch.randint(
            0, bank_count, (refresh_count,), generator=generator, device=bank.device
        )
        replacements = bank.index_select(0, sample_ids).clone()
        feature_std = bank.std(dim=0, unbiased=False).clamp_min(1e-6)
        noise = torch.randn(
            replacements.shape,
            generator=generator,
            device=bank.device,
            dtype=bank.dtype,
        )
        replacements = replacements + self.codebook_refresh_noise_scale * noise * feature_std

        self.codebook.data.index_copy_(0, chosen_dead, replacements.to(self.codebook.dtype))
        self.codebook_idle_steps.index_fill_(0, chosen_dead, 0)
        usage_fill = self.codebook_usage_ema[main_slice].mean()
        self.codebook_usage_ema.index_fill_(
            0,
            chosen_dead,
            float(usage_fill.item()),
        )
        self.codebook_last_refresh_count.fill_(refresh_count)

    @torch.no_grad()
    def _update_codebook_state(self, nearest_idx: torch.Tensor, x_fp32: torch.Tensor) -> None:
        counts = torch.bincount(
            nearest_idx, minlength=self.codebook.shape[0]
        ).to(device=self.codebook.device, dtype=torch.float32)

        if self._codebook_world_size() > 1:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        main_slice = self._codebook_main_slice()
        counts_main = counts[main_slice]
        total = counts_main.sum().clamp_min(1.0)
        usage = counts_main / total
        self.codebook_usage_ema[main_slice].mul_(self.codebook_usage_decay).add_(
            usage * (1.0 - self.codebook_usage_decay)
        )

        selected = counts_main > 0
        self.codebook_idle_steps[main_slice].add_(1)
        self.codebook_idle_steps[main_slice].masked_fill_(selected, 0)

        self.codebook_last_unique_codes.fill_(int(selected.sum().item()))
        self.codebook_last_top1_fraction.fill_(float((counts_main.max() / total).item()))
        topk_count = min(5, counts_main.numel())
        self.codebook_last_top5_fraction.fill_(float((counts_main.topk(topk_count).values.sum() / total).item()))
        usage_prob = self.codebook_usage_ema[main_slice] / self.codebook_usage_ema[main_slice].sum().clamp_min(1e-12)
        entropy = -(usage_prob * usage_prob.clamp_min(1e-12).log()).sum()
        self.codebook_last_effective_codes.fill_(float(torch.exp(entropy).item()))

        gathered_features = self._gather_feature_samples(x_fp32)
        self._append_recent_bank(gathered_features)

        self.codebook_step.add_(1)
        self._maybe_refresh_dead_codes()

    def set_curriculum_progress(self, progress: float) -> None:
        self.curriculum_progress.fill_(float(progress))

    def project_codebook_to_sphere_(self) -> None:
        target_norm = float(self.codebook_target_norm.item())
        if target_norm <= 0:
            return
        main_slice = self._codebook_main_slice()
        with torch.no_grad():
            main_codes = self.codebook.data[main_slice]
            main_norm = torch.norm(main_codes, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
            self.codebook.data[main_slice] = main_codes / main_norm * target_norm

    def _normalize_to_target_norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        target_norm = float(self.codebook_target_norm.item())
        if target_norm <= 0:
            return x
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(eps)
        return x / x_norm * target_norm

    def _chord_distance_to_angle(self, d: torch.Tensor) -> torch.Tensor:
        radius = float(self.codebook_target_norm.item())
        if radius <= 0:
            return torch.zeros_like(d)
        scaled = (d / (2.0 * radius)).clamp(min=0.0, max=1.0 - 1e-7)
        return 2.0 * torch.asin(scaled)

    def _sample_uniform_unit_vectors(self, num: int, dim: int, device, dtype) -> torch.Tensor:
        vec = torch.randn(num, dim, device=device, dtype=dtype)
        return F.normalize(vec, dim=-1)

    def _householder_rotate(self, x: torch.Tensor, mu_unit: torch.Tensor) -> torch.Tensor:
        e1 = torch.zeros_like(mu_unit)
        e1[:, 0] = 1.0
        u = e1 - mu_unit
        u_norm = torch.norm(u, p=2, dim=-1, keepdim=True)
        safe_u = u / u_norm.clamp_min(1e-6)
        rotated = x - 2.0 * (x * safe_u).sum(dim=-1, keepdim=True) * safe_u
        use_identity = (u_norm.squeeze(-1) < 1e-6).unsqueeze(-1)
        return torch.where(use_identity, x, rotated)

    def _sample_vmf(self, mu: torch.Tensor) -> torch.Tensor:
        kappa = float(self.codebook_vmf_kappa.item())
        if (not self.training) or kappa <= 0:
            return mu

        num_samples, dim = mu.shape
        mu_unit = F.normalize(mu, dim=-1)
        device = mu.device
        dtype = mu.dtype

        c = math.sqrt(4.0 * (kappa ** 2) + (dim - 1) ** 2)
        b_true = (-2.0 * kappa + c) / (dim - 1)
        b_app = (dim - 1) / (4.0 * kappa)
        b = min(b_app, b_true)
        a = (dim - 1 + 2.0 * kappa + c) / 4.0
        d = (4.0 * a * b) / (1.0 + b) - (dim - 1) * math.log(dim - 1)

        beta = torch.distributions.Beta(
            torch.tensor((dim - 1) / 2.0, device=device, dtype=dtype),
            torch.tensor((dim - 1) / 2.0, device=device, dtype=dtype),
        )

        w = torch.empty(num_samples, device=device, dtype=dtype)
        remaining = torch.arange(num_samples, device=device)

        while remaining.numel() > 0:
            e = beta.sample((remaining.numel(),))
            u = torch.rand(remaining.numel(), device=device, dtype=dtype)
            w_candidate = (1.0 - (1.0 + b) * e) / (1.0 - (1.0 - b) * e)
            t = (2.0 * a * b) / (1.0 - (1.0 - b) * e)
            accept = ((dim - 1) * torch.log(t) - t + d) >= torch.log(u)
            if accept.any():
                accepted_idx = remaining[accept]
                w[accepted_idx] = w_candidate[accept]
            remaining = remaining[~accept]

        v = self._sample_uniform_unit_vectors(num_samples, dim - 1, device, dtype)
        orth = torch.sqrt((1.0 - w.pow(2)).clamp_min(0.0)).unsqueeze(-1) * v
        base = torch.cat([w.unsqueeze(-1), orth], dim=-1)
        samples = self._householder_rotate(base, mu_unit)
        return samples * float(self.codebook_target_norm.item())

    def _get_codebook_lookup(self) -> torch.Tensor:
        self.project_codebook_to_sphere_()
        codebook_fp32 = self.codebook.float()
        if self.codebook_placeholder_slots == 0:
            return self._normalize_to_target_norm(codebook_fp32)

        placeholder = codebook_fp32[:self.codebook_placeholder_slots]
        main_codes = self._normalize_to_target_norm(codebook_fp32[self.codebook_placeholder_slots:])
        return torch.cat([placeholder, main_codes], dim=0)

    def _sample_variational_x(self, x_mu: torch.Tensor) -> torch.Tensor:
        return self._sample_vmf(x_mu)

    def _weighted_codebook_average(
        self,
        codebook_lookup: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        main_offset: int,
    ) -> torch.Tensor:
        topk = topk_indices.shape[-1]
        chunk = min(self.codebook_soft_update_chunk, topk)
        mixed = torch.zeros(
            topk_indices.shape[0],
            codebook_lookup.shape[-1],
            device=topk_indices.device,
            dtype=torch.float32,
        )

        for start in range(0, topk, chunk):
            end = min(start + chunk, topk)
            flat_ids = (topk_indices[:, start:end] + main_offset).reshape(-1)
            candidate_codes = codebook_lookup.index_select(0, flat_ids).view(
                topk_indices.shape[0], end - start, -1
            )
            mixed = mixed + (
                topk_weights[:, start:end].to(candidate_codes.dtype).unsqueeze(-1)
                * candidate_codes
            ).sum(dim=1)

        return mixed

    @torch.no_grad()
    def _assign_codes_with_capacity(
        self,
        topk_indices: torch.Tensor,
        topk_values: torch.Tensor,
        main_offset: int,
    ) -> torch.Tensor:
        num_tokens, topk = topk_indices.shape
        if num_tokens == 0:
            return torch.zeros(0, device=topk_indices.device, dtype=torch.long)

        assigned = torch.full(
            (num_tokens,),
            -1,
            device=topk_indices.device,
            dtype=torch.long,
        )

        if topk == 1 or self.codebook_assign_capacity <= 0:
            return topk_indices[:, 0] + main_offset

        candidate_scores = topk_values.float().clone()
        main_slice = self._codebook_main_slice()
        usage_alpha = float(self.codebook_usage_penalty_alpha)
        if usage_alpha > 0.0:
            usage = self.codebook_usage_ema[main_slice]
            usage_mean = usage.mean()
            usage_std = usage.std(unbiased=False).clamp_min(1e-6)
            usage_penalty = ((usage - usage_mean) / usage_std).clamp_min(0.0)
            gathered_usage = usage_penalty.index_select(0, topk_indices.reshape(-1)).view_as(candidate_scores)
            candidate_scores = candidate_scores + usage_alpha * gathered_usage

        idle_alpha = float(self.codebook_idle_reward_alpha)
        if idle_alpha > 0.0:
            idle = self.codebook_idle_steps[main_slice].float()
            idle_mean = idle.mean()
            idle_std = idle.std(unbiased=False).clamp_min(1.0)
            idle_reward = ((idle - idle_mean) / idle_std).clamp_min(0.0)
            gathered_idle = idle_reward.index_select(0, topk_indices.reshape(-1)).view_as(candidate_scores)
            candidate_scores = candidate_scores - idle_alpha * gathered_idle

        candidate_order = torch.argsort(candidate_scores, dim=-1)
        sorted_indices = topk_indices.gather(1, candidate_order)
        sorted_scores = candidate_scores.gather(1, candidate_order)

        usage = torch.zeros(
            self.codebook.shape[0] - main_offset,
            device=topk_indices.device,
            dtype=torch.long,
        )
        capacity = int(self.codebook_assign_capacity)

        if topk > 1:
            priority = sorted_scores[:, 1] - sorted_scores[:, 0]
            order = torch.argsort(priority, descending=True)
        else:
            order = torch.arange(num_tokens, device=topk_indices.device)

        for token_idx in order.tolist():
            candidates = sorted_indices[token_idx]
            chosen = -1
            for candidate in candidates.tolist():
                if usage[candidate] < capacity:
                    chosen = candidate
                    usage[candidate] += 1
                    break
            if chosen < 0:
                chosen = int(candidates[0].item())
                usage[chosen] += 1
            assigned[token_idx] = chosen + main_offset

        return assigned

    def encode_image_feature_rows(self, images, images_spatial_crop):
        batch_feature_rows = []

        sam_model = getattr(self, 'sam_model', None)
        qwen2_model = getattr(self, 'qwen2_model', None)
        if sam_model is None or images is None:
            return batch_feature_rows

        for image, crop_shape in zip(images, images_spatial_crop):
            patches = image[0]
            image_ori = image[1]

            if torch.sum(patches).item() != 0:
                local_features = self.projector(qwen2_model(sam_model(patches)))
                global_features = self.projector(qwen2_model(sam_model(image_ori)))
                global_features = global_features.view(-1, global_features.shape[-1])
                local_features = local_features.view(-1, local_features.shape[-1])
                global_local_features = torch.cat(
                    [local_features, global_features, self.view_seperator[None, :]],
                    dim=0,
                )
            else:
                global_features = self.projector(qwen2_model(sam_model(image_ori)))
                global_features = global_features.view(-1, global_features.shape[-1])
                global_local_features = torch.cat(
                    [global_features, self.view_seperator[None, :]],
                    dim=0,
                )

            batch_feature_rows.append(global_local_features)

        return batch_feature_rows

    def prepare_codebook_feature_rows(self, x_list):
        if len(x_list) == 0:
            zero = torch.zeros((), device=self.codebook.device, dtype=torch.float32)
            return [], [], zero, zero, []

        x_fp32_list = [x.float() for x in x_list]
        lengths = [x.shape[0] for x in x_fp32_list]
        all_x_raw = torch.cat(x_fp32_list, dim=0)

        main_masks = []
        for x_fp32 in x_fp32_list:
            mask = torch.ones(x_fp32.shape[0], device=x_fp32.device, dtype=torch.bool)
            mask[-1] = False
            main_masks.append(mask)
        main_mask = torch.cat(main_masks, dim=0)
        placeholder_mask = ~main_mask

        x_direct_all = all_x_raw.clone()
        nearest_idx_all = torch.empty(
            all_x_raw.shape[0], device=all_x_raw.device, dtype=torch.long
        )
        main_offset = self.codebook_placeholder_slots
        codebook_lookup = self._get_codebook_lookup().to(device=all_x_raw.device)
        mse_encoder_loss = torch.zeros((), device=all_x_raw.device, dtype=torch.float32)
        mse_codebook_loss = torch.zeros((), device=all_x_raw.device, dtype=torch.float32)

        if main_mask.any():
            raw_main_x = all_x_raw[main_mask]
            x_mu = self._normalize_to_target_norm(raw_main_x)
            x_sample = self._sample_variational_x(x_mu)
            x_direct_all[main_mask] = x_mu

            main_codebook = codebook_lookup[main_offset:]
            main_dist = torch.cdist(
                x_sample.unsqueeze(0),
                main_codebook.unsqueeze(0),
                p=2,
            ).squeeze(0)
            hard_topk = min(self.codebook_assign_topk, main_dist.shape[-1])
            hard_topk_values, hard_topk_indices = main_dist.topk(k=hard_topk, largest=False, dim=-1)
            nearest_idx_all[main_mask] = self._assign_codes_with_capacity(
                topk_indices=hard_topk_indices,
                topk_values=hard_topk_values,
                main_offset=main_offset,
            )

            soft_topk = min(self.codebook_soft_update_topk, main_dist.shape[-1])
            soft_topk_values, soft_topk_indices = main_dist.topk(k=soft_topk, largest=False, dim=-1)
            shifted_topk = soft_topk_values - soft_topk_values.min(dim=-1, keepdim=True).values
            scale = shifted_topk.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
            soft_topk_weights = torch.softmax(
                -shifted_topk / (scale * max(self.codebook_soft_update_temperature, 1e-6)),
                dim=-1,
            )

            with torch.no_grad():
                self.codebook_last_top1_distance.fill_(float(soft_topk_values[:, 0].mean().item()))
                top10_idx = min(10, soft_topk_values.shape[-1]) - 1
                top128_idx = min(128, soft_topk_values.shape[-1]) - 1
                self.codebook_last_top10_distance.fill_(float(soft_topk_values[:, top10_idx].mean().item()))
                self.codebook_last_top128_distance.fill_(float(soft_topk_values[:, top128_idx].mean().item()))
                self.codebook_last_top128_angle.fill_(
                    float(self._chord_distance_to_angle(soft_topk_values[:, top128_idx]).mean().item())
                )

            if self.training:
                self._update_codebook_state(nearest_idx_all[main_mask], x_sample.detach())

            z_q = codebook_lookup.index_select(0, nearest_idx_all[main_mask])
            if self.codebook_st_mode == "strict":
                z_quant_main = x_sample + (z_q - x_sample).detach()
            elif self.codebook_st_mode == "dual":
                z_quant_main = x_sample + z_q - x_sample.detach()
            else:
                z_quant_main = z_q
            z_soft = self._weighted_codebook_average(
                codebook_lookup,
                soft_topk_indices,
                soft_topk_weights,
                main_offset,
            )
            mse_encoder_loss = ((z_q.detach() - x_mu) ** 2).sum(dim=-1).mean()
            mse_codebook_loss = ((z_soft - x_mu.detach()) ** 2).sum(dim=-1).mean()
        else:
            z_quant_main = torch.zeros(
                0, all_x_raw.shape[-1], device=all_x_raw.device, dtype=torch.float32
            )

        if placeholder_mask.any():
            nearest_idx_all[placeholder_mask] = 0

        z_quant_all = x_direct_all.clone()
        if main_mask.any():
            z_quant_all[main_mask] = z_quant_main
        if placeholder_mask.any():
            z_quant_all[placeholder_mask] = codebook_lookup[0].to(z_quant_all.dtype)

        if (
            os.environ.get("CODEBOOK_DEBUG_PRINT", "0") == "1"
            and os.environ.get("LOCAL_RANK", "0") == "0"
            and main_mask.any()
        ):
            main_x_direct = x_direct_all[main_mask]
            main_z_hard = codebook_lookup.index_select(0, nearest_idx_all[main_mask])
            per_token_raw_mse = ((main_z_hard - main_x_direct) ** 2).mean(dim=-1).detach().cpu().tolist()
            per_token_l2 = torch.norm(main_z_hard - main_x_direct, p=2, dim=-1).detach().cpu().tolist()
            constrained_idx = nearest_idx_all[main_mask].detach().cpu().tolist()
            for token_idx, (codebook_token, raw_mse, l2_dist) in enumerate(
                zip(constrained_idx, per_token_raw_mse, per_token_l2)
            ):
                print(
                    f"[CODEBOOK] token_idx={token_idx} "
                    f"nearest_idx={codebook_token} "
                    f"raw_mse={raw_mse:.6f} "
                    f"l2={l2_dist:.6f}"
                )

        x_direct_list = []
        z_quant_list = []
        nearest_idx_list = []
        start = 0
        for length, original in zip(lengths, x_list):
            end = start + length
            x_direct_list.append(x_direct_all[start:end].to(dtype=original.dtype))
            z_quant_list.append(z_quant_all[start:end].to(dtype=original.dtype))
            nearest_idx_list.append(nearest_idx_all[start:end])
            start = end

        return x_direct_list, z_quant_list, mse_encoder_loss, mse_codebook_loss, nearest_idx_list

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        use_codebook: bool = True,
        prepared_feature_rows: Optional[List[torch.FloatTensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        total_mse_encoder_loss = None
        total_mse_codebook_loss = None

        if inputs_embeds is None:
            # inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.clone()

        batch_feature_rows = prepared_feature_rows
        if (
            batch_feature_rows is None
            and images is not None
            and images_seq_mask is not None
            and (input_ids.shape[1] != 1 or self.training)
            and torch.sum(images[0][1]).item() != 0
        ):
            raw_feature_rows = self.encode_image_feature_rows(images, images_spatial_crop)
            if raw_feature_rows:
                direct_rows, quant_rows, mse_encoder_loss, mse_codebook_loss, _ = (
                    self.prepare_codebook_feature_rows(raw_feature_rows)
                )
                batch_feature_rows = quant_rows if use_codebook else direct_rows
                total_mse_encoder_loss = mse_encoder_loss
                total_mse_codebook_loss = mse_codebook_loss

        if batch_feature_rows is not None:
            for idx, image_features in enumerate(batch_feature_rows):
                image_features = image_features.to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                mask = images_seq_mask[idx].unsqueeze(-1).to(inputs_embeds.device)
                updated_row = inputs_embeds[idx].masked_scatter(mask, image_features)
                inputs_embeds[idx] = updated_row

        if prepared_feature_rows is None:
            self._last_mse_encoder_loss = total_mse_encoder_loss
            self._last_mse_codebook_loss = total_mse_codebook_loss


        return super(DeepseekOCR2Model, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    

class DeepseekOCR2ForCausalLM(DeepseekV2ForCausalLM):

    config_class = DeepseekOCR2Config
    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(DeepseekV2ForCausalLM, self).__init__(config)
        self.model = DeepseekOCR2Model(config)

        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss = None
        self._last_loss_breakdown = None
        if labels is not None:
            curriculum_progress = float(self.model.curriculum_progress.item())
            lm_x_weight = max(0.0, 1.0 - min(curriculum_progress, 1.0))
            lm_z_weight = 1.0
            mse_encoder_weight = 0.05
            mse_codebook_weight = 0.1
            loss_fct = CrossEntropyLoss()
            shift_labels = labels[..., 1:].contiguous().view(-1)
            lm_loss_x = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            lm_loss_z = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            mse_encoder_loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            mse_codebook_loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)

            visual_feature_rows = None
            direct_rows = None
            quant_rows = None
            if images is not None and images_seq_mask is not None and torch.sum(images[0][1]).item() != 0:
                visual_feature_rows = self.model.encode_image_feature_rows(images, images_spatial_crop)
                if visual_feature_rows:
                    direct_rows, quant_rows, mse_encoder_loss, mse_codebook_loss, _ = (
                        self.model.prepare_codebook_feature_rows(visual_feature_rows)
                    )

            if lm_x_weight != 0.0:
                outputs_x = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    images=images,
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial_crop,
                    return_dict=return_dict,
                    use_codebook=False,
                    prepared_feature_rows=direct_rows,
                )
                logits_x = self.lm_head(outputs_x[0]).float()
                shift_logits_x = logits_x[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
                lm_loss_x = loss_fct(shift_logits_x, shift_labels.to(shift_logits_x.device))

            outputs_z = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
                return_dict=return_dict,
                use_codebook=True,
                prepared_feature_rows=quant_rows,
            )
            logits = self.lm_head(outputs_z[0]).float()
            outputs = outputs_z
            shift_logits_z = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            lm_loss_z = loss_fct(shift_logits_z, shift_labels.to(shift_logits_z.device))

            weighted_lm_loss_x = lm_x_weight * lm_loss_x
            weighted_lm_loss_z = lm_z_weight * lm_loss_z
            weighted_mse_encoder_loss = mse_encoder_weight * mse_encoder_loss
            weighted_mse_codebook_loss = mse_codebook_weight * mse_codebook_loss
            loss = (
                weighted_lm_loss_x
                + weighted_lm_loss_z
                + weighted_mse_encoder_loss
                + weighted_mse_codebook_loss
            )

            self.model._last_mse_encoder_loss = mse_encoder_loss
            self.model._last_mse_codebook_loss = mse_codebook_loss

            self._last_loss_breakdown = {
                "loss_total": loss.detach().float().item(),
                "loss_lm": (weighted_lm_loss_x + weighted_lm_loss_z).detach().float().item(),
                "lm_x_weight": float(lm_x_weight),
                "lm_z_weight": float(lm_z_weight),
                "loss_lm_x": lm_loss_x.detach().float().item(),
                "loss_lm_x_weighted": weighted_lm_loss_x.detach().float().item(),
                "loss_lm_z": lm_loss_z.detach().float().item(),
                "loss_lm_z_weighted": weighted_lm_loss_z.detach().float().item(),
                "loss_mse_encoder": mse_encoder_loss.detach().float().item(),
                "loss_mse_encoder_weighted": weighted_mse_encoder_loss.detach().float().item(),
                "loss_mse_codebook": mse_codebook_loss.detach().float().item(),
                "loss_mse_codebook_weighted": weighted_mse_codebook_loss.detach().float().item(),
                "codebook_unique_codes": int(self.model.codebook_last_unique_codes.item()),
                "codebook_top1_fraction": float(self.model.codebook_last_top1_fraction.item()),
                "codebook_top5_fraction": float(self.model.codebook_last_top5_fraction.item()),
                "codebook_effective_codes": float(self.model.codebook_last_effective_codes.item()),
                "codebook_target_norm": float(self.model.codebook_target_norm.item()),
                "codebook_noise_std": float(self.model.codebook_noise_std.item()),
                "codebook_vmf_kappa": float(self.model.codebook_vmf_kappa.item()),
                "codebook_top1_distance": float(self.model.codebook_last_top1_distance.item()),
                "codebook_top10_distance": float(self.model.codebook_last_top10_distance.item()),
                "codebook_top128_distance": float(self.model.codebook_last_top128_distance.item()),
                "codebook_top128_angle": float(self.model.codebook_last_top128_angle.item()),
                "codebook_refresh_count": int(self.model.codebook_last_refresh_count.item()),
                "loss_aux_in_total": 1.0,
            }
        else:
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
                return_dict=return_dict,
                use_codebook=True,
            )
            logits = self.lm_head(outputs[0]).float()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.get_seq_length()
                max_cache_length = None
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if self.generation_config.cache_implementation == "static":
        #     # generation with static cache
        #     cache_position = kwargs.get("cache_position", None)
        #     if cache_position is None:
        #         past_length = 0
        #     else:
        #         past_length = cache_position[-1] + 1
        #     input_ids = input_ids[:, past_length:]
        #     position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "images_seq_mask": kwargs.get("images_seq_mask", None),
                "images_spatial_crop": kwargs.get("images_spatial_crop", None),
            }
        )
        return model_inputs
    

    def disable_torch_init(self):
        """
        Disable the redundant torch default initialization to accelerate model creation.
        """
        import torch
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)



    def infer(self, tokenizer, prompt='', image_file='', output_path = '', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
        self.disable_torch_init()

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/images', exist_ok=True)

        if prompt and image_file:
            conversation = [
                {
                    "role": "<|User|>",
                    # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                    "content": f'{prompt}',
                    # "content": "君不见黄河之水天上来的下一句是什么？",
                    # "content": "<image>\nFree OCR. ",
                    # "content": "<image>\nParse the figure. ",
                    # "content": "<image>\nExtract the text in the image. ",
                    "images": [f'{image_file}'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        
        elif prompt:
            conversation = [
                {
                    "role": "<|User|>",
                    # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                    "content": f'{prompt}',
                    # "content": "君不见黄河之水天上来的下一句是什么？",
                    # "content": "<image>\nFree OCR. ",
                    # "content": "<image>\nParse the figure. ",
                    # "content": "<image>\nExtract the text in the image. ",
                    # "images": [f'{image_file}'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            assert False, f'prompt is none!'
        
        prompt = format_messages(conversations=conversation, sft_format='plain', system_prompt='')

        patch_size = 16
        downsample_ratio = 4
        images = load_pil_images(conversation)

        valid_img_tokens = 0
        ratio = 1

        image_draw = images[0].copy()

        w,h = image_draw.size
        # print(w, h)
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
    

        image_transform=BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        images_seq_mask = []

        image_token = '<image>'
        image_token_id = 128815
        text_splits = prompt.split(image_token)

        images_list, images_crop_list, images_seq_mask = [], [], []
        tokenized_str = []
        images_spatial_crop = []
        for text_sep, image in zip(text_splits, images):

            tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if crop_mode:

                if image.size[0] <= 768 and image.size[1] <= 768:
                    crop_ratio = [1, 1]

                else:
                    if crop_mode:
                        # best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
                        images_crop_raw, crop_ratio = dynamic_preprocess(image)
                    else:
                        # best_width, best_height = self.image_size, self.image_size
                        crop_ratio = [1, 1]
                
                """process the global view"""
                # image = image.resize((base_size, base_size))
                global_view = ImageOps.pad(image, (base_size, base_size),
                                        color=tuple(int(x * 255) for x in image_transform.mean))
                
                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                # elif base_size == 640:
                #     valid_img_tokens += int(100 * ratio)
                



                
                images_list.append(image_transform(global_view).to(torch_dtype))

                # global_view_tensor = image_transform(global_view).to(torch_dtype)

                width_crop_num, height_crop_num = crop_ratio

                images_spatial_crop.append([width_crop_num, height_crop_num])
                
                
                if width_crop_num > 1 or height_crop_num > 1:
                    """process the local views"""
                    
                    for i in range(len(images_crop_raw)):
                        images_crop_list.append(image_transform(images_crop_raw[i]).to(torch_dtype))
                
                if image_size == 768:
                    valid_img_tokens += len(images_crop_list) * 144

                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)



                """add image tokens"""

                

                tokenized_image = ([image_token_id] * num_queries_base) * num_queries_base
                tokenized_image += [image_token_id]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([image_token_id] * (num_queries * width_crop_num)) * (
                                num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))

            else:
                """process the global view with adaptive square resize"""
                orig_w, orig_h = image.size
                target_size = min(max(orig_w, orig_h), 768)

                print(f"adaptive resize: original=({orig_w}, {orig_h}), target=({target_size}, {target_size})")

                image = image.resize((target_size, target_size))
                global_view = image

                images_list.append(image_transform(global_view).to(torch_dtype))


                if target_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif target_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif target_size == 640:
                    valid_img_tokens += int(100 * 1)
                elif target_size == 512:
                    valid_img_tokens += int(64 * 1)
                elif target_size == 768:
                    valid_img_tokens += int(144 * 1)
                elif target_size == 896:
                    valid_img_tokens += int(196 * 1)
                else:
                    valid_img_tokens += int((target_size // 64) ** 2)

                width_crop_num, height_crop_num = 1, 1
                images_spatial_crop.append([width_crop_num, height_crop_num])

                """add image tokens"""
                num_queries = math.ceil((target_size // patch_size) / downsample_ratio)

                tokenized_image = ([image_token_id] * num_queries) * num_queries
                tokenized_image += [image_token_id]

                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
        

        """process the last text split"""
        tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos tokens"""
        bos_id = 0
        tokenized_str = [bos_id] + tokenized_str 
        images_seq_mask = [False] + images_seq_mask



        input_ids = torch.LongTensor(tokenized_str)


        

        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)


        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, image_size, image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, base_size, base_size))

        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, base_size, base_size))



        if not eval_mode:
            streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            with torch.autocast("cuda", dtype=torch_dtype):
                with torch.no_grad():
                    output_ids = self.generate(
                        input_ids.unsqueeze(0).cuda(),
                        images=[(images_crop.cuda(), images_ori.cuda())],
                        images_seq_mask = images_seq_mask.unsqueeze(0).cuda(),
                        images_spatial_crop = images_spatial_crop,
                        # do_sample=False,
                        # num_beams = 1,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        streamer=streamer,
                        max_new_tokens=8192,
                        no_repeat_ngram_size = 20,
                        use_cache = True
                        )

        else:
            with torch.autocast("cuda", dtype=torch_dtype):
                with torch.no_grad():
                    output_ids = self.generate(
                        input_ids.unsqueeze(0).cuda(),
                        images=[(images_crop.cuda(), images_ori.cuda())],
                        images_seq_mask = images_seq_mask.unsqueeze(0).cuda(),
                        images_spatial_crop = images_spatial_crop,
                        # do_sample=False,
                        # num_beams = 1,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=8192,
                        no_repeat_ngram_size = 35,
                        use_cache = True
                        )
                

        if '<image>' in conversation[0]['content'] and eval_mode:
                outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
                stop_str = '<｜end▁of▁sentence｜>'
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                # re_match
                outputs = outputs.strip()

                return outputs
        
        if '<image>' in conversation[0]['content'] and test_compress:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
            pure_texts_outputs_token_length = len(text_encode(tokenizer, outputs, bos=False, eos=False))
            print('='*50)
            print('image size: ', (w, h))
            print('valid image tokens: ', int(valid_img_tokens))
            print('output texts tokens (valid): ', pure_texts_outputs_token_length)
            print('compression ratio: ', round(pure_texts_outputs_token_length/valid_img_tokens, 2))
            print('='*50)


        if '<image>' in conversation[0]['content'] and save_results:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
            stop_str = '<｜end▁of▁sentence｜>'

            print('='*15 + 'save results:' + '='*15)
            
            # # # # conv.messages[-1][-1] = outputs
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            matches_ref, matches_images, mathes_other = re_match(outputs)
            # print(matches_ref)
            result = process_image_with_refs(image_draw, matches_ref, output_path)


            for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
                outputs = outputs.replace(a_match_image, '![](images/' + str(idx) + '.jpg)\n')
            
            for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
                outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')


            # if 'structural formula' in conversation[0]['content']:
            #     outputs = '<smiles>' + outputs + '</smiles>'
            with open(f'{output_path}/result.mmd', 'w', encoding = 'utf-8') as afile:
                afile.write(outputs)

            if 'line_type' in outputs:
                import matplotlib.pyplot as plt
                lines = eval(outputs)['Line']['line']

                line_type = eval(outputs)['Line']['line_type']
                # print(lines)

                endpoints = eval(outputs)['Line']['line_endpoint']

                fig, ax = plt.subplots(figsize=(3,3), dpi=200)
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)

                for idx, line in enumerate(lines):
                    try:
                        p0 = eval(line.split(' -- ')[0])
                        p1 = eval(line.split(' -- ')[-1])

                        if line_type[idx] == '--':
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                        else:
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth = 0.8, color = 'k')

                        ax.scatter(p0[0], p0[1], s=5, color = 'k')
                        ax.scatter(p1[0], p1[1], s=5, color = 'k')
                    except:
                        pass

                for endpoint in endpoints:

                    label = endpoint.split(': ')[0]
                    (x, y) = eval(endpoint.split(': ')[1])
                    ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                                fontsize=5, fontweight='light')
                

                plt.savefig(f'{output_path}/geo.jpg')
                plt.close()

            result.save(f"{output_path}/result_with_boxes.jpg")
