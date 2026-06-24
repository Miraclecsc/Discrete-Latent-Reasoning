import argparse
import importlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from PIL import Image
from PIL import ImageOps
from transformers import AutoModel
from tqdm import tqdm


DEFAULT_INPUT_JSONL = "data/train.jsonl"
DEFAULT_CHECKPOINT = "outputs/deepseek_codebook/checkpoint-last"
DEFAULT_OUTPUT_JSONL = "outputs/source2_codebook_input_ids.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export source2 samples with image codebook input ids from a trained checkpoint."
    )
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--source-name", default="source2")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--crop-mode", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_model_dtype(device: str):
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def get_base_model(model):
    return model.model if hasattr(model, "model") else model


def build_image_transform(remote_module):
    return remote_module.BasicImageTransform(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        normalize=True,
    )


def process_image(image, image_transform, image_size, base_size, crop_mode, dtype):
    patch_size = 16
    downsample_ratio = 4

    images_list = []
    images_crop_list = []
    images_spatial_crop = []

    if crop_mode:
        if image.size[0] <= 768 and image.size[1] <= 768:
            crop_ratio = (1, 1)
            images_crop_raw = []
        else:
            images_crop_raw, crop_ratio = remote_dynamic_preprocess(
                image,
                min_num=2,
                max_num=6,
                image_size=image_size,
                use_thumbnail=False,
            )

        global_view = ImageOps.pad(
            image,
            (base_size, base_size),
            color=tuple(int(x * 255) for x in image_transform.mean),
        )
        images_list.append(image_transform(global_view).to(dtype))
        width_crop_num, height_crop_num = crop_ratio
        images_spatial_crop.append([width_crop_num, height_crop_num])

        if width_crop_num > 1 or height_crop_num > 1:
            for crop_img in images_crop_raw:
                images_crop_list.append(image_transform(crop_img).to(dtype))

        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
        num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
        image_token_count = num_queries_base * num_queries_base + 1
        if width_crop_num > 1 or height_crop_num > 1:
            image_token_count += (num_queries * width_crop_num) * (num_queries * height_crop_num)
    else:
        images_spatial_crop.append([1, 1])
        orig_w, orig_h = image.size
        target_size = min(max(orig_w, orig_h), image_size)
        resized_image = image.resize((target_size, target_size), Image.LANCZOS)
        images_list.append(image_transform(resized_image).to(dtype))

        num_queries = math.ceil((target_size // patch_size) / downsample_ratio)
        image_token_count = num_queries * num_queries + 1

    images_ori = torch.stack(images_list, dim=0)
    if images_crop_list:
        images_crop = torch.stack(images_crop_list, dim=0)
    else:
        images_crop = torch.zeros((1, 3, base_size, base_size), dtype=dtype)

    images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
    return images_crop, images_ori, images_spatial_crop, image_token_count


def resolve_image_path(record, input_jsonl_path: Path):
    raw_path = record["rendered_image_path"]
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (input_jsonl_path.parent / path).resolve()


def count_matching_rows(input_jsonl, source_name):
    count = 0
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("source") == source_name:
                count += 1
    return count


def load_model(checkpoint, device):
    torch_dtype = resolve_model_dtype(device)
    print(f"[LOAD] checkpoint={checkpoint}")
    print(f"[LOAD] device={device} torch_dtype={torch_dtype}")
    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    base_model = get_base_model(model)
    base_model.codebook_assign_topk = 1
    base_model.codebook_assign_capacity = 0
    base_model.codebook_soft_update_topk = 1
    base_model.codebook_enable_refresh = False

    remote_module = importlib.import_module(base_model.__class__.__module__)
    return model, base_model, remote_module


def export_source2_records(args):
    input_jsonl_path = Path(args.input_jsonl)
    output_jsonl_path = Path(args.output_jsonl)

    if output_jsonl_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_jsonl_path}. Use --overwrite to replace it."
        )

    model, base_model, remote_module = load_model(args.checkpoint, args.device)
    image_transform = build_image_transform(remote_module)

    global remote_dynamic_preprocess
    remote_dynamic_preprocess = remote_module.dynamic_preprocess

    total = count_matching_rows(input_jsonl_path, args.source_name)
    if args.limit is not None:
        total = min(total, args.limit)

    processed = 0
    with open(input_jsonl_path, "r", encoding="utf-8") as fin, open(
        output_jsonl_path, "w", encoding="utf-8"
    ) as fout:
        progress = tqdm(total=total, desc=f"export {args.source_name}", dynamic_ncols=True)

        for line in fin:
            record = json.loads(line)
            if record.get("source") != args.source_name:
                continue
            if args.limit is not None and processed >= args.limit:
                break

            image_path = resolve_image_path(record, input_jsonl_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert("RGB")
            images_crop, images_ori, images_spatial_crop, image_token_count = process_image(
                image=image,
                image_transform=image_transform,
                image_size=args.image_size,
                base_size=args.base_size,
                crop_mode=args.crop_mode,
                dtype=model.dtype,
            )

            images = [
                (
                    images_crop.to(args.device),
                    images_ori.to(args.device),
                )
            ]
            images_spatial_crop = images_spatial_crop.to(args.device)

            with torch.inference_mode():
                feature_rows = base_model.encode_image_feature_rows(images, images_spatial_crop)
                _, _, _, _, nearest_idx_list = base_model.prepare_codebook_feature_rows(feature_rows)

            image_codebook_input_ids = nearest_idx_list[0].detach().cpu().tolist()
            output_record = {
                "id": record.get("id"),
                "source": record.get("source"),
                "text": record.get("text"),
                "original": record.get("original"),
                "rendered_image_path": record.get("rendered_image_path"),
                "resolved_image_path": str(image_path),
                "image_codebook_input_ids": image_codebook_input_ids,
                "image_codebook_token_count": len(image_codebook_input_ids),
                "expected_image_token_count": image_token_count,
                "codebook_placeholder_id": 0,
                "checkpoint": args.checkpoint,
                "preprocess": {
                    "crop_mode": args.crop_mode,
                    "image_size": args.image_size,
                    "base_size": args.base_size,
                    "assignment": "top1_nearest",
                    "noise": "disabled_in_eval",
                },
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

            processed += 1
            progress.update(1)
            if processed % args.log_every == 0:
                progress.set_postfix({"last_id": record.get("id"), "tokens": len(image_codebook_input_ids)})
                fout.flush()

        progress.close()

    print(f"[DONE] processed={processed}")
    print(f"[DONE] output={output_jsonl_path.resolve()}")


if __name__ == "__main__":
    export_source2_records(parse_args())
