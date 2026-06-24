#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT/latent_llm/scripts"

normalize_bool() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        0|false|no|n|off) return 1 ;;
        *)
            echo "Invalid boolean value: $1" >&2
            exit 1
            ;;
    esac
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29510}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/latent_llm/trl:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Uncomment these only if your machine needs explicit NCCL tuning.
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct}"
CODEBOOK_PATH="${CODEBOOK_PATH:-$PROJECT_ROOT/outputs/deepseek_codebook/latent/codebook.pt}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$PROJECT_ROOT/data/latent_pretrain.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/latent_pretrain}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PROJECT_ROOT/latent_llm/configs/deepspeed_zero2_bf16.json}"

MODEL_PATH_LOWER="${MODEL_PATH,,}"
if [[ -z "${GRADIENT_CHECKPOINTING:-}" ]]; then
    if [[ "$MODEL_PATH_LOWER" == *deepseek*moe*decoder* ]]; then
        GRADIENT_CHECKPOINTING=false
    else
        GRADIENT_CHECKPOINTING=true
    fi
fi
if [[ -z "${DDP_FIND_UNUSED_PARAMETERS:-}" ]]; then
    if [[ "$MODEL_PATH_LOWER" == *deepseek*moe*decoder* ]]; then
        DDP_FIND_UNUSED_PARAMETERS=false
    else
        DDP_FIND_UNUSED_PARAMETERS=true
    fi
fi

NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-10000}"
CODEBOOK_DIM="${CODEBOOK_DIM:-1280}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
OPTIM="${OPTIM:-adamw_torch}"
OPTIM_ARGS="${OPTIM_ARGS:-foreach=False}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
FREEZE_BASE_MODEL="${FREEZE_BASE_MODEL:-true}"
FREEZE_CODEBOOK="${FREEZE_CODEBOOK:-true}"
BF16="${BF16:-true}"

echo "=== Latent Pretrain 8-GPU Launch ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "CODEBOOK_PATH=$CODEBOOK_PATH"
echo "TRAIN_DATA_PATH=$TRAIN_DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DEEPSPEED_CONFIG=$DEEPSPEED_CONFIG"
echo "GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
echo "DDP_FIND_UNUSED_PARAMETERS=$DDP_FIND_UNUSED_PARAMETERS"
echo ""

CMD=(
    torchrun
    --nnodes=1
    --nproc_per_node="$GPUS_PER_NODE"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    train_latent_pretrain.py
    --model_name_or_path "$MODEL_PATH"
    --codebook_path "$CODEBOOK_PATH"
    --train_data_path "$TRAIN_DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --num_latent_tokens "$NUM_LATENT_TOKENS"
    --codebook_dim "$CODEBOOK_DIM"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --max_steps "$MAX_STEPS"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --warmup_ratio "$WARMUP_RATIO"
    --logging_steps "$LOGGING_STEPS"
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --optim "$OPTIM"
    --optim_args "$OPTIM_ARGS"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --deepspeed "$DEEPSPEED_CONFIG"
)

if normalize_bool "$GRADIENT_CHECKPOINTING"; then
    CMD+=(--gradient_checkpointing)
else
    CMD+=(--no-gradient_checkpointing)
fi

if normalize_bool "$DDP_FIND_UNUSED_PARAMETERS"; then
    CMD+=(--ddp_find_unused_parameters)
else
    CMD+=(--no-ddp_find_unused_parameters)
fi

if normalize_bool "$FREEZE_BASE_MODEL"; then
    CMD+=(--freeze_base_model)
else
    CMD+=(--no-freeze_base_model)
fi

if normalize_bool "$FREEZE_CODEBOOK"; then
    CMD+=(--freeze_codebook)
else
    CMD+=(--no-freeze_codebook)
fi

if normalize_bool "$BF16"; then
    CMD+=(--bf16)
else
    CMD+=(--no-bf16)
fi

if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
    CMD+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

CMD+=("$@")
exec "${CMD[@]}"
