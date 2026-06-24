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
export MASTER_PORT="${MASTER_PORT:-29600}"
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

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/latent_sft}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$PROJECT_ROOT/data/latent_grpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/latent_grpo}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PROJECT_ROOT/latent_llm/configs/deepspeed_zero2_bf16.json}"

MODEL_PATH_LOWER="${MODEL_PATH,,}"
if [[ -z "${GRADIENT_CHECKPOINTING:-}" ]]; then
    if [[ "$MODEL_PATH_LOWER" == *deepseek*moe*decoder* ]]; then
        GRADIENT_CHECKPOINTING=false
    else
        GRADIENT_CHECKPOINTING=true
    fi
fi

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-4}"
REWARD_MODE="${REWARD_MODE:-number}"
BETA="${BETA:-0.0}"

echo "=== Latent GRPO 8-GPU Launch ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "TRAIN_DATA_PATH=$TRAIN_DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DEEPSPEED_CONFIG=$DEEPSPEED_CONFIG"
echo "NUM_GENERATIONS=$NUM_GENERATIONS"
echo "REWARD_MODE=$REWARD_MODE"
echo "BETA=$BETA"
echo "GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
echo ""

CMD=(
    torchrun
    --nnodes=1
    --nproc_per_node="$GPUS_PER_NODE"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    train_latent_grpo.py
    --model_name_or_path "$MODEL_PATH"
    --train_data_path "$TRAIN_DATA_PATH"
    --output_dir "$OUTPUT_DIR"
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
    --num_generations "$NUM_GENERATIONS"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --steps_per_generation "$STEPS_PER_GENERATION"
    --reward_mode "$REWARD_MODE"
    --beta "$BETA"
    --bf16
    --deepspeed "$DEEPSPEED_CONFIG"
)

if normalize_bool "$GRADIENT_CHECKPOINTING"; then
    CMD+=(--gradient_checkpointing)
else
    CMD+=(--no-gradient_checkpointing)
fi

CMD+=("$@")
exec "${CMD[@]}"
