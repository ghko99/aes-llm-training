#!/bin/bash
# Multi-GPU training with standard HF + peft (no Unsloth)
# Automatically detects available GPUs

MODEL_PATH=${MODEL_PATH:-"/path/to/model"}

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected ${NUM_GPUS} GPUs"

accelerate launch \
    --multi_gpu \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    train.py \
    --model_path "$MODEL_PATH" \
    --no_unsloth \
    --max_seq_length 2560 \
    --batch_size 1 \
    --grad_accum $((32 / NUM_GPUS)) \
    --lr 2e-4 \
    --epochs 10 \
    --lora_r 16 \
    --lora_alpha 32
