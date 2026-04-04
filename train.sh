#!/bin/bash
# Single-GPU training with Unsloth

MODEL_PATH=${MODEL_PATH:-"/path/to/model"}

UNSLOTH_RETURN_LOGITS=1 python train.py \
    --model_path "$MODEL_PATH" \
    --max_seq_length 2560 \
    --batch_size 1 \
    --grad_accum 32 \
    --lr 2e-4 \
    --epochs 10 \
    --lora_r 16 \
    --lora_alpha 32
