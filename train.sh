#!/bin/bash
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate llm

UNSLOTH_RETURN_LOGITS=1 CUDA_VISIBLE_DEVICES=0 python -m aes_training.train \
    --max_seq_length 2560 \
    --batch_size 1 \
    --grad_accum 32 \
    --lr 2e-4 \
    --epochs 10 \
    --lora_r 16 \
    --lora_alpha 32
