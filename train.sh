#!/bin/bash
# Single-GPU training with Unsloth
# Uses the first available GPU automatically

UNSLOTH_RETURN_LOGITS=1 python train.py \
    --max_seq_length 2560 \
    --batch_size 1 \
    --grad_accum 32 \
    --lr 2e-4 \
    --epochs 10 \
    --lora_r 16 \
    --lora_alpha 32
