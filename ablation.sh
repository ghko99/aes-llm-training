#!/bin/bash
# Ablation study: loss function combinations for Kanana AES
#
# Combinations:
#   1. CE only              (baseline)
#   2. CE + NTL             (unweighted NTL)
#   3. CE + WNTL            (weighted NTL)
#   4. CE + SAL             (SAL only)
#   5. CE + NTL + SAL       (unweighted NTL + SAL)
#   6. CE + WNTL + SAL      (weighted NTL + SAL)

set -e
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate llm

GPU=${1:-0}

COMMON_ARGS="--max_seq_length 1536 --batch_size 4 --grad_accum 8 --lr 2e-4 --epochs 10 --lora_r 16 --lora_alpha 32"

echo "============================================"
echo "[1/6] CE only (baseline)"
echo "============================================"
UNSLOTH_RETURN_LOGITS=1 CUDA_VISIBLE_DEVICES=$GPU python -m aes_training.train \
    $COMMON_ARGS --no_ntl

echo "============================================"
echo "[2/6] CE + NTL"
echo "============================================"
UNSLOTH_RETURN_LOGITS=1 CUDA_VISIBLE_DEVICES=$GPU python -m aes_training.train \
    $COMMON_ARGS --no_weighted_ntl

echo "============================================"
echo "[4/6] CE + SAL"
echo "============================================"
UNSLOTH_RETURN_LOGITS=1 CUDA_VISIBLE_DEVICES=$GPU python -m aes_training.train \
    $COMMON_ARGS --no_ntl --use_sal

echo "============================================"
echo "[5/6] CE + NTL + SAL"
echo "============================================"
UNSLOTH_RETURN_LOGITS=1 CUDA_VISIBLE_DEVICES=$GPU python -m aes_training.train \
    $COMMON_ARGS --no_weighted_ntl --use_sal

echo "============================================"
echo "[6/6] CE + WNTL + SAL"
echo "============================================"
UNSLOTH_RETURN_LOGITS=1 CUDA_VISIBLE_DEVICES=$GPU python -m aes_training.train \
    $COMMON_ARGS --use_sal

echo "============================================"
echo "Ablation study complete!"
echo "============================================"
