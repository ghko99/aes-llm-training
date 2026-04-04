CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    train.py \
    --no_unsloth \
    --max_seq_length 2560 \
    --batch_size 1 \
    --grad_accum 16 \
    --lr 2e-4 \
    --epochs 10 \
    --lora_r 16 \
    --lora_alpha 32
