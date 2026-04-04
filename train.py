from __future__ import annotations

import os
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

"""Main training script for Kanana AES fine-tuning.

Supports two backends:
  - Unsloth (default): single-GPU, ~2x speedup
  - Standard HF + peft (--no_unsloth): multi-GPU via accelerate

Usage:
    # Single GPU with Unsloth
    python train.py [options]

    # Multi-GPU without Unsloth
    accelerate launch train.py --no_unsloth [options]

Key features:
  - Expanded LoRA targets (all linear projections)
  - Compact chat-template dataset (shorter instruction)
  - Configurable loss: CE + optional NTL (WNTL) + optional SAL
"""

import datetime
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from collator import AESCollator
from inference import run_inference
from evaluate import evaluate_results
from number_tokenizer import AutoNumberTokenizer
from trainer import AESTrainer


# ── Config ────────────────────────────────────────────────────────────

MODEL_PATH = "/home/khko/models/kanana"
DATASET_DIR = Path(__file__).resolve().parent / "aes_datasets"

TEST_SPLITS = {
    "test_14_1": "test_14_1.jsonl",
    "test_14_2": "test_14_2.jsonl",
    "test_14_3": "test_14_3.jsonl",
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Class weight computation ──────────────────────────────────────────

def _count_scores_by_pos(ds) -> list[Counter]:
    counts = [Counter() for _ in range(8)]
    for ex in ds:
        line = ex["assistant"].splitlines()[0].strip()
        scores = list(map(int, line.split()))
        if len(scores) != 8:
            continue
        for pos, s in enumerate(scores):
            if 1 <= s <= 9:
                counts[pos][s] += 1
    return counts


def _build_class_weights(
    counts_by_pos: list[Counter],
    min_w: float = 0.7,
    max_w: float = 2.5,
) -> list[dict[int, float]]:
    weights = []
    for pos in range(8):
        c = counts_by_pos[pos]
        total = sum(c[s] for s in range(1, 10))
        if total == 0:
            weights.append({s: 1.0 for s in range(1, 10)})
            continue

        w_pos = {}
        for s in range(1, 10):
            cs = max(1, c[s])
            w = math.sqrt(total / (9 * cs))
            w = max(min_w, min(max_w, w))
            w_pos[s] = w

        # Normalize to mean=1
        mean_w = sum(w_pos.values()) / 9.0
        for s in w_pos:
            w_pos[s] /= mean_w

        weights.append(w_pos)
    return weights


# ── Main ──────────────────────────────────────────────────────────────

def train(
    max_seq_length: int = 2560,
    batch_size: int = 2,
    grad_accum: int = 16,
    lr: float = 2e-4,
    epochs: int = 10,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_ntl: bool = True,
    use_sal: bool = False,
    use_weighted_ntl: bool = True,
    resume_checkpoint: str | None = None,
    use_unsloth: bool = True,
):
    set_seed(42)

    # Output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build tag from loss configuration
    parts = ["kanana"]
    if use_ntl:
        parts.append("wntl" if use_weighted_ntl else "ntl")
    if use_sal:
        parts.append("sal")
    if not use_ntl and not use_sal:
        parts.append("ce")
    tag = "_".join(parts)

    runs_dir = Path(__file__).resolve().parent / "runs"
    output_dir = str(runs_dir / f"{tag}_{timestamp}")

    if resume_checkpoint:
        from transformers.trainer_utils import get_last_checkpoint
        output_dir = resume_checkpoint
        ckpt = get_last_checkpoint(output_dir)
        if ckpt is None:
            raise ValueError(f"No checkpoint found in {output_dir}")
    else:
        ckpt = None

    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    train_ds = load_dataset("json", data_files=str(DATASET_DIR / "train.jsonl"))["train"]
    valid_ds = load_dataset("json", data_files=str(DATASET_DIR / "valid.jsonl"))["train"]

    test_datasets = {}
    for split_name, split_file in TEST_SPLITS.items():
        split_path = DATASET_DIR / split_file
        if split_path.exists():
            test_datasets[split_name] = load_dataset("json", data_files=str(split_path))["train"]

    print(f"Dataset: train={len(train_ds)}, valid={len(valid_ds)}")
    for name, ds in test_datasets.items():
        print(f"  {name}={len(ds)}")

    # Class weights
    score_weights = None
    if use_weighted_ntl:
        counts = _count_scores_by_pos(train_ds)
        score_weights = _build_class_weights(counts)
        weights_path = os.path.join(output_dir, "score_pos_class_weights.json")
        serializable = [{str(k): float(v) for k, v in w.items()} for w in score_weights]
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"Class weights saved: {weights_path}")

    # Model + LoRA
    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,  # auto-detect (bf16 on 4090)
            attn_implementation="sdpa",
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        use_bf16 = torch.cuda.is_bf16_supported()
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, use_fast=True, trust_remote_code=True,
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    # Wrap tokenizer for number token support
    num_tokenizer = AutoNumberTokenizer.from_pretrained(
        MODEL_PATH, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    if num_tokenizer.pad_token is None:
        num_tokenizer.pad_token = num_tokenizer.eos_token
    num_tokenizer.model_max_length = max_seq_length

    collator = AESCollator(tokenizer, max_seq_length=max_seq_length)

    # Pre-compute token lengths for length-grouped batching
    def _estimate_length(example):
        text = example["system"] + example["user"] + example.get("assistant", "")
        return {"length": len(tokenizer.encode(text, add_special_tokens=False))}

    train_ds = train_ds.map(_estimate_length, num_proc=4)
    valid_ds = valid_ds.map(_estimate_length, num_proc=4)

    # W&B init
    import wandb
    wandb.init(project="aes-training", name=f"{tag}_{timestamp}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=10,
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=lr,
        bf16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=False,
        seed=42,
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="length",
        report_to="wandb",
        run_name=f"{tag}_{timestamp}",
    )

    trainer = AESTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        use_ntl=use_ntl,
        use_sal=use_sal,
        num_tokenizer=num_tokenizer,
        score_pos_class_weights=score_weights if use_ntl else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    if ckpt:
        print(f"Resuming from: {ckpt}")
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    print("Training complete. Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Inference + Evaluation (per test split)
    if use_unsloth:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    else:
        model.eval()
    csv_paths = []
    for split_name, test_ds in test_datasets.items():
        print(f"\n{'='*60}")
        print(f"Running inference on {split_name} ({len(test_ds)} samples)...")
        print(f"{'='*60}")
        csv_path = run_inference(model, num_tokenizer, test_ds, output_dir, split_name=split_name)
        csv_paths.append(csv_path)
        print(f"\nEvaluating {split_name}...")
        evaluate_results(csv_path, output_dir, split_name=split_name)

    # Overall evaluation by merging all split CSVs
    if len(csv_paths) > 1:
        import pandas as pd
        dfs = []
        offset = 0
        for p in csv_paths:
            df = pd.read_csv(p)
            df["sample_idx"] = df["sample_idx"] + offset
            offset = df["sample_idx"].max() + 1
            dfs.append(df)
        merged = pd.concat(dfs, ignore_index=True)
        merged_path = str(Path(output_dir) / "inference_results" / "test_all_inference.csv")
        merged.to_csv(merged_path, index=False)
        print(f"\n{'='*60}")
        print(f"Evaluating test_all (merged)...")
        print(f"{'='*60}")
        evaluate_results(merged_path, output_dir, split_name="test_all")

    return output_dir


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kanana AES Training (Unsloth)")
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--no_ntl", action="store_true", help="Disable NTL loss")
    parser.add_argument("--use_sal", action="store_true", help="Enable SAL loss")
    parser.add_argument("--no_weighted_ntl", action="store_true")
    parser.add_argument("--no_unsloth", action="store_true",
                        help="Disable Unsloth (use standard HF+peft, supports multi-GPU)")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    train(
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_ntl=not args.no_ntl,
        use_sal=args.use_sal,
        use_weighted_ntl=not args.no_weighted_ntl,
        resume_checkpoint=args.resume,
        use_unsloth=not args.no_unsloth,
    )


if __name__ == "__main__":
    main()
