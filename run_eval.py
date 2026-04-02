"""Run inference + evaluation on a trained model checkpoint.

Usage (from aes_training directory):
    python run_eval.py --adapter_dir runs/kanana_wntl_20260318_221936
    python run_eval.py --adapter_dir runs/kanana_wntl_20260318_221936 --test_file dataset/test.jsonl
"""
import argparse
import csv
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from number_tokenizer import AutoNumberTokenizer
from evaluate import evaluate_results

MODEL_PATH = "/home/khko/models/kanana"
DATASET_DIR = Path(__file__).resolve().parent / "aes_datasets"

TEST_SPLITS = {
    "test_14_1": "test_14_1.jsonl",
    "test_14_2": "test_14_2.jsonl",
    "test_14_3": "test_14_3.jsonl",
}


def build_digit_token_id_map(tokenizer) -> dict:
    cand_by_num = {d: [] for d in range(1, 10)}
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        try:
            val = tokenizer.decode_number_token(tok)
        except ValueError:
            continue
        if val in range(1, 10) and float(val).is_integer():
            cand_by_num[int(val)].append(tok)

    digit_map = {}
    for d, toks in cand_by_num.items():
        if not toks:
            for t in [str(d), f" {d}"]:
                if t in vocab:
                    digit_map[d] = vocab[t]
                    break
            if d not in digit_map:
                raise ValueError(f"Cannot find token for digit {d}")
            continue
        if str(d) in toks:
            chosen = str(d)
        elif f" {d}" in toks:
            chosen = f" {d}"
        else:
            chosen = toks[0]
        digit_map[d] = vocab[chosen]
    return digit_map


def build_prompt(tokenizer, example: dict) -> str:
    msgs = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def run_eval(
    adapter_dir: str,
    test_file: str,
    out_dir: str,
    max_seq_length: int = 1536,
    max_new_tokens: int = 16,
):
    # Load model
    use_bf16 = torch.cuda.is_bf16_supported()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    # Load tokenizer
    tokenizer = AutoNumberTokenizer.from_pretrained(
        adapter_dir, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    # Dataset
    ds = load_dataset("json", data_files=test_file)["train"]
    digit_id_map = build_digit_token_id_map(tokenizer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_name = Path(test_file).stem.replace(".jsonl", "")
    out_path = out_dir / f"{split_name}_inference.csv"

    fieldnames = [
        "sample_idx", "gen_pos", "label",
        "chosen_token", "chosen_token_id",
    ] + [f"prob_{i}" for i in range(1, 10)]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ex in enumerate(tqdm(ds, desc="Inference")):
            prompt = build_prompt(tokenizer, ex)
            label = ex.get("assistant", "")

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)

            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_ids = gen_out.sequences[:, input_ids.size(1):]
            gen_len = gen_ids.size(1)
            scores = gen_out.scores

            for p in range(gen_len):
                logits = scores[p].squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                chosen_id = int(gen_ids[0, p].item())
                chosen_tok = tokenizer.decode([chosen_id], skip_special_tokens=True)

                row = {
                    "sample_idx": idx,
                    "gen_pos": p + 1,
                    "label": label,
                    "chosen_token": chosen_tok,
                    "chosen_token_id": chosen_id,
                }
                for d in range(1, 10):
                    row[f"prob_{d}"] = float(probs[digit_id_map[d]].item())
                writer.writerow(row)

    print(f"Inference complete: {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Run inference + evaluation on a trained model")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Path to saved adapter (e.g. runs/kanana_wntl_20260318_221936)")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Test jsonl file (default: run all test splits)")
    parser.add_argument("--max_seq_length", type=int, default=2560)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    out_dir = str(Path(args.adapter_dir) / "inference_results")

    if args.test_file:
        # Single file mode
        test_files = {Path(args.test_file).stem: args.test_file}
    else:
        # All test splits
        test_files = {}
        for split_name, split_file in TEST_SPLITS.items():
            split_path = DATASET_DIR / split_file
            if split_path.exists():
                test_files[split_name] = str(split_path)

    print(f"Adapter: {args.adapter_dir}")
    print(f"Output dir: {out_dir}")

    csv_paths = []
    for split_name, test_file in test_files.items():
        print(f"\n{'='*60}")
        print(f"Running inference on {split_name}: {test_file}")
        print(f"{'='*60}")
        csv_path = run_eval(
            adapter_dir=args.adapter_dir,
            test_file=test_file,
            out_dir=out_dir,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
        )
        csv_paths.append(csv_path)

        print(f"\nEvaluating {split_name}...")
        evaluate_results(csv_path, out_dir, split_name=split_name)

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
        merged_path = str(Path(out_dir) / "test_all_inference.csv")
        merged.to_csv(merged_path, index=False)
        print(f"\n{'='*60}")
        print(f"Evaluating test_all (merged)...")
        print(f"{'='*60}")
        evaluate_results(merged_path, out_dir, split_name="test_all")


if __name__ == "__main__":
    main()
