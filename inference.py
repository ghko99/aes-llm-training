"""Inference module — runs greedy decoding on test set and saves per-token probabilities.

Outputs CSV with columns:
  sample_idx, gen_pos, label, chosen_token, chosen_token_id, prob_1..prob_9
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from number_tokenizer import AutoNumberTokenizer



def build_digit_token_id_map(tokenizer) -> dict[int, int]:
    """Map digits 1-9 to their most canonical token IDs."""
    cand_by_num: dict[int, list[str]] = {d: [] for d in range(1, 10)}
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


def load_inference_model(adapter_dir: str, base_model: str = None):
    """Load quantized base model with LoRA adapter.

    Args:
        base_model: Path to base model. If None, reads from adapter config.
    """
    if base_model is None:
        from peft import PeftConfig
        base_model = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
    use_bf16 = torch.cuda.is_bf16_supported()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model


def _build_prompt(tokenizer, example: dict) -> str:
    """Build chat-template prompt from dataset sample."""
    msgs = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def run_test_and_save_csv(
    test_file: str,
    out_dir: str,
    adapter_dir: str,
    max_seq_length: int = 1024,
    max_new_tokens: int = 16,
):
    """Run inference on test set and save results to CSV."""
    model = load_inference_model(adapter_dir)
    tokenizer = AutoNumberTokenizer.from_pretrained(
        adapter_dir, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    ds = load_dataset("json", data_files=test_file)["train"]
    digit_id_map = build_digit_token_id_map(tokenizer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(adapter_dir).name}_inference.csv"

    fieldnames = [
        "sample_idx", "gen_pos", "label",
        "chosen_token", "chosen_token_id",
    ] + [f"prob_{i}" for i in range(1, 10)]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ex in enumerate(tqdm(ds, desc="Inference")):
            prompt = _build_prompt(tokenizer, ex)
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
                chosen_tok = tokenizer.decode(
                    [chosen_id], skip_special_tokens=True
                )

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


@torch.inference_mode()
def run_inference(model, tokenizer, test_dataset, out_dir: str, split_name: str = "test") -> str:
    """Run inference using an already-loaded model."""
    out_dir = Path(out_dir) / "inference_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}_inference.csv"

    digit_id_map = build_digit_token_id_map(tokenizer)
    max_seq_length = tokenizer.model_max_length

    fieldnames = [
        "sample_idx", "gen_pos", "label",
        "chosen_token", "chosen_token_id",
    ] + [f"prob_{i}" for i in range(1, 10)]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ex in enumerate(tqdm(test_dataset, desc="Inference")):
            prompt = _build_prompt(tokenizer, ex)
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
                max_new_tokens=16,
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
            scores_out = gen_out.scores

            for p in range(gen_len):
                logits = scores_out[p].squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                chosen_id = int(gen_ids[0, p].item())
                chosen_tok = tokenizer.decode(
                    [chosen_id], skip_special_tokens=True
                )

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
