"""Data collator for Kanana/Qwen chat-template AES training.

Handles:
  - Chat template formatting (system/user/assistant)
  - CE label masking (instruction → -100, output → learn)
  - NTL label extraction (score tokens only)
  - SAL label extraction (feedback tokens after scores)
"""
from __future__ import annotations

import unicodedata
from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


class AESCollator(DataCollatorForLanguageModeling):
    """Collator that formats chat-template samples and creates CE + NTL labels."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1536,
        score_token_len: int = 16,
    ):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.score_token_len = score_token_len

    def _format_prompt(self, example: dict[str, str]) -> str:
        """Build the prompt (system + user) using chat template."""
        msgs = [
            {"role": "system", "content": normalize_text(example["system"])},
            {"role": "user", "content": normalize_text(example["user"])},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # 1) Build full sequences: prompt + assistant output + eos
        prompts = [self._format_prompt(ex) for ex in examples]
        full_texts = [
            p + normalize_text(ex["assistant"]) + self.tokenizer.eos_token
            for p, ex in zip(prompts, examples)
        ]

        batch = self.tokenizer(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=False,
        )

        # 2) CE labels: mask instruction part with -100
        labels = batch["input_ids"].clone()

        prompt_lens = []
        for i, prompt in enumerate(prompts):
            p_ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )["input_ids"][0]
            p_len = int(p_ids.size(0))
            prompt_lens.append(p_len)
            labels[i, :p_len] = -100

        # Mask padding
        labels[batch["attention_mask"] == 0] = -100

        # 3) NTL labels: only score tokens (first score_token_len tokens of output)
        ntl_labels = torch.full_like(labels, -100)
        B, T = labels.shape

        for i in range(B):
            start = prompt_lens[i]
            end = min(start + self.score_token_len, T)
            if start < T and start < end:
                ntl_labels[i, start:end] = labels[i, start:end]

        # 4) SAL labels: feedback tokens (output tokens after score tokens)
        sal_labels = torch.full_like(labels, -100)
        for i in range(B):
            sal_start = prompt_lens[i] + self.score_token_len
            if sal_start < T:
                sal_labels[i, sal_start:] = labels[i, sal_start:]

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
            "ntl_labels": ntl_labels,
            "sal_labels": sal_labels,
        }
