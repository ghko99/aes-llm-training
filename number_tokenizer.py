"""Number-aware tokenizer wrapper for score token handling.

Identifies tokens that represent numeric values in the vocabulary,
used by NumberTokenLoss to compute expected values over digit predictions.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

NUMBER_REGEX = r"(\d+)(\.\d+)?"


class NumberEncodingTokenizer(PreTrainedTokenizerFast, ABC):
    @abstractmethod
    def get_num_token_ids(self) -> List[int]: ...

    @abstractmethod
    def get_num_tokens(self) -> List[str]: ...

    @abstractmethod
    def decode_number_token(self, token: str, ignore_order: bool = True) -> float: ...


class AutoNumberTokenizer(NumberEncodingTokenizer):
    """Tokenizer wrapper that detects number tokens in the vocabulary."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        vocab = self.get_vocab()
        self.num_tokens = []
        self.num_token_ids = []

        for token, tid in vocab.items():
            try:
                self.decode_number_token(token)
                self.num_tokens.append(token)
                self.num_token_ids.append(tid)
            except ValueError:
                continue

        if not self.num_tokens:
            raise ValueError("No number tokens found in vocabulary")

    def get_num_token_ids(self) -> List[int]:
        return self.num_token_ids

    def get_num_tokens(self) -> List[str]:
        return self.num_tokens

    def decode_number_token(self, token: str, ignore_order: bool = True) -> float:
        # Strip Qwen/Llama/Kanana space prefixes
        clean = token.lstrip("▁ Ġ")
        try:
            return float(clean)
        except ValueError:
            raise ValueError(f"Cannot convert token {token!r} to float")
