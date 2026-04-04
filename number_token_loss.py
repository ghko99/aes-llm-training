"""Number Token Loss — MSE on expected digit values.

Computes the expected numeric value from softmax over number tokens
and applies MSE loss against the true digit label (1-9).

Supports per-token weighting for class-imbalanced score distributions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from number_tokenizer import NumberEncodingTokenizer


class NumberTokenSelector:
    """Selects and maps number tokens from the full vocabulary."""

    def __init__(self, tokenizer: NumberEncodingTokenizer, vocab_size: int, device):
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        hashed = set(tokenizer.get_num_tokens())
        for token, tid in tokenizer.get_vocab().items():
            if token in hashed:
                try:
                    val = tokenizer.decode_number_token(token, ignore_order=True)
                    tval = torch.tensor(val, device=device, dtype=torch.float32)
                    if torch.isfinite(tval):
                        self.nvocab[tid] = float(val)
                except Exception:
                    pass

        self.number_token_mask = torch.isfinite(self.nvocab)
        self.number_token_indices = torch.nonzero(
            self.number_token_mask, as_tuple=False
        ).squeeze(-1)
        self.values_all = torch.nan_to_num(
            self.nvocab[self.number_token_indices], nan=0.0
        )

        # Digit (1-9) subset
        vals = self.values_all
        is_int = vals == torch.round(vals)
        self.mask_digits = is_int & (vals >= 1) & (vals <= 9)
        self.values_digits = vals[self.mask_digits]

    def select_number_tokens(self, logits: Tensor):
        return logits[:, :, self.number_token_mask], self.number_token_mask


class NumberTokenLoss:
    """MSE loss on expected digit value from softmax over number tokens."""

    def __init__(
        self,
        tokenizer: NumberEncodingTokenizer,
        vocab_size: int,
        device,
        loss_function=F.mse_loss,
    ):
        self.loss_function = loss_function
        self.selector = NumberTokenSelector(tokenizer, vocab_size, device)
        self.nvocab = self.selector.nvocab

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        if logits.numel() == 0 or labels.numel() == 0:
            return logits.sum() * 0.0

        # Select number token logits
        logits_num, _ = self.selector.select_number_tokens(logits)
        softmaxed = F.softmax(torch.clamp(logits_num, min=-50, max=50), dim=-1)

        # Full expected value (fallback)
        values_all = self.selector.values_all
        yhat_all = torch.sum(softmaxed * values_all, dim=-1)
        yhat_all = torch.nan_to_num(yhat_all, nan=0.0)

        # Digit-only expected value (re-normalized)
        mask_digits = self.selector.mask_digits
        values_digits = self.selector.values_digits
        if values_digits.numel() == 0:
            yhat_digits = yhat_all
            p_sum = torch.zeros_like(yhat_all)
        else:
            p_digits = softmaxed[..., mask_digits]
            p_sum = p_digits.sum(-1, keepdim=True)
            p_digits = p_digits / p_sum.clamp_min(1e-12)
            yhat_digits = torch.sum(p_digits * values_digits, dim=-1)
            yhat_digits = torch.nan_to_num(yhat_digits, nan=0.0)

        # True numeric values
        safe_labels = labels.masked_fill(labels == -100, 0)
        y_all = self.nvocab[safe_labels]
        y_all = torch.nan_to_num(y_all, nan=0.0)

        # Valid mask: not ignored AND digit label (1-9)
        is_digit = (y_all == torch.round(y_all)) & (y_all >= 1) & (y_all <= 9)
        valid = (labels != -100) & is_digit

        if valid.sum() == 0:
            return logits_num.sum() * 0.0

        # Choose digit vs full expected value
        use_digits = is_digit & (p_sum.squeeze(-1) > 1e-8)
        yhat = torch.where(use_digits, yhat_digits, yhat_all)

        # Loss computation
        if sample_weights is not None:
            per_elem = (yhat[valid] - y_all[valid]) ** 2
            w = sample_weights[valid]
            w = w / w.mean().clamp_min(1e-8)
            loss = (per_elem * w).mean()
        else:
            loss = self.loss_function(yhat[valid], y_all[valid])

        if torch.isnan(loss) or torch.isinf(loss):
            return logits_num.sum() * 0.0

        return loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
