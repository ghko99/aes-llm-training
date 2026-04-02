"""Custom trainer with CE + optional NTL + optional SAL loss.

Loss components:
  - CE: standard cross-entropy (always on)
  - NTL: Number Token Loss (MSE on expected digit value)
  - SAL: Semantic Alignment Loss (embedding-distance on feedback tokens)
  - WNTL: Weighted NTL with per-score class weights
  - Dynamic weighting: 0.5 * (CE + (CE/aux) * aux)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

from .number_token_loss import NumberTokenLoss


class AESTrainer(Trainer):
    """CE + optional NTL + optional SAL trainer for essay scoring."""

    def __init__(
        self,
        *args,
        use_ntl: bool = True,
        use_sal: bool = False,
        sal_topk: int = 64,
        num_tokenizer=None,
        score_pos_class_weights: Optional[List[Dict[int, float]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_ntl = use_ntl
        self.use_sal = use_sal
        self.sal_topk = int(sal_topk)

        self.num_tokenizer = num_tokenizer
        self.score_pos_class_weights = score_pos_class_weights

        device = self.args.device
        vocab_size = self.model.config.vocab_size

        # NTL criterion (only if enabled)
        if self.use_ntl:
            self.ntl_criterion = NumberTokenLoss(
                tokenizer=self.num_tokenizer,
                vocab_size=vocab_size,
                device=device,
                loss_function=torch.nn.functional.mse_loss,
            )

        # Score token ID → digit value mapping (for WNTL weighting)
        self._token_id_to_score: Dict[int, int] = {}
        if self.score_pos_class_weights is not None:
            if self.num_tokenizer is None:
                raise ValueError("num_tokenizer required for class weights")
            for s in range(1, 10):
                for text in (str(s), " " + str(s)):
                    enc = self.num_tokenizer.encode(text, add_special_tokens=False)
                    if len(enc) == 1:
                        self._token_id_to_score[enc[0]] = s

        self._last_logged_step = -1
        self._accum_losses: Dict[str, float] = {}
        self._accum_count: int = 0

    # ── Weighted NTL ──────────────────────────────────────────────────

    def _build_ntl_weights(
        self, ntl_labels: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        B, T = ntl_labels.shape
        weights = torch.ones(B, T, device=ntl_labels.device, dtype=torch.float32)

        if self.score_pos_class_weights is None:
            return weights

        for i in range(B):
            valid_idx = (ntl_labels[i] != -100).nonzero(as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                continue

            score_pos = 0
            for pos in valid_idx:
                tid = int(input_ids[i, pos].item())
                score_val = self._token_id_to_score.get(tid, None)
                if score_val is not None and score_pos < 8:
                    w = self.score_pos_class_weights[score_pos].get(score_val, 1.0)
                    weights[i, pos] = w
                    score_pos += 1

        return weights

    # ── SAL (Semantic Alignment Loss) ─────────────────────────────────

    @staticmethod
    def _get_embedding_matrix(model) -> torch.Tensor:
        if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
            return model.get_output_embeddings().weight
        if hasattr(model, "lm_head"):
            return model.lm_head.weight
        raise ValueError("Cannot find output embedding matrix (lm_head / get_output_embeddings).")

    def _sal_loss_topk(
        self, logits: torch.Tensor, sal_labels: torch.Tensor, model
    ) -> torch.Tensor:
        """Top-k embedding distance loss on feedback tokens (OOM-safe)."""
        logits_s = logits[:, :-1, :]         # [B, T-1, V]
        labels_s = sal_labels[:, 1:]         # [B, T-1]
        mask = labels_s.ne(-100)             # [B, T-1]

        if mask.sum().item() == 0:
            return logits_s.sum() * 0.0

        E = self._get_embedding_matrix(model)  # [V, D]
        B, Tm1, V = logits_s.shape
        D = E.size(1)

        labels_safe = labels_s.masked_fill(~mask, 0)
        k = min(self.sal_topk, V)

        lse = torch.logsumexp(logits_s, dim=-1)                  # [B, T-1]
        top_logits, topi = torch.topk(logits_s, k=k, dim=-1)     # [B, T-1, k]
        topv = torch.exp(top_logits - lse.unsqueeze(-1))         # [B, T-1, k]

        gt = labels_safe.unsqueeze(-1)
        topv = topv.masked_fill(topi.eq(gt), 0.0)
        topv = topv * mask.unsqueeze(-1)

        N = B * Tm1
        topi_f = topi.reshape(N, k)
        topv_f = topv.reshape(N, k).to(dtype=E.dtype)
        labels_f = labels_safe.reshape(N)
        mask_f = mask.reshape(N)

        total = torch.zeros((), device=logits.device, dtype=torch.float32)
        count = mask_f.sum().to(dtype=torch.float32).clamp_min(1.0)

        block_size = 128

        for s in range(0, N, block_size):
            e = min(s + block_size, N)

            idx_block = topi_f[s:e]
            w_block = topv_f[s:e]

            emb = E.index_select(0, idx_block.reshape(-1))
            emb = F.normalize(emb, p=2, dim=-1, eps=1e-12)
            emb = emb.view(e - s, k, D)
            q_block = (w_block.unsqueeze(-1) * emb).sum(dim=1)
            q_block = F.normalize(q_block, p=2, dim=-1, eps=1e-12)

            p_block = E.index_select(0, labels_f[s:e])
            p_block = F.normalize(p_block, p=2, dim=-1, eps=1e-12)

            cos = (p_block * q_block).sum(dim=-1)
            sal_tok = 1.0 - cos

            m = mask_f[s:e]
            if m.any():
                total = total + sal_tok[m].to(torch.float32).sum()

        return total / count

    # ── Total loss ────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Unsloth's prediction_step resets this to "0" after eval;
        # re-enable so logits are returned during training.
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        ntl_labels = inputs.get("ntl_labels", inputs.get("labels", None))
        sal_labels = inputs.get("sal_labels", None)

        inputs_fwd = {
            k: v for k, v in inputs.items() if k not in ("ntl_labels", "sal_labels")
        }

        # 1) CE loss
        base_loss, outputs = super().compute_loss(
            model, inputs_fwd, return_outputs=True, **kwargs
        )
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("Model outputs must include 'logits'.")

        eps = 1e-8

        # 2) NTL loss
        if self.use_ntl:
            logits_ntl = logits[:, :-1, :]
            labels_ntl = ntl_labels[:, 1:]

            if self.score_pos_class_weights is not None:
                ntl_weights = self._build_ntl_weights(ntl_labels, inputs_fwd["input_ids"])
                ntl_weights = ntl_weights[:, 1:]
                ntl_loss = self.ntl_criterion(logits_ntl, labels_ntl, sample_weights=ntl_weights)
            else:
                ntl_loss = self.ntl_criterion(logits_ntl, labels_ntl)
        else:
            ntl_loss = logits.sum() * 0.0

        # 3) SAL loss
        if self.use_sal and sal_labels is not None:
            sal_loss = self._sal_loss_topk(logits, sal_labels, model)
        else:
            sal_loss = logits.sum() * 0.0

        # 4) Dynamic weighting: 0.5 * (CE + (CE/aux) * aux)
        aux_loss = ntl_loss + sal_loss
        if self.use_ntl or self.use_sal:
            with torch.no_grad():
                dynamic_weight = base_loss.detach() / (aux_loss.detach() + eps)
            total_loss = 0.5 * (base_loss + dynamic_weight * aux_loss)
        else:
            total_loss = base_loss
            dynamic_weight = None

        # 5) Logging – accumulate across micro-batches, log the average
        if self.model.training:
            micro = {
                "loss_ce": base_loss.detach().item(),
                "loss_total": total_loss.detach().item(),
            }
            if self.use_ntl:
                micro["loss_ntl"] = ntl_loss.detach().item()
            if self.use_sal:
                micro["loss_sal"] = sal_loss.detach().item()
            if dynamic_weight is not None:
                micro["dynamic_weight"] = float(dynamic_weight.item())

            for k, v in micro.items():
                self._accum_losses[k] = self._accum_losses.get(k, 0.0) + v
            self._accum_count += 1

            ls = getattr(self.args, "logging_steps", 0) or 0
            ga = self.args.gradient_accumulation_steps
            do_log = (
                ls > 0
                and self.state.global_step % ls == 0
                and self._accum_count >= ga
                and self._last_logged_step != self.state.global_step
            )

            if do_log:
                log_dict = {
                    k: v / self._accum_count
                    for k, v in self._accum_losses.items()
                }
                self.log(log_dict)
                self._last_logged_step = self.state.global_step
                self._accum_losses = {}
                self._accum_count = 0

        return (total_loss, outputs) if return_outputs else total_loss
