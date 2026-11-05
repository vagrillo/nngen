"""Frozen MiniLM text encoder for conditioning.

Wraps sentence-transformers/all-MiniLM-L6-v2 and exposes:
  - encode(captions)  -> token-level last hidden states (B, L<=64, 384) + mask
    (used as K/V for the UNet cross-attention)
  - encode_pooled(captions) -> mean-pooled sentence embedding (B, 384)
The model is frozen (eval + no-grad); runs fine on CPU.
"""

from __future__ import annotations

import torch

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 64


class TextEncoder:
    def __init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval().requires_grad_(False)

    def encode(self, captions: list[str], max_len: int = MAX_LEN) -> tuple[torch.Tensor, torch.Tensor]:
        """Token embeddings + attention mask, trimmed/padded to `max_len`."""
        batch = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state          # (B, L, D)
        L = batch["input_ids"].shape[1]
        emb = hidden[:, :max_len]
        mask = batch["attention_mask"][:, :max_len].float()
        if L < max_len:  # pad to max_len when the whole batch is short
            emb = torch.nn.functional.pad(emb, (0, 0, 0, max_len - L))
            mask = torch.nn.functional.pad(mask, (0, max_len - L))
        return emb, mask

    def encode_pooled(self, captions: list[str]) -> torch.Tensor:
        """Attention-mask-weighted mean of token embeddings, (B, D)."""
        emb, mask = self.encode(captions)
        mask = mask.unsqueeze(-1)
        return (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


_NULL_CACHE: tuple[torch.Tensor, torch.Tensor] | None = None


def null_embedding(encoder: TextEncoder, max_len: int = MAX_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    """Embedding of the empty prompt, used for classifier-free guidance."""
    global _NULL_CACHE
    if _NULL_CACHE is None:
        _NULL_CACHE = encoder.encode([""], max_len=max_len)
    return _NULL_CACHE
