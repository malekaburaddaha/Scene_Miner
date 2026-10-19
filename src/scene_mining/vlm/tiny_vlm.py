"""TinyVLM — CPU-only two-layer transformer VLM stub.

Satisfies the BaseVLM interface without any pretrained weights,
GPU, or internet connection.  Intended for unit tests and CPU
smoke-training runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_vlm import BaseVLM


class TinyVLM(BaseVLM):
    """Minimal transformer VLM — fully CPU-capable, no downloads.

    Args:
        d_model: Hidden dimension.  Must match the QFormer ``d_vlm`` setting.
        n_heads: Attention heads.  Must evenly divide ``d_model``.
        n_layers: Number of TransformerEncoder layers.
        vocab_size: Vocabulary size for the token embedding table.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        vocab_size: int = 512,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(
        self,
        visual_tokens: torch.Tensor,        # (B, N_vis, d_model)
        input_ids: torch.Tensor,            # (B, L)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepend visual tokens to text embeddings and run encoder.

        Returns:
            ``(B, N_vis + L, d_model)``
        """
        text_emb = self.embed(input_ids)                    # (B, L, d_model)
        x = torch.cat([visual_tokens, text_emb], dim=1)    # (B, N_vis+L, d_model)
        return self.transformer(x)

    def generate(
        self,
        visual_tokens: torch.Tensor,
        prompts: list[str],
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Return stub captions (not real generation)."""
        return ["<tiny-vlm-stub>" for _ in prompts]
