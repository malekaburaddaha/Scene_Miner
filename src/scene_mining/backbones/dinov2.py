"""DINOv2 backbone adapter."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseBackbone


class DINOv2Backbone(BaseBackbone):
    """DINOv2 ViT-B/14 (or larger) patch feature extractor.

    Args:
        model_name: HuggingFace model ID (``"facebook/dinov2-base"`` etc.)
        proj_dim: Projection dimension for the spatial encoder.
        freeze: Whether to freeze backbone weights.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        proj_dim: int = 256,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return self.proj[-1].normalized_shape[0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 patch features.

        Args:
            images: (B, N_cam, 3, H, W) float32.

        Returns:
            (B, N_cam, N_patches, proj_dim)
        """
        B, N_cam, C, H, W = images.shape
        imgs_flat = images.view(B * N_cam, C, H, W)

        outputs = self.encoder(pixel_values=imgs_flat)
        # DINOv2 last_hidden_state: (B*N_cam, N_patches+1, hidden_dim)
        patch_feats = outputs.last_hidden_state[:, 1:]  # drop CLS

        patch_feats = self.proj(patch_feats)
        _, N_p, D = patch_feats.shape
        return patch_feats.view(B, N_cam, N_p, D)
