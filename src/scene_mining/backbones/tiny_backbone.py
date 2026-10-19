"""TinyBackbone — CPU-friendly patch embedder, no pretrained weights.

Replaces SigLIP2 / DINOv2 for unit tests and CPU smoke-training runs.
Uses a single Conv2d to tile each camera image into patch tokens, producing
the same ``(B, N_cam, N_patches, proj_dim)`` layout as the real backbones.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyBackbone(nn.Module):
    """Convolutional patch embedder — fully CPU-capable, no downloads.

    Args:
        proj_dim: Output feature dimension (match spatial encoder embed_dims).
        patch_size: Stride of the patch Conv2d (pixels). With image_h=64
            and patch_size=16 → 4×4 = 16 patches per camera.
        freeze: Ignored — kept for API compatibility with real backbones.
    """

    def __init__(
        self,
        proj_dim: int = 64,
        patch_size: int = 16,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, proj_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.kaiming_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: ``(B, N_cam, 3, H, W)`` float32

        Returns:
            ``(B, N_cam, N_patches, proj_dim)`` where
            ``N_patches = (H // patch_size) * (W // patch_size)``
        """
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)
        x = self.proj(x)                        # (B*N, proj_dim, ph, pw)
        _, D, ph, pw = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B*N, ph*pw, proj_dim)
        return x.view(B, N, ph * pw, D)
