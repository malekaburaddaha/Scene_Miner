"""Base backbone interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """Abstract base for all image backbone encoders.

    Subclasses must implement ``forward`` and expose ``feature_dim``
    (the channel dimension of output patch features).
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Output feature dimension (channels per patch token)."""
        ...

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract multi-camera patch features.

        Args:
            images: (B, N_cam, 3, H, W) float32 normalised [0,1].

        Returns:
            patch_features: (B, N_cam, N_patches, D) float32.
        """
        ...
