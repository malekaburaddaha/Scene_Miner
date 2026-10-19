"""SigLIP2 backbone adapter.

Wraps ``google/siglip2-base-patch16-224`` (or larger variants) from
HuggingFace transformers. Patch features are projected to a common
``proj_dim`` for the spatial encoder.

Text-image similarity is exposed for visual atomic functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone


class SigLIP2Backbone(BaseBackbone):
    """SigLIP2 image encoder as a multi-camera patch feature extractor.

    Args:
        model_name: HuggingFace model ID.
        proj_dim: Projection dimension for downstream spatial encoder.
        freeze: Whether to freeze the backbone weights.
        image_size: Input image resolution (H=W). SigLIP2 base uses 224.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        proj_dim: int = 256,
        freeze: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor

        self.model_name = model_name
        self.image_size = image_size

        # Load vision encoder only
        full_model = AutoModel.from_pretrained(model_name)
        self.encoder = full_model.vision_model
        # SigLIP2 is vision-only (no tokenizer); AutoImageProcessor avoids
        # a NoneType error in AutoProcessor's tokenizer-mapping lookup.
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Infer hidden dim from config
        hidden_dim = self.encoder.config.hidden_size
        self._feature_dim = hidden_dim

        # Projection to downstream dim
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
        """Extract SigLIP2 patch features.

        Args:
            images: (B, N_cam, 3, H, W) float32.

        Returns:
            (B, N_cam, N_patches, proj_dim)
        """
        B, N_cam, C, H, W = images.shape
        # Flatten cameras into batch dim for encoder
        imgs_flat = images.view(B * N_cam, C, H, W)

        # SigLIP2 outputs hidden states for all patches + CLS.
        # interpolate_pos_encoding=True handles images != 224×224 (e.g. synthetic 64×64).
        outputs = self.encoder(pixel_values=imgs_flat, interpolate_pos_encoding=True)
        # last_hidden_state: (B*N_cam, N_patches+1, hidden_dim)
        patch_feats = outputs.last_hidden_state[:, 1:, :]  # drop CLS

        # Project
        patch_feats = self.proj(patch_feats)  # (B*N_cam, N_patches, proj_dim)
        _, N_p, D = patch_feats.shape
        return patch_feats.view(B, N_cam, N_p, D)

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        """Encode text prompts to feature vectors (for visual atomic functions).

        Args:
            texts: List of text strings.
            device: Target device.

        Returns:
            (N_texts, hidden_dim) normalised float32 text embeddings.
        """
        from transformers import AutoModel, AutoTokenizer
        import torch

        device = device or next(self.encoder.parameters()).device
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=64
        ).to(device)

        full_model = AutoModel.from_pretrained(self.model_name).to(device)
        text_feats = full_model.text_model(**inputs).pooler_output  # (N, D_text)
        return F.normalize(text_feats, dim=-1)

    @torch.no_grad()
    def image_text_similarity(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between image and text features.

        Args:
            image_features: (N_images, D) normalised.
            text_embeddings: (N_texts, D) normalised.

        Returns:
            (N_images, N_texts) cosine similarity matrix.
        """
        img_norm = F.normalize(image_features, dim=-1)
        txt_norm = F.normalize(text_embeddings, dim=-1)
        return img_norm @ txt_norm.T
