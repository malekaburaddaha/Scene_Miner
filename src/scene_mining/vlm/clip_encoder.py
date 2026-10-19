"""CLIP text encoder for embedding text queries into the scene retrieval space.

Wraps HuggingFace ``openai/clip-vit-base-patch32`` (or compatible) and
produces L2-normalised 512-d text embeddings.  All parameters are frozen —
this module is inference-only.

Usage::

    encoder = CLIPTextEncoder()
    emb = encoder(["rainy intersection at night"])  # (1, 512)
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from scene_mining.models.scene_miner import SCENE_MINING_MODELS
    _HAS_REGISTRY = SCENE_MINING_MODELS is not None
except ImportError:
    _HAS_REGISTRY = False


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder for query embedding.

    Args:
        model_name: HuggingFace model ID for a CLIP checkpoint.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()

        from transformers import CLIPModel, CLIPTokenizer

        self.model: CLIPModel = CLIPModel.from_pretrained(model_name)
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(model_name)

        # Freeze all parameters — inference only
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of strings into L2-normalised embeddings.

        Args:
            texts: List of B strings.

        Returns:
            (B, 512) float32 tensor, L2-normalised, no gradients.
        """
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=77,
            return_tensors="pt",
        )
        # Move tokenised inputs to same device as model
        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        features = self.model.get_text_features(**tokens)  # (B, 512)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return features


# ── Registry ─────────────────────────────────────────────────────────────────
if _HAS_REGISTRY:
    SCENE_MINING_MODELS.register_module()(CLIPTextEncoder)
