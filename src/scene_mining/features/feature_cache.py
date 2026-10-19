"""FeatureCache: bridge between Path A (deep learning) and Path B (RefProg).

SigLIP2 patch features are computed ONCE per log during the backbone
forward pass and stored per (track_id, timestamp_ns). Visual atomic
functions then query the cache instead of re-encoding images.

Architecture:
- Per-track crop embeddings: SigLIP2 patch features mean-pooled within
  the 3D bounding box projection onto each camera image.
- Scene-level embeddings: mean over all valid patches in the front-center
  camera (for weather, time-of-day).
- BEV embeddings: 256-d BEV tokens from the spatial encoder (for scene
  description matching).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CachedFeature:
    """A single cached feature vector with associated metadata."""
    embedding: torch.Tensor   # (D,) float32
    visibility_score: float   # fraction of bbox visible in camera [0,1]
    camera_name: str
    timestamp_ns: int
    track_id: str


class FeatureCache:
    """Per-log feature cache for visual atomic functions.

    Stores:
    - _track_features: dict[(track_id, timestamp_ns)] → list[CachedFeature]
    - _scene_features: dict[timestamp_ns] → CachedFeature (front-center camera)
    - _bev_features: dict[timestamp_ns] → torch.Tensor (256-d BEV tokens mean)

    Args:
        log_id: AV2 log identifier.
        device: Device for similarity computations.
    """

    def __init__(self, log_id: str, device: torch.device | None = None) -> None:
        self.log_id = log_id
        self.device = device or torch.device("cpu")
        self._track_features: dict[tuple[str, int], list[CachedFeature]] = {}
        self._scene_features: dict[int, CachedFeature] = {}
        self._bev_features: dict[int, torch.Tensor] = {}

        # Text embedding cache (avoid re-encoding same text queries)
        self._text_cache: dict[str, torch.Tensor] = {}

        # Lazy SigLIP2 model for text encoding
        self._siglip2 = None

    # ── Population ────────────────────────────────────────────────────────────

    def add_track_feature(
        self,
        track_id: str,
        timestamp_ns: int,
        embedding: torch.Tensor,
        visibility_score: float,
        camera_name: str,
    ) -> None:
        """Add a cached crop embedding for a track at a timestamp."""
        key = (track_id, int(timestamp_ns))
        feat = CachedFeature(
            embedding=embedding.to(self.device),
            visibility_score=float(visibility_score),
            camera_name=camera_name,
            timestamp_ns=int(timestamp_ns),
            track_id=track_id,
        )
        self._track_features.setdefault(key, []).append(feat)

    def add_scene_feature(
        self,
        timestamp_ns: int,
        embedding: torch.Tensor,
        camera_name: str = "ring_front_center",
    ) -> None:
        """Add a scene-level feature for weather/time-of-day queries."""
        self._scene_features[int(timestamp_ns)] = CachedFeature(
            embedding=embedding.to(self.device),
            visibility_score=1.0,
            camera_name=camera_name,
            timestamp_ns=int(timestamp_ns),
            track_id="__scene__",
        )

    def add_bev_feature(self, timestamp_ns: int, bev_tokens: torch.Tensor) -> None:
        """Add BEV spatial tokens (mean-pooled across spatial dim)."""
        pooled = bev_tokens.mean(dim=0) if bev_tokens.dim() > 1 else bev_tokens
        self._bev_features[int(timestamp_ns)] = pooled.to(self.device)

    # ── Track feature queries ─────────────────────────────────────────────────

    def get_track_embedding(
        self,
        track_id: str,
        timestamp_ns: int | None = None,
    ) -> torch.Tensor | None:
        """Get visibility-weighted mean track embedding.

        If timestamp_ns is None, aggregates across all available timestamps.
        """
        if timestamp_ns is not None:
            feats = self._track_features.get((track_id, int(timestamp_ns)), [])
        else:
            feats = []
            for (tid, _), feat_list in self._track_features.items():
                if tid == track_id:
                    feats.extend(feat_list)

        if not feats:
            return None

        embeddings = torch.stack([f.embedding for f in feats])    # (N, D)
        weights = torch.tensor([f.visibility_score for f in feats], device=self.device)
        weights = weights / (weights.sum() + 1e-6)
        return (embeddings * weights.unsqueeze(-1)).sum(dim=0)

    def get_visibility_score(self, rec: Any) -> float | None:
        """Get the maximum visibility score for a track record."""
        feats = self._track_features.get((rec.track_id, int(rec.timestamp_ns)), [])
        if not feats:
            return None
        return max(f.visibility_score for f in feats)

    # ── Similarity queries ─────────────────────────────────────────────────────

    def color_similarity(self, rec: Any, color: str) -> float:
        """Cosine similarity between track embedding and color text."""
        track_emb = self.get_track_embedding(rec.track_id, rec.timestamp_ns)
        if track_emb is None:
            return 0.0
        text_emb = self._encode_text(f"a {color} colored object")
        return float(F.cosine_similarity(
            track_emb.unsqueeze(0), text_emb.unsqueeze(0)
        ).item())

    def attribute_similarity(self, rec: Any, attribute: str) -> float:
        """Cosine similarity between track embedding and attribute text."""
        track_emb = self.get_track_embedding(rec.track_id, rec.timestamp_ns)
        if track_emb is None:
            return 0.0
        text_emb = self._encode_text(attribute)
        return float(F.cosine_similarity(
            track_emb.unsqueeze(0), text_emb.unsqueeze(0)
        ).item())

    def weather_similarity(self, condition: str) -> float:
        """Scene-level similarity to a weather description."""
        scene_emb = self._aggregate_scene_embedding()
        if scene_emb is None:
            return 0.0
        text_emb = self._encode_text(f"driving in {condition} weather")
        return float(F.cosine_similarity(
            scene_emb.unsqueeze(0), text_emb.unsqueeze(0)
        ).item())

    def time_of_day_similarity(self, time: str) -> float:
        """Scene-level similarity to a time-of-day description."""
        scene_emb = self._aggregate_scene_embedding()
        if scene_emb is None:
            return 0.0
        text_emb = self._encode_text(f"driving at {time}")
        return float(F.cosine_similarity(
            scene_emb.unsqueeze(0), text_emb.unsqueeze(0)
        ).item())

    def scene_description_similarity(self, description: str) -> float:
        """BEV embedding similarity to a free-form scene description."""
        bev_emb = self._aggregate_bev_embedding()
        if bev_emb is None:
            return 0.0
        text_emb = self._encode_text(description)
        # BEV embedding may have different dim — project via cosine on min-dim
        b = bev_emb[:text_emb.shape[0]] if bev_emb.shape[0] > text_emb.shape[0] else bev_emb
        t = text_emb[:b.shape[0]]
        return float(F.cosine_similarity(b.unsqueeze(0), t.unsqueeze(0)).item())

    def vlm_vqa(self, rec: Any, question: str) -> bool:
        """VQA via VLM on the track crop (requires VLM to be set).

        Falls back to False if no VLM is available.
        """
        # VLM VQA path requires separate integration — return False as stub
        # Full implementation hooks into the VLM's generate() with crop patches
        return False

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_backbone_output(
        cls,
        log_id: str,
        backbone_outputs: list[dict],
        track_collection: Any,
        siglip2_model: Any,
        av2_dataset: Any,
        device: torch.device | None = None,
    ) -> "FeatureCache":
        """Build FeatureCache from batched backbone outputs for a full log.

        Args:
            log_id: AV2 log identifier.
            backbone_outputs: List of dicts, one per timestamp, each with:
                - ``patch_features``: (N_cam, N_patches, D) tensor
                - ``timestamp_ns``: int
                - ``bev_tokens``: (N_bev, D) tensor (optional)
            track_collection: TrackCollection for the log.
            siglip2_model: SigLIP2Backbone instance (for text encoding).
            av2_dataset: AV2Dataset (for calibration).
            device: Target device.

        Returns:
            Populated FeatureCache.
        """
        from scene_mining.features.track_crop_extractor import TrackCropExtractor

        cache = cls(log_id=log_id, device=device)
        cache._siglip2 = siglip2_model

        extractor = TrackCropExtractor(av2_dataset, log_id)

        for entry in backbone_outputs:
            ts = int(entry["timestamp_ns"])
            patch_features = entry["patch_features"]  # (N_cam, N_patches, D)
            bev_tokens = entry.get("bev_tokens")

            # Scene-level: mean of front-center camera patches
            front_idx = 0  # ring_front_center is index 0
            scene_emb = patch_features[front_idx].mean(dim=0)
            cache.add_scene_feature(ts, scene_emb, "ring_front_center")

            if bev_tokens is not None:
                cache.add_bev_feature(ts, bev_tokens)

            # Per-track crop features
            tracks_at_ts = track_collection.get_at_timestamp(ts)
            for rec in tracks_at_ts:
                crop_embs, vis_scores = extractor.extract(
                    rec, patch_features, ts
                )
                for cam_idx, (emb, vis) in enumerate(zip(crop_embs, vis_scores)):
                    if emb is not None:
                        cam_name = av2_dataset._get_camera_names()[cam_idx]
                        cache.add_track_feature(rec.track_id, ts, emb, vis, cam_name)

        return cache

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save cache to disk (pickle)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "log_id": self.log_id,
                "track_features": self._track_features,
                "scene_features": self._scene_features,
                "bev_features": self._bev_features,
            }, f)

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "FeatureCache":
        """Load cache from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        cache = cls(log_id=data["log_id"], device=device)
        cache._track_features = data["track_features"]
        cache._scene_features = data["scene_features"]
        cache._bev_features = data.get("bev_features", {})
        return cache

    # ── Private helpers ────────────────────────────────────────────────────────

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode a text string to a normalised embedding."""
        if text in self._text_cache:
            return self._text_cache[text]

        if self._siglip2 is not None:
            emb = self._siglip2.encode_text([text], device=self.device)[0]
        else:
            # Stub: random unit vector for testing without SigLIP2
            emb = F.normalize(torch.randn(768, device=self.device), dim=0)

        self._text_cache[text] = emb
        return emb

    def _aggregate_scene_embedding(self) -> torch.Tensor | None:
        """Mean scene embedding across all timestamps."""
        if not self._scene_features:
            return None
        embs = torch.stack([f.embedding for f in self._scene_features.values()])
        return embs.mean(dim=0)

    def _aggregate_bev_embedding(self) -> torch.Tensor | None:
        """Mean BEV embedding across all timestamps."""
        if not self._bev_features:
            return None
        embs = torch.stack(list(self._bev_features.values()))
        return embs.mean(dim=0)
