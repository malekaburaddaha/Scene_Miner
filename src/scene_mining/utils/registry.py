"""Shared mmengine registries for scene_mining.

Re-exports SCENE_MINING_MODELS from models.scene_miner and adds
SCENE_MINING_DATASETS and SCENE_MINING_LOSSES so that all three
registries are importable from a single, stable location.
"""

from __future__ import annotations

try:
    from mmengine.registry import Registry

    from scene_mining.models.scene_miner import SCENE_MINING_MODELS

    SCENE_MINING_DATASETS = Registry("scene_mining_datasets")
    SCENE_MINING_LOSSES = Registry("scene_mining_losses")

    _HAS_MMENGINE = True

except ImportError:
    SCENE_MINING_MODELS = None   # type: ignore[assignment]
    SCENE_MINING_DATASETS = None  # type: ignore[assignment]
    SCENE_MINING_LOSSES = None    # type: ignore[assignment]
    _HAS_MMENGINE = False


__all__ = [
    "SCENE_MINING_MODELS",
    "SCENE_MINING_DATASETS",
    "SCENE_MINING_LOSSES",
]
