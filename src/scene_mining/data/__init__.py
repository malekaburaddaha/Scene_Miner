"""WACV extension — unified multi-dataset training infrastructure.

Blends nuScenes and Waymo perception into one training stream behind a
single standardized sample contract (``UniversalDriveDataset``) plus a
domain-balancing sampler (``MultinomialDatasetSampler``).

Kept in a dedicated sub-package so the WACV work is isolated from the
frozen BMVC v8 code paths under ``scene_mining.datasets``.
"""
from scene_mining.data.universal_dataset import (
    UniversalDriveDataset, NuScenesUniversalDataset, WaymoUniversalDataset,
    universal_collate, STANDARD_KEYS,
)
from scene_mining.data.samplers import (
    MultinomialDatasetSampler, DistributedMultinomialSampler,
)

__all__ = [
    "UniversalDriveDataset", "NuScenesUniversalDataset", "WaymoUniversalDataset",
    "universal_collate", "STANDARD_KEYS",
    "MultinomialDatasetSampler", "DistributedMultinomialSampler",
]
