"""Config-driven dataset / dataloader builders.

Provides helpers to construct train/val/test DataLoaders with sensible
defaults and fallbacks (SyntheticDataset) so entrypoints can be config-driven.

All dataset construction goes through ``build_dataset(cfg, split)`` which
reads ``cfg.data.type`` to determine the dataset class.  This is the single
source of truth — no other code should instantiate dataset classes directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler


def _is_distributed() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


# ── Dataset construction ───────────────────────────────────────────────────────


def build_dataset(cfg, split: str | None = None) -> Optional[torch.utils.data.Dataset]:
    """Build a Dataset from cfg.data, dispatching on cfg.data.type.

    Args:
        cfg: mmengine Config (or any namespace) with a ``.data`` attribute
             containing at minimum a ``type`` key plus dataset-specific keys.
        split: Override cfg.data.split if provided.  Useful for building a
               val loader from a train config without modifying the config.

    Returns:
        Dataset instance, or None if construction fails (after printing a warning).
    """
    data = cfg.data
    dataset_type = data.get("type", None) if hasattr(data, "get") else getattr(data, "type", None)

    effective_split = split if split is not None else (
        data.get("split", "train") if hasattr(data, "get") else getattr(data, "split", "train")
    )

    n_tags = cfg.model.get("n_tags", 20) if (hasattr(cfg, "model") and hasattr(cfg.model, "get")) else (
        getattr(getattr(cfg, "model", None), "n_tags", 20) or 20
    )

    def _get(key, default=None):
        return data.get(key, default) if hasattr(data, "get") else getattr(data, key, default)

    try:
        if dataset_type in (None, "NuScenesDataset"):
            from scene_mining.datasets.nuscenes_dataset import NuScenesFrameDataset
            dataroot = Path(_get("dataroot", "./data/nuscenes"))
            if not dataroot.exists():
                raise FileNotFoundError(f"NuScenes root not found: {dataroot}")
            from scene_mining.datasets.nuscenes_dataset import N_OCC_CLASSES
            return NuScenesFrameDataset(
                dataroot=dataroot,
                version=_get("version", "v1.0-mini"),
                split=effective_split,
                n_tags=n_tags,
                load_occ_gt=_get("load_occ_gt", False),
                bev_h=_get("bev_h", 50),
                bev_w=_get("bev_w", 50),
                pc_range=_get("pc_range", None),
                n_occ_classes=_get("n_occ_classes", N_OCC_CLASSES),
                labels_dir=_get("labels_dir", None),
                tokenizer_path=_get("tokenizer_path", None),
                n_agents=_get("n_agents", 8),
                n_waypoints=_get("n_waypoints", 6),
                max_text_len=_get("max_text_len", 128),
                n_temporal_frames=_get("n_temporal_frames", 1),
            )

        elif dataset_type == "NuScenesMMDet3DDataset":
            from scene_mining.datasets.nuscenes_dataset import NuScenesMMDet3DDataset, N_OCC_CLASSES
            dataroot = Path(_get("dataroot", "./data/nuscenes"))
            # Determine pkl path: e.g. data/nuscenes/nuscenes_infos_train.pkl
            pkl_path = _get("pkl_path", None)
            if pkl_path is None:
                pkl_path = dataroot / f"nuscenes_infos_{effective_split}.pkl"
            else:
                pkl_path = Path(pkl_path)
            if not pkl_path.exists():
                raise FileNotFoundError(
                    f"mmdet3d pkl not found: {pkl_path}\n"
                    "Run: python scripts/prepare_nuscenes_data.py"
                )
            return NuScenesMMDet3DDataset(
                dataroot=dataroot,
                pkl_path=pkl_path,
                load_occ_gt=_get("load_occ_gt", False),
                bev_h=_get("bev_h", 50),
                bev_w=_get("bev_w", 50),
                pc_range=_get("pc_range", None),
                n_occ_classes=_get("n_occ_classes", N_OCC_CLASSES),
            )

        elif dataset_type == "AV2Dataset":
            from scene_mining.datasets.av2_dataset import AV2FrameDataset
            data_dir = Path(_get("data_dir", "./data/av2"))
            if not data_dir.exists():
                raise FileNotFoundError(f"AV2 root not found: {data_dir}")
            return AV2FrameDataset(
                data_dir=data_dir,
                split=effective_split,
                max_dist_m=_get("max_dist_m", 50.0),
                n_tags=n_tags,
                load_occ_gt=_get("load_occ_gt", False),
                bev_h=_get("bev_h", 50),
                bev_w=_get("bev_w", 50),
            )

        elif dataset_type == "SyntheticDataset":
            from scene_mining.datasets.synthetic_dataset import SyntheticDataset
            batch_size = (
                cfg.train.get("batch_size", 4) if (hasattr(cfg, "train") and hasattr(cfg.train, "get"))
                else getattr(getattr(cfg, "train", None), "batch_size", 4) or 4
            )
            return SyntheticDataset(
                n_samples=_get("n_samples", max(batch_size * 10, 64)),
                n_cams=_get("n_cams", 7),
                image_h=_get("image_h", 64),
                image_w=_get("image_w", 64),
                n_tags=n_tags,
                seq_len=_get("seq_len", 8),
                vocab_size=_get("vocab_size", 512),
            )

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type!r}")

    except Exception as exc:
        print(f"[builder] Could not build {dataset_type!r} ({exc}); "
              f"falling back to SyntheticDataset")
        try:
            from scene_mining.datasets.synthetic_dataset import SyntheticDataset
            return SyntheticDataset(n_samples=64, n_cams=7, image_h=64, image_w=64, n_tags=n_tags)
        except Exception:
            return None


def build_val_dataloader_for_eval(
    cfg,
    split: str = "val",
    batch_size: int = 1,
    num_workers: int = 2,
) -> Optional[DataLoader]:
    """Build a single-sample validation DataLoader for evaluation hooks and scripts.

    Forces ``load_occ_gt=True`` so occupancy metrics can be computed.
    The ``split`` parameter overrides ``cfg.data.split`` so a training config
    can produce a val loader without modification.

    Used by:
    - OccEvalHook in scripts/train.py
    - build_val_dataloader() in scripts/eval_occ.py
    """
    import types

    # Build a shallow copy of cfg with load_occ_gt forced on and split overridden.
    data_dict = {}
    data_src = cfg.data
    if hasattr(data_src, "keys"):
        for k in data_src.keys():
            data_dict[k] = data_src[k]
    elif hasattr(data_src, "__dict__"):
        data_dict = dict(data_src.__dict__)
    else:
        # mmengine ConfigDict / Config — iterate items
        try:
            data_dict = dict(data_src)
        except Exception:
            data_dict = {k: getattr(data_src, k) for k in dir(data_src) if not k.startswith("_")}

    data_dict["load_occ_gt"] = True
    data_dict["split"] = split

    cfg_copy = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")})
    cfg_copy.data = types.SimpleNamespace(**data_dict)

    dataset = build_dataset(cfg_copy, split=split)
    if dataset is None:
        return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ── DataLoader builders ────────────────────────────────────────────────────────


def build_dataloader(cfg, split: str = "train") -> Optional[DataLoader]:
    """Build a PyTorch DataLoader from ``cfg`` for ``split``.

    Returns None if a dataset could not be built (useful for optional val).
    """
    train_cfg = cfg.train if hasattr(cfg, "train") else None
    data_cfg  = cfg.data  if hasattr(cfg, "data")  else None

    def _get_int(obj, key, default):
        if obj is None:
            return default
        return int(obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default))

    batch_size  = _get_int(train_cfg, "batch_size",  _get_int(data_cfg, "batch_size",  4))
    num_workers = _get_int(train_cfg, "num_workers",  _get_int(data_cfg, "num_workers", 4))

    dataset = build_dataset(cfg, split=split)
    if dataset is None:
        return None

    sampler = DistributedSampler(dataset, shuffle=(split == "train")) if _is_distributed() else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == "train"),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=(split == "train"),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def build_train_dataloader(cfg) -> Optional[DataLoader]:
    return build_dataloader(cfg, split="train")


def build_val_dataloader(cfg) -> Optional[DataLoader]:
    return build_dataloader(cfg, split="val")
