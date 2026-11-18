#!/usr/bin/env python3
"""Unified cross-dataset sample contract for WACV multi-domain training.

The model needs to consume nuScenes and Waymo frames through ONE code
path.  The two sources have different native layouts (different camera
counts, different label availability, different text), so we define a
single standardized per-sample dictionary that every source must emit,
and wrap each native dataset in a thin adapter that maps onto it.

Standardized sample (every subclass returns EXACTLY these six fields,
plus a non-supervisory ``meta`` side-channel for provenance):

    camera_images : FloatTensor (6, 3, 448, 800)   — 6-view ring, /255
    bev_features  : dict
        intrinsics  : FloatTensor (6, 3, 3)
        extrinsics  : FloatTensor (6, 4, 4)         — ego→cam
        image_hw    : LongTensor  (2,)
        camera_valid: BoolTensor  (6,)              — False for padded views
    text_queries  : dict
        input_ids      : LongTensor (128,)
        attention_mask : LongTensor (128,)
    trajectory_gt : dict
        waypoints     : FloatTensor (8, 6, 2)       — ego frame, 6×0.5 s
        waypoint_mask : BoolTensor  (8, 6)
        agent_mask    : BoolTensor  (8,)
    risk_scalar   : FloatTensor ()                  — NaN where no GT exists
    semantic_tags : dict
        values     : FloatTensor (20,)              — multi-hot nuPrompt
        valid_mask : BoolTensor  (20,)              — which tags are supervised
    meta          : dict  {dataset, scene_token, risk_valid, frame_idx}

Design rationale (why a contract instead of two bespoke loaders):
  * The collate function can stack heterogeneous-origin samples into one
    batch ONLY if every field has a fixed dtype/shape regardless of
    source — that is exactly what this contract guarantees.
  * Missing supervision (e.g. Waymo has no physics-risk GT, and only 5/20
    tags) is represented by VALIDITY MASKS, not by absent keys.  The
    trainer masks the corresponding loss terms; shapes never vary.
  * ``bev_features`` carries the calibration the in-model BEV encoder
    lifts cameras with — neither dataset ships a precomputed BEV grid,
    so this is the honest content of that field.

Both subclasses use COMPOSITION (wrap the existing native dataset) rather
than copy-pasting its logic, so the frozen BMVC v8 datasets stay
untouched and battle-tested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

STANDARD_KEYS = (
    "camera_images", "bev_features", "text_queries",
    "trajectory_gt", "risk_scalar", "semantic_tags",
)


class UniversalDriveDataset(Dataset, ABC):
    """Abstract base enforcing the standardized cross-domain sample.

    Subclasses implement ``__len__`` and ``__getitem__``; the latter must
    return a dict produced by ``self._standardize(...)`` so shape/type
    invariants are validated centrally.
    """

    # Fixed structural constants — IDENTICAL across all sources so batches
    # of mixed origin stack cleanly.
    N_CAMERAS = 6
    IMAGE_HW = (448, 800)
    N_AGENTS = 8
    N_WAYPOINTS = 6
    N_TAGS = 20
    MAX_TEXT_LEN = 128

    dataset_name: str = "abstract"

    # Class-level tokenizer cache (loaded once, shared by all instances).
    _tokenizer = None
    _tokenizer_path: str | None = None

    # ── shared helpers ────────────────────────────────────────────────
    @classmethod
    def _get_tokenizer(cls, tokenizer_path: str):
        if cls._tokenizer is None or cls._tokenizer_path != tokenizer_path:
            from transformers import AutoTokenizer
            cls._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True)
            cls._tokenizer_path = tokenizer_path
        return cls._tokenizer

    def _tokenize(self, text: str) -> dict:
        """Tokenize free text to the fixed (MAX_TEXT_LEN,) contract."""
        if self._tokenizer is None:
            # No tokenizer configured → zero-pad fallback (shape preserved).
            ids = torch.zeros(self.MAX_TEXT_LEN, dtype=torch.long)
            return {"input_ids": ids, "attention_mask": ids.clone()}
        enc = self._tokenizer(
            text, max_length=self.MAX_TEXT_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"].squeeze(0).long(),
                "attention_mask": enc["attention_mask"].squeeze(0).long()}

    def _standardize(
        self, *,
        camera_images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        camera_valid: torch.Tensor,
        text_queries: dict,
        waypoints: torch.Tensor,
        waypoint_mask: torch.Tensor,
        agent_mask: torch.Tensor,
        risk_scalar: torch.Tensor,
        tag_values: torch.Tensor,
        tag_valid: torch.Tensor,
        scene_token: str,
        frame_idx: int = -1,
    ) -> dict:
        """Assemble + validate one standardized sample."""
        sample = {
            "camera_images": camera_images.float(),
            "bev_features": {
                "intrinsics": intrinsics.float(),
                "extrinsics": extrinsics.float(),
                "image_hw": torch.tensor(self.IMAGE_HW, dtype=torch.long),
                "camera_valid": camera_valid.bool(),
            },
            "text_queries": text_queries,
            "trajectory_gt": {
                "waypoints": waypoints.float(),
                "waypoint_mask": waypoint_mask.bool(),
                "agent_mask": agent_mask.bool(),
            },
            "risk_scalar": risk_scalar.float(),
            "semantic_tags": {
                "values": tag_values.float(),
                "valid_mask": tag_valid.bool(),
            },
            "meta": {
                "dataset": self.dataset_name,
                "scene_token": str(scene_token),
                "risk_valid": bool(torch.isfinite(risk_scalar).item()),
                "frame_idx": int(frame_idx),
            },
        }
        self.validate_sample(sample)
        return sample

    @classmethod
    def validate_sample(cls, s: dict) -> None:
        """Hard assertions: the six fields exist with the contract shapes."""
        missing = [k for k in STANDARD_KEYS if k not in s]
        assert not missing, f"sample missing standardized fields: {missing}"
        assert tuple(s["camera_images"].shape) == (
            cls.N_CAMERAS, 3, *cls.IMAGE_HW), \
            f"camera_images {tuple(s['camera_images'].shape)} != " \
            f"{(cls.N_CAMERAS, 3, *cls.IMAGE_HW)}"
        bev = s["bev_features"]
        assert tuple(bev["intrinsics"].shape) == (cls.N_CAMERAS, 3, 3)
        assert tuple(bev["extrinsics"].shape) == (cls.N_CAMERAS, 4, 4)
        tq = s["text_queries"]
        assert tuple(tq["input_ids"].shape) == (cls.MAX_TEXT_LEN,)
        tj = s["trajectory_gt"]
        assert tuple(tj["waypoints"].shape) == (
            cls.N_AGENTS, cls.N_WAYPOINTS, 2)
        assert tuple(tj["waypoint_mask"].shape) == (
            cls.N_AGENTS, cls.N_WAYPOINTS)
        assert s["risk_scalar"].ndim == 0, "risk_scalar must be a 0-d scalar"
        assert tuple(s["semantic_tags"]["values"].shape) == (cls.N_TAGS,)

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict: ...


# ──────────────────────────────────────────────────────────────────────
# nuScenes wrapper — fully supervised (all 20 tags + risk + trajectories)
# ──────────────────────────────────────────────────────────────────────
class NuScenesUniversalDataset(UniversalDriveDataset):
    """Adapter over the existing ``NuScenesFrameDataset`` (composition).

    The native dataset already emits images/intrinsics/extrinsics, the
    tokenized scene description, multi-hot tags, scalar risk, and the
    8×6×2 waypoint GT — so this wrapper is a pure field-remap.  nuScenes
    is the fully-supervised domain: all tags valid, risk valid, all 6
    cameras real.
    """

    dataset_name = "nuscenes"

    def __init__(self, inner, tokenizer_path: str | None = None):
        self._inner = inner
        if tokenizer_path is not None:
            self._get_tokenizer(tokenizer_path)

    def __len__(self) -> int:
        return len(self._inner)

    def _empty_like(self, scene_token: str = "EMPTY") -> dict:
        """Correctly-shaped, fully-masked sample for label-less frames."""
        z = torch.zeros
        return self._standardize(
            camera_images=z(self.N_CAMERAS, 3, *self.IMAGE_HW),
            intrinsics=torch.eye(3).repeat(self.N_CAMERAS, 1, 1),
            extrinsics=torch.eye(4).repeat(self.N_CAMERAS, 1, 1),
            camera_valid=torch.zeros(self.N_CAMERAS, dtype=torch.bool),
            text_queries=self._tokenize(""),
            waypoints=z(self.N_AGENTS, self.N_WAYPOINTS, 2),
            waypoint_mask=torch.zeros(self.N_AGENTS, self.N_WAYPOINTS, dtype=torch.bool),
            agent_mask=torch.zeros(self.N_AGENTS, dtype=torch.bool),
            risk_scalar=torch.tensor(float("nan")),
            tag_values=z(self.N_TAGS),
            tag_valid=torch.zeros(self.N_TAGS, dtype=torch.bool),
            scene_token=scene_token, frame_idx=-1,
        )

    def __getitem__(self, idx: int) -> dict:
        raw = self._inner[idx]
        if not raw:                       # native dataset returns {} for
            return self._empty_like()     # label-less frames — keep shapes.

        risk = raw.get("risk", torch.tensor(float("nan")))
        risk = risk if torch.is_tensor(risk) else torch.tensor(float(risk))
        return self._standardize(
            camera_images=raw["images"],
            intrinsics=raw["intrinsics"],
            extrinsics=raw["extrinsics"],
            camera_valid=torch.ones(self.N_CAMERAS, dtype=torch.bool),
            text_queries={
                "input_ids": raw["input_ids"].long(),
                "attention_mask": raw["attention_mask"].long(),
            },
            waypoints=raw["waypoints"],
            waypoint_mask=raw["waypoint_mask"],
            agent_mask=raw["agent_mask"],
            risk_scalar=risk.reshape(()),
            tag_values=raw["tags"],
            # nuScenes is fully supervised → every tag is valid.
            tag_valid=torch.ones(self.N_TAGS, dtype=torch.bool),
            scene_token=raw.get("scene_token", "nusc"),
        )


# ──────────────────────────────────────────────────────────────────────
# Waymo wrapper — partial supervision (5/20 tags, NO risk GT)
# ──────────────────────────────────────────────────────────────────────
class WaymoUniversalDataset(UniversalDriveDataset):
    """Adapter over ``WaymoZeroShotDataset`` (the Stage-B adapter).

    Native to the universal contract:
      * 5→6 camera padding is already handled by the adapter (CAM_BACK
        zero-filled), so ``camera_images`` is (6,3,448,800) directly.
      * Waymo perception has NO physics-risk GT → ``risk_scalar`` = NaN
        and ``meta.risk_valid`` = False (risk loss masked for Waymo).
      * Only 5/20 tags are Waymo-evaluable → ``semantic_tags.valid_mask``
        marks them; the other 15 are masked in the tag loss.
      * ``text_queries`` = a neutral prompt (Waymo has no scene caption),
        tokenized with the SAME tokenizer/length as nuScenes so shapes
        match.
    """

    dataset_name = "waymo"

    def __init__(self, extracted_root, tokenizer_path: str | None = None,
                 prompt: str = "Describe this driving scene:"):
        from scene_mining.adapters.waymo_adapter import WaymoZeroShotDataset
        self._inner = WaymoZeroShotDataset(extracted_root)
        self._prompt = prompt
        if tokenizer_path is not None:
            self._get_tokenizer(tokenizer_path)
        self._cached_text = self._tokenize(prompt)   # constant across frames

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        raw = self._inner[idx]              # adapter sample (6-cam already)
        return self._standardize(
            camera_images=raw["images"],
            intrinsics=raw["intrinsics"],
            extrinsics=raw["extrinsics"],
            camera_valid=raw["camera_valid"],
            text_queries={k: v.clone() for k, v in self._cached_text.items()},
            waypoints=raw["traj_gt"],
            waypoint_mask=raw["waypoint_mask"],
            agent_mask=raw["agent_mask"],
            # No physics-risk GT on Waymo perception → NaN (loss-masked).
            risk_scalar=torch.tensor(float("nan")),
            tag_values=raw["tag_target"],
            tag_valid=raw["tag_evaluable"],
            scene_token=raw["scene_token"],
            frame_idx=int(raw.get("frame_idx", -1)),
        )


# ──────────────────────────────────────────────────────────────────────
# Collation — stacks mixed-origin samples (the whole point of the contract)
# ──────────────────────────────────────────────────────────────────────
def universal_collate(batch: list[dict]) -> dict:
    """Stack a list of standardized samples into a batched dict.

    Every tensor field has a fixed shape across sources, so a plain
    stack works; ``meta`` is collected as a list (per-item provenance).
    """
    def stack(key, sub=None):
        if sub is None:
            return torch.stack([b[key] for b in batch], 0)
        return torch.stack([b[key][sub] for b in batch], 0)

    return {
        "camera_images": stack("camera_images"),
        "bev_features": {
            "intrinsics":   stack("bev_features", "intrinsics"),
            "extrinsics":   stack("bev_features", "extrinsics"),
            "image_hw":     stack("bev_features", "image_hw"),
            "camera_valid": stack("bev_features", "camera_valid"),
        },
        "text_queries": {
            "input_ids":      stack("text_queries", "input_ids"),
            "attention_mask": stack("text_queries", "attention_mask"),
        },
        "trajectory_gt": {
            "waypoints":     stack("trajectory_gt", "waypoints"),
            "waypoint_mask": stack("trajectory_gt", "waypoint_mask"),
            "agent_mask":    stack("trajectory_gt", "agent_mask"),
        },
        "risk_scalar": stack("risk_scalar"),
        "semantic_tags": {
            "values":     stack("semantic_tags", "values"),
            "valid_mask": stack("semantic_tags", "valid_mask"),
        },
        "meta": [b["meta"] for b in batch],
    }
