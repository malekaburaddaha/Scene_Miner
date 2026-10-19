#!/usr/bin/env python3
"""Stage B — Waymo perception → SceneMiner-v8 input adapter (zero-shot).

Consumes the neutral PNG/npz dump produced by
``scripts/waymo_extract_perception.py`` (Stage A) and re-packages each
frame into the exact tensor contract the frozen v8 model expects, so we
can measure the nuScenes→Waymo domain-shift gap without retraining.

Two domain bridges are implemented here, both **deliberately
conservative and explicitly documented** so the zero-shot numbers stay
honest:

  1. Camera-rig bridge (5 Waymo cams → 6 nuScenes slots).  Waymo has no
     rear camera and its SIDE cameras face ~90° outward, not rearward,
     so we map the three forward cameras 1:1 and place the two side
     cameras in the rear-side slots, leaving ``CAM_BACK`` zero-filled —
     reusing the dataset's own missing-camera convention
     (zeros image + eye(3)/eye(4)).  The geometric mismatch of the side
     cameras is itself part of the measured domain gap.

  2. Taxonomy bridge (Waymo stats → 20-class nuPrompt).  Waymo's
     perception ``Context.Stats`` only carries reliable signal for a
     subset of our 20 tags (time-of-day, weather, and 3 object classes).
     ``build_tag_target`` returns BOTH a 20-d target AND a 20-d
     "evaluable" mask; the eval must compute mAP only over the
     evaluable tags, never over the 15 tags Waymo cannot annotate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from scene_mining.datasets.nuscenes_dataset import (
    NUSCENES_CAMERA_NAMES, NUSCENES_IMAGE_SIZE, _resize_with_intrinsics,
)

# Canonical 20-class nuPrompt taxonomy (order must match the tag head).
NUPROMPT_TAG_NAMES = [
    "scene_dark", "scene_rainy", "scene_construction_zone", "scene_intersection",
    "scene_parking_lot", "scene_highway", "scene_school_zone", "scene_residential",
    "object_pedestrian", "object_bicycle", "object_motorcycle", "object_truck",
    "object_bus", "object_car", "object_emergency_vehicle", "object_animal",
    "behavior_jaywalking", "behavior_running", "behavior_erratic", "behavior_static",
]
TAG_INDEX = {n: i for i, n in enumerate(NUPROMPT_TAG_NAMES)}

# ── Camera-rig bridge ──────────────────────────────────────────────────
# nuScenes slots: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
#                 CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
# Waymo provides:  FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
WAYMO_TO_NUSCENES_CAM = {
    "CAM_FRONT":       "FRONT",
    "CAM_FRONT_LEFT":  "FRONT_LEFT",
    "CAM_FRONT_RIGHT": "FRONT_RIGHT",
    "CAM_BACK":        None,          # Waymo has no rear camera → zero-fill
    "CAM_BACK_LEFT":   "SIDE_LEFT",   # geometric approximation (90° vs 135°)
    "CAM_BACK_RIGHT":  "SIDE_RIGHT",
}
WAYMO_CAM_ORDER = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]


# ── Taxonomy bridge ────────────────────────────────────────────────────
# Each entry: nuPrompt tag → (rule, is_evaluable).  ``is_evaluable`` marks
# whether Waymo perception stats carry a reliable signal for the tag; the
# eval computes mAP only over evaluable tags.
def build_tag_target(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Map a Waymo frame's ``meta.json`` to a (20,) target + (20,) mask."""
    target = np.zeros(len(NUPROMPT_TAG_NAMES), dtype=np.float32)
    evaluable = np.zeros(len(NUPROMPT_TAG_NAMES), dtype=bool)

    tod = (meta.get("time_of_day") or "").lower()
    weather = (meta.get("weather") or "").lower()
    counts = meta.get("laser_object_counts", {}) or {}

    # Environmental tags with a reliable Waymo signal.
    evaluable[TAG_INDEX["scene_dark"]] = True
    target[TAG_INDEX["scene_dark"]] = 1.0 if "night" in tod else 0.0

    evaluable[TAG_INDEX["scene_rainy"]] = True
    target[TAG_INDEX["scene_rainy"]] = 1.0 if "rain" in weather else 0.0

    # Object tags with a reliable Waymo class signal.
    evaluable[TAG_INDEX["object_car"]] = True
    target[TAG_INDEX["object_car"]] = 1.0 if counts.get("VEHICLE", 0) > 0 else 0.0

    evaluable[TAG_INDEX["object_pedestrian"]] = True
    target[TAG_INDEX["object_pedestrian"]] = (
        1.0 if counts.get("PEDESTRIAN", 0) > 0 else 0.0)

    # Waymo CYCLIST ≈ our bicycle (two-wheeled rider).  Evaluable.
    evaluable[TAG_INDEX["object_bicycle"]] = True
    target[TAG_INDEX["object_bicycle"]] = (
        1.0 if counts.get("CYCLIST", 0) > 0 else 0.0)

    # The remaining 15 tags have NO reliable Waymo perception signal
    # (construction_zone, intersection, parking_lot, highway, school_zone,
    #  residential, motorcycle, truck, bus, emergency_vehicle, animal, and
    #  all four behavior tags) → left non-evaluable.
    return target, evaluable


def _load_six_camera_stack(frame_dir: Path):
    """Return (images (6,3,H,W), intrinsics (6,3,3), extrinsics (6,4,4),
    valid_mask (6,)) following the nuScenes contract, with Waymo's missing
    rear camera zero-filled exactly as NuScenesDataset does."""
    import cv2
    calib = np.load(frame_dir / "calib.npz", allow_pickle=True)
    waymo_order = list(calib["camera_order"])
    K_by_name = {nm: calib["intrinsics"][i]
                 for i, nm in enumerate(waymo_order)}
    ext_by_name = {nm: calib["extrinsics_cam2veh"][i]
                   for i, nm in enumerate(waymo_order)}

    H, W = NUSCENES_IMAGE_SIZE
    images, intrinsics, extrinsics, valid = [], [], [], []
    for slot in NUSCENES_CAMERA_NAMES:
        waymo_cam = WAYMO_TO_NUSCENES_CAM.get(slot)
        png = frame_dir / f"cam_{waymo_cam}.png" if waymo_cam else None
        if waymo_cam is None or png is None or not png.exists():
            images.append(np.zeros((3, H, W), dtype=np.float32))
            intrinsics.append(np.eye(3, dtype=np.float32))
            extrinsics.append(np.eye(4, dtype=np.float32))
            valid.append(False)
            continue
        bgr = cv2.imread(str(png), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        K = K_by_name[waymo_cam].astype(np.float32)
        img_resized, K_adj = _resize_with_intrinsics(rgb, K, H, W)
        img_f = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
        images.append(img_f)
        intrinsics.append(K_adj)
        # Waymo extrinsic is camera→vehicle; invert to vehicle→camera to
        # match the nuScenes ego2cam convention.
        ext_c2v = ext_by_name[waymo_cam].astype(np.float32)
        extrinsics.append(np.linalg.inv(ext_c2v).astype(np.float32))
        valid.append(True)
    return (np.stack(images), np.stack(intrinsics),
            np.stack(extrinsics), np.asarray(valid))


def _build_motion_gt(frame_dir: Path, n_agents: int = 8, n_steps: int = 6):
    """Future agent (x,y) in capture-ego frame → (N,6,2) gt + masks.

    Keeps the ``n_agents`` highest-confidence agents (here: closest to the
    ego, a reasonable salience proxy), matching the model's fixed N=8 slot
    structure.  Missing future steps are masked out for mask-aware ADE/FDE.
    """
    boxes = np.load(frame_dir / "boxes.npz", allow_pickle=True)
    future = json.loads((frame_dir / "future.json").read_text())
    ids = list(boxes["ids"])
    centers = boxes["centers"]              # (M,3) vehicle frame

    # Rank agents by planar distance to ego (closest first).
    if len(ids) == 0:
        traj = np.zeros((n_agents, n_steps, 2), dtype=np.float32)
        return traj, np.zeros(n_agents, bool), np.zeros((n_agents, n_steps), bool)
    dist = np.linalg.norm(centers[:, :2], axis=1)
    order = np.argsort(dist)[:n_agents]

    traj = np.zeros((n_agents, n_steps, 2), dtype=np.float32)
    agent_mask = np.zeros(n_agents, dtype=bool)
    wp_mask = np.zeros((n_agents, n_steps), dtype=bool)
    for slot, ai in enumerate(order):
        aid = ids[ai]
        steps = future.get(aid if isinstance(aid, str) else aid.decode()
                           if isinstance(aid, bytes) else str(aid), [])
        agent_mask[slot] = True
        for s in range(min(n_steps, len(steps))):
            pt = steps[s]
            if pt is None:
                continue
            traj[slot, s] = [float(pt[0]), float(pt[1])]
            wp_mask[slot, s] = True
    return traj, agent_mask, wp_mask


class WaymoZeroShotDataset(Dataset):
    """Torch Dataset over Stage-A extracted Waymo frames, yielding samples
    in the SceneMiner-v8 input contract (zero-shot — no Waymo training)."""

    def __init__(self, extracted_root: str | Path):
        self.root = Path(extracted_root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.frames = manifest

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        rec = self.frames[idx]
        fdir = self.root / rec["rel_dir"]
        meta = json.loads((fdir / "meta.json").read_text())

        images, intr, extr, cam_valid = _load_six_camera_stack(fdir)
        tag_target, tag_evaluable = build_tag_target(meta)
        traj_gt, agent_mask, wp_mask = _build_motion_gt(fdir)

        return {
            # Backbone input (matches NuScenesDataset packing).
            "images":       torch.from_numpy(images),          # (6,3,448,800)
            "intrinsics":   torch.from_numpy(intr),            # (6,3,3)
            "extrinsics":   torch.from_numpy(extr),            # (6,4,4)
            "camera_valid": torch.from_numpy(cam_valid),       # (6,)
            "camera_names": NUSCENES_CAMERA_NAMES,
            # Pseudo-label for retrieval: the Waymo segment is the "scene".
            "scene_token":  rec["segment"],
            # Tagging GT + evaluability.
            "tag_target":   torch.from_numpy(tag_target),      # (20,)
            "tag_evaluable": torch.from_numpy(tag_evaluable),  # (20,) bool
            # Motion GT (mask-aware).
            "traj_gt":      torch.from_numpy(traj_gt),         # (8,6,2)
            "agent_mask":   torch.from_numpy(agent_mask),      # (8,)
            "waypoint_mask": torch.from_numpy(wp_mask),        # (8,6)
            # Provenance.
            "segment":      rec["segment"],
            "frame_idx":    rec["frame_idx"],
        }
