"""nuScenes dataset adapter.

Wraps the nuscenes-devkit API to provide a BaseSceneDataset-compatible
interface for the scene mining pipeline.

nuScenes specifics:
- 6 ring cameras: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
  CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
- 2 Hz annotation rate (sample rate)
- Instance tokens used as track IDs
- 1000 scenes (~700 train, ~150 val, ~150 test) in the full dataset

Usage::

    from scene_mining.datasets.nuscenes_dataset import NuScenesDataset
    ds = NuScenesDataset(dataroot="/data/nuscenes", version="v1.0-trainval", split="val")
    log_ids = ds.get_log_ids()   # scene tokens
    tracks  = ds.get_annotations(log_ids[0])
    inp     = ds.adapt_to_backbone_input(log_ids[0], timestamp_ns)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from scene_mining.datasets.av2_dataset import BackboneInput
from scene_mining.datasets.base_dataset import BaseSceneDataset

# nuScenes canonical camera order
NUSCENES_CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Target (H, W) after resize for backbone input
NUSCENES_IMAGE_SIZE = (448, 800)

# NuScenes 10 detection classes mapped to semantic occupancy class indices.
# Background = 0. Maps top-level category prefix to class index.
# Used by the 3D-box rasterisation fallback (no panoptic data).
NUSCENES_DETECTION_CLASSES: dict[str, int] = {
    "car": 1,
    "truck": 2,
    "construction_vehicle": 3,
    "bus": 4,
    "trailer": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "pedestrian": 8,
    "traffic_cone": 9,
    "barrier": 10,
}
N_OCC_CLASSES = 11  # 0=background + 10 object classes (3D-box fallback)

# ── Panoptic / lidarseg semantic mapping ──────────────────────────────────────
# NuScenes panoptic encodes each LiDAR point as uint16:
#   category_index = value // 1000   (0..31)
#   instance_id    = value  % 1000
#
# We remap 32 panoptic categories → 17 BEV classes:
#   0         background / noise / unknown
#   1–10      dynamic objects (same as NUSCENES_DETECTION_CLASSES above)
#   11–16     static / flat classes (only available with panoptic labels)
#
# Category index → class index  (len=32, index = panoptic category_index)
PANOPTIC_CATEGORY_TO_CLASS: list[int] = [
    0,   # 0  noise                          → background
    0,   # 1  animal                         → background
    8,   # 2  human.pedestrian.adult         → pedestrian
    8,   # 3  human.pedestrian.child         → pedestrian
    8,   # 4  human.pedestrian.construction_worker → pedestrian
    8,   # 5  human.pedestrian.personal_mobility   → pedestrian
    8,   # 6  human.pedestrian.police_officer       → pedestrian
    8,   # 7  human.pedestrian.stroller      → pedestrian
    8,   # 8  human.pedestrian.wheelchair    → pedestrian
    10,  # 9  movable_object.barrier         → barrier
    0,   # 10 movable_object.debris          → background
    0,   # 11 movable_object.pushable_pullable → background
    9,   # 12 movable_object.trafficcone     → traffic_cone
    0,   # 13 static_object.bicycle_rack     → background
    7,   # 14 vehicle.bicycle                → bicycle
    4,   # 15 vehicle.bus.bendy              → bus
    4,   # 16 vehicle.bus.rigid              → bus
    1,   # 17 vehicle.car                    → car
    3,   # 18 vehicle.construction           → construction_vehicle
    1,   # 19 vehicle.emergency.ambulance    → car (treat as vehicle)
    1,   # 20 vehicle.emergency.police       → car (treat as vehicle)
    6,   # 21 vehicle.motorcycle             → motorcycle
    5,   # 22 vehicle.trailer               → trailer
    2,   # 23 vehicle.truck                  → truck
    11,  # 24 flat.driveable_surface         → driveable_surface
    12,  # 25 flat.other                     → flat_other
    13,  # 26 flat.sidewalk                  → sidewalk
    14,  # 27 flat.terrain                   → terrain
    15,  # 28 static.manmade                 → manmade
    0,   # 29 static.other                   → background
    16,  # 30 static.vegetation              → vegetation
    0,   # 31 vehicle.ego                    → background (self)
]
N_OCC_CLASSES_PANOPTIC = 17  # 0=background + 10 dynamic + 6 static


class NuScenesDataset(BaseSceneDataset):
    """nuScenes dataset adapter.

    Args:
        dataroot: Path to the nuScenes dataset root (contains ``samples/``).
        version: Dataset version string, e.g. ``"v1.0-trainval"``,
            ``"v1.0-mini"``.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        cameras: Subset of cameras to use. Defaults to all 6.
        verbose: Passed to the NuScenes constructor.
    """

    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-trainval",
        split: str = "val",
        cameras: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise ImportError(
                "nuscenes-devkit is required for NuScenesDataset. "
                "Install with: pip install nuscenes-devkit"
            ) from exc

        self.dataroot = Path(dataroot)
        self.version = version
        self.split = split
        self.cameras = cameras or NUSCENES_CAMERA_NAMES

        self._nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=verbose)
        self._scene_tokens: list[str] = self._filter_scene_tokens(split)

    # ── BaseSceneDataset implementation ───────────────────────────────────────

    def get_log_ids(self) -> list[str]:
        """Return scene tokens for the requested split."""
        return self._scene_tokens

    def get_annotations(self, log_id: str) -> "TrackCollection":
        """Load nuScenes annotations for *log_id* (a scene token).

        Returns:
            TrackCollection using instance tokens as track IDs.
        """
        from scene_mining.tracks.track_loader import TrackLoader

        return TrackLoader.from_nuscenes(log_id, self._nusc)

    def adapt_to_backbone_input(
        self,
        log_id: str,
        timestamp_ns: int,
        frame: dict[str, Any] | None = None,
    ) -> BackboneInput:
        """Build a BackboneInput for *log_id* at *timestamp_ns*.

        Finds the nuScenes sample closest to *timestamp_ns*, loads images
        for all configured cameras, resizes them, and returns a packed
        BackboneInput.

        Args:
            log_id: nuScenes scene token.
            timestamp_ns: Target timestamp in nanoseconds.
            frame: If provided, use these pre-loaded images instead of
                reading from disk.

        Returns:
            BackboneInput with stacked images [N_cam, 3, H, W].
        """
        sample_token = self._find_closest_sample(log_id, timestamp_ns)
        sample = self._nusc.get("sample", sample_token)

        images: list[np.ndarray] = []
        intrinsics_list: list[np.ndarray] = []
        extrinsics_list: list[np.ndarray] = []
        original_hw: list[tuple[int, int]] = []

        for cam in self.cameras:
            cam_token = sample["data"].get(cam)
            if cam_token is None:
                # Camera not present — fill with zeros
                H, W = NUSCENES_IMAGE_SIZE
                images.append(np.zeros((3, H, W), dtype=np.float32))
                intrinsics_list.append(np.eye(3, dtype=np.float32))
                extrinsics_list.append(np.eye(4, dtype=np.float32))
                original_hw.append((H, W))
                continue

            cam_data = self._nusc.get("sample_data", cam_token)
            img, K, ego2cam, orig_h, orig_w = self._load_camera(cam_data, frame, cam)

            target_h, target_w = NUSCENES_IMAGE_SIZE
            img_resized, K_adj = _resize_with_intrinsics(img, K, target_h, target_w)

            img_f = img_resized.astype(np.float32) / 255.0
            img_chw = np.transpose(img_f, (2, 0, 1))  # CHW

            images.append(img_chw)
            intrinsics_list.append(K_adj)
            extrinsics_list.append(ego2cam)
            original_hw.append((orig_h, orig_w))

        actual_ts = int(sample["timestamp"]) * 1000  # µs → ns

        return BackboneInput(
            log_id=log_id,
            timestamp_ns=actual_ts,
            images=np.stack(images, axis=0),
            intrinsics=np.stack(intrinsics_list, axis=0),
            extrinsics=np.stack(extrinsics_list, axis=0),
            camera_names=self.cameras,
            original_hw=original_hw,
        )

    # ── Temporal helpers (Phase 1: temporal BEV context) ──────────────────────

    def get_ego_pose_matrix(self, sample_token: str) -> np.ndarray:
        """Return the ego→world 4×4 matrix at *sample_token*'s LIDAR_TOP time.

        nuScenes stores ego poses per-sample_data, not per-sample.  We key off
        LIDAR_TOP because that's also where the GT waypoints are computed
        from.  Quaternion order in nuScenes is (w, x, y, z).
        """
        from pyquaternion import Quaternion
        sample = self._nusc.get("sample", sample_token)
        lidar_data = self._nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self._nusc.get("ego_pose", lidar_data["ego_pose_token"])
        q = Quaternion(ego_pose["rotation"])  # (w, x, y, z) → Quaternion
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = q.rotation_matrix.astype(np.float32)
        T[:3, 3] = np.asarray(ego_pose["translation"], dtype=np.float32)
        return T

    def adapt_to_temporal_input(
        self,
        log_id: str,
        sample_token: str,
        n_prev: int = 2,
    ) -> tuple[list[BackboneInput], np.ndarray]:
        """Return ``n_prev + 1`` BackboneInputs (oldest first, current last)
        and a stacked ego-pose array of shape ``(T, 4, 4)`` aligned with them.

        Walks the nuScenes ``prev`` field; if a prev sample doesn't exist
        (start of scene), the earliest available sample is repeated so the
        output always has exactly ``n_prev + 1`` entries.
        """
        # Walk back via prev.  Result ordered current → oldest, then reversed.
        tokens: list[str] = [sample_token]
        cur = sample_token
        for _ in range(n_prev):
            prev_token = self._nusc.get("sample", cur)["prev"]
            if prev_token:
                tokens.append(prev_token)
                cur = prev_token
            else:
                # Pad by repeating the earliest available frame.
                tokens.append(cur)
        tokens = list(reversed(tokens))     # oldest → current

        inputs: list[BackboneInput] = []
        poses: list[np.ndarray] = []
        for tok in tokens:
            sample = self._nusc.get("sample", tok)
            ts_ns = int(sample["timestamp"]) * 1000
            inputs.append(self.adapt_to_backbone_input(log_id, ts_ns))
            poses.append(self.get_ego_pose_matrix(tok))
        return inputs, np.stack(poses, axis=0).astype(np.float32)

    def get_lidar_timestamps(self, log_id: str) -> list[int]:
        """Return sample timestamps (ns) for the scene."""
        tokens = self._get_sample_tokens(log_id)
        timestamps = []
        for tok in tokens:
            s = self._nusc.get("sample", tok)
            timestamps.append(int(s["timestamp"]) * 1000)  # µs → ns
        return sorted(timestamps)

    # ── nuScenes-specific helpers ──────────────────────────────────────────────

    def _filter_scene_tokens(self, split: str) -> list[str]:
        """Return scene tokens for the requested split using nuscenes-devkit splits."""
        try:
            from nuscenes.utils.splits import create_splits_scenes
            split_map = create_splits_scenes()
            # split_map maps split name → list of scene names
            scene_names_in_split = set(split_map.get(split, []))
            tokens = [
                s["token"]
                for s in self._nusc.scene
                if s["name"] in scene_names_in_split
            ]
        except Exception:
            # Fallback: return all scenes
            tokens = [s["token"] for s in self._nusc.scene]
        return sorted(tokens)

    def _get_sample_tokens(self, scene_token: str) -> list[str]:
        """Return ordered sample tokens for a scene."""
        scene = self._nusc.get("scene", scene_token)
        tokens = []
        tok = scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            s = self._nusc.get("sample", tok)
            tok = s["next"]
        return tokens

    def _find_closest_sample(self, scene_token: str, timestamp_ns: int) -> str:
        """Return the sample token closest to *timestamp_ns*."""
        tokens = self._get_sample_tokens(scene_token)
        best_tok = tokens[0]
        best_diff = float("inf")
        for tok in tokens:
            s = self._nusc.get("sample", tok)
            diff = abs(int(s["timestamp"]) * 1000 - timestamp_ns)
            if diff < best_diff:
                best_diff = diff
                best_tok = tok
        return best_tok

    def _load_camera(
        self,
        cam_data: dict,
        frame: dict[str, Any] | None,
        cam_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Load image + calibration for one camera sample_data record.

        Returns:
            (image HWC uint8, K 3x3, ego2cam 4x4, orig_h, orig_w)
        """
        import cv2

        # Calibration
        cs = self._nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        K = np.array(cs["camera_intrinsic"], dtype=np.float32)
        cam2ego_t = np.array(cs["translation"], dtype=np.float32)
        cam2ego_q = cs["rotation"]  # [w, x, y, z]
        cam2ego = _quat_trans_to_SE3(cam2ego_q, cam2ego_t)
        ego2cam = np.linalg.inv(cam2ego).astype(np.float32)

        # Image
        if frame is not None and cam_name in frame:
            img = frame[cam_name]
        else:
            img_path = self.dataroot / cam_data["filename"]
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        return img, K, ego2cam, orig_h, orig_w


# ── Private helpers ────────────────────────────────────────────────────────────


def _quat_to_rotation_matrix(quat_wxyz: list[float]) -> np.ndarray:
    """Build a 3×3 rotation matrix from quaternion [w, x, y, z]."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)],
    ], dtype=np.float32)


def _quat_trans_to_SE3(quat_wxyz: list[float], translation: np.ndarray) -> np.ndarray:
    """Build a 4×4 SE3 matrix from quaternion [w,x,y,z] + translation."""
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)],
    ], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


def _resize_with_intrinsics(
    img: np.ndarray,
    K: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aspect-ratio-preserving resize + padding with intrinsic adjustment."""
    import cv2

    orig_h, orig_w = img.shape[:2]
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_right = target_w - new_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0,
    )

    K_adj = K.copy().astype(np.float32)
    K_adj[0, 0] *= scale
    K_adj[1, 1] *= scale
    K_adj[0, 2] = K[0, 2] * scale + pad_left
    K_adj[1, 2] = K[1, 2] * scale + pad_top

    return padded, K_adj


# ── PyTorch Dataset wrapper ────────────────────────────────────────────────────

class NuScenesFrameDataset(Dataset):
    """torch.utils.data.Dataset over individual nuScenes frames.

    Each item is one (scene, sample) pair, returning a dict compatible with
    SyntheticDataset so the same DataLoader / train_step code works for both.

    Loads supervision labels from four pre-computed JSON/PKL files produced
    by scripts/ (Tasks 1-5):
        - ``data/labels/nuprompt_tags.json``        — 20-dim binary tags per scene
        - ``data/labels/gt_risk_scores.json``       — physics-based risk scalar per scene
        - ``data/labels/gt_future_waypoints.pkl``   — per-agent 3 s future trajectories
        - ``data/labels/scene_descriptions.json``   — text descriptions for Motion Head / VLM

    If a labels file is missing, the dataset falls back to legacy keyword-based
    pseudo-labels (for the 2 keyword-matched tags/risk) or zero-padding
    (waypoints, text).

    Args:
        dataroot: Path to nuScenes root (contains ``samples/``).
        version: Dataset version, e.g. ``"v1.0-mini"``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        n_tags: Number of tag logits expected by the model head.
        seq_len: Max length of tokenized text / fallback dummy length.
        vocab_size: Vocab ceiling for random ``input_ids`` (legacy fallback).
        verbose: Passed to NuScenes constructor.
        load_occ_gt: Whether to load semantic BEV occupancy (for stage 0).
        labels_dir: Path to directory holding the 4 label files.
            Defaults to ``<dataroot>/../labels`` or ``./data/labels``.
        tokenizer_path: HuggingFace tokenizer path for scene descriptions.
            If None, returns zero-padded ``input_ids`` (legacy behaviour).
        n_agents: Max agents per keyframe for MotionHead supervision.
        n_waypoints: Future trajectory horizon (default 6 = 3 s at 0.5 s).
        max_text_len: Max tokens for scene descriptions.
    """

    # Legacy tag index → (keyword_group, risk_weight) used only when
    # nuprompt_tags.json / gt_risk_scores.json are not available.
    _TAG_KEYWORDS: list[tuple[list[str], float]] = [
        (["night", "dark", "evening", "dusk", "dawn"],          0.3),   # 0: night
        (["rain", "wet", "puddle", "rainy"],                     0.4),   # 1: rain
        (["highway", "freeway", "motorway", "overpass"],         0.0),   # 2: highway
        (["intersection", "junction", "crossroad", "crossing"],  0.2),   # 3: intersection
        (["turn", "turning", "curve", "roundabout"],             0.2),   # 4: turning
        (["traffic", "congestion", "busy", "heavy"],             0.1),   # 5: heavy traffic
        (["pedestrian", "cyclist", "bike", "walker"],            0.2),   # 6: vulnerable road user
        (["construction", "roadwork", "crane", "barrier"],       0.3),   # 7: construction zone
        (["parking", "parked", "lot", "garage"],                 0.0),   # 8: parking
        (["fog", "mist", "cloud", "overcast"],                   0.3),   # 9: low visibility
    ]

    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-mini",
        split: str = "train",
        n_tags: int = 20,
        seq_len: int = 8,
        vocab_size: int = 512,
        verbose: bool = False,
        load_occ_gt: bool = False,
        bev_h: int = 50,
        bev_w: int = 50,
        pc_range: list[float] | None = None,
        n_occ_classes: int = N_OCC_CLASSES,
        labels_dir: str | Path | None = None,
        tokenizer_path: str | None = None,
        n_agents: int = 8,
        n_waypoints: int = 6,
        max_text_len: int = 128,
        n_temporal_frames: int = 1,
    ) -> None:
        self._scene_ds = NuScenesDataset(
            dataroot=dataroot,
            version=version,
            split=split,
            verbose=verbose,
        )
        self.n_tags = n_tags
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.load_occ_gt = load_occ_gt
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.n_occ_classes = n_occ_classes
        self.n_agents = n_agents
        self.n_waypoints = n_waypoints
        self.max_text_len = max_text_len
        self.n_temporal_frames = max(1, int(n_temporal_frames))

        # Build panoptic token map: lidar_token → npz_path
        # The panoptic.json entries have sample_data_token = LIDAR_TOP token.
        self._panoptic_map: dict[str, Path] = {}
        _panoptic_candidates = [
            Path(dataroot) / f"nuScenes-panoptic-{version}",
            Path(dataroot) / "nuScenes-panoptic-v1.0-all",
        ]
        for _panoptic_root in _panoptic_candidates:
            _panoptic_json = _panoptic_root / version / "panoptic.json"
            if _panoptic_json.exists():
                import json as _json
                with open(_panoptic_json) as _f:
                    _entries = _json.load(_f)
                for _e in _entries:
                    self._panoptic_map[_e["sample_data_token"]] = _panoptic_root / _e["filename"]
                break

        # ── Load pre-computed supervision labels (Tasks 1-5) ─────────────
        self._labels_dir = self._resolve_labels_dir(dataroot, labels_dir)
        self._nuprompt_tags = self._load_json(self._labels_dir / "nuprompt_tags.json")
        self._gt_risk_scores = self._load_json(self._labels_dir / "gt_risk_scores.json")
        self._gt_waypoints = self._load_pickle(self._labels_dir / "gt_future_waypoints.pkl")
        self._scene_descriptions = self._load_json(self._labels_dir / "scene_descriptions.json")

        # ── Pre-tokenize scene descriptions if tokenizer provided ────────
        # This is done once at init (not per __getitem__) since descriptions
        # are per-scene, not per-keyframe.
        self._tokenized_descriptions: dict[str, dict[str, torch.Tensor]] = {}
        self._tokenizer = None
        if tokenizer_path is not None and self._scene_descriptions:
            self._tokenizer = self._load_tokenizer(tokenizer_path)
            if self._tokenizer is not None:
                for scene_token, entry in self._scene_descriptions.items():
                    text = entry["description"] if isinstance(entry, dict) else str(entry)
                    enc = self._tokenizer(
                        text,
                        max_length=max_text_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    self._tokenized_descriptions[scene_token] = {
                        "input_ids": enc["input_ids"].squeeze(0),
                        "attention_mask": enc["attention_mask"].squeeze(0),
                    }

        # Pre-build flat index: list of (scene_token, timestamp_ns)
        # Also cache per-scene pseudo-labels and build (log_id, ts) → sample_token map
        # so __getitem__ can look up waypoints by nuScenes sample_token.
        self._index: list[tuple[str, int]] = []
        self._scene_tags: dict[str, torch.Tensor] = {}   # scene_token → (n_tags,)
        self._scene_risk: dict[str, float] = {}           # scene_token → scalar
        self._sample_token_map: dict[tuple[str, int], str] = {}  # (log_id, ts) → sample_token

        nusc = self._scene_ds._nusc

        # Build (log_id, timestamp_ns) → sample_token lookup.
        # Note: get_lidar_timestamps returns nanoseconds (us * 1000), but
        # nuScenes sample_data.timestamp is microseconds. Multiply to match.
        for scene in nusc.scene:
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = nusc.get("sample", sample_token)
                sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                ts_ns = int(sd["timestamp"]) * 1000
                self._sample_token_map[(scene["token"], ts_ns)] = sample_token
                sample_token = sample["next"] if sample["next"] else None

        for log_id in self._scene_ds.get_log_ids():
            for ts in self._scene_ds.get_lidar_timestamps(log_id):
                self._index.append((log_id, ts))
            if log_id not in self._scene_tags:
                self._scene_tags[log_id] = self._resolve_tags(log_id, n_tags, nusc)
                self._scene_risk[log_id] = self._resolve_risk(log_id, nusc)

    # ── Label resolution helpers ─────────────────────────────────────────

    @staticmethod
    def _resolve_labels_dir(dataroot: str | Path, override: str | Path | None) -> Path:
        """Find the labels directory — explicit override, sibling to dataroot, or cwd default."""
        if override is not None:
            return Path(override)
        candidates = [
            Path(dataroot).parent / "labels",     # data/nuscenes → data/labels
            Path("data/labels"),                   # cwd fallback
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]  # non-existent path; _load_json will silently skip

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            return {}
        import json as _json
        with open(path) as f:
            return _json.load(f)

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        if not path.exists():
            return {}
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_tokenizer(tokenizer_path: str):
        """Load a HF tokenizer. Returns None on failure (falls back to legacy)."""
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to load tokenizer from {tokenizer_path}: {e}. "
                "Falling back to zero input_ids.",
                stacklevel=2,
            )
            return None

    def _resolve_tags(self, scene_token: str, n_tags: int, nusc) -> torch.Tensor:
        """Tag vector from nuprompt_tags.json, else keyword fallback."""
        if scene_token in self._nuprompt_tags:
            tag_list = self._nuprompt_tags[scene_token]
            tags = torch.tensor(tag_list, dtype=torch.float32)
            if tags.shape[0] < n_tags:
                tags = torch.cat([tags, torch.zeros(n_tags - tags.shape[0])])
            return tags[:n_tags]
        # Legacy fallback
        scene = nusc.get("scene", scene_token)
        desc = scene.get("description", "").lower()
        tags_vec, _ = self._labels_from_description(desc, n_tags)
        return tags_vec

    def _resolve_risk(self, scene_token: str, nusc) -> float:
        """Risk scalar from gt_risk_scores.json, else keyword fallback."""
        if scene_token in self._gt_risk_scores:
            return float(self._gt_risk_scores[scene_token])
        scene = nusc.get("scene", scene_token)
        desc = scene.get("description", "").lower()
        _, risk = self._labels_from_description(desc, self.n_tags)
        return risk

    def _load_waypoints(
        self, scene_token: str, sample_token: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (waypoints, waypoint_mask, agent_mask) for a keyframe.

        Returns:
            waypoints:      (n_agents, n_waypoints, 2) float32 ego-centric (x, y)
            waypoint_mask:  (n_agents, n_waypoints) uint8 — 1 = valid future step
            agent_mask:     (n_agents,) bool — True = real agent, False = padding
        """
        waypoints = torch.zeros(self.n_agents, self.n_waypoints, 2, dtype=torch.float32)
        waypoint_mask = torch.zeros(self.n_agents, self.n_waypoints, dtype=torch.uint8)
        agent_mask = torch.zeros(self.n_agents, dtype=torch.bool)

        scene_data = self._gt_waypoints.get(scene_token, {})
        keyframe_data = scene_data.get(sample_token, {})
        if not keyframe_data:
            return waypoints, waypoint_mask, agent_mask

        # Rank agents by proximity to ego at t=0 (first valid waypoint closest to origin).
        # This prioritises nearby agents, which matter most for collision risk and
        # matches how the Q-Former's limited N_agents slots should be allocated.
        ranked = []
        for ann_token, entry in keyframe_data.items():
            wp = entry["waypoints"]  # (6, 2) numpy
            msk = entry["mask"]      # (6,) numpy
            if msk.sum() == 0:
                continue
            # Use first valid waypoint's distance to origin (ego) as priority key
            first_valid = int(msk.argmax())
            dist = float((wp[first_valid] ** 2).sum() ** 0.5)
            ranked.append((dist, wp, msk))

        ranked.sort(key=lambda x: x[0])
        n_use = min(len(ranked), self.n_agents)
        for i in range(n_use):
            _, wp, msk = ranked[i]
            waypoints[i] = torch.from_numpy(wp).float()
            waypoint_mask[i] = torch.from_numpy(msk).to(torch.uint8)
            agent_mask[i] = True

        return waypoints, waypoint_mask, agent_mask

    def _labels_from_description(
        self, desc: str, n_tags: int
    ) -> tuple[torch.Tensor, float]:
        """Legacy: keyword-match scene description → binary tag vector + risk scalar.

        Kept for backward-compatibility fallback when nuprompt_tags.json /
        gt_risk_scores.json are unavailable.
        """
        tags = torch.zeros(n_tags, dtype=torch.float32)
        risk = 0.0
        for i, (keywords, risk_w) in enumerate(self._TAG_KEYWORDS):
            if i >= n_tags:
                break
            if any(kw in desc for kw in keywords):
                tags[i] = 1.0
                risk = max(risk, risk_w)
        return tags, risk

    def _load_occ_gt(self, log_id: str, timestamp_ns: int) -> torch.Tensor:
        """Load semantic BEV occupancy GT.

        Uses panoptic per-point labels when available (majority-vote projection),
        falling back to 3D bounding-box rasterisation otherwise.

        Returns:
            occ_gt (bev_h, bev_w) int64 tensor, values in [0, n_occ_classes-1].
        """
        nusc = self._scene_ds._nusc
        sample_token = self._scene_ds._find_closest_sample(log_id, timestamp_ns)
        sample = nusc.get("sample", sample_token)

        lidar_token = sample["data"].get("LIDAR_TOP")
        if lidar_token is None:
            return torch.zeros(self.bev_h, self.bev_w, dtype=torch.int64)

        if lidar_token in self._panoptic_map:
            return self._load_occ_gt_panoptic(lidar_token)
        return self._load_occ_gt_boxes(sample, lidar_token)

    def _load_occ_gt_panoptic(self, lidar_token: str) -> torch.Tensor:
        """Project panoptic per-point labels to BEV via majority vote.

        Each LiDAR point is mapped to a BEV cell; the most frequent
        non-background class wins per cell (background wins only if no
        foreground votes exist).

        Returns:
            (bev_h, bev_w) int64 tensor with values in [0, n_occ_classes-1].
        """
        nusc = self._scene_ds._nusc
        sd = nusc.get("sample_data", lidar_token)

        # ── Load LiDAR points in sensor frame (N, 5): x y z intensity ring ──
        lidar_path = self._scene_ds.dataroot / sd["filename"]
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
        xyz_lidar = points[:, :3]  # (N, 3)

        # ── Lidar → ego transform ─────────────────────────────────────────────
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        R_l2e = _quat_to_rotation_matrix(cs["rotation"]).astype(np.float64)
        t_l2e = np.array(cs["translation"], dtype=np.float64)
        xyz_ego = (R_l2e @ xyz_lidar.T.astype(np.float64) + t_l2e[:, None]).T  # (N, 3)

        # ── Load panoptic labels ───────────────────────────────────────────────
        npz_path = self._panoptic_map[lidar_token]
        panoptic_data = np.load(str(npz_path))["data"]          # (N,) uint16
        cat_idx = (panoptic_data // 1000).astype(np.int32)      # category 0..31
        # Map panoptic category → BEV class, clamp to configured n_classes
        cat_lut = np.array(PANOPTIC_CATEGORY_TO_CLASS, dtype=np.int32)
        class_per_pt = cat_lut[np.clip(cat_idx, 0, len(cat_lut) - 1)]
        class_per_pt = np.clip(class_per_pt, 0, self.n_occ_classes - 1)

        # ── Project to BEV grid ───────────────────────────────────────────────
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        cell_w = (xmax - xmin) / self.bev_w
        cell_h = (ymax - ymin) / self.bev_h

        col_idx = ((xyz_ego[:, 0] - xmin) / cell_w).astype(np.int32)  # x → col
        row_idx = ((xyz_ego[:, 1] - ymin) / cell_h).astype(np.int32)  # y → row

        valid = (
            (col_idx >= 0) & (col_idx < self.bev_w) &
            (row_idx >= 0) & (row_idx < self.bev_h)
        )
        col_idx = col_idx[valid]
        row_idx = row_idx[valid]
        cls_v   = class_per_pt[valid]

        # ── Majority vote per cell ─────────────────────────────────────────────
        n_cls = self.n_occ_classes
        occ_counts = np.zeros((self.bev_h, self.bev_w, n_cls), dtype=np.int32)
        np.add.at(occ_counts, (row_idx, col_idx, cls_v), 1)

        # Foreground = any class > 0
        if n_cls > 1:
            fg_counts = occ_counts[:, :, 1:]           # (H, W, n_fg)
            has_fg    = fg_counts.sum(axis=2) > 0
            occ       = np.where(has_fg, fg_counts.argmax(axis=2) + 1, 0)
        else:
            occ = (occ_counts[:, :, 0] > 0).astype(np.int64)

        return torch.from_numpy(occ.astype(np.int64))

    def _load_occ_gt_boxes(self, sample: dict, lidar_token: str) -> torch.Tensor:
        """Fallback: rasterise 3D bounding boxes into BEV (dynamic objects only).

        Returns:
            (bev_h, bev_w) int64 tensor, values in [0, 10].
        """
        nusc = self._scene_ds._nusc
        sd = nusc.get("sample_data", lidar_token)
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])

        t_e2g = np.array(ego_pose["translation"], dtype=np.float64)
        R_e2g = _quat_to_rotation_matrix(ego_pose["rotation"]).astype(np.float64)
        R_g2e = R_e2g.T

        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        cell_w = (xmax - xmin) / self.bev_w
        cell_h = (ymax - ymin) / self.bev_h

        xs = xmin + (np.arange(self.bev_w) + 0.5) * cell_w
        ys = ymin + (np.arange(self.bev_h) + 0.5) * cell_h
        grid_x, grid_y = np.meshgrid(xs, ys)

        occ = np.zeros((self.bev_h, self.bev_w), dtype=np.int64)

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cat_parts = ann["category_name"].split(".")
            class_idx = 0
            for part in cat_parts:
                if part in NUSCENES_DETECTION_CLASSES:
                    class_idx = NUSCENES_DETECTION_CLASSES[part]
                    break
            if class_idx == 0:
                continue

            c_global = np.array(ann["translation"], dtype=np.float64)
            c_ego = R_g2e @ (c_global - t_e2g)
            cx, cy = c_ego[0], c_ego[1]

            w_box, l_box, _ = ann["size"]
            half_w, half_l = w_box / 2.0, l_box / 2.0

            R_box_global = _quat_to_rotation_matrix(ann["rotation"]).astype(np.float64)
            R_box_ego = R_g2e @ R_box_global
            yaw = float(np.arctan2(R_box_ego[1, 0], R_box_ego[0, 0]))

            dx = grid_x - cx
            dy = grid_y - cy
            cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
            local_x = cos_y * dx - sin_y * dy
            local_y = sin_y * dx + cos_y * dy

            inside = (np.abs(local_x) <= half_l) & (np.abs(local_y) <= half_w)
            occ[inside] = class_idx

        return torch.from_numpy(occ)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        log_id, timestamp_ns = self._index[idx]
        inp = self._scene_ds.adapt_to_backbone_input(log_id, timestamp_ns)

        images = torch.from_numpy(inp.images).float()        # (N_cam, 3, H, W)
        intrinsics = torch.from_numpy(inp.intrinsics).float()
        extrinsics = torch.from_numpy(inp.extrinsics).float()

        _, _, H, W = images.shape
        tags = self._scene_tags[log_id]
        risk = torch.tensor(self._scene_risk[log_id], dtype=torch.float32)

        # ── Text tokens: real tokenised scene description, else zero-pad ──
        if log_id in self._tokenized_descriptions:
            tok = self._tokenized_descriptions[log_id]
            input_ids = tok["input_ids"]
            attention_mask = tok["attention_mask"]
        else:
            input_ids = torch.zeros(self.seq_len, dtype=torch.long)
            attention_mask = torch.zeros(self.seq_len, dtype=torch.long)

        # ── Future waypoints for MotionHead supervision ──────────────────
        # Look up sample_token by (log_id, timestamp); timestamp from
        # get_lidar_timestamps matches sample_data LIDAR_TOP timestamps.
        sample_token = self._sample_token_map.get((log_id, timestamp_ns), "")
        waypoints, waypoint_mask, agent_mask = self._load_waypoints(log_id, sample_token)

        item = {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "image_hw": (H, W),
            "tags": tags,
            "risk": risk,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "scene_token": log_id,
            "sample_token": sample_token,
            "waypoints": waypoints,
            "waypoint_mask": waypoint_mask,
            "agent_mask": agent_mask,
        }

        # ── Phase-1 temporal stack ────────────────────────────────────────
        # Emit T = n_temporal_frames frames (oldest first, current last) plus
        # their ego→world poses for ego-motion compensation in the model.
        # Only fired when explicitly enabled and we have a valid sample token;
        # default n_temporal_frames=1 keeps the legacy single-frame behaviour.
        if self.n_temporal_frames > 1 and sample_token:
            inputs, ego_poses = self._scene_ds.adapt_to_temporal_input(
                log_id, sample_token, n_prev=self.n_temporal_frames - 1,
            )
            item["temporal_images"]     = torch.from_numpy(
                np.stack([bi.images     for bi in inputs], axis=0)).float()
            item["temporal_intrinsics"] = torch.from_numpy(
                np.stack([bi.intrinsics for bi in inputs], axis=0)).float()
            item["temporal_extrinsics"] = torch.from_numpy(
                np.stack([bi.extrinsics for bi in inputs], axis=0)).float()
            item["temporal_ego_poses"]  = torch.from_numpy(ego_poses).float()

        if self.load_occ_gt:
            item["occ_gt"] = self._load_occ_gt(log_id, timestamp_ns)

        return item


# ── mmdet3d-pkl-based dataset ──────────────────────────────────────────────────

class NuScenesMMDet3DDataset(Dataset):
    """NuScenes dataset backed by pre-generated mmdet3d annotation pkl files.

    Faster than ``NuScenesFrameDataset`` because the pkl caches all annotation
    queries done at init time; nuscenes-devkit is only used at construction to
    verify the pkl exists.

    Generate pkl files with::

        python scripts/prepare_nuscenes_data.py \\
            --dataroot ./data/nuscenes --version v1.0-mini

    Args:
        dataroot: Path to NuScenes dataset root.
        pkl_path: Path to the annotation pkl file generated by prepare_nuscenes_data.py.
        image_size: Target (H, W) for all camera images.
        load_occ_gt: Whether to generate semantic BEV occupancy GT.
        bev_h: BEV grid height (number of rows).
        bev_w: BEV grid width (number of columns).
        pc_range: [xmin, ymin, zmin, xmax, ymax, zmax] in ego metres.
        n_occ_classes: Number of semantic occupancy classes (including background).
        cameras: Which cameras to include (default: all 6).
    """

    def __init__(
        self,
        dataroot: str | Path,
        pkl_path: str | Path,
        image_size: tuple[int, int] = NUSCENES_IMAGE_SIZE,
        load_occ_gt: bool = False,
        bev_h: int = 50,
        bev_w: int = 50,
        pc_range: list[float] | None = None,
        n_occ_classes: int = N_OCC_CLASSES,
        cameras: list[str] | None = None,
        seq_len: int = 8,
    ) -> None:
        import pickle as _pkl
        self.dataroot = Path(dataroot)
        self.image_size = image_size
        self.load_occ_gt = load_occ_gt
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.n_occ_classes = n_occ_classes
        self.cameras = cameras or NUSCENES_CAMERA_NAMES
        self.seq_len = seq_len

        with open(pkl_path, "rb") as f:
            self._infos: list[dict] = _pkl.load(f)

    def __len__(self) -> int:
        return len(self._infos)

    def __getitem__(self, idx: int) -> dict:
        info = self._infos[idx]

        images_list, intrinsics_list, extrinsics_list = [], [], []
        tgt_H, tgt_W = self.image_size

        for cam_name in self.cameras:
            cam_info = info["cams"].get(cam_name)
            if cam_info is None:
                # Pad with zeros if camera missing
                images_list.append(np.zeros((3, tgt_H, tgt_W), dtype=np.float32))
                intrinsics_list.append(np.eye(3, dtype=np.float32))
                extrinsics_list.append(np.eye(4, dtype=np.float32))
                continue

            img_path = self.dataroot / cam_info["data_path"]
            img, K_adj = _load_and_resize_image(str(img_path), self.image_size,
                                                cam_info["cam_intrinsic"])
            # sensor2ego is cam→ego; model expects ego→cam (i.e. inverse)
            s2e = cam_info["sensor2ego"].astype(np.float32)   # (4,4)
            e2c = np.linalg.inv(s2e)                          # ego→cam

            images_list.append(img)
            intrinsics_list.append(K_adj)
            extrinsics_list.append(e2c)

        images     = torch.from_numpy(np.stack(images_list)).float()     # (N, 3, H, W)
        intrinsics = torch.from_numpy(np.stack(intrinsics_list)).float() # (N, 3, 3)
        extrinsics = torch.from_numpy(np.stack(extrinsics_list)).float() # (N, 4, 4)

        item = dict(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=(tgt_H, tgt_W),
            tags=torch.zeros(20, dtype=torch.float32),
            risk=torch.tensor(0.0),
            input_ids=torch.zeros(self.seq_len, dtype=torch.long),
            scene_token=info.get("scene_token", info.get("token", "")),
        )

        if self.load_occ_gt:
            item["occ_gt"] = self._boxes_to_semantic_bev(info)

        return item

    def _boxes_to_semantic_bev(self, info: dict) -> torch.Tensor:
        """Rasterise gt_boxes from the pkl info dict to a semantic BEV map."""
        gt_boxes = info["gt_boxes"]   # (N, 9): cx,cy,cz, dx,dy,dz, yaw, vx,vy
        gt_names = info["gt_names"]   # list[str]

        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        cell_w = (xmax - xmin) / self.bev_w
        cell_h = (ymax - ymin) / self.bev_h

        xs = xmin + (np.arange(self.bev_w) + 0.5) * cell_w
        ys = ymin + (np.arange(self.bev_h) + 0.5) * cell_h
        grid_x, grid_y = np.meshgrid(xs, ys)

        occ = np.zeros((self.bev_h, self.bev_w), dtype=np.int64)

        for i, (box, name) in enumerate(zip(gt_boxes, gt_names)):
            class_idx = NUSCENES_DETECTION_CLASSES.get(name, 0)
            if class_idx == 0:
                continue
            cx, cy = float(box[0]), float(box[1])
            dx, dy = float(box[3]), float(box[4])   # dx=length, dy=width
            yaw = float(box[6])

            dx_cell, dy_cell = dx / 2.0, dy / 2.0
            cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
            local_x = cos_y * (grid_x - cx) - sin_y * (grid_y - cy)
            local_y = sin_y * (grid_x - cx) + cos_y * (grid_y - cy)
            inside = (np.abs(local_x) <= dx_cell) & (np.abs(local_y) <= dy_cell)
            occ[inside] = class_idx

        return torch.from_numpy(occ)


def _load_and_resize_image(
    path: str,
    target_hw: tuple[int, int],
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image, resize to target, return (chw float32, adjusted K)."""
    from PIL import Image as PILImage
    img = PILImage.open(path).convert("RGB")
    orig_w, orig_h = img.size
    tgt_H, tgt_W = target_hw
    img = img.resize((tgt_W, tgt_H), PILImage.BILINEAR)
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0   # (3,H,W)
    # Adjust intrinsic for resize
    K = intrinsic.copy().astype(np.float32)
    K[0, 0] *= tgt_W / orig_w
    K[1, 1] *= tgt_H / orig_h
    K[0, 2] *= tgt_W / orig_w
    K[1, 2] *= tgt_H / orig_h
    return arr, K
