"""Argoverse 2 (AV2) dataset adapter.

Provides:
- Per-log iteration over camera images + calibration
- Ground-truth TrackCollection loading
- HD vector map access (ArgoverseStaticMap)
- BackboneInput construction for the deep learning pipeline

AV2 specifics:
- 7 ring cameras: ring_front_center (portrait 2048×1550), all others landscape
- 10 Hz annotation rate (lidar timestamps)
- UUID string track IDs
- 30 annotated object categories
- HD vector map with lane graph, crosswalks, stop signs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# AV2 camera names in the canonical sensor order
AV2_CAMERA_NAMES = [
    "ring_front_center",
    "ring_front_right",
    "ring_front_left",
    "ring_side_right",
    "ring_side_left",
    "ring_rear_right",
    "ring_rear_left",
]

# Portrait camera — needs special handling for resize
AV2_PORTRAIT_CAMERAS = {"ring_front_center"}

# Target sizes for backbone input (longest-edge resize, then center-crop)
AV2_LANDSCAPE_SIZE = (448, 800)   # (H, W) for landscape ring cameras
AV2_PORTRAIT_SIZE = (800, 448)    # (H, W) for ring_front_center


@dataclass
class CameraCalibration:
    """Camera intrinsic + extrinsic for a single AV2 camera."""
    camera_name: str
    intrinsics: np.ndarray          # (3, 3) K matrix
    extrinsics_ego2cam: np.ndarray  # (4, 4) SE3: ego-vehicle → camera
    image_width: int
    image_height: int

    @property
    def extrinsics_cam2ego(self) -> np.ndarray:
        """Inverse transform: camera → ego-vehicle frame."""
        return np.linalg.inv(self.extrinsics_ego2cam)


@dataclass
class BackboneInput:
    """Packed multi-camera input for the SceneMiner backbone.

    Images are already resized/padded to uniform spatial dimensions per
    camera group (landscape vs portrait). Calibrations are correspondingly
    adjusted.
    """
    log_id: str
    timestamp_ns: int
    images: np.ndarray           # (N_cam, 3, H, W) float32 [0,1]
    intrinsics: np.ndarray       # (N_cam, 3, 3)
    extrinsics: np.ndarray       # (N_cam, 4, 4) ego2cam
    camera_names: list[str]
    # Original image dimensions before resize (for bbox reprojection)
    original_hw: list[tuple[int, int]]


class AV2Dataset:
    """Argoverse 2 sensor dataset adapter.

    Args:
        data_dir: Root of the AV2 dataset (contains ``sensor/`` subdirs).
        split: One of ``"train"``, ``"val"``, ``"test"``.
        max_dist_m: Pre-filter track annotations beyond this distance from
            the ego vehicle (0 = no filtering).

    Example::

        ds = AV2Dataset("/data/av2", split="val")
        log_ids = ds.get_log_ids()
        tracks = ds.get_annotations(log_ids[0])
        hd_map = ds.get_hd_map(log_ids[0])
        for ts, frame in ds.iter_log_frames(log_ids[0]):
            inp = ds.adapt_to_backbone_input(log_ids[0], ts, frame)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "val",
        max_dist_m: float = 0.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_dist_m = max_dist_m
        self._split_dir = self.data_dir / "sensor" / split

    # ── Log enumeration ───────────────────────────────────────────────────────

    def get_log_ids(self) -> list[str]:
        """Return all log IDs for this split."""
        return sorted(p.name for p in self._split_dir.iterdir() if p.is_dir())

    # ── Annotations (Ground Truth) ────────────────────────────────────────────

    def get_annotations(self, log_id: str) -> "TrackCollection":
        """Load AV2 GT annotations as a TrackCollection.

        Uses the av2 Python API to load the per-log annotation parquet.
        """
        from scene_mining.tracks.track_loader import TrackLoader

        ann_path = self._split_dir / log_id / "annotations.feather"
        if ann_path.exists():
            return TrackLoader.from_av2_parquet(log_id, ann_path)

        # Fallback: use av2 API directly
        return TrackLoader.from_av2(log_id, self._split_dir)

    # ── HD Map ────────────────────────────────────────────────────────────────

    def get_hd_map(self, log_id: str) -> Any:
        """Return the AV2 ArgoverseStaticMap for this log.

        Returns:
            av2.map.map_api.ArgoverseStaticMap instance, or None if
            the av2 package is not installed.
        """
        try:
            from av2.map.map_api import ArgoverseStaticMap
            map_dir = self._split_dir / log_id / "map"
            return ArgoverseStaticMap.from_map_dir(map_dir, build_raster=True)
        except Exception:
            return None

    # ── Camera Calibration ────────────────────────────────────────────────────

    def get_camera_calibration(
        self, log_id: str, camera_name: str
    ) -> CameraCalibration:
        """Load intrinsic + extrinsic for a single camera.

        Returns:
            CameraCalibration dataclass.
        """
        try:
            from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
            loader = AV2SensorDataLoader(
                data_dir=self._split_dir, labels_dir=self._split_dir
            )
            cam = loader.get_log_pinhole_camera(log_id=log_id, cam_name=camera_name)
            K = cam.intrinsics.K
            ego2cam = cam.extrinsics  # SE3 or (4,4) array
            if hasattr(ego2cam, "transform_matrix"):
                ego2cam = ego2cam.transform_matrix
            h, w = cam.height_px, cam.width_px
        except Exception:
            # Stub calibration for tests without av2 installed
            K = np.eye(3)
            ego2cam = np.eye(4)
            h, w = (2048, 1550) if camera_name in AV2_PORTRAIT_CAMERAS else (1550, 2048)

        return CameraCalibration(
            camera_name=camera_name,
            intrinsics=np.array(K, dtype=np.float32),
            extrinsics_ego2cam=np.array(ego2cam, dtype=np.float32),
            image_width=w,
            image_height=h,
        )

    def get_all_calibrations(self, log_id: str) -> dict[str, CameraCalibration]:
        """Return calibrations for all 7 ring cameras."""
        return {
            cam: self.get_camera_calibration(log_id, cam)
            for cam in AV2_CAMERA_NAMES
        }

    # ── Frame iteration ───────────────────────────────────────────────────────

    def iter_log_frames(
        self,
        log_id: str,
        cameras: list[str] | None = None,
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Iterate over lidar timestamps yielding (timestamp_ns, {cam: image}).

        Images are returned as uint8 HWC RGB arrays at their native resolution.
        Use ``adapt_to_backbone_input`` to resize for the deep learning pipeline.

        Args:
            log_id: AV2 log ID.
            cameras: List of camera names to load. Defaults to all 7.

        Yields:
            (timestamp_ns, {camera_name: np.ndarray HxWx3 uint8})
        """
        cameras = cameras or AV2_CAMERA_NAMES
        log_dir = self._split_dir / log_id

        try:
            from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
            from av2.utils.io import read_img

            loader = AV2SensorDataLoader(
                data_dir=self._split_dir, labels_dir=self._split_dir
            )
            timestamps = loader.get_ordered_log_lidar_timestamps(log_id=log_id)

            for ts in timestamps:
                frame: dict[str, np.ndarray] = {}
                for cam in cameras:
                    img_path = loader.get_closest_img_fpath(
                        log_id=log_id, cam_name=cam, lidar_timestamp_ns=ts
                    )
                    if img_path is not None and img_path.exists():
                        frame[cam] = read_img(img_path, channel_order="RGB")
                if frame:
                    yield int(ts), frame

        except ImportError:
            # Yield nothing if av2 not installed (for unit tests)
            return

    # ── Backbone input construction ────────────────────────────────────────────

    def adapt_to_backbone_input(
        self,
        log_id: str,
        timestamp_ns: int,
        frame: dict[str, np.ndarray],
        cameras: list[str] | None = None,
    ) -> BackboneInput:
        """Convert a raw camera frame dict to a BackboneInput tensor batch.

        Handles per-camera aspect-ratio-preserving resize (portrait vs
        landscape) and normalisation to float32 [0, 1].

        Args:
            log_id: AV2 log ID.
            timestamp_ns: Frame timestamp.
            frame: Dict mapping camera_name → uint8 HWC RGB image.
            cameras: Subset of cameras to use (default: all 7).

        Returns:
            BackboneInput with stacked images [N_cam, 3, H, W] and
            correspondingly adjusted calibration matrices.
        """
        import cv2

        cameras = cameras or AV2_CAMERA_NAMES
        calibs = self.get_all_calibrations(log_id)

        images: list[np.ndarray] = []
        intrinsics: list[np.ndarray] = []
        extrinsics: list[np.ndarray] = []
        original_hw: list[tuple[int, int]] = []

        for cam in cameras:
            img = frame.get(cam)
            calib = calibs.get(cam)
            if img is None or calib is None:
                # Pad with zeros if camera is missing
                target_h, target_w = _target_size(cam)
                img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                K_adj = np.eye(3, dtype=np.float32)
                if calib is not None:
                    K_adj = calib.intrinsics.copy()
            else:
                orig_h, orig_w = img.shape[:2]
                target_h, target_w = _target_size(cam)
                img_resized, K_adj = _resize_with_intrinsics(
                    img, calib.intrinsics, target_h, target_w
                )
                img = img_resized
                original_hw.append((orig_h, orig_w))

            # HWC uint8 → CHW float32 [0,1]
            img_f = img.astype(np.float32) / 255.0
            img_chw = np.transpose(img_f, (2, 0, 1))  # (3, H, W)
            images.append(img_chw)
            intrinsics.append(K_adj)
            extrinsics.append(calib.extrinsics_ego2cam if calib else np.eye(4, dtype=np.float32))

        return BackboneInput(
            log_id=log_id,
            timestamp_ns=timestamp_ns,
            images=np.stack(images, axis=0),           # (N_cam, 3, H, W)
            intrinsics=np.stack(intrinsics, axis=0),   # (N_cam, 3, 3)
            extrinsics=np.stack(extrinsics, axis=0),   # (N_cam, 4, 4)
            camera_names=cameras,
            original_hw=original_hw if original_hw else [(0, 0)] * len(cameras),
        )


# ── Private helpers ────────────────────────────────────────────────────────────


def _target_size(camera_name: str) -> tuple[int, int]:
    """Return (H, W) target resolution for a given camera."""
    if camera_name in AV2_PORTRAIT_CAMERAS:
        return AV2_PORTRAIT_SIZE
    return AV2_LANDSCAPE_SIZE


def _resize_with_intrinsics(
    img: np.ndarray,
    K: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize image and adjust intrinsics matrix accordingly.

    Performs aspect-ratio-preserving resize to fit within (target_h, target_w),
    then center-crops or pads to exactly (target_h, target_w).

    Returns:
        (resized_image HxWx3, adjusted_K 3x3)
    """
    import cv2

    orig_h, orig_w = img.shape[:2]
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size (center pad)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_right = target_w - new_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )

    # Adjust K: scale focal lengths and principal point
    K_adj = K.copy().astype(np.float32)
    K_adj[0, 0] *= scale              # fx
    K_adj[1, 1] *= scale              # fy
    K_adj[0, 2] = K[0, 2] * scale + pad_left   # cx
    K_adj[1, 2] = K[1, 2] * scale + pad_top    # cy

    return padded, K_adj


# ── torch.utils.data.Dataset wrapper ──────────────────────────────────────────


class AV2FrameDataset:
    """torch.utils.data.Dataset over individual AV2 frames.

    Wraps AV2Dataset similarly to how NuScenesFrameDataset wraps NuScenesDataset,
    providing __len__ and __getitem__ for DataLoader compatibility.

    Args:
        data_dir: Path to AV2 root (contains sensor/train, sensor/val).
        split: "train" or "val".
        max_dist_m: Passed to AV2Dataset for annotation distance filtering.
        n_tags: Tag dimension for the returned tag vector.
        load_occ_gt: Whether to load BEV occupancy GT. AV2 occ GT is not yet
            implemented — returns zeros if True.
        bev_h: BEV grid height (used only for the dummy occ_gt tensor).
        bev_w: BEV grid width (used only for the dummy occ_gt tensor).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        max_dist_m: float = 50.0,
        n_tags: int = 20,
        load_occ_gt: bool = False,
        bev_h: int = 50,
        bev_w: int = 50,
    ) -> None:
        import torch  # local import so the module can be imported without torch

        self._scene_ds = AV2Dataset(data_dir=data_dir, split=split, max_dist_m=max_dist_m)
        self.n_tags = n_tags
        self.load_occ_gt = load_occ_gt
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Build flat (log_id, timestamp_ns) index.
        # TODO: cache frame paths for O(1) lookup rather than re-iterating at __getitem__ time.
        self._index: list[tuple[str, int]] = []
        for log_id in self._scene_ds.get_log_ids():
            for ts, _ in self._scene_ds.iter_log_frames(log_id):
                self._index.append((log_id, ts))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        import torch

        log_id, timestamp_ns = self._index[idx]

        # Retrieve the frame dict for this timestamp.
        # TODO: replace with indexed lookup once frame path caching is implemented.
        frame: dict = {}
        for ts, f in self._scene_ds.iter_log_frames(log_id):
            if ts == timestamp_ns:
                frame = f
                break

        inp = self._scene_ds.adapt_to_backbone_input(log_id, timestamp_ns, frame)
        images     = torch.from_numpy(inp.images).float()       # (N_cam, 3, H, W)
        intrinsics = torch.from_numpy(inp.intrinsics).float()   # (N_cam, 3, 3)
        extrinsics = torch.from_numpy(inp.extrinsics).float()   # (N_cam, 4, 4)
        _, _, H, W = images.shape

        item = dict(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=(H, W),
            tags=torch.zeros(self.n_tags),
            risk=torch.tensor(0.0),
            input_ids=torch.zeros(8, dtype=torch.long),
        )
        if self.load_occ_gt:
            # AV2 LiDAR-projected BEV occupancy GT not yet implemented.
            item["occ_gt"] = torch.zeros(self.bev_h, self.bev_w, dtype=torch.float32)
        return item
