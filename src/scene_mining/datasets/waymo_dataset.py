"""Waymo Open Dataset adapter (stub).

This module provides a BaseSceneDataset-compatible interface for Waymo.
Full implementation requires ``waymo-open-dataset-tf-2-12-0`` and
TensorFlow >= 2.12.

If the waymo_open_dataset package is not installed, all methods raise
``NotImplementedError`` with a helpful installation message.

Waymo specifics:
- 5 cameras: FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
- 10 Hz annotation rate
- Object IDs are integer track IDs (cast to strings)
- TFRecord-based data format

Usage::

    from scene_mining.datasets.waymo_dataset import WaymoDataset
    ds = WaymoDataset(data_dir="/data/waymo/validation", split="val")
    log_ids = ds.get_log_ids()
    tracks = ds.get_annotations(log_ids[0])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scene_mining.datasets.av2_dataset import BackboneInput
from scene_mining.datasets.base_dataset import BaseSceneDataset

# Waymo canonical camera names (matches waymo_open_dataset.dataset_pb2)
WAYMO_CAMERA_NAMES = [
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
]

# Target (H, W) for backbone input
WAYMO_IMAGE_SIZE = (448, 800)

_INSTALL_MSG = (
    "The Waymo Open Dataset adapter requires:\n"
    "  pip install waymo-open-dataset-tf-2-12-0\n"
    "  pip install tensorflow>=2.12.0\n"
    "These packages are declared in the 'waymo' extras of pyproject.toml:\n"
    "  pip install -e '.[waymo]'"
)


def _require_waymo() -> None:
    """Raise NotImplementedError if waymo_open_dataset is not installed."""
    try:
        import waymo_open_dataset  # noqa: F401
    except ImportError as exc:
        raise NotImplementedError(_INSTALL_MSG) from exc


class WaymoDataset(BaseSceneDataset):
    """Waymo Open Dataset adapter.

    Args:
        data_dir: Path to directory containing Waymo TFRecord files.
        split: One of ``"train"``, ``"val"``.
        cameras: Subset of cameras. Defaults to all 5.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "val",
        cameras: list[str] | None = None,
    ) -> None:
        _require_waymo()
        self.data_dir = Path(data_dir)
        self.split = split
        self.cameras = cameras or WAYMO_CAMERA_NAMES
        self._tfrecords: list[Path] = sorted(self.data_dir.glob("*.tfrecord"))

    # ── BaseSceneDataset implementation ───────────────────────────────────────

    def get_log_ids(self) -> list[str]:
        """Return TFRecord stem names as log IDs."""
        _require_waymo()
        return [p.stem for p in self._tfrecords]

    def get_annotations(self, log_id: str) -> "TrackCollection":
        """Load Waymo GT annotations for *log_id*.

        Returns:
            TrackCollection with all annotated objects across the segment.
        """
        _require_waymo()
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
        from scene_mining.tracks.track import TrackCollection, TrackRecord

        tfrecord_path = self._find_tfrecord(log_id)
        records: list[TrackRecord] = []

        dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
        for raw_frame in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(raw_frame.numpy())
            timestamp_ns = int(frame.timestamp_micros) * 1000  # µs → ns

            for label in frame.laser_labels:
                box = label.box
                rec = TrackRecord(
                    timestamp_ns=timestamp_ns,
                    track_id=str(label.id),
                    confidence=1.0,
                    class_name=_waymo_label_to_class(label.type),
                    translation_m=np.array(
                        [box.center_x, box.center_y, box.center_z],
                        dtype=np.float64,
                    ),
                    size=np.array(
                        [box.length, box.width, box.height], dtype=np.float64
                    ),
                    yaw=float(box.heading),
                )
                records.append(rec)

        return TrackCollection(log_id=log_id, records=records)

    def adapt_to_backbone_input(
        self,
        log_id: str,
        timestamp_ns: int,
        frame: dict[str, Any] | None = None,
    ) -> BackboneInput:
        """Build a BackboneInput for *log_id* at *timestamp_ns*."""
        _require_waymo()
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
        from waymo_open_dataset.utils import frame_utils

        tfrecord_path = self._find_tfrecord(log_id)
        dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")

        target_frame = self._find_closest_frame(dataset, dataset_pb2, timestamp_ns)
        if target_frame is None:
            raise ValueError(
                f"No Waymo frame found near timestamp {timestamp_ns} in {log_id}"
            )

        images_list, K_list, ego2cam_list, original_hw = [], [], [], []

        for cam_name in self.cameras:
            cam_idx = WAYMO_CAMERA_NAMES.index(cam_name) + 1  # Waymo uses 1-indexed
            img_np, K, ego2cam = _extract_camera(target_frame, cam_idx)

            if img_np is None:
                H, W = WAYMO_IMAGE_SIZE
                img_np = np.zeros((H, W, 3), dtype=np.uint8)
                K = np.eye(3, dtype=np.float32)
                ego2cam = np.eye(4, dtype=np.float32)

            orig_h, orig_w = img_np.shape[:2]
            target_h, target_w = WAYMO_IMAGE_SIZE
            img_resized, K_adj = _resize_with_intrinsics(img_np, K, target_h, target_w)

            img_f = img_resized.astype(np.float32) / 255.0
            images_list.append(np.transpose(img_f, (2, 0, 1)))
            K_list.append(K_adj)
            ego2cam_list.append(ego2cam)
            original_hw.append((orig_h, orig_w))

        actual_ts = int(target_frame.timestamp_micros) * 1000

        return BackboneInput(
            log_id=log_id,
            timestamp_ns=actual_ts,
            images=np.stack(images_list, axis=0),
            intrinsics=np.stack(K_list, axis=0),
            extrinsics=np.stack(ego2cam_list, axis=0),
            camera_names=self.cameras,
            original_hw=original_hw,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _find_tfrecord(self, log_id: str) -> Path:
        for p in self._tfrecords:
            if p.stem == log_id:
                return p
        raise FileNotFoundError(f"TFRecord not found for log_id={log_id!r}")

    def _find_closest_frame(
        self,
        dataset: Any,
        dataset_pb2: Any,
        timestamp_ns: int,
    ) -> Any | None:
        """Scan the TFRecord and return the frame closest to *timestamp_ns*."""
        best_frame = None
        best_diff = float("inf")
        for raw in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(raw.numpy())
            diff = abs(int(frame.timestamp_micros) * 1000 - timestamp_ns)
            if diff < best_diff:
                best_diff = diff
                best_frame = frame
        return best_frame


# ── Module-level helpers ───────────────────────────────────────────────────────


def _waymo_label_to_class(label_type: int) -> str:
    """Map Waymo label type enum value to a class name string."""
    # waymo_open_dataset.label_pb2.Label.Type values
    _MAP = {
        0: "UNKNOWN",
        1: "VEHICLE",
        2: "PEDESTRIAN",
        3: "SIGN",
        4: "CYCLIST",
    }
    return _MAP.get(label_type, "UNKNOWN")


def _extract_camera(
    frame: Any, cam_idx: int
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Extract image + calibration for camera *cam_idx* from a Waymo frame.

    Returns:
        (image HWC uint8 or None, K 3x3, ego2cam 4x4)
    """
    import cv2

    # Find image
    img_np = None
    for cam_image in frame.images:
        if cam_image.name == cam_idx:
            buf = np.frombuffer(cam_image.image, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            break

    # Find calibration
    K = np.eye(3, dtype=np.float32)
    ego2cam = np.eye(4, dtype=np.float32)
    for calib in frame.context.camera_calibrations:
        if calib.name == cam_idx:
            f_u, f_v, c_u, c_v = calib.intrinsic[:4]
            K = np.array([[f_u, 0, c_u], [0, f_v, c_v], [0, 0, 1]], dtype=np.float32)
            # calib.extrinsic.transform is a flat 4x4 row-major matrix
            T = np.array(calib.extrinsic.transform, dtype=np.float32).reshape(4, 4)
            ego2cam = np.linalg.inv(T)  # vehicle2sensor
            break

    return img_np, K, ego2cam


def _resize_with_intrinsics(
    img: np.ndarray,
    K: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aspect-ratio-preserving resize + centre-padding, with intrinsic adjustment."""
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

    K_adj = K.copy()
    K_adj[0, 0] *= scale
    K_adj[1, 1] *= scale
    K_adj[0, 2] = K[0, 2] * scale + pad_left
    K_adj[1, 2] = K[1, 2] * scale + pad_top
    return padded, K_adj
