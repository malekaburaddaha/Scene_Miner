"""Load 3D tracks from AV2, nuScenes, Waymo, or raw pickle/parquet files.

TrackLoader is the single entry point for converting any supported
annotation format into a TrackCollection.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .track import TrackCollection, TrackRecord


class TrackLoader:
    """Factory for loading TrackCollection from various sources.

    Supported formats:
    - AV2 ground-truth annotations (av2 Python API)
    - RefAV-style pickle (dict mapping log_id → list[dict])
    - Parquet files (AV2 per-log annotation parquet)
    - nuScenes JSON sample_annotation
    """

    # ── AV2 Ground-Truth ────────────────────────────────────────────────

    @classmethod
    def from_av2(cls, log_id: str, av2_dataroot: str | Path) -> TrackCollection:
        """Load AV2 ground-truth annotations for a single log.

        Requires the ``av2`` package (``pip install av2``).
        """
        from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
        from av2.utils.typing import NDArrayFloat

        dataroot = Path(av2_dataroot)
        loader = AV2SensorDataLoader(data_dir=dataroot, labels_dir=dataroot)
        sweep_uuids = loader.get_ordered_log_cam_fpaths(log_id=log_id)

        records: list[TrackRecord] = []
        city_SE3_ego_dict = loader.get_city_SE3_ego(log_id=log_id)

        for timestamp_ns, annotations in loader.get_labels_at_lidar_tss(log_id).items():
            for ann in annotations:
                translation = ann.dst_SE3_object.translation
                quat = ann.dst_SE3_object.rotation
                yaw = _quat_to_yaw(quat)
                rec = TrackRecord(
                    timestamp_ns=int(timestamp_ns),
                    track_id=ann.track_uuid,
                    confidence=1.0,
                    class_name=ann.category.value,
                    translation_m=np.array(translation, dtype=np.float64),
                    size=np.array(
                        [ann.length_m, ann.width_m, ann.height_m], dtype=np.float64
                    ),
                    yaw=yaw,
                )
                records.append(rec)

        return TrackCollection(log_id=log_id, records=records)

    @classmethod
    def from_av2_parquet(cls, log_id: str, parquet_path: str | Path) -> TrackCollection:
        """Load AV2 annotations from a single per-log annotation parquet file.

        Parquet columns expected (AV2 format):
            timestamp_ns, track_uuid, category, tx_m, ty_m, tz_m,
            length_m, width_m, height_m, qw, qx, qy, qz, num_interior_pts
        """
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        records: list[TrackRecord] = []

        for _, row in df.iterrows():
            yaw = _quat_to_yaw(
                np.array([row["qw"], row["qx"], row["qy"], row["qz"]], dtype=np.float64)
            )
            rec = TrackRecord(
                timestamp_ns=int(row["timestamp_ns"]),
                track_id=str(row["track_uuid"]),
                confidence=1.0,
                class_name=str(row["category"]),
                translation_m=np.array(
                    [row["tx_m"], row["ty_m"], row["tz_m"]], dtype=np.float64
                ),
                size=np.array(
                    [row["length_m"], row["width_m"], row["height_m"]], dtype=np.float64
                ),
                yaw=yaw,
            )
            records.append(rec)

        return TrackCollection(log_id=log_id, records=records)

    # ── RefAV pickle (predicted tracks from LE3DE2E or similar) ─────────

    @classmethod
    def from_pickle(cls, log_id: str, pickle_path: str | Path) -> TrackCollection:
        """Load tracks from a RefAV-style pickle.

        Expected pickle content: either
        - A dict mapping log_id → list[dict]
        - A list[dict] (single-log dump)

        Each dict should have keys matching TrackRecord fields.
        """
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            raw = data.get(log_id, [])
        else:
            raw = data

        records = [_dict_to_record(d) for d in raw]
        return TrackCollection(log_id=log_id, records=records)

    # ── nuScenes ─────────────────────────────────────────────────────────

    @classmethod
    def from_nuscenes(cls, scene_token: str, nusc: Any) -> TrackCollection:
        """Load tracks for a nuScenes scene.

        Args:
            scene_token: nuScenes scene token.
            nusc: NuScenes instance (``from nuscenes.nuscenes import NuScenes``).
        """
        scene = nusc.get("scene", scene_token)
        log_id = scene_token

        records: list[TrackRecord] = []
        sample_token = scene["first_sample_token"]

        while sample_token:
            sample = nusc.get("sample", sample_token)
            timestamp_ns = sample["timestamp"] * 1000  # nuScenes uses µs

            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                instance = nusc.get("instance", ann["instance_token"])
                category_name = ann["category_name"]

                # Map nuScenes category to a simplified class name
                class_name = _nuscenes_category_to_class(category_name)
                rotation = ann["rotation"]  # [w, x, y, z]
                yaw = _quat_to_yaw(np.array(rotation, dtype=np.float64))

                rec = TrackRecord(
                    timestamp_ns=int(timestamp_ns),
                    track_id=ann["instance_token"],
                    confidence=1.0,
                    class_name=class_name,
                    translation_m=np.array(ann["translation"], dtype=np.float64),
                    size=np.array(ann["size"][::-1], dtype=np.float64),  # WLH → LWH
                    yaw=yaw,
                )
                records.append(rec)

            sample_token = sample["next"]

        return TrackCollection(log_id=log_id, records=records)

    # ── Generic list-of-dicts ─────────────────────────────────────────────

    @classmethod
    def from_records(cls, log_id: str, records: list[TrackRecord]) -> TrackCollection:
        """Wrap a pre-built list of TrackRecord objects."""
        return TrackCollection(log_id=log_id, records=records)

    @classmethod
    def from_dicts(cls, log_id: str, dicts: list[dict]) -> TrackCollection:
        """Build from a list of plain dicts (flexible field mapping)."""
        records = [_dict_to_record(d) for d in dicts]
        return TrackCollection(log_id=log_id, records=records)


# ── Private helpers ──────────────────────────────────────────────────────────


def _quat_to_yaw(quat: np.ndarray) -> float:
    """Convert quaternion [w, x, y, z] to yaw angle (rotation about z-axis)."""
    w, x, y, z = quat
    # yaw = atan2(2(wz + xy), 1 - 2(y² + z²))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _dict_to_record(d: dict) -> TrackRecord:
    """Convert a flat dict to a TrackRecord.

    Accepts both snake_case and alternative field names.
    """
    translation = d.get("translation_m") or d.get("translation") or d.get("center")
    size = d.get("size") or d.get("dimensions")
    yaw = d.get("yaw") or d.get("heading") or 0.0

    if isinstance(yaw, (list, np.ndarray)):
        yaw = float(yaw[0])

    if isinstance(translation, (list, tuple)):
        translation = np.array(translation, dtype=np.float64)
    if isinstance(size, (list, tuple)):
        size = np.array(size, dtype=np.float64)

    # Support quaternion → yaw conversion if quat fields present
    if "qw" in d:
        quat = np.array([d["qw"], d["qx"], d["qy"], d["qz"]], dtype=np.float64)
        yaw = _quat_to_yaw(quat)

    return TrackRecord(
        timestamp_ns=int(d.get("timestamp_ns") or d.get("timestamp", 0)),
        track_id=str(d.get("track_id") or d.get("track_uuid") or d.get("instance_token", "")),
        confidence=float(d.get("confidence") or d.get("score", 1.0)),
        class_name=str(d.get("class_name") or d.get("category") or d.get("label", "UNKNOWN")),
        translation_m=np.asarray(translation, dtype=np.float64),
        size=np.asarray(size, dtype=np.float64),
        yaw=float(yaw),
    )


def _nuscenes_category_to_class(category_name: str) -> str:
    """Map nuScenes dotted category name to a simplified class name."""
    mapping = {
        "vehicle.car": "REGULAR_VEHICLE",
        "vehicle.truck": "TRUCK",
        "vehicle.bus.rigid": "BUS",
        "vehicle.bus.bendy": "ARTICULATED_BUS",
        "vehicle.motorcycle": "MOTORCYCLE",
        "vehicle.bicycle": "BICYCLIST",
        "vehicle.trailer": "VEHICULAR_TRAILER",
        "vehicle.construction": "LARGE_VEHICLE",
        "vehicle.emergency.police": "REGULAR_VEHICLE",
        "vehicle.emergency.ambulance": "LARGE_VEHICLE",
        "human.pedestrian.adult": "PEDESTRIAN",
        "human.pedestrian.child": "PEDESTRIAN",
        "human.pedestrian.wheelchair": "WHEELCHAIR",
        "human.pedestrian.stroller": "STROLLER",
        "human.pedestrian.personal_mobility": "WHEELED_RIDER",
        "human.pedestrian.police_officer": "PEDESTRIAN",
        "human.pedestrian.construction_worker": "PEDESTRIAN",
        "animal": "ANIMAL",
        "movable_object.barrier": "BOLLARD",
        "movable_object.trafficcone": "CONSTRUCTION_CONE",
        "movable_object.pushable_pullable": "WHEELED_DEVICE",
        "movable_object.debris": "BOLLARD",
        "static_object.bicycle_rack": "BICYCLE",
    }
    return mapping.get(category_name, category_name.upper().replace(".", "_"))
