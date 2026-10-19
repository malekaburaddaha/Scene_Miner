"""Build a synthetic ego-vehicle TrackRecord sequence from AV2 pose data.

The ego vehicle has no annotation in AV2 ground truth, but its pose is
available via city_SE3_ego transforms. This module synthesises TrackRecord
objects for the ego so that atomic functions (following, near_objects, etc.)
can treat the ego uniformly alongside other agents.
"""

from __future__ import annotations

import numpy as np

from .track import TrackRecord

# Conventional ego vehicle track ID — stable across the codebase
EGO_TRACK_ID = "EGO_VEHICLE"

# Approximate AV2 ego vehicle physical dimensions (Jaguar I-PACE)
EGO_LENGTH_M = 4.68
EGO_WIDTH_M = 1.89
EGO_HEIGHT_M = 1.55


def build_ego_track(
    log_id: str,
    timestamps_ns: list[int],
    city_SE3_ego: dict[int, "SE3"],  # type: ignore[name-defined]
) -> list[TrackRecord]:
    """Build ego vehicle TrackRecord list from city_SE3_ego transforms.

    Args:
        log_id: AV2 log identifier (unused here; kept for symmetry).
        timestamps_ns: Ordered list of lidar timestamp values (nanoseconds).
        city_SE3_ego: Dict mapping timestamp_ns → av2 SE3 (city ← ego).
            Each SE3 provides `.translation` [x, y, z] and `.rotation_matrix` [3,3]
            in the city frame.

    Returns:
        List of TrackRecord objects (one per timestamp), with ego frame
        positions set to [0, 0, 0] (by definition of the ego coordinate frame).
        Velocity/acceleration are computed by TrackCollection.__post_init__
        from the translation series.

    Note:
        AV2 annotations are already in the ego frame at each timestamp, so
        the ego position is always (0, 0, 0) in its own frame. However, for
        computing ego motion in the *city* frame (needed for ``following``
        and other relative-motion checks), we embed the ego's city-frame
        translations here and flag them with a special class_name.

        Atomic functions that need ego motion in ego-frame should call
        ``relative_position`` against the ego TrackRecord, not use
        ``translation_m`` directly.
    """
    records: list[TrackRecord] = []

    for ts in timestamps_ns:
        pose = city_SE3_ego.get(ts)
        if pose is None:
            continue

        # Extract city-frame translation
        tx, ty, tz = pose.translation  # (3,)

        # Extract yaw from rotation matrix: R @ [1,0,0] → heading vector
        # R[:, 0] is the forward direction in the city frame
        heading_city = pose.rotation_matrix[:, 0]
        yaw = float(np.arctan2(heading_city[1], heading_city[0]))

        rec = TrackRecord(
            timestamp_ns=int(ts),
            track_id=EGO_TRACK_ID,
            confidence=1.0,
            class_name="EGO_VEHICLE",
            translation_m=np.array([tx, ty, tz], dtype=np.float64),
            size=np.array([EGO_LENGTH_M, EGO_WIDTH_M, EGO_HEIGHT_M], dtype=np.float64),
            yaw=yaw,
        )
        records.append(rec)

    return records


def build_ego_track_from_poses(
    log_id: str,
    poses: list[dict],
) -> list[TrackRecord]:
    """Build ego track from a list of pose dicts (format-agnostic).

    Each dict must contain:
        - ``timestamp_ns`` (int)
        - ``x``, ``y``, ``z`` (float) — city-frame position
        - ``yaw`` (float, radians) — heading in city frame
    """
    records: list[TrackRecord] = []
    for p in sorted(poses, key=lambda d: d["timestamp_ns"]):
        rec = TrackRecord(
            timestamp_ns=int(p["timestamp_ns"]),
            track_id=EGO_TRACK_ID,
            confidence=1.0,
            class_name="EGO_VEHICLE",
            translation_m=np.array([p["x"], p["y"], p.get("z", 0.0)], dtype=np.float64),
            size=np.array([EGO_LENGTH_M, EGO_WIDTH_M, EGO_HEIGHT_M], dtype=np.float64),
            yaw=float(p["yaw"]),
        )
        records.append(rec)
    return records
