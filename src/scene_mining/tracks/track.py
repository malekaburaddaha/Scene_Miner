"""3D track data structures for autonomous driving scenario mining.

TrackRecord: single detection at one timestamp.
TrackCollection: all tracks for a driving log, with kinematic derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# AV2 super-categories used by atomic functions
VEHICLE_CATEGORIES = frozenset({
    "REGULAR_VEHICLE", "LARGE_VEHICLE", "BOX_TRUCK", "TRUCK", "TRUCK_CAB",
    "VEHICULAR_TRAILER", "BUS", "SCHOOL_BUS", "ARTICULATED_BUS",
    "RAILED_VEHICLE", "MOTORCYCLE", "BICYCLIST",
})

# AV2's full 30 annotated categories
AV2_CATEGORIES = frozenset({
    "REGULAR_VEHICLE", "PEDESTRIAN", "BICYCLIST", "MOTORCYCLIST",
    "WHEELED_RIDER", "BOLLARD", "CONSTRUCTION_CONE", "SIGN",
    "CONSTRUCTION_BARREL", "STOP_SIGN", "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "VEHICULAR_TRAILER",
    "TRUCK_CAB", "SCHOOL_BUS", "ARTICULATED_BUS", "MESSAGE_BOARD_TRAILER",
    "BICYCLE", "MOTORCYCLE", "WHEELED_DEVICE", "WHEELCHAIR",
    "STROLLER", "DOG", "ANIMAL", "OFFICIAL_SIGNALER",
    "RAILED_VEHICLE", "EGO_VEHICLE",
})


@dataclass
class TrackRecord:
    """Single object detection at one timestamp.

    Follows the RefAV wire format exactly, with derived kinematics added.
    All positions are in the ego-vehicle coordinate frame (right-hand,
    x=forward, y=left, z=up, origin at rear axle center).
    """

    timestamp_ns: int
    track_id: str                       # UUID string (AV2) — use str universally
    confidence: float
    class_name: str                     # One of AV2_CATEGORIES
    translation_m: np.ndarray           # shape (3,) — [x, y, z]
    size: np.ndarray                    # shape (3,) — [length, width, height]
    yaw: float                          # radians, counter-clockwise from x-axis

    # Derived fields — populated by TrackCollection at load time
    velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration_mps2: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Visual feature cache key — set by FeatureCache
    feature_cache_key: Optional[str] = None

    @property
    def speed(self) -> float:
        """Scalar ground-plane speed in m/s (XY only)."""
        return float(np.linalg.norm(self.velocity_mps[:2]))

    @property
    def forward_speed(self) -> float:
        """Speed component in the heading direction (signed)."""
        heading = self.heading_vector
        return float(np.dot(self.velocity_mps[:2], heading[:2]))

    @property
    def lateral_speed(self) -> float:
        """Speed component perpendicular to heading (signed, left=positive)."""
        left = self.left_vector
        return float(np.dot(self.velocity_mps[:2], left[:2]))

    @property
    def forward_acceleration(self) -> float:
        """Acceleration component in heading direction (m/s²)."""
        return float(np.dot(self.acceleration_mps2[:2], self.heading_vector[:2]))

    @property
    def lateral_acceleration(self) -> float:
        """Acceleration component perpendicular to heading (m/s², left=positive)."""
        return float(np.dot(self.acceleration_mps2[:2], self.left_vector[:2]))

    @property
    def heading_vector(self) -> np.ndarray:
        """Unit vector in the heading (yaw) direction."""
        return np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])

    @property
    def left_vector(self) -> np.ndarray:
        """Unit vector 90° left of heading."""
        return np.array([-np.sin(self.yaw), np.cos(self.yaw), 0.0])

    @property
    def corners_bev(self) -> np.ndarray:
        """Bird's-eye-view corners of the 3D bounding box, shape (4, 2).

        Returns corners in order: front-left, front-right, rear-right, rear-left.
        """
        l, w = self.size[0] / 2, self.size[1] / 2
        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        # Local corners (length along x, width along y)
        local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]], dtype=float)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        rotated = (R @ local.T).T
        return rotated + self.translation_m[:2]

    def distance_to(self, other: "TrackRecord") -> float:
        """Euclidean ground-plane distance to another track record."""
        return float(np.linalg.norm(self.translation_m[:2] - other.translation_m[:2]))

    def relative_position(self, other: "TrackRecord") -> np.ndarray:
        """Position of `other` in this track's local coordinate frame.

        Returns (forward, left, up) components.
        """
        delta = other.translation_m - self.translation_m
        fwd = np.dot(delta, self.heading_vector)
        left = np.dot(delta, self.left_vector)
        return np.array([fwd, left, delta[2]])

    def __repr__(self) -> str:
        return (
            f"TrackRecord(id={self.track_id[:8]}, cls={self.class_name}, "
            f"t={self.timestamp_ns}, pos=({self.translation_m[0]:.1f}, "
            f"{self.translation_m[1]:.1f}), spd={self.speed:.1f}m/s)"
        )


@dataclass
class TrackCollection:
    """All track records for a single driving log.

    Organized for fast lookup by track ID and by timestamp. Kinematic
    quantities (velocity, acceleration) are derived via finite differences
    at construction time.
    """

    log_id: str
    records: list[TrackRecord]

    # Index structures built by __post_init__
    by_track: dict[str, list[TrackRecord]] = field(default_factory=dict, init=False)
    by_timestamp: dict[int, list[TrackRecord]] = field(default_factory=dict, init=False)
    timestamps: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._build_indexes()
        self._compute_kinematics()

    def _build_indexes(self) -> None:
        self.by_track = {}
        self.by_timestamp = {}
        for rec in self.records:
            self.by_track.setdefault(rec.track_id, []).append(rec)
            self.by_timestamp.setdefault(rec.timestamp_ns, []).append(rec)
        # Sort each track's records by time
        for tid in self.by_track:
            self.by_track[tid].sort(key=lambda r: r.timestamp_ns)
        self.timestamps = sorted(self.by_timestamp.keys())

    def _compute_kinematics(self) -> None:
        """Finite-difference velocity and acceleration for each track."""
        for recs in self.by_track.values():
            n = len(recs)
            for i, rec in enumerate(recs):
                if i == 0:
                    rec.velocity_mps = np.zeros(3)
                else:
                    dt = (recs[i].timestamp_ns - recs[i - 1].timestamp_ns) * 1e-9
                    dt = max(dt, 1e-6)
                    rec.velocity_mps = (
                        recs[i].translation_m - recs[i - 1].translation_m
                    ) / dt

                if i < 2:
                    rec.acceleration_mps2 = np.zeros(3)
                else:
                    dt = (recs[i].timestamp_ns - recs[i - 1].timestamp_ns) * 1e-9
                    dt = max(dt, 1e-6)
                    rec.acceleration_mps2 = (
                        recs[i].velocity_mps - recs[i - 1].velocity_mps
                    ) / dt

    # ── Query helpers ──────────────────────────────────────────────────────

    def get_track(self, track_id: str) -> list[TrackRecord]:
        """All records for a track, sorted by timestamp."""
        return self.by_track.get(track_id, [])

    def get_at_timestamp(self, timestamp_ns: int) -> list[TrackRecord]:
        """All records at a given timestamp."""
        return self.by_timestamp.get(timestamp_ns, [])

    def get_track_ids(self, category: str | None = None) -> list[str]:
        """All track IDs, optionally filtered by category."""
        if category is None:
            return list(self.by_track.keys())
        return [
            tid
            for tid, recs in self.by_track.items()
            if recs and recs[0].class_name == category
        ]

    def get_record(self, track_id: str, timestamp_ns: int) -> TrackRecord | None:
        """Single record for (track_id, timestamp_ns), or None."""
        for rec in self.by_timestamp.get(timestamp_ns, []):
            if rec.track_id == track_id:
                return rec
        return None

    def filter_by_distance(
        self, max_dist_m: float, origin: np.ndarray | None = None
    ) -> "TrackCollection":
        """Return new collection with records within max_dist_m of origin.

        origin defaults to [0, 0, 0] (ego vehicle position).
        """
        if origin is None:
            origin = np.zeros(3)
        filtered = [
            r
            for r in self.records
            if np.linalg.norm(r.translation_m[:2] - origin[:2]) <= max_dist_m
        ]
        return TrackCollection(log_id=self.log_id, records=filtered)

    def filter_by_confidence(self, min_confidence: float) -> "TrackCollection":
        filtered = [r for r in self.records if r.confidence >= min_confidence]
        return TrackCollection(log_id=self.log_id, records=filtered)

    def filter_top_k_per_category(self, k: int) -> "TrackCollection":
        """Keep top-k tracks per category by summed confidence (RefAV baseline)."""
        from collections import defaultdict

        # Sum confidence per track
        track_conf: dict[str, float] = defaultdict(float)
        track_cat: dict[str, str] = {}
        for rec in self.records:
            track_conf[rec.track_id] += rec.confidence
            track_cat[rec.track_id] = rec.class_name

        # Group tracks by category, sort by confidence
        by_cat: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for tid, total_conf in track_conf.items():
            by_cat[track_cat[tid]].append((total_conf, tid))

        kept_ids: set[str] = set()
        for cat, entries in by_cat.items():
            entries.sort(reverse=True)
            kept_ids.update(tid for _, tid in entries[:k])

        filtered = [r for r in self.records if r.track_id in kept_ids]
        return TrackCollection(log_id=self.log_id, records=filtered)

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return (
            f"TrackCollection(log={self.log_id}, "
            f"n_records={len(self.records)}, "
            f"n_tracks={len(self.by_track)}, "
            f"n_timestamps={len(self.timestamps)})"
        )
