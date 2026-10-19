"""Kinematic state atomic functions.

Each function evaluates a per-track motion predicate and returns a
ScenarioDict containing tracks (and timestamps) where the predicate holds.

Convention: all thresholds have physically meaningful defaults derived
from the RefAV paper and standard autonomous driving practice.
"""

from __future__ import annotations

import numpy as np

from scene_mining.tracks import TrackCollection
from scene_mining.tracks.track import TrackRecord

from .base import ScenarioDict

# ── Default thresholds ───────────────────────────────────────────────────────

_STATIONARY_SPEED_MPS = 0.5        # below this → stationary
_MOVING_SPEED_MPS = 1.0            # above this → moving
_ACCEL_THRESHOLD_MPS2 = 1.0        # |forward_accel| above this → accelerating/decelerating
_TURNING_YAW_RATE_RPS = 0.1        # rad/s above this → turning
_LATERAL_ACCEL_THRESHOLD = 0.5     # |lateral_accel| above this → has_lateral_acceleration
_LANE_CHANGE_LATERAL_SPEED = 0.3   # m/s lateral speed → changing lanes


def _active_timestamps(
    tracks: TrackCollection,
    tid: str,
    pred_fn,
    candidates: ScenarioDict | None,
) -> list[int]:
    """Return timestamps where pred_fn(TrackRecord) is True for track tid.

    Restricts to timestamps already present in candidates[tid] if given.
    """
    recs = tracks.by_track.get(tid, [])
    if not recs:
        return []

    allowed_ts: set[int] | None = None
    if candidates is not None and tid in candidates:
        # Collect all timestamps from the candidate's related-object lists
        # An empty related dict ({}) means all timestamps are allowed
        inner = candidates[tid]
        if inner:
            allowed_ts = set()
            for ts_list in inner.values():
                allowed_ts.update(ts_list)
        # If inner is empty, allowed_ts stays None → all timestamps valid

    result: list[int] = []
    for rec in recs:
        if allowed_ts is not None and rec.timestamp_ns not in allowed_ts:
            continue
        if pred_fn(rec):
            result.append(rec.timestamp_ns)

    return result


def _build_result(
    tracks: TrackCollection,
    candidates: ScenarioDict | None,
    pred_fn,
    min_timestamps: int = 1,
) -> ScenarioDict:
    """Generic builder: collect tracks where pred_fn holds at ≥ min_timestamps."""
    check_ids = list(candidates.keys()) if candidates is not None else list(tracks.by_track)
    result: ScenarioDict = {}
    for tid in check_ids:
        ts = _active_timestamps(tracks, tid, pred_fn, candidates)
        if len(ts) >= min_timestamps:
            result[tid] = {}
    return result


# ── Atomic functions ─────────────────────────────────────────────────────────


def moving(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    min_speed_mps: float = _MOVING_SPEED_MPS,
) -> ScenarioDict:
    """Return tracks that are moving (speed > min_speed_mps).

    Args:
        tracks: Full TrackCollection for the log.
        candidates: Optional pre-filter ScenarioDict.
        min_speed_mps: Speed threshold in m/s (default 1.0).

    Returns:
        ScenarioDict with tracks moving at any timestamp.
    """
    return _build_result(tracks, candidates, lambda r: r.speed > min_speed_mps)


def stationary(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    max_speed_mps: float = _STATIONARY_SPEED_MPS,
) -> ScenarioDict:
    """Return tracks that are stationary (speed < max_speed_mps).

    Args:
        tracks: Full TrackCollection for the log.
        candidates: Optional pre-filter ScenarioDict.
        max_speed_mps: Speed threshold in m/s (default 0.5).

    Returns:
        ScenarioDict with stationary tracks.
    """
    return _build_result(tracks, candidates, lambda r: r.speed < max_speed_mps)


def accelerating(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    min_accel_mps2: float = _ACCEL_THRESHOLD_MPS2,
) -> ScenarioDict:
    """Return tracks with positive forward acceleration > min_accel_mps2.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        min_accel_mps2: Minimum forward acceleration threshold (m/s²).

    Returns:
        ScenarioDict with tracks that are speeding up.
    """
    return _build_result(tracks, candidates, lambda r: r.forward_acceleration > min_accel_mps2)


def decelerating(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    min_decel_mps2: float = _ACCEL_THRESHOLD_MPS2,
) -> ScenarioDict:
    """Return tracks with negative forward acceleration (decelerating).

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        min_decel_mps2: Minimum deceleration magnitude (m/s²), default 1.0.

    Returns:
        ScenarioDict with tracks that are slowing down.
    """
    return _build_result(
        tracks, candidates, lambda r: r.forward_acceleration < -min_decel_mps2
    )


def turning(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    min_yaw_rate_rps: float = _TURNING_YAW_RATE_RPS,
    direction: str | None = None,
) -> ScenarioDict:
    """Return tracks that are turning.

    Yaw rate is estimated from the heading change between consecutive
    records for each track.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        min_yaw_rate_rps: Minimum |yaw rate| in rad/s to qualify as turning
            (default 0.1 rad/s ≈ 5.7°/s).
        direction: Optional ``"left"`` or ``"right"`` to restrict direction.
            ``"left"`` = positive yaw rate; ``"right"`` = negative.

    Returns:
        ScenarioDict with tracks that are turning at qualifying timestamps.
    """
    result: ScenarioDict = {}
    check_ids = list(candidates.keys()) if candidates is not None else list(tracks.by_track)

    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        if len(recs) < 2:
            continue

        turning_ts: list[int] = []
        for i in range(1, len(recs)):
            dt = (recs[i].timestamp_ns - recs[i - 1].timestamp_ns) * 1e-9
            if dt < 1e-6:
                continue
            dyaw = _angle_diff(recs[i].yaw, recs[i - 1].yaw)
            yaw_rate = dyaw / dt

            if abs(yaw_rate) < min_yaw_rate_rps:
                continue
            if direction == "left" and yaw_rate <= 0:
                continue
            if direction == "right" and yaw_rate >= 0:
                continue
            turning_ts.append(recs[i].timestamp_ns)

        if turning_ts:
            result[tid] = {}

    return result


def changing_lanes(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    min_lateral_speed_mps: float = _LANE_CHANGE_LATERAL_SPEED,
) -> ScenarioDict:
    """Return tracks that are performing a lane change.

    Detected as sustained lateral speed > min_lateral_speed_mps while
    also moving forward (speed > 2.0 m/s).

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        min_lateral_speed_mps: Lateral speed threshold in m/s (default 0.3).

    Returns:
        ScenarioDict with lane-changing tracks.
    """
    def _pred(r: TrackRecord) -> bool:
        return r.speed > 2.0 and abs(r.lateral_speed) > min_lateral_speed_mps

    return _build_result(tracks, candidates, _pred)


def has_velocity(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    min_speed_mps: float = 0.1,
    max_speed_mps: float = float("inf"),
) -> ScenarioDict:
    """Return tracks whose speed falls within [min_speed_mps, max_speed_mps].

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        min_speed_mps: Minimum speed (inclusive, default 0.1 m/s).
        max_speed_mps: Maximum speed (inclusive, default inf).

    Returns:
        ScenarioDict with matching tracks.
    """
    def _pred(r: TrackRecord) -> bool:
        return min_speed_mps <= r.speed <= max_speed_mps

    return _build_result(tracks, candidates, _pred)


def has_lateral_acceleration(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    min_lat_accel_mps2: float = _LATERAL_ACCEL_THRESHOLD,
) -> ScenarioDict:
    """Return tracks with |lateral_acceleration| > min_lat_accel_mps2.

    High lateral acceleration indicates hard cornering or evasive maneuvers.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        min_lat_accel_mps2: Lateral acceleration magnitude threshold (m/s²).

    Returns:
        ScenarioDict with tracks exhibiting lateral acceleration.
    """
    return _build_result(
        tracks,
        candidates,
        lambda r: abs(r.lateral_acceleration) > min_lat_accel_mps2,
    )


# ── Private geometry helper ───────────────────────────────────────────────────


def _angle_diff(a: float, b: float) -> float:
    """Signed shortest angular difference a - b in (-π, π]."""
    diff = a - b
    # Wrap to (-π, π]
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff <= -np.pi:
        diff += 2 * np.pi
    return diff
