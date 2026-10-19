"""Spatial and temporal relationship atomic functions.

Each function evaluates inter-agent relationships and returns a
ScenarioDict mapping candidate track_ids to related track_ids and the
timestamps at which the relationship holds.

Key implementation note on ``direction`` semantics (RefAV CVPR 2025
disambiguation):
    "direction" always refers to the orientation of RELATED objects
    relative to TRACK candidates, NOT the absolute compass direction.
    For example, ``has_objects_in_relative_direction(tracks, candidates,
    direction="front")`` returns tracks that have related objects in
    FRONT of them (i.e. the related object is forward of the candidate).
"""

from __future__ import annotations

import numpy as np

from scene_mining.tracks import TrackCollection
from scene_mining.tracks.track import TrackRecord

from .base import ScenarioDict

# ── Default thresholds ────────────────────────────────────────────────────────

_NEAR_DISTANCE_M = 10.0           # near_objects default radius
_HEADING_ANGLE_DEG = 45.0         # heading_toward cone half-angle
_FACING_ANGLE_DEG = 30.0          # facing_toward cone half-angle
_FOLLOWING_DIST_M = 15.0          # max longitudinal gap for following
_FOLLOWING_LATERAL_M = 3.0        # max lateral offset for following
_FOLLOWING_HEADING_DEG = 20.0     # max heading difference for following
_CROSSING_LATERAL_SPEED_MPS = 0.5 # being_crossed_by lateral speed threshold
_SAME_LANE_LATERAL_M = 2.0        # lateral distance → same lane
_SAME_LANE_DIST_M = 60.0          # longitudinal range for same-lane check


def near_objects(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    related_candidates: ScenarioDict | None = None,
    max_dist_m: float = _NEAR_DISTANCE_M,
    category: str | None = None,
) -> ScenarioDict:
    """Return tracks that have at least one other object within max_dist_m.

    Args:
        tracks: Full TrackCollection.
        candidates: Tracks to evaluate as the primary set.
        related_candidates: Optional restriction on which tracks can be
            the *related* object. If None, all other tracks qualify.
        max_dist_m: Maximum euclidean ground-plane distance (m).
        category: Optional category filter on the related object.

    Returns:
        ScenarioDict mapping candidate_id → {related_id → [timestamp_ns]}.
    """
    check_ids = _resolve_ids(candidates, tracks)
    related_ids = _resolve_ids(related_candidates, tracks) if related_candidates else None

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        for rec in recs:
            nearby = tracks.get_at_timestamp(rec.timestamp_ns)
            for other in nearby:
                if other.track_id == tid:
                    continue
                if related_ids is not None and other.track_id not in related_ids:
                    continue
                if category and other.class_name != category:
                    continue
                dist = rec.distance_to(other)
                if dist <= max_dist_m:
                    _add_to_result(result, tid, other.track_id, rec.timestamp_ns)

    return _drop_empty_timestamps(result)


def heading_toward(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    related_candidates: ScenarioDict | None = None,
    max_dist_m: float = 30.0,
    cone_half_angle_deg: float = _HEADING_ANGLE_DEG,
) -> ScenarioDict:
    """Return tracks whose heading points toward a related object.

    A candidate track is "heading toward" a related object when:
    1. The related object is within ``max_dist_m``.
    2. The angle between the candidate's heading vector and the vector
       from candidate to related object is ≤ cone_half_angle_deg.

    Args:
        tracks: Full TrackCollection.
        candidates: Primary tracks to evaluate.
        related_candidates: Optional restriction on related objects.
        max_dist_m: Maximum distance to related object (m).
        cone_half_angle_deg: Half-angle of heading cone in degrees.

    Returns:
        ScenarioDict mapping candidate → {related → [timestamp_ns]}.
    """
    check_ids = _resolve_ids(candidates, tracks)
    related_ids = _resolve_ids(related_candidates, tracks) if related_candidates else None
    cos_thresh = np.cos(np.radians(cone_half_angle_deg))

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        for rec in recs:
            nearby = tracks.get_at_timestamp(rec.timestamp_ns)
            for other in nearby:
                if other.track_id == tid:
                    continue
                if related_ids is not None and other.track_id not in related_ids:
                    continue
                dist = rec.distance_to(other)
                if dist > max_dist_m or dist < 1e-3:
                    continue
                vec_to_other = (other.translation_m[:2] - rec.translation_m[:2]) / dist
                heading = rec.heading_vector[:2]
                cos_angle = float(np.dot(heading, vec_to_other))
                if cos_angle >= cos_thresh:
                    _add_to_result(result, tid, other.track_id, rec.timestamp_ns)

    return _drop_empty_timestamps(result)


def facing_toward(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    related_candidates: ScenarioDict | None = None,
    max_dist_m: float = 30.0,
    cone_half_angle_deg: float = _FACING_ANGLE_DEG,
) -> ScenarioDict:
    """Return tracks that face toward a related object AND the related object
    faces back toward the candidate (mutual facing).

    This detects face-to-face scenarios like two pedestrians walking
    toward each other, or a vehicle facing a cyclist head-on.

    Both objects must have their heading pointing toward the other.

    Args:
        tracks: Full TrackCollection.
        candidates: Primary tracks to evaluate.
        related_candidates: Optional restriction on related objects.
        max_dist_m: Maximum distance between the two objects (m).
        cone_half_angle_deg: Half-angle of facing cone in degrees.

    Returns:
        ScenarioDict mapping candidate → {related → [timestamp_ns]}.
    """
    cos_thresh = np.cos(np.radians(cone_half_angle_deg))
    check_ids = _resolve_ids(candidates, tracks)
    related_ids = _resolve_ids(related_candidates, tracks) if related_candidates else None

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        for rec in recs:
            nearby = tracks.get_at_timestamp(rec.timestamp_ns)
            for other in nearby:
                if other.track_id == tid:
                    continue
                if related_ids is not None and other.track_id not in related_ids:
                    continue
                dist = rec.distance_to(other)
                if dist > max_dist_m or dist < 1e-3:
                    continue
                # Vector from rec to other (normalised)
                d12 = (other.translation_m[:2] - rec.translation_m[:2]) / dist
                # Both headings must point toward each other
                cos1 = float(np.dot(rec.heading_vector[:2], d12))
                cos2 = float(np.dot(other.heading_vector[:2], -d12))
                if cos1 >= cos_thresh and cos2 >= cos_thresh:
                    _add_to_result(result, tid, other.track_id, rec.timestamp_ns)

    return _drop_empty_timestamps(result)


def being_crossed_by(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    related_candidates: ScenarioDict | None = None,
    max_dist_m: float = 15.0,
    min_lateral_speed_mps: float = _CROSSING_LATERAL_SPEED_MPS,
) -> ScenarioDict:
    """Return tracks being crossed by another object.

    An object B is "crossing" track A when:
    1. B is within max_dist_m of A.
    2. B is moving in the lateral direction of A (|lateral_speed of B
       relative to A's heading| > min_lateral_speed_mps).
    3. B's relative position is close to A's lateral axis (|forward offset|
       < A's length/2 + some margin).

    Args:
        tracks: Full TrackCollection.
        candidates: Tracks being crossed (the stationary/slower objects).
        related_candidates: Crossing objects (typically pedestrians).
        max_dist_m: Maximum distance to the crossing object (m).
        min_lateral_speed_mps: Minimum lateral speed of crossing object (m/s).

    Returns:
        ScenarioDict mapping candidate → {crossing_obj → [timestamp_ns]}.
    """
    check_ids = _resolve_ids(candidates, tracks)
    related_ids = _resolve_ids(related_candidates, tracks) if related_candidates else None

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        for rec in recs:
            nearby = tracks.get_at_timestamp(rec.timestamp_ns)
            for other in nearby:
                if other.track_id == tid:
                    continue
                if related_ids is not None and other.track_id not in related_ids:
                    continue
                dist = rec.distance_to(other)
                if dist > max_dist_m:
                    continue
                # Lateral speed of other object in rec's frame
                lateral_spd = abs(np.dot(other.velocity_mps[:2], rec.left_vector[:2]))
                if lateral_spd < min_lateral_speed_mps:
                    continue
                # Other must be near the front-back axis of rec
                rel = rec.relative_position(other)
                margin = rec.size[0] * 0.5 + 2.0
                if abs(rel[0]) <= margin:
                    _add_to_result(result, tid, other.track_id, rec.timestamp_ns)

    return _drop_empty_timestamps(result)


def following(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    related_candidates: ScenarioDict | None = None,
    max_dist_m: float = _FOLLOWING_DIST_M,
    max_lateral_offset_m: float = _FOLLOWING_LATERAL_M,
    max_heading_diff_deg: float = _FOLLOWING_HEADING_DEG,
    min_speed_mps: float = 1.0,
) -> ScenarioDict:
    """Return tracks that are following another object.

    A candidate track A is following related track B when, over multiple
    consecutive timestamps:
    1. B is in front of A (positive forward component).
    2. The longitudinal distance is ≤ max_dist_m.
    3. The lateral offset is ≤ max_lateral_offset_m.
    4. The heading difference is ≤ max_heading_diff_deg.
    5. A is moving (speed > min_speed_mps).

    Args:
        tracks: Full TrackCollection.
        candidates: Following objects (the ones behind).
        related_candidates: Objects being followed (the ones in front).
        max_dist_m: Maximum following distance (m).
        max_lateral_offset_m: Maximum allowed lateral offset (m).
        max_heading_diff_deg: Maximum heading difference (degrees).
        min_speed_mps: Minimum speed for the following object (m/s).

    Returns:
        ScenarioDict mapping follower → {leader → [timestamp_ns]}.
    """
    check_ids = _resolve_ids(candidates, tracks)
    related_ids = _resolve_ids(related_candidates, tracks) if related_candidates else None
    cos_heading = np.cos(np.radians(max_heading_diff_deg))

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        for rec in recs:
            if rec.speed < min_speed_mps:
                continue
            nearby = tracks.get_at_timestamp(rec.timestamp_ns)
            for other in nearby:
                if other.track_id == tid:
                    continue
                if related_ids is not None and other.track_id not in related_ids:
                    continue
                rel = rec.relative_position(other)  # (forward, left, up)
                fwd, lat = rel[0], rel[1]
                if fwd <= 0 or fwd > max_dist_m:
                    continue
                if abs(lat) > max_lateral_offset_m:
                    continue
                # Heading alignment
                head_cos = float(
                    np.dot(rec.heading_vector[:2], other.heading_vector[:2])
                )
                if head_cos < cos_heading:
                    continue
                _add_to_result(result, tid, other.track_id, rec.timestamp_ns)

    return _drop_empty_timestamps(result)


def in_same_lane(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    related_candidates: ScenarioDict | None = None,
    max_lateral_m: float = _SAME_LANE_LATERAL_M,
    max_longitudinal_m: float = _SAME_LANE_DIST_M,
) -> ScenarioDict:
    """Return tracks that share the same lane as a related object.

    Two objects are considered in the same lane when:
    1. They are aligned in heading (heading dot product > 0.85).
    2. The lateral separation in the candidate's frame is ≤ max_lateral_m.
    3. The longitudinal separation is ≤ max_longitudinal_m.

    Args:
        tracks: Full TrackCollection.
        candidates: Primary track set.
        related_candidates: Related track set.
        max_lateral_m: Maximum lateral offset (m), default 2.0.
        max_longitudinal_m: Maximum longitudinal distance (m), default 60.0.

    Returns:
        ScenarioDict mapping candidate → {related → [timestamp_ns]}.
    """
    check_ids = _resolve_ids(candidates, tracks)
    related_ids = _resolve_ids(related_candidates, tracks) if related_candidates else None

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        for rec in recs:
            nearby = tracks.get_at_timestamp(rec.timestamp_ns)
            for other in nearby:
                if other.track_id == tid:
                    continue
                if related_ids is not None and other.track_id not in related_ids:
                    continue
                # Heading alignment
                head_cos = float(
                    np.dot(rec.heading_vector[:2], other.heading_vector[:2])
                )
                if head_cos < 0.85:
                    continue
                rel = rec.relative_position(other)
                if abs(rel[1]) > max_lateral_m:
                    continue
                if abs(rel[0]) > max_longitudinal_m:
                    continue
                _add_to_result(result, tid, other.track_id, rec.timestamp_ns)

    return _drop_empty_timestamps(result)


def has_objects_in_relative_direction(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    related_candidates: ScenarioDict | None = None,
    direction: str = "front",
    max_dist_m: float = 30.0,
    cone_half_angle_deg: float = 60.0,
) -> ScenarioDict:
    """Return tracks that have other objects in a relative spatial direction.

    Direction semantics (CRITICAL — RefAV disambiguation):
        ``direction`` refers to the location of the RELATED object
        relative to the CANDIDATE (not compass direction).

    Args:
        tracks: Full TrackCollection.
        candidates: Primary track set (the reference frame).
        related_candidates: Optional restriction on related objects.
        direction: One of ``"front"``, ``"rear"``, ``"left"``, ``"right"``.
        max_dist_m: Search radius in metres.
        cone_half_angle_deg: Half-angle of the directional cone.

    Returns:
        ScenarioDict mapping candidate → {related → [timestamp_ns]}.
    """
    check_ids = _resolve_ids(candidates, tracks)
    related_ids = _resolve_ids(related_candidates, tracks) if related_candidates else None
    cos_thresh = np.cos(np.radians(cone_half_angle_deg))

    # Direction → reference unit vector in the candidate's local frame
    dir_map = {
        "front": np.array([1.0, 0.0]),
        "rear": np.array([-1.0, 0.0]),
        "left": np.array([0.0, 1.0]),
        "right": np.array([0.0, -1.0]),
    }
    if direction not in dir_map:
        raise ValueError(f"direction must be one of {list(dir_map.keys())}, got {direction!r}")
    ref_vec = dir_map[direction]

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        for rec in recs:
            nearby = tracks.get_at_timestamp(rec.timestamp_ns)
            for other in nearby:
                if other.track_id == tid:
                    continue
                if related_ids is not None and other.track_id not in related_ids:
                    continue
                dist = rec.distance_to(other)
                if dist > max_dist_m or dist < 1e-3:
                    continue
                # Compute relative position in candidate's frame
                rel = rec.relative_position(other)[:2]  # (forward, left)
                rel_norm = rel / (np.linalg.norm(rel) + 1e-9)
                cos_angle = float(np.dot(rel_norm, ref_vec))
                if cos_angle >= cos_thresh:
                    _add_to_result(result, tid, other.track_id, rec.timestamp_ns)

    return _drop_empty_timestamps(result)


def heading_in_relative_direction_to(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    related_candidates: ScenarioDict | None = None,
    direction: str = "same",
    max_dist_m: float = 50.0,
) -> ScenarioDict:
    """Return track pairs based on heading relationship.

    Args:
        tracks: Full TrackCollection.
        candidates: Primary track set.
        related_candidates: Optional restriction on related objects.
        direction: One of:
            ``"same"`` — headings within 45° of each other.
            ``"opposite"`` — headings 135°–180° apart (oncoming traffic).
            ``"perpendicular"`` — headings 45°–135° apart.
        max_dist_m: Maximum distance between the pair (m).

    Returns:
        ScenarioDict mapping candidate → {related → [timestamp_ns]}.
    """
    check_ids = _resolve_ids(candidates, tracks)
    related_ids = _resolve_ids(related_candidates, tracks) if related_candidates else None

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        for rec in recs:
            nearby = tracks.get_at_timestamp(rec.timestamp_ns)
            for other in nearby:
                if other.track_id == tid:
                    continue
                if related_ids is not None and other.track_id not in related_ids:
                    continue
                dist = rec.distance_to(other)
                if dist > max_dist_m:
                    continue
                cos_h = float(np.dot(rec.heading_vector[:2], other.heading_vector[:2]))
                angle_deg = float(np.degrees(np.arccos(np.clip(abs(cos_h), 0, 1))))

                if direction == "same" and cos_h > np.cos(np.radians(45)):
                    _add_to_result(result, tid, other.track_id, rec.timestamp_ns)
                elif direction == "opposite" and cos_h < np.cos(np.radians(135)):
                    _add_to_result(result, tid, other.track_id, rec.timestamp_ns)
                elif direction == "perpendicular" and 45 <= angle_deg <= 135:
                    _add_to_result(result, tid, other.track_id, rec.timestamp_ns)

    return _drop_empty_timestamps(result)


# ── Private helpers ────────────────────────────────────────────────────────────


def _resolve_ids(sd: ScenarioDict | None, tracks: TrackCollection) -> list[str]:
    if sd is None:
        return list(tracks.by_track.keys())
    return list(sd.keys())


def _add_to_result(
    result: ScenarioDict,
    track_id: str,
    related_id: str,
    timestamp_ns: int,
) -> None:
    if track_id not in result:
        result[track_id] = {}
    inner = result[track_id]
    if related_id not in inner:
        inner[related_id] = []
    inner[related_id].append(timestamp_ns)


def _drop_empty_timestamps(result: ScenarioDict) -> ScenarioDict:
    """Remove related entries with empty timestamp lists."""
    clean: ScenarioDict = {}
    for tid, related in result.items():
        clean_related = {rid: ts for rid, ts in related.items() if ts}
        if clean_related:
            clean[tid] = clean_related
    return clean
