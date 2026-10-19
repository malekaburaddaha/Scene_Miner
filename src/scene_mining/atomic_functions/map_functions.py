"""HD map atomic functions.

These functions query the AV2 HD map (ArgoverseStaticMap) to filter tracks
by their spatial relationship to map elements: intersections, pedestrian
crossings, stop signs, drivable area, lane types, etc.

When no HD map is available (map=None), functions gracefully degrade by
returning all candidates (no filtering applied) with a warning.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from scene_mining.tracks import TrackCollection

from .base import ScenarioDict

if TYPE_CHECKING:
    # Optional — only imported when actually used
    pass


def on_intersection(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    hd_map: Any | None = None,
) -> ScenarioDict:
    """Return tracks whose position is within an intersection region.

    Requires AV2 ArgoverseStaticMap. Falls back to identity (no filtering)
    when map is None.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        hd_map: av2.map.map_api.ArgoverseStaticMap instance or None.

    Returns:
        ScenarioDict with tracks on intersection at any timestamp.
    """
    return _map_filter(
        tracks, candidates, hd_map,
        check_fn=_check_on_intersection,
    )


def near_intersection(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    hd_map: Any | None = None,
    max_dist_m: float = 20.0,
) -> ScenarioDict:
    """Return tracks within max_dist_m of an intersection boundary.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        hd_map: ArgoverseStaticMap or None.
        max_dist_m: Distance threshold in metres (default 20.0).

    Returns:
        ScenarioDict with tracks near intersection.
    """
    return _map_filter(
        tracks, candidates, hd_map,
        check_fn=lambda pos, m: _check_near_intersection(pos, m, max_dist_m),
    )


def at_pedestrian_crossing(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    hd_map: Any | None = None,
    buffer_m: float = 2.0,
) -> ScenarioDict:
    """Return tracks at or near a pedestrian crossing (crosswalk).

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        hd_map: ArgoverseStaticMap or None.
        buffer_m: Extend crosswalk polygon by this margin (m).

    Returns:
        ScenarioDict with tracks at pedestrian crossings.
    """
    return _map_filter(
        tracks, candidates, hd_map,
        check_fn=lambda pos, m: _check_at_crosswalk(pos, m, buffer_m),
    )


def at_stop_sign(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    hd_map: Any | None = None,
    max_dist_m: float = 15.0,
) -> ScenarioDict:
    """Return tracks near a stop sign.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        hd_map: ArgoverseStaticMap or None.
        max_dist_m: Maximum distance to nearest stop sign (m).

    Returns:
        ScenarioDict with tracks near stop signs.
    """
    return _map_filter(
        tracks, candidates, hd_map,
        check_fn=lambda pos, m: _check_near_stop_sign(pos, m, max_dist_m),
    )


def in_drivable_area(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    hd_map: Any | None = None,
) -> ScenarioDict:
    """Return tracks whose position is within the drivable area.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        hd_map: ArgoverseStaticMap or None.

    Returns:
        ScenarioDict with tracks in drivable area.
    """
    return _map_filter(
        tracks, candidates, hd_map,
        check_fn=_check_in_drivable_area,
    )


def on_road(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    hd_map: Any | None = None,
) -> ScenarioDict:
    """Return tracks on road surface (lanes + bike lanes + intersections).

    This is broader than ``in_drivable_area`` — it also includes bike
    lanes and dedicated road segments not marked as drivable for cars.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        hd_map: ArgoverseStaticMap or None.

    Returns:
        ScenarioDict with tracks on road.
    """
    return _map_filter(
        tracks, candidates, hd_map,
        check_fn=_check_on_road,
    )


def on_lane_type(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    hd_map: Any | None = None,
    lane_type: str = "VEHICLE",
) -> ScenarioDict:
    """Return tracks on a specific lane type.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        hd_map: ArgoverseStaticMap or None.
        lane_type: AV2 LaneType string. Common values:
            ``"VEHICLE"``, ``"BIKE"``, ``"BUS"``.

    Returns:
        ScenarioDict with tracks on the specified lane type.
    """
    return _map_filter(
        tracks, candidates, hd_map,
        check_fn=lambda pos, m: _check_lane_type(pos, m, lane_type),
    )


# ── Internal map query helpers ────────────────────────────────────────────────


def _map_filter(
    tracks: TrackCollection,
    candidates: ScenarioDict | None,
    hd_map: Any | None,
    check_fn,
) -> ScenarioDict:
    """Generic map-based filter helper.

    Args:
        check_fn: Callable(position_xy: np.ndarray, hd_map) → bool
    """
    if hd_map is None:
        warnings.warn(
            "HD map not provided; map filter returning all candidates unfiltered.",
            RuntimeWarning,
            stacklevel=3,
        )
        if candidates is not None:
            return dict(candidates)
        return {tid: {} for tid in tracks.by_track}

    check_ids = list(candidates.keys()) if candidates is not None else list(tracks.by_track)

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        if not recs:
            continue
        # Check at any timestamp
        matched = any(
            check_fn(rec.translation_m[:2], hd_map) for rec in recs
        )
        if matched:
            result[tid] = {}

    return result


def _check_on_intersection(pos: np.ndarray, hd_map: Any) -> bool:
    """Check if position is inside any intersection polygon."""
    try:
        from shapely.geometry import Point
        pt = Point(float(pos[0]), float(pos[1]))
        for seg in hd_map.get_scenario_lane_segments():
            if seg.is_intersection:
                # Build polygon from lane boundary
                poly = _lane_segment_to_polygon(seg)
                if poly is not None and poly.contains(pt):
                    return True
    except Exception:
        pass
    return False


def _check_near_intersection(pos: np.ndarray, hd_map: Any, max_dist_m: float) -> bool:
    """Check if position is within max_dist_m of any intersection polygon."""
    try:
        from shapely.geometry import Point
        pt = Point(float(pos[0]), float(pos[1]))
        for seg in hd_map.get_scenario_lane_segments():
            if seg.is_intersection:
                poly = _lane_segment_to_polygon(seg)
                if poly is not None and pt.distance(poly) <= max_dist_m:
                    return True
    except Exception:
        pass
    return False


def _check_at_crosswalk(pos: np.ndarray, hd_map: Any, buffer_m: float) -> bool:
    """Check if position is at a pedestrian crosswalk."""
    try:
        from shapely.geometry import Point
        pt = Point(float(pos[0]), float(pos[1]))
        for crosswalk in hd_map.get_scenario_ped_crossings():
            # Crosswalk has .polygon attribute (3D, take XY)
            pts_xy = crosswalk.polygon[:, :2]
            from shapely.geometry import Polygon
            poly = Polygon(pts_xy).buffer(buffer_m)
            if poly.contains(pt):
                return True
    except Exception:
        pass
    return False


def _check_near_stop_sign(pos: np.ndarray, hd_map: Any, max_dist_m: float) -> bool:
    """Check if position is within max_dist_m of a stop sign."""
    try:
        for seg in hd_map.get_scenario_lane_segments():
            if hasattr(seg, "has_stop_sign") and seg.has_stop_sign:
                # Lane segment centroid as proxy for stop sign location
                centroid = _lane_segment_centroid(seg)
                if centroid is not None:
                    dist = np.linalg.norm(pos - centroid)
                    if dist <= max_dist_m:
                        return True
    except Exception:
        pass
    return False


def _check_in_drivable_area(pos: np.ndarray, hd_map: Any) -> bool:
    """Check if position is in the drivable area."""
    try:
        return bool(hd_map.get_raster_layer_points_boolean(
            pos.reshape(1, 2), "drivable_area"
        )[0])
    except Exception:
        pass
    return False


def _check_on_road(pos: np.ndarray, hd_map: Any) -> bool:
    """Check if position is on any road surface."""
    try:
        # Try drivable area first (fastest raster query)
        if _check_in_drivable_area(pos, hd_map):
            return True
        # Fallback: check lane segments
        from shapely.geometry import Point
        pt = Point(float(pos[0]), float(pos[1]))
        for seg in hd_map.get_scenario_lane_segments():
            poly = _lane_segment_to_polygon(seg)
            if poly is not None and poly.contains(pt):
                return True
    except Exception:
        pass
    return False


def _check_lane_type(pos: np.ndarray, hd_map: Any, lane_type: str) -> bool:
    """Check if position is on a lane of the specified type."""
    try:
        from shapely.geometry import Point
        pt = Point(float(pos[0]), float(pos[1]))
        for seg in hd_map.get_scenario_lane_segments():
            if str(seg.lane_type).upper() != lane_type.upper():
                continue
            poly = _lane_segment_to_polygon(seg)
            if poly is not None and poly.contains(pt):
                return True
    except Exception:
        pass
    return False


def _lane_segment_to_polygon(seg: Any):
    """Convert an AV2 LaneSegment to a Shapely Polygon, or None on failure."""
    try:
        from shapely.geometry import Polygon
        left = seg.left_lane_boundary.xyz[:, :2]
        right = seg.right_lane_boundary.xyz[:, :2]
        coords = np.vstack([left, right[::-1]])
        return Polygon(coords)
    except Exception:
        return None


def _lane_segment_centroid(seg: Any) -> np.ndarray | None:
    """Return the XY centroid of a lane segment, or None on failure."""
    try:
        left = seg.left_lane_boundary.xyz[:, :2]
        right = seg.right_lane_boundary.xyz[:, :2]
        return np.mean(np.vstack([left, right]), axis=0)
    except Exception:
        return None
