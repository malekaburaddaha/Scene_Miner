"""Category filter atomic functions.

These functions filter a TrackCollection by object category and return
a ScenarioDict containing only tracks of the requested class(es).
"""

from __future__ import annotations

from scene_mining.tracks import TrackCollection
from scene_mining.tracks.track import VEHICLE_CATEGORIES

from .base import ScenarioDict


def get_objects_of_category(
    tracks: TrackCollection,
    category: str,
    candidates: ScenarioDict | None = None,
) -> ScenarioDict:
    """Return all track IDs whose class_name matches ``category``.

    The special alias ``"VEHICLE"`` expands to all vehicle sub-categories
    (REGULAR_VEHICLE, LARGE_VEHICLE, BUS, etc.) as defined in
    ``VEHICLE_CATEGORIES``.

    Args:
        tracks: Full TrackCollection for the log.
        category: AV2 category string or the alias ``"VEHICLE"``.
        candidates: Optional pre-filtered ScenarioDict. When provided,
            only tracks present in ``candidates`` are checked.

    Returns:
        ScenarioDict mapping matching track_ids to ``{}`` (no related object).
    """
    if category == "VEHICLE":
        target_cats = VEHICLE_CATEGORIES
    else:
        target_cats = frozenset({category})

    check_ids = list(candidates.keys()) if candidates is not None else list(tracks.by_track)

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        if not recs:
            continue
        # Use majority class_name (first occurrence for GT; last for predicted)
        if recs[0].class_name in target_cats:
            result[tid] = {}

    return result


def is_category(
    tracks: TrackCollection,
    candidates: ScenarioDict,
    category: str,
) -> ScenarioDict:
    """Filter an existing ScenarioDict to keep only tracks in ``category``.

    Unlike ``get_objects_of_category``, this function *requires* a
    ``candidates`` dict and preserves the related-object structure.

    Args:
        tracks: Full TrackCollection.
        candidates: Existing ScenarioDict to filter.
        category: AV2 category or ``"VEHICLE"`` alias.

    Returns:
        Filtered ScenarioDict retaining only matching candidates.
    """
    if category == "VEHICLE":
        target_cats = VEHICLE_CATEGORIES
    else:
        target_cats = frozenset({category})

    result: ScenarioDict = {}
    for tid, related in candidates.items():
        recs = tracks.by_track.get(tid, [])
        if recs and recs[0].class_name in target_cats:
            result[tid] = related

    return result
