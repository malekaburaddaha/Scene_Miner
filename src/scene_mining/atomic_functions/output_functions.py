"""Output formatting atomic function.

``output_scenario`` is the mandatory final call in every synthesised
RefProg program. It validates and normalises the intermediate ScenarioDict
into the canonical wire format expected by the evaluator.
"""

from __future__ import annotations

from scene_mining.tracks import TrackCollection

from .base import ScenarioDict


def output_scenario(
    tracks: TrackCollection,
    scenario: ScenarioDict,
    min_timestamps: int = 1,
    deduplicate: bool = True,
) -> ScenarioDict:
    """Validate and return the final ScenarioDict for a query.

    This function serves as the mandatory return statement in synthesised
    RefProg programs. It performs lightweight validation and normalisation:

    1. Removes tracks not present in ``tracks.by_track`` (stale IDs).
    2. Removes related objects with fewer than ``min_timestamps`` entries.
    3. Deduplicates and sorts timestamp lists.
    4. Tracks with no valid related objects but an empty inner dict ({})
       are retained (they represent unrelated referred objects).

    Args:
        tracks: Full TrackCollection for the log (used for ID validation).
        scenario: Intermediate ScenarioDict from the program.
        min_timestamps: Minimum number of timestamps required for a
            related-object entry to be retained (default 1).
        deduplicate: Whether to deduplicate and sort timestamp lists.

    Returns:
        Normalised ScenarioDict ready for evaluation.

    Example (typical usage in a synthesised program)::

        vehicles = get_objects_of_category(tracks, "VEHICLE")
        pedestrians = get_objects_of_category(tracks, "PEDESTRIAN")
        near = near_objects(tracks, candidates=vehicles,
                            related_candidates=pedestrians, max_dist_m=5.0)
        return output_scenario(tracks, near)
    """
    valid_ids = set(tracks.by_track.keys())
    result: ScenarioDict = {}

    for tid, related in scenario.items():
        if tid not in valid_ids:
            continue

        if not related:
            # Empty related dict → referred object with no required relation
            result[tid] = {}
            continue

        clean_related: dict[str, list[int]] = {}
        for rid, ts_list in related.items():
            if rid not in valid_ids:
                continue
            if deduplicate:
                ts_list = sorted(set(ts_list))
            if len(ts_list) < min_timestamps:
                continue
            clean_related[rid] = ts_list

        if clean_related:
            result[tid] = clean_related

    return result
