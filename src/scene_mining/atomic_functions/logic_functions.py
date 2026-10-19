"""Boolean logic combinators over ScenarioDicts.

These functions compose the outputs of other atomic functions to express
AND, OR, NOT, and relationship-reversal operations.
"""

from __future__ import annotations

from .base import ScenarioDict


def scenario_and(
    *scenarios: ScenarioDict,
) -> ScenarioDict:
    """Intersection of multiple ScenarioDicts (AND semantics).

    Returns only tracks present in ALL input ScenarioDicts. The related-
    object structure of the FIRST scenario is preserved for retained tracks.

    For timestamp-level AND (intersect active timestamps), use the
    ``candidates`` parameter of individual atomic functions instead.

    Args:
        *scenarios: Two or more ScenarioDicts to intersect.

    Returns:
        ScenarioDict with tracks present in all inputs.

    Example::

        cyclists_near_intersection = scenario_and(
            get_objects_of_category(tracks, "BICYCLIST"),
            near_intersection(tracks, hd_map=hd_map),
        )
    """
    if not scenarios:
        return {}
    result = dict(scenarios[0])
    for sd in scenarios[1:]:
        keep = set(result.keys()) & set(sd.keys())
        result = {tid: result[tid] for tid in keep}
    return result


def scenario_or(
    *scenarios: ScenarioDict,
) -> ScenarioDict:
    """Union of multiple ScenarioDicts (OR semantics).

    Returns all tracks present in ANY input ScenarioDict. When a track
    appears in multiple inputs, the related-object dict is merged (union
    of related IDs; union of timestamp lists).

    Args:
        *scenarios: Two or more ScenarioDicts to union.

    Returns:
        ScenarioDict with all tracks from any input.

    Example::

        all_hazards = scenario_or(
            get_objects_of_category(tracks, "PEDESTRIAN"),
            get_objects_of_category(tracks, "BICYCLIST"),
        )
    """
    result: ScenarioDict = {}
    for sd in scenarios:
        for tid, related in sd.items():
            if tid not in result:
                result[tid] = {}
            for rid, ts_list in related.items():
                if rid in result[tid]:
                    result[tid][rid] = sorted(set(result[tid][rid]) | set(ts_list))
                else:
                    result[tid][rid] = list(ts_list)
    return result


def scenario_not(
    all_tracks: ScenarioDict,
    excluded: ScenarioDict,
) -> ScenarioDict:
    """Set difference: tracks in ``all_tracks`` but NOT in ``excluded``.

    Args:
        all_tracks: Base ScenarioDict (e.g. all vehicles).
        excluded: ScenarioDict whose track IDs should be removed.

    Returns:
        ScenarioDict with excluded track IDs removed.

    Example::

        non_stationary_vehicles = scenario_not(
            get_objects_of_category(tracks, "VEHICLE"),
            stationary(tracks),
        )
    """
    excluded_ids = set(excluded.keys())
    return {tid: rel for tid, rel in all_tracks.items() if tid not in excluded_ids}


def reverse_relationship(
    scenario: ScenarioDict,
) -> ScenarioDict:
    """Swap referred and related objects in a ScenarioDict.

    Transforms {A: {B: [ts]}} → {B: {A: [ts]}}.

    This is useful for changing the perspective of a query. For example,
    ``following(tracks, vehicles, cyclists)`` returns vehicles following
    cyclists. Applying ``reverse_relationship`` gives the cyclists being
    followed by vehicles.

    Args:
        scenario: Input ScenarioDict with (referred → related) mappings.

    Returns:
        Reversed ScenarioDict with (related → {referred → timestamps}).

    Example::

        cyclists_being_followed = reverse_relationship(
            following(tracks, candidates=vehicles, related_candidates=cyclists)
        )
    """
    result: ScenarioDict = {}
    for tid, related in scenario.items():
        for rid, ts_list in related.items():
            if rid not in result:
                result[rid] = {}
            if tid in result[rid]:
                result[rid][tid] = sorted(set(result[rid][tid]) | set(ts_list))
            else:
                result[rid][tid] = list(ts_list)
    return result
