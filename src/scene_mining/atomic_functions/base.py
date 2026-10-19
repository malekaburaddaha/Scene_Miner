"""Base types and contracts for the atomic function API.

ScenarioDict is the central data structure flowing through all atomic
functions. It encodes the answer to a spatio-temporal query as a mapping:

    track_uuid  →  {related_uuid → [timestamp_ns, ...]}

Semantics:
  - A track_uuid present in the dict is a *referred* object.
  - The inner dict maps each *related* object UUID to the list of
    timestamps at which the relationship holds.
  - An empty inner dict ({}) means the referred object has no required
    related object (e.g. ``moving(tracks)`` produces this).
  - An empty list [] for a related UUID means the related object exists
    but the relationship held at no sampled timestamp (effectively absent;
    functions should not emit these).

Example::

    {
        "abc-1234": {
            "xyz-5678": [100000000, 200000000, 300000000],
        },
        "def-9999": {},
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scene_mining.tracks import TrackCollection

# Central type alias — import this everywhere instead of writing out the full type
ScenarioDict = dict[str, dict[str, list[int]]]


class AtomicFunctionBase(ABC):
    """Abstract base class for atomic query functions.

    Concrete implementations are typically module-level functions (not
    class instances), but this ABC documents the contract that all
    atomic functions must satisfy and provides shared helpers.

    The function signature is::

        fn(tracks: TrackCollection,
           candidates: ScenarioDict | None = None,
           **kwargs) -> ScenarioDict

    Args:
        tracks: Full TrackCollection for the driving log.
        candidates: Optional pre-filtered ScenarioDict from a previous
            atomic function. When provided, only the track IDs in
            ``candidates`` are evaluated. This enables efficient function
            composition (AND-filter semantics).
        **kwargs: Function-specific parameters (thresholds, category
            names, etc.).

    Returns:
        ScenarioDict with entries for tracks that satisfy the predicate.
    """

    @abstractmethod
    def __call__(
        self,
        tracks: "TrackCollection",
        candidates: ScenarioDict | None = None,
        **kwargs,
    ) -> ScenarioDict:
        ...

    # ── Shared helpers ────────────────────────────────────────────────────

    @staticmethod
    def candidate_ids(candidates: ScenarioDict | None, tracks: "TrackCollection") -> list[str]:
        """Return the list of candidate track IDs to evaluate.

        If ``candidates`` is None, returns all track IDs in ``tracks``.
        Otherwise returns only those present in ``candidates``.
        """
        if candidates is None:
            return list(tracks.by_track.keys())
        return list(candidates.keys())

    @staticmethod
    def intersect_timestamps(
        a: list[int], b: list[int] | None
    ) -> list[int]:
        """Return timestamps present in both ``a`` and ``b``.

        If ``b`` is None (no prior constraint), returns ``a`` unchanged.
        """
        if b is None:
            return a
        b_set = set(b)
        return [t for t in a if t in b_set]

    @staticmethod
    def merge_related(
        existing: dict[str, list[int]],
        new_related: dict[str, list[int]],
    ) -> dict[str, list[int]]:
        """Union-merge two related-object dicts (extend timestamp lists)."""
        merged = dict(existing)
        for uid, ts_list in new_related.items():
            if uid in merged:
                merged[uid] = sorted(set(merged[uid]) | set(ts_list))
            else:
                merged[uid] = ts_list
        return merged


def empty_scenario() -> ScenarioDict:
    """Return an empty ScenarioDict (no matches)."""
    return {}


def single_track(track_id: str, timestamps: list[int]) -> ScenarioDict:
    """Return a ScenarioDict with one unrelated referred track."""
    return {track_id: {}}


def merge_scenarios(*scenarios: ScenarioDict) -> ScenarioDict:
    """Union-merge multiple ScenarioDicts (OR semantics for tracks)."""
    result: ScenarioDict = {}
    for sd in scenarios:
        for tid, related in sd.items():
            if tid in result:
                # Merge related dicts
                for rid, ts_list in related.items():
                    if rid in result[tid]:
                        result[tid][rid] = sorted(set(result[tid][rid]) | set(ts_list))
                    else:
                        result[tid][rid] = ts_list
            else:
                result[tid] = dict(related)
    return result
