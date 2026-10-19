"""Visual attribute atomic functions (requires FeatureCache).

These functions bridge the RefProg (Path B) pipeline with the deep
learning (Path A) pipeline via the FeatureCache. They extend the base
28 RefAV atomic functions to cover visual attributes that cannot be
expressed geometrically.

All functions accept an optional ``feature_cache`` argument. When None,
they raise a RuntimeError because visual queries cannot be answered
without visual features.

FeatureCache is imported lazily to avoid circular imports and to keep
the atomic_functions module usable in environments without the full
deep learning stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scene_mining.tracks import TrackCollection

from .base import ScenarioDict

if TYPE_CHECKING:
    pass  # FeatureCache type hint — imported lazily at runtime


def is_color(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    color: str = "red",
    feature_cache: Any | None = None,
    similarity_threshold: float = 0.25,
) -> ScenarioDict:
    """Return tracks whose dominant color matches ``color``.

    Uses SigLIP2 zero-shot text-image cosine similarity on per-track
    crop embeddings from the FeatureCache.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        color: Color name (e.g. ``"red"``, ``"white"``, ``"silver"``).
        feature_cache: FeatureCache instance (required).
        similarity_threshold: Minimum cosine similarity to count as a match.

    Returns:
        ScenarioDict with color-matching tracks.
    """
    _require_cache(feature_cache, "is_color")
    check_ids = list(candidates.keys()) if candidates is not None else list(tracks.by_track)

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        if not recs:
            continue
        # Check at first available timestamp (color is typically stable)
        for rec in recs:
            sim = feature_cache.color_similarity(rec, color)
            if sim >= similarity_threshold:
                result[tid] = {}
                break

    return result


def is_weather(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    condition: str = "rainy",
    feature_cache: Any | None = None,
    similarity_threshold: float = 0.20,
) -> ScenarioDict:
    """Return tracks (or all candidates) if the scene weather matches.

    Weather is a scene-level attribute, so ALL candidates in the log
    are either included or excluded together based on scene-level
    SigLIP2 similarity to the weather description.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        condition: Weather description (e.g. ``"rainy"``, ``"foggy"``,
            ``"snowy"``, ``"clear"``).
        feature_cache: FeatureCache instance (required).
        similarity_threshold: Minimum scene-level cosine similarity.

    Returns:
        ScenarioDict (all candidates if weather matches, empty if not).
    """
    _require_cache(feature_cache, "is_weather")
    sim = feature_cache.weather_similarity(condition)
    if sim < similarity_threshold:
        return {}

    check_ids = list(candidates.keys()) if candidates is not None else list(tracks.by_track)
    return {tid: {} for tid in check_ids}


def is_time_of_day(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    time: str = "night",
    feature_cache: Any | None = None,
    similarity_threshold: float = 0.20,
) -> ScenarioDict:
    """Return tracks if the scene matches the time-of-day description.

    Like ``is_weather``, this is a scene-level predicate applied uniformly
    to all candidates.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        time: Time description (e.g. ``"night"``, ``"dawn"``, ``"dusk"``,
            ``"day"``).
        feature_cache: FeatureCache instance (required).
        similarity_threshold: Minimum scene-level cosine similarity.

    Returns:
        ScenarioDict (all candidates if time matches, empty if not).
    """
    _require_cache(feature_cache, "is_time_of_day")
    sim = feature_cache.time_of_day_similarity(time)
    if sim < similarity_threshold:
        return {}

    check_ids = list(candidates.keys()) if candidates is not None else list(tracks.by_track)
    return {tid: {} for tid in check_ids}


def has_visual_attribute(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    attribute: str = "wearing a helmet",
    feature_cache: Any | None = None,
    similarity_threshold: float = 0.22,
    use_vlm_vqa: bool = False,
) -> ScenarioDict:
    """Return tracks that have the specified visual attribute.

    Two modes:
    - ``use_vlm_vqa=False`` (default): SigLIP2 zero-shot similarity on
      per-track crop embeddings. Fast, handles simple attributes.
    - ``use_vlm_vqa=True``: Qwen2.5-VL VQA on per-track crops for complex
      compositional attributes. Slower but more accurate.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        attribute: Natural-language attribute description.
        feature_cache: FeatureCache instance (required).
        similarity_threshold: Cosine similarity threshold (SigLIP2 mode).
        use_vlm_vqa: Use VLM VQA instead of zero-shot similarity.

    Returns:
        ScenarioDict with attribute-matching tracks.
    """
    _require_cache(feature_cache, "has_visual_attribute")
    check_ids = list(candidates.keys()) if candidates is not None else list(tracks.by_track)

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        if not recs:
            continue

        if use_vlm_vqa:
            # VLM VQA path — ask the VLM about the track crop
            matched = feature_cache.vlm_vqa(recs[0], f"Does this object have {attribute}?")
        else:
            # SigLIP2 zero-shot similarity
            matched = feature_cache.attribute_similarity(recs[0], attribute) >= similarity_threshold

        if matched:
            result[tid] = {}

    return result


def has_occlusion(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    feature_cache: Any | None = None,
    min_occlusion_fraction: float = 0.5,
) -> ScenarioDict:
    """Return tracks that are significantly occluded in camera views.

    Occlusion is computed from the visibility score recorded in
    FeatureCache during bbox projection — pixels outside the image or
    behind closer objects reduce the visibility score.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        feature_cache: FeatureCache instance (required).
        min_occlusion_fraction: Fraction of bbox that must be occluded
            (1.0 - visibility_score). Default 0.5 (≥50% occluded).

    Returns:
        ScenarioDict with heavily occluded tracks.
    """
    _require_cache(feature_cache, "has_occlusion")
    check_ids = list(candidates.keys()) if candidates is not None else list(tracks.by_track)

    result: ScenarioDict = {}
    for tid in check_ids:
        recs = tracks.by_track.get(tid, [])
        for rec in recs:
            vis = feature_cache.get_visibility_score(rec)
            if vis is not None and (1.0 - vis) >= min_occlusion_fraction:
                result[tid] = {}
                break

    return result


def is_scene_description(
    tracks: TrackCollection,
    candidates: ScenarioDict | None = None,
    description: str = "highway scene at night",
    feature_cache: Any | None = None,
    similarity_threshold: float = 0.30,
) -> ScenarioDict:
    """Return tracks if the overall scene matches a free-form description.

    Uses the 256-d BEV embedding (from SceneMiner backbone) compared
    against the VLM caption embedding for the description.

    Args:
        tracks: Full TrackCollection.
        candidates: Optional pre-filter ScenarioDict.
        description: Free-form scene description.
        feature_cache: FeatureCache instance (required).
        similarity_threshold: Minimum cosine similarity.

    Returns:
        ScenarioDict (all candidates if scene matches, empty if not).
    """
    _require_cache(feature_cache, "is_scene_description")
    sim = feature_cache.scene_description_similarity(description)
    if sim < similarity_threshold:
        return {}

    check_ids = list(candidates.keys()) if candidates is not None else list(tracks.by_track)
    return {tid: {} for tid in check_ids}


# ── Private helpers ────────────────────────────────────────────────────────────


def _require_cache(cache: Any | None, fn_name: str) -> None:
    if cache is None:
        raise RuntimeError(
            f"{fn_name}() requires a FeatureCache instance. "
            "Pass feature_cache=<cache> or use a HYBRID execution path "
            "that builds the cache before running visual atomic functions."
        )
