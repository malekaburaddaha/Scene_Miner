"""Build the system prompt for RefProg LLM code synthesis.

The system prompt injects the full atomic function API documentation so
the LLM can write correct Python code without hallucinating function
signatures.
"""

from __future__ import annotations

import inspect

# Import all atomic functions to auto-extract docstrings
from scene_mining.atomic_functions import (
    accelerating,
    at_pedestrian_crossing,
    at_stop_sign,
    being_crossed_by,
    changing_lanes,
    decelerating,
    facing_toward,
    following,
    get_objects_of_category,
    has_lateral_acceleration,
    has_objects_in_relative_direction,
    has_velocity,
    heading_in_relative_direction_to,
    heading_toward,
    in_drivable_area,
    in_same_lane,
    is_category,
    moving,
    near_intersection,
    near_objects,
    on_intersection,
    on_lane_type,
    on_road,
    output_scenario,
    reverse_relationship,
    scenario_and,
    scenario_not,
    scenario_or,
    stationary,
    turning,
)

# Visual functions (optional — included when feature_cache is available)
from scene_mining.atomic_functions.visual_functions import (
    has_occlusion,
    has_visual_attribute,
    is_color,
    is_scene_description,
    is_time_of_day,
    is_weather,
)

_BASE_FUNCTIONS = [
    get_objects_of_category, is_category,
    moving, stationary, accelerating, decelerating,
    turning, changing_lanes, has_velocity, has_lateral_acceleration,
    near_objects, heading_toward, facing_toward, being_crossed_by,
    following, in_same_lane, has_objects_in_relative_direction,
    heading_in_relative_direction_to,
    on_intersection, near_intersection, at_pedestrian_crossing,
    at_stop_sign, in_drivable_area, on_road, on_lane_type,
    scenario_and, scenario_or, scenario_not, reverse_relationship,
    output_scenario,
]

_VISUAL_FUNCTIONS = [
    is_color, is_weather, is_time_of_day, has_visual_attribute,
    has_occlusion, is_scene_description,
]


def build_system_prompt(
    include_visual: bool = True,
    include_examples: bool = True,
) -> str:
    """Build the full system prompt for RefProg synthesis.

    Args:
        include_visual: Include visual atomic function API (requires FeatureCache).
        include_examples: Include in-context composition examples.

    Returns:
        System prompt string to pass to the LLM.
    """
    sections = [_HEADER]
    sections.append(_build_api_section(_BASE_FUNCTIONS, title="## Base Atomic Functions"))
    if include_visual:
        sections.append(
            _build_api_section(_VISUAL_FUNCTIONS, title="## Visual Atomic Functions")
        )
        sections.append(_VISUAL_NOTE)
    sections.append(_DIRECTION_DISAMBIGUATION)
    sections.append(_SCENARIO_DICT_SECTION)
    if include_examples:
        sections.append(_EXAMPLES_SECTION)
    sections.append(_INSTRUCTIONS)
    return "\n\n".join(sections)


def build_user_prompt(
    query: str,
    log_id: str | None = None,
    batch_queries: list[str] | None = None,
) -> str:
    """Build the user-turn prompt for a single query or a batch.

    Args:
        query: Natural language scene query.
        log_id: Optional log identifier (for context).
        batch_queries: If provided, synthesise code that handles ALL
            queries in a single function (batch mode for consistency).

    Returns:
        User prompt string.
    """
    if batch_queries:
        queries_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(batch_queries))
        return (
            f"Write a Python function for each of the following queries.\n"
            f"Queries:\n{queries_str}\n\n"
            f"Return a dict mapping query index (1-based) to ScenarioDict."
        )
    ctx = f" (log_id={log_id!r})" if log_id else ""
    return f"Query{ctx}: {query}\n\nWrite a Python function `run(tracks, hd_map=None, feature_cache=None)` that returns the ScenarioDict."


# ── Prompt sections ────────────────────────────────────────────────────────────

_HEADER = """\
You are an expert autonomous driving scenario mining system. Your task is to
write Python code that uses the provided atomic function API to identify
objects and timestamps matching a natural language scene description.

You have access to a `tracks` (TrackCollection) and optionally `hd_map`
(ArgoverseStaticMap) and `feature_cache` (FeatureCache) objects."""


_DIRECTION_DISAMBIGUATION = """\
## CRITICAL: Direction Parameter Semantics

The `direction` parameter in spatial functions ALWAYS refers to the location
of the RELATED object relative to the CANDIDATE track — NOT an absolute
compass direction.

Examples:
- `has_objects_in_relative_direction(tracks, vehicles, pedestrians, direction="front")`
  → finds vehicles that have pedestrians IN FRONT OF THEM
- `heading_in_relative_direction_to(tracks, A, B, direction="opposite")`
  → finds A-B pairs where B is heading in the OPPOSITE direction of A

This convention is opposite to what might be intuitively expected. Always think:
"The related object is [direction] of the candidate track.\""""


_SCENARIO_DICT_SECTION = """\
## ScenarioDict Format

All functions return a `ScenarioDict`:
```python
ScenarioDict = dict[str, dict[str, list[int]]]
# track_uuid → {related_uuid → [timestamp_ns, ...]}
# Empty inner dict {} = referred object, no required relation
```

You MUST call `output_scenario(tracks, <result>)` as the final return value.
Never return raw intermediate ScenarioDicts."""


_VISUAL_NOTE = """\
## Visual Functions Note

Visual functions (`is_color`, `is_weather`, etc.) require `feature_cache` to be
provided. Always check `if feature_cache is not None` before calling them, or
accept that the function will raise RuntimeError if cache is missing.

For HYBRID queries, apply geometric filters first (faster), then visual filters
on the reduced candidate set."""


_EXAMPLES_SECTION = """\
## Examples

### Example 1: Vehicle following cyclist at intersection
```python
def run(tracks, hd_map=None, feature_cache=None):
    cyclists = get_objects_of_category(tracks, "BICYCLIST")
    vehicles = get_objects_of_category(tracks, "VEHICLE")
    at_intersection = on_intersection(tracks, candidates=vehicles, hd_map=hd_map)
    following_vehicles = following(
        tracks,
        candidates=at_intersection,
        related_candidates=cyclists,
        max_dist_m=15.0,
    )
    return output_scenario(tracks, following_vehicles)
```

### Example 2: Pedestrian crossing in front of stationary vehicle
```python
def run(tracks, hd_map=None, feature_cache=None):
    vehicles = stationary(tracks, get_objects_of_category(tracks, "VEHICLE"))
    pedestrians = get_objects_of_category(tracks, "PEDESTRIAN")
    crossing = being_crossed_by(
        tracks,
        candidates=vehicles,
        related_candidates=pedestrians,
    )
    return output_scenario(tracks, crossing)
```

### Example 3: Red car near intersection (hybrid: geometry + visual)
```python
def run(tracks, hd_map=None, feature_cache=None):
    cars = get_objects_of_category(tracks, "REGULAR_VEHICLE")
    near_int = near_intersection(tracks, candidates=cars, hd_map=hd_map)
    if feature_cache is not None:
        red_cars = is_color(tracks, candidates=near_int, color="red",
                            feature_cache=feature_cache)
    else:
        red_cars = near_int  # degrade gracefully without visual features
    return output_scenario(tracks, red_cars)
```

### Example 4: Two vehicles heading in opposite directions
```python
def run(tracks, hd_map=None, feature_cache=None):
    vehicles = get_objects_of_category(tracks, "VEHICLE")
    oncoming = heading_in_relative_direction_to(
        tracks,
        candidates=vehicles,
        related_candidates=vehicles,
        direction="opposite",
        max_dist_m=30.0,
    )
    return output_scenario(tracks, oncoming)
```"""


_INSTRUCTIONS = """\
## Instructions

1. Write a single Python function `run(tracks, hd_map=None, feature_cache=None)`.
2. Import atomic functions at the top of the function body or assume they are
   already in scope (they are injected into the execution namespace).
3. Use `candidates=` to chain filters efficiently (avoid re-scanning all tracks).
4. Call `output_scenario(tracks, result)` as the final return.
5. Do NOT use for-loops to iterate over tracks manually — use atomic functions.
6. Do NOT import external libraries (numpy, shapely, etc. are available if needed).
7. Return ONLY the Python code block, no explanation."""


def _build_api_section(functions: list, title: str) -> str:
    """Build an API documentation section from function docstrings."""
    lines = [title]
    for fn in functions:
        sig = _get_signature(fn)
        doc = inspect.getdoc(fn) or ""
        # Keep only first paragraph of docstring (up to first blank line)
        short_doc = doc.split("\n\n")[0].strip()
        lines.append(f"\n### `{fn.__name__}{sig}`\n{short_doc}")
    return "\n".join(lines)


def _get_signature(fn) -> str:
    """Return a compact function signature string."""
    try:
        sig = inspect.signature(fn)
        return str(sig)
    except Exception:
        return "(...)"
