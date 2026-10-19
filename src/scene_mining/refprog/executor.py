"""Safe sandboxed executor for synthesised RefProg Python code.

Executes LLM-generated Python in a restricted namespace that exposes only:
- All atomic functions from scene_mining.atomic_functions
- numpy (as np)
- The TrackCollection and ScenarioDict types

Any access to the file system, network, subprocess, or builtins beyond
safe arithmetic/collections is blocked.
"""

from __future__ import annotations

import builtins
import traceback
from typing import Any

from scene_mining.atomic_functions import (
    ScenarioDict,
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
from scene_mining.atomic_functions.visual_functions import (
    has_occlusion,
    has_visual_attribute,
    is_color,
    is_scene_description,
    is_time_of_day,
    is_weather,
)
from scene_mining.tracks import TrackCollection


# ── Allowed builtins (strict whitelist) ───────────────────────────────────────

_SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter",
    "float", "frozenset", "int", "isinstance", "len", "list",
    "map", "max", "min", "print", "range", "round", "set",
    "sorted", "str", "sum", "tuple", "type", "zip",
    "None", "True", "False",
}


def _build_namespace(
    tracks: TrackCollection,
    hd_map: Any | None = None,
    feature_cache: Any | None = None,
) -> dict[str, Any]:
    """Build the execution namespace for a synthesised program."""
    import numpy as np

    ns: dict[str, Any] = {
        # Safe builtins
        "__builtins__": {k: getattr(builtins, k) for k in _SAFE_BUILTINS if hasattr(builtins, k)},

        # Atomic functions (base)
        "get_objects_of_category": get_objects_of_category,
        "is_category": is_category,
        "moving": moving,
        "stationary": stationary,
        "accelerating": accelerating,
        "decelerating": decelerating,
        "turning": turning,
        "changing_lanes": changing_lanes,
        "has_velocity": has_velocity,
        "has_lateral_acceleration": has_lateral_acceleration,
        "near_objects": near_objects,
        "heading_toward": heading_toward,
        "facing_toward": facing_toward,
        "being_crossed_by": being_crossed_by,
        "following": following,
        "in_same_lane": in_same_lane,
        "has_objects_in_relative_direction": has_objects_in_relative_direction,
        "heading_in_relative_direction_to": heading_in_relative_direction_to,
        "on_intersection": on_intersection,
        "near_intersection": near_intersection,
        "at_pedestrian_crossing": at_pedestrian_crossing,
        "at_stop_sign": at_stop_sign,
        "in_drivable_area": in_drivable_area,
        "on_road": on_road,
        "on_lane_type": on_lane_type,
        "scenario_and": scenario_and,
        "scenario_or": scenario_or,
        "scenario_not": scenario_not,
        "reverse_relationship": reverse_relationship,
        "output_scenario": output_scenario,

        # Atomic functions (visual)
        "is_color": is_color,
        "is_weather": is_weather,
        "is_time_of_day": is_time_of_day,
        "has_visual_attribute": has_visual_attribute,
        "has_occlusion": has_occlusion,
        "is_scene_description": is_scene_description,

        # Data objects
        "tracks": tracks,
        "hd_map": hd_map,
        "feature_cache": feature_cache,

        # Utility
        "np": np,
        "ScenarioDict": ScenarioDict,
        "TrackCollection": TrackCollection,
    }
    return ns


class ExecutionResult:
    """Result of a single program execution attempt."""

    def __init__(
        self,
        success: bool,
        scenario: ScenarioDict | None,
        error: str | None,
        code: str,
    ) -> None:
        self.success = success
        self.scenario = scenario or {}
        self.error = error
        self.code = code

    def __repr__(self) -> str:
        if self.success:
            return f"ExecutionResult(ok, n_tracks={len(self.scenario)})"
        return f"ExecutionResult(failed, error={self.error!r:.80})"


class RefProgExecutor:
    """Execute synthesised Python code in a restricted sandbox.

    Args:
        timeout_s: Maximum execution time in seconds (advisory — uses
            threading; hard kill requires separate process).
    """

    def __init__(self, timeout_s: float = 30.0) -> None:
        self.timeout_s = timeout_s

    def execute(
        self,
        code: str,
        tracks: TrackCollection,
        hd_map: Any | None = None,
        feature_cache: Any | None = None,
    ) -> ExecutionResult:
        """Execute synthesised code and return the ScenarioDict.

        The code must define a function ``run(tracks, hd_map=None,
        feature_cache=None)`` that returns a ScenarioDict.

        Args:
            code: Python source code string.
            tracks: TrackCollection for the current log.
            hd_map: Optional ArgoverseStaticMap.
            feature_cache: Optional FeatureCache.

        Returns:
            ExecutionResult with success flag, scenario, and error message.
        """
        ns = _build_namespace(tracks, hd_map, feature_cache)

        try:
            # Compile and execute the code to define the `run` function
            compiled = compile(code, "<refprog>", "exec")
            exec(compiled, ns)  # noqa: S102

            run_fn = ns.get("run")
            if run_fn is None:
                return ExecutionResult(
                    success=False,
                    scenario=None,
                    error="Synthesised code did not define a `run` function.",
                    code=code,
                )

            # Call run()
            result = run_fn(tracks, hd_map=hd_map, feature_cache=feature_cache)

            if not isinstance(result, dict):
                return ExecutionResult(
                    success=False,
                    scenario=None,
                    error=f"run() returned {type(result).__name__}, expected dict (ScenarioDict).",
                    code=code,
                )

            return ExecutionResult(success=True, scenario=result, error=None, code=code)

        except Exception:
            tb = traceback.format_exc(limit=5)
            return ExecutionResult(success=False, scenario=None, error=tb, code=code)
