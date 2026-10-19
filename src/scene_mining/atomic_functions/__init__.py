"""Atomic scene query functions for program synthesis (RefProg API).

All functions share the same contract:
    fn(tracks, candidates, ...) -> ScenarioDict

The ``ScenarioDict`` maps track_uuid → {related_uuid → [timestamp_ns]}.
An empty inner dict means the candidate has no required related object.

This module re-exports all public atomic functions so that synthesised
programs can import them with a single ``from scene_mining.atomic_functions import *``.
"""

from .base import ScenarioDict, AtomicFunctionBase
from .category_functions import get_objects_of_category, is_category
from .logic_functions import scenario_and, scenario_or, scenario_not, reverse_relationship
from .map_functions import (
    at_pedestrian_crossing,
    at_stop_sign,
    in_drivable_area,
    near_intersection,
    on_intersection,
    on_lane_type,
    on_road,
)
from .output_functions import output_scenario
from .relationship_functions import (
    being_crossed_by,
    facing_toward,
    following,
    has_objects_in_relative_direction,
    heading_in_relative_direction_to,
    heading_toward,
    in_same_lane,
    near_objects,
)
from .state_functions import (
    accelerating,
    changing_lanes,
    decelerating,
    has_lateral_acceleration,
    has_velocity,
    moving,
    stationary,
    turning,
)

__all__ = [
    # Base
    "ScenarioDict",
    "AtomicFunctionBase",
    # Category
    "get_objects_of_category",
    "is_category",
    # State
    "moving",
    "stationary",
    "accelerating",
    "decelerating",
    "turning",
    "changing_lanes",
    "has_velocity",
    "has_lateral_acceleration",
    # Relationship
    "near_objects",
    "heading_toward",
    "facing_toward",
    "being_crossed_by",
    "following",
    "in_same_lane",
    "has_objects_in_relative_direction",
    "heading_in_relative_direction_to",
    # Map
    "on_intersection",
    "near_intersection",
    "at_pedestrian_crossing",
    "at_stop_sign",
    "in_drivable_area",
    "on_road",
    "on_lane_type",
    # Logic
    "scenario_and",
    "scenario_or",
    "scenario_not",
    "reverse_relationship",
    # Output
    "output_scenario",
]
