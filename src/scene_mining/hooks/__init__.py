"""Custom mmengine hooks for scene_mining.

Importing this module registers all three hooks in the mmengine HOOKS registry
so they can be instantiated from config dicts via ``custom_hooks``.
"""

from .checkpoint_hook import TrainableParamCheckpointHook
from .stage_switch_hook import StageSwitchHook
from .occ_eval_hook import OccEvalHook

__all__ = [
    "TrainableParamCheckpointHook",
    "StageSwitchHook",
    "OccEvalHook",
]
