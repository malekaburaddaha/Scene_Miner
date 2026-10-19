"""StageSwitchHook — manages frozen/unfrozen parameter groups across training stages.

Registered in the mmengine HOOKS registry as ``"StageSwitchHook"``.

Config usage::

    custom_hooks = [
        dict(type="StageSwitchHook", priority="HIGH"),
    ]

All stage epoch counts and per-stage LRs are read from ``runner.cfg.train``
at ``before_run`` time so they don't need to be duplicated in the hook config.
"""

from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _rebuild_optimizer(runner, param_groups: list[dict]) -> None:
    """Update optimizer param groups WITHOUT clearing existing Adam state.

    mmengine wraps the optimizer in OptimWrapper; we access the raw optimizer
    via ``runner.optim_wrapper.optimizer``.

    Strategy:
    - Preserve ``opt.state`` for params that remain trainable so Adam momentum
      is continuous across stage transitions (eliminates loss spikes).
    - Prune ``opt.state`` only for params no longer in any group (free memory).
    - New params added in this transition have no state yet — AdamW initialises
      them lazily on the first optimizer step (standard behaviour).
    """
    opt = runner.optim_wrapper.optimizer

    old_param_ids = {id(p) for pg in opt.param_groups for p in pg["params"]}
    new_param_ids = {id(p) for pg in param_groups for p in pg["params"]}

    # Replace group metadata only — do NOT touch opt.state
    opt.param_groups.clear()
    for pg in param_groups:
        opt.add_param_group(pg)

    # Prune state for params that are no longer trainable
    stale_ids = old_param_ids - new_param_ids
    stale_params = [p for p in list(opt.state.keys()) if id(p) in stale_ids]
    for p in stale_params:
        del opt.state[p]


@HOOKS.register_module()
class StageSwitchHook(Hook):
    """Switch frozen/unfrozen parameter groups at stage-boundary epochs.

    Reads stage configuration from ``runner.cfg.train`` so the hook config
    needs no params — all values come from the training schedule config.

    Supports both 3-stage (stage1/2/3) and 4-stage (stage0/1/2/3) schedules.
    After each transition, the optimizer's param groups are rebuilt in-place
    while preserving existing Adam state for continuity.
    """

    def before_run(self, runner) -> None:
        tr = runner.cfg.train

        self._stage0_epochs     = getattr(tr, "stage0_epochs", 0)
        self._stage1_epochs     = tr.stage1_epochs
        self._stage2_epochs     = tr.stage2_epochs
        self._stage3_epochs     = tr.stage3_epochs

        # 1-based epoch boundaries (inclusive end of each stage)
        self._end_stage0 = self._stage0_epochs
        self._end_stage1 = self._end_stage0 + self._stage1_epochs
        self._end_stage2 = self._end_stage1 + self._stage2_epochs

        self._base_lr           = tr.base_lr
        self._backbone_lr       = tr.stage2_backbone_lr
        self._lora_lr           = tr.stage3_lora_lr
        self._n_backbone_blocks = getattr(tr, "n_backbone_blocks", 4)

    def after_train_epoch(self, runner) -> None:
        epoch  = runner.epoch   # 1-based
        module = getattr(runner.model, "module", runner.model)

        if (self._stage0_epochs > 0
                and epoch == self._end_stage0
                and self._stage1_epochs > 0):
            print(f"[stage] epoch {epoch}: stage0 → stage1")
            module.freeze_for_stage1()
            _rebuild_optimizer(runner, [
                dict(params=module.trainable_parameters(), lr=self._base_lr),
            ])

        elif (self._stage1_epochs > 0
              and epoch == self._end_stage1
              and self._stage2_epochs > 0):
            n = self._n_backbone_blocks
            print(f"[stage] epoch {epoch}: stage1 → stage2 "
                  f"(unfreeze last {n} backbone blocks)")
            module.freeze_for_stage2(n)
            backbone_params = [
                p for _, p in module.backbone.named_parameters() if p.requires_grad
            ]
            other_params = [
                p for name, p in module.named_parameters()
                if p.requires_grad and not name.startswith("backbone.")
            ]
            _rebuild_optimizer(runner, [
                dict(params=other_params,   lr=self._base_lr),
                dict(params=backbone_params, lr=self._backbone_lr),
            ])

        elif (self._stage2_epochs > 0
              and epoch == self._end_stage2
              and self._stage3_epochs > 0):
            print(f"[stage] epoch {epoch}: stage2 → stage3 (enable VLM LoRA)")
            module.freeze_for_stage3()
            lora_params = [
                p for _, p in module.vlm.named_parameters() if p.requires_grad
            ]
            other_params = [
                p for name, p in module.named_parameters()
                if p.requires_grad and not name.startswith("vlm.")
            ]
            _rebuild_optimizer(runner, [
                dict(params=other_params, lr=self._base_lr),
                dict(params=lora_params,  lr=self._lora_lr),
            ])
