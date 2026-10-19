"""DEPRECATED: StagedTrainer — replaced by hooks-based training in scripts/train.py.

This module is kept for backward compatibility only.  The active training path is
``scripts/train.py::train_with_mmengine()`` which uses ``StageSwitchHook``,
``TrainableParamCheckpointHook``, and ``OccEvalHook`` registered on a single
mmengine Runner.  StagedTrainer will be removed in a future release.

Stage 1: Freeze backbone + VLM; train spatial encoder + Q-Former + heads.
Stage 2: Unfreeze last N backbone layers (lr=1e-5).
Stage 3: Enable VLM LoRA (lr=1e-4 for LoRA params).
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    import mmengine
    from mmengine.hooks import CheckpointHook, LoggerHook
    from mmengine.optim import AmpOptimWrapper
    from mmengine.runner import Runner

    _HAS_MMENGINE = True
except ImportError:
    _HAS_MMENGINE = False

logger = logging.getLogger(__name__)


class StagedTrainer:
    """Three-stage training controller for SceneMiner.

    Each stage runs for a configurable number of epochs, loading the
    checkpoint saved by the previous stage before starting.

    Args:
        model: SceneMiner instance (or any nn.Module with
            ``freeze_for_stage1/2/3()`` methods).
        train_dataloader: PyTorch DataLoader or mmengine DataLoader config dict.
        cfg: Stage config dict. Recognised keys:

            - ``stage1_epochs`` (int, default 10): epochs for stage 1
            - ``stage2_epochs`` (int, default 5): epochs for stage 2
            - ``stage3_epochs`` (int, default 5): epochs for stage 3
            - ``base_lr`` (float, default 1e-4): learning rate for stages 1/2/3 params
            - ``stage2_backbone_lr`` (float, default 1e-5): lr for unfrozen backbone blocks
            - ``stage3_lora_lr`` (float, default 1e-4): lr for LoRA params
            - ``n_backbone_blocks`` (int, default 4): backbone blocks to unfreeze in stage 2
            - ``weight_decay`` (float, default 0.01)
            - ``grad_clip_norm`` (float, default 1.0)
            - ``amp`` (bool, default True): use AMP mixed-precision

        work_dir: Directory for checkpoints and logs.
        val_dataloader: Optional validation DataLoader (for val_cfg).
        val_evaluator: Optional mmengine evaluator config.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: Any,
        cfg: dict | None = None,
        work_dir: str = "work_dirs/scene_miner",
        val_dataloader: Any = None,
        val_evaluator: Any = None,
    ) -> None:
        warnings.warn(
            "StagedTrainer is deprecated and will be removed in a future version. "
            "Use scripts/train.py with mmengine Runner (train_with_mmengine) and "
            "StageSwitchHook instead — this is the active training path.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.val_evaluator = val_evaluator
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        cfg = cfg or {}
        self.stage1_epochs = int(cfg.get("stage1_epochs", 10))
        self.stage2_epochs = int(cfg.get("stage2_epochs", 5))
        self.stage3_epochs = int(cfg.get("stage3_epochs", 5))
        self.base_lr = float(cfg.get("base_lr", 1e-4))
        self.stage2_backbone_lr = float(cfg.get("stage2_backbone_lr", 1e-5))
        self.stage3_lora_lr = float(cfg.get("stage3_lora_lr", 1e-4))
        self.n_backbone_blocks = int(cfg.get("n_backbone_blocks", 4))
        self.weight_decay = float(cfg.get("weight_decay", 0.01))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
        self.use_amp = bool(cfg.get("amp", True))

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, resume_stage: int = 1) -> None:
        """Run the full staged training pipeline.

        Args:
            resume_stage: Stage to start from (1, 2, or 3).
                Previous stage checkpoint must exist in ``work_dir``.
        """
        if resume_stage not in (1, 2, 3):
            raise ValueError(f"resume_stage must be 1, 2, or 3; got {resume_stage}")

        if resume_stage <= 1:
            logger.info("=== Stage 1: training spatial encoder + Q-Former + heads ===")
            self._run_stage(stage=1, n_epochs=self.stage1_epochs, resume_from=None)

        if resume_stage <= 2:
            ckpt = self._find_stage_checkpoint(1)
            logger.info("=== Stage 2: + last %d backbone blocks (lr=%.1e) ===",
                        self.n_backbone_blocks, self.stage2_backbone_lr)
            self._run_stage(stage=2, n_epochs=self.stage2_epochs, resume_from=ckpt)

        if resume_stage <= 3:
            ckpt = self._find_stage_checkpoint(2)
            logger.info("=== Stage 3: + VLM LoRA adapters (lr=%.1e) ===",
                        self.stage3_lora_lr)
            self._run_stage(stage=3, n_epochs=self.stage3_epochs, resume_from=ckpt)

        logger.info("Staged training complete. Checkpoints in: %s", self.work_dir)

    # ── Stage execution ────────────────────────────────────────────────────────

    def _run_stage(self, stage: int, n_epochs: int, resume_from: str | None) -> None:
        """Apply freeze, build optimizer, run one training stage."""
        # 1. Apply the freeze policy for this stage
        self._apply_freeze(stage)

        if _HAS_MMENGINE:
            self._run_stage_mmengine(stage, n_epochs, resume_from)
        else:
            self._run_stage_pytorch(stage, n_epochs, resume_from)

    def _apply_freeze(self, stage: int) -> None:
        """Call the model's freeze method for the given stage."""
        if hasattr(self.model, "freeze_for_stage1") and stage == 1:
            self.model.freeze_for_stage1()
        elif hasattr(self.model, "freeze_for_stage2") and stage == 2:
            self.model.freeze_for_stage2(n_backbone_blocks=self.n_backbone_blocks)
        elif hasattr(self.model, "freeze_for_stage3") and stage == 3:
            self.model.freeze_for_stage3()

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Stage %d — trainable parameters: %s", stage, f"{n_trainable:,}")

    # ── mmengine path ──────────────────────────────────────────────────────────

    def _run_stage_mmengine(
        self, stage: int, n_epochs: int, resume_from: str | None
    ) -> None:
        """Build and run an mmengine Runner for one stage."""
        stage_work_dir = str(self.work_dir / f"stage{stage}")
        os.makedirs(stage_work_dir, exist_ok=True)

        optim_wrapper_cfg = self._build_optim_wrapper_cfg(stage)

        runner_cfg = dict(
            model=self.model,
            work_dir=stage_work_dir,
            train_dataloader=self.train_dataloader,
            optim_wrapper=optim_wrapper_cfg,
            train_cfg=dict(
                type="EpochBasedTrainLoop",
                max_epochs=n_epochs,
                val_interval=max(1, n_epochs // 5),
            ),
            param_scheduler=dict(
                type="CosineAnnealingLR",
                T_max=n_epochs,
                eta_min=self.base_lr * 0.01,
                begin=0,
                end=n_epochs,
                by_epoch=True,
            ),
            hooks=[
                dict(
                    type="CheckpointHook",
                    interval=1,
                    save_best="auto",
                    by_epoch=True,
                    max_keep_ckpts=2,
                ),
                dict(type="LoggerHook", interval=50),
                dict(type="IterTimerHook"),
            ],
            default_scope="scene_mining",
            resume=resume_from is not None,
            load_from=resume_from,
        )

        if self.val_dataloader is not None:
            runner_cfg["val_dataloader"] = self.val_dataloader
            runner_cfg["val_cfg"] = dict(type="ValLoop")
            if self.val_evaluator is not None:
                runner_cfg["val_evaluator"] = self.val_evaluator

        runner = Runner.from_cfg(mmengine.Config(runner_cfg))
        runner.train()

    def _build_optim_wrapper_cfg(self, stage: int) -> dict:
        """Build per-stage optim_wrapper config with per-group LRs."""
        if stage == 1:
            # All trainable params at base_lr
            return dict(
                type="OptimWrapper",
                optimizer=dict(
                    type="AdamW",
                    lr=self.base_lr,
                    weight_decay=self.weight_decay,
                ),
                clip_grad=dict(max_norm=self.grad_clip_norm),
            )

        if stage == 2:
            # Two param groups: backbone blocks at reduced LR, rest at base_lr
            return dict(
                type="OptimWrapper",
                optimizer=dict(
                    type="AdamW",
                    lr=self.base_lr,
                    weight_decay=self.weight_decay,
                    paramwise_cfg=dict(
                        custom_keys={
                            "backbone": dict(lr_mult=self.stage2_backbone_lr / self.base_lr),
                        }
                    ),
                ),
                clip_grad=dict(max_norm=self.grad_clip_norm),
            )

        # Stage 3: LoRA params at stage3_lora_lr, rest at base_lr
        return dict(
            type="OptimWrapper",
            optimizer=dict(
                type="AdamW",
                lr=self.base_lr,
                weight_decay=self.weight_decay,
                paramwise_cfg=dict(
                    custom_keys={
                        "lora_": dict(lr_mult=self.stage3_lora_lr / self.base_lr),
                    }
                ),
            ),
            clip_grad=dict(max_norm=self.grad_clip_norm),
        )

    # ── Pure-PyTorch fallback (no mmengine) ───────────────────────────────────

    def _run_stage_pytorch(
        self, stage: int, n_epochs: int, resume_from: str | None
    ) -> None:
        """Minimal PyTorch training loop (fallback when mmengine is absent)."""
        device = next(self.model.parameters()).device
        optimizer = self._build_pytorch_optimizer(stage)

        start_epoch = 0
        if resume_from and Path(resume_from).exists():
            checkpoint = torch.load(resume_from, map_location=device)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint.get("optimizer", {}))
            start_epoch = checkpoint.get("epoch", 0)
            logger.info("Resumed stage %d from %s (epoch %d)", stage, resume_from, start_epoch)

        scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=self.base_lr * 0.01
        )

        self.model.train()
        for epoch in range(start_epoch, n_epochs):
            for batch in self.train_dataloader:
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = self._compute_loss(batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = self._compute_loss(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                    optimizer.step()
            scheduler.step()
            logger.info("Stage %d  epoch %d/%d", stage, epoch + 1, n_epochs)

        ckpt_path = self._stage_checkpoint_path(stage)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": n_epochs,
                "stage": stage,
            },
            ckpt_path,
        )
        logger.info("Stage %d checkpoint saved to %s", stage, ckpt_path)

    def _build_pytorch_optimizer(self, stage: int) -> torch.optim.Optimizer:
        """Build AdamW with per-group LRs for the given stage."""
        if stage == 1:
            return torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.base_lr,
                weight_decay=self.weight_decay,
            )

        backbone_params, lora_params, other_params = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(p)
            elif "lora_" in name:
                lora_params.append(p)
            else:
                other_params.append(p)

        param_groups = [dict(params=other_params, lr=self.base_lr)]
        if backbone_params:
            param_groups.append(dict(params=backbone_params, lr=self.stage2_backbone_lr))
        if lora_params:
            param_groups.append(dict(params=lora_params, lr=self.stage3_lora_lr))

        return torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """Forward pass returning a scalar loss tensor.

        Override this method when using the pure-PyTorch fallback to
        supply your own loss computation.
        """
        raise NotImplementedError(
            "Override StagedTrainer._compute_loss() to provide a loss "
            "when running without mmengine."
        )

    # ── Checkpoint helpers ─────────────────────────────────────────────────────

    def _stage_checkpoint_path(self, stage: int) -> str:
        """Return the expected checkpoint path for the given stage."""
        # mmengine saves to stage{N}/epoch_{max_epochs}.pth by default;
        # the PyTorch fallback uses stage{N}_final.pth at the top level.
        mmengine_path = self.work_dir / f"stage{stage}" / f"epoch_{self._stage_epochs(stage)}.pth"
        if mmengine_path.exists():
            return str(mmengine_path)
        return str(self.work_dir / f"stage{stage}_final.pth")

    def _find_stage_checkpoint(self, stage: int) -> str | None:
        """Find the latest checkpoint for the given stage, or None."""
        path = self._stage_checkpoint_path(stage)
        if Path(path).exists():
            return path
        # Also check for best checkpoint saved by CheckpointHook
        best = self.work_dir / f"stage{stage}" / "best_auto.pth"
        if best.exists():
            return str(best)
        logger.warning(
            "No checkpoint found for stage %d. Expected at %s", stage, path
        )
        return None

    def _stage_epochs(self, stage: int) -> int:
        return {1: self.stage1_epochs, 2: self.stage2_epochs, 3: self.stage3_epochs}[stage]
