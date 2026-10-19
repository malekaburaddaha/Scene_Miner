"""TrainableParamCheckpointHook — saves only trainable parameters.

Registered in the mmengine HOOKS registry as ``"TrainableParamCheckpointHook"``.

Config usage::

    custom_hooks = [
        dict(type="TrainableParamCheckpointHook", interval=1, max_keep_ckpts=3),
    ]
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from pathlib import Path

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class TrainableParamCheckpointHook(Hook):
    """Save only parameters that require gradients at the end of each epoch.

    The default mmengine CheckpointHook serialises the full model to CPU RAM
    (~6-7 GB for Qwen2.5-VL-3B), which can trigger the OOM killer on nodes
    with limited host memory.  This hook saves only the trainable parameters
    (~450 MB in Stage 1: BEVFormer + QFormer + heads), which fits comfortably.

    Args:
        interval: Save every N epochs.
        max_keep_ckpts: Maximum number of checkpoints to keep on disk.
    """

    def __init__(self, interval: int = 1, max_keep_ckpts: int = 3) -> None:
        self._interval = interval
        self._max_keep = max_keep_ckpts
        self._saved: list[Path] = []

    def after_train_epoch(self, runner) -> None:
        # Only rank 0 writes checkpoints in distributed training.
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        epoch = runner.epoch  # 1-based after epoch completes
        if epoch % self._interval != 0:
            return

        out_dir = Path(runner.work_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"epoch_{epoch}_trainable.pth"

        # In DDP, access underlying module for parameter iteration.
        module = getattr(runner.model, "module", runner.model)
        trainable = {
            name: param.detach().cpu()
            for name, param in module.named_parameters()
            if param.requires_grad
        }
        torch.save({"epoch": epoch, "state_dict": trainable}, save_path)
        print(f"[ckpt] saved {len(trainable)} trainable tensors → {save_path}")

        self._saved.append(save_path)
        while len(self._saved) > self._max_keep:
            oldest = self._saved.pop(0)
            if oldest.exists():
                oldest.unlink()
