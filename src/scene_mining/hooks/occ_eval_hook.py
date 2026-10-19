"""OccEvalHook — per-epoch BEV occupancy evaluation and visualization.

Registered in the mmengine HOOKS registry as ``"OccEvalHook"``.

Config usage::

    custom_hooks = [
        dict(type="OccEvalHook", n_vis_frames=5, priority="LOW"),
    ]

The hook reads ``runner.cfg`` and ``runner.work_dir`` at runtime so no config
paths need to be duplicated in the hook dict.  The val split is read from
``runner.cfg.data.val_split`` (default: ``"val"``).

Evaluation frequency:
- Every epoch when ``occ_head`` has trainable parameters (stage0 / pre-training).
- Every 5 epochs when ``occ_head`` is frozen (stage1+) — for monitoring only.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from scene_mining.evaluation.occ_metrics import (
    save_occ_vis, compute_occ_metrics,
    save_semantic_occ_vis, compute_semantic_occ_metrics,
)


@HOOKS.register_module()
class OccEvalHook(Hook):
    """Evaluate BEV occupancy at the end of each (or every 5th) epoch.

    Metrics (mean IoU / precision / recall) are logged to the mmengine
    message_hub so they appear alongside training loss in the run log.
    The first ``n_vis_frames`` frames are rendered as 3-panel BEV figures
    and saved under ``{work_dir}/vis_occ/epoch_{N:03d}/``.

    Args:
        n_vis_frames: Number of frames to visualize per epoch.
    """

    def __init__(self, n_vis_frames: int = 5) -> None:
        self._n_vis_frames = n_vis_frames
        self._val_loader = None   # built lazily on first call

    # ── Runner lifecycle ───────────────────────────────────────────────────────

    def before_run(self, runner) -> None:
        self._cfg      = runner.cfg
        self._vis_base = Path(runner.work_dir) / "vis_occ"
        # Read val split from data config; default "val"
        data = runner.cfg.data
        if hasattr(data, "get"):
            self._val_split = data.get("val_split", "val")
        else:
            self._val_split = getattr(data, "val_split", "val")

    # ── Epoch hook ────────────────────────────────────────────────────────────

    def after_train_epoch(self, runner) -> None:
        # Rank 0 only in distributed training
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        module = getattr(runner.model, "module", runner.model)
        if not hasattr(module, "occ_head") or module.occ_head is None:
            return

        epoch = runner.epoch   # 1-based

        # When occ_head is frozen, only evaluate every 5 epochs (monitoring).
        occ_head_trainable = any(p.requires_grad for p in module.occ_head.parameters())
        if not occ_head_trainable and epoch % 5 != 0:
            return

        # Build val loader once (lazily)
        if self._val_loader is None:
            self._val_loader = self._build_val_loader()
        if self._val_loader is None:
            return

        device = next(module.parameters()).device
        epoch_vis_dir = self._vis_base / f"epoch_{epoch:03d}"

        was_training = module.training
        module.eval()

        all_metrics: list[dict] = []
        n_saved = 0
        semantic_mode: bool | None = None   # determined on first batch

        with torch.no_grad():
            for frame_idx, batch in enumerate(self._val_loader):
                occ_gt_b = batch.get("occ_gt")
                if occ_gt_b is None:
                    continue

                images     = batch["images"].to(device)
                intrinsics = batch["intrinsics"].to(device)
                extrinsics = batch["extrinsics"].to(device)
                occ_gt_b   = occ_gt_b.to(device)

                raw_hw = batch.get("image_hw")
                image_hw = _parse_image_hw(raw_hw)

                out = module(images, intrinsics, extrinsics, image_hw=image_hw)
                if out.occ_logits is None:
                    continue

                # Determine mode once from logits shape
                if semantic_mode is None:
                    semantic_mode = (out.occ_logits.dim() == 4)  # (B, C, H, W)

                if semantic_mode:
                    n_classes = out.occ_logits.shape[1]
                    pred_classes = out.occ_logits[0].argmax(0)   # (H, W)
                    gt_classes   = occ_gt_b[0].long()            # (H, W)
                    m = compute_semantic_occ_metrics(pred_classes, gt_classes, n_classes)
                    all_metrics.append(m)

                    if n_saved < self._n_vis_frames:
                        save_semantic_occ_vis(
                            gt_classes=gt_classes,
                            pred_classes=pred_classes,
                            miou=m["miou"],
                            per_class_iou={k: v for k, v in m.items() if k != "miou"},
                            epoch=epoch,
                            frame_idx=frame_idx,
                            save_path=epoch_vis_dir / f"frame_{frame_idx:04d}.png",
                            n_classes=n_classes,
                        )
                        n_saved += 1
                else:
                    prob    = torch.sigmoid(out.occ_logits[0])
                    pred    = prob >= 0.5
                    gt_bool = occ_gt_b[0].bool()
                    m = compute_occ_metrics(pred, gt_bool)
                    all_metrics.append(m)

                    if n_saved < self._n_vis_frames:
                        save_occ_vis(
                            gt=occ_gt_b[0].float(),
                            pred_prob=prob,
                            pred_binary=pred,
                            iou=m["iou"], prec=m["precision"], rec=m["recall"],
                            epoch=epoch,
                            frame_idx=frame_idx,
                            save_path=epoch_vis_dir / f"frame_{frame_idx:04d}.png",
                        )
                        n_saved += 1

        if was_training:
            module.train()

        if not all_metrics:
            return

        if semantic_mode:
            mean_miou = float(np.mean([m["miou"] for m in all_metrics]))
            runner.message_hub.update_scalar("val/occ_miou", mean_miou)
            # Log per-class IoU averaged across frames
            all_keys = [k for k in all_metrics[0] if k != "miou"]
            for key in all_keys:
                vals = [m[key] for m in all_metrics if key in m]
                runner.message_hub.update_scalar(f"val/occ_{key}_iou", float(np.mean(vals)))
            print(
                f"[occ_eval] epoch {epoch:3d}  mIoU={mean_miou:.4f}  "
                f"n={len(all_metrics)}  vis→{epoch_vis_dir}/"
            )
        else:
            mean_iou  = float(np.mean([m["iou"] for m in all_metrics]))
            mean_prec = float(np.mean([m["precision"] for m in all_metrics]))
            mean_rec  = float(np.mean([m["recall"] for m in all_metrics]))
            runner.message_hub.update_scalar("val/occ_iou",      mean_iou)
            runner.message_hub.update_scalar("val/occ_precision", mean_prec)
            runner.message_hub.update_scalar("val/occ_recall",    mean_rec)
            print(
                f"[occ_eval] epoch {epoch:3d}  "
                f"IoU={mean_iou:.4f}  P={mean_prec:.4f}  R={mean_rec:.4f}  "
                f"n={len(all_metrics)}  vis→{epoch_vis_dir}/"
            )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_val_loader(self):
        from scene_mining.datasets.builder import build_val_dataloader_for_eval

        loader = build_val_dataloader_for_eval(
            self._cfg, split=self._val_split, batch_size=1, num_workers=2
        )
        if loader is None:
            print(f"[occ_eval] Could not build val dataset ({self._val_split!r})")
        else:
            print(f"[occ_eval] val dataset ({self._val_split!r}): "
                  f"{len(loader.dataset)} frames")
        return loader


def _parse_image_hw(raw_hw) -> tuple[int, int] | None:
    """Parse the image_hw field from a DataLoader batch into (H, W) or None."""
    if isinstance(raw_hw, torch.Tensor):
        return (int(raw_hw[0, 0]), int(raw_hw[0, 1]))
    if isinstance(raw_hw, (list, tuple)) and len(raw_hw) == 2:
        h, w = raw_hw[0], raw_hw[1]
        return (
            int(h[0]) if isinstance(h, torch.Tensor) else int(h),
            int(w[0]) if isinstance(w, torch.Tensor) else int(w),
        )
    return None
