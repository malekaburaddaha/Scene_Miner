"""Shared BEV occupancy metrics and visualization utilities.

Used by both the in-training OccEvalHook (scripts/train.py) and the
standalone evaluation script (scripts/eval_occ.py).  Centralising here
breaks the fragile cross-script import that eval_occ.py previously had
on train.py.
"""

from __future__ import annotations

from pathlib import Path

import torch

__all__ = ["save_occ_vis", "compute_occ_metrics", "save_semantic_occ_vis", "compute_semantic_occ_metrics"]

# Semantic class names — 17 classes (0=background, 1-10=dynamic, 11-16=static)
# Matches PANOPTIC_CATEGORY_TO_CLASS mapping in nuscenes_dataset.py.
OCC_CLASS_NAMES = [
    "background",           # 0
    "car",                  # 1
    "truck",                # 2
    "construction_vehicle", # 3
    "bus",                  # 4
    "trailer",              # 5
    "motorcycle",           # 6
    "bicycle",              # 7
    "pedestrian",           # 8
    "traffic_cone",         # 9
    "barrier",              # 10
    "driveable_surface",    # 11
    "flat_other",           # 12
    "sidewalk",             # 13
    "terrain",              # 14
    "manmade",              # 15
    "vegetation",           # 16
]

# Per-class RGB colours for semantic BEV visualisation
OCC_CLASS_COLORS = [
    (0.93, 0.93, 0.93),  # 0:  background         — light grey
    (0.10, 0.30, 0.90),  # 1:  car                — blue
    (0.20, 0.55, 0.90),  # 2:  truck              — light blue
    (0.00, 0.80, 0.80),  # 3:  construction_veh   — cyan
    (0.00, 0.75, 0.40),  # 4:  bus                — teal-green
    (0.10, 0.60, 0.10),  # 5:  trailer            — green
    (1.00, 0.50, 0.00),  # 6:  motorcycle         — orange
    (1.00, 0.85, 0.00),  # 7:  bicycle            — yellow
    (0.90, 0.10, 0.10),  # 8:  pedestrian         — red
    (0.90, 0.00, 0.90),  # 9:  traffic_cone       — magenta
    (0.50, 0.00, 0.50),  # 10: barrier            — purple
    (0.60, 0.60, 0.60),  # 11: driveable_surface  — mid grey
    (0.75, 0.70, 0.55),  # 12: flat_other         — tan
    (0.55, 0.55, 0.80),  # 13: sidewalk           — lavender
    (0.40, 0.70, 0.30),  # 14: terrain            — olive green
    (0.70, 0.55, 0.35),  # 15: manmade            — brown
    (0.20, 0.65, 0.20),  # 16: vegetation         — dark green
]


def save_occ_vis(
    gt,           # (H, W) float32 tensor {0,1}
    pred_prob,    # (H, W) float32 tensor [0,1]
    pred_binary,  # (H, W) bool tensor
    iou: float,
    prec: float,
    rec: float,
    epoch: int,
    frame_idx: int,
    save_path: "Path",
) -> None:
    """Save 3-panel BEV occupancy figure: GT | probability heatmap | TP/FP/FN overlay."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    gt_np   = gt.cpu().float().numpy()
    prob_np = pred_prob.cpu().float().numpy()
    pred_np = pred_binary.cpu().numpy().astype(np.float32)
    H, W    = gt_np.shape
    cy, cx  = H // 2, W // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Epoch {epoch}  Frame {frame_idx}  |  "
        f"IoU={iou:.3f}  P={prec:.3f}  R={rec:.3f}",
        fontsize=11,
    )

    # Panel 1: GT
    axes[0].imshow(gt_np, cmap="Blues", origin="lower", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[0].plot(cx, cy, "r+", markersize=10, markeredgewidth=2, label="ego")
    axes[0].set_title("GT BEV Occupancy")
    axes[0].set_xlabel("BEV x"); axes[0].set_ylabel("BEV y")
    axes[0].legend(fontsize=8)

    # Panel 2: Predicted probability
    im = axes[1].imshow(prob_np, cmap="hot", origin="lower", vmin=0, vmax=1,
                        interpolation="nearest")
    axes[1].plot(cx, cy, "b+", markersize=10, markeredgewidth=2)
    axes[1].set_title("Predicted Occupancy Probability")
    axes[1].set_xlabel("BEV x")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: TP/FP/FN overlay
    gt_b   = gt_np > 0.5
    pred_b = pred_np > 0.5
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[gt_b   & pred_b]  = [0.0, 0.8, 0.0]    # TP: green
    overlay[~gt_b  & pred_b]  = [0.9, 0.1, 0.1]    # FP: red
    overlay[gt_b   & ~pred_b] = [0.1, 0.3, 0.9]    # FN: blue
    overlay[~gt_b  & ~pred_b] = [0.95, 0.95, 0.95] # TN: light grey
    axes[2].imshow(overlay, origin="lower", interpolation="nearest")
    axes[2].plot(cx, cy, "k+", markersize=10, markeredgewidth=2)
    axes[2].set_title("Overlay  (green=TP  red=FP  blue=FN)")
    axes[2].set_xlabel("BEV x")
    axes[2].legend(handles=[
        mpatches.Patch(color=[0.0, 0.8, 0.0], label="TP"),
        mpatches.Patch(color=[0.9, 0.1, 0.1], label="FP"),
        mpatches.Patch(color=[0.1, 0.3, 0.9], label="FN"),
        mpatches.Patch(color=[0.95, 0.95, 0.95], label="TN"),
    ], loc="upper right", fontsize=8)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def compute_occ_metrics(
    pred_binary: torch.Tensor,   # (H, W) bool
    gt: torch.Tensor,             # (H, W) bool
) -> dict[str, float]:
    """Return IoU, precision, recall, F1 for binary BEV occupancy."""
    TP = (pred_binary & gt).sum().item()
    FP = (pred_binary & ~gt).sum().item()
    FN = (~pred_binary & gt).sum().item()
    iou       = TP / (TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return dict(iou=iou, precision=precision, recall=recall, f1=f1)


def compute_semantic_occ_metrics(
    pred_classes: torch.Tensor,  # (H, W) int64, values in [0, n_classes-1]
    gt_classes: torch.Tensor,    # (H, W) int64
    n_classes: int = 11,
) -> dict[str, float]:
    """Per-class IoU and mIoU over foreground classes (1..n_classes-1)."""
    pred = pred_classes.cpu()
    gt   = gt_classes.cpu()
    per_class_iou: dict[str, float] = {}
    iou_values = []
    for c in range(1, n_classes):   # skip background (0)
        tp = ((pred == c) & (gt == c)).sum().item()
        fp = ((pred == c) & (gt != c)).sum().item()
        fn = ((pred != c) & (gt == c)).sum().item()
        iou = tp / (tp + fp + fn + 1e-6)
        name = OCC_CLASS_NAMES[c] if c < len(OCC_CLASS_NAMES) else f"class_{c}"
        per_class_iou[name] = iou
        if (gt == c).any():   # only count classes present in GT for mIoU
            iou_values.append(iou)
    miou = float(sum(iou_values) / len(iou_values)) if iou_values else 0.0
    return dict(miou=miou, **per_class_iou)


def save_semantic_occ_vis(
    gt_classes,       # (H, W) int64 tensor
    pred_classes,     # (H, W) int64 tensor
    miou: float,
    per_class_iou: dict[str, float],
    epoch: int,
    frame_idx: int,
    save_path: "Path",
    n_classes: int = 11,
) -> None:
    """Save 2-panel semantic BEV figure: GT class map | Predicted class map."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    def _to_rgb(class_map):
        H, W = class_map.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        for c in range(n_classes):
            color = OCC_CLASS_COLORS[c] if c < len(OCC_CLASS_COLORS) else (0.5, 0.5, 0.5)
            rgb[class_map == c] = color
        return rgb

    gt_np   = gt_classes.cpu().numpy()
    pred_np = pred_classes.cpu().numpy()
    H, W    = gt_np.shape
    cy, cx  = H // 2, W // 2

    gt_rgb   = _to_rgb(gt_np)
    pred_rgb = _to_rgb(pred_np)

    # Top-5 present foreground classes for legend
    present = sorted(
        [(iou, name) for name, iou in per_class_iou.items() if iou > 0],
        reverse=True,
    )[:5]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"Epoch {epoch}  Frame {frame_idx}  |  mIoU={miou:.3f}",
        fontsize=11,
    )

    axes[0].imshow(gt_rgb, origin="lower", interpolation="nearest")
    axes[0].plot(cx, cy, "w+", markersize=10, markeredgewidth=2)
    axes[0].set_title("GT Semantic BEV")
    axes[0].set_xlabel("BEV x"); axes[0].set_ylabel("BEV y")

    axes[1].imshow(pred_rgb, origin="lower", interpolation="nearest")
    axes[1].plot(cx, cy, "w+", markersize=10, markeredgewidth=2)
    axes[1].set_title("Predicted Semantic BEV")
    axes[1].set_xlabel("BEV x")

    patches = [
        mpatches.Patch(color=OCC_CLASS_COLORS[c], label=OCC_CLASS_NAMES[c])
        for c in range(1, n_classes)
        if c < len(OCC_CLASS_NAMES)
    ]
    axes[1].legend(handles=patches, loc="upper right", fontsize=7, ncol=2)

    # Per-class IoU annotation
    if present:
        iou_text = "\n".join(f"{name}: {iou:.3f}" for iou, name in present)
        axes[0].text(0.02, 0.02, iou_text, transform=axes[0].transAxes,
                     fontsize=7, verticalalignment="bottom",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
