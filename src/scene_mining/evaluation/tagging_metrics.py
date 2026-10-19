"""Multi-label tagging evaluation metrics.

Functions:
- compute_micro_f1: Micro-averaged F1
- compute_macro_f1: Macro-averaged F1
- compute_tag_map: Mean Average Precision across tags

All functions accept ``torch.Tensor`` inputs so they can be called
directly inside a training loop or from an evaluation script.

Usage::

    import torch
    from scene_mining.evaluation.tagging_metrics import (
        compute_micro_f1, compute_macro_f1, compute_tag_map,
    )

    preds   = torch.rand(64, 20)   # [N, T] logits or probabilities
    targets = torch.randint(0, 2, (64, 20)).float()  # [N, T] binary

    print(compute_micro_f1(preds, targets))
    print(compute_macro_f1(preds, targets))
    print(compute_tag_map(preds, targets))
"""

from __future__ import annotations

import torch


def compute_micro_f1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute micro-averaged F1 across all tags and samples.

    Micro-averaging pools TP, FP, FN across all tags before computing F1,
    giving equal weight to each *sample-tag* pair.

    Args:
        preds: ``[N, T]`` float tensor (logits or probabilities).
        targets: ``[N, T]`` binary float tensor (ground-truth labels).
        threshold: Decision threshold on sigmoid-transformed predictions.

    Returns:
        Micro-F1 in [0, 1].
    """
    pred_bin = _binarise(preds, threshold)
    tp = (pred_bin * targets).sum().item()
    fp = (pred_bin * (1 - targets)).sum().item()
    fn = ((1 - pred_bin) * targets).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def compute_macro_f1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute macro-averaged F1 across tags.

    Macro-averaging computes F1 independently for each tag and averages,
    giving equal weight to each *tag*.

    Args:
        preds: ``[N, T]`` float tensor (logits or probabilities).
        targets: ``[N, T]`` binary float tensor.
        threshold: Decision threshold.

    Returns:
        Macro-F1 in [0, 1].
    """
    pred_bin = _binarise(preds, threshold)
    T = preds.shape[1]
    f1s = []

    for t in range(T):
        tp = (pred_bin[:, t] * targets[:, t]).sum().item()
        fp = (pred_bin[:, t] * (1 - targets[:, t])).sum().item()
        fn = ((1 - pred_bin[:, t]) * targets[:, t]).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)

    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def compute_tag_map(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute mean Average Precision (mAP) across tags.

    For each tag, computes the Average Precision of ranking samples by
    predicted score, then averages across all tags that have at least
    one positive sample.

    Args:
        preds: ``[N, T]`` float tensor (logits or probabilities; higher
            values are treated as more confident positives).
        targets: ``[N, T]`` binary float tensor.

    Returns:
        Tag mAP in [0, 1].
    """
    preds = preds.float()
    targets = targets.float()
    T = preds.shape[1]
    aps = []

    for t in range(T):
        t_targets = targets[:, t]
        n_pos = t_targets.sum().item()
        if n_pos == 0:
            continue  # skip tags with no positives

        t_scores = preds[:, t]
        sorted_idx = t_scores.argsort(descending=True)
        t_targets_sorted = t_targets[sorted_idx]

        n_hits = 0.0
        running_precision = 0.0
        for rank, gt in enumerate(t_targets_sorted.tolist(), start=1):
            if gt:
                n_hits += 1
                running_precision += n_hits / rank

        ap = running_precision / n_pos if n_pos > 0 else 0.0
        aps.append(ap)

    return float(sum(aps) / len(aps)) if aps else 0.0


# ── Private helpers ────────────────────────────────────────────────────────────


def _binarise(preds: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply sigmoid if preds are logits, then threshold to binary."""
    preds = preds.float()
    if preds.min() < 0 or preds.max() > 1:
        preds = torch.sigmoid(preds)
    return (preds >= threshold).float()
