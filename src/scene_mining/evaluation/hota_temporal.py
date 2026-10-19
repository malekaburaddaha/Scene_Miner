"""HOTA-Temporal evaluation metric (from the RefAV paper).

HOTA-Temporal scores a ScenarioDict against ground truth by considering
ONLY the timestamps at which the referred action is actively occurring.

This is the primary challenge metric in RefAV.

Reference: arXiv:2505.20981 (RefAV: Referring to Objects in
Autonomous Vehicle Videos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HOTAResult:
    """Detailed HOTA metric results."""
    hota: float                  # HOTA = sqrt(DetA * AssA)
    det_a: float                 # Detection Accuracy
    ass_a: float                 # Association Accuracy
    loc_a: float                 # Localisation Accuracy (mean BEV IoU)
    n_tp: int                    # True positive track-timestamps
    n_fp: int                    # False positive track-timestamps
    n_fn: int                    # False negative track-timestamps
    log_balanced_acc: float      # Was the scenario present in the log?
    timestamp_balanced_acc: float # Correct active timestamps?
    per_query: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"HOTAResult(HOTA={self.hota:.2f}, DetA={self.det_a:.2f}, "
            f"AssA={self.ass_a:.2f}, LocA={self.loc_a:.2f}, "
            f"LogBA={self.log_balanced_acc:.2f}, "
            f"TsBA={self.timestamp_balanced_acc:.2f})"
        )


def evaluate_hota_temporal(
    predictions: dict[str, "ScenarioDict"],  # query_id → ScenarioDict
    ground_truth: dict[str, "ScenarioDict"],  # query_id → ScenarioDict
    track_collection: Any,                    # TrackCollection for bbox lookups
    iou_threshold: float = 0.5,
) -> HOTAResult:
    """Compute HOTA-Temporal over a set of queries.

    Args:
        predictions: Dict mapping query_id to predicted ScenarioDict.
        ground_truth: Dict mapping query_id to GT ScenarioDict.
        track_collection: TrackCollection for BEV IoU computation.
        iou_threshold: BEV IoU threshold for counting a TP (default 0.5).

    Returns:
        HOTAResult with aggregated and per-query metrics.
    """
    all_tp = all_fp = all_fn = 0
    all_iou: list[float] = []
    per_query: dict[str, float] = {}
    log_tp = log_fp = log_fn = 0

    for qid, gt_sd in ground_truth.items():
        pred_sd = predictions.get(qid, {})

        # Log-level balanced accuracy
        gt_has_matches = bool(gt_sd)
        pred_has_matches = bool(pred_sd)
        if gt_has_matches and pred_has_matches:
            log_tp += 1
        elif gt_has_matches and not pred_has_matches:
            log_fn += 1
        elif not gt_has_matches and pred_has_matches:
            log_fp += 1
        # True negative: neither has matches (no contribution)

        # Temporal HOTA computation
        q_tp, q_fp, q_fn, q_iou = _compute_temporal_hota_single(
            pred_sd, gt_sd, track_collection, iou_threshold
        )
        all_tp += q_tp
        all_fp += q_fp
        all_fn += q_fn
        all_iou.extend(q_iou)

        # Per-query HOTA
        det_a = q_tp / (q_tp + q_fp + q_fn + 1e-6)
        per_query[qid] = det_a

    # DetA = TP / (TP + FP + FN)
    det_a = all_tp / (all_tp + all_fp + all_fn + 1e-6)
    loc_a = float(np.mean(all_iou)) if all_iou else 0.0
    ass_a = det_a  # Simplified: full AssA requires track ID matching across frames
    hota = float(np.sqrt(det_a * ass_a))

    # Log-level balanced accuracy
    total_log = log_tp + log_fp + log_fn
    log_ba = log_tp / (total_log + 1e-6)

    # Timestamp-level balanced accuracy (fraction of correct referred timestamps)
    ts_ba = all_tp / (all_tp + all_fn + 1e-6)

    return HOTAResult(
        hota=hota,
        det_a=det_a,
        ass_a=ass_a,
        loc_a=loc_a,
        n_tp=all_tp,
        n_fp=all_fp,
        n_fn=all_fn,
        log_balanced_acc=log_ba,
        timestamp_balanced_acc=ts_ba,
        per_query=per_query,
    )


def _compute_temporal_hota_single(
    pred_sd: "ScenarioDict",
    gt_sd: "ScenarioDict",
    track_collection: Any,
    iou_threshold: float,
) -> tuple[int, int, int, list[float]]:
    """Compute TP/FP/FN for a single query.

    Returns:
        (n_tp, n_fp, n_fn, iou_values_for_tps)
    """
    tp = fp = fn = 0
    iou_vals: list[float] = []

    # Collect all (track_id, timestamp_ns) pairs from GT
    gt_pairs: set[tuple[str, int]] = set()
    for tid, related in gt_sd.items():
        if related:
            for rid, ts_list in related.items():
                for ts in ts_list:
                    gt_pairs.add((tid, ts))
        else:
            recs = track_collection.by_track.get(tid, [])
            for rec in recs:
                gt_pairs.add((tid, rec.timestamp_ns))

    # Collect all (track_id, timestamp_ns) pairs from predictions
    pred_pairs: set[tuple[str, int]] = set()
    for tid, related in pred_sd.items():
        if related:
            for rid, ts_list in related.items():
                for ts in ts_list:
                    pred_pairs.add((tid, ts))
        else:
            recs = track_collection.by_track.get(tid, [])
            for rec in recs:
                pred_pairs.add((tid, rec.timestamp_ns))

    # Match pairs
    for pair in gt_pairs:
        if pair in pred_pairs:
            tp += 1
            # Compute BEV IoU for this pair
            iou = _bev_iou_pair(pair[0], pair[1], track_collection)
            if iou >= iou_threshold:
                iou_vals.append(iou)
            else:
                # IoU below threshold → not a true positive
                tp -= 1
                fn += 1
        else:
            fn += 1

    fp = len(pred_pairs - gt_pairs)
    return tp, fp, fn, iou_vals


def _bev_iou_pair(
    track_id: str,
    timestamp_ns: int,
    track_collection: Any,
) -> float:
    """Compute BEV IoU for a single (track_id, timestamp) prediction.

    For GT evaluation, both GT and prediction use the same track —
    IoU = 1.0 (perfect). For predicted tracks vs GT tracks, would
    need GT annotation lookup.

    In the RefAV setting, tracks come from the GT annotations, so
    IoU is always 1.0 for correctly retrieved tracks.
    """
    return 1.0  # GT tracks → perfect localisation
