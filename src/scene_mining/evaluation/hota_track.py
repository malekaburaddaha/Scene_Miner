"""HOTA-Track evaluation metric.

HOTA-Track counts ALL timestamps in a referred track as TP (not just the
active-action timestamps). This is the easier, less informative variant
compared to HOTA-Temporal.

See hota_temporal.py for the primary challenge metric.
"""

from __future__ import annotations

from scene_mining.evaluation.hota_temporal import HOTAResult, evaluate_hota_temporal
from scene_mining.atomic_functions.base import ScenarioDict


def evaluate_hota_track(
    predictions: dict[str, ScenarioDict],
    ground_truth: dict[str, ScenarioDict],
    track_collection,
    iou_threshold: float = 0.5,
) -> HOTAResult:
    """Compute HOTA-Track (all track timestamps counted, not just active).

    Converts temporal ScenarioDicts to full-track ScenarioDicts by
    expanding each referred track to include ALL its timestamps, then
    calls evaluate_hota_temporal.

    Args:
        predictions: Dict mapping query_id to predicted ScenarioDict.
        ground_truth: Dict mapping query_id to GT ScenarioDict.
        track_collection: TrackCollection for bbox lookups.
        iou_threshold: BEV IoU threshold.

    Returns:
        HOTAResult (using full track duration as ground truth).
    """
    # Expand GT to full-track mode
    gt_full = {}
    for qid, gt_sd in ground_truth.items():
        expanded: ScenarioDict = {}
        for tid in gt_sd:
            expanded[tid] = {}  # empty → all timestamps
        gt_full[qid] = expanded

    # Expand predictions similarly
    pred_full = {}
    for qid, pred_sd in predictions.items():
        expanded: ScenarioDict = {}
        for tid in pred_sd:
            expanded[tid] = {}
        pred_full[qid] = expanded

    return evaluate_hota_temporal(
        pred_full, gt_full, track_collection, iou_threshold
    )
