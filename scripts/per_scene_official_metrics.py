#!/usr/bin/env python3
"""Per-scene 3D detection difficulty using the OFFICIAL nuScenes algorithm.

A hand-rolled matcher gave an inverted signal (night scenes scoring 150x higher
than day). The official global eval on the same predictions returns
mAP 0.4164 / NDS 0.5173 -- exactly published BEVFormer-base -- so the predictions
are fine and the matcher was wrong. This script therefore reuses the official
nuscenes.eval.detection.algo.accumulate / calc_ap on a PER-SCENE basis, so the
per-scene numbers are produced by the same code that produces the headline mAP.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--eval-set", default="val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from nuscenes import NuScenes
    from nuscenes.eval.detection.data_classes import DetectionConfig
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.common.loaders import (load_prediction, load_gt,
                                              add_center_dist, filter_eval_boxes)
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.algo import accumulate, calc_ap

    cfg: DetectionConfig = config_factory("detection_cvpr_2019")
    nusc = NuScenes(version=a.version, dataroot=a.dataroot, verbose=False)
    pred_boxes, meta = load_prediction(a.results_json, cfg.max_boxes_per_sample,
                                       DetectionBox, verbose=False)
    gt_boxes = load_gt(nusc, a.eval_set, DetectionBox, verbose=False)
    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, verbose=False)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=False)
    print(f"[official] {len(pred_boxes.sample_tokens)} sample tokens", flush=True)

    # group sample tokens by scene
    scene_of = {}
    for st in gt_boxes.sample_tokens:
        scene_of[st] = nusc.get("sample", st)["scene_token"]
    by_scene = defaultdict(list)
    for st, sc in scene_of.items(): by_scene[sc].append(st)
    print(f"[official] {len(by_scene)} scenes", flush=True)

    from nuscenes.eval.common.data_classes import EvalBoxes
    def subset(boxes: EvalBoxes, toks):
        e = EvalBoxes()
        for t in toks:
            if t in boxes.boxes: e.add_boxes(t, boxes[t])
        return e

    out = {}
    for i, (sc, toks) in enumerate(sorted(by_scene.items())):
        gp, pp = subset(gt_boxes, toks), subset(pred_boxes, toks)
        aps, ngt = [], 0
        per_cls = {}
        for cls in cfg.class_names:
            ngt_c = sum(1 for t in toks for b in gp[t] if b.detection_name == cls)
            ngt += ngt_c
            if ngt_c == 0: continue
            cls_ap = []
            for d in cfg.dist_ths:
                md = accumulate(gp, pp, cls, cfg.dist_fcn_callable, d)
                cls_ap.append(calc_ap(md, cfg.min_recall, cfg.min_precision))
            per_cls[cls] = {"n_gt": ngt_c, "AP_mean": float(np.mean(cls_ap)),
                            "AP_by_dist": [float(x) for x in cls_ap]}
            aps.append(float(np.mean(cls_ap)))
        out[sc] = {"n_samples": len(toks), "n_gt": ngt,
                   "n_classes_present": len(per_cls),
                   "mAP": float(np.mean(aps)) if aps else float("nan"),
                   "per_class": per_cls}
        if (i+1) % 25 == 0: print(f"  [official] {i+1}/{len(by_scene)} scenes", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    m = np.array([v["mAP"] for v in out.values() if v["mAP"] == v["mAP"]])
    print(f"[official] per-scene mAP: mean {m.mean():.4f} sd {m.std():.4f} "
          f"range [{m.min():.3f},{m.max():.3f}]  (global eval was 0.4164)")
    print(f"[official] wrote {a.out}")

if __name__ == "__main__":
    main()
