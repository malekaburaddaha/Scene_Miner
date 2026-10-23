#!/usr/bin/env python3
"""MILESTONE 16 - per-scene CAM_FRONT pedestrian / VRU metrics on val150.

Adds, per scene, recall and translation error for vulnerable road users
(classes pedestrian, bicycle, motorcycle) inside the CAM_FRONT sector, at the
same fixed operating point as Milestone 11 (score > 0.3) and the same greedy
same-class centre-distance matching at 2 m, in descending confidence. The other
front-sector metrics (mAP, FP rate, mATE, mAOE, recall > / <= 30 m) are taken
unchanged from Milestone 11's per-scene table, and everything is written to one
per-scene file for the retrieval study.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
from val150_camfront_errors import quat_to_yaw_inv_rot

VRU = {"pedestrian", "bicycle", "motorcycle"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", default="work_dirs/bevformer_out/official_eval/pts_bbox/results_nusc.json")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--m11", default="results/cvpr/milestone11_camfront_errors.json")
    ap.add_argument("--out", default="work_dirs/val150_per_scene_front.json")
    a = ap.parse_args()
    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
    from nuscenes.eval.detection.data_classes import DetectionBox
    cfg = config_factory("detection_cvpr_2019")
    nusc = NuScenes(version="v1.0-trainval", dataroot=a.dataroot, verbose=False)
    m11 = json.load(open(a.m11)); half = np.radians(m11["camfront_half_fov_deg"])
    per = m11["per_scene"]
    pred, _ = load_prediction(a.results_json, cfg.max_boxes_per_sample, DetectionBox, verbose=False)
    gt = load_gt(nusc, "val", DetectionBox, verbose=False)
    pred = filter_eval_boxes(nusc, add_center_dist(nusc, pred), cfg.class_range, verbose=False)
    gt = filter_eval_boxes(nusc, add_center_dist(nusc, gt), cfg.class_range, verbose=False)
    by_scene = defaultdict(list)
    for st in gt.sample_tokens:
        by_scene[nusc.get("sample", st)["scene_token"]].append(st)

    def front(boxes, st, R):
        out = []
        for b in (boxes[st] if st in boxes.boxes else []):
            if b.detection_name not in VRU: continue
            v = R @ np.array(b.ego_translation[:2])
            if abs(np.arctan2(v[1], v[0])) <= half: out.append(b)
        return out

    for sc in per:
        n_gt = n_hit = 0; terr = []
        for st in by_scene.get(sc, []):
            sdr = nusc.get("sample_data", nusc.get("sample", st)["data"]["LIDAR_TOP"])
            R = quat_to_yaw_inv_rot(nusc.get("ego_pose", sdr["ego_pose_token"])["rotation"])
            G = front(gt, st, R)
            P = sorted([b for b in front(pred, st, R) if b.detection_score >= 0.3],
                       key=lambda b: -b.detection_score)
            taken = set()
            for p in P:
                best, bi = 2.0, None
                for i, g in enumerate(G):
                    if i in taken or g.detection_name != p.detection_name: continue
                    d = float(np.linalg.norm(np.array(g.translation[:2]) - np.array(p.translation[:2])))
                    if d < best: best, bi = d, i
                if bi is not None: taken.add(bi); terr.append(best)
            n_gt += len(G); n_hit += len(taken)
        per[sc]["vru_n_gt"] = n_gt
        per[sc]["vru_recall"] = n_hit / n_gt if n_gt else float("nan")
        per[sc]["vru_ATE"] = float(np.mean(terr)) if terr else float("nan")
    Path(a.out).write_text(json.dumps(per, indent=1))
    v = [p["vru_recall"] for p in per.values() if p["vru_recall"] == p["vru_recall"]]
    print(f"{len(per)} scenes; {len(v)} with front-sector VRU GT; mean VRU recall {np.mean(v):.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
