#!/usr/bin/env python3
"""TASK 1a - per-scene perception difficulty from INDEPENDENT sources.

mmdet3d/mmdet are not installed in this env (AssertionError on import), so a
BEVFormer forward pass is not runnable tonight. Two self-contained substitutes,
BOTH independent of our VLM attribute pipeline:

  (A) ANNOTATION-DERIVED  - nuScenes human-annotated visibility levels
      (v0-40 / v40-60 / v60-80 / v80-100) plus object geometry. If a human
      annotator marked objects <40% visible, perception IS hard there. This is a
      human label that shares no channel with our supervision -> also attacks
      the circularity critique.
  (B) MODEL-DERIVED - an off-the-shelf COCO Faster R-CNN (torchvision) run on
      the cached thumbnails; we record mean/max score and the count of confident
      detections. Annotation-free: a detector that sees less, scores lower.

Outputs results/overnight_oct18/scene_difficulty.json
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

def annotation_difficulty(dataroot, version, tokens):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    by_tok = {s["token"]: s for s in nusc.scene}
    VIS = {"1": 0.20, "2": 0.50, "3": 0.70, "4": 0.90}   # midpoint of each band
    out = {}
    for tk in tokens:
        s = by_tok.get(tk)
        if s is None: continue
        vis, dists, nobj, nvru, nlow = [], [], 0, 0, 0
        st = s["first_sample_token"]
        while st:
            samp = nusc.get("sample", st)
            sd = nusc.get("sample_data", samp["data"]["LIDAR_TOP"])
            ego = np.array(nusc.get("ego_pose", sd["ego_pose_token"])["translation"][:2])
            for at in samp["anns"]:
                ann = nusc.get("sample_annotation", at)
                p = np.array(ann["translation"][:2])
                d = float(np.linalg.norm(p - ego))
                if d > 50.0: continue
                nobj += 1; dists.append(d)
                v = ann.get("visibility_token")
                if v in VIS:
                    vis.append(VIS[v])
                    if v in ("1", "2"): nlow += 1
                if ann["category_name"].startswith(("human.", "vehicle.bicycle",
                                                    "vehicle.motorcycle")):
                    nvru += 1
            st = samp["next"]
        n_s = max(1, len(vis))
        out[tk] = {
            "n_obj_50m": nobj,
            "mean_visibility": float(np.mean(vis)) if vis else 1.0,
            "frac_low_visibility": nlow / n_s,
            "mean_dist": float(np.mean(dists)) if dists else 0.0,
            "n_vru": nvru,
        }
    return out

def detector_difficulty(thumbs, tokens, batch=8, score_thr=0.5):
    import torch, torchvision
    from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2,
                                              FasterRCNN_ResNet50_FPN_V2_Weights)
    from PIL import Image
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=w).to(dev).eval()
    tf = w.transforms()
    out, t0 = {}, time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(tokens), batch):
            chunk = tokens[i:i + batch]
            imgs, keep = [], []
            for t in chunk:
                p = Path(thumbs) / f"{t}.jpg"
                if p.exists():
                    imgs.append(tf(Image.open(p).convert("RGB")).to(dev)); keep.append(t)
            if not imgs: continue
            preds = model(imgs)
            for t, pr in zip(keep, preds):
                sc = pr["scores"].detach().cpu().numpy()
                conf = sc[sc >= score_thr]
                out[t] = {"det_n_conf": int(len(conf)),
                          "det_mean_score": float(conf.mean()) if len(conf) else 0.0,
                          "det_max_score": float(sc.max()) if len(sc) else 0.0,
                          "det_sum_score": float(conf.sum())}
            if (i // batch) % 20 == 0:
                print(f"  [det] {i+len(chunk)}/{len(tokens)}  {time.perf_counter()-t0:.0f}s", flush=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work_dirs/unified_filter/features_overnight.npz")
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", default="results/overnight_oct18/scene_difficulty.json")
    ap.add_argument("--skip-detector", action="store_true")
    ap.add_argument("--skip-annotations", action="store_true")
    a = ap.parse_args()

    z = np.load(a.features, allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]; dom = z["domain"].astype(int)
    nusc_toks = [t for t, d in zip(toks, dom) if d == 0]
    res = {"annotation": {}, "detector": {}}
    if not a.skip_annotations:
        print(f"[diff] annotation-derived over {len(nusc_toks)} nuScenes scenes...", flush=True)
        res["annotation"] = annotation_difficulty(a.dataroot, a.version, nusc_toks)
        print(f"  done: {len(res['annotation'])} scenes", flush=True)
    if not a.skip_detector:
        print(f"[diff] detector-derived over all {len(toks)} scenes...", flush=True)
        res["detector"] = detector_difficulty(a.thumbs, toks)
        print(f"  done: {len(res['detector'])} scenes", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"[diff] wrote {a.out}")
