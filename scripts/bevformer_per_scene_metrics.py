#!/usr/bin/env python3
"""Per-scene 3D detection quality from BEVFormer predictions.

Full nuScenes NDS needs the five TP sub-metrics and a global eval pass; what the
gate needs is a per-scene difficulty signal, so we compute the per-scene
analogue of nuScenes mAP directly: greedy center-distance matching per class at
the official thresholds {0.5, 1, 2, 4} m, then AP per class per threshold,
averaged. Also reports per-scene recall at each threshold.

Run inside the bevformer env (mmdet3d 0.17.1).
"""
from __future__ import annotations
import argparse, json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

THRESH = [0.5, 1.0, 2.0, 4.0]

def ap_from_matches(scores, tps, n_gt):
    """VOC-style AP from score-sorted TP indicators."""
    if n_gt == 0 or len(scores) == 0: return float("nan")
    o = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, dtype=float)[o]
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(1e-9, ctp + cfp)
    # 101-point interpolation
    ap, prev = 0.0, 0.0
    for r in np.linspace(0, 1, 101):
        p = prec[rec >= r].max() if (rec >= r).any() else 0.0
        ap += p
    return float(ap / 101)

def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--results", required=True)
    ap_.add_argument("--infos", required=True)
    ap_.add_argument("--out", required=True)
    a = ap_.parse_args()

    CLASSES = ['car','truck','construction_vehicle','bus','trailer','barrier',
               'motorcycle','bicycle','pedestrian','traffic_cone']
    CIDX = {c:i for i,c in enumerate(CLASSES)}

    res = pickle.load(open(a.results, "rb"))
    if isinstance(res, dict): res = res.get("bbox_results", res.get("results", res))
    d = pickle.load(open(a.infos, "rb"))
    infos = d["infos"] if isinstance(d, dict) else d
    print(f"[metrics] {len(res)} predictions, {len(infos)} infos", flush=True)
    n = min(len(res), len(infos))

    # CLASS-AWARE matching. The previous class-agnostic version let ~300
    # predictions/sample cover any small GT set, making recall a function of
    # object count (r(n_gt,night) = -0.36) rather than of detection quality,
    # and inverting the night signal. Matching within class fixes that.
    per_scene = defaultdict(lambda: {"n_gt": 0, "n_pred": 0, "samples": 0,
                                     "cls_gt": defaultdict(int),
                                     "cls_score": defaultdict(list),
                                     "cls_tp": defaultdict(lambda: defaultdict(list))})
    for i in range(n):
        info = infos[i]; r = res[i]
        sc = info["scene_token"]
        pb = r.get("pts_bbox", r) if isinstance(r, dict) else r
        boxes = pb.get("boxes_3d"); scores = pb.get("scores_3d"); labels = pb.get("labels_3d")
        if boxes is None: continue
        P = np.asarray(boxes.tensor.cpu().numpy() if hasattr(boxes, "tensor") else boxes)
        S = np.asarray(scores.cpu().numpy() if hasattr(scores, "cpu") else scores)
        L = np.asarray(labels.cpu().numpy() if hasattr(labels, "cpu") else labels)
        G = np.asarray(info.get("gt_boxes", np.zeros((0, 7))))
        GN = np.asarray([str(x) for x in info.get("gt_names", [])])
        keep_g = np.array([g in CIDX for g in GN], dtype=bool)
        G, GN = (G[keep_g], GN[keep_g]) if len(GN) else (G, GN)
        e = per_scene[sc]; e["samples"] += 1; e["n_pred"] += len(S); e["n_gt"] += len(G)
        for c in set(GN.tolist()): e["cls_gt"][c] += int((GN == c).sum())
        order = np.argsort(-S)
        used = {c: {t: set() for t in THRESH} for c in CLASSES}
        for oi in order:
            ci = int(L[oi])
            if ci >= len(CLASSES): continue
            cname = CLASSES[ci]
            e["cls_score"][cname].append(float(S[oi]))
            gsel = np.where(GN == cname)[0] if len(GN) else np.array([], dtype=int)
            dists = (np.linalg.norm(G[gsel][:, :2] - P[oi, :2][None, :], axis=1)
                     if len(gsel) else np.array([]))
            for t in THRESH:
                hit = 0.0
                if len(dists):
                    cand = [gsel[j] for j in np.argsort(dists)
                            if gsel[j] not in used[cname][t] and dists[j] <= t]
                    if cand: used[cname][t].add(cand[0]); hit = 1.0
                e["cls_tp"][cname][t].append(hit)
    out = {}
    for sc, e in per_scene.items():
        rec = {"samples": e["samples"], "n_gt": e["n_gt"], "n_pred": e["n_pred"]}
        aps = {}
        for t in THRESH:
            per_cls, tp_tot = [], 0.0
            for c, ngt in e["cls_gt"].items():
                if ngt == 0: continue
                v = ap_from_matches(e["cls_score"].get(c, []), e["cls_tp"][c][t], ngt)
                if v == v: per_cls.append(v)
                tp_tot += float(np.sum(e["cls_tp"][c][t]))
            aps[f"AP@{t}"] = float(np.mean(per_cls)) if per_cls else float("nan")
            rec[f"recall@{t}"] = float(tp_tot / max(1, e["n_gt"]))
        rec.update(aps)
        vals = [v for k, v in aps.items() if v == v]
        rec["mAP_dist"] = float(np.mean(vals)) if vals else float("nan")
        out[sc] = rec
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    m = [v["mAP_dist"] for v in out.values() if v["mAP_dist"] == v["mAP_dist"]]
    print(f"[metrics] {len(out)} scenes; mAP_dist mean {np.mean(m):.4f} "
          f"sd {np.std(m):.4f} range [{np.min(m):.3f},{np.max(m):.3f}]")
    print(f"[metrics] wrote {a.out}")

if __name__ == "__main__":
    main()
