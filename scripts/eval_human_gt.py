#!/usr/bin/env python3
"""TASK 5 - evaluate every ranker against HUMAN ground truth.

This is the evaluation where the cached-VLM lookup stops being an oracle. When
relevance is defined by a predicate over the VLM's own flags, the lookup scores
1.000 by construction (see results/overnight_oct18/step1_attribute_index.json).
Grade against human labels instead and the lookup inherits the VLM's error,
while SceneMiner can in principle recover positives the VLM missed by fusing
BEV embedding e, tags t, risk r and the visual token v.

Usage (any time once you have >= ~200 labelled scenes):
  PYTHONPATH=src python scripts/eval_human_gt.py \
      --raters results/overnight_oct18/annotations/*.json
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import (ap_binary, p_at, ndcg_at, load_attributes)

ATTRS = ["droplets", "glare", "wet_road", "low_contrast"]

def kappa(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); n = len(a)
    if n == 0: return float("nan")
    po = float((a == b).mean())
    p1, p2 = a.mean(), b.mean()
    pe = p1*p2 + (1-p1)*(1-p2)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")

def load_raters(paths):
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        name = Path(p).stem
        out[name] = {t: r["labels"] for t, r in d.items()
                     if r.get("flag") not in ("skip", "unsure")}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+",
                    default=glob.glob("results/overnight_oct18/annotations/*.json"))
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--sm-scores", default="results/overnight_oct18/sm_scores.json")
    ap.add_argument("--min-labels", type=int, default=100)
    ap.add_argument("--out", default="results/day1_gate/human_gt_eval.json")
    a = ap.parse_args()

    paths = [p for p in a.raters if Path(p).exists()
             and Path(p).name not in ("sample_600.json",)]
    raters = load_raters(paths)
    if not raters:
        print("[human-gt] no annotation files yet -- run scripts/annotate_ui.py first")
        return
    for nm, d in raters.items(): print(f"[human-gt] rater '{nm}': {len(d)} labelled scenes")
    common = set.intersection(*[set(d) for d in raters.values()]) if len(raters) > 1 else set(next(iter(raters.values())))
    if len(common) < a.min_labels:
        print(f"[human-gt] only {len(common)} scenes labelled; need >= {a.min_labels}. "
              f"Re-run when more are done.")
        if len(common) == 0: return

    W = Path(a.work)
    z = np.load(W/"features_overnight.npz", allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]
    vlm = json.load(open(W/"binary_flags_with_logits.json"))
    res = {"n_raters": len(raters), "n_common": len(common)}

    # ---------- 1. agreement ----------
    agree = {}
    names = sorted(raters)
    toks = sorted(common)
    for at in ATTRS:
        rec = {}
        if len(names) > 1:
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    x = [raters[names[i]][t].get(at, False) for t in toks]
                    y = [raters[names[j]][t].get(at, False) for t in toks]
                    rec[f"kappa_{names[i]}_vs_{names[j]}"] = kappa(x, y)
        # consensus = all raters agree True (strict); else majority
        cons = []
        for t in toks:
            v = [raters[n][t].get(at, False) for n in names]
            cons.append(sum(v) > len(v)/2)
        mv = [bool(vlm.get(t, {}).get(at, False)) for t in toks]
        rec["kappa_human_vs_vlm"] = kappa(cons, mv)
        tp = sum(1 for c, m in zip(cons, mv) if c and m)
        fp = sum(1 for c, m in zip(cons, mv) if (not c) and m)
        fn = sum(1 for c, m in zip(cons, mv) if c and (not m))
        rec.update({"human_positives": int(sum(cons)), "vlm_positives": int(sum(mv)),
                    "vlm_false_positives": fp, "vlm_false_negatives": fn,
                    "vlm_precision": tp/max(1, tp+fp), "vlm_recall": tp/max(1, tp+fn)})
        agree[at] = rec
        print(f"  {at:<14} human+={sum(cons):3d} vlm+={sum(mv):3d}  "
              f"kappa(h,vlm)={rec['kappa_human_vs_vlm']:+.3f}  "
              f"VLM FP={fp} FN={fn}  P={rec['vlm_precision']:.2f} R={rec['vlm_recall']:.2f}")
    res["agreement"] = agree

    # ---------- 2. retrieval graded against HUMAN labels ----------
    sm = json.load(open(a.sm_scores)) if Path(a.sm_scores).exists() else {}
    idx = {t: i for i, t in enumerate(corpus)}
    sub = [t for t in toks if t in idx]
    QMAP = {"droplets": "water droplets smearing the camera lens",
            "glare": "blinding headlight bloom from oncoming traffic",
            "wet_road": "wet reflective asphalt after rainfall",
            "low_contrast": None}
    rows = {}
    for at in ATTRS:
        human = np.array([sum(raters[n][t].get(at, False) for n in names) > len(names)/2
                          for t in sub])
        if human.sum() == 0 or human.sum() == len(sub):
            rows[at] = {"skipped": "degenerate human labels"}; continue
        rankers = {}
        # flat VLM lookup: binary flag, random tie-break
        flat = np.array([float(vlm.get(t, {}).get(at, False)) for t in sub])
        rng = np.random.default_rng(0); acc = []
        for _ in range(500):
            o = np.lexsort((rng.random(len(sub)), -flat)); acc.append(ap_binary(o, human))
        rankers["lookup_flat"] = {"mAP": float(np.mean(acc))}
        # logit VLM lookup: continuous margin -> a real ranking
        marg = np.array([float(vlm.get(t, {}).get(at+"_margin", -9.0)) for t in sub])
        o = np.argsort(-marg)
        rankers["lookup_logit"] = {"mAP": ap_binary(o, human), "P@5": p_at(o, human),
                                   "NDCG@10": ndcg_at(o, human.astype(float))}
        # SceneMiner
        q = QMAP.get(at)
        if q and q in sm:
            full = np.asarray(sm[q], float)
            s = np.array([full[idx[t]] for t in sub])
            o = np.argsort(-s)
            rankers["sceneminer"] = {"mAP": ap_binary(o, human), "P@5": p_at(o, human),
                                     "NDCG@10": ndcg_at(o, human.astype(float))}
            # does SceneMiner recover human-positives the VLM missed?
            missed = human & ~np.array([bool(vlm.get(t, {}).get(at, False)) for t in sub])
            if missed.sum() > 0:
                rank_of = np.empty(len(sub)); rank_of[o] = np.arange(len(sub))
                rankers["sceneminer"]["vlm_missed_positives"] = int(missed.sum())
                rankers["sceneminer"]["median_rank_of_vlm_missed"] = float(np.median(rank_of[missed]))
                rankers["sceneminer"]["median_rank_pct"] = float(np.median(rank_of[missed])/len(sub))
        rows[at] = {"n_human_positive": int(human.sum()), "n_scenes": len(sub),
                    "chance": float(human.mean()), "rankers": rankers}
        print(f"\n  == {at} (human+ {int(human.sum())}/{len(sub)}, chance {human.mean():.3f}) ==")
        for k, v in rankers.items():
            extra = ""
            if "median_rank_pct" in v:
                extra = (f"  | VLM missed {v['vlm_missed_positives']}, SM median rank "
                         f"{v['median_rank_pct']*100:.0f}%")
            print(f"    {k:<14} mAP={v['mAP']:.4f}{extra}")
    res["retrieval_vs_human"] = rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\n[human-gt] wrote {a.out}")

if __name__ == "__main__":
    main()
