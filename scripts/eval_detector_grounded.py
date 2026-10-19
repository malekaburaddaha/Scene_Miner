#!/usr/bin/env python3
"""TASK 1b - THE DECISIVE THESIS TEST.

Relevance-by-predicate makes a cached lookup a tautological oracle (Step 1:
mAP 1.000). This script re-grounds the comparison in PERCEPTION DIFFICULTY,
which is NOT a cached attribute and cannot be read from the lookup table.

Rankers: (a) flat lookup  (b) graded lookup  (c) metadata-matched  (d) SceneMiner
Measures:
  1. Mean difficulty of the top-k each ranker returns (equal bucket size).
  2. Decile curve: difficulty vs SceneMiner retrieval decile.
  3. WITHIN-BUCKET: restrict to scenes the lookup marks attribute=1 - where the
     lookup is provably blind (all tied) - and ask whether SceneMiner orders
     that bucket by difficulty. Spearman rho + top-k vs bucket mean.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import (MACRO8, DEGRAD, load_attributes,
                                  predicate_mask, conjunct_fraction)

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/d) if d > 0 else 0.0

def perm_p(a, b, rho, n=5000, seed=0):
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n):
        if abs(spearman(a, rng.permutation(b))) >= abs(rho): cnt += 1
    return (cnt + 1) / (n + 1)

def build_difficulty(diff, toks):
    """Composite difficulty in [0,1]; higher = harder. z-score then average."""
    out, comps = {}, {}
    ann, det = diff.get("annotation", {}), diff.get("detector", {})
    def z(v):
        v = np.asarray(v, float); s = v.std()
        return (v - v.mean())/s if s > 1e-9 else np.zeros_like(v)
    have_ann = [t for t in toks if t in ann]
    if have_ann:
        lowvis = z([ann[t]["frac_low_visibility"] for t in have_ann])
        meanvis = -z([ann[t]["mean_visibility"] for t in have_ann])   # low vis = hard
        dist   = z([ann[t]["mean_dist"] for t in have_ann])
        comps["annotation"] = dict(zip(have_ann, (lowvis + meanvis + dist)/3.0))
    have_det = [t for t in toks if t in det]
    if have_det:
        nconf = -z([det[t]["det_n_conf"] for t in have_det])          # fewer dets = hard
        msc   = -z([det[t]["det_mean_score"] for t in have_det])      # lower conf = hard
        comps["detector"] = dict(zip(have_det, (nconf + msc)/2.0))
    for t in toks:
        vals = [c[t] for c in comps.values() if t in c]
        if vals: out[t] = float(np.mean(vals))
    return out, comps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--difficulty", nargs="+",
                    default=["results/overnight_oct18/scene_difficulty_annot.json"])
    ap.add_argument("--sm-scores", default="results/overnight_oct18/sm_scores.json")
    ap.add_argument("--split", default="benchmark_cvpr/split_906_602.json")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--nuscenes-only", action="store_true", default=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", default="results/overnight_oct18/task1_detector_grounded.json")
    a = ap.parse_args()

    W = Path(a.work)
    z = np.load(W/"features_overnight.npz", allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]; dom = z["domain"].astype(int)
    attrs = load_attributes(W)

    diff = {"annotation": {}, "detector": {}}
    for f in a.difficulty:
        if Path(f).exists():
            d = json.load(open(f))
            for k in ("annotation", "detector"): diff[k].update(d.get(k, {}))
    print(f"[task1] difficulty sources: annotation={len(diff['annotation'])} "
          f"detector={len(diff['detector'])}")

    sm = json.load(open(a.sm_scores)) if Path(a.sm_scores).exists() else {}
    print(f"[task1] sceneminer scores for {len(sm)} queries")

    sel = np.ones(len(toks), bool)
    if a.nuscenes_only: sel &= (dom == 0)
    if a.eval_only:
        ev = set(json.load(open(a.split))["eval"]); sel &= np.array([t in ev for t in toks])
    idx = np.where(sel)[0]; sub = [toks[i] for i in idx]
    D, comps = build_difficulty(diff, sub)
    sub = [t for t in sub if t in D]
    dv = np.array([D[t] for t in sub])
    print(f"[task1] scope: {len(sub)} scenes with a difficulty value "
          f"(nuScenes_only={a.nuscenes_only}, eval_only={a.eval_only})")
    print(f"[task1] difficulty: mean {dv.mean():+.3f} sd {dv.std():.3f} "
          f"range [{dv.min():+.2f},{dv.max():+.2f}]")

    rng = np.random.default_rng(0)
    rows, within = [], []
    for qname, qset in (("degrad", DEGRAD), ("macro8", MACRO8)):
        for q, pred in qset:
            flat = predicate_mask(attrs, sub, pred).astype(float)
            grad = conjunct_fraction(attrs, sub, pred)
            meta_keys = {k: v for k, v in pred.items() if k in ("weather", "lighting")}
            meta = (predicate_mask(attrs, sub, meta_keys).astype(float)
                    if meta_keys else np.zeros(len(sub)))
            smv = None
            if q in sm:
                full = np.asarray(sm[q], float); smv = full[idx][[True]*len(idx)]
                smv = np.array([full[i] for i, t in zip(idx, [toks[j] for j in idx]) if t in D])
            nb = int(flat.sum())
            if nb < 10 or smv is None: continue

            k = min(a.topk, nb)
            def topk_mean(score, k):
                o = np.argsort(-score)[:k]; return float(dv[o].mean())
            r = {"query": q, "set": qname, "bucket": nb, "k": k,
                 "difficulty_topk": {
                     "flat_lookup": float(np.mean([dv[i] for i in
                                      rng.choice(np.where(flat > 0)[0], k, replace=False)])),
                     "graded_lookup": topk_mean(grad, k),
                     "metadata": topk_mean(meta, k) if meta.sum() > 0 else float("nan"),
                     "sceneminer": topk_mean(smv, k),
                     "random": float(np.mean(dv[rng.choice(len(sub), k, replace=False)])),
                     "corpus_mean": float(dv.mean())}}
            rows.append(r)

            # ---------- WITHIN-BUCKET: where the lookup is blind ----------
            b = np.where(flat > 0)[0]
            if len(b) >= 20:
                bd, bs = dv[b], smv[b]
                rho = spearman(bs, bd)
                kk = max(5, len(b)//5)
                top = bd[np.argsort(-bs)[:kk]]
                within.append({"query": q, "set": qname, "bucket": int(len(b)),
                               "spearman_sm_vs_difficulty": rho,
                               "perm_p": perm_p(bs, bd, rho, 2000),
                               "bucket_mean_difficulty": float(bd.mean()),
                               "sm_top20pct_difficulty": float(top.mean()),
                               "lift_over_bucket_mean": float(top.mean()-bd.mean())})

    # ---------- decile curve over the union of degradation buckets ----------
    dec = {}
    for q, pred in DEGRAD:
        if q not in sm: continue
        full = np.asarray(sm[q], float)
        smv = np.array([full[i] for i, t in zip(idx, [toks[j] for j in idx]) if t in D])
        o = np.argsort(-smv); parts = np.array_split(o, 10)
        dec[q] = [float(dv[p].mean()) for p in parts]
    if dec:
        M = np.array(list(dec.values()))
        dec["MEAN"] = M.mean(0).tolist()

    out = {"scope": {"n_scenes": len(sub), "nuscenes_only": a.nuscenes_only,
                     "eval_only": a.eval_only,
                     "difficulty_components": list(comps)},
           "per_query_topk": rows, "within_bucket": within,
           "decile_curve": dec}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    # ---------- report ----------
    if rows:
        print(f"\n=== TOP-{a.topk} MEAN DIFFICULTY (higher = harder scenes retrieved) ===")
        keys = ["random", "flat_lookup", "metadata", "graded_lookup", "sceneminer"]
        print(f"{'ranker':<16}{'degrad':>10}{'macro8':>10}{'all':>10}")
        for kk in keys:
            vs = {s: [r["difficulty_topk"][kk] for r in rows if r["set"] == s
                      and not np.isnan(r["difficulty_topk"][kk])] for s in ("degrad","macro8")}
            allv = vs["degrad"] + vs["macro8"]
            f = lambda v: f"{np.mean(v):+.3f}" if v else "   n/a"
            print(f"{kk:<16}{f(vs['degrad']):>10}{f(vs['macro8']):>10}{f(allv):>10}")
    if within:
        rhos = [w["spearman_sm_vs_difficulty"] for w in within]
        lifts = [w["lift_over_bucket_mean"] for w in within]
        sig = sum(1 for w in within if w["perm_p"] < 0.05)
        print(f"\n=== WITHIN-BUCKET (lookup is blind here; n={len(within)} queries) ===")
        print(f"  mean Spearman(SceneMiner score, difficulty) = {np.mean(rhos):+.3f}")
        print(f"  significant at p<0.05: {sig}/{len(within)}")
        print(f"  mean difficulty lift of SM top-20% over bucket mean = {np.mean(lifts):+.3f}")
    if "MEAN" in dec:
        print(f"\n=== DECILE CURVE (mean difficulty, decile 1 = top-ranked) ===")
        print("  " + "  ".join(f"D{i+1}:{v:+.2f}" for i, v in enumerate(dec["MEAN"])))
    print(f"\n[task1] wrote {a.out}")

if __name__ == "__main__":
    main()
