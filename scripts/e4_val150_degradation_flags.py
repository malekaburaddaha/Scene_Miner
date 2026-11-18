#!/usr/bin/env python3
"""MILESTONE 41 - E4: the val150 degraded/clean split and its front-sector table.

Re-derives, from the cached files only:
  * the flag rule: a val150 scene is "degraded" when ANY of the four zero-shot
    Qwen2.5-VL-3B yes/no log-odds on its CAM_FRONT keyframe (wet road, glare,
    droplets, low contrast; questions in scripts/val150_camfront_errors.py) is > 0
    (work_dirs/val150_camfront_signals.npz);
  * the count per attribute and the any-degradation split;
  * every front-sector metric mean from the stored per-scene table
    (results/cvpr/milestone11_camfront_errors.json), the two-sided permutation p of
    each degraded-vs-clean difference (10,000 relabellings) for all 5 partitions x 6
    metrics = 30 tests, and Benjamini-Hochberg q over those 30;
  * the published p-values (Milestone 11, 1000 relabellings) with their BH q, for comparison.
-> results/cvpr/milestone41_e4_val150_flags.json
"""
import json
from pathlib import Path
import numpy as np

ATTRS = ["wet_road", "glare", "droplets", "low_contrast"]
METS = ["mAP_front", "fp_rate", "mATE_front", "recall_far", "mAOE_front", "recall_near"]


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def main():
    z = np.load("work_dirs/val150_camfront_signals.npz", allow_pickle=False)
    scenes = [str(s) for s in z["scenes"]]; M = z["margin"]
    m11 = json.load(open("results/cvpr/milestone11_camfront_errors.json")); per = m11["per_scene"]
    assert set(scenes) == set(per), "margin cache and per-scene table cover different scenes"
    labels = {at: M[:, j] > 0 for j, at in enumerate(ATTRS)}
    labels["any_degradation"] = (M > 0).any(1)
    rng = np.random.default_rng(0); NP = 10000
    rows, tests = {}, []
    for name, pos in labels.items():
        A = [s for s, f in zip(scenes, pos) if f]; B = [s for s, f in zip(scenes, pos) if not f]
        rows[name] = {"n_degraded": len(A), "n_clean": len(B), "metrics": {}}
        for mt in METS:
            va = np.array([per[s][mt] for s in A], float); vb = np.array([per[s][mt] for s in B], float)
            va, vb = va[~np.isnan(va)], vb[~np.isnan(vb)]
            if len(va) < 3 or len(vb) < 3: continue
            obs = va.mean() - vb.mean(); pooled = np.concatenate([va, vb]); cnt = 0
            for _ in range(NP):
                x = rng.permutation(pooled); cnt += abs(x[:len(va)].mean() - x[len(va):].mean()) >= abs(obs)
            pub = m11["partitions"][name]["metrics"].get(mt, {})
            rows[name]["metrics"][mt] = {"degraded": float(va.mean()), "clean": float(vb.mean()), "delta": float(obs),
                                         "n_degraded_valid": len(va), "n_clean_valid": len(vb),
                                         "perm_p": (cnt + 1) / (NP + 1), "published_perm_p": pub.get("perm_p"),
                                         "published_delta": pub.get("delta")}
            tests.append((name, mt))
    q = bh([rows[n]["metrics"][m]["perm_p"] for n, m in tests])
    qp = bh([rows[n]["metrics"][m]["published_perm_p"] for n, m in tests])
    for (n, m), a_, b_ in zip(tests, q, qp):
        rows[n]["metrics"][m]["q_BH"] = float(a_); rows[n]["metrics"][m]["published_q_BH"] = float(b_)
    out = {"doc": __doc__, "n_scenes": len(scenes), "n_tests": len(tests),
           "rule": "degraded = any of the 4 zero-shot Qwen2.5-VL-3B CAM_FRONT yes/no log-odds > 0",
           "per_attribute_positive": {at: int(labels[at].sum()) for at in ATTRS}, "partitions": rows}
    Path("results/cvpr/milestone41_e4_val150_flags.json").write_text(json.dumps(out, indent=1))
    print(f"[E4] {len(scenes)} val150 scenes; per attribute >0: {out['per_attribute_positive']}; tests {len(tests)}")
    r = rows["any_degradation"]
    print(f"[E4] any_degradation: {r['n_degraded']} degraded vs {r['n_clean']} clean")
    for mt, v in r["metrics"].items():
        print(f"   {mt:<12} degraded {v['degraded']:.4f} clean {v['clean']:.4f} delta {v['delta']:+.4f} "
              f"p {v['perm_p']:.4f} q {v['q_BH']:.3f} | published p {v['published_perm_p']:.4f} q {v['published_q_BH']:.3f}")
    print(f"[E4] tests with q<0.05 (new / published): {int((q < 0.05).sum())} / {int((qp < 0.05).sum())}; "
          f"q<0.10: {int((q < 0.10).sum())} / {int((qp < 0.10).sum())}")


if __name__ == "__main__":
    main()
