#!/usr/bin/env python3
"""MILESTONE 30 - fold a second human rater into the evaluation.

Reads results/rater2/<rater>.json (written by scripts/run_rater2_ui.sh) and, on the
160-scene subset, reports per attribute:
  * Cohen's kappa between rater 1 and rater 2, overall and per stratum
    (rater-1 positives / VLM-boundary cases), with raw agreement and prevalence;
  * mAP of SceneMiner, zero-shot SigLIP2, the VLM lookup and late fusion
    z(SigLIP2)+z(lookup) against rater 2, next to the same rankers against rater 1
    on the same scenes, so the two graders are compared on identical ground.
-> results/cvpr/rater2_<rater>.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_human_gt_full import load_human, ATTRS, QUERY


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rater", required=True)
    ap.add_argument("--labels", default=None, help="override path to the rater-2 JSON")
    a = ap.parse_args()
    sub = json.load(open("benchmark_cvpr/rater2_subset_160.json"))
    r1 = load_human("results/day1_gate/malek_repaired.json", "B")
    r2 = load_human(a.labels or f"results/rater2/{a.rater}.json", "B")
    toks = [t for t in sub["tokens"] if t in r1 and t in r2]
    print(f"[rater2] {len(toks)}/{len(sub['tokens'])} scenes labelled by both raters")
    out = {"rater2": a.rater, "n_scenes": len(toks), "attributes": {}}
    from eval_related_work_baselines import Enc, Rankers, load_corpus, zs, ap_of
    enc = Enc(); C = load_corpus()
    z = np.load("work_dirs/unified_filter/features_v2.npz"); R = Rankers(enc, {}, z["risk"].reshape(-1))
    cid = {t: i for i, t in enumerate(C["toks"])}; ix = np.array([cid[t] for t in toks])
    for at in ATTRS:
        y1 = np.array([r1[t][at] for t in toks]); y2 = np.array([r2[t][at] for t in toks])
        st = np.array([sub["stratum"][t].split(":")[1] for t in toks])
        rec = {"kappa": kappa(y1, y2), "agreement": float((y1 == y2).mean()),
               "prevalence_r1": float(y1.mean()), "prevalence_r2": float(y2.mean()), "by_stratum": {}}
        for s_ in sorted(set(st)):
            m = st == s_
            rec["by_stratum"][s_] = {"n": int(m.sum()), "kappa": kappa(y1[m], y2[m]), "agreement": float((y1[m] == y2[m]).mean())}
        q = QUERY[at]
        S = {"SceneMiner": R.sceneminer(C, q), "SigLIP2": R.siglip2(C, q), "lookup": R.lookup(C, q, at)}
        S["late fusion SigLIP2+lookup"] = zs(S["SigLIP2"]) + zs(S["lookup"])
        rec["mAP"] = {k: {"vs_rater1": ap_of(np.asarray(v)[..., ix], y1, per_seed=np.ndim(v) == 2),
                          "vs_rater2": ap_of(np.asarray(v)[..., ix], y2, per_seed=np.ndim(v) == 2)} for k, v in S.items()}
        out["attributes"][at] = rec
        print(f"  {at:<13} kappa {rec['kappa']:+.3f}  agree {rec['agreement']:.3f}  "
              + "  ".join(f"{k}: {v['vs_rater1']:.3f}/{v['vs_rater2']:.3f}" for k, v in rec["mAP"].items()))
    out["macro"] = {"kappa": float(np.mean([out["attributes"][a_]["kappa"] for a_ in ATTRS])),
                    "mAP": {k: {r: float(np.mean([out["attributes"][a_]["mAP"][k][r] for a_ in ATTRS]))
                                for r in ("vs_rater1", "vs_rater2")} for k in out["attributes"][ATTRS[0]]["mAP"]}}
    print("  macro kappa", round(out["macro"]["kappa"], 3), out["macro"]["mAP"])
    json.dump(out, open(f"results/cvpr/rater2_{a.rater}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
