#!/usr/bin/env python3
"""MILESTONE 33 - the second human rater (160 scenes) in the evaluation.

Inputs (anonymised; the `rater` field is "rater1" / "rater2"):
  benchmark_cvpr/human_rater1_600.json   rater 1, 600 scenes (550 after unsure/skip)
  benchmark_cvpr/human_rater2_160.json   rater 2, the 160-scene subset
  benchmark_cvpr/rater2_subset_160.json  how the subset was drawn: per attribute,
      20 rater-1 positives and 20 VLM-boundary scenes (margin closest to 0). The
      subset over-samples positives and hard cases on purpose, so agreement on it
      is a conservative (lower) estimate of agreement on the full sample.

Per attribute and macro-averaged:
  1. human-human agreement: Cohen's kappa (scene-bootstrap 95% CI), raw agreement,
     positive / negative specific agreement, prevalence; by stratum (rater-1
     positive vs VLM boundary, over all 160 scenes and over the 40 drawn for the
     attribute);
  2. VLM-human kappa: the Stage-1 VLM's binary flag against each rater on the same
     scenes;
  3. mAP of every Table 2 method and the late-fusion control graded against
     rater 2, against rater 1 on the same 160 scenes, against the two-rater
     consensus (only scenes where they agree), and against the union / intersection
     of the two raters' positives. Human-GT protocol: per-seed AP averaged over
     seeds. mAP on the subset is not comparable with the 550-scene numbers (the
     subset is enriched for positives); rankers are compared on identical ground.
  4. paired scene bootstrap of SceneMiner minus each key ranker, macro mAP, rater 2.
-> results/cvpr/milestone33_rater2_results.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_human_gt_full import load_human, ATTRS, QUERY
from eval_related_work_baselines import (Enc, Rankers, load_corpus, load_captions, zs, ap_of, METHODS, W)

R1 = "benchmark_cvpr/human_rater1_600.json"
R2 = "benchmark_cvpr/human_rater2_160.json"
NB = 2000


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def agree_stats(a, b, rng=None):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n11, n00 = int((a & b).sum()), int((~a & ~b).sum()); n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    degenerate = a.all() or not a.any() or b.all() or not b.any()      # kappa undefined if a rater is constant
    r = {"n": len(a), "kappa": None if degenerate else kappa(a, b), "agreement": float((a == b).mean()),
         "positive_agreement": float(2 * n11 / (2 * n11 + n10 + n01)) if (2 * n11 + n10 + n01) else float("nan"),
         "negative_agreement": float(2 * n00 / (2 * n00 + n10 + n01)) if (2 * n00 + n10 + n01) else float("nan"),
         "prevalence_a": float(a.mean()), "prevalence_b": float(b.mean()), "table": [[n11, n10], [n01, n00]]}
    if rng is not None:
        ks = [kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(NB))]
        r["kappa_ci95"] = [float(np.nanpercentile(ks, 2.5)), float(np.nanpercentile(ks, 97.5))]
    return r


def main():
    t0 = time.time(); rng = np.random.default_rng(0)
    sub = json.load(open("benchmark_cvpr/rater2_subset_160.json"))
    r1, r2 = load_human(R1, "B"), load_human(R2, "B")
    toks = [t for t in sub["tokens"] if t in r1 and t in r2]
    print(f"[rater2] {len(toks)}/{len(sub['tokens'])} scenes labelled by both raters", flush=True)
    kind = np.array([sub["stratum"][t].split(":")[1] for t in toks])
    drawn = np.array([sub["stratum"][t].split(":")[0] for t in toks])

    enc = Enc(); C = load_corpus()
    prog = Path("results/cvpr/refav_llm_programs.json")
    programs = json.load(open(prog))["programs"] if prog.exists() else {}
    z = np.load(W/"features_v2.npz", allow_pickle=False)
    R = Rankers(enc, load_captions("neutral"), z["risk"].reshape(-1), load_captions("enumerating"), programs)
    cid = {t: i for i, t in enumerate(C["toks"])}; ix = np.array([cid[t] for t in toks])

    def call(meth, q, at):
        f = getattr(R, meth)
        return f(C, q, at, 1) if meth == "lookup" else f(C, q)

    out = {"doc": __doc__, "n_scenes": len(toks), "strata": {k: int((kind == k).sum()) for k in sorted(set(kind))},
           "attributes": {}, "methods": [m[0] for m in METHODS] + ["Late fusion SigLIP2 + lookup"]}
    scores = {}                                                   # method -> attr -> scores on the 160 (seeds x n or n)
    for at in ATTRS:
        y1 = np.array([r1[t][at] for t in toks]); y2 = np.array([r2[t][at] for t in toks])
        vlm = C["flags"][at][ix]
        rec = {"human_human": agree_stats(y1, y2, rng), "by_stratum": {}, "drawn_for_attribute": {},
               "vlm_vs_rater1": agree_stats(vlm, y1, rng), "vlm_vs_rater2": agree_stats(vlm, y2, rng)}
        for k in sorted(set(kind)):
            m = kind == k; rec["by_stratum"][k] = agree_stats(y1[m], y2[m])
            m2 = m & (drawn == at); rec["drawn_for_attribute"][k] = agree_stats(y1[m2], y2[m2])
            if k == "rater1_positive":                                   # all rater-1 positive by construction
                rec["drawn_for_attribute"][k]["rater2_confirms"] = float(y2[m2].mean())
        cons = y1 == y2
        grades = {"rater2": (np.ones(len(toks), bool), y2), "rater1": (np.ones(len(toks), bool), y1),
                  "consensus": (cons, y1), "union": (np.ones(len(toks), bool), y1 | y2),
                  "intersection": (np.ones(len(toks), bool), y1 & y2)}
        rec["n_consensus"] = int(cons.sum()); rec["positives"] = {g: int(l[m].sum()) for g, (m, l) in grades.items()}
        rec["mAP"] = {}
        sig = lup = None
        for label, meth, _ in METHODS:
            s = call(meth, QUERY[at], at)
            if s is None: rec["mAP"][label] = None; continue
            s = np.asarray(s, float)[..., ix]; scores.setdefault(label, {})[at] = s
            if meth == "siglip2": sig = s
            if meth == "lookup": lup = s
        full_sig, full_lup = np.asarray(call("siglip2", QUERY[at], at), float), np.asarray(call("lookup", QUERY[at], at), float)
        scores.setdefault("Late fusion SigLIP2 + lookup", {})[at] = (zs(full_sig) + zs(full_lup))[ix]
        for label in out["methods"]:
            if at not in scores.get(label, {}): rec["mAP"][label] = None; continue
            s = scores[label][at]
            rec["mAP"][label] = {g: ap_of(s[..., m], l[m], per_seed=np.ndim(s) == 2) for g, (m, l) in grades.items()}
        out["attributes"][at] = rec
        hh = rec["human_human"]
        print(f"  {at:<13} kappa {hh['kappa']:+.3f} [{hh['kappa_ci95'][0]:+.2f},{hh['kappa_ci95'][1]:+.2f}]  agree {hh['agreement']:.3f}"
              f"  VLM-r1 {rec['vlm_vs_rater1']['kappa']:+.3f}  VLM-r2 {rec['vlm_vs_rater2']['kappa']:+.3f}  consensus n={rec['n_consensus']}", flush=True)

    A = out["attributes"]
    mac = lambda f: float(np.mean([f(A[a]) for a in ATTRS]))
    out["macro"] = {
        "human_human": {k: mac(lambda r: r["human_human"][k]) for k in ("kappa", "agreement", "positive_agreement", "negative_agreement")},
        "vlm_vs_rater1_kappa": mac(lambda r: r["vlm_vs_rater1"]["kappa"]),
        "vlm_vs_rater2_kappa": mac(lambda r: r["vlm_vs_rater2"]["kappa"]),
        "by_stratum_kappa": {k: mac(lambda r: r["by_stratum"][k]["kappa"]) for k in out["strata"]},
        "mAP": {lab: (None if any(A[a]["mAP"][lab] is None for a in ATTRS) else
                      {g: mac(lambda r: r["mAP"][lab][g]) for g in ("rater2", "rater1", "consensus", "union", "intersection")})
                for lab in out["methods"]}}
    # pooled kappa over all 4 x 160 decisions (a second summary, less sensitive to low-prevalence attributes)
    Y1 = np.concatenate([[r1[t][a] for t in toks] for a in ATTRS]); Y2 = np.concatenate([[r2[t][a] for t in toks] for a in ATTRS])
    out["macro"]["pooled_kappa"] = kappa(Y1, Y2)

    # paired scene bootstrap, macro mAP against rater 2 (per-seed protocol inside each resample)
    def macro_ap(lab, idx, grade):
        v = []
        for a in ATTRS:
            s = scores[lab][a]; y = np.array([(r2 if grade == "rater2" else r1)[toks[i]][a] for i in idx])
            v.append(ap_of(s[..., idx], y, per_seed=np.ndim(s) == 2))
        return float(np.mean(v))
    ref = "SceneMiner"; out["paired_bootstrap_rater2"] = {}
    boots = [rng.integers(0, len(toks), len(toks)) for _ in range(1000)]
    for lab in ("SceneMiner (portable)", "SigLIP2 ViT-B/16", "Qwen2.5-VL lookup", "Late fusion SigLIP2 + lookup", "Naive uncalibrated fusion"):
        d = np.array([macro_ap(ref, b, "rater2") - macro_ap(lab, b, "rater2") for b in boots])
        point = out["macro"]["mAP"][ref]["rater2"] - out["macro"]["mAP"][lab]["rater2"]
        out["paired_bootstrap_rater2"][lab] = {"delta": float(point), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                               "p_one_sided": float((1 + (d <= 0).sum()) / (len(d) + 1))}
        print(f"  {ref} - {lab:<30} {point:+.3f}  CI [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]", flush=True)

    M = out["macro"]
    print(f"  macro kappa {M['human_human']['kappa']:.3f} (pooled {M['pooled_kappa']:.3f}); VLM-r1 {M['vlm_vs_rater1_kappa']:.3f} VLM-r2 {M['vlm_vs_rater2_kappa']:.3f}")
    for lab in out["methods"]:
        v = M["mAP"][lab]
        if v: print(f"  {lab:<40} " + "  ".join(f"{g} {x:.3f}" for g, x in v.items()))
    out["seconds"] = time.time() - t0
    json.dump(out, open("results/cvpr/milestone33_rater2_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
