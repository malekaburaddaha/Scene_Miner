#!/usr/bin/env python3
"""MILESTONE 41 - E1: paired scene-bootstrap CIs on human-labelled degradation.

SceneMiner (full) and SceneMiner (portable) against training-free late fusion
z(SigLIP2) + z(lookup) and the Qwen2.5-VL attribute lookup, per attribute and
macro-averaged, graded by
  rater 1 on all 550 scenes (Mode B labels, as Table 2), and
  rater 2 on the 160 double-labelled scenes (as Table 3).
Protocol: trained methods score AP per seed and average the five seeds (the human-GT
protocol behind 0.794 / 0.762); each bootstrap resample draws scenes with
replacement and re-scores every method on the same resample (paired). Late fusion is
z-scored over the scenes of the published number it reproduces (the 550 scenes for
rater 1, the full corpus for rater 2); the other choice is reported as a check.
p: two-sided bootstrap p = 2 min(P*(d <= 0), P*(d >= 0)), floored at 1/(B+1).
With SM_CONTROL=<checkpoint family> (E2: v26tf) the training-free control is scored too and
compared with both SceneMiner configurations; output then goes to SM_OUT.
-> results/cvpr/milestone41_e1_bootstrap.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from paths import HUMAN_R1, HUMAN_R2
from eval_human_gt_full import load_human, ATTRS, QUERY
from eval_related_work_baselines import Enc, Rankers, load_corpus, load_captions, zs, ap_of, W
from eval_semantic_edge_cases import load_v23

NB = int(os.environ.get("SM_NBOOT", 2000))
REFS = ["SceneMiner (full)", "SceneMiner (portable)"]
OTHERS = ["Late fusion (SigLIP2 + lookup)", "Qwen2.5-VL attribute lookup"]
CONTROL = os.environ.get("SM_CONTROL")
if CONTROL: OTHERS = OTHERS + ["Training-free control"]


def main():
    t0 = time.time()
    enc = Enc(); C = load_corpus()
    z = np.load(W/"features_v2.npz", allow_pickle=False)
    R = Rankers(enc, {}, z["risk"].reshape(-1))
    cid = {t: i for i, t in enumerate(C["toks"])}
    r1, r2 = load_human(HUMAN_R1, "B"), load_human(HUMAN_R2, "B")
    sub = json.load(open("benchmark_cvpr/rater2_subset_160.json"))
    sets = {"rater1_550": ([t for t in C["toks"] if t in r1], r1),
            "rater2_160": ([t for t in sub["tokens"] if t in r1 and t in r2], r2)}
    full = {}
    for at in ATTRS:
        q = QUERY[at]
        full[at] = {"SceneMiner (full)": np.asarray(R.sceneminer(C, q), float),
                    "SceneMiner (portable)": np.asarray(R.sceneminer_portable(C, q), float),
                    "sig": np.asarray(R.siglip2(C, q), float), "look": np.asarray(R.lookup(C, q, at, 1), float)}
    if CONTROL:
        keep = R.M25; R.M25 = load_v23(z["risk"].reshape(-1), which=CONTROL); [setattr(m, "use_abstain", True) for m in R.M25]
        for at in ATTRS: full[at]["Training-free control"] = np.asarray(R.sceneminer(C, QUERY[at]), float)
        R.M25 = keep
    out = {"doc": __doc__, "n_boot": NB, "sets": {}}
    rng = np.random.default_rng(0)
    for name, (toks, lab) in sets.items():
        ix = np.array([cid[t] for t in toks]); n = len(toks)
        Y = {at: np.array([lab[t][at] for t in toks]) for at in ATTRS}
        S = {}
        for at in ATTRS:
            f = full[at]; s_sub, l_sub = f["sig"][ix], f["look"][ix]
            lf_local, lf_corpus = zs(s_sub) + zs(l_sub), (zs(f["sig"]) + zs(f["look"]))[ix]
            S[at] = {"SceneMiner (full)": f["SceneMiner (full)"][:, ix], "SceneMiner (portable)": f["SceneMiner (portable)"][:, ix],
                     "Late fusion (SigLIP2 + lookup)": lf_local if name == "rater1_550" else lf_corpus,
                     "Late fusion, other z-scoring set": lf_corpus if name == "rater1_550" else lf_local,
                     "Qwen2.5-VL attribute lookup": l_sub}
            if CONTROL: S[at]["Training-free control"] = f["Training-free control"][:, ix]
        meths = list(S[ATTRS[0]])
        ap = lambda m, at, idx: ap_of(S[at][m][..., idx], Y[at][idx], per_seed=np.ndim(S[at][m]) == 2)
        allix = np.arange(n)
        point = {m: {at: ap(m, at, allix) for at in ATTRS} for m in meths}
        for m in meths: point[m]["macro"] = float(np.mean([point[m][at] for at in ATTRS]))
        boots = [rng.integers(0, n, n) for _ in range(NB)]
        bt = {m: {at: np.empty(NB) for at in ATTRS} for m in REFS + OTHERS}
        for b, idx in enumerate(boots):
            for m in REFS + OTHERS:
                for at in ATTRS: bt[m][at][b] = ap(m, at, idx)
        for m in bt: bt[m]["macro"] = np.mean([bt[m][at] for at in ATTRS], 0)
        cmp = {}
        for a in REFS:
            for o in OTHERS:
                for k in ATTRS + ["macro"]:
                    d = bt[a][k] - bt[o][k]
                    p = 2 * min((d <= 0).mean(), (d >= 0).mean()); p = max(p, 1 / (NB + 1))
                    cmp.setdefault(f"{a} - {o}", {})[k] = {"delta": point[a][k] - point[o][k],
                                                           "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                                           "p_two_sided": float(min(1.0, p))}
        out["sets"][name] = {"n_scenes": n, "positives": {at: int(Y[at].sum()) for at in ATTRS}, "mAP": point, "paired": cmp}
        print(f"== {name} (n={n})", flush=True)
        for m in meths: print(f"   {m:<36} " + " ".join(f"{k} {point[m][k]:.3f}" for k in ATTRS + ["macro"]), flush=True)
        for c, v in cmp.items():
            print(f"   {c:<64} " + " | ".join(f"{k} {v[k]['delta']:+.3f} [{v[k]['ci95'][0]:+.3f},{v[k]['ci95'][1]:+.3f}] p={v[k]['p_two_sided']:.3f}"
                                            for k in ATTRS + ["macro"]), flush=True)
    out["seconds"] = time.time() - t0
    Path(os.environ.get("SM_OUT", "results/cvpr/milestone41_e1_bootstrap.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
