#!/usr/bin/env python3
"""MILESTONE 16 - does retrieval surface scenes where 3D perception breaks?

On the 150 uncontaminated nuScenes val scenes, each ranker ranks the scenes for
every query; we then read the BEVFormer CAM_FRONT-sector metrics of what it
retrieved: the top-25 and each score quartile.

Queries: the 62 sensor-degradation queries of the frozen 150-query suite (not
chosen for this study) and the 20 atomic Stratum-5 semantic queries.

Rankers (portable signals only; nothing else exists for val150):
  SceneMiner v2.3 portable, v2.3A portable (post-hoc abstention variant),
  zero-shot SigLIP2, logit VLM lookup (the query's attribute column), metadata
  matched (human scene description: night for glare/low-contrast, rain for
  droplets/wet-road), random. A ranker with nothing to say about a query
  (lookup or metadata on a semantic query) is a random ranker for that query.

Statistics. The null is NOT independent random subsets per query: queries
overlap heavily in what they retrieve, which would make that test
anti-conservative. Instead the per-scene metric vector is permuted across the
150 scenes (2000 times) while every ranker's retrieved sets are held fixed, and
the aggregate over queries recomputed. One-sided p in the "worse" direction.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_semantic_edge_cases import text_embed, load_v23, portable, TRAIN_ORDER

dev = torch.device("cuda")
METRICS = {"mAP_front": -1, "fp_rate": +1, "recall_far": -1, "mATE_front": +1,
           "vru_recall": -1, "vru_ATE": +1}          # sign: +1 means higher = worse
META_OF = {"droplets": "rain", "wet_road": "rain", "glare": "night", "low_contrast": "night"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", default="work_dirs/val150_per_scene_front.json")
    ap.add_argument("--gt", default="work_dirs/unified_filter/human_semantic_gt.json")
    ap.add_argument("--suite", default="benchmark_cvpr/queries_390.json")
    ap.add_argument("--semantic", default="benchmark_cvpr/queries_semantic_edge_cases.json")
    ap.add_argument("--val-margins", default="work_dirs/val150_camfront_signals.npz")
    ap.add_argument("--val-siglip", default="work_dirs/val150_siglip.npz")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--out", default="results/cvpr/milestone16_val150_retrieval_to_3d.json")
    ap.add_argument("--model", default="v23", choices=["v23", "v24"])
    a = ap.parse_args()
    per = json.load(open(a.per_scene)); G = json.load(open(a.gt))["scenes"]
    vz = np.load(a.val_siglip, allow_pickle=False); scenes = [str(t) for t in vz["scenes"]]
    S = torch.from_numpy(vz["siglip"]).float().to(dev)
    mz = np.load(a.val_margins, allow_pickle=False); msc = [str(t) for t in mz["scenes"]]
    ASKED = ["wet_road", "glare", "droplets", "low_contrast"]
    VM = mz["margin"][[msc.index(t) for t in scenes]][:, [ASKED.index(k) for k in TRAIN_ORDER]]
    MZt = torch.from_numpy((VM - VM.mean(0)) / (VM.std(0) + 1e-6)).float().to(dev)
    X = {m: np.array([per[t][m] for t in scenes], float) for m in METRICS}
    n = len(scenes)

    deg = [q for q in json.load(open(a.suite))["eval"] if q["stratum"] == "sensor_degradation"]
    sem = [q for q in json.load(open(a.semantic))["queries"] if q["stratum"] == "semantic_atomic"]
    QS = [(q["text"], "degradation", q.get("attribute") or next(iter(q["predicate"]))) for q in deg] + \
         [(q["text"], "semantic", q["semantic"]) for q in sem]
    QT = text_embed([q[0] for q in QS]); QN = F.normalize(QT, dim=-1)
    M = load_v23(np.array([0.417797, 0.081112]), which=a.model)   # risk stats unused in portable mode
    for m in M: m.use_abstain = (a.model == "v24")
    rng = np.random.default_rng(0)

    # rankings: ranker -> list of score vectors (None = no information -> random)
    SM = "SceneMiner_v2.4" if a.model == "v24" else "SceneMiner_v2.3"
    R = {k: [] for k in [SM, "zeroshot_SigLIP2", "logit_VLM_lookup", "metadata_matched", "random"]}
    for qi, (text, kind, attr) in enumerate(QS):
        qv = QT[qi].unsqueeze(0).expand(n, -1)
        R[SM].append(portable(M, qv, S, MZt, abstain=(a.model == "v24")))
        R["zeroshot_SigLIP2"].append((S @ QN[qi]).cpu().numpy())
        if kind == "degradation":
            R["logit_VLM_lookup"].append(VM[:, TRAIN_ORDER.index(attr)])
            R["metadata_matched"].append(np.array([float(G[t][META_OF[attr]]) for t in scenes])
                                         + rng.random(n) * 1e-6)
        else:
            R["logit_VLM_lookup"].append(None); R["metadata_matched"].append(None)
        R["random"].append(None)

    def sets(scores, k, seed):
        """top-k index set per query; None -> a fixed random draw per query."""
        r = np.random.default_rng(seed); out = []
        for s in scores:
            out.append(r.permutation(n)[:k] if s is None else np.argsort(-s)[:k])
        return out

    def quartiles(scores, seed):
        r = np.random.default_rng(seed); out = []
        for s in scores:
            o = r.permutation(n) if s is None else np.argsort(-s)
            out.append(np.array_split(o, 4))
        return out

    kinds = np.array([q[1] for q in QS])
    res = {"n_scenes": n, "topk": a.topk, "n_queries": {"degradation": int((kinds == "degradation").sum()),
                                                       "semantic": int((kinds == "semantic").sum())},
           "corpus_mean": {m: float(np.nanmean(v)) for m, v in X.items()},
           "topk": {}, "quartiles": {}, "doc": __doc__}
    for kind in ["degradation", "semantic", "all"]:
        qsel = np.where((kinds == kind) | (kind == "all"))[0]
        res["topk"][kind] = {}; res["quartiles"][kind] = {}
        for rk, scores in R.items():
            # random rankers: average 50 draws for the point estimate
            draws = 50 if all(scores[i] is None for i in qsel) else 1
            agg = {m: [] for m in METRICS}; qagg = {m: np.zeros(4) for m in METRICS}
            for d_ in range(draws):
                T = sets([scores[i] for i in qsel], a.topk, d_)
                Qs = quartiles([scores[i] for i in qsel], d_)
                for m, v in X.items():
                    agg[m].append(np.mean([np.nanmean(v[t]) for t in T]))
                    qagg[m] += np.array([np.mean([np.nanmean(v[q[j]]) for q in Qs]) for j in range(4)]) / draws
            out = {}
            T = sets([scores[i] for i in qsel], a.topk, 0)
            for m, sgn in METRICS.items():
                obs = float(np.mean(agg[m])); base = float(np.nanmean(X[m]))
                # scene-label permutation null with retrieved sets held fixed
                v = X[m]; null = np.empty(a.perm)
                for p in range(a.perm):
                    vp = v[rng.permutation(n)]
                    null[p] = np.mean([np.nanmean(vp[t]) for t in T])
                p_worse = float(((sgn * (null - base) >= sgn * (obs - base)).sum() + 1) / (a.perm + 1))
                out[m] = {"topk_mean": obs, "corpus_mean": base, "delta": obs - base,
                          "delta_sigma": (obs - base) / (np.nanstd(v) + 1e-12),
                          "null_sd": float(null.std()), "p_worse": p_worse}
            res["topk"][kind][rk] = out
            res["quartiles"][kind][rk] = {m: qagg[m].tolist() for m in METRICS}
    # ---- pairwise: is ranker A's top-k worse than ranker B's? same scene-label
    # permutation, applied to both rankers at once, so the null keeps both sets fixed
    res["pairwise"] = {}
    qsel = np.where(kinds == "degradation")[0]
    for A_, B_ in [(SM, "metadata_matched"), (SM, "zeroshot_SigLIP2"),
                   (SM, "logit_VLM_lookup"), ("metadata_matched", "zeroshot_SigLIP2")]:
        TA = sets([R[A_][i] for i in qsel], a.topk, 0); TB = sets([R[B_][i] for i in qsel], a.topk, 0)
        for m, sgn in METRICS.items():
            v = X[m]
            f = lambda vv: np.mean([np.nanmean(vv[t]) for t in TA]) - np.mean([np.nanmean(vv[t]) for t in TB])
            obs = f(v); null = np.array([f(v[rng.permutation(n)]) for _ in range(a.perm)])
            res["pairwise"][f"{A_} - {B_} | {m}"] = {
                "delta": float(obs), "p_two_sided": float(((np.abs(null) >= abs(obs)).sum() + 1) / (a.perm + 1))}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("\n=== pairwise, degradation queries, top-k ===")
    for k, v in res["pairwise"].items():
        print(f"  {k:<58} delta {v['delta']:+.4f}  p={v['p_two_sided']:.3f}")
    for kind in ["degradation", "semantic"]:
        print(f"\n=== {kind} queries (n={res['n_queries'][kind]}): top-{a.topk} vs corpus mean ===")
        print(f"  {'ranker':<20}" + "".join(f"{m:>17}" for m in METRICS))
        for rk, o in res["topk"][kind].items():
            print(f"  {rk:<20}" + "".join(f"{o[m]['delta']:>+9.4f} p{o[m]['p_worse']:.3f}" for m in METRICS))
        print(f"  quartiles (Q1 = top-scored) front mAP / FP rate:")
        for rk, o in res["quartiles"][kind].items():
            print(f"  {rk:<20} mAP " + " ".join(f"{x:.4f}" for x in o["mAP_front"]) +
                  "  | FP " + " ".join(f"{x:.4f}" for x in o["fp_rate"]))
    print(f"\ncorpus means: " + "  ".join(f"{m} {v:.4f}" for m, v in res["corpus_mean"].items()))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
