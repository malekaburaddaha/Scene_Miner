#!/usr/bin/env python3
"""MILESTONE 41 - E3: retrieval-to-detection-error on val150, controlled for night/rain.

Same rankers, queries and per-scene front-sector metrics as Table 6
(scripts/val150_retrieval_to_3d.py --model v26): the 62 degradation queries of the
150-query suite; SceneMiner v2.6 portable with recorded night/rain metadata, the VLM
margin lookup, night/rain metadata, zero-shot SigLIP2 and random; each ranker's top 25.
Strata from the human scene description: day-dry (neither night nor rain) and
night-or-rain.

  A. Table 6 as published: top-25 mean minus the val150 mean, one-sided p under the
     scene permutation null (all 150 scenes exchangeable).
  B. Stratum-preserving null: the same top-25 sets, but per-scene metrics are permuted
     only within each stratum, so the null keeps each set's night/rain composition.
     Reported: the excess over the stratum-matched expectation and its one-sided p.
  C. Within-stratum retrieval: each ranker ranks only the scenes of one stratum and
     keeps the same share of it as top-25 does of 150 (day-dry: 18 of 111;
     night-or-rain: 7 of 39); delta to that stratum's mean, permutation within it.
  D. Paired tests, SceneMiner minus {night/rain metadata, SigLIP2, lookup}: both
     rankers' sets fixed, metrics permuted (globally, and within strata), two-sided.
Also the night/rain share of every ranker's top 25.
-> results/cvpr/milestone41_e3_val150_stratified.json
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from paths import DEVICE
from eval_semantic_edge_cases import text_embed, load_v23, portable, TRAIN_ORDER

dev = torch.device(DEVICE)
METRICS = {"mAP_front": -1, "fp_rate": +1, "recall_far": -1, "vru_recall": -1}    # +1: higher is worse
META_OF = {"droplets": "rain", "wet_road": "rain", "glare": "night", "low_contrast": "night"}
NP = int(os.environ.get("SM_NPERM", 5000))
SM, MD, SG, LK, RD = "SceneMiner", "night/rain metadata", "zero-shot SigLIP2", "VLM margin lookup", "random"


def main():
    per = json.load(open("work_dirs/val150_per_scene_front.json"))
    G = json.load(open("work_dirs/unified_filter/human_semantic_gt.json"))["scenes"]
    vz = np.load("work_dirs/val150_siglip.npz", allow_pickle=False); scenes = [str(t) for t in vz["scenes"]]
    S = torch.from_numpy(vz["siglip"]).float().to(dev)
    mz = np.load("work_dirs/val150_camfront_signals.npz", allow_pickle=False); msc = [str(t) for t in mz["scenes"]]
    ASKED = ["wet_road", "glare", "droplets", "low_contrast"]
    VM = mz["margin"][[msc.index(t) for t in scenes]][:, [ASKED.index(k) for k in TRAIN_ORDER]]
    MZt = torch.from_numpy((VM - VM.mean(0)) / (VM.std(0) + 1e-6)).float().to(dev)
    X = {m: np.array([per[t][m] for t in scenes], float) for m in METRICS}
    n = len(scenes)
    adverse = np.array([bool(G[t]["night"] or G[t]["rain"]) for t in scenes])
    strata = {"day-dry": np.where(~adverse)[0], "night-or-rain": np.where(adverse)[0]}
    deg = [q for q in json.load(open("benchmark_cvpr/queries_390.json"))["eval"] if q["stratum"] == "sensor_degradation"]
    QS = [(q["text"], q.get("attribute") or next(iter(q["predicate"]))) for q in deg]
    QT = text_embed([q[0] for q in QS]); QN = F.normalize(QT, dim=-1)
    M = load_v23(np.array([0.417797, 0.081112]), which="v26")
    VPR = torch.zeros(n, 26, device=dev)
    for i, t in enumerate(scenes):
        VPR[i, 20] = float(G[t]["night"]); VPR[i, 21] = float(not G[t]["night"])
        VPR[i, 22] = float(G[t]["rain"]); VPR[i, 23] = float(not G[t]["rain"])
    rng = np.random.default_rng(0)          # metadata tie-breaks: the same stream as val150_retrieval_to_3d.py
    rnd = np.random.default_rng(1)          # the random ranker
    R = {k: [] for k in (SM, LK, MD, SG, RD)}
    for qi, (text, attr) in enumerate(QS):
        qv = QT[qi].unsqueeze(0).expand(n, -1)
        R[SM].append(portable(M, qv, S, MZt, abstain=True, PR=VPR))
        R[SG].append((S @ QN[qi]).cpu().numpy())
        R[LK].append(VM[:, TRAIN_ORDER.index(attr)])
        R[MD].append(np.array([float(G[t][META_OF[attr]]) for t in scenes]) + rng.random(n) * 1e-6)
        R[RD].append(rnd.random(n))
    out = {"doc": __doc__, "n_scenes": n, "n_queries": len(QS), "n_perm": NP,
           "strata": {k: int(len(v)) for k, v in strata.items()},
           "degraded_flags_by_stratum": {k: int(((VM > 0).any(1))[v].sum()) for k, v in strata.items()}}

    def topsets(scores, k, pool=None):
        return np.stack([(np.argsort(-s) if pool is None else pool[np.argsort(-s[pool])])[:k] for s in scores])

    def setmean(v, T):
        """mean over queries of the per-set nanmean (sets with no valid value, possible for
        VRU recall in the small within-stratum sets, are skipped)"""
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return float(np.nanmean(np.nanmean(v[T], 1)))

    def perm_within(v, groups):
        vp = v.copy()
        for g in groups: vp[g] = v[rng.permutation(g)]
        return vp

    rng = np.random.default_rng(2)          # permutations
    allg = [np.arange(n)]; strg = list(strata.values())
    T25 = {r: topsets(s, 25) for r, s in R.items()}
    out["top25_night_or_rain_share"] = {r: float(np.mean([adverse[t].mean() for t in T])) for r, T in T25.items()}
    draws = [topsets([np.random.default_rng(1000 + d).random(n) for _ in QS], 25) for d in range(50)]
    out["random_50_draw_delta"] = {m: float(np.mean([setmean(X[m], T) for T in draws]) - np.nanmean(X[m])) for m in METRICS}
    # A and B
    out["A_global_null"], out["B_stratum_null"] = {}, {}
    for r, T in T25.items():
        out["A_global_null"][r], out["B_stratum_null"][r] = {}, {}
        for m, sg in METRICS.items():
            v = X[m]; obs = setmean(v, T); base = float(np.nanmean(v))
            nullA = np.array([setmean(perm_within(v, allg), T) for _ in range(NP)])
            nullB = np.array([setmean(perm_within(v, strg), T) for _ in range(NP)])
            out["A_global_null"][r][m] = {"delta": obs - base, "p_worse": float(((sg * (nullA - base) >= sg * (obs - base)).sum() + 1) / (NP + 1))}
            eB = float(nullB.mean())
            out["B_stratum_null"][r][m] = {"delta_vs_mean": obs - base, "stratum_matched_expectation_delta": eB - base,
                                           "excess": obs - eB, "p_worse": float(((sg * (nullB - eB) >= sg * (obs - eB)).sum() + 1) / (NP + 1))}
    # C
    out["C_within_stratum"] = {}
    for name, pool in strata.items():
        k = int(round(25 * len(pool) / n)); res = {"k": k, "n": len(pool), "rankers": {}}
        for r, scores in R.items():
            T = topsets(scores, k, pool); res["rankers"][r] = {}
            for m, sg in METRICS.items():
                v = X[m]; obs = setmean(v, T); base = float(np.nanmean(v[pool]))
                null = np.array([setmean(perm_within(v, [pool]), T) for _ in range(NP)])
                res["rankers"][r][m] = {"delta": obs - base, "p_worse": float(((sg * (null - base) >= sg * (obs - base)).sum() + 1) / (NP + 1))}
        out["C_within_stratum"][name] = res
    # D
    out["D_paired"] = {}
    for other in (MD, SG, LK):
        for scope, groups, sets_ in [("global", allg, (T25[SM], T25[other])), ("stratum-preserving", strg, (T25[SM], T25[other]))] + \
                [(f"within {nm}", [pool], (topsets(R[SM], int(round(25 * len(pool) / n)), pool),
                                           topsets(R[other], int(round(25 * len(pool) / n)), pool))) for nm, pool in strata.items()]:
            TA, TB = sets_
            for m in METRICS:
                v = X[m]; f = lambda vv: setmean(vv, TA) - setmean(vv, TB); obs = f(v)
                null = np.array([f(perm_within(v, groups)) for _ in range(NP)])
                out["D_paired"].setdefault(f"{SM} - {other}", {}).setdefault(scope, {})[m] = {
                    "delta": obs, "p_two_sided": float(((np.abs(null) >= abs(obs)).sum() + 1) / (NP + 1))}
    Path("results/cvpr/milestone41_e3_val150_stratified.json").write_text(json.dumps(out, indent=1))
    f = lambda d: " ".join(f"{m} {d[m].get('delta', d[m].get('excess')):+.4f} (p={d[m].get('p_worse', d[m].get('p_two_sided')):.4f})" for m in METRICS)
    print("strata", out["strata"], "degraded flags", out["degraded_flags_by_stratum"])
    print("night/rain share of top-25:", {k: round(v, 3) for k, v in out["top25_night_or_rain_share"].items()})
    for sec in ("A_global_null", "B_stratum_null"):
        print(f"== {sec}")
        for r, d in out[sec].items():
            print(f"   {r:<20} " + " ".join(f"{m} {d[m]['delta'] if 'delta' in d[m] else d[m]['excess']:+.4f}"
                                            f"{'' if sec[0] == 'A' else ' (exp ' + format(d[m]['stratum_matched_expectation_delta'], '+.4f') + ')'} p={d[m]['p_worse']:.4f}" for m in METRICS))
    for nm, res in out["C_within_stratum"].items():
        print(f"== C within {nm} (top {res['k']} of {res['n']})")
        for r, d in res["rankers"].items(): print(f"   {r:<20} {f(d)}")
    print("== D paired")
    for k, scopes in out["D_paired"].items():
        for sc, d in scopes.items(): print(f"   {k:<36} [{sc}] {f(d)}")


if __name__ == "__main__":
    main()
