#!/usr/bin/env python3
"""MILESTONE 41 - Table 4 conjunction-column check.

Scores every Table 4 ranker on the 258 nuScenes evaluation scenes over two query sets:
  s5_20      the 20 Stratum-5 conjunction queries (the set behind Table 2's semantic
             column; 8 night/rain, 12 degradation)
  synergy_30 the set behind the published Table 4 column (milestone27_compositional_synergy):
             11 generated object x condition conjunctions with >= 5 positives plus the 19
             Stratum-5 conjunction phrasings whose object flag is one of the four generated
             ones (the "rare category" phrasing is excluded)
SceneMiner (monocular, as in Tables 2 and 4) as five-seed ensemble and as per-seed mean.
Late fusions z-score over the 258 scenes. Paired tests: SceneMiner ensemble minus each
ranker, sign-flip permutation (10,000) over queries, and over predicates for synergy_30.
-> results/cvpr/milestone41_table4_check.json
"""
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from paths import HUMAN_R1, DEVICE
from eval_attribute_index import ap_binary
from eval_human_gt_full import load_human
from eval_semantic_edge_cases import text_embed, load_v23, zs, COND_COL, META_OF

dev = torch.device(DEVICE)
SEM = {"pedestrian_close": "pedestrians close to the vehicle", "vru": "a cyclist or pedestrian on the road",
       "construction_zone": "a construction zone with cones", "heavy_vehicle": "a truck or bus"}


def perm(d, n=10000, seed=0):
    d = np.asarray(d, float); fl = np.random.default_rng(seed).choice([-1, 1], (n, len(d)))
    return float(((np.abs((fl * d).mean(1)) >= abs(d.mean())).sum() + 1) / (n + 1))


def main():
    W = Path("work_dirs/unified_filter")
    G = json.load(open(W/"human_semantic_gt.json"))["scenes"]
    z = np.load(W/"features_v2.npz", allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]; cidx = {t: i for i, t in enumerate(corpus)}
    hum = load_human(HUMAN_R1, "B")
    ev = set(json.load(open("benchmark_cvpr/split_906_602.json"))["eval"])
    toks = [t for t in corpus if t in ev and t in G and t in hum]; idx = np.array([cidx[t] for t in toks])
    syn = json.load(open("results/cvpr/milestone27_compositional_synergy.json"))
    s5 = [q for q in json.load(open("benchmark_cvpr/queries_semantic_edge_cases.json"))["queries"]
          if q["stratum"] == "semantic_compositional"]
    sets = {"s5_20": [{"text": q["text"], "predicate": q["predicate"], "condition": q["condition"]} for q in s5],
            "synergy_30": [{"text": p["text"], "predicate": p["predicate"], "condition": p["condition"]} for p in syn["per_query"]]}
    truth = lambda t, k: G[t][k] if (k in G[t] and k not in ("droplets", "glare", "wet_road", "low_contrast")) else hum[t][k]
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    emb, tlog, risk, sig, mz, prior = T("emb"), T("tag_logits"), T("risk"), T("siglip_img"), T("vlm_margin_z"), T("prior")
    qwen = torch.load(W/"V_frozen_cache.pt", map_location=dev).float()
    M = load_v23(risk.reshape(-1).cpu().numpy(), which="v26"); [setattr(m, "use_abstain", True) for m in M]
    vlm = json.load(open(W/"binary_flags_with_logits.json")); strict = json.load(open(W/"lowcontrast_strict.json"))
    meta = json.load(open(W/"deterministic_gt.json"))
    col = lambda t, c: float(((strict, "low_contrast_strict") if c == "low_contrast" else (vlm, c))[0]
                             .get(t, {}).get(("low_contrast_strict" if c == "low_contrast" else c) + "_margin", -9.))
    N = len(corpus); off = torch.zeros(N, device=dev)
    out = {"doc": __doc__, "n_scenes": len(toks), "sets": {}}
    for name, Q in sets.items():
        QT = text_embed([q["text"] for q in Q]); QN = F.normalize(QT, dim=-1)
        rows = {}
        for qi, q in enumerate(Q):
            rel = np.array([all(truth(t, k) == v for k, v in q["predicate"].items()) for t in toks])
            cc = COND_COL[q["condition"]]; fld, val = META_OF[q["condition"]]
            cos = (sig[idx] @ QN[qi]).cpu().numpy(); look = np.array([col(t, cc) for t in toks])
            md = np.array([float(meta.get(t, {}).get(fld) == val) for t in toks])
            qv = QT[qi].unsqueeze(0).expand(N, -1)
            with torch.no_grad():
                S = np.stack([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=off).cpu().numpy()[idx] for m in M])
            ap = lambda s: ap_binary(np.argsort(-s), rel)
            r = {"SceneMiner (5-seed ensemble)": ap(S.mean(0)), "SceneMiner (per-seed mean)": float(np.mean([ap(s) for s in S])),
                 "zero-shot SigLIP2": ap(cos), "logit VLM lookup": ap(look), "fusion SigLIP2 + lookup": ap(zs(cos) + zs(look)),
                 "fusion SigLIP2 + metadata": ap(zs(cos) + md)}
            for k, v in r.items(): rows.setdefault(k, []).append(v)
        res = {"n_queries": len(Q), "mAP": {k: float(np.mean(v)) for k, v in rows.items()}, "tests_by_query": {}}
        ref = np.array(rows["SceneMiner (5-seed ensemble)"])
        for k in rows:
            if k.startswith("SceneMiner"): continue
            d = ref - np.array(rows[k]); res["tests_by_query"][k] = {"delta": float(d.mean()), "perm_p": perm(d), "wins": int((d > 0).sum())}
        if name == "synergy_30":
            grp = {}
            for i, q in enumerate(Q): grp.setdefault(json.dumps(q["predicate"], sort_keys=True), []).append(i)
            res["tests_by_predicate"] = {}
            for k in rows:
                if k.startswith("SceneMiner"): continue
                d = np.array([np.mean([ref[i] - rows[k][i] for i in g]) for g in grp.values()])
                res["tests_by_predicate"][k] = {"delta": float(d.mean()), "perm_p": perm(d), "wins": int((d > 0).sum()), "n_predicates": len(d)}
        out["sets"][name] = res
        print(f"== {name} ({len(Q)} queries)")
        for k, v in res["mAP"].items(): print(f"   {k:<30} {v:.4f}")
        for kind in ("tests_by_query", "tests_by_predicate"):
            for k, v in res.get(kind, {}).items(): print(f"   [{kind}] SceneMiner - {k:<26} {v['delta']:+.4f} p={v['perm_p']:.4f} wins={v['wins']}")
    Path("results/cvpr/milestone41_table4_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
