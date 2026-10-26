#!/usr/bin/env python3
"""MILESTONE 19 - where does fusion beat both single-modality baselines?

Three regimes, every one graded by HUMANS:
  atomic degradation   550 rater-labelled scenes, 4 attributes (Milestone 18)
  atomic semantics     Stratum-5 atomic queries, corpus 258 + clean val150
  conjunctions         semantic object (nuScenes 3D annotation) x adverse
                       condition (night/rain from the human scene description;
                       glare/wet_road/droplets/low_contrast from the rater)

Conjunctions are generated, not hand-picked: the 4 semantic flags
(pedestrian_close, vru, construction_zone, heavy_vehicle) are crossed with the 6
conditions, every cell with >= 5 positives is kept, and each gets one templated
query "{object phrase} {condition phrase}". The Stratum-5 hand phrasings are
reported alongside. Conjunctions involving VLM-defined degradation are NOT
graded on val150: there the only degradation labels are the VLM's own margins,
which would make the ground truth identical to one of the rankers.

Controls: the obvious objection to any fusion win is "you just added SigLIP2
and the lookup". So late fusion z(SigLIP2) + z(lookup) is a ranker here, as is
z(SigLIP2) + metadata.
"""
from __future__ import annotations
from paths import HUMAN_R1
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_semantic_edge_cases import (text_embed, load_v23, portable, metrics, paired,
                                      TRAIN_ORDER, COND_COL, META_OF, zs)
from eval_human_gt_full import load_human, ATTRS

dev = torch.device("cuda")
import os
FINAL = os.environ.get("SM_FINAL", "v24")
SEM = {"pedestrian_close": "pedestrians near the road", "vru": "a cyclist or motorcyclist",
       "construction_zone": "a construction zone with traffic cones",
       "heavy_vehicle": "a bus or large truck"}
COND = {"night": "at night", "rain": "in the rain", "glare": "with strong headlight glare",
        "wet_road": "on a wet reflective road", "droplets": "seen through raindrops on the lens",
        "low_contrast": "in a dark low-contrast scene"}


def main():
    W = Path("work_dirs/unified_filter")
    G = json.load(open("work_dirs/unified_filter/human_semantic_gt.json"))["scenes"]
    z = np.load(W/"features_v2.npz", allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]; cidx = {t: i for i, t in enumerate(corpus)}
    hum = load_human(HUMAN_R1, "B")
    ev = set(json.load(open("benchmark_cvpr/split_906_602.json"))["eval"])
    toks = [t for t in corpus if t in ev and t in G and t in hum]; idx = np.array([cidx[t] for t in toks])
    truth = lambda t, k: G[t][k] if k in ("night", "rain") or k in SEM else hum[t][k]

    # ---- generated conjunctions (corpus 258)
    gen = []
    for s_, sp in SEM.items():
        for c_, cp in COND.items():
            n = sum(truth(t, s_) and truth(t, c_) for t in toks)
            if n >= 5: gen.append({"text": f"{sp} {cp}", "predicate": {s_: True, c_: True},
                                   "semantic": s_, "condition": c_, "source": "generated", "n_pos": n})
    s5 = [dict(q, source="stratum5") for q in json.load(open("benchmark_cvpr/queries_semantic_edge_cases.json"))["queries"]
          if q["stratum"] == "semantic_compositional" and q["semantic"] in SEM]
    Q = gen + s5
    print(f"[synergy] {len(gen)} generated conjunctions with >=5 positives, {len(s5)} Stratum-5 phrasings", flush=True)

    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    emb, tlog, risk = T("emb"), T("tag_logits"), T("risk")
    sig, mz, prior = T("siglip_img"), T("vlm_margin_z"), T("prior")
    qwen = torch.load(W/"V_frozen_cache.pt", map_location=dev).float()
    rk = risk.reshape(-1).cpu().numpy()
    M24 = load_v23(rk, which=FINAL); [setattr(m, "use_abstain", True) for m in M24]
    M23 = load_v23(rk, which="v23")
    vlm = json.load(open(W/"binary_flags_with_logits.json")); strict = json.load(open(W/"lowcontrast_strict.json"))
    meta = json.load(open(W/"deterministic_gt.json"))
    QT = text_embed([q["text"] for q in Q]); QN = F.normalize(QT, dim=-1)
    N = len(corpus); on, off = torch.ones(N, device=dev), torch.zeros(N, device=dev)

    def col(t, c):
        src, key = (strict, "low_contrast_strict") if c == "low_contrast" else (vlm, c)
        return float(src.get(t, {}).get(key + "_margin", -9.))

    rows = {}; per = []
    for qi, q in enumerate(Q):
        rel = np.array([all(truth(t, k) == v for k, v in q["predicate"].items()) for t in toks])
        cc = COND_COL[q["condition"]]
        qv = QT[qi].unsqueeze(0).expand(N, -1)
        R = {}
        R["zeroshot_SigLIP2"] = (sig[idx] @ QN[qi]).cpu().numpy()
        R["logit_VLM_lookup"] = np.array([col(t, cc) for t in toks])
        fld, val = META_OF[q["condition"]]
        mt = np.array([float(meta.get(t, {}).get(fld) == val) for t in toks])
        R["metadata_matched"] = mt + np.random.default_rng(qi).random(len(toks)) * 1e-6
        R["late_fusion SigLIP2+lookup"] = zs(R["zeroshot_SigLIP2"]) + zs(R["logit_VLM_lookup"])
        R["late_fusion SigLIP2+metadata"] = zs(R["zeroshot_SigLIP2"]) + mt
        with torch.no_grad():
            R["v2.4_monocular"] = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=off)
                                           .cpu().numpy() for m in M24], 0)[idx]
            R["v2.4_full [contam.]"] = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=on)
                                                .cpu().numpy() for m in M24], 0)[idx]
            for m in M23: m.use_abstain = True
            R["v2.3A_monocular"] = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=off)
                                            .cpu().numpy() for m in M23], 0)[idx]
            for m in M23: m.use_abstain = False
            R["v2.3_monocular"] = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=off)
                                           .cpu().numpy() for m in M23], 0)[idx]
            _, g24, _ = M24[0].concept(QT[qi:qi+1]); _, g23, _ = M23[0].concept(QT[qi:qi+1])
        res = {k: metrics(v, rel) for k, v in R.items()}
        for k, v in res.items(): rows.setdefault(k, []).append(v)
        per.append({**{k: q[k] for k in ("text", "predicate", "source", "semantic", "condition")},
                    "n_pos": int(rel.sum()), "gamma_v24": float(g24), "gamma_v23": float(g23),
                    "mAP": {k: v["mAP"] for k, v in res.items()}})

    def summ(sel):
        return {k: float(np.mean([rows[k][i]["mAP"] for i in sel])) for k in rows}
    out = {"doc": __doc__, "n_scenes": len(toks), "per_query": per, "summary": {}, "tests": {}}
    for name, sel in [("generated", [i for i, q in enumerate(Q) if q["source"] == "generated"]),
                      ("stratum5_phrasings", [i for i, q in enumerate(Q) if q["source"] == "stratum5"]),
                      ("all", list(range(len(Q))))]:
        out["summary"][name] = summ(sel)
    # paired permutation by PREDICATE over all phrasings (generated + Stratum 5)
    grp = {}
    for i, q in enumerate(Q): grp.setdefault(json.dumps(q["predicate"], sort_keys=True), []).append(i)
    rng = np.random.default_rng(0)
    for a_ in ["v2.4_monocular", "v2.3A_monocular"]:
        for b_ in ["zeroshot_SigLIP2", "logit_VLM_lookup", "late_fusion SigLIP2+lookup", "metadata_matched",
                   "late_fusion SigLIP2+metadata"]:
            d = np.array([np.mean([rows[a_][i]["mAP"] - rows[b_][i]["mAP"] for i in g]) for g in grp.values()])
            fl = rng.choice([-1, 1], size=(10000, len(d)))
            p = float(((np.abs((fl * d).mean(1)) >= abs(d.mean())).sum() + 1) / 10001)
            out["tests"][f"{a_} - {b_}"] = {"delta": float(d.mean()), "perm_p": p, "n_predicates": len(d),
                                            "wins": int((d > 0).sum())}
    # predicate-level table
    out["by_predicate"] = {k: {"n_pos": per[g[0]]["n_pos"],
                               **{r: float(np.mean([rows[r][i]["mAP"] for i in g])) for r in rows}}
                           for k, g in grp.items()}
    Path(os.environ.get("SM_OUT", "results/cvpr/milestone19_compositional_synergy.json")).write_text(json.dumps(out, indent=1))
    for name, d in out["summary"].items():
        print(f"\n-- {name} --")
        for k, v in sorted(d.items(), key=lambda kv: -kv[1]): print(f"   {k:<30}{v:.4f}")
    print("\n-- paired by predicate --")
    for k, v in out["tests"].items():
        print(f"   {k:<50} {v['delta']:+.4f} p={v['perm_p']:.4f} wins {v['wins']}/{v['n_predicates']}")
    print("\n-- by predicate (v2.4 mono / SigLIP2 / lookup / late fusion / metadata) --")
    for k, v in out["by_predicate"].items():
        print(f"   {'x'.join(json.loads(k)):<32} pos {v['n_pos']:>2}  {v['v2.4_monocular']:.3f} {v['zeroshot_SigLIP2']:.3f} "
              f"{v['logit_VLM_lookup']:.3f} {v['late_fusion SigLIP2+lookup']:.3f} {v['metadata_matched']:.3f}")


if __name__ == "__main__":
    main()
