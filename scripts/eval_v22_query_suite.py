#!/usr/bin/env python3
"""MILESTONE 10 - SceneMiner v2.2 on the frozen 150-query eval suite.

Scope and a standing caveat. Relevance in this suite is a PREDICATE OVER THE
CACHED VLM ATTRIBUTES, so an index that reads those same attributes scores
exactly 1.000 by construction (MILESTONE 4). The suite is therefore NOT a test
of whether a ranker finds degraded scenes - it is a test of whether a ranker can
recover a known predicate from language alone, which is why the tautological
`attribute_index` row is printed as a CEILING rather than a baseline. The
informative rows are the `compositional_heldout` and `negated` strata, whose
predicates never appear in training.

It also serves a second purpose: validating two v2.2 design choices IN DOMAIN,
so they do not rest on BDD100K measurements (which would turn an external test
set into a validation set). Both are ablated at inference:
  * PartNorm      - `normalize=False` restores v2.1's incommensurate blend;
  * mean-centring - zeroing `text_mean` restores the raw text-text selector.
"""
from __future__ import annotations
import argparse, glob, json, sys
import os
SEED_GLOB = os.environ.get("SM_SEED_GLOB", "s*")      # Milestone 34: one seed at a time for +/- sd
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, p_at, ndcg_at, load_attributes, predicate_mask

ATTRS = ["droplets", "glare", "wet_road", "low_contrast"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="benchmark_cvpr/queries_390.json")
    ap.add_argument("--split-file", default="benchmark_cvpr/split_906_602.json")
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--ckpts", default="work_dirs/sceneminer_v22/" + SEED_GLOB + "/final.pt")
    ap.add_argument("--out", default="results/cvpr/milestone10_query_suite_150.json")
    a = ap.parse_args()
    dev = torch.device("cuda"); W = Path(a.work)

    z = np.load(a.features, allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    emb, tlog, risk = T("emb"), T("tag_logits"), T("risk")
    sig, mz, prior = T("siglip_img"), T("vlm_margin_z"), T("prior")
    qwen = torch.zeros(len(corpus), 2048, device=dev)
    vc = W/"V_frozen_cache.pt"
    if vc.exists(): qwen = torch.load(vc, map_location=dev).float()
    rk = risk.reshape(-1).cpu().numpy()
    assert list(z["margin_attrs"]) == ATTRS, list(z["margin_attrs"])

    # EVAL SPLIT ONLY - no evaluated scene ever contributed a gradient
    ev_toks = set(json.load(open(a.split_file))["eval"])
    keep = np.array([t in ev_toks for t in corpus])
    idx = np.where(keep)[0]
    toks = [corpus[i] for i in idx]
    print(f"[suite] {len(toks)} eval-split scenes", flush=True)

    attrs = load_attributes(W)
    for f in ("binary_flags_with_logits.json", "lowcontrast_strict.json"):
        p = W/f
        if p.exists():
            for t, r in json.load(open(p)).items():
                rec = {k: v for k, v in r.items() if not k.endswith(("_margin","_logp_yes"))}
                if "low_contrast_strict" in rec: rec["low_contrast"] = rec.pop("low_contrast_strict")
                attrs.setdefault(t, {}).update(rec)
    vlm = json.load(open(W/"binary_flags_with_logits.json"))
    strict = json.load(open(W/"lowcontrast_strict.json")) if (W/"lowcontrast_strict.json").exists() else {}

    QS = json.load(open(a.queries))["eval"]
    from transformers import AutoModel, GemmaTokenizerFast
    P = "google/siglip2-base-patch16-224"
    sm = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    tk = GemmaTokenizerFast.from_pretrained(P)
    ft = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    with torch.no_grad():
        ti = tk([q["text"] for q in QS], padding="max_length", truncation=True,
                max_length=64, return_tensors="pt").to(dev)
        QT = ft(sm.get_text_features(**ti))
    del sm; torch.cuda.empty_cache()

    from scene_mining.filter.sceneminer_v2_2 import SceneMinerV22
    from scene_mining.filter.sceneminer_v2 import SceneMinerV21
    M22 = []
    for c in sorted(glob.glob(a.ckpts)):
        d = torch.load(c, map_location=dev, weights_only=False)
        m = SceneMinerV22(d_qwen=qwen.shape[1], d_siglip=sig.shape[1],
                          n_margins=mz.shape[1], n_meta=6, d_text=QT.shape[1],
                          risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
        m.load_state_dict(d["state_dict"]); m.eval(); M22.append(m)
    M21 = []
    for c in sorted(glob.glob("work_dirs/sceneminer_v21/" + SEED_GLOB + "/final.pt")):
        d = torch.load(c, map_location=dev, weights_only=False)
        m = SceneMinerV21(d_qwen=qwen.shape[1], d_siglip=sig.shape[1],
                          n_margins=mz.shape[1], n_meta=6, d_text=QT.shape[1],
                          risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
        m.load_state_dict(d["state_dict"]); m.eval(); M21.append(m)
    print(f"[suite] {len(M22)} v2.2 seeds, {len(M21)} v2.1 seeds", flush=True)

    N = len(corpus)
    on, off = torch.ones(N, device=dev), torch.zeros(N, device=dev)
    av3 = torch.ones(N, 3, device=dev)
    av2 = torch.stack([on, on, off], -1)
    # saved text_mean, so the centring ablation can restore it
    saved_mu = [m.text_mean.clone() for m in M22]

    rows, per_stratum = {}, {}
    for qi, q in enumerate(QS):
        rel_all = predicate_mask(attrs, corpus, q["predicate"])
        rel = np.asarray(rel_all)[idx].astype(bool)
        if rel.sum() == 0 or rel.sum() == len(rel): continue
        qv = QT[qi].unsqueeze(0).expand(N, -1)
        R = {}
        # tautological ceiling: an index that reads the same cached attributes
        R["attribute_index(TAUTOLOGY)"] = rel.astype(float) + \
            np.random.default_rng(qi).random(len(rel)) * 1e-6
        # a-priori margin lookup: the attribute the query's predicate names
        at = q.get("attribute") or next((k for k in q["predicate"] if k in ATTRS), None)
        if at:
            src = strict if (at == "low_contrast" and strict) else vlm
            key = "low_contrast_strict" if (at == "low_contrast" and strict) else at
            R["logit_VLM_lookup"] = np.array(
                [float(src.get(t, {}).get(key+"_margin", -9.)) for t in toks])
        with torch.no_grad():
            parts = M22[0].base_parts(qv, qv, sig, mz, prior).cpu().numpy()
            R["zeroshot_SigLIP2"] = parts[idx, 0]
            R["SceneMiner_v2.2"] = np.mean(
                [m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=on)
                 .cpu().numpy()[idx] for m in M22], 0)
            R["SceneMiner_v2.2_monocular"] = np.mean(
                [m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=off)
                 .cpu().numpy()[idx] for m in M22], 0)
            R["SceneMiner_v2.1"] = np.mean(
                [m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior)
                 .cpu().numpy()[idx] for m in M21], 0)
            # --- ablation 1: no PartNorm (v2.1's incommensurate blend) -------
            ab = []
            for m in M22:
                p_raw = m.base_parts(qv, qv, sig, mz, prior, normalize=False)
                w = m.route(qv, av3)
                g = torch.tanh(m.gate_head(qv)).squeeze(-1)
                r_ = m.residual_score(qv, emb, tlog, risk, qwen, sig, mz, on)
                ab.append(((w * p_raw).sum(-1) + g * r_).cpu().numpy()[idx])
            R["v2.2_ablate_PartNorm"] = np.mean(ab, 0)
            # --- ablation 2: raw (uncentred) text-text channel selector -----
            for m in M22: m.text_mean.zero_()
            R["v2.2_ablate_text_centring"] = np.mean(
                [m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=on)
                 .cpu().numpy()[idx] for m in M22], 0)
            for m, mu in zip(M22, saved_mu): m.text_mean.copy_(mu)
        for nm, s in R.items():
            o = np.argsort(-s)
            v = (ap_binary(o, rel), p_at(o, rel), ndcg_at(o, rel.astype(float)))
            rows.setdefault(nm, []).append(v)
            per_stratum.setdefault(q["stratum"], {}).setdefault(nm, []).append(v[0])

    out = {"n_scenes": len(toks), "n_queries_scored": len(rows["zeroshot_SigLIP2"]),
           "caveat": "relevance is a predicate over the cached VLM attributes, so "
                     "attribute_index is a tautological CEILING (1.000), not a baseline",
           "overall": {}, "per_stratum": {}}
    print(f"\n{'ranker':<34}{'mAP':>8}{'P@5':>8}{'NDCG@10':>9}")
    for nm in sorted(rows, key=lambda k: -np.mean([r[0] for r in rows[k]])):
        A = np.array(rows[nm])
        out["overall"][nm] = {"mAP": float(A[:,0].mean()), "P@5": float(A[:,1].mean()),
                              "NDCG@10": float(A[:,2].mean())}
        print(f"{nm:<34}{A[:,0].mean():>8.4f}{A[:,1].mean():>8.4f}{A[:,2].mean():>9.4f}")
    for st in sorted(per_stratum):
        out["per_stratum"][st] = {k: float(np.mean(v)) for k, v in per_stratum[st].items()}
        print(f"\n  -- {st} (n={len(next(iter(per_stratum[st].values())))}) --")
        for k, v in sorted(per_stratum[st].items(), key=lambda kv: -np.mean(kv[1])):
            print(f"    {k:<36}{np.mean(v):>8.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[suite] wrote {a.out}")


if __name__ == "__main__":
    main()
