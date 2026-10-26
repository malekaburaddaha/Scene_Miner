#!/usr/bin/env python3
"""MILESTONE 14 - SceneMiner v2.3 on all three benchmarks, with ablations.

(A) 550 human-labelled scenes: zero-leakage, monocular, 5-fold human-calibrated
    (routing weights only, refit on 4/5 and scored on the held-out 1/5), and
    VLM false-negative recovery. Same protocol as Milestones 6 and 10.
(B) BDD100K, 1000 images, 12 queries: portable base score (no 3D rig, meta
    channel declared absent), margin columns permuted into TRAINING order,
    no tuning of any kind. Broken down into the 4 queries that name one of the
    four cached attributes (in-vocabulary) and the 8 that do not - a split
    fixed from the query wording, not from gamma.
(C) 150-query suite on the 602 eval-split scenes, per stratum, plus the top-5
    nuScenes domain skew.

Every ablation flips one inference-time switch of the trained v2.3 seeds, so a
row differs from the full model in exactly one mechanism:
  no_gate     - concept gate off (margin weight not scaled by gamma)
  no_rescue   - asymmetric FN rescue off
  no_resnorm  - raw residual, as in v2.2
  base_only   - the routed base score with no residual at all
  abstain     - POST-HOC (added in Milestone 15 after the semantic results):
                interpolate toward standardised SigLIP2 by (1 - gamma)
"""
from __future__ import annotations
from paths import EXT, HUMAN_R1
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, p_at, ndcg_at, load_attributes, predicate_mask
from eval_human_gt_full import load_human, ATTRS, QUERY

dev = torch.device("cuda")
SEG = re.compile(r"#c\d+$")
ABL = ["full", "abstain", "no_gate", "no_rescue", "no_resnorm", "base_only"]
DEFAULT_ABSTAIN = False          # v2.4: abstention is part of the model (set in main)
BDD_INVOCAB = {"wet reflective asphalt after rainfall": "wet_road",
               "water droplets smearing the camera lens": "droplets",
               "a low contrast scene where vehicles are barely distinguishable from the background": "low_contrast",
               "blinding headlight bloom from oncoming traffic": "glare"}


def set_mode(models, mode):
    for m in models:
        m.use_gate = mode != "no_gate"
        m.use_rescue = mode != "no_rescue"
        m.use_resnorm = mode != "no_resnorm"
        m.use_abstain = (mode == "abstain") or (DEFAULT_ABSTAIN and mode != "no_abstain")
        if hasattr(m, "use_macro_prior"): m.use_macro_prior = mode != "no_macro_prior"


def score(models, mode, qv, F_, has_3d):
    """seed-stacked scores (S, N) for one ablation mode."""
    set_mode(models, mode)
    out = []
    with torch.no_grad():
        for m in models:
            s, base, res, w, parts, gate = m(qv, qv, *F_, has_3d=has_3d, return_parts=True)
            out.append((base if mode == "base_only" else s).cpu().numpy())
    set_mode(models, "full")
    return np.stack(out)


def text_embed(texts):
    from transformers import AutoModel, GemmaTokenizerFast
    P = "google/siglip2-base-patch16-224"
    sm = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    tk = GemmaTokenizerFast.from_pretrained(P)
    ft = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    with torch.no_grad():
        ti = tk(texts, padding="max_length", truncation=True, max_length=64,
                return_tensors="pt").to(dev)
        E = ft(sm.get_text_features(**ti))
    del sm; torch.cuda.empty_cache()
    return E


def load_models(pat, d_qwen, d_sig, n_m, d_text, rk, cls_name="v23"):
    from scene_mining.filter.sceneminer_v2_3 import SceneMinerV23
    if cls_name == "v24":
        from scene_mining.filter.sceneminer_v2_4 import SceneMinerV24 as SceneMinerV23
    if cls_name == "v25":
        from scene_mining.filter.sceneminer_v2_5 import SceneMinerV25 as SceneMinerV23
    if cls_name == "v26":
        from scene_mining.filter.sceneminer_v2_6 import SceneMinerV26 as SceneMinerV23
    M = []
    for c in sorted(glob.glob(pat)):
        d = torch.load(c, map_location=dev, weights_only=False)
        m = SceneMinerV23(d_qwen=d_qwen, d_siglip=d_sig, n_margins=n_m, n_meta=6,
                          d_text=d_text, risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
        m.load_state_dict(d["state_dict"]); m.eval(); M.append(m)
    return M


def fn_recovery(s, H, V):
    miss = H & ~V
    if miss.sum() == 0: return None
    o = np.argsort(-s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    return {"n_missed": int(miss.sum()), "n_in_top20pct": int((r[miss] < 0.2*len(s)).sum()),
            "median_rank_pct": float(np.median(r[miss]) / len(s))}


# --------------------------------------------------------------------- (A)
def bench_human(a, Z, M, QT):
    corpus, cidx, F_ = Z["corpus"], Z["cidx"], Z["F"]
    N = len(corpus); on = torch.ones(N, device=dev); off = torch.zeros(N, device=dev)
    W = Path(a.work)
    vlm = json.load(open(W/"binary_flags_with_logits.json"))
    strict = json.load(open(W/"lowcontrast_strict.json")) if (W/"lowcontrast_strict.json").exists() else {}
    hum = load_human(a.human, "B")
    toks = [t for t in corpus if t in hum]; idx = np.array([cidx[t] for t in toks])
    out = {"n_scenes": len(toks), "attributes": {}}
    for ai, at in enumerate(ATTRS):
        H = np.array([hum[t][at] for t in toks])
        src = strict if (at == "low_contrast" and strict) else vlm
        key = "low_contrast_strict" if (at == "low_contrast" and strict) else at
        V = np.array([bool(src.get(t, {}).get(key, False)) for t in toks])
        qv = QT[ai].unsqueeze(0).expand(N, -1)
        R, rec = {}, {}
        look = np.array([float(src.get(t, {}).get(key+"_margin", -9.)) for t in toks])
        with torch.no_grad():
            parts = torch.stack([m.base_parts(qv, qv, F_[4], F_[5], F_[6]) for m in M]).cpu().numpy()[:, idx]
            g = torch.stack([torch.tanh(m.gate_head(qv)).squeeze(-1) for m in M]).cpu().numpy()[:, idx]
            res = np.stack([(m.forward(qv, qv, *F_, has_3d=on, return_parts=True)[2]).cpu().numpy()
                            for m in M])[:, idx]
            route = np.stack([m.route(qv, torch.ones(N, 3, device=dev)).mean(0).cpu().numpy() for m in M])
            _, gam, _ = M[0].concept(QT[ai:ai+1])
        R["logit_VLM_lookup"] = {"mAP": ap_binary(np.argsort(-look), H)}
        cos = F.cosine_similarity(F.normalize(qv, -1), F.normalize(F_[4], -1)).cpu().numpy()[idx]
        R["zeroshot_SigLIP2"] = {"mAP": ap_binary(np.argsort(-cos), H)}
        rec["logit_VLM_lookup"] = fn_recovery(look, H, V)
        rec["zeroshot_SigLIP2"] = fn_recovery(cos, H, V)
        for mode in ABL:
            S = score(M, mode, qv, F_, on)[:, idx]
            aps = [ap_binary(np.argsort(-S[k]), H) for k in range(len(M))]
            R[f"v2.3_{mode}"] = {"mAP": float(np.mean(aps)), "mAP_std": float(np.std(aps))}
            rec[f"v2.3_{mode}"] = fn_recovery(S.mean(0), H, V)
            rec[f"v2.3_{mode}_seed0"] = fn_recovery(S[0], H, V)
        S = score(M, "full", qv, F_, off)[:, idx]
        R["v2.3_monocular"] = {"mAP": float(np.mean([ap_binary(np.argsort(-S[k]), H) for k in range(len(M))]))}
        # 5-fold calibration of routing weights only (Milestone 6 protocol)
        P3 = parts.mean(0); gr = (g * res).mean(0)
        order = np.random.default_rng(0).permutation(len(toks)); folds = np.array_split(order, 5)
        cal = []
        for f in range(5):
            te = folds[f]; tr = np.concatenate([folds[j] for j in range(5) if j != f])
            best, bw = -1, None
            for w0 in np.linspace(0, 1, 11):
                for w1 in np.linspace(0, 1-w0, int(round((1-w0)*10))+1):
                    w2 = 1-w0-w1
                    st = P3 @ np.array([w0, w1, w2]) + gr
                    v = ap_binary(np.argsort(-st[tr]), H[tr])
                    if v > best: best, bw = v, (w0, w1, w2)
            st = P3 @ np.array(bw) + gr
            cal.append(ap_binary(np.argsort(-st[te]), H[te]))
        R["v2.3_human_calibrated_5fold"] = {"mAP": float(np.mean(cal)), "mAP_std": float(np.std(cal))}
        out["attributes"][at] = {"n_human_pos": int(H.sum()), "gamma": float(gam),
                                 "routing_mean": route.mean(0).tolist(),
                                 "rankers": R, "vlm_fn_recovery": rec}
        print(f"  [A] {at:<13} gamma {float(gam):.3f}  full {R['v2.3_full']['mAP']:.4f}  "
              f"cal {R['v2.3_human_calibrated_5fold']['mAP']:.4f}  base {R['v2.3_base_only']['mAP']:.4f}  "
              f"FN {rec['v2.3_full']['n_in_top20pct'] if rec['v2.3_full'] else '-'}"
              f"/{rec['v2.3_full']['n_missed'] if rec['v2.3_full'] else '-'}", flush=True)
    keys = out["attributes"]["droplets"]["rankers"].keys()
    out["macro_mAP"] = {k: float(np.mean([out["attributes"][at]["rankers"][k]["mAP"] for at in ATTRS]))
                        for k in keys}
    return out


# --------------------------------------------------------------------- (B)
def bench_bdd(a, M):
    from eval_external_bdd100k import EVAL_Q, QUESTIONS, CHANNEL
    root = Path(a.bdd)
    labels = json.load(open(root/"labels.json")); names = sorted(labels)
    z = np.load(root/"signals.npz", allow_pickle=False)
    TRAIN_ORDER = ["droplets", "glare", "wet_road", "low_contrast"]
    ASKED = [k for k, _ in QUESTIONS]
    MARG = z["margin"][:, [ASKED.index(k) for k in TRAIN_ORDER]]
    MZ = (MARG - MARG.mean(0, keepdims=True)) / (MARG.std(0, keepdims=True) + 1e-6)
    S = torch.from_numpy(z["siglip"]).float().to(dev); Mz = torch.from_numpy(MZ).float().to(dev)
    N = len(names); PR = torch.zeros(N, 26, device=dev)
    av2 = torch.stack([torch.ones(N, device=dev), torch.ones(N, device=dev), torch.zeros(N, device=dev)], -1)
    # raw text features, as in training and in benches (A)/(C); Milestone 23 found this
    # line L2-normalised them, which changed the scale seen by the learned selectors
    QT = text_embed([q for q, _ in EVAL_Q]); QN = F.normalize(QT, dim=-1)
    rows, per_q = {}, []
    for qi, (qt, (field, pos)) in enumerate(EVAL_Q):
        rel = np.array([labels[n][field] in pos for n in names])
        if rel.sum() in (0, N): continue
        qv = QT[qi].unsqueeze(0).expand(N, -1)
        cos = (S @ QN[qi]).cpu().numpy()
        att, sgn = CHANNEL[qt]; j = TRAIN_ORDER.index(att)
        R = {"zeroshot_SigLIP2": cos, "logit_VLM_lookup": sgn * MZ[:, j]}
        with torch.no_grad():
            for mode in (["full", "no_abstain", "no_gate"] if DEFAULT_ABSTAIN else ["full", "abstain", "no_gate", "no_rescue"]):
                set_mode(M, mode)
                R[f"v2.3_{mode}"] = np.mean([m.base_score(qv, S, Mz, PR, av2).cpu().numpy()
                                             for m in M], 0)
            set_mode(M, "full")
            a_, gam, c = M[0].concept(QT[qi:qi+1])
            w = np.mean([m.route(qv[:1], av2[:1])[0].cpu().numpy() for m in M], 0)
        cat = "in_vocab" if qt in BDD_INVOCAB else "out_of_vocab"
        per_q.append({"query": qt, "category": cat, "gamma": float(gam),
                      "concept_argmax": TRAIN_ORDER[int(a_.argmax())],
                      "concept_pmax": float(a_.max()),
                      "apriori_channel": att, "routing": w.tolist(),
                      "mAP": {k: ap_binary(np.argsort(-v), rel) for k, v in R.items()}})
        for k, v in R.items():
            o = np.argsort(-v)
            rows.setdefault(k, []).append((ap_binary(o, rel), p_at(o, rel), ndcg_at(o, rel.astype(float)), cat))
    out = {"n_images": N, "n_queries": len(per_q), "per_query": per_q, "rankers": {}}
    for k, L in rows.items():
        A = np.array([r[:3] for r in L], float); cats = np.array([r[3] for r in L])
        out["rankers"][k] = {"mAP": float(A[:, 0].mean()), "P@5": float(A[:, 1].mean()),
                             "NDCG@10": float(A[:, 2].mean()),
                             "mAP_in_vocab": float(A[cats == "in_vocab", 0].mean()),
                             "mAP_out_of_vocab": float(A[cats == "out_of_vocab", 0].mean())}
    for q in per_q:
        print(f"  [B] {q['category'][:6]:<7} gamma {q['gamma']:.3f} {q['concept_argmax']:<13}"
              f"(a-priori {q['apriori_channel']:<12}) v2.3 {q['mAP']['v2.3_full']:.3f} "
              f"SigLIP {q['mAP']['zeroshot_SigLIP2']:.3f}  {q['query'][:44]}", flush=True)
    return out


# --------------------------------------------------------------------- (C)
def bench_suite(a, Z, M):
    from eval_domain_calibrated_v2 import mmr
    corpus, F_ = Z["corpus"], Z["F"]
    N = len(corpus); on = torch.ones(N, device=dev)
    ev = set(json.load(open(a.split))["eval"])
    idx = np.array([i for i, t in enumerate(corpus) if t in ev]); toks = [corpus[i] for i in idx]
    dom = Z["dom"][idx]; Vn = F.normalize(F_[4], dim=-1).cpu().numpy()[idx]
    W = Path(a.work); attrs = load_attributes(W)
    for f in ("binary_flags_with_logits.json", "lowcontrast_strict.json"):
        p = W/f
        if p.exists():
            for t, r in json.load(open(p)).items():
                rec = {k: v for k, v in r.items() if not k.endswith(("_margin", "_logp_yes"))}
                if "low_contrast_strict" in rec: rec["low_contrast"] = rec.pop("low_contrast_strict")
                attrs.setdefault(t, {}).update(rec)
    QS = json.load(open(a.suite))["eval"]
    QT = text_embed([q["text"] for q in QS])
    rows, strat, skew = {}, {}, {}
    gam_by = {}
    for qi, q in enumerate(QS):
        rel = predicate_mask(attrs, toks, q["predicate"])
        if rel.sum() in (0, len(rel)): continue
        qv = QT[qi].unsqueeze(0).expand(N, -1)
        R = {"zeroshot_SigLIP2": F.cosine_similarity(F.normalize(qv, -1), F.normalize(F_[4], -1)).cpu().numpy()[idx]}
        for mode in ABL:
            R[f"v2.3_{mode}"] = score(M, mode, qv, F_, on).mean(0)[idx]
        with torch.no_grad():
            _, gam, _ = M[0].concept(QT[qi:qi+1])
        gam_by.setdefault(q["stratum"], []).append(float(gam))
        for k, s in R.items():
            o = np.argsort(-s)
            rows.setdefault(k, []).append((ap_binary(o, rel), p_at(o, rel), ndcg_at(o, rel.astype(float))))
            strat.setdefault(q["stratum"], {}).setdefault(k, []).append(ap_binary(o, rel))
            sel = mmr(o, Vn, toks, 5)
            skew.setdefault(k, []).append(float(np.mean(dom[sel] == 0)) - float(np.mean(dom[rel] == 0)))
    out = {"n_scenes": len(toks), "overall": {}, "per_stratum": {}, "skew_pts": {},
           "gamma_by_stratum": {k: float(np.mean(v)) for k, v in gam_by.items()}}
    rng = np.random.default_rng(0)
    for k, L in rows.items():
        A = np.array(L); sk = np.array(skew[k]) * 100
        bs = [sk[rng.integers(0, len(sk), len(sk))].mean() for _ in range(1000)]
        out["overall"][k] = {"mAP": float(A[:, 0].mean()), "P@5": float(A[:, 1].mean()),
                             "NDCG@10": float(A[:, 2].mean())}
        out["skew_pts"][k] = {"mean": float(sk.mean()),
                              "CI95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    for st, d in strat.items():
        out["per_stratum"][st] = {k: float(np.mean(v)) for k, v in d.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="all", choices=["all", "human", "bdd", "suite"])
    ap.add_argument("--ckpts", default="work_dirs/sceneminer_v23/s*/final.pt")
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--human", default=HUMAN_R1)
    ap.add_argument("--bdd", default=EXT + "/bdd100k")
    ap.add_argument("--suite", default="benchmark_cvpr/queries_390.json")
    ap.add_argument("--split", default="benchmark_cvpr/split_906_602.json")
    ap.add_argument("--out", default="results/cvpr/milestone14_v2_3.json")
    ap.add_argument("--model", default="v23", choices=["v23", "v24", "v25", "v26"])
    a = ap.parse_args()
    z = np.load(a.features, allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    qwen = torch.zeros(len(corpus), 2048, device=dev)
    vc = Path(a.work)/"V_frozen_cache.pt"
    if vc.exists(): qwen = torch.load(vc, map_location=dev).float()
    risk = T("risk"); rk = risk.reshape(-1).cpu().numpy()
    Z = {"corpus": corpus, "cidx": {t: i for i, t in enumerate(corpus)},
         "dom": z["domain"].astype(int),
         "F": (T("emb"), T("tag_logits"), risk, qwen, T("siglip_img"), T("vlm_margin_z"), T("prior"))}
    assert list(z["margin_attrs"]) == ATTRS
    global ABL, DEFAULT_ABSTAIN
    if a.model in ("v24", "v25", "v26"):
        DEFAULT_ABSTAIN = True
        ABL = ["full", "no_abstain", "no_gate", "no_rescue", "no_resnorm", "base_only"]
        if a.model in ("v25", "v26"): ABL.insert(2, "no_macro_prior")
    M = load_models(a.ckpts, qwen.shape[1], 768, 4, 768, rk, a.model)
    set_mode(M, "full")
    print(f"[v2.3] {len(M)} seeds", flush=True)
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    if a.bench in ("all", "human"):
        out["A_human_gt"] = bench_human(a, Z, M, text_embed([QUERY[x] for x in ATTRS]))
    if a.bench in ("all", "bdd"):
        out["B_bdd100k"] = bench_bdd(a, M)
    if a.bench in ("all", "suite"):
        out["C_suite150"] = bench_suite(a, Z, M)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[v2.3] wrote {a.out}")


if __name__ == "__main__":
    main()
