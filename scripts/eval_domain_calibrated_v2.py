#!/usr/bin/env python3
"""MILESTONE 11 - negative-tail / quantile-floor domain calibration.

Problem (measured in MILESTONE 7). Retrieved top-k over-represent nuScenes:
SceneMiner v2.1's top-5 is 65.6 % nuScenes against a 56.9 % relevant base rate,
a skew of +8.7 points. The obvious fix - z-score each ranker's scores within
each domain - OVERCORRECTS badly: skew swings to -23.8 points and mAP falls
0.5312 -> 0.4925, a loss of 0.0387, more than ten times the 0.003 budget.

Why it overcorrects. A whole-distribution z-score estimates each domain's
location and scale from ALL of that domain's scores, relevant ones included.
Relevance prevalence differs between the two domains, so the estimate absorbs
the very signal we want to keep: subtracting it removes genuine differences in
how much of each domain is relevant, not just the nuisance scale offset.

Negative-tail calibration. Estimate the domain offset from the NEGATIVE TAIL
only - the scores at or below a quantile floor `q`, which are overwhelmingly
irrelevant scenes and therefore carry the nuisance offset without the relevance
signal - and apply it with a shrinkage factor `alpha`:

    mu_d, sigma_d  from  {s in domain d : s <= Quantile_q(s_d)}
    s' = s - alpha * (s - (s - mu_d) / sigma_d)        [alpha = 1 is a full tail z-score]

PROTOCOL. (q, alpha) are chosen on the 240 TRAIN queries, minimising |skew|
subject to a mAP loss of at most 0.003, and the selected configuration is then
reported once on the 150 held-out EVAL queries. Evaluated SCENES are always from
the 602-scene eval split, so no scored scene contributed a gradient. Fitting on
train queries rather than on the reported eval queries is what keeps the
reported skew an out-of-sample number.
"""
from __future__ import annotations
from paths import DEVICE
import argparse, glob, json, re, sys
import os
SEED_GLOB = os.environ.get("SM_SEED_GLOB", "s*")      # Milestone 34: one seed at a time for +/- sd
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, p_at, ndcg_at, load_attributes, predicate_mask

SEG = re.compile(r"#c\d+$")
MARG_COL = {"droplets": 0, "glare": 1, "wet_road": 2, "low_contrast": 3}


def mmr(order, Vn, toks, k, per_source=1, sim_thr=0.95):
    sel, cnt = [], {}
    for i in order:
        if len(sel) >= k: break
        src = SEG.sub("", toks[i])
        if cnt.get(src, 0) >= per_source: continue
        if sel and float((Vn[i] @ Vn[sel].T).max()) > sim_thr: continue
        sel.append(int(i)); cnt[src] = cnt.get(src, 0) + 1
    for i in order:
        if len(sel) >= k: break
        if int(i) not in sel: sel.append(int(i))
    return sel[:k]


def calibrate(s, dom, q=0.5, alpha=1.0):
    """Negative-tail domain calibration. q=1.0 recovers the whole-distribution
    z-score of MILESTONE 7; alpha=0 is a no-op."""
    if alpha == 0.0: return s.astype(float)
    out = s.astype(float).copy()
    for d in (0, 1):
        m = dom == d
        if m.sum() < 3: continue
        sd = s[m]
        thr = np.quantile(sd, q)
        tail = sd[sd <= thr]
        if tail.size < 3: tail = sd
        mu, sg = float(tail.mean()), float(tail.std())
        zs = (sd - mu) / (sg + 1e-9)
        out[m] = (1.0 - alpha) * sd + alpha * zs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--suite", default="benchmark_cvpr/queries_390.json")
    ap.add_argument("--split", default="benchmark_cvpr/split_906_602.json")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--skew-target", type=float, default=2.0)
    ap.add_argument("--map-budget", type=float, default=0.003)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results/cvpr/milestone11_domain_calibration.json")
    ap.add_argument("--rankers", default="SceneMiner_v2.2,SceneMiner_v2.1,zeroshot_SigLIP2,logit_VLM_lookup")
    a = ap.parse_args()
    dev = torch.device(DEVICE); W = Path(a.work)

    z = np.load(a.features, allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]; cidx = {t: i for i, t in enumerate(corpus)}
    evset = set(json.load(open(a.split))["eval"])
    toks = [t for t in corpus if t in evset]
    idx = np.array([cidx[t] for t in toks])
    dom = z["domain"].astype(int)[idx]
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    emb, tlog, risk = T("emb"), T("tag_logits"), T("risk")
    sig, mzz, prior = T("siglip_img"), T("vlm_margin_z"), T("prior")
    qwen = torch.zeros(len(corpus), 2048, device=dev)
    vc = W/"V_frozen_cache.pt"
    if vc.exists(): qwen = torch.load(vc, map_location=dev).float()
    rk = risk.reshape(-1).cpu().numpy()
    Vn = F.normalize(sig, dim=-1).cpu().numpy()[idx]
    print(f"[cal] {len(toks)} eval-split scenes "
          f"({int((dom==0).sum())} nuScenes / {int((dom==1).sum())} Waymo)", flush=True)

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
    SUITE = json.load(open(a.suite))

    from transformers import AutoModel, GemmaTokenizerFast
    P = "google/siglip2-base-patch16-224"
    sm = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    tk = GemmaTokenizerFast.from_pretrained(P)
    ft = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    QE = {}
    for part in ("train", "eval"):
        with torch.no_grad():
            ti = tk([q["text"] for q in SUITE[part]], padding="max_length",
                    truncation=True, max_length=64, return_tensors="pt").to(dev)
            QE[part] = ft(sm.get_text_features(**ti))
    del sm; torch.cuda.empty_cache()

    from scene_mining.filter.sceneminer_v2 import SceneMinerV21
    from scene_mining.filter.sceneminer_v2_2 import SceneMinerV22
    from scene_mining.filter.sceneminer_v2_3 import SceneMinerV23
    def load(cls, pat):
        out = []
        for c in sorted(glob.glob(pat)):
            d = torch.load(c, map_location=dev, weights_only=False)
            m = cls(d_qwen=qwen.shape[1], d_siglip=sig.shape[1], n_margins=mzz.shape[1],
                    n_meta=6, d_text=QE["eval"].shape[1],
                    risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
            m.load_state_dict(d["state_dict"]); m.eval(); out.append(m)
        return out
    M21 = load(SceneMinerV21, "work_dirs/sceneminer_v21/" + SEED_GLOB + "/final.pt")
    M22 = load(SceneMinerV22, "work_dirs/sceneminer_v22/" + SEED_GLOB + "/final.pt")
    M23 = load(SceneMinerV23, "work_dirs/sceneminer_v23/" + SEED_GLOB + "/final.pt")
    print(f"[cal] {len(M21)} v2.1 seeds, {len(M22)} v2.2 seeds", flush=True)

    on = torch.ones(len(corpus), device=dev)

    def score_all(part):
        """-> {ranker: [(scores, rel) per query]}"""
        out = {}
        rng = np.random.default_rng(0)
        for qi, q in enumerate(SUITE[part]):
            rel = predicate_mask(attrs, toks, q["predicate"])
            if rel.sum() == 0 or rel.sum() == len(toks): continue
            qv = QE[part][qi].unsqueeze(0).expand(len(corpus), -1)
            R = {}
            with torch.no_grad():
                R["zeroshot_SigLIP2"] = F.cosine_similarity(
                    F.normalize(QE[part][qi].unsqueeze(0), -1),
                    F.normalize(sig, -1)).cpu().numpy()[idx]
                R["SceneMiner_v2.1"] = np.mean(
                    [m(qv, qv, emb, tlog, risk, qwen, sig, mzz, prior)
                     .cpu().numpy()[idx] for m in M21], 0)
                R["SceneMiner_v2.3"] = np.mean(
                    [m(qv, qv, emb, tlog, risk, qwen, sig, mzz, prior, has_3d=on)
                     .cpu().numpy()[idx] for m in M23], 0)
                R["SceneMiner_v2.2"] = np.mean(
                    [m(qv, qv, emb, tlog, risk, qwen, sig, mzz, prior, has_3d=on)
                     .cpu().numpy()[idx] for m in M22], 0)
            at = next((k for k in MARG_COL if k in q["predicate"]), None)
            if at is not None:
                src = strict if (at == "low_contrast" and strict) else vlm
                key = "low_contrast_strict" if (at == "low_contrast" and strict) else at
                R["logit_VLM_lookup"] = np.array(
                    [float(src.get(t, {}).get(key+"_margin", -9.)) for t in toks])
            for nm, s in R.items():
                out.setdefault(nm, []).append((s, rel))
        return out

    TR, EV = score_all("train"), score_all("eval")
    print(f"[cal] {len(next(iter(TR.values())))} train / "
          f"{len(next(iter(EV.values())))} eval queries scored", flush=True)

    def measure(pairs, q, alpha):
        aps, skews = [], []
        for s, rel in pairs:
            sc = calibrate(s, dom, q, alpha)
            o = np.argsort(-sc)
            aps.append(ap_binary(o, rel))
            sel = mmr(o, Vn, toks, a.topk)
            skews.append(float(np.mean(dom[sel] == 0)) - float(np.mean(dom[rel] == 0)))
        return float(np.mean(aps)), float(np.mean(skews) * 100)

    GRID_Q = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]
    GRID_A = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
    out = {"n_eval_scenes": len(toks), "topk": a.topk,
           "skew_target_pts": a.skew_target, "map_budget": a.map_budget,
           "protocol": "(q, alpha) selected on the 240 TRAIN queries, reported "
                       "once on the 150 held-out EVAL queries; scored scenes are "
                       "always from the 602-scene eval split",
           "rankers": {}}
    for nm in a.rankers.split(","):
        if nm not in TR: continue
        base_tr = measure(TR[nm], 1.0, 0.0)
        base_ev = measure(EV[nm], 1.0, 0.0)
        # full z-score, the MILESTONE 7 method, for reference
        m7_ev = measure(EV[nm], 1.0, 1.0)
        feas, allc = [], []
        for q in GRID_Q:
            for al in GRID_A:
                if al == 0.0 and q != GRID_Q[0]: continue
                ap_, sk = measure(TR[nm], q, al)
                rec = {"q": q, "alpha": al, "train_mAP": ap_, "train_skew": sk,
                       "train_dmAP": ap_ - base_tr[0]}
                allc.append(rec)
                if ap_ - base_tr[0] >= -a.map_budget: feas.append(rec)
        pick = min(feas or allc, key=lambda r: abs(r["train_skew"]))
        ev_ap, ev_sk = measure(EV[nm], pick["q"], pick["alpha"])
        # bootstrap CI over queries for the selected configuration
        rng = np.random.default_rng(0)
        pe = EV[nm]; n = len(pe)
        bs_ap, bs_sk, bs_d = [], [], []
        raw_ap = [ap_binary(np.argsort(-s), rel) for s, rel in pe]
        cal_ap = [ap_binary(np.argsort(-calibrate(s, dom, pick["q"], pick["alpha"])), rel)
                  for s, rel in pe]
        raw_sk = [float(np.mean(dom[mmr(np.argsort(-s), Vn, toks, a.topk)] == 0))
                  - float(np.mean(dom[rel] == 0)) for s, rel in pe]
        cal_sk = [float(np.mean(dom[mmr(np.argsort(-calibrate(s, dom, pick["q"], pick["alpha"])),
                                        Vn, toks, a.topk)] == 0))
                  - float(np.mean(dom[rel] == 0)) for s, rel in pe]
        raw_ap, cal_ap = np.array(raw_ap), np.array(cal_ap)
        raw_sk, cal_sk = np.array(raw_sk) * 100, np.array(cal_sk) * 100
        bs_rsk = []
        for _ in range(a.boot):
            b = rng.integers(0, n, n)
            bs_ap.append(cal_ap[b].mean()); bs_sk.append(cal_sk[b].mean())
            bs_d.append(cal_ap[b].mean() - raw_ap[b].mean())
            bs_rsk.append(raw_sk[b].mean())
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        # paired permutation test on the mAP change
        obs = cal_ap.mean() - raw_ap.mean()
        cnt = 0
        for _ in range(a.boot):
            fl = rng.integers(0, 2, n).astype(bool)
            d_ = np.where(fl, -(cal_ap - raw_ap), cal_ap - raw_ap)
            if abs(d_.mean()) >= abs(obs): cnt += 1
        out["rankers"][nm] = {
            "raw": {"mAP": base_ev[0], "skew_pts": base_ev[1],
                    "skew_CI95": [float(np.percentile(bs_rsk, 2.5)),
                                  float(np.percentile(bs_rsk, 97.5))],
                    "skew_within_target": bool(abs(base_ev[1]) <= a.skew_target)},
            "milestone7_full_zscore": {"mAP": m7_ev[0], "skew_pts": m7_ev[1],
                                       "dmAP": m7_ev[0] - base_ev[0]},
            "selected": {"q": pick["q"], "alpha": pick["alpha"],
                         "train_skew_pts": pick["train_skew"],
                         "train_dmAP": pick["train_dmAP"]},
            "eval": {"mAP": ev_ap, "skew_pts": ev_sk, "dmAP": ev_ap - base_ev[0],
                     "mAP_CI95": ci(bs_ap), "skew_CI95": ci(bs_sk),
                     "dmAP_CI95": ci(bs_d),
                     "perm_p_dmAP": (cnt + 1) / (a.boot + 1)},
            "targets_met": {"skew_within_target": bool(abs(ev_sk) <= a.skew_target),
                            "within_map_budget": bool(ev_ap - base_ev[0] >= -a.map_budget)},
            "best_option": ("no_calibration" if abs(base_ev[1]) <= abs(ev_sk)
                            else "negative_tail"),
            "skew_target_resolvable": bool(
                (np.percentile(bs_rsk, 97.5) - np.percentile(bs_rsk, 2.5))
                <= 2 * a.skew_target),
        }
        r = out["rankers"][nm]
        print(f"\n=== {nm} ===")
        rci = [float(np.percentile(bs_rsk, 2.5)), float(np.percentile(bs_rsk, 97.5))]
        print(f"  raw                       mAP {base_ev[0]:.4f}  skew {base_ev[1]:+.1f} pts  "
              f"(95% CI {rci[0]:+.1f}..{rci[1]:+.1f}, width {rci[1]-rci[0]:.1f} pts)")
        print(f"  full z-score (M7)         mAP {m7_ev[0]:.4f}  skew {m7_ev[1]:+.1f} pts  "
              f"dmAP {m7_ev[0]-base_ev[0]:+.4f}")
        print(f"  negative-tail q={pick['q']} a={pick['alpha']}  "
              f"mAP {ev_ap:.4f}  skew {ev_sk:+.1f} pts  dmAP {ev_ap-base_ev[0]:+.4f}")
        print(f"    95% CI  mAP {ci(bs_ap)[0]:.4f}..{ci(bs_ap)[1]:.4f}   "
              f"skew {ci(bs_sk)[0]:+.1f}..{ci(bs_sk)[1]:+.1f}   "
              f"dmAP {ci(bs_d)[0]:+.4f}..{ci(bs_d)[1]:+.4f}  p={r['eval']['perm_p_dmAP']:.3f}")
        print(f"    better option: {r['best_option']};  "
              f"target resolvable at this n: "
              f"{'yes' if r['skew_target_resolvable'] else 'NO - CI wider than the target band'}")
        print(f"    skew |{abs(ev_sk):.1f}| <= {a.skew_target}: "
              f"{'MET' if abs(ev_sk)<=a.skew_target else 'NOT MET'};  "
              f"dmAP >= -{a.map_budget}: "
              f"{'MET' if ev_ap-base_ev[0]>=-a.map_budget else 'NOT MET'}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[cal] wrote {a.out}")


if __name__ == "__main__":
    main()
