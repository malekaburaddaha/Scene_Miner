#!/usr/bin/env python3
"""MILESTONE 10 - SceneMiner v2.2 vs baselines on human GT, plus 5-fold human
calibration of the query-routing weights.

Identical protocol to the v2.1 evaluation (MILESTONE 6) so the two are directly
comparable; only the checkpoint family changes. v2.2 additionally reports the
MONOCULAR score (has_3d_rig=False), which is the quantity that actually
transfers to a camera-only corpus such as BDD100K.

Two modes:
  zero-leakage : routing+residual trained only on the 906 train split
  calibrated   : 5-fold CV over the human-labelled eval scenes. In each fold the
                 ROUTING WEIGHTS ONLY are refit on 4/5 of the human scenes and
                 evaluated strictly on the held-out 1/5. The residual and all
                 adapters stay frozen, so the only thing the human labels buy is
                 "which modality should this query trust".
"""
from __future__ import annotations
import argparse, glob, json, sys
import os
SEED_GLOB = os.environ.get("SM_SEED_GLOB", "s*")      # Milestone 34: one seed at a time for +/- sd
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, p_at, ndcg_at, load_attributes
from eval_human_gt_full import load_human, ATTRS, QUERY

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="results/day1_gate/malek_repaired.json")
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results/cvpr/milestone10_sceneminer_v2_2.json")
    a = ap.parse_args()
    dev = torch.device("cuda"); W = Path(a.work)
    z = np.load(a.features, allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]; cidx = {t:i for i,t in enumerate(corpus)}
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    emb, tlog, risk = T("emb"), T("tag_logits"), T("risk")
    sig, mz, prior = T("siglip_img"), T("vlm_margin_z"), T("prior")
    qwen = torch.zeros(len(corpus), 2048, device=dev)
    vc = W/"V_frozen_cache.pt"
    if vc.exists(): qwen = torch.load(vc, map_location=dev).float()
    rk = risk.reshape(-1).cpu().numpy()

    from transformers import AutoModel, GemmaTokenizerFast
    P = "google/siglip2-base-patch16-224"
    sm_ = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    tk = GemmaTokenizerFast.from_pretrained(P)
    feat = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    qs = [QUERY[x] for x in ATTRS]
    with torch.no_grad():
        ti = tk(qs, padding="max_length", truncation=True, max_length=64,
                return_tensors="pt").to(dev)
        QT = feat(sm_.get_text_features(**ti))
    del sm_; torch.cuda.empty_cache()

    from scene_mining.filter.sceneminer_v2_2 import SceneMinerV22
    models = []
    for ck in sorted(glob.glob("work_dirs/sceneminer_v22/" + SEED_GLOB + "/final.pt")):
        d = torch.load(ck, map_location=dev, weights_only=False)
        m = SceneMinerV22(d_qwen=qwen.shape[1], d_siglip=sig.shape[1],
                          n_margins=mz.shape[1], n_meta=6, d_text=QT.shape[1],
                          risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
        m.load_state_dict(d["state_dict"]); m.eval(); models.append(m)
    print(f"[v2.2] {len(models)} seeds", flush=True)

    # cache the 3 base parts + residual per query (frozen pieces)
    cache = {}
    with torch.no_grad():
        for qi, q in enumerate(qs):
            qq = QT[qi].unsqueeze(0).expand(len(corpus), -1)
            N = len(corpus)
            on, off = torch.ones(N, device=dev), torch.zeros(N, device=dev)
            av3 = torch.ones(N, 3, device=dev)
            av2 = torch.stack([on, on, off], -1)        # monocular: no meta prior
            parts = models[0].base_parts(qq, qq, sig, mz, prior)          # (N,3)
            res = torch.stack([m.residual_score(qq, emb, tlog, risk, qwen, sig, mz, on)
                               for m in models], 0)                        # (S,N)
            gates = torch.stack([torch.tanh(m.gate_head(qq)).squeeze(-1) for m in models], 0)
            full = torch.stack([m(qq, qq, emb, tlog, risk, qwen, sig, mz, prior, has_3d=on)
                                for m in models], 0)                       # (S,N)
            mono = torch.stack([m(qq, qq, emb, tlog, risk, qwen, sig, mz, prior, has_3d=off)
                                for m in models], 0)                       # (S,N)
            route = torch.stack([m.route(qq, av3).mean(0) for m in models], 0)   # (S,3)
            route_m = torch.stack([m.route(qq, av2).mean(0) for m in models], 0)
            cache[q] = {"parts": parts.cpu().numpy(), "res": res.cpu().numpy(),
                        "gate": gates.cpu().numpy(), "full": full.cpu().numpy(),
                        "mono": mono.cpu().numpy(),
                        "route": route.cpu().numpy(), "route_mono": route_m.cpu().numpy()}

    vlm = json.load(open(W/"binary_flags_with_logits.json"))
    strict = json.load(open(W/"lowcontrast_strict.json")) if (W/"lowcontrast_strict.json").exists() else {}
    res_out = {}
    for mode in ["A", "B"]:
        hum = load_human(a.human, mode)
        toks = [t for t in corpus if t in hum]; idx = [cidx[t] for t in toks]
        print(f"\n===== MODE {mode}: {len(toks)} scenes =====")
        mres = {"n_scenes": len(toks), "attributes": {}}
        for at in ATTRS:
            H = np.array([hum[t][at] for t in toks])
            if H.sum() == 0 or H.sum() == len(toks): continue
            q = QUERY[at]; C = cache[q]
            src = strict if (at=="low_contrast" and strict) else vlm
            key = "low_contrast_strict" if (at=="low_contrast" and strict) else at
            rank = {}
            def add(nm, s):
                o = np.argsort(-s)
                rank[nm] = {"mAP": ap_binary(o,H), "P@5": p_at(o,H),
                            "NDCG@10": ndcg_at(o,H.astype(float))}
            add("logit_VLM_lookup", np.array([float(src.get(t,{}).get(key+"_margin",-9.)) for t in toks]))
            add("zeroshot_SigLIP2", C["parts"][idx,0])
            add("metadata_prior", C["parts"][idx,2])
            seeds = [ap_binary(np.argsort(-C["full"][s][idx]), H) for s in range(len(models))]
            rank["SceneMiner_v2.2_zeroleak"] = {"mAP": float(np.mean(seeds)),
                                                "mAP_std": float(np.std(seeds)),
                                                "n_seeds": len(seeds)}
            mseeds = [ap_binary(np.argsort(-C["mono"][s_][idx]), H)
                      for s_ in range(len(models))]
            rank["SceneMiner_v2.2_monocular"] = {"mAP": float(np.mean(mseeds)),
                                                 "mAP_std": float(np.std(mseeds)),
                                                 "n_seeds": len(models),
                                                 "note": "has_3d_rig=False, routing renormalised over 2 channels"}
            # ---------- 5-fold human calibration of the ROUTING ONLY ----------
            rng = np.random.default_rng(0); order = rng.permutation(len(toks))
            folds = np.array_split(order, a.folds)
            cal_ap = []
            for f in range(a.folds):
                te = folds[f]; tr = np.concatenate([folds[j] for j in range(a.folds) if j!=f])
                P3 = C["parts"][idx]                       # (n,3)
                resid = C["res"][:, idx].mean(0)           # seed-mean residual
                g = C["gate"][:, idx].mean(0)
                # grid-search a simplex weight on the TRAIN fold only
                best, bw = -1, None
                for w0 in np.linspace(0,1,11):
                    for w1 in np.linspace(0,1-w0,int((1-w0)*10)+1):
                        w2 = 1-w0-w1
                        s_tr = (P3[:,0]*w0 + P3[:,1]*w1 + P3[:,2]*w2) + g*resid
                        v = ap_binary(np.argsort(-s_tr[tr]), H[tr])
                        if v > best: best, bw = v, (w0,w1,w2)
                s_te = (P3[:,0]*bw[0] + P3[:,1]*bw[1] + P3[:,2]*bw[2]) + g*resid
                cal_ap.append(ap_binary(np.argsort(-s_te[te]), H[te]))
            rank["SceneMiner_v2.2_human_calibrated_5fold"] = {
                "mAP": float(np.mean(cal_ap)), "mAP_std": float(np.std(cal_ap)),
                "folds": a.folds, "note": "routing refit on 4/5 human scenes, scored on held-out 1/5"}
            # VLM false-negative recovery
            Vb = np.array([bool(src.get(t,{}).get(key,False)) for t in toks])
            miss = H & ~Vb; recov = {}
            if miss.sum() > 0:
                for nm, s in [("SceneMiner_v2.1", C["full"][0][idx]),
                              ("zeroshot_SigLIP2", C["parts"][idx,0]),
                              ("logit_VLM_lookup", np.array([float(src.get(t,{}).get(key+"_margin",-9.)) for t in toks]))]:
                    o = np.argsort(-s); r = np.empty(len(toks)); r[o] = np.arange(len(toks))
                    recov[nm] = {"n_missed": int(miss.sum()),
                                 "n_in_top20pct": int((r[miss] < 0.2*len(toks)).sum()),
                                 "median_rank_pct": float(np.median(r[miss])/len(toks))}
            mres["attributes"][at] = {"n_human_pos": int(H.sum()), "chance": float(H.mean()),
                                      "rankers": rank, "vlm_fn_recovery": recov,
                                      "mean_routing_seed0": C["route"][0].tolist(),
                                      "mean_routing_all_seeds": C["route"].mean(0).tolist(),
                                      "mean_routing_monocular": C["route_mono"].mean(0).tolist()}
            print(f"\n  -- {at} (human+ {int(H.sum())}/{len(toks)}) route(siglip,margin,meta)="
                  f"{np.round(C['route'][0],3)} --")
            for k,v in sorted(rank.items(), key=lambda kv:-kv[1]["mAP"]):
                print(f"    {k:<42}{v['mAP']:>8.4f}")
        res_out[f"mode_{mode}"] = mres
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res_out, indent=2))
    print(f"\n[v2.2] wrote {a.out}")

if __name__ == "__main__":
    main()
