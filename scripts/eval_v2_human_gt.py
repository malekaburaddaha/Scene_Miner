#!/usr/bin/env python3
"""MILESTONE 2 - SceneMiner v2 vs all baselines against human ground truth."""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, p_at, ndcg_at, load_attributes
from eval_human_gt_full import load_human, ATTRS, QUERY, META, prf, kappa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="results/day1_gate/malek_repaired.json")
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--out", default="results/cvpr/milestone2_sceneminer_v2.json")
    a = ap.parse_args()
    dev = torch.device("cuda"); W = Path(a.work)
    z = np.load(a.features, allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]; cidx = {t: i for i, t in enumerate(corpus)}
    emb = torch.from_numpy(z["emb"]).float().to(dev)
    tlog = torch.from_numpy(z["tag_logits"]).float().to(dev)
    risk = torch.from_numpy(z["risk"]).float().to(dev)
    sig = torch.from_numpy(z["siglip_img"]).float().to(dev)
    mz = torch.from_numpy(z["vlm_margin_z"]).float().to(dev)
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

    from scene_mining.filter.sceneminer_v2 import SceneMinerV2
    v2_scores = []      # per seed: {query: np.array}
    for ck in sorted(glob.glob("work_dirs/sceneminer_v2/s*/final.pt")):
        d = torch.load(ck, map_location=dev, weights_only=False)
        m = SceneMinerV2(d_qwen=qwen.shape[1], d_siglip=sig.shape[1],
                         n_margins=mz.shape[1], d_text=QT.shape[1],
                         risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
        m.load_state_dict(d["state_dict"]); m.eval()
        sc = {}
        with torch.no_grad():
            for qi, q in enumerate(qs):
                qq = QT[qi].unsqueeze(0).expand(len(corpus), -1)
                s, _ = m(qq, qq, emb, tlog, risk, qwen, sig, mz)
                sc[q] = s.float().cpu().numpy()
        v2_scores.append(sc)
    print(f"[v2-eval] {len(v2_scores)} seeds loaded", flush=True)
    # base-only reference (gate forced to 0) = the zero-shot prior
    m0 = SceneMinerV2(d_qwen=qwen.shape[1], d_siglip=sig.shape[1],
                      n_margins=mz.shape[1], d_text=QT.shape[1],
                      risk_stats=(float(rk.mean()), float(rk.std()))).to(dev).eval()
    base_sc = {}
    with torch.no_grad():
        for qi, q in enumerate(qs):
            qq = QT[qi].unsqueeze(0).expand(len(corpus), -1)
            base_sc[q] = m0.base_score(qq, sig, mz).float().cpu().numpy()

    vlm = json.load(open(W/"binary_flags_with_logits.json"))
    strict = json.load(open(W/"lowcontrast_strict.json")) if (W/"lowcontrast_strict.json").exists() else {}
    det = load_attributes(W)
    old = json.load(open("results/cvpr/human_gt_600_both_modes.json")) \
          if Path("results/cvpr/human_gt_600_both_modes.json").exists() else {}

    res = {}
    for mode in ["A", "B"]:
        hum = load_human(a.human, mode)
        toks = [t for t in corpus if t in hum]
        idx = [cidx[t] for t in toks]
        print(f"\n===== MODE {mode}: {len(toks)} scenes =====")
        mres = {"n_scenes": len(toks), "attributes": {}}
        for at in ATTRS:
            H = np.array([hum[t][at] for t in toks])
            if H.sum() == 0 or H.sum() == len(toks): continue
            q = QUERY[at]
            rank = {}
            def add(nm, s):
                o = np.argsort(-s)
                rank[nm] = {"mAP": ap_binary(o, H), "P@5": p_at(o, H),
                            "NDCG@10": ndcg_at(o, H.astype(float))}
            src = strict if (at == "low_contrast" and strict) else vlm
            key = "low_contrast_strict" if (at == "low_contrast" and strict) else at
            add("logit_VLM_lookup", np.array([float(src.get(t,{}).get(key+"_margin",-9.)) for t in toks]))
            add("v2_base_only(zero-shot prior)", base_sc[q][idx])
            seeds = [ap_binary(np.argsort(-sc[q][idx]), H) for sc in v2_scores]
            o0 = np.argsort(-v2_scores[0][q][idx])
            rank["SceneMiner_v2_5seed"] = {"mAP": float(np.mean(seeds)),
                                           "mAP_std": float(np.std(seeds)),
                                           "P@5": p_at(o0, H),
                                           "NDCG@10": ndcg_at(o0, H.astype(float)),
                                           "n_seeds": len(seeds)}
            # carry v1 numbers across for a like-for-like line
            if old:
                prev = old.get(f"mode_{mode}", {}).get("attributes", {}).get(at, {}).get("rankers", {})
                for k in ("flat_VLM_lookup","zeroshot_SigLIP2","metadata_matched",
                          "SceneMiner_D_final","SceneMiner_rowE_5seed"):
                    if k in prev: rank[k+"_v1"] = {"mAP": prev[k]["mAP"]}
            # VLM false-negative recovery for v2
            Vb = np.array([bool(src.get(t,{}).get(key,False)) for t in toks])
            miss = H & ~Vb; recov = {}
            if miss.sum() > 0:
                for nm, s in [("SceneMiner_v2", v2_scores[0][q][idx]),
                              ("v2_base_only", base_sc[q][idx])]:
                    o = np.argsort(-s); r = np.empty(len(toks)); r[o] = np.arange(len(toks))
                    recov[nm] = {"median_rank_pct": float(np.median(r[miss])/len(toks)),
                                 "n_in_top20pct": int((r[miss] < 0.2*len(toks)).sum()),
                                 "n_missed": int(miss.sum())}
            mres["attributes"][at] = {"n_human_pos": int(H.sum()), "chance": float(H.mean()),
                                      "rankers": rank, "vlm_fn_recovery": recov}
            print(f"\n  -- {at} (human+ {int(H.sum())}/{len(toks)}) --")
            for k, v in sorted(rank.items(), key=lambda kv: -kv[1]["mAP"]):
                print(f"    {k:<34}{v['mAP']:>8.4f}")
        res[f"mode_{mode}"] = mres
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\n[v2-eval] wrote {a.out}")

if __name__ == "__main__":
    main()
