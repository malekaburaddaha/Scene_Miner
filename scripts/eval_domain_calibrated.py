#!/usr/bin/env python3
"""MILESTONE 7 - Table 3 on the 150-query suite + per-domain score calibration.

Domain skew: retrieved top-k over-represent nuScenes relative to the relevant
base rate. Fix = z-score each ranker's scores WITHIN nuScenes and WITHIN Waymo
before global ranking and MMR, so a scene competes against its own domain's
score distribution rather than against a cross-domain scale offset.

Reports every ranker on all four strata, with and without calibration, plus the
nuScenes share of the top-k against the relevant base rate.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, p_at, ndcg_at, load_attributes, predicate_mask

SEG = re.compile(r"#c\d+$")

def mmr(order, z, Vn, toks, k, per_source=1, sim_thr=0.95, lam=0.7):
    sel, cnt = [], {}
    for i in order:
        if len(sel) >= k: break
        src = SEG.sub("", toks[i])
        if cnt.get(src, 0) >= per_source: continue
        if sel and float((Vn[i] @ Vn[sel].T).max()) > sim_thr: continue
        sel.append(i); cnt[src] = cnt.get(src, 0) + 1
    for i in order:
        if len(sel) >= k: break
        if i not in sel: sel.append(int(i))
    return sel[:k]

def domain_z(s, dom):
    out = np.empty_like(s, dtype=float)
    for d in (0, 1):
        m = dom == d
        if m.sum() > 1:
            out[m] = (s[m] - s[m].mean()) / (s[m].std() + 1e-9)
        else:
            out[m] = s[m]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--suite", default="benchmark_cvpr/queries_390.json")
    ap.add_argument("--split", default="benchmark_cvpr/split_906_602.json")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default="results/cvpr/milestone7_tables2_3.json")
    a = ap.parse_args()
    dev = torch.device("cuda"); W = Path(a.work)
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
    Vn = torch.nn.functional.normalize(sig, dim=-1).cpu().numpy()[idx]

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
    suite = json.load(open(a.suite))["eval"]

    # ---- text encoders ----
    from transformers import (AutoModel, GemmaTokenizerFast,
                              CLIPTextModelWithProjection, CLIPTokenizerFast)
    import torch.nn.functional as F
    P = "google/siglip2-base-patch16-224"
    sm_ = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    tkz = GemmaTokenizerFast.from_pretrained(P)
    feat = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    texts = [q["text"] for q in suite]
    with torch.no_grad():
        ti = tkz(texts, padding="max_length", truncation=True, max_length=64,
                 return_tensors="pt").to(dev)
        QS = feat(sm_.get_text_features(**ti))
    del sm_; torch.cuda.empty_cache()
    ct = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    ce = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True).to(dev).eval()
    with torch.no_grad():
        tt = ct(texts, padding=True, truncation=True, return_tensors="pt").to(dev)
        QC = F.normalize(ce(**tt).text_embeds.float(), dim=-1)
    del ce; torch.cuda.empty_cache()
    clip_img = None   # CLIP image embeddings not cached; SigLIP2 is the vision baseline

    # ---- SceneMiner v2.1 (5 seeds) ----
    import glob
    from scene_mining.filter.sceneminer_v2 import SceneMinerV21
    models = []
    for ck in sorted(glob.glob("work_dirs/sceneminer_v21/s*/final.pt")):
        d = torch.load(ck, map_location=dev, weights_only=False)
        m = SceneMinerV21(d_qwen=qwen.shape[1], d_siglip=sig.shape[1],
                          n_margins=mzz.shape[1], n_meta=6, d_text=QS.shape[1],
                          risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
        m.load_state_dict(d["state_dict"]); m.eval(); models.append(m)
    print(f"[tables] {len(models)} v2.1 seeds, {len(suite)} eval queries, "
          f"{len(toks)} eval scenes", flush=True)

    MARG_COL = {"droplets":0, "glare":1, "wet_road":2, "low_contrast":3}
    rows, skew = {}, {}
    rng = np.random.default_rng(0)
    for qi, q in enumerate(suite):
        rel = predicate_mask(attrs, toks, q["predicate"])
        if rel.sum() == 0 or rel.sum() == len(toks): continue
        st = q["stratum"]
        qs_v = QS[qi].unsqueeze(0).expand(len(corpus), -1)
        rankers = {}
        with torch.no_grad():
            rankers["zeroshot_SigLIP2"] = F.cosine_similarity(
                F.normalize(QS[qi].unsqueeze(0), -1), F.normalize(sig, -1)).cpu().numpy()[idx]
            sm_s = np.mean([m(qs_v, qs_v, emb, tlog, risk, qwen, sig, mzz, prior
                              ).cpu().numpy()[idx] for m in models], 0)
            rankers["SceneMiner_v2.1"] = sm_s
        # lookup baselines keyed on the attribute the predicate names
        at = next((k for k in MARG_COL if k in q["predicate"]), None)
        if at is not None:
            src = strict if (at == "low_contrast" and strict) else vlm
            key = "low_contrast_strict" if (at == "low_contrast" and strict) else at
            rankers["logit_VLM_lookup"] = np.array(
                [float(src.get(t,{}).get(key+"_margin",-9.)) for t in toks])
            rankers["flat_VLM_lookup"] = np.array(
                [float(src.get(t,{}).get(key,False)) for t in toks]) + rng.random(len(toks))*1e-6
        rankers["metadata_prior"] = np.array(
            [float(all(attrs.get(t,{}).get(k)==v for k,v in q["predicate"].items()
                       if k in ("lighting","weather","density"))) for t in toks]) \
            + rng.random(len(toks))*1e-6
        for nm, s in rankers.items():
            for cal in (False, True):
                sc = domain_z(s, dom) if cal else s
                o = np.argsort(-sc)
                key = f"{nm}|{'cal' if cal else 'raw'}"
                rows.setdefault(key, {}).setdefault(st, []).append(
                    (ap_binary(o, rel), p_at(o, rel), ndcg_at(o, rel.astype(float))))
                sel = mmr(o, sc, Vn, toks, a.topk)
                nus = float(np.mean(dom[sel] == 0))
                skew.setdefault(key, []).append((nus, float(np.mean(dom[rel] == 0))))
    out = {"n_queries": sum(len(v) for v in next(iter(rows.values())).values()),
           "n_scenes": len(toks), "tables": {}, "domain_skew": {}}
    STR = ["sensor_degradation","macro_environmental","compositional_heldout","negated"]
    print(f"\n{'ranker':<28}" + "".join(f"{s[:11]:>13}" for s in STR) + f"{'ALL':>9}")
    for key in sorted(rows, key=lambda k: -np.mean([m[0] for v in rows[k].values() for m in v])):
        per = {s: float(np.mean([m[0] for m in rows[key].get(s, [])])) if rows[key].get(s) else float("nan")
               for s in STR}
        allv = float(np.mean([m[0] for v in rows[key].values() for m in v]))
        out["tables"][key] = {"mAP_by_stratum": per, "mAP_all": allv,
                              "P@5_all": float(np.mean([m[1] for v in rows[key].values() for m in v])),
                              "NDCG@10_all": float(np.mean([m[2] for v in rows[key].values() for m in v]))}
        print(f"{key:<28}" + "".join(f"{per[s]:>13.4f}" for s in STR) + f"{allv:>9.4f}")
    print(f"\n{'ranker':<28}{'nuSc share top-5':>18}{'relevant base':>15}{'skew pts':>10}")
    for key in sorted(skew):
        arr = np.array(skew[key]); sh, base = arr[:,0].mean(), arr[:,1].mean()
        out["domain_skew"][key] = {"nuscenes_share_top5": float(sh),
                                   "relevant_base_rate": float(base),
                                   "skew_points": float((sh-base)*100)}
        print(f"{key:<28}{sh:>18.3f}{base:>15.3f}{(sh-base)*100:>+10.1f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[tables] wrote {a.out}")

if __name__ == "__main__":
    main()
