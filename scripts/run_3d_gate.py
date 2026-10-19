#!/usr/bin/env python3
"""MILESTONE 1 - the 3D gate, grounded in OFFICIAL nuScenes per-scene mAP.

Difficulty = per-scene nuScenes mAP from bevformer_base, computed with the
official accumulate/calc_ap (validated: the same predictions give global
mAP 0.4164 / NDS 0.5173, matching the published checkpoint). LOWER mAP = HARDER.

Rankers compared at equal bucket size: random, metadata-matched, flat VLM
lookup, logit VLM lookup, zero-shot SigLIP2, SceneMiner. Reports top-k mean
difficulty, the decile curve, and the within-positive-bucket ordering test
(where a binary lookup is blind because every scene in the bucket is tied).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import load_attributes

ATTR_Q = {
 "droplets":     "water droplets smearing the camera lens",
 "glare":        "blinding headlight bloom from oncoming traffic",
 "wet_road":     "wet reflective asphalt after rainfall",
 "low_contrast": "a low contrast scene where vehicles are barely distinguishable from the background",
}
META = {"droplets": {"weather":"rain"}, "glare": {"lighting":"night"},
        "wet_road": {"weather":"rain"}, "low_contrast": {"lighting":"night"}}

def spearman(a,b):
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    ra-=ra.mean(); rb-=rb.mean(); d=np.sqrt((ra**2).sum()*(rb**2).sum())
    return float((ra*rb).sum()/d) if d>0 else 0.0

def perm_p(a,b,rho,n=2000,seed=0):
    rng=np.random.default_rng(seed); c=0
    for _ in range(n):
        if abs(spearman(a,rng.permutation(b)))>=abs(rho): c+=1
    return (c+1)/(n+1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--per-scene", required=True)
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--sm-scores", default="results/overnight_oct18/sm_scores.json")
    ap.add_argument("--siglip-scores", default="results/cvpr/siglip_scores.json")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()

    W=Path(a.work)
    diff=json.load(open(a.per_scene))
    z=np.load(W/"features_overnight.npz",allow_pickle=False)
    corpus=[str(t) for t in z["scene_tokens"]]; cidx={t:i for i,t in enumerate(corpus)}
    toks=[t for t in corpus if t in diff and diff[t]["mAP"]==diff[t]["mAP"]]
    D=np.array([diff[t]["mAP"] for t in toks])
    HARD=-D   # higher = harder
    attrs=load_attributes(W)
    vlm=json.load(open(W/"binary_flags_with_logits.json"))
    strict=json.load(open(W/"lowcontrast_strict.json")) if (W/"lowcontrast_strict.json").exists() else {}
    sm=json.load(open(a.sm_scores)) if Path(a.sm_scores).exists() else {}
    sig=json.load(open(a.siglip_scores)) if Path(a.siglip_scores).exists() else {}
    print(f"[gate] {len(toks)} scenes with official per-scene mAP")
    print(f"[gate] difficulty: mAP mean {D.mean():.4f} sd {D.std():.4f} "
          f"range [{D.min():.3f},{D.max():.3f}]")

    rng=np.random.default_rng(0); rows=[]; within=[]; deciles={}
    for at,q in ATTR_Q.items():
        src = strict if (at=="low_contrast" and strict) else vlm
        key = "low_contrast_strict" if (at=="low_contrast" and strict) else at
        flat=np.array([float(src.get(t,{}).get(key,False)) for t in toks])
        marg=np.array([float(src.get(t,{}).get(key+"_margin",-9.0)) for t in toks])
        meta=np.array([float(all(attrs.get(t,{}).get(k)==v for k,v in META[at].items()))
                       for t in toks])
        smv = np.array([sm[q][cidx[t]] for t in toks]) if q in sm else None
        sgv = np.array([sig[q][cidx[t]] for t in toks]) if q in sig else None
        nb=int(flat.sum())
        if nb<10: 
            print(f"  [skip] {at}: bucket {nb} too small"); continue
        k=min(a.topk,nb)
        topk=lambda s: float(np.mean(HARD[np.argsort(-s)[:k]]))
        r={"attribute":at,"bucket":nb,"k":k,
           "random":float(np.mean(HARD[rng.choice(len(toks),k,replace=False)])),
           "corpus_mean":float(HARD.mean()),
           "flat_lookup":float(np.mean(HARD[rng.choice(np.where(flat>0)[0],k,replace=False)])),
           "logit_lookup":topk(marg),
           "metadata":topk(meta) if meta.sum()>0 else float("nan"),
           "zeroshot_siglip2":topk(sgv) if sgv is not None else float("nan"),
           "sceneminer":topk(smv) if smv is not None else float("nan")}
        rows.append(r)
        # within-bucket: lookup is tied across the whole positive bucket
        b=np.where(flat>0)[0]
        if len(b)>=20 and smv is not None:
            bd,bs=HARD[b],smv[b]; rho=spearman(bs,bd); kk=max(5,len(b)//5)
            top=bd[np.argsort(-bs)[:kk]]
            w={"attribute":at,"bucket":int(len(b)),
               "spearman_sceneminer_vs_hardness":rho,"perm_p":perm_p(bs,bd,rho),
               "bucket_mean_hardness":float(bd.mean()),
               "sm_top20pct_hardness":float(top.mean()),
               "lift":float(top.mean()-bd.mean())}
            if sgv is not None:
                w["spearman_siglip_vs_hardness"]=spearman(sgv[b],bd)
            if marg is not None:
                w["spearman_logit_vs_hardness"]=spearman(marg[b],bd)
            within.append(w)
        if smv is not None:
            o=np.argsort(-smv); deciles[at]=[float(HARD[p].mean()) for p in np.array_split(o,10)]
    if deciles:
        deciles["MEAN"]=np.array(list(deciles.values())).mean(0).tolist()

    out={"n_scenes":len(toks),"difficulty":"official per-scene nuScenes mAP (negated)",
         "topk":rows,"within_bucket":within,"decile_curve":deciles}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(out,indent=2))

    print(f"\n=== TOP-{a.topk} MEAN HARDNESS (higher = found harder scenes) ===")
    ks=["random","flat_lookup","metadata","logit_lookup","zeroshot_siglip2","sceneminer"]
    print(f"{'ranker':<20}"+"".join(f"{r['attribute'][:9]:>12}" for r in rows)+f"{'MEAN':>10}")
    for kk in ks:
        vs=[r[kk] for r in rows]
        print(f"{kk:<20}"+"".join(f"{v:>12.4f}" for v in vs)+f"{np.nanmean(vs):>10.4f}")
    if within:
        print(f"\n=== WITHIN-BUCKET (lookup is tied here) ===")
        for w in within:
            print(f"  {w['attribute']:<14} n={w['bucket']:4d}  SM rho={w['spearman_sceneminer_vs_hardness']:+.3f} "
                  f"(p={w['perm_p']:.3f})  SigLIP rho={w.get('spearman_siglip_vs_hardness',float('nan')):+.3f}  "
                  f"lift={w['lift']:+.4f}")
    if "MEAN" in deciles:
        print(f"\n=== DECILE CURVE (SceneMiner rank -> hardness) ===")
        print("  "+"  ".join(f"D{i+1}:{v:+.3f}" for i,v in enumerate(deciles["MEAN"])))
    print(f"\n[gate] wrote {a.out}")

if __name__=="__main__":
    main()
