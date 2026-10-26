#!/usr/bin/env python3
"""Human-GT evaluation, both modes, all five rankers.

Mode A (strict): exclude flag in {skip, unsure}.
Mode B (full)  : additionally KEEP flag=="skip" rows whose toggles are all-False,
                 treating them as genuine 'clear scene' negatives. Rows flagged
                 'unsure' are excluded in both modes.

This is the evaluation where the cached-VLM lookup is NOT an oracle: relevance
comes from human labels, so the lookup inherits the VLM's own error.
"""
from __future__ import annotations
from paths import HUMAN_R1, QWEN_VL
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, p_at, ndcg_at, load_attributes

ATTRS = ["droplets", "glare", "wet_road", "low_contrast"]
QUERY = {
 "droplets":     "water droplets smearing the camera lens",
 "glare":        "blinding headlight bloom from oncoming traffic",
 "wet_road":     "wet reflective asphalt after rainfall",
 "low_contrast": "a low contrast scene where vehicles are barely distinguishable "
                 "from the background",
}
# metadata-matched proxy: nuScenes/Waymo recorded weather+lighting only
META = {"droplets": {"weather":"rain"}, "glare": {"lighting":"night"},
        "wet_road": {"weather":"rain"}, "low_contrast": {"lighting":"night"}}

def kappa(a,b):
    a=np.asarray(a,bool); b=np.asarray(b,bool); n=len(a)
    if n==0: return float("nan")
    po=float((a==b).mean()); p1,p2=a.mean(),b.mean()
    pe=p1*p2+(1-p1)*(1-p2)
    return float((po-pe)/(1-pe)) if pe<1 else float("nan")

def prf(human, pred):
    tp=int((human&pred).sum()); fp=int((~human&pred).sum())
    fn=int((human&~pred).sum()); tn=int((~human&~pred).sum())
    P=tp/max(1,tp+fp); R=tp/max(1,tp+fn)
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,"precision":P,"recall":R,
            "F1":2*P*R/max(1e-9,P+R),"FPR":fp/max(1,fp+tn)}

def load_human(path, mode):
    d=json.loads(Path(path).read_text()); out={}
    for t,r in d.items():
        fl=r.get("flag"); lab=r["labels"]
        if fl=="unsure": continue
        if fl=="skip":
            if mode=="A": continue
            if any(lab.values()): continue          # ambiguous skip -> drop
        out[t]={k:bool(lab.get(k,False)) for k in ATTRS}
    return out

@torch.no_grad()
def siglip_scores(toks, thumbs, queries, batch=64):
    from transformers import AutoImageProcessor, AutoModel, GemmaTokenizerFast
    from PIL import Image
    import torch.nn.functional as F
    P="google/siglip2-base-patch16-224"; dev=torch.device("cuda")
    m=AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    ip=AutoImageProcessor.from_pretrained(P); tk=GemmaTokenizerFast.from_pretrained(P)
    feat=lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    V=torch.zeros(len(toks), m.config.text_config.hidden_size, device=dev)
    for i in range(0,len(toks),batch):
        ims=[Image.open(Path(thumbs)/f"{t}.jpg").convert("RGB") for t in toks[i:i+batch]]
        V[i:i+batch]=feat(m.get_image_features(**ip(images=ims,return_tensors="pt").to(dev)))
    ti=tk(queries,padding="max_length",truncation=True,max_length=64,return_tensors="pt").to(dev)
    Q=feat(m.get_text_features(**ti))
    V=F.normalize(V,dim=-1); Q=F.normalize(Q,dim=-1)
    out={q:(V@Q[i]).cpu().numpy() for i,q in enumerate(queries)}
    del m; torch.cuda.empty_cache(); return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--human", default=HUMAN_R1)
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--qwen", default=QWEN_VL)
    ap.add_argument("--out", default="results/day1_gate/human_gt_full.json")
    a=ap.parse_args()
    W=Path(a.work)
    z=np.load(W/"features_overnight.npz",allow_pickle=False)
    corpus=[str(t) for t in z["scene_tokens"]]; cidx={t:i for i,t in enumerate(corpus)}
    vlm=json.load(open(W/"binary_flags_with_logits.json"))
    strict=json.load(open(W/"lowcontrast_strict.json")) if (W/"lowcontrast_strict.json").exists() else {}
    det=load_attributes(W)

    # --- model scores over the full corpus, computed once ---
    qs=[QUERY[a_] for a_ in ATTRS]
    print("[scores] SigLIP2 zero-shot ...", flush=True)
    sig=siglip_scores(corpus, a.thumbs, qs)
    print("[scores] SceneMiner D_final + 5-seed row E ...", flush=True)
    import eval_cvpr_suites as E
    QS=[(QUERY[a_],{}) for a_ in ATTRS]
    sm_d = E.score_model("work_dirs/ablation/D_final/final.pt", corpus, W, a, QS, False, False, 64, 0.30)
    sm_e=[]
    import glob
    for ck in sorted(glob.glob("work_dirs/cvpr_ablation/E_s*/final.pt")):
        sm_e.append(E.score_model(ck, corpus, W, a, QS, False, False, 1, 0.30))
    print(f"[scores] done ({len(sm_e)} row-E seeds)", flush=True)

    rng=np.random.default_rng(0); res={}
    for mode in ["A","B"]:
        hum=load_human(a.human, mode)
        toks=[t for t in corpus if t in hum]
        print(f"\n{'='*78}\nMODE {mode}: {len(toks)} scenes\n{'='*78}")
        mres={"n_scenes":len(toks),"attributes":{}}
        for at in ATTRS:
            H=np.array([hum[t][at] for t in toks])
            src = strict if (at=="low_contrast" and strict) else vlm
            key = "low_contrast_strict" if (at=="low_contrast" and strict) else at
            Vb=np.array([bool(src.get(t,{}).get(key,False)) for t in toks])
            Vm=np.array([float(src.get(t,{}).get(key+"_margin",-9.0)) for t in toks])
            if H.sum()==0 or H.sum()==len(toks):
                mres["attributes"][at]={"skipped":"degenerate"}; continue
            cls=prf(H,Vb); cls["kappa"]=kappa(H,Vb)
            print(f"\n--- {at}  (human+ {int(H.sum())}/{len(toks)}, chance {H.mean():.3f}) ---")
            print(f"  VLM classification: P={cls['precision']:.3f} R={cls['recall']:.3f} "
                  f"F1={cls['F1']:.3f} FPR={cls['FPR']:.3f} kappa={cls['kappa']:+.3f} "
                  f"(FP={cls['FP']} FN={cls['FN']})")
            idx=[cidx[t] for t in toks]
            rank={}
            # (a) flat VLM lookup, random tie-break
            acc=[ap_binary(np.lexsort((rng.random(len(toks)),-Vb.astype(float))),H) for _ in range(500)]
            rank["flat_VLM_lookup"]={"mAP":float(np.mean(acc))}
            def add(nm, s):
                o=np.argsort(-s)
                rank[nm]={"mAP":ap_binary(o,H),"P@5":p_at(o,H),"NDCG@10":ndcg_at(o,H.astype(float))}
                return o
            add("logit_VLM_lookup", Vm)
            mp=np.array([float(all(det.get(t,{}).get(k)==v for k,v in META[at].items())) for t in toks])
            add("metadata_matched", mp)
            add("zeroshot_SigLIP2", sig[QUERY[at]][idx])
            o_d=add("SceneMiner_D_final", sm_d[QUERY[at]][idx])
            seedm=[]
            for s_ in sm_e:
                o=np.argsort(-s_[QUERY[at]][idx]); seedm.append(ap_binary(o,H))
            if seedm:
                rank["SceneMiner_rowE_5seed"]={"mAP":float(np.mean(seedm)),
                                               "mAP_std":float(np.std(seedm)),
                                               "n_seeds":len(seedm)}
            # ---- VLM false-negative recovery ----
            miss = H & ~Vb
            recov={}
            if miss.sum()>0:
                for nm,s in [("logit_VLM_lookup",Vm),
                             ("zeroshot_SigLIP2",sig[QUERY[at]][idx]),
                             ("SceneMiner_D_final",sm_d[QUERY[at]][idx])]:
                    o=np.argsort(-s); r=np.empty(len(toks)); r[o]=np.arange(len(toks))
                    recov[nm]={"median_rank_pct":float(np.median(r[miss])/len(toks)),
                               "n_in_top20pct":int((r[miss]<0.2*len(toks)).sum())}
                # flat lookup ranks every FN in the random tail by construction
                nneg=int((~Vb).sum())
                recov["flat_VLM_lookup"]={"median_rank_pct":float((len(toks)-nneg/2)/len(toks)),
                                          "n_in_top20pct":0,
                                          "note":"FNs sit below the positive bucket by construction"}
            print(f"  {'ranker':<24}{'mAP':>8}{'P@5':>8}{'NDCG@10':>9}")
            for nm,v in rank.items():
                print(f"  {nm:<24}{v['mAP']:>8.4f}{v.get('P@5',float('nan')):>8.4f}"
                      f"{v.get('NDCG@10',float('nan')):>9.4f}")
            if recov:
                print(f"  VLM false negatives: {int(miss.sum())} scenes human+ but VLM-")
                for nm,v in recov.items():
                    print(f"    {nm:<24} median rank {v['median_rank_pct']*100:5.1f}%  "
                          f"in top-20%: {v['n_in_top20pct']}")
            mres["attributes"][at]={"n_human_pos":int(H.sum()),"chance":float(H.mean()),
                                    "vlm_classification":cls,"rankers":rank,
                                    "vlm_fn_recovery":recov}
        res[f"mode_{mode}"]=mres
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(res,indent=2))
    print(f"\n[human-gt] wrote {a.out}")

if __name__=="__main__":
    main()
