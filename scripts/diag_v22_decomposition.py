#!/usr/bin/env python3
"""MILESTONE 14 - decompose v2.2 into base and gated residual on the 550 human
scenes, and inspect the 20 VLM-missed glare scenes. This is the diagnostic that
located the residual-dominance problem ResNorm fixes."""
from paths import HUMAN_R1
import glob, json, sys, numpy as np, torch
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
from eval_human_gt_full import load_human, ATTRS, QUERY
from scene_mining.filter.sceneminer_v2_2 import SceneMinerV22
dev=torch.device("cuda"); W="work_dirs/unified_filter"
z=np.load(f"{W}/features_v2.npz",allow_pickle=False)
corpus=[str(t) for t in z["scene_tokens"]]; cidx={t:i for i,t in enumerate(corpus)}
T=lambda k: torch.from_numpy(z[k]).float().to(dev)
sig,mz,prior=T("siglip_img"),T("vlm_margin_z"),T("prior")
from transformers import AutoModel, GemmaTokenizerFast
P="google/siglip2-base-patch16-224"
sm=AutoModel.from_pretrained(P,use_safetensors=True).to(dev).eval(); tk=GemmaTokenizerFast.from_pretrained(P)
ft=lambda o:(o if torch.is_tensor(o) else o.pooler_output).float()
with torch.no_grad():
    QT=ft(sm.get_text_features(**tk([QUERY[a] for a in ATTRS],padding="max_length",truncation=True,max_length=64,return_tensors="pt").to(dev)))
M=[]
for c in sorted(glob.glob("work_dirs/sceneminer_v22/s*/final.pt")):
    d=torch.load(c,map_location=dev,weights_only=False)
    m=SceneMinerV22(d_siglip=768,n_margins=4,n_meta=6,d_text=768).to(dev); m.load_state_dict(d["state_dict"],strict=False); m.eval(); M.append(m)
hum=load_human(HUMAN_R1,"B")
vlm=json.load(open(f"{W}/binary_flags_with_logits.json"))
toks=[t for t in corpus if t in hum]; idx=np.array([cidx[t] for t in toks]); N=len(corpus)
av=torch.ones(N,3,device=dev)
for ai,at in enumerate(ATTRS):
    qv=QT[ai].unsqueeze(0).expand(N,-1)
    with torch.no_grad():
        p=np.mean([m.base_parts(qv,qv,sig,mz,prior).cpu().numpy() for m in M],0)[idx]
        w=np.mean([m.route(qv,av).mean(0).cpu().numpy() for m in M],0)
    # per-query standardise over the 550 scenes for interpretability
    zs=(p-p.mean(0))/p.std(0)
    H=np.array([hum[t][at] for t in toks]); V=np.array([bool(vlm.get(t,{}).get(at,False)) for t in toks])
    miss=H&~V
    print(f"{at:<13} routing[siglip,margin,meta]={np.round(w,3)}  human+={H.sum()} VLM-missed={miss.sum()}")
    if miss.sum():
        print(f"   on VLM-missed positives: mean z_siglip {zs[miss,0].mean():+.2f}  mean z_margin {zs[miss,1].mean():+.2f}  "
              f"frac z_siglip>+1: {np.mean(zs[miss,0]>1):.2f}  frac z_margin<-0.5: {np.mean(zs[miss,1]<-0.5):.2f}")
        print(f"   on all negatives:       mean z_siglip {zs[~H,0].mean():+.2f}  mean z_margin {zs[~H,1].mean():+.2f}")
        # how many missed positives in top 20% under siglip alone?
        o=np.argsort(-zs[:,0]); r=np.empty(len(o)); r[o]=np.arange(len(o))
        print(f"   SigLIP2-alone puts {int((r[miss]<0.2*len(o)).sum())}/{miss.sum()} missed positives in top 20%")

print("\n=== decomposition: base vs residual vs full (5-seed mean) ===")
from eval_attribute_index import ap_binary
emb,tlog,risk=T("emb"),T("tag_logits"),T("risk")
qwen=torch.load(f"{W}/V_frozen_cache.pt",map_location=dev).float()
on=torch.ones(N,device=dev)
for ai,at in enumerate(ATTRS):
    qv=QT[ai].unsqueeze(0).expand(N,-1)
    with torch.no_grad():
        full,base,res,g=[],[],[],[]
        for m in M:
            s,b,r,w,pp,gg=m(qv,qv,emb,tlog,risk,qwen,sig,mz,prior,has_3d=on,return_parts=True)
            full.append(s.cpu().numpy()); base.append(b.cpu().numpy()); res.append((gg*r).cpu().numpy())
    full,base,res=[np.mean(x,0)[idx] for x in (full,base,res)]
    H=np.array([hum[t][at] for t in toks]); V=np.array([bool(vlm.get(t,{}).get(at,False)) for t in toks]); miss=H&~V
    def rec(s):
        o=np.argsort(-s); r=np.empty(len(o)); r[o]=np.arange(len(o)); return int((r[miss]<0.2*len(o)).sum())
    print(f"{at:<13} mAP base {ap_binary(np.argsort(-base),H):.4f}  gated-res {ap_binary(np.argsort(-res),H):.4f}  "
          f"full {ap_binary(np.argsort(-full),H):.4f} | FN-rec base {rec(base)}/{miss.sum()} res {rec(res)} full {rec(full)} "
          f"| std base {base.std():.3f} gated-res {res.std():.3f}")
