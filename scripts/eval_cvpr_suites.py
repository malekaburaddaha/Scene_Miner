#!/usr/bin/env python3
"""Evaluate the CVPR ablation (Table 2) and the fair LOAO suite on the 602-scene
eval split. All metrics are computed ONLY over eval-split scenes; no evaluated
scene contributed a gradient (enforced at train time by --split-file).

LOAO fairness: for the held-out attribute, the lookup baseline is DENIED that
attribute and must fall back to its best remaining proxy predicate, exactly as
SceneMiner must. Otherwise the lookup trivially wins by reading the cached flag.
"""
from __future__ import annotations
from paths import QWEN_VL
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import (MACRO8, DEGRAD, load_attributes, predicate_mask,
                                  ap_binary, p_at, ndcg_at)

# best proxy a lookup can use once an attribute is withheld from it
PROXY = {"droplets":     {"weather": "rain"},
         "glare":        {"lighting": "night"},
         "wet_road":     {"weather": "rain"},
         "low_contrast": {"lighting": "night"}}

def queries_for(attr):
    """Queries whose predicate mentions `attr` (the held-out-attribute suite)."""
    return [(q, p) for q, p in (DEGRAD + MACRO8) if attr in p]

def score_model(ckpt, toks, W, args, queries, legacy_scoring=False,
                legacy_scalar=False, lora_r=1, tau=0.30):
    from scene_mining.filter.lora_grounded_filter import (LoRAGroundedFilter,
                                                          QwenVisualEncoder)
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
    import torch.nn.functional as F
    from PIL import Image
    dev = torch.device("cuda")
    z = np.load(W/"features_overnight.npz", allow_pickle=False)
    all_toks = [str(t) for t in z["scene_tokens"]]
    keep = np.array([t in set(toks) for t in all_toks])
    emb = torch.from_numpy(z["emb"]).float().to(dev)
    tlog = torch.from_numpy(z["tag_logits"]).float().to(dev)
    risk = torch.from_numpy(z["risk"]).float().to(dev)
    rk = risk.reshape(-1).cpu().numpy()
    enc = QwenVisualEncoder(args.qwen, lora_r=lora_r, lora_alpha=128,
                            device=str(dev), img_max_side=336)
    enc.model.to(dev)
    m = LoRAGroundedFilter(enc, tau=tau, legacy_scoring=legacy_scoring,
                           legacy_scalar=legacy_scalar,
                           risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(ck["trainable_state_dict"], strict=False); m.eval()
    # The visual tower is FROZEN for every row we evaluate (lora_r=1), so V is
    # identical across all 45 checkpoints. Compute once, cache to disk, reuse.
    cache = Path("work_dirs/unified_filter/V_frozen_cache.pt")
    if cache.exists():
        V = torch.load(cache, map_location=dev)
    else:
        V = torch.zeros(len(all_toks), enc.d_model, device=dev)
        with torch.no_grad():
            for i in range(0, len(all_toks), 16):
                ims = [Image.open(Path(args.thumbs)/f"{t}.jpg").convert("RGB")
                       for t in all_toks[i:i+16]]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    V[i:i+16] = enc.encode(ims)
        torch.save(V.cpu(), cache); V = V.to(dev)
        print(f"  [cache] wrote {cache}", flush=True)
    ctok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    cenc = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True).to(dev).eval()
    out = {}
    with torch.no_grad():
        for q, _ in queries:
            tt = ctok([q], padding=True, truncation=True, return_tensors="pt").to(dev)
            qc = F.normalize(cenc(**tt).text_embeds.float(), dim=-1).expand(len(all_toks), -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = m.score(qc, emb, tlog, risk, V).float().cpu().numpy()
            out[q] = s[keep]
    del m, enc; torch.cuda.empty_cache()
    return out

def metrics(scores, toks, attrs, queries):
    ap, p5, nd = [], [], []
    for q, pred in queries:
        rel = predicate_mask(attrs, toks, pred)
        if rel.sum() == 0 or rel.sum() == len(toks): continue
        o = np.argsort(-scores[q])
        ap.append(ap_binary(o, rel)); p5.append(p_at(o, rel))
        nd.append(ndcg_at(o, rel.astype(float)))
    return (float(np.mean(ap)) if ap else float("nan"),
            float(np.mean(p5)) if p5 else float("nan"),
            float(np.mean(nd)) if nd else float("nan"), len(ap))

def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--work", default="work_dirs/unified_filter")
    ap_.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap_.add_argument("--qwen", default=QWEN_VL)
    ap_.add_argument("--split", default="benchmark_cvpr/split_906_602.json")
    ap_.add_argument("--mode", choices=["ablation", "loao"], required=True)
    ap_.add_argument("--out", required=True)
    a = ap_.parse_args()
    W = Path(a.work)
    _evset = set(json.load(open(a.split))["eval"])
    # CRITICAL: scores are gathered as s[keep] in CORPUS order, so the token list
    # used for relevance must be in corpus order too. The split file stores
    # `eval` sorted; using it directly misaligns scores with labels and drives
    # every row to chance.
    _z0 = np.load(W/"features_overnight.npz", allow_pickle=False)
    ev = [str(t) for t in _z0["scene_tokens"] if str(t) in _evset]
    assert len(ev) == len(_evset), f"{len(ev)} != {len(_evset)}"
    attrs = load_attributes(W)
    # prefer the 4-attribute extraction with logits where available
    newf = W/"binary_flags_with_logits.json"
    if newf.exists():
        for t, r in json.load(open(newf)).items():
            attrs.setdefault(t, {}).update({k: v for k, v in r.items()
                                            if not k.endswith(("_margin", "_logp_yes"))})
    res = {}
    if a.mode == "ablation":
        QS = DEGRAD + MACRO8
        LEG = {"A": (True, True), "A1": (True, True), "A2": (True, True),
               "B": (True, False), "E": (False, False)}
        TAU = {"A": 1.0, "A1": 1.0, "A2": 1.0, "B": 1.0, "E": 0.30}
        for row in ["A", "A1", "A2", "B", "E"]:
            per = []
            for s in range(5):
                ck = Path(f"work_dirs/cvpr_ablation/{row}_s{s}/final.pt")
                if not ck.exists(): continue
                sc = score_model(str(ck), ev, W, a, QS, *LEG[row], 1, TAU[row])
                per.append(metrics(sc, ev, attrs, QS))
                print(f"  {row} s{s}: mAP={per[-1][0]:.4f} P@5={per[-1][1]:.4f}", flush=True)
            if per:
                A = np.array([p[:3] for p in per])
                res[row] = {"n_seeds": len(per),
                            "mAP": float(A[:,0].mean()), "mAP_std": float(A[:,0].std()),
                            "P@5": float(A[:,1].mean()), "P@5_std": float(A[:,1].std()),
                            "NDCG@10": float(A[:,2].mean()), "NDCG@10_std": float(A[:,2].std())}
                print(f"ROW {row}: mAP {res[row]['mAP']:.4f} +/- {res[row]['mAP_std']:.4f} "
                      f"({len(per)} seeds)", flush=True)
    else:
        for attr in ["droplets", "glare", "wet_road", "low_contrast"]:
            QS = queries_for(attr)
            if not QS: continue
            per = []
            for s in range(5):
                ck = Path(f"work_dirs/cvpr_loao/{attr}_s{s}/final.pt")
                if not ck.exists(): continue
                sc = score_model(str(ck), ev, W, a, QS, False, False, 1, 0.30)
                per.append(metrics(sc, ev, attrs, QS))
                print(f"  LOAO[{attr}] s{s}: mAP={per[-1][0]:.4f}", flush=True)
            # lookup DENIED the held-out attribute -> best proxy predicate
            lk = []
            for q, pred in QS:
                rel = predicate_mask(attrs, ev, pred)
                if rel.sum() == 0: continue
                prox = {k: v for k, v in pred.items() if k != attr}
                prox.update(PROXY.get(attr, {}))
                tier = predicate_mask(attrs, ev, prox).astype(float)
                rng = np.random.default_rng(0); acc = []
                for _ in range(200):
                    o = np.lexsort((rng.random(len(ev)), -tier))
                    acc.append(ap_binary(o, rel))
                lk.append(float(np.mean(acc)))
            r = {"n_queries": len(QS), "lookup_denied_mAP": float(np.mean(lk)) if lk else float("nan")}
            if per:
                A = np.array([p[:3] for p in per])
                r.update({"n_seeds": len(per), "sceneminer_mAP": float(A[:,0].mean()),
                          "sceneminer_mAP_std": float(A[:,0].std()),
                          "sceneminer_P@5": float(A[:,1].mean()),
                          "sceneminer_NDCG@10": float(A[:,2].mean())})
            res[attr] = r
            print(f"LOAO {attr}: SM {r.get('sceneminer_mAP', float('nan')):.4f} "
                  f"vs lookup-denied {r['lookup_denied_mAP']:.4f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[eval] wrote {a.out}")

if __name__ == "__main__":
    main()
