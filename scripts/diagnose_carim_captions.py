#!/usr/bin/env python3
"""MILESTONE 23 - why the CARIM-style baseline is weak on the 550 human-GT scenes:
dense vs BM25 parts, and how often captions mention an attribute only to negate it.
-> results/cvpr/milestone23_carim_diagnostic.json"""
from paths import HUMAN_R1
import sys, json, re, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_related_work_baselines import Enc, BM25, zs, load_captions, ap_of
from eval_human_gt_full import load_human, ATTRS, QUERY

cap = load_captions(); z = np.load("work_dirs/unified_filter/features_v2.npz"); toks = [str(t) for t in z["scene_tokens"]]
docs = [cap[f"corpus|{t}"] for t in toks]; enc = Enc(); E = enc.sig2_t(docs); bm = BM25(docs)
hum = load_human(HUMAN_R1, "B"); ht = [t for t in toks if t in hum]
idx = np.array([toks.index(t) for t in ht])
out = {}
for at in ATTRS:
    H = np.array([hum[t][at] for t in ht]); q = QUERY[at]
    d = (E @ enc.sig2_t([q])[0]).cpu().numpy(); b = bm.score(q)
    out[at] = {"dense": ap_of(d[idx], H), "bm25": ap_of(b[idx], H), "sum": ap_of((zs(d) + zs(b))[idx], H),
               "prevalence": float(H.mean())}
KW = {"droplets": r"droplet|raindrop", "glare": r"glare|flare|bloom", "wet_road": r"\bwet\b|reflect|puddle",
      "low_contrast": r"low visibility|poor visibility|dark|dim"}
for at, rx in KW.items():
    out[at]["caption_mentions_frac"] = float(np.mean([bool(re.search(rx, c.lower())) for c in docs]))
    out[at]["caption_negated_mentions_frac"] = float(np.mean(
        [bool(re.search(r"\b(no|not|without|free of|absence of)\b[^.]{0,60}(" + rx + ")", c.lower())) for c in docs]))
for k, v in out.items(): print(k, {a: round(b, 3) for a, b in v.items()})
json.dump({"doc": "CARIM-style diagnostic on the 550 human-GT scenes: dense vs BM25 parts, and how often captions "
                  "mention the attribute only to negate it", "attributes": out},
          open("results/cvpr/milestone23_carim_diagnostic.json", "w"), indent=1)
