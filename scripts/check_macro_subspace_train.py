#!/usr/bin/env python3
"""MILESTONE 23 - should gamma_macro be computed inside the context subspace?
Training text only. Targets:
  * the 96 compound degradation phrasings whose predicate names a context
    (lighting/weather/density) - gamma_macro should put its mass on THAT class;
  * the 84 context phrasings (leave-one-out) - mass on their own class;
  * the 60 atomic degradation phrasings and 12 generic phrases - no context named,
    so little context mass.
Variants: 'full' (current, unprojected) and 'subspace' (query and bank
projected onto the span of the 7 context-class means before scoring)."""
import json, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_semantic_edge_cases import text_embed
from scene_mining.filter.sceneminer_v2_4 import build_bank

tr = json.load(open("benchmark_cvpr/queries_390.json"))["train"]
texts, names, cls, is_c = build_bank(tr)
deg = [q for q in tr if q["stratum"] == "sensor_degradation"]
E = F.normalize(text_embed(texts).float().cpu(), dim=-1)
B = F.normalize(E - F.normalize(E.mean(0), dim=-1), dim=-1)
macro_ids = [k for k, n in enumerate(names) if n.startswith("macro:")]
M = torch.stack([B[torch.as_tensor(cls == k)].mean(0) for k in macro_ids]); Q, _ = torch.linalg.qr(M.T)
def ctx_of(pred):
    for k, v in pred.items():
        if k in ("lighting", "weather", "density"): return f"macro:{k}={v}"
    return None
target = []
for i, (t, c) in enumerate(zip(texts, cls)):
    if c < 4:
        q = next(q for q in deg if q["text"] == t); target.append(ctx_of(q["predicate"]))
    elif names[c].startswith("macro:"): target.append(names[c])
    else: target.append(None)
res = {}
for variant in ["full", "subspace"]:
    X = (B @ Q) @ Q.T if variant == "subspace" else B
    X = F.normalize(X, dim=-1)
    for tau in [0.01, 0.02, 0.03, 0.05]:
        C = (X @ X.T).numpy(); n = len(texts); mass_ok, acc, spurious = [], [], []
        for i in range(n):
            mask = np.ones(n, bool); mask[i] = False                 # leave-one-out
            S = np.full(len(names), -np.inf)
            for k in range(len(names)):
                mk = mask & (cls == k)
                if mk.any(): S[k] = tau * (np.logaddexp.reduce(C[i, mk] / tau) - np.log(mk.sum()))
            z = S / tau; z -= z[np.isfinite(z)].max(); p = np.exp(z); p[~np.isfinite(S)] = 0; p /= p.sum()
            pm = {names[k]: p[k] for k in macro_ids}
            if target[i] is not None:
                acc.append(max(pm, key=pm.get) == target[i]); mass_ok.append(pm.get(target[i], 0))
            else:
                spurious.append(sum(pm.values()))
        res[f"{variant}|{tau}"] = {"ctx_argmax_acc": float(np.mean(acc)), "ctx_mass_median": float(np.median(mass_ok)),
                                   "ctx_mass_p10": float(np.percentile(mass_ok, 10)),
                                   "no_ctx_mass_p90": float(np.percentile(spurious, 90))}
        r = res[f"{variant}|{tau}"]
        print(f"{variant:<9} tau {tau:<5} context argmax acc {r['ctx_argmax_acc']:.3f}  mass median {r['ctx_mass_median']:.3f} "
              f"p10 {r['ctx_mass_p10']:.3f} | spurious mass p90 (no context named) {r['no_ctx_mass_p90']:.3f}")
json.dump(res, open("results/cvpr/milestone23_macro_subspace_train.json", "w"), indent=1)
