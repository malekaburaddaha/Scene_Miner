#!/usr/bin/env python3
"""MILESTONE 22 - choose the degradation-gate variant on TRAINING text only.

Positives: the 156 sensor-degradation phrasings of the 390-query train split,
scored LEAVE-GROUP-OUT (a base phrasing and all its "... in dense traffic /
on an empty road / in daylight" compounds are removed together), so a compound
cannot be recognised merely because its siblings are in the bank.
Negatives: the 84 macro phrasings and 12 generic phrases (leave-one-out), plus
synthetic macro x macro conjunctions joined from two training macro phrasings of
different predicate type ("<a> with <b>").
Variants:
  joint     - the v2.4 gate: one softmax over 4 concepts + 7 macro + 12 OOV
  project   - remove the macro subspace (span of the 7 macro class means) from
              query and bank, then the same softmax on the residuals
  project_l - partial removal, lambda in {0.5, 0.75}
"""
import json, sys, itertools
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_semantic_edge_cases import text_embed
from scene_mining.filter.sceneminer_v2_4 import build_bank

tr = json.load(open("benchmark_cvpr/queries_390.json"))["train"]
texts, names, cls, is_c = build_bank(tr)
deg_tr = [q for q in tr if q["stratum"] == "sensor_degradation"]
atoms = [q["text"] for q in deg_tr if len(q["predicate"]) == 1]
def base_of(t):
    m = [a for a in atoms if t.startswith(a)]
    return max(m, key=len) if m else t
group = np.array([base_of(t) if c < 4 else f"__{i}" for i, (t, c) in enumerate(zip(texts, cls))])
macro_q = [q for q in tr if q["stratum"] == "macro_environmental"]
rng = np.random.default_rng(0); syn = []
keys = sorted({list(q["predicate"])[0] for q in macro_q})
for _ in range(60):
    k1, k2 = rng.choice(keys, 2, replace=False)
    a = rng.choice([q["text"] for q in macro_q if list(q["predicate"])[0] == k1])
    b = rng.choice([q["text"] for q in macro_q if list(q["predicate"])[0] == k2])
    syn.append(f"{a} with {b}")
E = F.normalize(text_embed(texts + syn).float().cpu(), dim=-1)
EB, ES = E[:len(texts)], E[len(texts):]
mu = F.normalize(EB.mean(0), dim=-1)
centre = lambda X: F.normalize(X - mu, dim=-1)
B0, S0 = centre(EB), centre(ES)
n_cls = len(names)
macro_ids = [k for k, n in enumerate(names) if n.startswith("macro:")]
Mmeans = torch.stack([B0[torch.as_tensor(cls == k)].mean(0) for k in macro_ids])
Q_, _ = torch.linalg.qr(Mmeans.T)                     # (d, 7) orthonormal basis

def proj(X, lam):
    return F.normalize(X - lam * (X @ Q_) @ Q_.T, dim=-1) if lam > 0 else X

def gamma(Qe, Be, tau, excl=None):
    C = (Qe @ Be.T).numpy(); out = np.zeros(len(Qe))
    for i in range(len(Qe)):
        m = np.ones(len(Be), bool)
        if excl is not None: m &= excl[i]
        S = np.full(n_cls, -np.inf)
        for k in range(n_cls):
            mk = m & (cls == k)
            if mk.any():
                x = C[i, mk] / tau; S[k] = tau * (np.logaddexp.reduce(x) - np.log(mk.sum()))
        z = S / tau; z -= z[np.isfinite(z)].max(); p = np.exp(z); p[~np.isfinite(S)] = 0; p /= p.sum()
        out[i] = p[:4].sum()
    return out

res = {}
pos = np.where(cls < 4)[0]; neg = np.where(cls >= 4)[0]
comp = np.array([len(q["predicate"]) > 1 for q in deg_tr])
for name, lam in [("joint", 0.0), ("project_0.5", 0.5), ("project_0.75", 0.75), ("project", 1.0)]:
    Bp, Sp = proj(B0, lam), proj(S0, lam)
    for tau in [0.01, 0.02, 0.03, 0.05]:
        exg = np.array([[group[j] != group[i] for j in range(len(texts))] for i in range(len(texts))])
        g = gamma(Bp, Bp, tau, excl=exg)
        gs = gamma(Sp, Bp, tau)
        gp, gn = g[pos], g[neg]
        r = {"pos_atomic_min": float(gp[~comp].min()), "pos_atomic_med": float(np.median(gp[~comp])),
             "pos_compound_p10": float(np.percentile(gp[comp], 10)), "pos_compound_med": float(np.median(gp[comp])),
             "pos_frac_gt_0.9": float((gp > 0.9).mean()),
             "neg_p95": float(np.percentile(gn, 95)), "neg_max": float(gn.max()),
             "synthetic_macroxmacro_p95": float(np.percentile(gs, 95)), "synthetic_max": float(gs.max())}
        r["sep"] = float(np.percentile(gp, 5) - max(np.percentile(gn, 95), np.percentile(gs, 95)))
        res[f"{name}|{tau}"] = r
        print(f"{name:<13} tau {tau:<5} pos>0.9 {r['pos_frac_gt_0.9']:.2f} compound p10 {r['pos_compound_p10']:.3f} "
              f"med {r['pos_compound_med']:.3f} | neg p95 {r['neg_p95']:.3f} max {r['neg_max']:.3f} | "
              f"syn p95 {r['synthetic_macroxmacro_p95']:.3f} max {r['synthetic_max']:.3f} | sep {r['sep']:+.3f}")
json.dump({"doc": __doc__, "results": res, "synthetic_examples": syn[:8]},
          open("results/cvpr/milestone22_gate_variants_train.json", "w"), indent=1)
