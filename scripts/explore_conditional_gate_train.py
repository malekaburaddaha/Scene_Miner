#!/usr/bin/env python3
"""MILESTONE 27 - conditional context removal for gamma_deg, chosen on TRAINING text only.

v2.5 removes half of the macro-context subspace from EVERY query before computing
gamma_deg. A query that names no context at all (an out-of-vocabulary phrase such
as "snow covering the road") is then projected too, which can move its gamma_deg.
The variants below remove context directions only when the query activates a
context class, judged on the UNPROJECTED class scores S_k (the same soft-max class
scores the gate uses):

  uncond        v2.5: remove lam * span(all 7 macro means) from every query
  cond_all      remove the whole subspace only if some macro class beats the best
                generic (OOV) class: max_m S_m > max_g S_g
  cond_sub      remove only the span of the ACTIVE macro classes (S_m > max_g S_g)

Protocol and selection rule are those of Milestone 22 (explore_factored_gate_train.py):
positives are the 156 degradation phrasings scored leave-GROUP-out, negatives the
84 macro + 12 generic phrasings (leave-one-out) and 60 synthetic macro x macro
conjunctions. Rule, fixed before running: per variant and lambda, the smoothest tau
whose separation stays >= 0.99; among those, maximise the share of positives with
gamma_deg > 0.9 subject to every negative (bank and synthetic) <= 0.2; ties go to
the lower maximum negative. No evaluation query, scene or label is read.
"""
import json, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_semantic_edge_cases import text_embed
from scene_mining.filter.sceneminer_v2_4 import build_bank

tr = json.load(open("benchmark_cvpr/queries_390.json"))["train"]
texts, names, cls, is_c = build_bank(tr)
cls = np.asarray(cls)
deg_tr = [q for q in tr if q["stratum"] == "sensor_degradation"]
atoms = [q["text"] for q in deg_tr if len(q["predicate"]) == 1]


def base_of(t):
    m = [a for a in atoms if t.startswith(a)]
    return max(m, key=len) if m else t


group = np.array([base_of(t) if c < 4 else f"__{i}" for i, (t, c) in enumerate(zip(texts, cls))])
macro_q = [q for q in tr if q["stratum"] == "macro_environmental"]
rng = np.random.default_rng(0); syn = []
keys = sorted({list(q["predicate"])[0] for q in macro_q})
for _ in range(60):                       # identical draw to Milestone 22
    k1, k2 = rng.choice(keys, 2, replace=False)
    a = rng.choice([q["text"] for q in macro_q if list(q["predicate"])[0] == k1])
    b = rng.choice([q["text"] for q in macro_q if list(q["predicate"])[0] == k2])
    syn.append(f"{a} with {b}")
E = F.normalize(text_embed(texts + syn).float().cpu(), dim=-1).double()
EB, ES = E[:len(texts)], E[len(texts):]
mu = F.normalize(EB.mean(0), dim=-1)
centre = lambda X: F.normalize(X - mu, dim=-1)
B0, S0 = centre(EB), centre(ES)
n_cls = len(names)
macro_ids = [k for k, n in enumerate(names) if n.startswith("macro:")]
oov_ids = [k for k, n in enumerate(names) if n.startswith("oov:")]
Mmeans = torch.stack([B0[torch.as_tensor(cls == k)].mean(0) for k in macro_ids])     # (7, d)


def class_scores(q, Bk, tau, m):
    """soft-max class scores of one query against the bank rows allowed by m."""
    c = (Bk @ q).numpy() / tau
    S = np.full(n_cls, -np.inf)
    for k in range(n_cls):
        mk = m & (cls == k)
        if mk.any(): S[k] = tau * (np.logaddexp.reduce(c[mk]) - np.log(mk.sum()))
    return S


def gamma_from(S, tau):
    z = S / tau; z = z - z[np.isfinite(z)].max(); p = np.exp(z); p[~np.isfinite(S)] = 0; p /= p.sum()
    return p[:4].sum()


def proj_with(basis, X, lam):
    if basis is None or lam == 0: return X
    return F.normalize(X - lam * (X @ basis) @ basis.T, dim=-1)


def gamma_one(q, variant, lam, tau, m):
    S_un = class_scores(q, B0, tau, m)
    if variant == "uncond":
        act = list(range(len(macro_ids)))
    else:
        ref = max(S_un[k] for k in oov_ids)
        act = [j for j, k in enumerate(macro_ids) if S_un[k] > ref]
    if not act:
        return gamma_from(S_un, tau), 0
    if variant == "cond_sub":
        basis, _ = torch.linalg.qr(Mmeans[act].T)
    else:
        basis, _ = torch.linalg.qr(Mmeans.T)
    qp = proj_with(basis, q[None], lam)[0]; Bp = proj_with(basis, B0, lam)
    return gamma_from(class_scores(qp, Bp, tau, m), tau), len(act)


pos = np.where(cls < 4)[0]; neg = np.where(cls >= 4)[0]
comp = np.array([len(q["predicate"]) > 1 for q in deg_tr])
allm = np.ones(len(texts), bool)
res = {}
for variant in ["uncond", "cond_all", "cond_sub"]:
    for lam in [0.5, 0.75, 1.0]:
        for tau in [0.01, 0.02, 0.03, 0.05]:
            g = np.array([gamma_one(B0[i], variant, lam, tau, group != group[i])[0] for i in range(len(texts))])
            ga = [gamma_one(S0[i], variant, lam, tau, allm) for i in range(len(syn))]
            gs = np.array([x[0] for x in ga])
            gp, gn = g[pos], g[neg]
            r = {"pos_atomic_min": float(gp[~comp].min()), "pos_compound_p10": float(np.percentile(gp[comp], 10)),
                 "pos_compound_med": float(np.median(gp[comp])), "pos_frac_gt_0.9": float((gp > 0.9).mean()),
                 "neg_p95": float(np.percentile(gn, 95)), "neg_max": float(gn.max()),
                 "synthetic_p95": float(np.percentile(gs, 95)), "synthetic_max": float(gs.max())}
            r["sep"] = float(np.percentile(gp, 5) - max(np.percentile(gn, 95), np.percentile(gs, 95)))
            res[f"{variant}|{lam}|{tau}"] = r
            print(f"{variant:<9} lam {lam:<4} tau {tau:<5} pos>0.9 {r['pos_frac_gt_0.9']:.2f} comp p10 "
                  f"{r['pos_compound_p10']:.3f} | neg max {r['neg_max']:.3f} syn max {r['synthetic_max']:.3f} "
                  f"| sep {r['sep']:+.3f}", flush=True)

# the pre-declared selection rule
cands = []
for v in ["uncond", "cond_all", "cond_sub"]:
    for lam in [0.5, 0.75, 1.0]:
        ok = [t for t in [0.01, 0.02, 0.03, 0.05] if res[f"{v}|{lam}|{t}"]["sep"] >= 0.99]
        if not ok: continue
        t = max(ok); r = res[f"{v}|{lam}|{t}"]
        if max(r["neg_max"], r["synthetic_max"]) <= 0.2:
            cands.append((r["pos_frac_gt_0.9"], -max(r["neg_max"], r["synthetic_max"]), v, lam, t))
cands.sort(reverse=True)
sel = {"variant": cands[0][2], "lam": cands[0][3], "tau": cands[0][4]} if cands else None
print("candidates (pos>0.9, -max neg, variant, lam, tau):", cands[:6])
print("SELECTED:", sel)
json.dump({"doc": __doc__, "results": res, "candidates": cands, "selected": sel},
          open("results/cvpr/milestone27_conditional_gate_train.json", "w"), indent=1)
