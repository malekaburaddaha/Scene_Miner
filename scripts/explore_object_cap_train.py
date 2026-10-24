#!/usr/bin/env python3
"""MILESTONE 32 - should gamma_deg be capped when a degradation query also names
something outside the degradation/context vocabulary? Decided on TRAINING text only.

Rule under test: gamma_deg' = min(gamma_deg, 0.5) when detector(q) > theta, so the
SigLIP2 cosine keeps at least half the influence on "object x degradation" queries.
The cap value 0.5 is fixed (task specification), not tuned.

Training-only stand-in for "something else": the 12 generic out-of-vocabulary phrases
of the bank ("a toll booth", "a parking lot", ...). Synthetic conjunctions join a
training degradation phrasing with one generic phrase; when one is scored, that
generic phrase is removed from the bank (no self-match). No evaluation query, scene or
score is read.

Detectors (unprojected, mean-centred SigLIP2 text space, tau = 0.02 class scores):
  D1  best generic-class score minus best concept-class score (leave-out as above)
  D2  1 - squared norm of the query's projection onto the span of the 4 concept and
      7 context class means (the share of the query outside the degradation/context
      vocabulary)
Pre-declared rule (fixed before running): theta = 99th percentile of the detector over
the 156 training degradation phrasings scored leave-group-out (<= 1% wrongly capped);
pick the detector with the higher recall on the synthetic conjunctions; adopt the cap
only if that recall is >= 0.80. Also reported: the flag rate on the 240 training
queries (the cap is applied at inference to the v2.6 checkpoints, which is consistent
only if training queries are essentially never flagged).
-> results/cvpr/milestone32_object_cap_train.json
"""
import json, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_semantic_edge_cases import text_embed
from scene_mining.filter.sceneminer_v2_4 import build_bank

TAU = 0.02
tr_all = json.load(open("benchmark_cvpr/queries_390.json"))["train"]
texts, names, cls, is_c = build_bank(tr_all)
cls = np.asarray(cls)
deg_tr = [q for q in tr_all if q["stratum"] == "sensor_degradation"]
atoms = [q["text"] for q in deg_tr if len(q["predicate"]) == 1]


def base_of(t):
    m = [a for a in atoms if t.startswith(a)]
    return max(m, key=len) if m else t


group = np.array([base_of(t) if c < 4 else f"__{i}" for i, (t, c) in enumerate(zip(texts, cls))])
gen_ids = [k for k, n in enumerate(names) if n.startswith("oov:")]
mac_ids = [k for k, n in enumerate(names) if n.startswith("macro:")]
gen_text = {k: texts[int(np.where(cls == k)[0][0])] for k in gen_ids}
rng = np.random.default_rng(0)
deg_atoms = [i for i in range(len(texts)) if cls[i] < 4 and texts[i] in atoms]
syn, syn_gen = [], []
templates = ["{d} near {g}", "{d} at {g}", "{g} with {d}"]
for k in gen_ids:
    for j, i in enumerate(rng.choice(deg_atoms, 10, replace=False)):
        g = gen_text[k]; g_ = g[0].lower() + g[1:]
        syn.append(templates[j % 3].format(d=texts[i], g=g_)); syn_gen.append(k)
train_q = [q["text"] for q in tr_all]
E = F.normalize(text_embed(texts + syn + train_q).float().cpu(), dim=-1).double()
EB, ES, ET = E[:len(texts)], E[len(texts):len(texts) + len(syn)], E[len(texts) + len(syn):]
mu = F.normalize(EB.mean(0), dim=-1)
centre = lambda X: F.normalize(X - mu, dim=-1)
B0, S0, T0 = centre(EB), centre(ES), centre(ET)
means = torch.stack([B0[torch.as_tensor(cls == k)].mean(0) for k in list(range(4)) + mac_ids])
Q_, _ = torch.linalg.qr(means.T)


def class_scores(q, m):
    c = (B0 @ q).numpy() / TAU; S = np.full(len(names), -np.inf)
    for k in range(len(names)):
        mk = m & (cls == k)
        if mk.any(): S[k] = TAU * (np.logaddexp.reduce(c[mk]) - np.log(mk.sum()))
    return S


def D1(q, m):
    S = class_scores(q, m)
    g = [S[k] for k in gen_ids if np.isfinite(S[k])]
    return max(g) - max(S[:4])


def D2(q):
    return float(1 - ((q @ Q_) ** 2).sum())


allm = np.ones(len(texts), bool)
pos = [i for i in range(len(texts)) if cls[i] < 4]
res = {}
for name, f in [("D1", "d1"), ("D2", "d2")]:
    if name == "D1":
        p = np.array([D1(B0[i], group != group[i]) for i in pos])
        s = np.array([D1(S0[j], allm & (cls != syn_gen[j])) for j in range(len(syn))])
        t = np.array([D1(T0[j], allm) for j in range(len(train_q))])
    else:
        p = np.array([D2(B0[i]) for i in pos]); s = np.array([D2(S0[j]) for j in range(len(syn))])
        t = np.array([D2(T0[j]) for j in range(len(train_q))])
    theta = float(np.percentile(p, 99))
    res[name] = {"theta": theta, "false_flag_degradation": float((p > theta).mean()),
                 "recall_synthetic": float((s > theta).mean()), "flag_rate_training_queries": float((t > theta).mean()),
                 "pos_quantiles": [float(x) for x in np.percentile(p, [5, 50, 95, 99])],
                 "syn_quantiles": [float(x) for x in np.percentile(s, [5, 25, 50, 95])]}
    print(name, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in res[name].items()})
best = max(res, key=lambda k: res[k]["recall_synthetic"])
adopt = res[best]["recall_synthetic"] >= 0.80
out = {"doc": __doc__, "detectors": res, "selected": best, "adopt": adopt,
       "synthetic_examples": syn[:6], "n_synthetic": len(syn)}
print("SELECTED", best, "ADOPT" if adopt else "NOT ADOPTED (recall < 0.80)")
json.dump(out, open("results/cvpr/milestone32_object_cap_train.json", "w"), indent=1)
