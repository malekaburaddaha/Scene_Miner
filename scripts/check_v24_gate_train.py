#!/usr/bin/env python3
"""MILESTONE 18 - choose tau and verify the v2.4 concept gate on TRAINING text only.

Every phrasing of the 390-query TRAIN split is scored leave-one-out (removed
from its own class before scoring). Nothing from any evaluation set is read.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from scene_mining.filter.sceneminer_v2_4 import SceneMinerV24, build_bank, TRAIN_ORDER
from eval_semantic_edge_cases import text_embed

tr = json.load(open("benchmark_cvpr/queries_390.json"))["train"]
texts, names, cls, is_c = build_bank(tr)
E = text_embed(texts).cpu()
m = SceneMinerV24()
tau, rep = m.set_bank(E, cls, is_c, mean_bank=E)
print(f"bank: {len(texts)} phrasings, {len(names)} classes ({is_c.sum()} concepts)")
print(f"{'tau':>6}{'pos p05':>9}{'pos min':>9}{'pos med':>9}{'neg p95':>9}{'neg max':>9}{'neg med':>9}{'sep':>8}")
for t, r in rep.items():
    print(f"{t:>6}{r['pos_p05']:>9.3f}{r['pos_min']:>9.3f}{r['pos_median']:>9.3f}{r['neg_p95']:>9.3f}"
          f"{r['neg_max']:>9.3f}{r['neg_median']:>9.3f}{r['sep']:>8.3f}{'  <- chosen' if t == tau else ''}")
# per-group LOO gamma at the chosen tau
from scene_mining.filter.sceneminer_v2_4 import class_scores_np, gamma_np
B = m._centre(E); C = (B @ B.T).numpy()
g = gamma_np(class_scores_np(C, cls, len(names), tau, exclude=np.arange(len(texts))), is_c, tau)
out = {"tau": tau, "grid": rep, "per_group": {}}
print("\nLOO gamma by group at chosen tau:")
for k, nm in enumerate(names):
    sel = cls == k
    if nm.startswith("oov:"): continue
    gg = g[sel]
    out["per_group"][nm] = {"n": int(sel.sum()), "min": float(gg.min()), "median": float(np.median(gg)),
                            "frac_gt_0.90": float((gg > 0.9).mean()), "frac_lt_0.10": float((gg < 0.1).mean())}
    print(f"  {nm:<28} n={sel.sum():>3} min {gg.min():.3f} median {np.median(gg):.3f} "
          f">0.9: {(gg>0.9).mean():.2f}  <0.1: {(gg<0.1).mean():.2f}")
oo = g[np.array([n.startswith('oov:') for n in np.array(names)[cls]])]
out["per_group"]["oov_bank"] = {"n": len(oo), "max": float(oo.max()), "median": float(np.median(oo))}
print(f"  {'OOV bank (12)':<28} max {oo.max():.3f} median {np.median(oo):.3f}")
deg = [i for i, q in enumerate([q for q in tr if q['stratum']=='sensor_degradation'])]
comp = np.array([len(q["predicate"]) > 1 for q in tr if q["stratum"] == "sensor_degradation"])
gd = g[cls < 4]
print(f"\ndegradation phrasings: atomic (n={(~comp).sum()}) min {gd[~comp].min():.3f} median {np.median(gd[~comp]):.3f}; "
      f"compound (n={comp.sum()}) min {gd[comp].min():.3f} median {np.median(gd[comp]):.3f}")
print("lowest-gamma degradation phrasings:")
dt = [t for t, c in zip(texts, cls) if c < 4]
for i in np.argsort(gd)[:6]: print(f"   {gd[i]:.3f}  {dt[i]}")
mt = [t for t, c in zip(texts, cls) if 4 <= c and not names[c].startswith('oov:')]
gm = g[[i for i, c in enumerate(cls) if 4 <= c and not names[c].startswith('oov:')]]
print("highest-gamma macro phrasings:")
for i in np.argsort(-gm)[:6]: print(f"   {gm[i]:.3f}  {mt[i]}")
out["atomic_degr"] = {"min": float(gd[~comp].min()), "median": float(np.median(gd[~comp]))}
out["compound_degr"] = {"min": float(gd[comp].min()), "median": float(np.median(gd[comp]))}
json.dump(out, open("results/cvpr/milestone18_gate_train_check.json", "w"), indent=1)
