#!/usr/bin/env python3
"""Corrected label-entropy null, erank(L).

BUG (as shipped): L was built multi-hot over the ~11 ATOMIC binary attributes,
so rank(L) <= 11 and erank(L) <= 11 by construction. That can never reach the
47.6 reported in Sec. 4.3 -- the ceiling is arithmetic, not empirical.

FIX: the supervision the representation is actually trained to express is the
QUERY-PREDICATE relevance matrix: one column per query, L[i,j] = 1 iff scene i
satisfies query j's predicate. Its rank is bounded by the number of queries,
not by the number of atomic attributes, and it is what the contrastive
objective consumes. Both constructions are reported so the ceiling is explicit.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

def erank(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False); s = s[s > eps]
    if s.size == 0: return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + eps)).sum()))

def atomic_L(attrs, toks, keys):
    cols = []
    for k in keys:
        vals = sorted({attrs.get(t, {}).get(k) for t in toks} - {None}, key=str)
        if len(vals) <= 1: continue
        if set(vals) <= {True, False}: cols.append((k, True))
        else: cols += [(k, v) for v in vals]
    L = np.array([[1.0 if attrs.get(t, {}).get(k) == v else 0.0 for k, v in cols]
                  for t in toks])
    return L, cols

def predicate_L(attrs, toks, queries):
    """One column per query predicate - the supervision actually trained on."""
    cols = []
    for q, pred in queries:
        col = np.array([all(attrs.get(t, {}).get(k) == v for k, v in pred.items())
                        for t in toks], dtype=float)
        if 0 < col.sum() < len(toks): cols.append((q, col))
    L = np.stack([c for _, c in cols], 1) if cols else np.zeros((len(toks), 0))
    return L, [q for q, _ in cols]

if __name__ == "__main__":
    from eval_attribute_index import load_attributes, MACRO8, DEGRAD
    W = Path("work_dirs/unified_filter")
    z = np.load(W/"features_overnight.npz", allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]
    attrs = load_attributes(W)
    for f in ("binary_flags_with_logits.json", "lowcontrast_strict.json"):
        p = W/f
        if p.exists():
            for t, r in json.load(open(p)).items():
                attrs.setdefault(t, {}).update(
                    {k: v for k, v in r.items() if not k.endswith(("_margin","_logp_yes"))})
    keys = ["lighting","weather","density","droplets","glare","wet_road",
            "low_contrast","low_contrast_strict"]
    La, ca = atomic_L(attrs, toks, keys)
    queries = DEGRAD + MACRO8
    Lp, cp = predicate_L(attrs, toks, queries)
    cfg_a = len(np.unique(La, axis=0)); cfg_p = len(np.unique(Lp, axis=0))
    out = {
      "atomic": {"n_cols": len(ca), "rank_ceiling": len(ca),
                 "erank_L": erank(La), "n_distinct_configs": int(cfg_a)},
      "predicate": {"n_cols": len(cp), "rank_ceiling": len(cp),
                    "erank_L": erank(Lp), "n_distinct_configs": int(cfg_p),
                    "matrix_rank": int(np.linalg.matrix_rank(Lp - Lp.mean(0)))},
    }
    Path("results/cvpr").mkdir(parents=True, exist_ok=True)
    Path("results/cvpr/milestone1_label_null.json").write_text(json.dumps(out, indent=2))
    print("=== corrected label-entropy null ===")
    print(f"  ATOMIC    : {len(ca):3d} cols -> erank(L) = {out['atomic']['erank_L']:6.2f} "
          f"(ceiling {len(ca)}, {cfg_a} distinct configs)")
    print(f"  PREDICATE : {len(cp):3d} cols -> erank(L) = {out['predicate']['erank_L']:6.2f} "
          f"(ceiling {len(cp)}, rank {out['predicate']['matrix_rank']}, {cfg_p} distinct configs)")
    print("\n  The atomic construction can never exceed its column count; Sec. 4.3's")
    print("  47.6 is only reachable on a predicate/query matrix with >=48 columns.")
