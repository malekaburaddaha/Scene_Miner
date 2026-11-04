#!/usr/bin/env python3
"""MILESTONE 39 - what the concept gate decides for example queries (main-paper table).

For each query: gamma_deg, the dominant class of the unprojected bank posterior, the
context classes it activates and the margin column it selects, from the final v2.6
checkpoints (the gate is identical across seeds: same training bank, same tau; the
script checks this). Queries are taken verbatim from the evaluation suite, the
BDD100K queries, Stratum 5 and the main text.
-> results/cvpr/milestone39_gate_examples.json and paper/tab_gate.tex
"""
import json, sys
from pathlib import Path
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import eval_semantic_edge_cases as E
from scene_mining.filter.sceneminer_v2_4 import build_bank

QUERIES = [
    ("water droplets on the lens at night", "degradation + context"),
    ("slick pavement in the dark", "degradation + context"),
    ("blinding headlight bloom from oncoming traffic", "degradation"),
    ("snow covering the road", "out of vocabulary"),
    ("driving after dark", "context"),
    ("bright daytime driving", "context"),
    ("a vehicle at night", "object + context"),
    ("pedestrians crossing the road at night", "object + context"),
    ("a bus at a stop", "object"),
]
ATTR = ["droplets", "glare", "wet road", "low contrast"]


def main():
    import numpy as np
    risk = np.load("work_dirs/unified_filter/features_v2.npz")["risk"]
    M = E.load_v23(risk.reshape(-1), which="v26")
    _, names, _, _ = build_bank(json.load(open("benchmark_cvpr/queries_390.json"))["train"])
    QT = E.text_embed([q for q, _ in QUERIES])
    out = []
    with torch.no_grad():
        G = torch.stack([m.concept(QT)[1] for m in M])                      # (seeds, Q)
        assert float((G - G[0]).abs().max()) < 1e-5, "gate differs across seeds"
        m = M[0]
        a, g, _ = m.concept(QT)
        act = m.active_context(QT)
        P = torch.softmax(m._class_scores(m._centre(QT), m.bank) / m.tau_g, -1)
    for i, (q, kind) in enumerate(QUERIES):
        top = int(P[i].argmax())
        ctx = [names[int(k)].split("=")[-1] for k, on in zip(m.macro_ids, act[i]) if on]
        out.append({"query": q, "kind": kind, "gamma_deg": float(g[i]), "top_class": names[top],
                    "top_class_p": float(P[i, top]), "active_context": ctx,
                    "margin_column": ATTR[int(a[i].argmax())], "margin_weight": float(a[i].max())})
        print(f"{q:50s} gamma={float(g[i]):.3f} top={names[top]} ctx={ctx}")
    Path("results/cvpr/milestone39_gate_examples.json").write_text(json.dumps({"doc": __doc__, "rows": out}, indent=1))
    L = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lccl@{}}", "\\toprule", "query & context & $\\gdeg$ & ranked by\\\\", "\\midrule"]
    for r in out:
        ctx = ", ".join(r["active_context"]) or "--"
        mode = f"head: {r['margin_column']}" if r["gamma_deg"] >= 0.5 else "SigLIP2" + (" $+$ meta" if r["active_context"] else "")
        L.append(f"``{r['query']}'' & {ctx} & {r['gamma_deg']:.3f} & {mode}\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tab_gate.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
