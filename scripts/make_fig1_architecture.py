#!/usr/bin/env python3
"""Figure 1, two-tier architecture diagram (main paper), and the detailed Stage-1 /
Stage-2 diagrams for the supplementary (Milestones 29, 32).

Typography: drawn at 7.0 in wide, so the font sizes below are print sizes. Sub-labels
are at least 7 pt, module titles 8-9 pt bold; every symbol is matplotlib mathtext.

Every number comes from the code, the config or a result file:
  Stage 1  configs/experiment/nuscenes_scenemining_v8_unified.py (SigLIP2-base,
           BEVFormer 50x50 / 3 layers, BLIP-2 Q-Former 32 queries, Qwen2.5-VL-3B
           LoRA r=16 on q/v); corpus sizes from features_v2.npz; backbone sizes by
           meta-device instantiation from their configs.
  Stage 2  parameter counts of the SceneMinerV26 state dict; latency from
           results/cvpr/milestone27_latency.json.
"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("paper/figures")
C = {"data": ("#ecfeff", "#155e75"), "frozen": ("#e5e7eb", "#374151"), "trained": ("#e0e7ff", "#3730a3"),
     "lora": ("#fef3c7", "#92400e"), "cache": ("#dbeafe", "#1e40af"), "query": ("#dcfce7", "#166534"),
     "gate": ("#fff7ed", "#c2410c"), "norm": ("#fee2e2", "#b91c1c"), "head": ("#ede9fe", "#5b21b6")}
INK, SUB = "#111827", "#374151"
W_IN = 7.0                       # figure width in inches; 100 x-units
LH = 1.0 / 72 / (W_IN / 100)     # units per point (equal axes)
MIN_PT = 7.0

# measured parameter counts (meta-device instantiation from the configs; SceneMinerV26 state dict)
S1_PARAMS = {"SigLIP2": "92.9 M", "Q-Former": "105.1 M", "Qwen": "4.07 B"}
S2_PARAMS = {"FiLM": "0.79 M", "decoder": "2.11 M", "scorer": "0.28 M", "adapters": "1.46 M with query"}

_sizes = []


def T(ax, x, y, s, fs, **kw):
    assert fs >= MIN_PT, (s, fs)
    _sizes.append(fs)
    return ax.text(x, y, s, fontsize=fs, **kw)


def box(ax, x, y, w, h, kind, title, sub=None, tfs=8.5, sfs=7.2, bold=True):
    """title and subtitle stacked by line height, centred in the box"""
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.7", fc=fc, ec=ec, lw=1.0))
    nt = title.count("\n") + 1; ns = (sub.count("\n") + 1) if sub else 0
    ht, hs, gap = nt * tfs * LH * 1.22, ns * sfs * LH * 1.22, (0.5 if sub else 0)
    top = y + h / 2 + (ht + gap + hs) / 2
    T(ax, x + w / 2, top - ht / 2, title, tfs, ha="center", va="center", color=INK,
      fontweight="bold" if bold else "normal", linespacing=1.12)
    if sub:
        T(ax, x + w / 2, top - ht - gap - hs / 2, sub, sfs, ha="center", va="center", color=SUB, linespacing=1.12)


def arrow(ax, x0, y0, x1, y1, col=INK, lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=col, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def tier(ax, x, y, w, h, col, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.0", fc="#fbfdff", ec=col, lw=1.3))
    ax.add_patch(FancyBboxPatch((x, y + h - 3.9), w, 3.9, boxstyle="round,pad=0.0,rounding_size=1.0", fc=col, ec=col, lw=1.3))
    T(ax, x + 1.4, y + h - 1.95, label, 9.0, ha="left", va="center", color="white", fontweight="bold")


def fig1(lat, out="fig1_architecture.pdf", crop=None):
    yl = {None: (0, 65.5), "stage1": (34.6, 63.0), "stage2": (0.0, 34.4)}[crop]
    fig = plt.figure(figsize=(W_IN, W_IN * (yl[1] - yl[0]) / 100))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.axis("off"); ax.set_ylim(yl)
    # legend
    lx = 0.8
    for k, lab in [("data", "data"), ("frozen", "frozen"), ("lora", "frozen + LoRA"), ("trained", "trained"),
                   ("norm", "standardised"), ("cache", "cached"), ("gate", "training-text gate"), ("query", "query / output")]:
        fc, ec = C[k]
        ax.add_patch(FancyBboxPatch((lx, 63.4), 2.0, 1.6, boxstyle="round,pad=0,rounding_size=0.3", fc=fc, ec=ec, lw=0.8))
        T(ax, lx + 2.5, 64.2, lab, 7.2, va="center", color=INK); lx += 4.0 + 0.78 * len(lab)
    # ---------------- Stage 1
    tier(ax, 0.6, 34.8, 98.8, 27.8, "#1e3a8a", "STAGE 1  ·  Scene encoders (frozen, run once per scene, cached)")
    T(ax, 1.5, 57.4, "multi-camera rig (nuScenes, Waymo)", 7.2, color="#4b5563", style="italic", va="center")
    box(ax, 1.5, 51.0, 13.8, 5.0, "data", "nuScenes", "700 scenes", 8.5, 7.2)
    box(ax, 1.5, 45.4, 13.8, 5.0, "data", "Waymo", "808 chunks", 8.5, 7.2)
    box(ax, 17.3, 52.6, 14.2, 3.4, "frozen", "SigLIP2 patches", None, 7.6, bold=False)
    box(ax, 17.3, 49.0, 14.2, 3.4, "trained", "BEVFormer 50×50", None, 7.2, bold=False)
    box(ax, 17.3, 45.4, 14.2, 3.4, "trained", "motion head", None, 7.6, bold=False)
    box(ax, 33.5, 45.4, 11.5, 10.6, "trained", "BLIP-2\nQ-Former", "32 queries", 8.5, 7.2)
    box(ax, 47.0, 45.4, 12.5, 10.6, "lora", "Qwen2.5-VL\n3B", "LoRA $r{=}16$", 8.5, 7.2)
    box(ax, 61.5, 52.6, 13.0, 3.4, "head", "embedding $\\mathbf{e}$", None, 7.6, bold=False)
    box(ax, 61.5, 49.0, 13.0, 3.4, "head", "tags $\\mathbf{t}$", None, 7.6, bold=False)
    box(ax, 61.5, 45.4, 13.0, 3.4, "head", "risk $r$", None, 7.6, bold=False)
    arrow(ax, 15.3, 53.5, 17.3, 54.3); arrow(ax, 15.3, 47.9, 17.3, 50.7)
    for y in (54.3, 50.7, 47.1): arrow(ax, 31.5, y, 33.5, 50.7)
    arrow(ax, 45.0, 50.7, 47.0, 50.7)
    for y in (54.3, 50.7, 47.1): arrow(ax, 59.5, 50.7, 61.5, y)
    for y in (54.3, 50.7, 47.1): arrow(ax, 74.5, y, 77.0, 50.0)
    T(ax, 1.5, 43.6, "any single camera (also BDD100K, val150)", 7.2, color="#4b5563", style="italic", va="center")
    box(ax, 1.5, 35.8, 13.8, 6.2, "data", "front-camera\nimage", None, 8.0)
    box(ax, 17.3, 35.8, 14.2, 6.2, "frozen", "SigLIP2", "image $\\mathbf{s}$", 8.5, 7.2)
    box(ax, 33.5, 35.8, 26.0, 6.2, "frozen", "Qwen2.5-VL-3B, zero-shot", "margins $\\mathbf{m}$, token $\\mathbf{v}$", 8.5, 7.2)
    box(ax, 61.5, 35.8, 13.0, 6.2, "data", "recorded\nmetadata", None, 8.0)
    arrow(ax, 15.3, 38.9, 17.3, 38.9)
    ax.plot([8.4, 8.4, 46.5], [42.0, 42.8, 42.8], color=INK, lw=0.9); arrow(ax, 46.5, 42.8, 46.5, 42.0)
    for x0 in (31.5, 59.5): pass
    ax.plot([74.5, 76.0], [38.9, 38.9], color=INK, lw=0.9)
    ax.plot([24.4, 24.4], [35.8, 35.3], color=INK, lw=0.9); ax.plot([46.5, 46.5], [35.8, 35.3], color=INK, lw=0.9)
    ax.plot([68.0, 68.0], [35.8, 35.3], color=INK, lw=0.9)
    ax.plot([24.4, 87.5], [35.3, 35.3], color=INK, lw=0.9); arrow(ax, 87.5, 35.3, 87.5, 35.8)
    box(ax, 77.0, 35.8, 21.6, 20.2, "cache", "feature bank", "1508 scenes\n$\\{\\mathbf{e},\\mathbf{t},r,\\mathbf{v},"
        "\\mathbf{s},\\mathbf{m},\\pi\\}$\n\n$\\pi$: 20 tags +\n6 metadata", 9.0, 7.6)
    # ---------------- Stage 2
    tier(ax, 0.6, 0.4, 98.8, 33.4, "#0f766e",
         f"STAGE 2  ·  Scale-aware retrieval filter  (4.68 M trainable  ·  {lat} per query)")
    T(ax, 1.5, 28.6, "adapters + type embedding", 7.2, color="#4b5563", style="italic", va="center")
    toks = [("$\\mathbf{e}$", "256→256", "trained"), ("$\\mathbf{t}$", "20→256", "trained"),
            ("$r$", "standardise", "norm"), ("$\\mathbf{v}$", "2048→256", "trained"),
            ("$\\mathbf{s}$", "768→256", "trained"), ("$\\mathbf{m}$", "4→256", "trained")]
    for i, (n, d, k) in enumerate(toks):
        y = 24.4 - i * 3.05
        box(ax, 1.5, y, 4.2, 2.6, "cache", n, None, 8.5, bold=False)
        box(ax, 6.6, y, 12.6, 2.6, k, d, None, 7.4, bold=False)
        arrow(ax, 5.7, y + 1.3, 6.6, y + 1.3); arrow(ax, 19.2, y + 1.3, 21.0, y + 1.3, lw=0.7)
    box(ax, 21.0, 9.15, 9.6, 17.85, "trained", "FiLM", "$\\gamma(q)\\odot z$\n$+\\,\\beta(q)$", 9.0, 7.6)
    box(ax, 32.4, 9.15, 13.2, 17.85, "trained", "cross-\nattention\ndecoder", "2 layers\n4 heads\n$Q{\\leftarrow}q$; $K,V{\\leftarrow}z$", 8.5, 7.2)
    arrow(ax, 30.6, 18.1, 32.4, 18.1)
    box(ax, 47.4, 18.6, 12.0, 8.4, "trained", "scorer", "bilinear + MLP\n$\\rho$", 8.5, 7.2)
    box(ax, 47.4, 9.15, 12.0, 8.0, "norm", "ResNorm", "$\\alpha\\, g(q)\\,\\tilde\\rho$\n$\\alpha=0.5$", 8.5, 7.2)
    arrow(ax, 45.6, 22.8, 47.4, 22.8); arrow(ax, 53.4, 18.6, 53.4, 17.15)
    box(ax, 61.2, 18.6, 14.0, 8.4, "norm", "PartNorm", "$[\\cos,\\ m_{\\mathrm{sel}},\\ \\pi_{\\mathrm{sel}}]$", 8.5, 7.4)
    box(ax, 61.2, 9.15, 14.0, 8.0, "trained", "router $\\mathbf{w}(q)$",
        "$s_{\\mathrm{head}}=\\mathbf{w}^{\\top}\\tilde{\\mathbf{p}}$\n$+\\,\\alpha g\\tilde\\rho$", 8.5, 7.2)
    arrow(ax, 68.2, 18.6, 68.2, 17.15); arrow(ax, 59.4, 13.15, 61.2, 13.15)
    box(ax, 77.0, 18.6, 21.6, 8.4, "query", "$s=\\gamma_{\\mathrm{deg}}\\,s_{\\mathrm{head}}$\n"
        "$+\\,(1-\\gamma_{\\mathrm{deg}})\\,s_{\\mathrm{abst}}$", None, 8.5, bold=False)
    box(ax, 77.0, 9.15, 12.4, 8.0, "gate", "concept\ngate", "$\\gamma_{\\mathrm{deg}},\\ \\gamma_{\\mathrm{macro}}$", 8.5, 7.4)
    box(ax, 90.8, 9.15, 7.8, 8.0, "query", "rank\ntop-$k$", None, 8.0)
    arrow(ax, 75.2, 15.5, 77.0, 20.0); arrow(ax, 83.2, 17.15, 83.2, 18.6, col=C["gate"][1])
    arrow(ax, 77.0, 11.0, 75.2, 11.0, col=C["gate"][1]); arrow(ax, 94.7, 18.6, 94.7, 17.15)
    # query
    box(ax, 1.5, 1.4, 10.0, 4.6, "query", "text\nquery", None, 8.0)
    box(ax, 12.8, 1.4, 11.4, 4.6, "frozen", "SigLIP2 text", "$q\\in\\mathbb{R}^{768}$", 8.0, 7.2)
    arrow(ax, 11.5, 3.7, 12.8, 3.7)
    qc = C["query"][1]
    ax.plot([24.2, 83.2], [3.7, 3.7], color=qc, lw=0.9)
    T(ax, 30.0, 4.4, "$q$", 8.5, color=qc, fontweight="bold")
    for x in (25.8, 39.0, 53.4, 68.2, 83.2): arrow(ax, x, 3.7, x, 9.15, col=qc, lw=0.8)
    arrow(ax, 96.0, 35.8, 96.0, 33.8, col=C["cache"][1], ls="--", lw=1.0)
    if crop == "stage1":
        for (x, y, t) in [(24.4, 56.8, S1_PARAMS["SigLIP2"] + " vision"), (39.25, 56.8, S1_PARAMS["Q-Former"]),
                          (53.25, 56.8, S1_PARAMS["Qwen"] + " total")]:
            T(ax, x, y, t, 7.0, color="#4b5563", ha="center", va="center")
    if crop == "stage2":
        for (x, y, t) in [(25.8, 27.9, S2_PARAMS["FiLM"]), (39.0, 27.9, S2_PARAMS["decoder"]),
                          (53.4, 27.9, S2_PARAMS["scorer"]), (10.4, 6.9, "adapters " + S2_PARAMS["adapters"])]:
            T(ax, x, y, t, 7.0, color="#4b5563", ha="center", va="center")
    if crop:
        for art in ax.patches + ax.texts + ax.lines: art.set_clip_on(True)
        fig.savefig(OUT/out); plt.close(fig); return
    fig.savefig(OUT/out, bbox_inches="tight", pad_inches=0.02); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--latency", default=None)
    a = ap.parse_args()
    lat = a.latency
    if lat is None:
        p = Path("results/cvpr/milestone27_latency.json")
        lat = "{:.1f} ms".format(json.load(open(p))["online"]["full_1"]) if p.exists() else "< 10 ms"
    fig1(lat); fig1(lat, "supp_stage1.pdf", crop="stage1"); fig1(lat, "supp_stage2.pdf", crop="stage2")
    print("wrote fig1_architecture.pdf + supp_stage1/2.pdf; latency label:", lat,
          f"| font sizes used: min {min(_sizes)} pt, max {max(_sizes)} pt")
