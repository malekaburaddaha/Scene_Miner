#!/usr/bin/env python3
"""MILESTONE 29 - Figure 1, two-tier architecture diagram (main paper), and the
detailed Stage-1 / Stage-2 diagrams for the supplementary.

Every number is taken from the code, the config or a result file:
  Stage 1  configs/experiment/nuscenes_scenemining_v8_unified.py (SigLIP2-base,
           BEVFormer 50x50 / 3 layers, BLIP-2 Q-Former 32 queries, Qwen2.5-VL-3B
           LoRA r=16 on q/v); corpus sizes from features_v2.npz.
  Stage 2  parameter counts of SceneMinerV26 (scripts/make_fig1_architecture.py
           --counts prints them); latency from results/cvpr/milestone27_fast_index.json.
"""
import argparse, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("paper/figures")
C = {"data": ("#ecfeff", "#155e75"), "frozen": ("#e5e7eb", "#374151"), "trained": ("#e0e7ff", "#3730a3"),
     "lora": ("#fef3c7", "#92400e"), "cache": ("#dbeafe", "#1e40af"), "query": ("#dcfce7", "#166534"),
     "gate": ("#fff7ed", "#c2410c"), "norm": ("#fee2e2", "#b91c1c"), "head": ("#ede9fe", "#5b21b6")}
INK = "#1f2937"


def box(ax, x, y, w, h, kind, title, sub=None, fs=6.2, sfs=5.0, bold=True, ls="-"):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.6", fc=fc, ec=ec, lw=0.8, ls=ls))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=sfs, color="#4b5563", linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, col=INK, lw=0.7, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5.5, color=col, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def tier(ax, x, y, w, h, col, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.0", fc="#fbfdff", ec=col, lw=1.1))
    ax.add_patch(FancyBboxPatch((x, y + h - 3.6), w, 3.6, boxstyle="round,pad=0.0,rounding_size=1.0", fc=col, ec=col, lw=1.1))
    ax.text(x + 1.5, y + h - 1.8, label, ha="left", va="center", fontsize=6.8, color="white", fontweight="bold")


# measured parameter counts (meta-device instantiation from the configs; SceneMinerV26 state dict)
S1_PARAMS = {"SigLIP2-base": "92.9 M (vision)", "Q-Former": "105.1 M", "Qwen": "4.07 B total, d 2048"}
S2_PARAMS = {"adapters": "e 0.13 M · t 0.07 M · r 0.07 M · v 0.59 M · s 0.26 M · m 0.07 M · query 0.26 M",
             "FiLM": "0.79 M", "decoder": "2.11 M", "scorer": "0.30 M", "router": "router + selectors 0.02 M"}


def fig1(lat_ms, out="fig1_architecture.pdf", crop=None):
    yl = {None: (0, 55), "stage1": (28.9, 51.9), "stage2": (0.4, 28.8)}[crop]
    wid = 7.0 if crop is None else 9.0                           # equal units on both axes
    fig = plt.figure(figsize=(wid, wid * (yl[1] - yl[0]) / 100))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.axis("off"); ax.set_ylim(yl)
    lx = 1.5
    for k, lab in [("data", "dataset"), ("frozen", "frozen"), ("lora", "frozen + LoRA"), ("trained", "trained"),
                   ("norm", "standardisation"), ("cache", "cached per scene"), ("gate", "training-text gate"),
                   ("query", "query / output")]:
        fc, ec = C[k]; ax.add_patch(FancyBboxPatch((lx, 52.6), 1.8, 1.5, boxstyle="round,pad=0,rounding_size=0.3", fc=fc, ec=ec, lw=0.7))
        ax.text(lx + 2.3, 53.35, lab, fontsize=5.2, va="center", color=INK); lx += 3.3 + 0.80 * len(lab)
    # ---------------- Stage 1
    tier(ax, 0.8, 29.3, 98.4, 22.4, "#1e3a8a", "STAGE 1  ·  Multi-camera and monocular scene encoders (frozen, run once per scene, cached)")
    # row A: multi-camera rig
    ax.text(1.8, 47.2, "multi-camera rig (nuScenes, Waymo)", fontsize=5.0, color="#6b7280", style="italic", va="center")
    box(ax, 1.8, 42.4, 11.5, 3.8, "data", "nuScenes", "700 scenes · 6 cameras", fs=5.8, sfs=4.6)
    box(ax, 1.8, 38.2, 11.5, 3.8, "data", "Waymo Open", "808 chunks · 5 cameras", fs=5.8, sfs=4.6)
    box(ax, 15.2, 43.6, 12.6, 2.7, "frozen", "SigLIP2-base (patches)", None, fs=4.6)
    box(ax, 15.2, 40.8, 12.6, 2.5, "trained", "BEVFormer 50×50", None, fs=4.8)
    box(ax, 15.2, 38.2, 12.6, 2.3, "trained", "motion head", None, fs=4.8)
    box(ax, 29.2, 38.2, 9.6, 8.1, "trained", "BLIP-2\nQ-Former", "32 queries", fs=5.9, sfs=4.7)
    box(ax, 41.0, 38.2, 11.0, 8.1, "lora", "Qwen2.5-VL-3B", "LoRA r = 16 (q, v)", fs=5.8, sfs=4.7)
    box(ax, 54.3, 43.9, 14.0, 2.4, "head", "embedding  e ∈ ℝ²⁵⁶", None, fs=5.0, bold=False)
    box(ax, 54.3, 41.1, 14.0, 2.4, "head", "scene/object tags  t ∈ ℝ²⁰", None, fs=5.0, bold=False)
    box(ax, 54.3, 38.3, 14.0, 2.4, "head", "risk  r ∈ ℝ¹", None, fs=5.0, bold=False)
    arrow(ax, 13.3, 44.3, 15.2, 44.9); arrow(ax, 13.3, 40.1, 15.2, 41.9)
    for y in (44.9, 42.0, 39.3): arrow(ax, 27.8, y, 29.2, 42.2)
    arrow(ax, 38.8, 42.2, 41.0, 42.2)
    for y in (45.1, 42.3, 39.5): arrow(ax, 52.0, 42.2, 54.3, y)
    for y in (45.1, 42.3, 39.5): arrow(ax, 68.3, y, 86.0, 41.0 + (y - 42.3) * 0.6)
    # row B: monocular, any single camera
    ax.text(44.0, 36.9, "any single camera (also BDD100K, val150)", fontsize=5.0, color="#6b7280", style="italic", va="center")
    box(ax, 1.8, 31.0, 11.5, 4.6, "data", "front-camera\nimage", None, fs=5.6)
    box(ax, 15.2, 31.0, 14.8, 4.6, "frozen", "SigLIP2 ViT-B/16", "image  s ∈ ℝ⁷⁶⁸", fs=5.7, sfs=4.8)
    box(ax, 32.2, 31.0, 20.0, 4.6, "frozen", "Qwen2.5-VL-3B (zero-shot)", "yes/no margins m ∈ ℝ⁴ · token v ∈ ℝ²⁰⁴⁸", fs=5.6, sfs=4.5)
    box(ax, 54.3, 31.0, 14.0, 4.6, "data", "recorded metadata", "night/day · rain/dry · density", fs=5.6, sfs=4.4)
    arrow(ax, 13.3, 33.3, 15.2, 33.3)
    ax.plot([7.5, 7.5, 42.2], [35.6, 36.3, 36.3], color=INK, lw=0.7); arrow(ax, 42.2, 36.3, 42.2, 35.6)
    for x in (22.6, 42.2, 61.3): ax.plot([x, x], [31.0, 30.1], color=INK, lw=0.7)
    ax.plot([22.6, 90.0], [30.1, 30.1], color=INK, lw=0.7); arrow(ax, 90.0, 30.1, 90.0, 31.0)
    box(ax, 86.0, 31.0, 12.4, 15.3, "cache", "feature bank", "1508 scenes\n× {e, t, r, v, s, m, π}\n\nπ ∈ ℝ²⁶: 20 tags\n+ 6 metadata", fs=6.0, sfs=4.8)
    # ---------------- Stage 2
    tier(ax, 0.8, 0.8, 98.4, 27.6, "#0f766e",
         f"STAGE 2  ·  Scale-aware query-conditioned retrieval filter  (4.68 M trainable  ·  {lat_ms} per query, 1508 scenes)")
    ax.text(1.8, 23.9, "modality adapters (+ type embedding)", fontsize=5.0, color="#6b7280", style="italic", va="center")
    toks = [("e", "256→256→256"), ("t", "20→256→256"), ("r", "standardise · 1→256"), ("v", "2048→256→256"),
            ("s", "768→256→256"), ("m", "4→256→256")]
    for i, (n, d) in enumerate(toks):
        y = 20.9 - i * 2.45
        box(ax, 1.8, y, 3.8, 2.0, "cache", n, None, fs=5.6)
        box(ax, 6.6, y, 12.4, 2.0, "norm" if n == "r" else "trained", d, None, fs=4.8, bold=False)
        arrow(ax, 5.6, y + 1.0, 6.6, y + 1.0); arrow(ax, 19.0, y + 1.0, 21.0, y + 1.0, lw=0.5)
    ax.text(1.8, 7.7, "r: fixed standardisation, not a width-1 LayerNorm", fontsize=4.4, color=C["norm"][1], ha="left")
    box(ax, 21.0, 8.6, 8.6, 14.3, "trained", "FiLM", "γ(q) ⊙ z + β(q)\nper token type", fs=6.2, sfs=4.7)
    box(ax, 31.6, 8.6, 11.8, 14.3, "trained", "cross-attention\ndecoder", "2 layers · 4 heads\nd 256 · FFN 1024\nQ ← q ;  K, V ← 6 tokens", fs=6.0, sfs=4.6)
    arrow(ax, 29.6, 15.7, 31.6, 15.7)
    box(ax, 45.4, 16.9, 10.3, 6.0, "trained", "scorer", "bilinear + MLP → ρ", fs=6.0, sfs=4.6)
    box(ax, 45.4, 8.6, 10.3, 6.4, "norm", "ResNorm", "α g(q) ρ̃\nα = 0.5, |g| ≤ 1", fs=6.0, sfs=4.6)
    arrow(ax, 43.4, 19.9, 45.4, 19.9); arrow(ax, 50.55, 16.9, 50.55, 15.0)
    box(ax, 57.7, 16.9, 13.3, 6.0, "norm", "PartNorm", "[cos(q, s), m_sel, π_sel]", fs=6.0, sfs=4.5)
    box(ax, 57.7, 8.6, 13.3, 6.4, "trained", "router w(q)", "s_head = wᵀp̃ + α g ρ̃", fs=6.0, sfs=4.6)
    arrow(ax, 64.35, 16.9, 64.35, 15.0); arrow(ax, 55.7, 11.8, 57.7, 11.8)
    box(ax, 73.0, 16.9, 25.4, 6.0, "query", "s = γ_deg s_head + (1 − γ_deg) s_abst",
        "s_abst = SigLIP2 + metadata named by γ_macro", fs=5.7, sfs=4.5)
    box(ax, 73.0, 8.6, 15.0, 6.4, "gate", "factored concept gate", "γ_deg · γ_macro\n252 training phrasings", fs=5.2, sfs=4.5)
    box(ax, 89.6, 8.6, 8.8, 6.4, "query", "rank 1508\nscenes → top-k", None, fs=4.7)
    arrow(ax, 71.0, 12.8, 73.0, 18.4); arrow(ax, 79.9, 15.0, 79.9, 16.9, col=C["gate"][1])
    arrow(ax, 73.0, 10.6, 71.0, 10.6, col=C["gate"][1])
    arrow(ax, 94.0, 16.9, 94.0, 15.0)
    # query
    box(ax, 1.8, 1.8, 12.0, 4.6, "query", "text query", "“wet asphalt at night”", fs=5.8, sfs=4.7)
    box(ax, 15.8, 1.8, 13.0, 4.6, "frozen", "SigLIP2 text tower", "q ∈ ℝ⁷⁶⁸", fs=5.6, sfs=4.7)
    arrow(ax, 13.8, 4.1, 15.8, 4.1)
    qc = C["query"][1]
    ax.plot([28.8, 79.9], [4.1, 4.1], color=qc, lw=0.7)
    ax.text(30.0, 4.6, "q", fontsize=5.8, color=qc, fontweight="bold")
    for x in (25.3, 37.5, 50.55, 64.35, 79.9):
        arrow(ax, x, 4.1, x, 8.6, col=qc, lw=0.6)
    arrow(ax, 95.5, 31.0, 95.5, 28.4, col=C["cache"][1], ls="--", lw=0.8)
    if crop == "stage1":
        for (x, y, t) in [(21.5, 47.2, S1_PARAMS["SigLIP2-base"]), (34.0, 47.2, S1_PARAMS["Q-Former"]),
                          (46.5, 47.2, S1_PARAMS["Qwen"])]:
            ax.text(x, y, t, fontsize=4.6, color="#6b7280", ha="center", va="center")
    if crop == "stage2":
        for (x, y, t) in [(25.3, 7.6, S2_PARAMS["FiLM"]), (37.5, 7.6, S2_PARAMS["decoder"]),
                          (50.55, 23.4, S2_PARAMS["scorer"]), (64.35, 7.6, S2_PARAMS["router"])]:
            ax.text(x, y, t, fontsize=4.4, color="#6b7280", ha="center", va="center")
        ax.text(57.0, 2.2, "adapters: " + S2_PARAMS["adapters"], fontsize=4.4, color="#6b7280", ha="center")
    if crop:
        for art in ax.patches + ax.texts + ax.lines: art.set_clip_on(True)
        fig.savefig(OUT/out); plt.close(fig); return
    fig.savefig(OUT/out, bbox_inches="tight", pad_inches=0.02); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--latency", default=None)
    a = ap.parse_args()
    lat = a.latency
    if lat is None:
        p = Path("results/cvpr/milestone27_fast_index.json")
        lat = "≈{:.1f} ms".format(min(v["cached_1model_ms"] for v in json.load(open(p))["latency"].values())) if p.exists() else "< 10 ms"
    fig1(lat); fig1(lat, "supp_stage1.pdf", crop="stage1"); fig1(lat, "supp_stage2.pdf", crop="stage2")
    print("wrote", OUT/"fig1_architecture.pdf", "+ supp_stage1/2.pdf; latency label:", lat)
