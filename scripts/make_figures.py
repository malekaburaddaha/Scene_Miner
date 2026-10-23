#!/usr/bin/env python3
"""MILESTONE 12 - the three paper figures. Every number is read from the result
JSONs written by Milestones 1-11; nothing here is hard-coded by hand."""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
C1, C2, C3, CG = "#2b6cb0", "#c05621", "#2f855a", "#4a5568"
OUT = Path("paper/figures")


# ---------------------------------------------------------------- figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, txt, fc, ec, fs=7.0, bold=False, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=0.8, linestyle=ls))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x1, y1, x2, y2, c=CG, lw=0.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=c,
                                     lw=lw, mutation_scale=7, shrinkA=0, shrinkB=0))

    ax.text(1, 43.4, "Stage 1 — frozen, cached once", fontsize=8, fontweight="bold", color=CG)
    ax.text(55.5, 43.4, "Stage 2 — SceneMiner v2.3, 4.68 M trainable", fontsize=8, fontweight="bold", color=C1)
    ax.plot([53, 53], [1, 42], color="#cbd5e0", lw=0.8, ls=(0, (3, 3)))

    box(1, 31.5, 22, 8, "BEVFormer 50$\\times$50\n+ MotionHead", "#ebf8ff", C1, fs=6.8)
    box(1, 21.8, 22, 8, "BLIP-2 Q-Former\n(20 scene tags)", "#ebf8ff", C1, fs=6.8)
    box(1, 12.1, 22, 8, "Qwen2.5-VL-3B\n(4 degradation margins)", "#ebf8ff", C1, fs=6.5)
    box(1, 2.4, 22, 8, "SigLIP2 ViT-B/16\n(image + text towers)", "#ebf8ff", C1, fs=6.8)

    box(27, 14, 20, 20,
        "cached signals\n\n$\\mathbf{e}$  BEV embed (256)\n$\\mathbf{t}$  tags (20)\n"
        "$r$  risk (1)\n$\\mathbf{v}$  visual tok (2048)\n$\\mathbf{m}$  margins (4)",
        "#f7fafc", CG, fs=6.6)
    for y in (35.5, 25.8, 16.1, 6.4): arrow(23, y, 27, 24)

    box(56, 33.5, 19, 7.5, "query text $q$\n(SigLIP2 text tower)", "#fffaf0", C2, fs=6.8)
    box(78.5, 33.5, 20.5, 7.5,
        "concept gate $\\gamma(q)$\nmargin column by text\nalignment; off when OOV", "#fff5f5", C2,
        fs=6.0, ls=(0, (2.5, 1.8)))

    box(56, 20, 19, 10, "PartNorm\nbase parts $\\mathbf{p}(q)$\n[cos, margin, prior]",
        "#f0fff4", C3, fs=6.6)
    box(80, 20, 19, 10, "router $\\mathbf{w}(q)$\nmargin weight $\\times\\gamma(q)$\n"
        "renorm. on availability", "#f0fff4", C3, fs=6.1)
    box(56, 10, 43, 7,
        "cross-attn. fusion $\\rightarrow\\rho$; ResNorm bounds it: $\\alpha\\,g(q)\\,\\tilde\\rho$, $\\alpha=0.5$",
        "#f0fff4", C3, fs=6.6)
    box(56, 1.5, 43, 6.5,
        "$s(q,i)=\\mathbf{w}(q)^{\\!\\top}\\mathbf{p}(q,i)+\\alpha\\,g(q)\\,\\tilde\\rho(q,i)$\n"
        "$g\\!=\\!0$ at init $\\Rightarrow\\; s\\equiv s_{\\mathrm{base}}$",
        "#ffffff", C3, fs=6.8, bold=True)

    arrow(65.5, 33.5, 65.5, 30)
    arrow(88.8, 33.5, 88.8, 30, c=C2)
    arrow(75, 25, 80, 25)
    arrow(47, 24, 56, 25)
    arrow(65.5, 20, 65.5, 17); arrow(89.5, 20, 89.5, 17)
    arrow(77.5, 10, 77.5, 8)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT/"fig1_architecture.pdf", bbox_inches="tight")
    plt.close(fig); print("  fig1_architecture.pdf")


# ---------------------------------------------------------------- figure 3
def fig3():
    suite = json.load(open("results/cvpr/milestone10_query_suite_150.json"))["overall"]
    val = json.load(open("results/cvpr/bevformer_val_per_scene_official.json"))
    trn = json.load(open("results/cvpr/bevformer_ours278_per_scene_official.json"))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    suite = dict(suite)
    suite["SceneMiner_v2.3"] = json.load(open("results/cvpr/milestone14_v2_3.json"))["C_suite150"]["overall"]["v2.3_full"]
    order = ["attribute_index(TAUTOLOGY)", "SceneMiner_v2.1", "SceneMiner_v2.3",
             "zeroshot_SigLIP2", "logit_VLM_lookup"]
    lab = ["attribute\nindex", "SceneMiner\nv2.1", "SceneMiner\nv2.3",
           "zero-shot\nSigLIP2", "VLM margin\nlookup"]
    vals = [suite[k]["mAP"] for k in order]
    cols = ["#c53030", C1, C1, CG, CG]
    axa.bar(range(len(vals)), vals, color=cols, width=0.6, zorder=3)
    for i, v in enumerate(vals):
        axa.text(i, v + 0.022, f"{v:.3f}", ha="center", fontsize=7,
                 fontweight="bold" if i == 0 else "normal")
    axa.axhline(1.0, color="#c53030", lw=0.7, ls=":", zorder=2)
    axa.set_xticks(range(len(vals))); axa.set_xticklabels(lab, fontsize=6.4)
    axa.set_ylabel("mAP, 150-query suite"); axa.set_ylim(0, 1.3)
    axa.grid(axis="y", lw=0.4, color="#e2e8f0", zorder=0)
    axa.set_title("(a) Benchmark circularity", fontsize=8, fontweight="bold", pad=14)
    axa.text(-0.45, 1.19, "relevance $=$ a predicate over the\nsame cached attributes "
             "$\\Rightarrow$ the index\nscores 1.000 by construction",
             fontsize=6.0, color="#c53030", ha="left", va="center", linespacing=1.4)
    for sp in ("top", "right"): axa.spines[sp].set_visible(False)

    mv = np.array([v["mAP"] for v in val.values() if v["mAP"] == v["mAP"]])
    mt = np.array([v["mAP"] for v in trn.values() if v["mAP"] == v["mAP"]])
    bins = np.linspace(0, 1, 26)
    axb.hist(mt, bins=bins, color="#c53030", alpha=0.72, zorder=3,
             label=f"train split, seen by\nthe detector ($n$={len(mt)})")
    axb.hist(mv, bins=bins, color=C1, alpha=0.72, zorder=3,
             label=f"nuScenes val, unseen\n($n$={len(mv)})")
    axb.axvline(mt.mean(), color="#c53030", lw=1.1, zorder=4)
    axb.axvline(mv.mean(), color=C1, lw=1.1, zorder=4)
    axb.annotate("", xy=(mt.mean(), 68), xytext=(mv.mean(), 68),
                 arrowprops=dict(arrowstyle="<|-|>", color="k", lw=0.8, mutation_scale=7))
    axb.text(mv.mean() + 0.018, 72.5,
             f"$+${mt.mean()-mv.mean():.4f} mAP\n($+${100*(mt.mean()-mv.mean())/mv.mean():.0f}%)",
             ha="left", va="bottom", fontsize=6.6, fontweight="bold")
    axb.set_xlabel("per-scene mAP (official nuScenes protocol)")
    axb.set_ylabel("scenes"); axb.set_ylim(0, 82); axb.set_xlim(0, 1)
    axb.legend(fontsize=5.9, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.52),
               handlelength=1.1, labelspacing=0.7, borderpad=0.2)
    axb.grid(axis="y", lw=0.4, color="#e2e8f0", zorder=0)
    axb.set_title("(b) Detector training-set contamination", fontsize=8,
                  fontweight="bold", pad=14)
    for sp in ("top", "right"): axb.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT/"fig3_audit.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3_audit.pdf  (contamination +{mt.mean()-mv.mean():.4f} mAP)")
    return {"tautology_mAP": vals[0], "contamination_delta": float(mt.mean()-mv.mean()),
            "train_mAP": float(mt.mean()), "val_mAP": float(mv.mean())}


# ---------------------------------------------------------------- figure 2
QUERIES = ["water droplets smearing the camera lens",
           "wet reflective asphalt after rainfall",
           "a low contrast scene where vehicles are barely distinguishable from the background"]
SHORT = ["\"water droplets smearing the camera lens\"",
         "\"wet reflective asphalt after rainfall\"",
         "\"a low contrast scene, vehicles barely distinguishable\""]


def fig2(args):
    import torch, torch.nn.functional as F
    from PIL import Image
    dev = torch.device("cuda")
    z = np.load("work_dirs/unified_filter/features_v2.npz", allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]
    dom = z["domain"].astype(int)
    evset = set(json.load(open("benchmark_cvpr/split_906_602.json"))["eval"])
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    emb, tlog, risk = T("emb"), T("tag_logits"), T("risk")
    sig, mz, prior = T("siglip_img"), T("vlm_margin_z"), T("prior")
    qwen = torch.zeros(len(corpus), 2048, device=dev)
    vc = Path("work_dirs/unified_filter/V_frozen_cache.pt")
    if vc.exists(): qwen = torch.load(vc, map_location=dev).float()
    rk = risk.reshape(-1).cpu().numpy()

    from transformers import AutoModel, GemmaTokenizerFast
    P = "google/siglip2-base-patch16-224"
    sm = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    tkz = GemmaTokenizerFast.from_pretrained(P)
    ft = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    with torch.no_grad():
        ti = tkz(QUERIES, padding="max_length", truncation=True, max_length=64,
                 return_tensors="pt").to(dev)
        QT = ft(sm.get_text_features(**ti))
    del sm; torch.cuda.empty_cache()

    from eval_semantic_edge_cases import load_v23
    M = load_v23(rk)

    # BDD100K signals, with the margin columns in TRAINING order
    broot = Path("/nfs/turbo/umd-srawa/scene_mining/datasets/external_adverse/bdd100k")
    bz = np.load(broot/"signals.npz", allow_pickle=False)
    bnames = sorted(json.load(open(broot/"labels.json")))
    ASKED = ["wet_road", "glare", "droplets", "low_contrast"]
    TRAIN_ORDER = ["droplets", "glare", "wet_road", "low_contrast"]
    BM = bz["margin"][:, [ASKED.index(k) for k in TRAIN_ORDER]]
    BMZ = (BM - BM.mean(0, keepdims=True)) / (BM.std(0, keepdims=True) + 1e-6)
    BS = torch.from_numpy(bz["siglip"]).float().to(dev)
    BMZt = torch.from_numpy(BMZ).float().to(dev)
    BPR = torch.zeros(len(bnames), 26, device=dev)
    nb = len(bnames)
    av2 = torch.stack([torch.ones(nb, device=dev), torch.ones(nb, device=dev),
                       torch.zeros(nb, device=dev)], -1)
    on = torch.ones(len(corpus), device=dev)

    import textwrap
    n_in, n_bdd = 3, 2
    fig, axes = plt.subplots(len(QUERIES), n_in + n_bdd,
                             figsize=(7.2, 1.08 * len(QUERIES) + 0.5))
    fig.subplots_adjust(left=0.205, right=0.995, top=0.895, bottom=0.015,
                        wspace=0.05, hspace=0.17)
    thumbs = Path("work_dirs/unified_filter/scene_thumbs")
    for r, q in enumerate(QUERIES):
        qv = QT[r].unsqueeze(0).expand(len(corpus), -1)
        with torch.no_grad():
            s = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=on)
                         .cpu().numpy() for m in M], 0)
            qb = QT[r].unsqueeze(0).expand(nb, -1)
            sb = np.mean([m.base_score(qb, BS, BMZt, BPR, av2).cpu().numpy() for m in M], 0)
        # in-domain: eval split only, one scene per segment, thumb must exist
        cand = [i for i in np.argsort(-s)
                if corpus[i] in evset and (thumbs/f"{corpus[i]}.jpg").exists()]
        seen, pick = set(), []
        for i in cand:
            seg = corpus[i].split("#c")[0]
            if seg in seen: continue
            seen.add(seg); pick.append(i)
            if len(pick) == n_in: break
        for c, i in enumerate(pick):
            axes[r, c].imshow(Image.open(thumbs/f"{corpus[i]}.jpg").convert("RGB"))
            axes[r, c].set_title(f"{'nuScenes' if dom[i]==0 else 'Waymo'}  rank {c+1}",
                                 fontsize=5.8, color=C1, pad=1.6)
            axes[r, c].set_anchor("N")
        for c, i in enumerate(np.argsort(-sb)[:n_bdd]):
            ax = axes[r, n_in + c]
            ax.imshow(Image.open(broot/"images"/bnames[i]).convert("RGB"))
            ax.set_anchor("N")
            ax.set_title(f"BDD100K  rank {c+1}", fontsize=5.8, color=C2, pad=1.6)
        for c in range(n_in + n_bdd):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_color(C1 if c < n_in else C2); sp.set_lw(0.9)
        axes[r, 0].set_ylabel(textwrap.fill(SHORT[r], 26), fontsize=6.0, labelpad=5,
                              rotation=0, ha="right", va="center", style="italic",
                              linespacing=1.4)
    fig.suptitle("SceneMiner v2.3 top retrievals — trained on nuScenes $+$ Waymo, "
                 "applied unchanged to monocular BDD100K (external)",
                 fontsize=7.6, y=0.985)
    fig.savefig(OUT/"fig2_qualitative.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig); print("  fig2_qualitative.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fig2", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("[figures]")
    fig1(); meta = fig3()
    if not args.skip_fig2: fig2(args)
    Path("results/cvpr/milestone12_figure_numbers.json").write_text(json.dumps(meta, indent=2))
    print(f"[figures] wrote {OUT}")


# ---------------------------------------------------------------- val150 quartiles
# Categorical slots 1-4 of the validated reference palette, fixed order; random
# is a neutral dashed reference. Contrast WARN on aqua/yellow is relieved by a
# distinct marker per series and direct end labels (identity never colour-only).
SER = [("SceneMiner_v2.3", "SceneMiner v2.3", "#2a78d6", "o"),
       ("zeroshot_SigLIP2", "zero-shot SigLIP2", "#eb6834", "s"),
       ("logit_VLM_lookup", "VLM margin lookup", "#1baf7a", "^"),
       ("metadata_matched", "night/rain metadata", "#eda100", "D")]


def quartile_panels(axes, d, kind="degradation", label_end=True):
    Q = d["quartiles"][kind]
    for ax, m, yl in zip(axes, ["mAP_front", "fp_rate"],
                         ["CAM_FRONT-sector mAP", "ghost FP rate (score > 0.3)"]):
        x = np.arange(1, 5)
        ax.plot(x, Q["random"][m], color="#8a8a85", lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.text(4.12, Q["random"][m][-1], "random", fontsize=6.2, color="#55554f", va="center")
        for key, lab, col, mk in SER:
            y = Q[key][m]
            ax.plot(x, y, color=col, lw=2.0, marker=mk, ms=4.6, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(["Q1\n(top-scored)", "Q2", "Q3", "Q4"], fontsize=6.6)
        ax.set_xlim(0.8, 4.9); ax.set_ylabel(yl, fontsize=7)
        ax.grid(axis="y", lw=0.4, color="#e2e8f0", zorder=0)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6.6)


def fig_val150_quartiles():
    d = json.load(open("results/cvpr/milestone16_val150_retrieval_to_3d.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    quartile_panels(axes, d)
    axes[0].set_title("(a) detection quality by retrieval quartile", fontsize=7.6, fontweight="bold")
    axes[1].set_title("(b) false positives by retrieval quartile", fontsize=7.6, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(pad=0.4, rect=[0, 0.07, 1, 1])
    fig.savefig(OUT/"fig_val150_quartiles.pdf", bbox_inches="tight")
    plt.close(fig); print("  fig_val150_quartiles.pdf")


# ---------------------------------------------------------------- supplementary strips
SUPP_Q = [("water droplets smearing the camera lens", False),
          ("blinding headlight bloom from oncoming traffic", False),
          ("wet reflective asphalt after rainfall", False),
          ("a low contrast scene where vehicles are barely distinguishable from the background", False),
          ("road construction zone with traffic cones and barriers", True),
          ("large truck or bus ahead", True)]


def fig_supp_strips():
    """2 nuScenes + 2 Waymo + 2 BDD100K per query. Degradation queries use v2.3;
    semantic queries use v2.3A (post-hoc abstention) in MONOCULAR mode, because
    the BEV and tag channels are contaminated for object queries on these scenes."""
    import textwrap, torch
    from PIL import Image
    dev = torch.device("cuda")
    z = np.load("work_dirs/unified_filter/features_v2.npz", allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]; dom = z["domain"].astype(int)
    evset = set(json.load(open("benchmark_cvpr/split_906_602.json"))["eval"])
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    emb, tlog, risk = T("emb"), T("tag_logits"), T("risk")
    sig, mz, prior = T("siglip_img"), T("vlm_margin_z"), T("prior")
    qwen = torch.load("work_dirs/unified_filter/V_frozen_cache.pt", map_location=dev).float()
    from eval_semantic_edge_cases import text_embed, load_v23
    M = load_v23(risk.reshape(-1).cpu().numpy())
    QT = text_embed([q for q, _ in SUPP_Q])
    broot = Path("/nfs/turbo/umd-srawa/scene_mining/datasets/external_adverse/bdd100k")
    bz = np.load(broot/"signals.npz", allow_pickle=False); bn = sorted(json.load(open(broot/"labels.json")))
    ASK = ["wet_road", "glare", "droplets", "low_contrast"]; TO = ["droplets", "glare", "wet_road", "low_contrast"]
    BM = bz["margin"][:, [ASK.index(k) for k in TO]]; BM = (BM - BM.mean(0)) / (BM.std(0) + 1e-6)
    BS = torch.from_numpy(bz["siglip"]).float().to(dev); BMt = torch.from_numpy(BM).float().to(dev)
    nb = len(bn); PR = torch.zeros(nb, 26, device=dev)
    av = torch.stack([torch.ones(nb, device=dev), torch.ones(nb, device=dev), torch.zeros(nb, device=dev)], -1)
    N = len(corpus); on, off = torch.ones(N, device=dev), torch.zeros(N, device=dev)
    thumbs = Path("work_dirs/unified_filter/scene_thumbs")
    fig, axes = plt.subplots(len(SUPP_Q), 6, figsize=(7.2, 0.70 * len(SUPP_Q) + 0.45))
    fig.subplots_adjust(left=0.2, right=0.995, top=0.92, bottom=0.01, wspace=0.05, hspace=0.30)
    for r, (q, semantic) in enumerate(SUPP_Q):
        for m in M: m.use_abstain = semantic
        qv = QT[r].unsqueeze(0).expand(N, -1)
        with torch.no_grad():
            s = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior,
                           has_3d=off if semantic else on).cpu().numpy() for m in M], 0)
            qb = QT[r].unsqueeze(0).expand(nb, -1)
            sb = np.mean([m.base_score(qb, BS, BMt, PR, av).cpu().numpy() for m in M], 0)
        for m in M: m.use_abstain = False
        cols = []
        for d_ in (0, 1):
            seen, got = set(), []
            for i in np.argsort(-s):
                if corpus[i] not in evset or dom[i] != d_ or not (thumbs/f"{corpus[i]}.jpg").exists(): continue
                seg = corpus[i].split("#c")[0]
                if seg in seen: continue
                seen.add(seg); got.append(i)
                if len(got) == 2: break
            cols += [("nuScenes" if d_ == 0 else "Waymo", thumbs/f"{corpus[i]}.jpg", C1) for i in got]
        cols += [("BDD100K", broot/"images"/bn[i], C2) for i in np.argsort(-sb)[:2]]
        for c, (name, path, col) in enumerate(cols):
            ax = axes[r, c]; ax.imshow(Image.open(path).convert("RGB")); ax.set_anchor("N")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name} #{c % 2 + 1}", fontsize=5.6, color=col, pad=1.4)
            for sp in ax.spines.values(): sp.set_color(col); sp.set_lw(0.9)
        tag = "  [v2.3A, monocular]" if semantic else ""
        axes[r, 0].set_ylabel(textwrap.fill(f"\"{q}\"", 28) + tag, fontsize=5.6, rotation=0,
                              ha="right", va="center", style="italic", labelpad=5, linespacing=1.35)
    fig.suptitle("Top-2 retrievals per corpus: nuScenes and Waymo (602-scene eval split), "
                 "BDD100K (external, monocular)", fontsize=7.2, y=0.985)
    fig.savefig(OUT/"supp_strips.pdf", bbox_inches="tight", dpi=170)
    plt.close(fig); print("  supp_strips.pdf")


# ---------------------------------------------------------------- figure 4
def fig4():
    """(left) val150 CAM_FRONT mAP and ghost FP by retrieval quartile, degradation
    queries; (right) mAP per query stratum. One measure per axis - no dual axes."""
    d16 = json.load(open("results/cvpr/milestone16_val150_retrieval_to_3d.json"))
    m14 = json.load(open("results/cvpr/milestone14_v2_3.json"))["C_suite150"]["per_stratum"]
    m10 = json.load(open("results/cvpr/milestone10_query_suite_150.json"))["per_stratum"]
    m15 = json.load(open("results/cvpr/milestone15_semantic_edge_cases.json"))["corpus_258"]["summary"]
    fig = plt.figure(figsize=(7.0, 2.85))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.12, wspace=0.42)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    quartile_panels([a1, a2], d16)
    a1.set_ylabel("front mAP", fontsize=6.8); a2.set_ylabel("ghost FP rate", fontsize=6.8)
    plt.setp(a1.get_xticklabels(), visible=False)
    a1.set_title("(a) val150: 3D quality of retrieved scenes", fontsize=7.4, fontweight="bold")
    ax = fig.add_subplot(gs[:, 1])
    rows = [("S1 sensor degr.", "sensor_degradation"), ("S2 macro-env.", "macro_environmental"),
            ("S3 compositional", "compositional_heldout"), ("S4 negated", "negated"),
            ("S5a semantic atomic*", "semantic_atomic"), ("S5b semantic x degr.*", "semantic_compositional")]
    ser = [("SceneMiner v2.3", "#2a78d6", "o", lambda s: m14[s]["v2.3_full"] if s in m14 else m15[s]["v2.3_monocular"]["mAP"]),
           ("v2.3A (post-hoc)", "#6250d6", "v", lambda s: m14[s]["v2.3_abstain"] if s in m14 else m15[s]["v2.3A_monocular"]["mAP"]),
           ("zero-shot SigLIP2", "#eb6834", "s", lambda s: m14[s]["zeroshot_SigLIP2"] if s in m14 else m15[s]["zeroshot_SigLIP2"]["mAP"]),
           ("VLM margin lookup", "#1baf7a", "^", lambda s: m10[s]["logit_VLM_lookup"] if s in m10 else m15[s]["logit_VLM_lookup"]["mAP"])]
    y = np.arange(len(rows))[::-1]
    for k, (lab, col, mk, f) in enumerate(ser):
        xs = [f(s) for _, s in rows]
        ax.scatter(xs, y + (k - 1.5) * 0.19, color=col, marker=mk, s=34, zorder=3,
                   edgecolor="white", linewidth=0.7, label=lab)
    # S5 is graded by humans; S1-S4 by the VLM's own attributes (see caption)
    ax.axhspan(-0.55, 1.5, color="#f1f0ec", zorder=0, lw=0)
    ax.text(0.84, 1.42, "human GT", fontsize=5.6, color="#55554f", ha="right", va="top")
    ax.text(0.84, 5.55, "VLM-attribute GT", fontsize=5.6, color="#55554f", ha="right", va="top")
    ax.set_yticks(y); ax.set_yticklabels([r for r, _ in rows], fontsize=6.4)
    ax.set_ylim(-0.6, 5.65)
    ax.set_xlabel("mAP", fontsize=6.8); ax.set_xlim(0.0, 0.85); ax.tick_params(labelsize=6.4)
    ax.grid(axis="x", lw=0.4, color="#e2e8f0", zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title("(b) retrieval mAP by query stratum", fontsize=7.4, fontweight="bold")
    hb, lb = ax.get_legend_handles_labels(); ha, la = a1.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    extra = [(h, l) for h, l in zip(ha, la) if l == "night/rain metadata"]
    rnd = Line2D([], [], color="#8a8a85", lw=1.2, ls=(0, (4, 3)))
    fig.legend(hb + [e[0] for e in extra] + [rnd], lb + [e[1] for e in extra] + ["random"],
               loc="lower center", ncol=6, fontsize=5.9, frameon=False, bbox_to_anchor=(0.5, -0.07),
               handlelength=1.8, columnspacing=1.2)
    fig.savefig(OUT/"fig4_front_sector_degradation_and_semantics.pdf", bbox_inches="tight")
    plt.close(fig); print("  fig4_front_sector_degradation_and_semantics.pdf")
