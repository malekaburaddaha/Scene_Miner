#!/usr/bin/env python3
"""MILESTONE 24 - two-part qualitative showcase (main Fig. 2) and the
supplementary semantic strips and failure-comparison strip.

Every row shows SceneMiner's top-2 retrievals from EACH of nuScenes and Waymo
(602-scene eval split, one image per driving segment) and from BDD100K (1000
external images, portable signals), so no dataset dominates.

Box callouts come only from HUMAN 3D annotations, projected into the image that
was retrieved:
  nuScenes  sample_annotation boxes of the scene's first keyframe (the frame the
            thumbnail was cut from; verified by pixel correlation), visibility >= 40%.
  Waymo     the segment's labelled 3D boxes for the matched frame (the thumbnail is
            matched to its source frame by pixel correlation, r >= 0.95, else no box
            is drawn). Waymo labels have no truck/bus or cone/barrier classes, so
            those rows draw no Waymo boxes.
  BDD100K   no box labels are on disk; no callouts are drawn.
A cell whose frame has no annotated object of the row's class says so.
"""
from __future__ import annotations
from paths import DEVICE
from paths import EXT, HUMAN_R1
import os
import argparse, glob, json, sys, textwrap
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
LABEL_FONT = "/usr/share/fonts/dejavu/DejaVuSansCondensed-Bold.ttf"      # narrow bold for query labels
LABEL = lambda size: FontProperties(fname=LABEL_FONT, size=size)
from PIL import Image
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

dev = torch.device(DEVICE)
OUT = Path("paper/figures")
NUSC = Path("data/nuscenes")
THUMBS = Path("work_dirs/unified_filter/scene_thumbs")
BROOT = Path(EXT + "/bdd100k")
COLS = [("nuScenes", "#2a78d6"), ("Waymo", "#1baf7a"), ("BDD100K", "#eb6834")]
BOX_C = "#ffd400"

PART_A = [("water droplets smearing the camera lens", None),
          ("blinding headlight bloom from oncoming traffic", None),
          ("a low contrast scene on wet reflective asphalt", None)]
PART_B = [("pedestrians crossing the road at night", "ped"),
          ("cyclists or motorcyclists on the road", "cyc"),
          ("a construction zone with traffic cones and barriers", "cons"),
          ("a large truck or bus on a wet road", "heavy")]
NUSC_CLS = {"ped": ("human.pedestrian.",), "cyc": ("vehicle.bicycle", "vehicle.motorcycle"),
            "cons": ("movable_object.trafficcone", "movable_object.barrier", "vehicle.construction"),
            "heavy": ("vehicle.truck", "vehicle.bus", "vehicle.trailer"),
            "rare": ("human.pedestrian.child", "human.pedestrian.stroller", "human.pedestrian.wheelchair",
                     "human.pedestrian.personal_mobility", "human.pedestrian.police_officer",
                     "human.pedestrian.construction_worker", "animal"),
            "emerg": ("vehicle.emergency.",)}
# Waymo types: 1 vehicle, 2 pedestrian, 3 sign, 4 cyclist. Trucks, buses and motorcycles are
# labelled VEHICLE, so those rows select VEHICLE boxes by their labelled 3D size. Heavy vehicles:
# length >= 6.0 m or height >= 2.8 m (a 2.3 m height bound admits tall vans, e.g. 5.3 x 2.5 x 2.45 m).
WAYMO_RULE = {"ped": lambda t, l, w, h: t == 2,
              "cyc": lambda t, l, w, h: t == 4 or (t == 1 and l < 2.8 and w < 1.2),
              "heavy": lambda t, l, w, h: t == 1 and (l >= 6.0 or h >= 2.8)}
BDD_CLS = {"ped": ("pedestrian", "other person"), "cyc": ("rider", "bicycle", "motorcycle"),
           "heavy": ("truck", "bus")}


# ---------------------------------------------------------------- ranking
class Ranker:
    def __init__(self):
        from eval_semantic_edge_cases import load_v23
        z = np.load("work_dirs/unified_filter/features_v2.npz", allow_pickle=False)
        self.corpus = [str(t) for t in z["scene_tokens"]]; self.dom = z["domain"].astype(int)
        self.ev = set(json.load(open("benchmark_cvpr/split_906_602.json"))["eval"])
        T = lambda k: torch.from_numpy(z[k]).float().to(dev)
        self.F = (T("emb"), T("tag_logits"), T("risk"),
                  torch.load("work_dirs/unified_filter/V_frozen_cache.pt", map_location=dev).float(),
                  T("siglip_img"), T("vlm_margin_z"), T("prior"))
        self.M = load_v23(z["risk"].reshape(-1), which=os.environ.get("SM_FINAL", "v26"))
        for m in self.M: m.use_abstain = True
        bz = np.load(BROOT/"signals.npz", allow_pickle=False); self.bn = sorted(json.load(open(BROOT/"labels.json")))
        ASK = ["wet_road", "glare", "droplets", "low_contrast"]; TO = ["droplets", "glare", "wet_road", "low_contrast"]
        BM = bz["margin"][:, [ASK.index(k) for k in TO]]; BM = (BM - BM.mean(0)) / (BM.std(0) + 1e-6)
        self.BS = torch.from_numpy(bz["siglip"]).float().to(dev); self.BM = torch.from_numpy(BM).float().to(dev)

    def top(self, qraw, n_per=2, used=None):
        """top n_per per corpus; `used` holds driving segments already shown in this
        figure, which are skipped so no segment appears twice. Returns (dataset, id, rank)."""
        used = set() if used is None else used
        from eval_semantic_edge_cases import text_embed
        N, nb = len(self.corpus), len(self.bn)
        qv = text_embed([qraw])[0]
        with torch.no_grad():
            s = np.mean([m(qv.expand(N, -1), qv.expand(N, -1), *self.F, has_3d=torch.ones(N, device=dev)).cpu().numpy()
                         for m in self.M], 0)
            av = torch.stack([torch.ones(nb, device=dev), torch.ones(nb, device=dev), torch.zeros(nb, device=dev)], -1)
            sb = np.mean([m.base_score(qv.expand(nb, -1), self.BS, self.BM, torch.zeros(nb, 26, device=dev), av)
                          .cpu().numpy() for m in self.M], 0)
        cells = []
        for d in (0, 1):
            got, rank = [], 0
            for i in np.argsort(-s):
                t = self.corpus[i]
                if t not in self.ev or self.dom[i] != d or not (THUMBS/f"{t}.jpg").exists(): continue
                rank += 1
                seg = t.split("#c")[0]
                if seg in used: continue
                used.add(seg); got.append(("nusc" if d == 0 else "waymo", t, rank))
                if len(got) == n_per: break
            cells += got
        got = []
        for rank, i in enumerate(np.argsort(-sb), 1):
            seg = "bdd:" + self.bn[i].split("-")[0]                      # BDD100K video id
            if seg in used: continue
            used.add(seg); got.append(("bdd", self.bn[i], rank))
            if len(got) == n_per: break
        return cells + got


# ---------------------------------------------------------------- annotations
class Annot:
    def __init__(self):
        from nuscenes.nuscenes import NuScenes
        self.nusc = NuScenes("v1.0-trainval", dataroot=str(NUSC), verbose=False)
        self.scene = {s["token"]: s for s in self.nusc.scene}

    def nusc_frame(self, tok, kind):
        """first keyframe CAM_FRONT image + projected human boxes of the row's class."""
        from nuscenes.utils.geometry_utils import BoxVisibility, view_points
        s = self.scene[tok]["first_sample_token"]
        sd = self.nusc.get("sample", s)["data"]["CAM_FRONT"]
        path, boxes, K = self.nusc.get_sample_data(sd, box_vis_level=BoxVisibility.ANY)
        img = Image.open(path).convert("RGB")
        th = np.asarray(Image.open(THUMBS/f"{tok}.jpg").convert("L").resize((200, 112)), float)
        r = float(np.corrcoef(th.ravel(), np.asarray(img.convert("L").resize((200, 112)), float).ravel())[0, 1])
        out = []
        if kind:
            for b in boxes:
                a = self.nusc.get("sample_annotation", b.token)
                if not a["category_name"].startswith(NUSC_CLS[kind]) or int(a["visibility_token"]) < 2: continue
                if b.center[2] < 1: continue
                c = view_points(b.corners(), K, normalize=True)[:2]
                out.append(_clip(c, img.size))
        return img, [o for o in out if o is not None], r


def _clip(c, size):
    W, H = size
    x0, x1 = max(c[0].min(), 0), min(c[0].max(), W - 1); y0, y1 = max(c[1].min(), 0), min(c[1].max(), H - 1)
    return (x0, y0, x1, y1) if x1 - x0 > 2 and y1 - y0 > 2 else None


def waymo_frame(tok, kind):
    """Matched source frame of a Waymo chunk thumbnail, and the projected labelled
    boxes of the row's class. Returns (image, boxes, r) or the thumbnail if no
    frame matches."""
    seg, ci = tok.split("#c")
    th = np.asarray(Image.open(THUMBS/f"{tok}.jpg").convert("L").resize((200, 112)), float)
    best = (-1, None)
    for fd in sorted(glob.glob(f"work_dirs/waymo_zeroshot/extracted_all_*/{seg}/*/")) or \
              sorted(glob.glob(f"work_dirs/waymo_zeroshot/extracted*/{seg}/*/")):
        p = Path(fd)/"cam_FRONT.png"
        if not p.exists(): continue
        im = Image.open(p).convert("L"); w, h = im.size; sc = min(448 / h, 800 / w)
        nw, nh = int(w * sc), int(h * sc); can = Image.new("L", (800, 448)); can.paste(im.resize((nw, nh)), ((800 - nw) // 2, (448 - nh) // 2))
        r = float(np.corrcoef(th.ravel(), np.asarray(can.resize((200, 112)), float).ravel())[0, 1])
        if r > best[0]: best = (r, Path(fd))
    r, fd = best
    if fd is None or r < 0.95:
        return Image.open(THUMBS/f"{tok}.jpg").convert("RGB"), [], r
    img = Image.open(fd/"cam_FRONT.png").convert("RGB")
    out = []
    if kind in WAYMO_RULE:
        cal = np.load(fd/"calib.npz", allow_pickle=True); i = list(cal["camera_order"]).index("FRONT")
        K, T = cal["intrinsics"][i], np.linalg.inv(cal["extrinsics_cam2veh"][i])
        b = np.load(fd/"boxes.npz")
        for t, c, d, hd in zip(b["types"], b["centers"], b["dims"], b["headings"]):
            l, w_, h_ = d
            if not WAYMO_RULE[kind](int(t), l, w_, h_): continue
            xs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
            ys = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w_ / 2; zs = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * h_ / 2
            R = np.array([[np.cos(hd), -np.sin(hd)], [np.sin(hd), np.cos(hd)]])
            xy = R @ np.stack([xs, ys]); P = np.stack([xy[0] + c[0], xy[1] + c[1], zs + c[2], np.ones(8)])
            Pc = (T @ P)[:3]
            if (Pc[0] < 1).any() or np.linalg.norm(c[:2]) > 50: continue
            u = K[0, 2] - K[0, 0] * Pc[1] / Pc[0]; v = K[1, 2] - K[1, 1] * Pc[2] / Pc[0]
            bb = _clip(np.stack([u, v]), img.size)
            if bb is not None: out.append(bb)
    return img, out, r


# ---------------------------------------------------------------- drawing
def bdd_frame(name, kind, BB):
    img = Image.open(BROOT/"images"/name).convert("RGB"); W, H = img.size; out = []
    for b in BB.get(name, []) if kind in BDD_CLS else []:
        if b["label"] in BDD_CLS[kind]:
            x, y, w, h = b["bbox"]; out.append((x * W, y * H, (x + w) * W, (y + h) * H))
    return img, out


def draw_cell(ax, img, boxes, kind, has_labels, name_col, idx, title=True):
    name, col = name_col
    W, H = img.size
    # display window cropped to 16:9 (Waymo frames are 3:2); boxes stay in pixel coordinates
    Hc = min(H, W / (16 / 9)); y0c = (H - Hc) * 0.6
    ax.imshow(img); ax.set_xlim(0, W); ax.set_ylim(y0c + Hc, y0c); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(f"{name} #{idx}", fontsize=8.5, color=col, pad=2.0, fontweight="bold")
    for sp in ax.spines.values(): sp.set_color(col); sp.set_lw(1.6)
    for (x0, y0, x1, y1) in boxes:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=BOX_C, lw=0.5))          # hairline, so the object stays visible
    if kind is None: return
    if boxes:
        # zoom callout on the smallest annotated object
        x0, y0, x1, y1 = min(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        if (x1 - x0) * (y1 - y0) < 0.02 * W * H:
            # inset: 45% of the thumbnail height, 42% of its width; window aspect matched to the inset
            iw, ih = 0.42, 0.45
            half = max(x1 - x0, y1 - y0) * 1.6 + 0.02 * W; hv = half * (ih * 9) / (iw * 16)
            cx = float(np.clip((x0 + x1) / 2, half, W - half)); cy = float(np.clip((y0 + y1) / 2, hv, H - hv))
            ins = ax.inset_axes([1 - iw - 0.01, 1 - ih - 0.015, iw, ih])
            ins.imshow(img, aspect="auto"); ins.set_xlim(cx - half, cx + half); ins.set_ylim(cy + hv, cy - hv)
            ins.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=BOX_C, lw=0.5))
            ins.set_xticks([]); ins.set_yticks([])
            for sp in ins.spines.values(): sp.set_color(BOX_C); sp.set_lw(1.0)
        ax.text(0.02, 0.04, f"{len(boxes)} annotated", transform=ax.transAxes, fontsize=7.5, color="black",
                fontweight="bold", bbox=dict(fc=BOX_C, ec="#111111", lw=0.6, pad=1.0, alpha=0.95))
    elif has_labels:       # the dataset labels this class, and this frame has none
        ax.text(0.02, 0.04, "0 annotated", transform=ax.transAxes, fontsize=7.5, color="white",
                fontweight="bold", bbox=dict(fc="#c0392b", ec="none", pad=1.0, alpha=0.95))


def render(rows, out, title, R, A, n_per=2, part_breaks=()):
    nr = len(rows) + len(part_breaks)
    # full text width (7 in); a narrow bold label column; thumbnails 16:9 with small gaps
    LEFT, RIGHT, WS = 0.135, 0.998, 0.015
    cell_w = 7.0 * (RIGHT - LEFT) / (3 * n_per + (3 * n_per - 1) * WS)
    row_h = cell_w * 9 / 16
    hr = []; layout = []
    for i, r in enumerate(rows):
        for (bi, lab) in part_breaks:
            if bi == i: hr.append(0.34 / row_h); layout.append(("hdr", lab))
        hr.append(1.0); layout.append(("row", r))
    H = row_h * (len(rows) * 1.0 + 0.34 / row_h * len(part_breaks)) * 1.12 + 0.25
    fig = plt.figure(figsize=(7.0, H))
    gs = fig.add_gridspec(len(layout), 3 * n_per, height_ratios=hr, left=LEFT, right=RIGHT, top=0.985, bottom=0.004,
                          wspace=WS, hspace=0.12)
    log = []; first = True; used = set()
    BB = json.load(open(BROOT/"boxes.json"))
    for li, (typ, r) in enumerate(layout):
        if typ == "hdr":
            ax = fig.add_subplot(gs[li, :]); ax.axis("off")
            ax.text(-LEFT / (RIGHT - LEFT), 0.40, r, transform=ax.transAxes, fontsize=9.0, fontweight="bold", va="bottom")
            first = True; continue
        q, kind = r
        cells = R.top(q, n_per, used); entry = {"query": q, "cells": []}
        for c, (src, tok, rank) in enumerate(cells):
            ax = fig.add_subplot(gs[li, c])
            if src == "nusc":
                img, boxes, rr = A.nusc_frame(tok, kind); lab = kind is not None
            elif src == "waymo":
                img, boxes, rr = waymo_frame(tok, kind); lab = kind in WAYMO_RULE and rr >= 0.95
            else:
                (img, boxes), rr = bdd_frame(tok, kind, BB), None; lab = kind in BDD_CLS
            draw_cell(ax, img, boxes, kind, lab, COLS[c // n_per], c % n_per + 1, title=first)
            entry["cells"].append({"dataset": src, "id": tok, "rank_in_corpus": rank, "frame_match_r": rr,
                                   "class_labelled": bool(lab), "n_boxes": len(boxes)})
            if c == 0:
                ax.set_ylabel(textwrap.fill(q, 17), fontproperties=LABEL(8.5), rotation=0, ha="right",
                              va="center", labelpad=3, linespacing=1.05)
        first = False; log.append(entry); print(f"  {q[:50]:<50} " + " ".join(f"{e['dataset'][:2]}:{e['n_boxes']}" for e in entry["cells"]), flush=True)
    if title: fig.suptitle(title, fontsize=7.0, y=0.999)
    fig.savefig(OUT/out, bbox_inches="tight", pad_inches=0.02, dpi=200); plt.close(fig)
    cells = [c for e in log for c in e["cells"]]
    ids = [c["id"] for c in cells]
    assert len(ids) == len(set(ids)), "duplicate image in figure"
    lab = [c for c in cells if c["class_labelled"]]
    summary = {"n_cells": len(cells), "unique_images": len(set(ids)),
               "cells_with_class_labels": len(lab), "cells_with_annotated_object": sum(c["n_boxes"] > 0 for c in lab),
               "by_dataset": {d: {"labelled": sum(c["dataset"] == d for c in lab),
                                  "hit": sum(c["dataset"] == d and c["n_boxes"] > 0 for c in lab)}
                              for d in ("nusc", "waymo", "bdd")},
               "max_rank_shown": max(c["rank_in_corpus"] for c in cells)}
    print("  summary:", summary, flush=True)
    return {"rows": log, "summary": summary}


# ---------------------------------------------------------------- supplementary failure strip
def failure_strip(out):
    """BEV-TSR-style vs Qwen2.5-VL lookup vs SceneMiner top-3 on the 258 nuScenes
    scenes with human semantic GT. Rows chosen by a FIXED rule from the comparison
    per-query APs: the atomic and the compositional query with the largest
    SceneMiner margin over the better of the two baselines, plus the query with
    the largest margin AGAINST SceneMiner (shown so the strip is not one-sided)."""
    from eval_related_work_baselines import Enc, Rankers, load_corpus
    from eval_human_gt_full import load_human, ATTRS
    from eval_semantic_edge_cases import COND_COL
    res = json.load(open(os.environ.get("SM_CMP", "results/cvpr/milestone28_related_work_comparison.json")))
    S5 = json.load(open("benchmark_cvpr/queries_semantic_edge_cases.json"))["queries"]
    PQ = res["per_query"]; pick = []
    for st in ("atomic", "compositional"):
        Q = [q for q in S5 if q["stratum"] == f"semantic_{st}"]
        sm = np.array(PQ["SceneMiner"][f"s5_corpus_{st}"])
        base = np.maximum(PQ["BEV-TSR-style text-to-BEV"][f"s5_corpus_{st}"], PQ["Qwen2.5-VL lookup"][f"s5_corpus_{st}"])
        pick.append((Q[int(np.argmax(sm - base))], "largest SceneMiner margin", float((sm - base).max())))
        if st == "compositional":
            pick.append((Q[int(np.argmin(sm - base))], "largest margin against SceneMiner", float((sm - base).min())))
    enc = Enc(); z = np.load("work_dirs/unified_filter/features_v2.npz", allow_pickle=False)
    Rk = Rankers(enc, {}, z["risk"].reshape(-1)); C = load_corpus()
    G = json.load(open("work_dirs/unified_filter/human_semantic_gt.json"))["scenes"]
    hum = load_human(HUMAN_R1, "B")
    ev = set(json.load(open("benchmark_cvpr/split_906_602.json"))["eval"])
    toks = [t for t in C["toks"] if t in ev and t in G and t in hum]; cid = {t: i for i, t in enumerate(C["toks"])}
    idx = np.array([cid[t] for t in toks])
    truth = lambda t, k: G[t][k] if (k in G[t] and k not in ATTRS) else hum[t][k]
    meths = [("BEV-TSR-style", lambda q: Rk.bev_tsr(C, q["text"]).mean(0)),
             ("Qwen2.5-VL lookup", lambda q: Rk.lookup(C, q["text"], COND_COL[q["condition"]] if q["condition"] else None)),
             ("SceneMiner", lambda q: Rk.sceneminer(C, q["text"], mono=True).mean(0))]
    # full width: a header row (legend, method names), then per query a text line and 9 thumbnails
    WS, GAP, HDR, LAB = 0.03, 0.10, 0.42, 0.22          # GAP: spacer column between method groups
    wr = [1, 1, 1, GAP, 1, 1, 1, GAP, 1, 1, 1]
    cell_w = 7.0 / (sum(wr) + 10 * WS); row_h = cell_w * 9 / 16
    hr = [HDR] + [LAB, row_h] * len(pick)
    fig = plt.figure(figsize=(7.0, sum(hr)))
    gs = fig.add_gridspec(len(hr), 11, height_ratios=hr, width_ratios=wr, left=0, right=1, top=1, bottom=0,
                          wspace=WS, hspace=0)
    col = [0, 1, 2, 4, 5, 6, 8, 9, 10]
    axes = np.array([[fig.add_subplot(gs[2 + 2 * r, c]) for c in col] for r in range(len(pick))])
    hdr = fig.add_subplot(gs[0, :]); hdr.axis("off")
    hdr.text(0.5, 0.97, "Top-3 on the 258 nuScenes scenes with human semantic labels (✓ relevant, ✗ not); "
             "† BEV encoder supervised on these scenes.", ha="center", va="top", fontsize=7.5)
    for mi, mn in enumerate(["BEV-TSR-style †", "Qwen2.5-VL lookup", "SceneMiner"]):
        hdr.text((mi + 0.5) / 3, 0.02, mn, ha="center", va="bottom", fontsize=9, fontweight="bold")
    rng = np.random.default_rng(0); log = []
    for r, (q, why, margin) in enumerate(pick):
        rel = {t: all(truth(t, k) == v for k, v in q["predicate"].items()) for t in toks}
        for mi, (mn, f) in enumerate(meths):
            s = np.asarray(f(q), float)[idx]
            const = np.ptp(s) == 0
            if const: s = s + rng.random(len(s)) * 1e-9          # constant ranking: arbitrary order
            for k, i in enumerate(np.argsort(-s)[:3]):
                ax = axes[r, mi * 3 + k]; t = toks[i]
                ax.imshow(Image.open(THUMBS/f"{t}.jpg").convert("RGB")); ax.set_xticks([]); ax.set_yticks([])
                ok = rel[t]; colr = "#1baf7a" if ok else "#c0392b"
                for sp in ax.spines.values(): sp.set_color(colr); sp.set_lw(2.0)
                ax.text(0.04, 0.07, "✓" if ok else "✗", transform=ax.transAxes, fontsize=9, color="white",
                        fontweight="bold", bbox=dict(fc=colr, ec="none", pad=0.6))
                if const and k == 0:
                    ax.text(0.97, 0.94, "constant score:\narbitrary order", transform=ax.transAxes, fontsize=7.0,
                            ha="right", va="top", color="white", bbox=dict(fc="#444444", ec="none", pad=0.5))
            log.append({"query": q["text"], "method": mn, "top3_relevant": [bool(rel[toks[i]]) for i in np.argsort(-s)[:3]]})
        lab = fig.add_subplot(gs[1 + 2 * r, :]); lab.axis("off")
        lab.text(0.0, 0.12, f"“{q['text']}”", fontproperties=LABEL(8.5), va="bottom")
        lab.text(1.0, 0.12, f"[{why}]", fontsize=7.5, style="italic", ha="right", va="bottom")
    fig.savefig(OUT/out, bbox_inches="tight", pad_inches=0.02, dpi=200); plt.close(fig)
    return {"rule": failure_strip.__doc__, "rows": [{"query": q["text"], "why": w, "AP_margin": m} for q, w, m in pick],
            "top3": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "main", "supp", "fail"])
    a = ap.parse_args()
    lp = Path("results/cvpr/milestone24_qualitative_log.json")
    log = json.loads(lp.read_text()) if lp.exists() else {}
    log["doc"] = __doc__
    if a.only in ("all", "main", "supp"):
        R, A = Ranker(), Annot()
    if a.only in ("all", "main"):
        log["fig2"] = render(PART_A + PART_B, "fig2_qualitative.pdf", None, R, A,
                             part_breaks=[(0, "(a) Sensor and surface degradations"),
                                          (3, "(b) Road users and objects (yellow: human annotations; inset: smallest; red: none in frame)")])
    if a.only in ("all", "supp"):
        SQ = [("pedestrians walking near the roadway", "ped"), ("a pedestrian close to the vehicle", "ped"),
              ("a cyclist sharing the road", "cyc"), ("a motorcycle in traffic", "cyc"),
              ("road works with cones and barriers", "cons"), ("a bus at a stop", "heavy"),
              ("a child or an unusual road user near the road", "rare"),
              ("a police car or an emergency vehicle", "emerg"),
              ("pedestrians on a rainy night", "ped"), ("a truck ahead in heavy rain", "heavy")]
        log["supp_semantic"] = render(SQ, "supp_semantic_strips.pdf", None, R, A)
    if a.only in ("all", "fail"):
        log["supp_failure"] = failure_strip("supp_failure_strip.pdf")
    Path("results/cvpr/milestone24_qualitative_log.json").write_text(json.dumps(log, indent=1))
    print("wrote results/cvpr/milestone24_qualitative_log.json")


if __name__ == "__main__":
    main()
