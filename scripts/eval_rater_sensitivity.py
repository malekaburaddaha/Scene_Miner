#!/usr/bin/env python3
"""MILESTONE 19 - single-rater sensitivity on the 550 human-labelled scenes.

1. Rater's own uncertainty. The 50 scenes the rater flagged "unsure" carry NO
   labels (none was set), so they cannot be scored; Mode A (drop skipped
   scenes) vs Mode B (keep unambiguous skips as negatives) is the available
   check, and is reported.
2. Boundary-exclusion subsets. "Boundary" is defined by CLIP ViT-B/32 - a model
   that NONE of the rankers uses, so the subset is not selected by any of the
   compared systems. A scene is on the boundary for an attribute when CLIP's
   attribute-vs-neutral probability is near 0.5; the 10 % and 20 % most
   ambiguous scenes are removed in turn.
3. Monte Carlo second rater. For a target Cohen's kappa (0.90, 0.85, 0.75)
   with prevalence held fixed, f positives and f negatives are flipped per
   attribute, f = N (1 - p_e)(1 - kappa) / 2. Which labels flip is drawn either
   uniformly or weighted toward CLIP-ambiguous scenes; 500 trials each. Ranker
   scores are fixed; only the labels move.
Rankers: SceneMiner v2.4 (5-seed mean), logit VLM lookup, zero-shot SigLIP2,
and the two late-fusion controls z(SigLIP2)+z(lookup), z(SigLIP2)+metadata.
"""
from __future__ import annotations
from paths import DEVICE
from paths import HUMAN_R1
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary
from eval_human_gt_full import load_human, ATTRS, QUERY, META
from eval_semantic_edge_cases import text_embed, load_v23, zs
from annotate_ui import find_thumb

dev = torch.device(DEVICE)
import os
FINAL = os.environ.get("SM_FINAL", "v24")
CLIP_PROMPT = {"droplets": "a photo taken through a camera lens with water droplets on it",
               "glare": "a photo with strong headlight or sun glare",
               "wet_road": "a photo of a wet reflective road",
               "low_contrast": "a dark, low contrast photo with poor visibility"}
NEUTRAL = "a clear, normal photo of a road"


def clip_probs(toks):
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    P = "openai/clip-vit-base-patch32"
    m = CLIPModel.from_pretrained(P, use_safetensors=True).to(dev).eval(); pr = CLIPProcessor.from_pretrained(P)
    th = Path("work_dirs/unified_filter/scene_thumbs")
    txt = [CLIP_PROMPT[a] for a in ATTRS] + [NEUTRAL]
    ft = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()   # transformers 5: projected
    with torch.no_grad():
        T = F.normalize(ft(m.get_text_features(**pr(text=txt, return_tensors="pt", padding=True).to(dev))), dim=-1)
        out = []
        for i in range(0, len(toks), 32):
            ims = [Image.open(find_thumb(th, t)).convert("RGB") for t in toks[i:i+32]]
            I = F.normalize(ft(m.get_image_features(**pr(images=ims, return_tensors="pt").to(dev))), dim=-1)
            L = 100.0 * I @ T.T                                   # CLIP logit scale
            p = torch.stack([torch.softmax(torch.stack([L[:, k], L[:, 4]], -1), -1)[:, 0]
                             for k in range(4)], -1)
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def main():
    W = Path("work_dirs/unified_filter")
    z = np.load(W/"features_v2.npz", allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]; cidx = {t: i for i, t in enumerate(corpus)}
    humB = load_human(HUMAN_R1, "B")
    humA = load_human(HUMAN_R1, "A")
    toks = [t for t in corpus if t in humB]; idx = np.array([cidx[t] for t in toks])
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    emb, tlog, risk = T("emb"), T("tag_logits"), T("risk")
    sig, mz, prior = T("siglip_img"), T("vlm_margin_z"), T("prior")
    qwen = torch.load(W/"V_frozen_cache.pt", map_location=dev).float()
    M = load_v23(risk.reshape(-1).cpu().numpy(), which=FINAL); [setattr(m, "use_abstain", True) for m in M]
    vlm = json.load(open(W/"binary_flags_with_logits.json")); strict = json.load(open(W/"lowcontrast_strict.json"))
    meta = json.load(open(W/"deterministic_gt.json"))
    QT = text_embed([QUERY[a] for a in ATTRS]); QN = F.normalize(QT, dim=-1)
    N = len(corpus); on = torch.ones(N, device=dev)
    S = {}
    for ai, at in enumerate(ATTRS):
        qv = QT[ai].unsqueeze(0).expand(N, -1)
        src, key = (strict, "low_contrast_strict") if at == "low_contrast" else (vlm, at)
        with torch.no_grad():
            v24 = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=on).cpu().numpy() for m in M], 0)[idx]
        cos = (sig[idx] @ QN[ai]).cpu().numpy()
        look = np.array([float(src.get(t, {}).get(key + "_margin", -9.)) for t in toks])
        fld, val = next(iter(META[at].items()))
        md = np.array([float(meta.get(t, {}).get(fld) == val) for t in toks])
        S[at] = {"SceneMiner_v2.4": v24, "logit_VLM_lookup": look, "zeroshot_SigLIP2": cos,
                 "late_fusion SigLIP2+lookup": zs(cos) + zs(look), "late_fusion SigLIP2+metadata": zs(cos) + md}
    RK = list(S[ATTRS[0]])
    H = {at: np.array([humB[t][at] for t in toks]) for at in ATTRS}
    ap = lambda s, y: ap_binary(np.argsort(-s), y)
    out = {"doc": __doc__, "n_scenes": len(toks), "n_unsure_unlabelled": 50}

    def table(keep):
        r = {k: {at: ap(S[at][k][keep[at]], H[at][keep[at]]) for at in ATTRS} for k in RK}
        for k in RK: r[k]["macro"] = float(np.mean([r[k][at] for at in ATTRS]))
        return r
    allk = {at: np.ones(len(toks), bool) for at in ATTRS}
    out["full_550"] = table(allk)
    tA = [t for t in toks if t in humA]; kA = np.array([t in set(tA) for t in toks])
    HA = {at: np.array([humA[t][at] if t in humA else False for t in toks]) for at in ATTRS}
    rA = {k: {at: ap(S[at][k][kA], HA[at][kA]) for at in ATTRS} for k in RK}
    for k in RK: rA[k]["macro"] = float(np.mean([rA[k][at] for at in ATTRS]))
    out["mode_A"] = {"n": int(kA.sum()), "table": rA}

    CP = clip_probs(toks); amb = 1.0 - 2.0 * np.abs(CP - 0.5)          # 1 = maximally ambiguous
    out["clip_ambiguity_mean"] = {at: float(amb[:, i].mean()) for i, at in enumerate(ATTRS)}
    out["boundary_excluded"] = {}
    for frac in (0.10, 0.20):
        keep = {at: amb[:, i] < np.quantile(amb[:, i], 1 - frac) for i, at in enumerate(ATTRS)}
        out["boundary_excluded"][f"drop_{int(frac*100)}pct"] = table(keep)

    rng = np.random.default_rng(0); n = len(toks)
    out["monte_carlo"] = {}
    for kappa in (0.90, 0.85, 0.75):
        for scheme in ("uniform", "clip_boundary"):
            wins = {f"{a} > {b}": {at: 0 for at in ATTRS + ["macro"]} for a, b in
                    [("SceneMiner_v2.4", "logit_VLM_lookup"), ("SceneMiner_v2.4", "zeroshot_SigLIP2"),
                     ("SceneMiner_v2.4", "late_fusion SigLIP2+lookup"), ("SceneMiner_v2.4", "late_fusion SigLIP2+metadata")]}
            maps = {k: [] for k in RK}; kap = []
            for trial in range(500):
                res = {k: {} for k in RK}
                for i, at in enumerate(ATTRS):
                    y = H[at].copy(); P = y.sum(); pe = (P/n)**2 + (1-P/n)**2
                    f = int(round(n * (1 - pe) * (1 - kappa) / 2))
                    w = (amb[:, i] + 1e-3) if scheme == "clip_boundary" else np.ones(n)
                    pos, neg = np.where(y)[0], np.where(~y)[0]
                    fp = rng.choice(pos, min(f, len(pos)), replace=False, p=w[pos]/w[pos].sum())
                    fn = rng.choice(neg, min(f, len(neg)), replace=False, p=w[neg]/w[neg].sum())
                    y2 = y.copy(); y2[fp] = False; y2[fn] = True
                    po = (y == y2).mean(); kap.append((po - pe) / (1 - pe))
                    for k in RK: res[k][at] = ap(S[at][k], y2)
                for k in RK:
                    res[k]["macro"] = float(np.mean([res[k][at] for at in ATTRS])); maps[k].append(res[k]["macro"])
                for key in wins:
                    a, b = key.split(" > ")
                    for at in ATTRS + ["macro"]:
                        wins[key][at] += int(res[a][at] > res[b][at])
            out["monte_carlo"][f"kappa_{kappa}_{scheme}"] = {
                "realised_kappa_mean": float(np.mean(kap)),
                "win_rate": {k: {at: v / 500 for at, v in d.items()} for k, d in wins.items()},
                "macro_mAP_mean": {k: float(np.mean(v)) for k, v in maps.items()},
                "macro_mAP_p05": {k: float(np.percentile(v, 5)) for k, v in maps.items()}}
    Path(os.environ.get("SM_OUT", "results/cvpr/milestone19_rater_sensitivity.json")).write_text(json.dumps(out, indent=1))
    def show(name, t):
        print(f"\n-- {name} --  " + "  ".join(f"{k[:22]} {t[k]['macro']:.4f}" for k in RK))
    show("full 550 (Mode B)", out["full_550"]); show(f"Mode A ({out['mode_A']['n']})", out["mode_A"]["table"])
    for k, t in out["boundary_excluded"].items(): show(f"CLIP boundary {k}", t)
    print("\n-- Monte Carlo second rater (500 trials each): share of trials SceneMiner wins --")
    for k, d in out["monte_carlo"].items():
        print(f"  {k:<28} kappa~{d['realised_kappa_mean']:.3f}")
        for c, w in d["win_rate"].items():
            print(f"     {c:<52} macro {w['macro']:.3f} | " + " ".join(f"{at[:4]} {w[at]:.2f}" for at in ATTRS))


if __name__ == "__main__":
    main()
