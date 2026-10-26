#!/usr/bin/env python3
"""MILESTONE 34 - a larger SigLIP2 backbone: So400m/14 at 384 px instead of ViT-B/16.

--part embed --data {corpus,val150,bdd,holdout}
    normalised So400m image embeddings of the same image files the ViT-B/16 features
    were computed from -> work_dirs/emb_siglip2_so400m_{data}.npz (emb, names)
--part summarize
    joins the So400m run of eval_related_work_baselines.py (zero-shot SigLIP2 and
    SceneMiner retrained on So400m features, full and portable, five seeds) with the
    ViT-B/16 run of Table 2 -> results/cvpr/milestone34_larger_backbone.json

The So400m SceneMiner is trained with the unchanged v2.6 recipe (scripts/
train_sceneminer_v26.py --siglip-repo/--siglip-emb); its concept bank and gate are
rebuilt in the So400m text space from the same training phrasings. Nothing is tuned.
"""
from paths import EXT, SIGLIP2_SO400M
import argparse, json
from pathlib import Path
import numpy as np, torch

REPO = SIGLIP2_SO400M
BDD = Path(EXT + "/bdd100k")
HOLD = Path(EXT + "/bdd100k_holdout")
BASE = "results/cvpr/milestone28_related_work_comparison.json"
BIG = "results/cvpr/milestone34_so400m_comparison.json"


def sources(data):
    if data == "corpus":
        toks = [str(t) for t in np.load("work_dirs/unified_filter/features_v2.npz")["scene_tokens"]]
        return toks, [f"work_dirs/unified_filter/scene_thumbs/{t}.jpg" for t in toks]
    if data == "val150":
        sc = [str(t) for t in np.load("work_dirs/val150_siglip.npz")["scenes"]]
        return sc, [f"work_dirs/val150_thumbs/{t}.jpg" for t in sc]
    root = BDD if data == "bdd" else HOLD
    names = sorted(json.load(open(root/"labels.json")))
    return names, [str(root/"images"/n) for n in names]


@torch.no_grad()
def embed(data, batch=32):
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    import torch.nn.functional as F
    dev = torch.device("cuda")
    m = AutoModel.from_pretrained(REPO, use_safetensors=True, torch_dtype=torch.float32).to(dev).eval()
    ip = AutoImageProcessor.from_pretrained(REPO)
    names, paths = sources(data); X = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i+batch]]
        o = m.get_image_features(**ip(images=ims, return_tensors="pt").to(dev))
        X.append(F.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1).cpu().numpy())
    E = np.concatenate(X).astype(np.float32)
    np.savez_compressed(f"work_dirs/emb_siglip2_so400m_{data}.npz", emb=E, names=np.array(names))
    print(f"[embed] {data} {E.shape}", flush=True)


def summarize():
    B, S = json.load(open(BASE)), json.load(open(BIG))
    keys = ["human_gt", "bdd", "s5_corpus_atomic", "s5_corpus_compositional", "s5_val150_atomic",
            "s5_val150_compositional", "suite", "suite_skew_pts"]
    rows = {}
    for lab, src, name in [("SigLIP2 ViT-B/16", B, "SigLIP2 ViT-B/16"), ("SigLIP2 So400m/14", S, "SigLIP2 ViT-B/16"),
                           ("SceneMiner (portable), ViT-B/16", B, "SceneMiner (portable)"),
                           ("SceneMiner (portable), So400m/14", S, "SceneMiner (portable)"),
                           ("SceneMiner, ViT-B/16", B, "SceneMiner"), ("SceneMiner, So400m/14", S, "SceneMiner")]:
        m = src["methods"][name]
        r = {k: m.get(k) for k in keys}; r["human_gt_per_attr"] = m.get("human_gt_per_attr")
        ho = src["holdout"]["modes"]["metadata_free"]["methods"].get(name)
        r["holdout"] = ho["mAP_all"] if ho else None
        rows[lab] = r
    tests = {}
    for tag, src in [("ViT-B/16", B), ("So400m/14", S)]:
        P = src.get("paired_vs_sceneminer", {}).get("SigLIP2 ViT-B/16", {})
        H = src["holdout"]["modes"]["metadata_free"]["paired_vs_sceneminer_portable"].get("SigLIP2 ViT-B/16", {}).get("all")
        tests[f"SceneMiner minus SigLIP2, {tag}"] = {**{k: v for k, v in P.items()}, "holdout": H}
    out = {"doc": __doc__, "backbone_repo": "google/siglip2-so400m-patch14-384", "rows": rows, "paired": tests}
    json.dump(out, open("results/cvpr/milestone34_larger_backbone.json", "w"), indent=1)
    f = lambda x: "  -- " if x is None else f"{x:.3f}"
    print(f"{'':<36}" + "".join(f"{k[:12]:>13}" for k in keys + ["holdout"]))
    for lab, r in rows.items():
        print(f"{lab:<36}" + "".join(f"{f(r[k]):>13}" for k in keys + ["holdout"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["embed", "summarize"], required=True)
    ap.add_argument("--data", choices=["corpus", "val150", "bdd", "holdout"])
    a = ap.parse_args()
    embed(a.data) if a.part == "embed" else summarize()
