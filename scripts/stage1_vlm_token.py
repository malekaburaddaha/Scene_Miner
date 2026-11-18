#!/usr/bin/env python3
"""Stage 1 (single-camera path) - the 2048-d VLM token v of every scene.

v = mean over the image tokens of the last hidden layer of zero-shot Qwen2.5-VL-3B,
for the CAM_FRONT keyframe thumbnail and the prompt "Describe the driving conditions."
(QwenVisualEncoder in src/scene_mining/filter/lora_grounded_filter.py; images
downscaled to a 336 px long side). The encoder wraps the model in a freshly
initialised rank-1 LoRA whose B matrix is zero, so it computes the frozen model
exactly. This is the computation that produced work_dirs/unified_filter/V_frozen_cache.pt
(it was first written inline in scripts/eval_cvpr_suites.py).

  python scripts/stage1_vlm_token.py                      # write the cache
  python scripts/stage1_vlm_token.py --check 4            # recompute 4 scenes, compare
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from paths import DEVICE, QWEN_VL


def encoder(path, dev):
    from scene_mining.filter.lora_grounded_filter import QwenVisualEncoder
    enc = QwenVisualEncoder(path, lora_r=1, lora_alpha=128, device=str(dev), img_max_side=336)
    enc.model.to(dev); return enc


def encode(enc, ims, dev):
    from PIL import Image
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        return enc.encode([Image.open(p).convert("RGB") for p in ims])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--out", default="work_dirs/unified_filter/V_frozen_cache.pt")
    ap.add_argument("--qwen", default=QWEN_VL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--check", type=int, default=0, help="recompute this many scenes and compare with --out")
    a = ap.parse_args()
    dev = torch.device(DEVICE)
    toks = [str(t) for t in np.load(a.features, allow_pickle=False)["scene_tokens"]]
    enc = encoder(a.qwen, dev)
    if a.check:
        V = torch.load(a.out, map_location="cpu")
        ix = np.random.default_rng(0).choice(len(toks), a.check, replace=False)
        new = torch.cat([encode(enc, [Path(a.thumbs)/f"{toks[i]}.jpg"], dev).cpu() for i in ix])
        cos = F.cosine_similarity(new, V[ix], dim=-1)
        rel = (new - V[ix]).norm(dim=-1) / V[ix].norm(dim=-1)
        print(f"[v] {a.check} scenes on {dev}: cosine to cache min {float(cos.min()):.5f} mean {float(cos.mean()):.5f}; "
              f"relative L2 error max {float(rel.max()):.4f}")
        return
    V = torch.zeros(len(toks), enc.d_model)
    for i in range(0, len(toks), a.batch):
        V[i:i + a.batch] = encode(enc, [Path(a.thumbs)/f"{t}.jpg" for t in toks[i:i + a.batch]], dev).cpu()
        if (i // a.batch) % 10 == 0: print(f"  [v] {i + a.batch}/{len(toks)}", flush=True)
    torch.save(V, a.out); print(f"[v] wrote {a.out} {tuple(V.shape)}")


if __name__ == "__main__":
    main()
