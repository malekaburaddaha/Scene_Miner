#!/usr/bin/env python3
"""MILESTONE 2 - cache the signals SceneMiner v1 never saw.

v1 fused BEVFormer e / tags t / risk r / Qwen v. Against human labels it lost to
two things it did not have access to:
  * zero-shot SigLIP2 (0.913 mAP on droplets vs our 0.908, 0.843 on wet_road
    vs our 0.225)
  * the continuous VLM logit margins (0.733 mean vs our 0.420)
Both are cheap and cacheable. This script writes them into a v2 feature store.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="work_dirs/unified_filter/features_v2.npz")
    a = ap.parse_args()
    W = Path(a.work)
    z = np.load(W/"features_overnight.npz", allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]

    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    import torch.nn.functional as F
    P = "google/siglip2-base-patch16-224"
    dev = torch.device("cuda")
    m = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    ip = AutoImageProcessor.from_pretrained(P)
    feat = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    d = m.config.text_config.hidden_size
    S = torch.zeros(len(toks), d, device=dev)
    with torch.no_grad():
        for i in range(0, len(toks), a.batch):
            ims = [Image.open(Path(a.thumbs)/f"{t}.jpg").convert("RGB")
                   for t in toks[i:i+a.batch]]
            S[i:i+a.batch] = feat(m.get_image_features(**ip(images=ims, return_tensors="pt").to(dev)))
            if (i//a.batch) % 10 == 0: print(f"  [siglip] {i+len(ims)}/{len(toks)}", flush=True)
    Sn = F.normalize(S, dim=-1).cpu().numpy().astype(np.float32)

    # 4 continuous VLM logit margins (sharpened low_contrast where available)
    vlm = json.load(open(W/"binary_flags_with_logits.json"))
    strict_p = W/"lowcontrast_strict.json"
    strict = json.load(open(strict_p)) if strict_p.exists() else {}
    ATTRS = ["droplets", "glare", "wet_road", "low_contrast"]
    M = np.zeros((len(toks), len(ATTRS)), dtype=np.float32)
    for i, t in enumerate(toks):
        for j, at in enumerate(ATTRS):
            if at == "low_contrast" and strict:
                M[i, j] = strict.get(t, {}).get("low_contrast_strict_margin", -9.0)
            else:
                M[i, j] = vlm.get(t, {}).get(at+"_margin", -9.0)
    # standardise margins so they enter the head on a comparable scale
    Mz = (M - M.mean(0, keepdims=True)) / (M.std(0, keepdims=True) + 1e-6)

    out = {k: z[k] for k in z.files}
    out["siglip_img"] = Sn
    out["vlm_margin"] = M
    out["vlm_margin_z"] = Mz
    out["margin_attrs"] = np.array(ATTRS)
    np.savez_compressed(a.out, **out)
    print(f"[v2] wrote {a.out}")
    print(f"  siglip_img  {Sn.shape}  vlm_margin {M.shape} (attrs {ATTRS})")
    print(f"  margin means {M.mean(0).round(2)}  sds {M.std(0).round(2)}")

if __name__ == "__main__":
    main()
