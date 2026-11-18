#!/usr/bin/env python3
"""MILESTONE 23 - image embeddings for the CLIP and SigLIP (v1) baselines.
Same images and order as everywhere else: corpus (features_v2 scene order),
BDD100K (sorted names), val150 (sorted CAM_FRONT thumbnails)."""
from paths import DEVICE
from paths import EXT
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "scripts")
from annotate_ui import find_thumb
from PIL import Image
from transformers import AutoModel, AutoImageProcessor as AutoProcessor

dev = torch.device(DEVICE)
z = np.load("work_dirs/unified_filter/features_v2.npz", allow_pickle=False)
th = Path("work_dirs/unified_filter/scene_thumbs")
broot = Path(EXT + "/bdd100k")
SETS = {"corpus": [find_thumb(th, str(t)) for t in z["scene_tokens"]],
        "bdd": [broot/"images"/n for n in sorted(json.load(open(broot/"labels.json")))],
        "val150": sorted(Path("work_dirs/val150_thumbs").glob("*.jpg"))}
ft = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
for name, repo in [("siglip1", "google/siglip-base-patch16-224")] if Path("work_dirs/emb_clip.npz").exists() else [("clip", "openai/clip-vit-base-patch32"), ("siglip1", "google/siglip-base-patch16-224")]:
    m = AutoModel.from_pretrained(repo, use_safetensors=True).to(dev).eval()
    pr = AutoProcessor.from_pretrained(repo)
    out = {}
    for sname, paths in SETS.items():
        X = []
        with torch.no_grad():
            for i in range(0, len(paths), 64):
                ims = [Image.open(p).convert("RGB") for p in paths[i:i+64]]
                X.append(F.normalize(ft(m.get_image_features(**pr(images=ims, return_tensors="pt").to(dev))), dim=-1).cpu().numpy())
        out[sname] = np.concatenate(X); print(name, sname, out[sname].shape, flush=True)
    np.savez_compressed(f"work_dirs/emb_{name}.npz", **out)
    del m; torch.cuda.empty_cache()
