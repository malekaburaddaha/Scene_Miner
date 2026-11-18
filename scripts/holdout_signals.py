#!/usr/bin/env python3
"""MILESTONE 28 - frozen Stage-1 signals for the untouched BDD100K holdout.

--part margins --shard i --nshard n : Qwen2.5-VL yes/no log-odds for the four
    QUESTIONS of eval_external_bdd100k.py (same prompts and token sets), one shard.
--part merge : SigLIP2 / CLIP / SigLIP image embeddings and the merged margins
    -> <holdout>/signals.npz (same layout as the 1,000-image BDD100K cache) and
    work_dirs/emb_{clip,siglip1}_holdout.npz. No query is scored here.
"""
from paths import DEVICE
from paths import EXT, QWEN_VL
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "scripts")
from eval_external_bdd100k import QUESTIONS

H = Path(EXT + "/bdd100k_holdout")
dev = torch.device(DEVICE)


def margins(shard, nshard, qwen):
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    names = sorted(json.load(open(H/"labels.json")))[shard::nshard]
    proc = AutoProcessor.from_pretrained(qwen)
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen, torch_dtype=torch.bfloat16).to(dev).eval()
    tok = proc.tokenizer
    one = lambda ss: sorted({tok.encode(s, add_special_tokens=False)[0] for s in ss
                             if len(tok.encode(s, add_special_tokens=False)) == 1})
    YES, NO = one(["Yes", "yes", " Yes", " yes"]), one(["No", "no", " No", " no"])
    out = np.zeros((len(names), len(QUESTIONS)), np.float32)
    for i, n in enumerate(names):
        img = Image.open(H/"images"/n).convert("RGB")
        for j, (_, q) in enumerate(QUESTIONS):
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
            t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[t], images=[img], return_tensors="pt").to(dev)
            with torch.no_grad():
                g = qm.generate(**inp, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True)
            lp = torch.log_softmax(g.scores[0][0].float(), -1)
            out[i, j] = float(torch.logsumexp(lp[YES], 0) - torch.logsumexp(lp[NO], 0))
        if (i + 1) % 100 == 0: print(f"[margins {shard}] {i+1}/{len(names)}", flush=True)
    np.savez(f"work_dirs/holdout_margins_shard{shard}.npz", names=np.array(names), margin=out)
    print(f"[margins {shard}] DONE", flush=True)


def merge():
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel
    names = sorted(json.load(open(H/"labels.json")))
    M = {}
    for f in glob.glob("work_dirs/holdout_margins_shard*.npz"):
        z = np.load(f); M.update(dict(zip([str(x) for x in z["names"]], z["margin"])))
    marg = np.stack([M[n] for n in names])
    ft = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    embs = {}
    for key, repo in [("siglip", "google/siglip2-base-patch16-224"), ("clip", "openai/clip-vit-base-patch32"),
                      ("siglip1", "google/siglip-base-patch16-224")]:
        m = (CLIPModel if key == "clip" else AutoModel).from_pretrained(repo, use_safetensors=True).to(dev).eval()
        ip = AutoImageProcessor.from_pretrained(repo); X = []
        with torch.no_grad():
            for i in range(0, len(names), 64):
                ims = [Image.open(H/"images"/n).convert("RGB") for n in names[i:i+64]]
                X.append(F.normalize(ft(m.get_image_features(**ip(images=ims, return_tensors="pt").to(dev))), dim=-1).cpu().numpy())
        embs[key] = np.concatenate(X); del m; torch.cuda.empty_cache()
    np.savez_compressed(H/"signals.npz", siglip=embs["siglip"], margin=marg, names=np.array(names))
    np.savez_compressed("work_dirs/emb_clip_holdout.npz", holdout=embs["clip"])
    np.savez_compressed("work_dirs/emb_siglip1_holdout.npz", holdout=embs["siglip1"])
    print("merged", marg.shape, {k: v.shape for k, v in embs.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["margins", "merge"], required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--qwen", default=QWEN_VL)
    a = ap.parse_args()
    margins(a.shard, a.nshard, a.qwen) if a.part == "margins" else merge()
