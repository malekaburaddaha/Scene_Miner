#!/usr/bin/env python3
"""Stage 1 (single-camera path) - recompute a sample of the cached signals and compare.

For N random corpus scenes, from the CAM_FRONT thumbnail alone:
  * SigLIP2 ViT-B/16 image embedding s (scripts/cache_v2_signals.py)
  * the four zero-shot Qwen2.5-VL-3B margins m = log P(yes) - log P(no) of the first
    generated token: wet road, glare, droplets (scripts/extract_vlm_logits.py) and the
    strict low-contrast question (scripts/extract_lowcontrast_strict.py)
against work_dirs/unified_filter/features_v2.npz. The 26-d prior is checked exactly by
scripts/stage1_build_prior.py --check and the VLM token by scripts/stage1_vlm_token.py --check.
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from paths import DEVICE, QWEN_VL
from extract_vlm_logits import QUESTIONS as Q3, yes_no_ids
from extract_lowcontrast_strict import QUESTIONS as QLC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--qwen", default=QWEN_VL)
    ap.add_argument("-n", type=int, default=4)
    a = ap.parse_args()
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image
    dev = torch.device(DEVICE)
    z = np.load(a.features, allow_pickle=False); toks = [str(t) for t in z["scene_tokens"]]
    ix = np.random.default_rng(0).choice(len(toks), a.n, replace=False)
    ims = [Image.open(Path(a.thumbs)/f"{toks[i]}.jpg").convert("RGB") for i in ix]
    P = "google/siglip2-base-patch16-224"
    m = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval(); ip = AutoImageProcessor.from_pretrained(P)
    with torch.no_grad():
        o = m.get_image_features(**ip(images=ims, return_tensors="pt").to(dev))
        S = F.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1).cpu()
    cos = (S * torch.from_numpy(z["siglip_img"][ix])).sum(-1)
    print(f"[check] SigLIP2 image embedding, {a.n} scenes: cosine to cache min {float(cos.min()):.5f}", flush=True)
    del m
    proc = AutoProcessor.from_pretrained(a.qwen)
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.qwen, torch_dtype=torch.bfloat16).to(dev).eval()
    YES, NO = yes_no_ids(proc.tokenizer)
    qs = {k: q for k, q in Q3}; qs["low_contrast"] = QLC[0][1]
    order = [str(x) for x in z["margin_attrs"]]                 # droplets, glare, wet_road, low_contrast
    M = np.zeros((a.n, 4))
    for r, img in enumerate(ims):
        for j, at in enumerate(order):
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": qs[at]}]}]
            inp = proc(text=[proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)], images=[img],
                       return_tensors="pt").to(dev)
            with torch.no_grad():
                g = qm.generate(**inp, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True)
            lp = torch.log_softmax(g.scores[0][0].float(), -1)
            M[r, j] = float(torch.logsumexp(lp[YES], 0) - torch.logsumexp(lp[NO], 0))
    C = z["vlm_margin"][ix]
    print(f"[check] Qwen2.5-VL margins ({', '.join(order)}), {a.n} scenes: max |diff| {np.abs(M - C).max():.3f} "
          f"(cached range {C.min():.2f}..{C.max():.2f}); sign agreement {(np.sign(M) == np.sign(C)).mean():.3f}", flush=True)
    for r in range(a.n): print("   recomputed", M[r].round(3), " cached", C[r].round(3))


if __name__ == "__main__":
    main()
