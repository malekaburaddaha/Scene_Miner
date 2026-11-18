#!/usr/bin/env python3
"""MILESTONE 23 - per-image cost of one Qwen2.5-VL caption (the CARIM-style
baseline's offline step), timed on 19 images spread over the three corpora.
-> results/cvpr/milestone23_caption_timing.json (does not touch the caption shards)"""
from paths import DEVICE
from paths import QWEN_VL
import sys, time, json, numpy as np, torch
sys.path.insert(0, "scripts")
from caption_scenes_qwen import images, PROMPT
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

Q = QWEN_VL; dev = torch.device(DEVICE)
proc = AutoProcessor.from_pretrained(Q)
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(Q, torch_dtype=torch.bfloat16).to(dev).eval()
ts = []
for j, (src, key, path) in enumerate(images()[::133][:20]):
    img = Image.open(path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    torch.cuda.synchronize(); a = time.perf_counter()
    inp = proc(text=[t], images=[img], return_tensors="pt").to(dev)
    with torch.no_grad(): m.generate(**inp, max_new_tokens=96, do_sample=False)
    torch.cuda.synchronize()
    if j > 0: ts.append(time.perf_counter() - a)       # first call is warm-up
r = {"caption_s_per_image_median": float(np.median(ts)), "caption_s_per_image_mean": float(np.mean(ts)),
     "n_timed": len(ts), "gpu": "A40", "model": "Qwen2.5-VL-3B bf16, greedy, 96 new tokens"}
print(r); json.dump(r, open("results/cvpr/milestone23_caption_timing.json", "w"), indent=1)
