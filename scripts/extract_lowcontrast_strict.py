#!/usr/bin/env python3
"""TASK 3 - Qwen2.5-VL binary attribute extraction WITH Yes/No log-prob margins.

Extracts droplets, glare, wet_road AND the missing 4th attribute low_contrast,
saving for each: the binary decision and the continuous log-prob margin
log P(Yes) - log P(No) at the first generated token. The margin is what lets the
cached-attribute baseline be tested in its STRONGEST form (continuous ranking),
which binary_flags_{a,b} could not support (pure bools, disjoint shards).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

QUESTIONS = [
    # STRICTER: the original low_contrast prompt hit 55.6% prevalence with
    # 288/600 borderline margins - it was firing on any slightly flat image.
    # This version demands a consequence for perception, not an aesthetic.
    ("low_contrast_strict",
     "Are object boundaries severely obscured by darkness, fog, or bloom such "
     "that vehicles within 30 m are barely distinguishable from the background? "
     "Answer only yes or no."),
]

def yes_no_ids(tok):
    yes, no = set(), set()
    for s in ["Yes", "yes", " Yes", " yes", "YES"]:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1: yes.add(ids[0])
    for s in ["No", "no", " No", " no", "NO"]:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1: no.add(ids[0])
    return sorted(yes), sorted(no)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work_dirs/unified_filter/features_overnight.npz")
    ap.add_argument("--thumbs",   default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--model",    default="/nfs/turbo/umd-srawa/hf_cache/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out",      required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image
    dev = torch.device("cuda")
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    tok = proc.tokenizer
    YES, NO = yes_no_ids(tok)
    assert YES and NO, "could not resolve single-token Yes/No ids"
    print(f"[shard {args.shard}] yes_ids={YES} no_ids={NO}", flush=True)

    z = np.load(args.features, allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]
    mine = toks[args.shard::args.nshards]
    print(f"[shard {args.shard}] {len(mine)} scenes x {len(QUESTIONS)} questions", flush=True)

    out, t0 = {}, time.perf_counter()
    for i, tkn in enumerate(mine):
        p = Path(args.thumbs) / f"{tkn}.jpg"
        if not p.exists(): continue
        img = Image.open(p).convert("RGB")
        rec = {}
        for key, q in QUESTIONS:
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=[img], return_tensors="pt").to(dev)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=1, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True)
            logits = gen.scores[0][0].float()                 # (vocab,)
            lp = torch.log_softmax(logits, dim=-1)
            # aggregate probability mass over casing variants, then take log
            p_yes = torch.logsumexp(lp[YES], dim=0).item()
            p_no  = torch.logsumexp(lp[NO],  dim=0).item()
            rec[key] = bool(p_yes > p_no)
            rec[key + "_margin"] = float(p_yes - p_no)        # >0 => yes
            rec[key + "_logp_yes"] = float(p_yes)
        out[tkn] = rec
        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"[shard {args.shard}] {i+1}/{len(mine)}  {el:.0f}s  "
                  f"eta {el/(i+1)*(len(mine)-i-1):.0f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[shard {args.shard}] wrote {args.out} ({len(out)} scenes) "
          f"in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
