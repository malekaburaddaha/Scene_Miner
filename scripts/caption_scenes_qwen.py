#!/usr/bin/env python3
"""MILESTONE 23 - rich VLM captions for the CARIM-style caption-retrieval baseline.

CARIM retrieves scenes by matching queries against VLM-generated captions. The
cached captions in this project are closed-vocabulary slots ("clear day, light
traffic, urban street") with no lens or object content, which would make a
caption baseline unfairly weak, so we generate one rich caption per image with
the same Qwen2.5-VL-3B used in Stage 1, greedy decoding, fixed prompt.
Images: the 1508 corpus thumbnails, the 1000 BDD100K images, the 150 val150
CAM_FRONT keyframes. Sharded: --shard i --nshard n.
"""
from paths import EXT, QWEN_VL
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "scripts")
from annotate_ui import find_thumb

NEUTRAL_PROMPT = ("Describe this driving scene concisely in two sentences, stating the lighting, weather, "
                  "road surface condition, traffic density, and visible road users or obstacles. Mention only "
                  "conditions and objects that are present.")
PROMPT = ("Describe this driving scene in two or three sentences. Cover the weather and lighting, "
          "the road surface, the condition of the camera view (water droplets, glare, blur, low "
          "visibility), and every kind of road user or object you can see (pedestrians, children, "
          "cyclists, motorcyclists, cars, trucks, buses, construction cones, barriers or workers).")


def images():
    z = np.load("work_dirs/unified_filter/features_v2.npz", allow_pickle=False)
    th = Path("work_dirs/unified_filter/scene_thumbs")
    out = [("corpus", str(t), str(find_thumb(th, str(t)))) for t in z["scene_tokens"]]
    broot = Path(EXT + "/bdd100k")
    out += [("bdd", n, str(broot/"images"/n)) for n in sorted(json.load(open(broot/"labels.json")))]
    out += [("val150", p.stem, str(p)) for p in sorted(Path("work_dirs/val150_thumbs").glob("*.jpg"))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True); ap.add_argument("--nshard", type=int, required=True)
    ap.add_argument("--qwen", default=QWEN_VL)
    ap.add_argument("--prompt", default="enumerating", choices=["enumerating", "neutral"],
                    help="neutral (Milestone 28): affirmative prompt that does not list the degradations")
    ap.add_argument("--items", default=None, help="optional JSON list of [src, key, path] to caption instead")
    a = ap.parse_args()
    prompt = NEUTRAL_PROMPT if a.prompt == "neutral" else PROMPT
    tag = "" if a.prompt == "enumerating" else "_neutral"
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    dev = torch.device("cuda")
    proc = AutoProcessor.from_pretrained(a.qwen)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.qwen, torch_dtype=torch.bfloat16).to(dev).eval()
    items = (json.load(open(a.items)) if a.items else images())[a.shard::a.nshard]
    outp = Path(f"work_dirs/qwen_captions{tag}{'_' + Path(a.items).stem if a.items else ''}_shard{a.shard}.json")
    done = json.load(open(outp)) if outp.exists() else {}
    for i, (src, key, path) in enumerate(items):
        k = f"{src}|{key}"
        if k in done: continue
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[t], images=[img], return_tensors="pt").to(dev)
        with torch.no_grad():
            g = m.generate(**inp, max_new_tokens=96, do_sample=False)
        done[k] = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        if (i + 1) % 50 == 0:
            outp.write_text(json.dumps(done)); print(f"[shard {a.shard}] {i+1}/{len(items)}", flush=True)
    outp.write_text(json.dumps(done)); print(f"[shard {a.shard}] DONE {len(done)}", flush=True)


if __name__ == "__main__":
    main()
