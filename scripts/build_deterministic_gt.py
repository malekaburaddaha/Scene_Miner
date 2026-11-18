#!/usr/bin/env python3
"""Phase 1 — Deterministic, high-precision ground truth for the 1,508-scene
bank. Replaces zero-shot VLM pseudo-labels wherever exact metadata exists.

Attributes per scene:
  density  : from 3D GT boxes — mean vehicle count within 50 m of ego
             (nuScenes: devkit sample_annotation vs ego_pose;
              Waymo: boxes.npz centers, type==1) →
             dense (>10) / moderate (4–10) / sparse (<4)
  lighting : explicit metadata — nuScenes scene.description contains
             'night' → night; Waymo meta.time_of_day == 'Night' → night;
             'Dawn/Dusk' → dusk; else day.
  weather  : nuScenes description contains 'rain' → rain;
             Waymo meta.weather == 'rain' → rain; else dry.
  wet_road / glare / droplets : UNSTRUCTURED visual conditions — filled by
             a separate high-precision BINARY VLM verification pass
             (--vlm-pass): three yes/no questions per scene, answers
             parsed strictly; far more reliable than open captioning.

Output: work_dirs/unified_filter/deterministic_gt.json
"""
from __future__ import annotations
from paths import QWEN_VL

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


# ── nuScenes: exact vehicle counts + description attributes ────────────
def nusc_attributes(scene_tokens: list[str], dataroot: str):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=dataroot, verbose=False)
    by_tok = {s["token"]: s for s in nusc.scene}
    out = {}
    for tok in scene_tokens:
        s = by_tok.get(tok)
        if s is None:
            continue
        desc = s["description"].lower()
        lighting = "night" if "night" in desc else "day"
        weather = "rain" if "rain" in desc else "dry"
        counts = []
        st = s["first_sample_token"]
        while st:
            samp = nusc.get("sample", st)
            sd = nusc.get("sample_data", samp["data"]["LIDAR_TOP"])
            ego = np.array(nusc.get("ego_pose", sd["ego_pose_token"])["translation"][:2])
            n = 0
            for at in samp["anns"]:
                ann = nusc.get("sample_annotation", at)
                if ann["category_name"].startswith("vehicle."):
                    p = np.array(ann["translation"][:2])
                    if np.linalg.norm(p - ego) <= 50.0:
                        n += 1
            counts.append(n)
            st = samp["next"]
        v50 = float(np.mean(counts)) if counts else 0.0
        out[tok] = {"veh_50m": v50, "lighting": lighting, "weather": weather}
    return out


# ── Waymo: exact counts from boxes.npz + explicit meta ─────────────────
def waymo_attributes(chunk_tokens: list[str], roots: list[str]):
    # map segment → extracted dir
    seg_dir = {}
    for r in roots:
        for d in Path(r).iterdir():
            if d.is_dir():
                seg_dir[d.name] = d
    out = {}
    for tok in chunk_tokens:
        seg, cidx = tok.split("#c")
        d = seg_dir.get(seg)
        if d is None:
            continue
        frames = sorted(p for p in d.iterdir() if p.is_dir())
        chunk = frames[int(cidx) * 4:(int(cidx) + 1) * 4]
        counts = []
        meta = None
        for f in chunk:
            try:
                b = np.load(f / "boxes.npz", allow_pickle=True)
                veh = (b["types"] == 1)
                dist = np.linalg.norm(b["centers"][:, :2], axis=1)
                counts.append(int((veh & (dist <= 50.0)).sum()))
                if meta is None:
                    meta = json.loads((f / "meta.json").read_text())
            except Exception:
                continue
        if meta is None:
            continue
        tod = (meta.get("time_of_day") or "").lower()
        lighting = ("night" if "night" in tod
                    else "dusk" if "dawn" in tod or "dusk" in tod else "day")
        weather = "rain" if "rain" in (meta.get("weather") or "").lower() else "dry"
        out[tok] = {"veh_50m": float(np.mean(counts)) if counts else 0.0,
                    "lighting": lighting, "weather": weather}
    return out


def density_class(v: float) -> str:
    return "dense" if v > 10 else ("moderate" if v >= 4 else "sparse")


# ── Binary VLM verification pass for unstructured visual conditions ────
BINARY_QS = [
    ("wet_road", "Is the road surface visibly wet or reflective from water? "
                 "Answer only yes or no."),
    ("glare", "Is there strong light glare, bloom, or lens flare from "
              "headlights, streetlights, or the sun? Answer only yes or no."),
    ("droplets", "Are there visible water droplets or rain smear on the "
                 "camera lens? Answer only yes or no."),
]


def vlm_binary_pass(tokens, thumbs, model_path, device, start=0, end=None):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image
    proc = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=str(device)).eval()
    out = {}
    toks = tokens[start:end]
    for i, tok in enumerate(toks):
        p = Path(thumbs) / f"{tok}.jpg"
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        flags = {}
        for key, q in BINARY_QS:
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img}, {"type": "text", "text": q}]}]
            ti = proc.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
            inp = proc(text=[ti], images=[img], return_tensors="pt").to(device)
            with torch.no_grad():
                oid = model.generate(**inp, max_new_tokens=4, do_sample=False)
            ans = proc.batch_decode(oid[:, inp["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0].lower()
            flags[key] = bool(re.search(r"\byes\b", ans))
        out[tok] = flags
        if (i + 1) % 100 == 0:
            print(f"  [vlm-binary {start}] {i+1}/{len(toks)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work_dirs/unified_filter/features_overnight.npz")
    ap.add_argument("--nusc-root", default="data/nuscenes")
    ap.add_argument("--waymo-roots", nargs="+",
                    default=[f"work_dirs/waymo_zeroshot/extracted_all_{i}" for i in range(3)])
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--model", default=QWEN_VL)
    ap.add_argument("--out", default="work_dirs/unified_filter/deterministic_gt.json")
    ap.add_argument("--vlm-pass", action="store_true",
                    help="Run ONLY the binary VLM flags (sharded via --start/--end).")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    z = np.load(args.features, allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]
    dom = z["domain"]

    if args.vlm_pass:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        flags = vlm_binary_pass(toks, args.thumbs, args.model, device,
                                args.start, args.end)
        Path(args.out).write_text(json.dumps(flags, indent=1))
        print(f"[det-gt] wrote {len(flags)} binary-flag records → {args.out}")
        return

    nusc_toks = [t for t, d in zip(toks, dom) if d == 0]
    waymo_toks = [t for t, d in zip(toks, dom) if d == 1]
    print(f"[det-gt] nuScenes {len(nusc_toks)} | Waymo {len(waymo_toks)}")
    gt = {}
    gt.update(nusc_attributes(nusc_toks, args.nusc_root))
    print(f"[det-gt] nuScenes attributes done ({len(gt)})")
    gt.update(waymo_attributes(waymo_toks, args.waymo_roots))
    print(f"[det-gt] + Waymo attributes ({len(gt)})")
    for tok in gt:
        gt[tok]["density"] = density_class(gt[tok]["veh_50m"])
    hist = Counter((v["lighting"], v["density"], v["weather"]) for v in gt.values())
    print("[det-gt] top combos:", hist.most_common(8))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(gt, indent=1))
    print(f"[det-gt] wrote {args.out} ({len(gt)} scenes)")


if __name__ == "__main__":
    main()
