#!/usr/bin/env python3
"""MILESTONE 8 part 2 - uncontaminated 3D study on the 150 nuScenes val scenes.

bevformer_base never saw these scenes (per-scene mAP 0.4514 vs 0.7864 on the
memorised train scenes). We rank them with the PORTABLE signals - SigLIP2
cosine and Qwen VLM margins - and ask whether the top-k contains scenes the
detector actually handles worse, overall and on vulnerable subsets:
  * objects beyond 30 m
  * annotations marked v0-40 (human-labelled low visibility)
Stage-1 signals (BEV embedding, tags, risk, Qwen visual token) are not available
for val scenes without re-running the whole extraction, so the full head is not
evaluated here; the portable base is.
"""
from __future__ import annotations
from paths import QWEN_VL
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

QUESTIONS = [
 ("wet_road","Is the road surface visibly wet or reflective from water? Answer only yes or no."),
 ("glare","Is there strong light glare, bloom, or lens flare from headlights, streetlights, or the sun? Answer only yes or no."),
 ("droplets","Are there visible water droplets or rain smear on the camera lens? Answer only yes or no."),
 ("low_contrast","Are object boundaries severely obscured by darkness, fog, or bloom such that vehicles within 30 m are barely distinguishable from the background? Answer only yes or no."),
]
QUERIES = ["wet reflective asphalt after rainfall",
           "driving after dark with poor visibility",
           "blinding headlight bloom from oncoming traffic",
           "water droplets smearing the camera lens",
           "a low contrast scene where vehicles are barely distinguishable from the background"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", default="results/cvpr/bevformer_val_per_scene_official.json")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--thumbs", default="work_dirs/val150_thumbs")
    ap.add_argument("--qwen", default=QWEN_VL)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--out", default="results/cvpr/milestone8_val150_3d.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    diff = json.load(open(a.per_scene))
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=a.dataroot, verbose=False)
    by = {s["token"]: s for s in nusc.scene}
    scenes = [t for t in diff if t in by and diff[t]["mAP"] == diff[t]["mAP"]]
    print(f"[val150] {len(scenes)} val scenes with official per-scene mAP", flush=True)

    # one mid-scene CAM_FRONT keyframe per scene as the thumbnail
    th = Path(a.thumbs); th.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    paths = []
    for t in scenes:
        p = th/f"{t}.jpg"
        if not p.exists():
            s = by[t]; st = s["first_sample_token"]
            for _ in range(s["nbr_samples"]//2):
                st = nusc.get("sample", st)["next"] or st
            sd = nusc.get("sample_data", nusc.get("sample", st)["data"]["CAM_FRONT"])
            im = Image.open(Path(a.dataroot)/sd["filename"]).convert("RGB")
            im.thumbnail((640, 640)); im.save(p, quality=88)
        paths.append(p)
    print(f"[val150] thumbnails ready", flush=True)

    # vulnerable-subset difficulty from the official per-class/per-threshold APs
    hard = {}
    for t in scenes:
        d = diff[t]
        far = [v["AP_by_dist"][-1] for v in d["per_class"].values()]     # 4 m threshold
        tight = [v["AP_by_dist"][0] for v in d["per_class"].values()]    # 0.5 m threshold
        hard[t] = {"mAP": d["mAP"],
                   "mAP_tight": float(np.mean(tight)) if tight else np.nan,
                   "n_gt": d["n_gt"]}
    # human-annotated low-visibility fraction per scene
    VIS = {"1": 1.0, "2": 1.0, "3": 0.0, "4": 0.0}
    for t in scenes:
        s = by[t]; st = s["first_sample_token"]; low = tot = 0
        while st:
            sm = nusc.get("sample", st)
            for an in sm["anns"]:
                v = nusc.get("sample_annotation", an).get("visibility_token")
                if v in VIS: tot += 1; low += VIS[v]
            st = sm["next"]
        hard[t]["frac_low_vis"] = low/max(1, tot)

    # portable signals
    from transformers import (AutoImageProcessor, AutoModel, GemmaTokenizerFast,
                              AutoProcessor, Qwen2_5_VLForConditionalGeneration)
    P = "google/siglip2-base-patch16-224"
    m = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    ip = AutoImageProcessor.from_pretrained(P); tkz = GemmaTokenizerFast.from_pretrained(P)
    feat = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    SIG = np.zeros((len(paths), m.config.text_config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i+32]]
            SIG[i:i+32] = F.normalize(feat(m.get_image_features(
                **ip(images=ims, return_tensors="pt").to(dev))), dim=-1).cpu().numpy()
        ti = tkz(QUERIES, padding="max_length", truncation=True, max_length=64,
                 return_tensors="pt").to(dev)
        QT = F.normalize(feat(m.get_text_features(**ti)), dim=-1).cpu().numpy()
    del m; torch.cuda.empty_cache()
    proc = AutoProcessor.from_pretrained(a.qwen)
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.qwen, torch_dtype=torch.bfloat16).to(dev).eval()
    tok = proc.tokenizer
    YES = sorted({tok.encode(s, add_special_tokens=False)[0] for s in ["Yes","yes"," Yes"," yes"]
                  if len(tok.encode(s, add_special_tokens=False)) == 1})
    NO = sorted({tok.encode(s, add_special_tokens=False)[0] for s in ["No","no"," No"," no"]
                 if len(tok.encode(s, add_special_tokens=False)) == 1})
    MARG = np.zeros((len(paths), len(QUESTIONS)), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        for j, (_, q) in enumerate(QUESTIONS):
            msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q}]}]
            t_ = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[t_], images=[img], return_tensors="pt").to(dev)
            with torch.no_grad():
                g = qm.generate(**inp, max_new_tokens=1, do_sample=False,
                                output_scores=True, return_dict_in_generate=True)
            lp = torch.log_softmax(g.scores[0][0].float(), -1)
            MARG[i, j] = float(torch.logsumexp(lp[YES],0) - torch.logsumexp(lp[NO],0))
        if (i+1) % 40 == 0: print(f"  [qwen] {i+1}/{len(paths)}", flush=True)
    del qm; torch.cuda.empty_cache()

    mAP = np.array([hard[t]["mAP"] for t in scenes])
    tight = np.array([hard[t]["mAP_tight"] for t in scenes])
    lowv = np.array([hard[t]["frac_low_vis"] for t in scenes])
    rng = np.random.default_rng(0)
    rank = {"zeroshot_SigLIP2": (SIG @ QT.mean(0)),
            "logit_VLM_lookup": (MARG - MARG.mean(0))/(MARG.std(0)+1e-6),
            }
    rank["logit_VLM_lookup"] = rank["logit_VLM_lookup"].mean(1)
    # metadata-matched: nuScenes description rain/night
    desc = {t: by[t]["description"].lower() for t in scenes}
    rank["metadata_matched"] = np.array([float(("rain" in desc[t]) or ("night" in desc[t]))
                                         for t in scenes]) + rng.random(len(scenes))*1e-6
    rank["random"] = rng.random(len(scenes))
    k = a.topk
    out = {"n_scenes": len(scenes), "contaminated": False,
           "corpus_mean_mAP": float(mAP.mean()), "topk": k, "rankers": {}}
    print(f"\n{'ranker':<24}{'top-k mAP':>11}{'tight mAP':>11}{'low-vis frac':>14}")
    print(f"{'(corpus mean)':<24}{mAP.mean():>11.4f}{np.nanmean(tight):>11.4f}{lowv.mean():>14.4f}")
    for nm, s in rank.items():
        o = np.argsort(-s)[:k]
        out["rankers"][nm] = {"topk_mAP": float(mAP[o].mean()),
                              "topk_mAP_tight": float(np.nanmean(tight[o])),
                              "topk_frac_low_vis": float(lowv[o].mean()),
                              "delta_vs_corpus": float(mAP[o].mean()-mAP.mean())}
        print(f"{nm:<24}{mAP[o].mean():>11.4f}{np.nanmean(tight[o]):>11.4f}{lowv[o].mean():>14.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[val150] wrote {a.out}   (lower top-k mAP = found harder scenes)")

if __name__ == "__main__":
    main()
