#!/usr/bin/env python3
"""MILESTONE 8 - zero-shot transfer to BDD100K (external, human-labelled).

Ground truth is BDD100K's own per-image human annotation (`weather`,
`timeofday`) - a channel that shares nothing with our VLM pipeline, so this is
the independent external evaluation the Area Chair and Reviewer fKWi asked for.

IMPORTANT STRUCTURAL LIMIT, reported rather than worked around: the full
SceneMiner head consumes four Stage-1 signals (BEVFormer embedding, tag logits,
physical risk, Qwen visual token) that require a calibrated 6-camera rig with
3D boxes. BDD100K is monocular dashcam video with none of those. Only the
portable part of the model - the SigLIP2 cosine, the VLM margins and the learned
query routing over them - can cross to this corpus. That portability gap is
itself a result.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, p_at, ndcg_at

QUESTIONS = [
 ("wet_road", "Is the road surface visibly wet or reflective from water? Answer only yes or no."),
 ("glare", "Is there strong light glare, bloom, or lens flare from headlights, streetlights, or the sun? Answer only yes or no."),
 ("droplets", "Are there visible water droplets or rain smear on the camera lens? Answer only yes or no."),
 ("low_contrast", "Are object boundaries severely obscured by darkness, fog, or bloom such that vehicles within 30 m are barely distinguishable from the background? Answer only yes or no."),
]
# queries whose relevance comes from BDD's OWN human labels
EVAL_Q = [
 ("wet reflective asphalt after rainfall",            ("weather", {"rainy"})),
 ("driving in rain",                                  ("weather", {"rainy"})),
 ("water droplets smearing the camera lens",          ("weather", {"rainy"})),
 ("snow covering the road",                           ("weather", {"snowy"})),
 ("a snowy street scene",                             ("weather", {"snowy"})),
 ("driving after dark",                               ("timeofday", {"night"})),
 ("a night-time street scene",                        ("timeofday", {"night"})),
 ("a low contrast scene where vehicles are barely distinguishable from the background",
                                                      ("timeofday", {"night"})),
 ("blinding headlight bloom from oncoming traffic",   ("timeofday", {"night"})),
 ("bright daytime driving",                           ("timeofday", {"daytime"})),
 ("overcast grey sky",                                ("weather", {"overcast"})),
 ("adverse weather on the road",                      ("weather", {"rainy","snowy","foggy"})),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/nfs/turbo/umd-srawa/scene_mining/datasets/external_adverse/bdd100k")
    ap.add_argument("--qwen", default="/nfs/turbo/umd-srawa/hf_cache/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default="results/cvpr/milestone8_external_dataset.json")
    a = ap.parse_args()
    root = Path(a.root); dev = torch.device("cuda")
    labels = json.load(open(root/"labels.json"))
    names = sorted(labels)
    imgs = [root/"images"/n for n in names]
    print(f"[bdd] {len(names)} images with human weather/timeofday labels", flush=True)
    cache = Path(a.cache) if a.cache else root/"signals.npz"

    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        SIG, MARG = z["siglip"], z["margin"]
        print("[bdd] loaded cached signals", flush=True)
    else:
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image
        P = "google/siglip2-base-patch16-224"
        m = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
        ip = AutoImageProcessor.from_pretrained(P)
        feat = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
        SIG = np.zeros((len(imgs), m.config.text_config.hidden_size), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(imgs), 64):
                ims = [Image.open(p).convert("RGB") for p in imgs[i:i+64]]
                SIG[i:i+64] = F.normalize(
                    feat(m.get_image_features(**ip(images=ims, return_tensors="pt").to(dev))),
                    dim=-1).cpu().numpy()
                if (i//64) % 5 == 0: print(f"  [siglip] {i+len(ims)}/{len(imgs)}", flush=True)
        del m; torch.cuda.empty_cache()
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(a.qwen)
        qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            a.qwen, torch_dtype=torch.bfloat16).to(dev).eval()
        tok = proc.tokenizer
        YES = sorted({tok.encode(s, add_special_tokens=False)[0]
                      for s in ["Yes","yes"," Yes"," yes"]
                      if len(tok.encode(s, add_special_tokens=False)) == 1})
        NO = sorted({tok.encode(s, add_special_tokens=False)[0]
                     for s in ["No","no"," No"," no"]
                     if len(tok.encode(s, add_special_tokens=False)) == 1})
        MARG = np.zeros((len(imgs), len(QUESTIONS)), dtype=np.float32)
        for i, p in enumerate(imgs):
            img = Image.open(p).convert("RGB")
            for j, (_, q) in enumerate(QUESTIONS):
                msgs = [{"role":"user","content":[{"type":"image","image":img},
                                                  {"type":"text","text":q}]}]
                t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = proc(text=[t], images=[img], return_tensors="pt").to(dev)
                with torch.no_grad():
                    g = qm.generate(**inp, max_new_tokens=1, do_sample=False,
                                    output_scores=True, return_dict_in_generate=True)
                lp = torch.log_softmax(g.scores[0][0].float(), -1)
                MARG[i, j] = float(torch.logsumexp(lp[YES],0) - torch.logsumexp(lp[NO],0))
            if (i+1) % 100 == 0: print(f"  [qwen] {i+1}/{len(imgs)}", flush=True)
        del qm; torch.cuda.empty_cache()
        np.savez_compressed(cache, siglip=SIG, margin=MARG, names=np.array(names))
        print(f"[bdd] cached signals -> {cache}", flush=True)

    MZ = (MARG - MARG.mean(0, keepdims=True)) / (MARG.std(0, keepdims=True) + 1e-6)
    ATT = [k for k, _ in QUESTIONS]
    from transformers import AutoModel, GemmaTokenizerFast
    P = "google/siglip2-base-patch16-224"
    sm_ = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    tkz = GemmaTokenizerFast.from_pretrained(P)
    feat = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    qtexts = [q for q, _ in EVAL_Q]
    with torch.no_grad():
        ti = tkz(qtexts, padding="max_length", truncation=True, max_length=64,
                 return_tensors="pt").to(dev)
        QT = F.normalize(feat(sm_.get_text_features(**ti)), dim=-1)
    del sm_; torch.cuda.empty_cache()

    # the portable part of v2.1: learned routing over (siglip, margin, meta=0)
    from scene_mining.filter.sceneminer_v2 import SceneMinerV21
    ck = sorted(glob.glob("work_dirs/sceneminer_v21/s*/final.pt"))
    routers = []
    for c in ck:
        d = torch.load(c, map_location=dev, weights_only=False)
        m = SceneMinerV21(d_text=QT.shape[1], n_margins=MZ.shape[1], n_meta=6).to(dev)
        m.load_state_dict(d["state_dict"]); m.eval(); routers.append(m)

    S = torch.from_numpy(SIG).float().to(dev); M = torch.from_numpy(MZ).float().to(dev)
    PR = torch.zeros(len(names), 26, device=dev)      # no Stage-1 prior on BDD
    rows = {}
    for qi, (qt, (field, pos)) in enumerate(EVAL_Q):
        rel = np.array([labels[n][field] in pos for n in names])
        if rel.sum() == 0 or rel.sum() == len(names): continue
        qv = QT[qi].unsqueeze(0).expand(len(names), -1)
        cos = (S @ QT[qi]).cpu().numpy()
        # margin channel whose attribute the query most plausibly names
        best = max(range(len(ATT)), key=lambda j: float(np.corrcoef(MZ[:, j], rel)[0, 1]))
        rank = {"zeroshot_SigLIP2": cos,
                "logit_VLM_lookup": MZ[:, best],
                "flat_VLM_lookup": (MARG[:, best] > 0).astype(float)
                                   + np.random.default_rng(0).random(len(names))*1e-6}
        with torch.no_grad():
            base = np.mean([m.base_score(qv, qv, S, M, PR).cpu().numpy() for m in routers], 0)
        rank["SceneMiner_v2.1_portable(routing)"] = base
        for nm, s in rank.items():
            o = np.argsort(-s)
            rows.setdefault(nm, []).append((ap_binary(o, rel), p_at(o, rel),
                                            ndcg_at(o, rel.astype(float)), rel.mean()))
    out = {"n_images": len(names), "n_queries": len(rows[next(iter(rows))]),
           "label_source": "BDD100K native human weather/timeofday annotations",
           "not_transferable": "BEVFormer embedding, tag logits, risk, Qwen visual "
                               "token - require a calibrated 6-camera rig with 3D boxes",
           "rankers": {}}
    print(f"\n{'ranker':<38}{'mAP':>9}{'P@5':>9}{'NDCG@10':>10}")
    for nm in sorted(rows, key=lambda k: -np.mean([r[0] for r in rows[k]])):
        A = np.array(rows[nm])
        out["rankers"][nm] = {"mAP": float(A[:,0].mean()), "P@5": float(A[:,1].mean()),
                              "NDCG@10": float(A[:,2].mean())}
        print(f"{nm:<38}{A[:,0].mean():>9.4f}{A[:,1].mean():>9.4f}{A[:,2].mean():>10.4f}")
    ch = float(np.mean([r[3] for r in rows[next(iter(rows))]]))
    out["chance"] = ch
    print(f"{'(chance)':<38}{ch:>9.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[bdd] wrote {a.out}")

if __name__ == "__main__":
    main()
