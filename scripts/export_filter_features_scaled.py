#!/usr/bin/env python3
"""Scaled feature export for the Unified Filter — nuScenes TRAIN + Waymo.

Differences from export_filter_features.py (val-only):
  * Scene bank = nuScenes v1.0-trainval TRAIN split (700 scenes) + the 50
    Waymo segments (400 frames) → ~750 scenes.
  * Features come from the WACV LoRA-FM checkpoint through the WACV
    forward (train_wacv.forward_model), which includes the zero-agent
    attention sanitizer — NaN prevention at the source. A finite-filter
    still guards aggregation (belt and suspenders).
  * Per-scene tag VALIDITY is exported: nuScenes scenes are valid on all
    20 tags, Waymo scenes only on the 5 perception-evaluable tags. Query
    relevance gains a parallel `query_valid` matrix: a scene is EXCLUDED
    from a query's ranking universe (neither positive nor negative) when
    the query's required tags are not evaluable for it.
  * One representative CAM_FRONT thumbnail per scene is saved for the
    visual stress-test grids.

Usage (1 GPU, ~85 min):
  python scripts/export_filter_features_scaled.py \
      --out work_dirs/unified_filter/features_scaled.npz \
      --thumbs work_dirs/unified_filter/scene_thumbs
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_wacv import build_model, load_ckpt, forward_model, TOKENIZER  # noqa
from scene_mining.data import (                                          # noqa
    NuScenesUniversalDataset, WaymoUniversalDataset, universal_collate)

TAG_NAMES = [
    "scene_dark", "scene_rainy", "scene_construction_zone", "scene_intersection",
    "scene_parking_lot", "scene_highway", "scene_school_zone", "scene_residential",
    "object_pedestrian", "object_bicycle", "object_motorcycle", "object_truck",
    "object_bus", "object_car", "object_emergency_vehicle", "object_animal",
    "behavior_jaywalking", "behavior_running", "behavior_erratic", "behavior_static",
]
PHRASE = {
    "scene_dark": "at night", "scene_rainy": "in heavy rain",
    "scene_construction_zone": "in a construction zone",
    "scene_intersection": "at a busy intersection",
    "scene_parking_lot": "in a parking lot", "scene_highway": "on a highway",
    "scene_school_zone": "in a school zone",
    "scene_residential": "on a residential street",
    "object_pedestrian": "with pedestrians", "object_bicycle": "with a cyclist",
    "object_motorcycle": "with a motorcycle", "object_truck": "with a truck",
    "object_bus": "with a bus", "object_car": "with surrounding cars",
    "object_emergency_vehicle": "with an emergency vehicle",
    "object_animal": "with an animal on the road",
    "behavior_jaywalking": "with a jaywalking pedestrian",
    "behavior_running": "with someone running",
    "behavior_erratic": "with erratic behavior",
    "behavior_static": "with stopped agents",
}
TEMPLATES = ["a driving scene {}", "an urban driving scenario {}",
             "footage of the road {}", "a hard scene {}"]
RISK_TEMPLATES = ["a dangerous near-miss situation {}",
                  "a high-risk driving scene {}"]


def build_queries(tags_gt, tag_valid, risk, max_pair_queries=160, seed=0):
    """→ texts, query_pos (Q,N), query_valid (Q,N), query_tags (Q,20)."""
    rng = np.random.default_rng(seed)
    n = tags_gt.shape[0]
    r = np.nan_to_num(risk[:, 0], nan=0.0)
    hi_risk = r >= np.quantile(r, 0.75)
    texts, pos_l, val_l, qtag_l = [], [], [], []

    def add(text, need: list[int], extra_mask=None):
        need_v = np.ones(20, bool); need_v[:] = False
        for t in need: need_v[t] = True
        valid = tag_valid[:, need].all(axis=1)          # scene can judge query
        pos = valid & (tags_gt[:, need] >= 0.5).all(axis=1)
        if extra_mask is not None:
            pos = pos & extra_mask
        n_eff = valid.sum()
        if pos.sum() >= 3 and pos.sum() <= n_eff - 3:
            texts.append(text); pos_l.append(pos); val_l.append(valid)
            qtag_l.append(need_v)

    for ti, tag in enumerate(TAG_NAMES):
        for tpl in TEMPLATES[:2]:
            add(tpl.format(PHRASE[tag]), [ti])
        add(RISK_TEMPLATES[0].format(PHRASE[tag]), [ti], extra_mask=hi_risk)
    pairs = list(itertools.combinations(range(len(TAG_NAMES)), 2))
    rng.shuffle(pairs)
    added = 0
    for ti, tj in pairs:
        if added >= max_pair_queries:
            break
        before = len(texts)
        add(f"a driving scene {PHRASE[TAG_NAMES[ti]]} {PHRASE[TAG_NAMES[tj]]}",
            [ti, tj])
        added += len(texts) - before
    return texts, np.stack(pos_l), np.stack(val_l), np.stack(qtag_l)


@torch.no_grad()
def clip_embed(texts, device, batch=64,
               model_name="openai/clip-vit-base-patch32"):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
    tok = CLIPTokenizerFast.from_pretrained(model_name)
    enc = CLIPTextModelWithProjection.from_pretrained(
        model_name, use_safetensors=True).to(device).eval()
    out = []
    for i in range(0, len(texts), batch):
        t = tok(texts[i:i + batch], padding=True, truncation=True,
                return_tensors="pt").to(device)
        out.append(torch.nn.functional.normalize(
            enc(**t).text_embeds.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def sweep(model, loader, device, thumbs_dir: Path, label: str):
    """Per-frame features; saves one CAM_FRONT thumb per new scene."""
    import cv2
    E, TL, R, TG, TV, toks = [], [], [], [], [], []
    seen_thumb = set()
    for bi, batch in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            out = forward_model(model, batch, device)
        E.append(out.embeddings.detach().float().cpu())
        TL.append(out.tag_logits.detach().float().cpu())
        R.append(out.risk_scores.detach().float().cpu().view(-1, 1))
        TG.append(batch["semantic_tags"]["values"].float())
        TV.append(batch["semantic_tags"]["valid_mask"].bool())
        for b, m in enumerate(batch["meta"]):
            tok = m["scene_token"]; toks.append(tok)
            if tok not in seen_thumb:
                seen_thumb.add(tok)
                img = batch["camera_images"][b, 0].numpy()      # CAM_FRONT CHW
                bgr = cv2.cvtColor(
                    (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(thumbs_dir / f"{tok}.jpg"), bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
        if (bi + 1) % 200 == 0:
            print(f"    [{label}] {len(toks)} frames", flush=True)
    return (torch.cat(E), torch.cat(TL), torch.cat(R),
            torch.cat(TG), torch.cat(TV), toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="configs/experiment/nuscenes_scenemining_v8_unified.py")
    ap.add_argument("--base-checkpoint",
                    default="work_dirs/nuscenes_scenemining_v4_motion_mask/epoch_1.pth")
    ap.add_argument("--v8-checkpoint",
                    default="work_dirs/nuscenes_scenemining_v8_unified/epoch_1_trainable.pth")
    ap.add_argument("--wacv-checkpoint",
                    default="work_dirs/wacv_unified_lora_fm/wacv_epoch2.pth")
    ap.add_argument("--nusc-root", default="data/nuscenes")
    ap.add_argument("--nusc-split", default="train")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--waymo", default="work_dirs/waymo_zeroshot/extracted_large")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thumbs", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thumbs = Path(args.thumbs); thumbs.mkdir(parents=True, exist_ok=True)
    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    margs = SimpleNamespace(use_lora=True, freeze_motion=False,
                            base_checkpoint=args.base_checkpoint,
                            v8_checkpoint=args.v8_checkpoint)
    model, _ = build_model(cfg, device, margs, rank=0)
    load_ckpt(model, None, None, args.wacv_checkpoint, rank=0)
    model.eval()

    # nuScenes train
    from scene_mining.datasets.nuscenes_dataset import NuScenesFrameDataset
    inner = NuScenesFrameDataset(
        dataroot=args.nusc_root, version="v1.0-trainval", split=args.nusc_split,
        labels_dir=args.labels, tokenizer_path=TOKENIZER,
        n_agents=8, n_waypoints=6, max_text_len=128)
    nds = NuScenesUniversalDataset(inner, tokenizer_path=TOKENIZER)
    nl = DataLoader(nds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=universal_collate)
    print(f"[scaled] nuScenes {args.nusc_split}: {len(nds)} frames")
    E1, TL1, R1, TG1, TV1, tok1 = sweep(model, nl, device, thumbs, "nusc")

    # Waymo
    wds = WaymoUniversalDataset(args.waymo, tokenizer_path=TOKENIZER)
    wl = DataLoader(wds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=universal_collate)
    print(f"[scaled] Waymo: {len(wds)} frames")
    E2, TL2, R2, TG2, TV2, tok2 = sweep(model, wl, device, thumbs, "waymo")
    del model; torch.cuda.empty_cache()

    E = torch.cat([E1, E2]); TL = torch.cat([TL1, TL2])
    R = torch.cat([R1, R2]); TG = torch.cat([TG1, TG2])
    TV = torch.cat([TV1, TV2]); toks = tok1 + tok2
    domain = np.array([0] * len(tok1) + [1] * len(tok2))

    finite = (torch.isfinite(E).all(1) & torch.isfinite(TL).all(1)
              & torch.isfinite(R).view(-1))
    print(f"[scaled] dropping {int((~finite).sum())}/{len(finite)} "
          f"non-finite frames")
    E = torch.nn.functional.normalize(E, dim=1).numpy()

    by_scene = defaultdict(list)
    for i, t in enumerate(toks):
        if finite[i]:
            by_scene[t].append(i)
    scenes = sorted(by_scene)
    S = np.stack([E[by_scene[s]].mean(0) for s in scenes])
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    TLs = np.stack([TL.numpy()[by_scene[s]].mean(0) for s in scenes])
    TGs = np.stack([TG.numpy()[by_scene[s]].max(0) for s in scenes])
    TVs = np.stack([TV.numpy()[by_scene[s]].all(0) for s in scenes])
    RKs = np.stack([[np.nan_to_num(R.numpy()[by_scene[s]], nan=0.0).mean()]
                    for s in scenes])
    DOM = np.array([domain[by_scene[s][0]] for s in scenes])
    print(f"[scaled] {len(scenes)} scenes "
          f"({int((DOM==0).sum())} nuScenes + {int((DOM==1).sum())} Waymo)")

    texts, qpos, qval, qtags = build_queries(TGs, TVs, RKs)
    print(f"[scaled] {len(texts)} queries "
          f"(mean positives {qpos.sum(1).mean():.1f})")
    qclip = clip_embed(texts, device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, emb=S.astype(np.float32), tag_logits=TLs.astype(np.float32),
        tags_gt=TGs.astype(np.float32), tag_valid=TVs,
        risk=RKs.astype(np.float32), domain=DOM.astype(np.int8),
        scene_tokens=np.array(scenes),
        queries_clip=qclip.astype(np.float32), query_pos=qpos,
        query_valid=qval, query_tags=qtags, query_texts=np.array(texts))
    print(f"[scaled] wrote {args.out}")


if __name__ == "__main__":
    main()
