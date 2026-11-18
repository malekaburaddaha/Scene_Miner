#!/usr/bin/env python3
"""Overnight Phase 1b — balanced 1400+ scene bank.

Waymo side: forwards ALL frames from extracted_all_{0,1,2} (202 segments x
16 frames) through the WACV LoRA-FM model (sanitized forward → NaN-free)
and aggregates them into TEMPORAL CHUNK scenes: 4 consecutive extracted
frames = 1 scene, token "<segment>#c<k>" (a 20 s Waymo segment yields 4
scene-chunks, matching nuScenes' 20 s scene granularity at 4x density).
Saves one thumbnail per chunk.

nuScenes side: REUSES the 700 cached scene features from
features_scaled.npz (the images did not change; re-forwarding 28k frames
would waste ~1.2 h of the compute window).

Output: features_overnight.npz — emb / tag_logits / tags_gt / tag_valid /
risk / domain / scene_tokens for ~700 + ~808 scenes. Finite-filter applied
before every aggregation (strict NaN discipline).
"""
from __future__ import annotations

import argparse
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
from scene_mining.data import WaymoUniversalDataset, universal_collate   # noqa


@torch.no_grad()
def sweep_waymo(model, root, device, thumbs_dir: Path, chunk: int, bs, nw):
    import cv2
    ds = WaymoUniversalDataset(str(root), tokenizer_path=TOKENIZER)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                    collate_fn=universal_collate)
    E, TL, R, TG, TV, toks, fidx = [], [], [], [], [], [], []
    for bi, batch in enumerate(dl):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            out = forward_model(model, batch, device)
        E.append(out.embeddings.detach().float().cpu())
        TL.append(out.tag_logits.detach().float().cpu())
        R.append(out.risk_scores.detach().float().cpu().view(-1, 1))
        TG.append(batch["semantic_tags"]["values"].float())
        TV.append(batch["semantic_tags"]["valid_mask"].bool())
        for b, m in enumerate(batch["meta"]):
            toks.append(m["scene_token"])
            fidx.append(int(m.get("frame_idx", len(fidx))))
        if (bi + 1) % 100 == 0:
            print(f"    [{root.name}] {len(toks)} frames", flush=True)
    # order frames within each segment by frame_idx, then chunk
    E, TL, R = torch.cat(E), torch.cat(TL), torch.cat(R)
    TG, TV = torch.cat(TG), torch.cat(TV)
    finite = (torch.isfinite(E).all(1) & torch.isfinite(TL).all(1)
              & torch.isfinite(R).view(-1))
    print(f"    [{root.name}] non-finite frames dropped: "
          f"{int((~finite).sum())}/{len(finite)}")
    by_seg = defaultdict(list)
    for i, (t, f) in enumerate(zip(toks, fidx)):
        if finite[i]:
            by_seg[t].append((f, i))
    S, TLs, TGs, TVs, RKs, chunk_toks = [], [], [], [], [], []
    En = torch.nn.functional.normalize(E, dim=1).numpy()
    for seg in sorted(by_seg):
        frames = [i for _, i in sorted(by_seg[seg])]
        for k in range(0, len(frames), chunk):
            grp = frames[k:k + chunk]
            if len(grp) < max(2, chunk // 2):
                continue
            tokn = f"{seg}#c{k // chunk}"
            v = En[grp].mean(0); v /= np.linalg.norm(v) + 1e-9
            S.append(v)
            TLs.append(TL.numpy()[grp].mean(0))
            TGs.append(TG.numpy()[grp].max(0))
            TVs.append(TV.numpy()[grp].all(0))
            RKs.append([np.nan_to_num(R.numpy()[grp], nan=0.0).mean()])
            chunk_toks.append(tokn)
            # thumbnail = first frame of the chunk (re-encode from dataset)
            img = ds[grp[0]]["camera_images"][0].numpy()
            bgr = cv2.cvtColor((img.transpose(1, 2, 0) * 255)
                               .clip(0, 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(thumbs_dir / f"{tokn}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    return (np.stack(S), np.stack(TLs), np.stack(TGs), np.stack(TVs),
            np.stack(RKs), chunk_toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-roots", nargs="+", required=True)
    ap.add_argument("--nusc-features",
                    default="work_dirs/unified_filter/features_scaled.npz")
    ap.add_argument("--config",
                    default="configs/experiment/nuscenes_scenemining_v8_unified.py")
    ap.add_argument("--base-checkpoint",
                    default="work_dirs/nuscenes_scenemining_v4_motion_mask/epoch_1.pth")
    ap.add_argument("--v8-checkpoint",
                    default="work_dirs/nuscenes_scenemining_v8_unified/epoch_1_trainable.pth")
    ap.add_argument("--wacv-checkpoint",
                    default="work_dirs/wacv_unified_lora_fm/wacv_epoch2.pth")
    ap.add_argument("--out", default="work_dirs/unified_filter/features_overnight.npz")
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--chunk", type=int, default=4)
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

    Sw, TLw, TGw, TVw, RKw, tokw = [], [], [], [], [], []
    for root in args.extracted_roots:
        s, tl, tg, tv, rk, tk = sweep_waymo(
            model, Path(root), device, thumbs, args.chunk,
            args.batch_size, args.num_workers)
        Sw.append(s); TLw.append(tl); TGw.append(tg); TVw.append(tv)
        RKw.append(rk); tokw += tk
    Sw = np.concatenate(Sw); TLw = np.concatenate(TLw)
    TGw = np.concatenate(TGw); TVw = np.concatenate(TVw)
    RKw = np.concatenate(RKw)
    print(f"[overnight] Waymo scene-chunks: {len(tokw)}")

    zn = np.load(args.nusc_features, allow_pickle=False)
    keep = zn["domain"] == 0                      # cached nuScenes rows only
    S = np.concatenate([zn["emb"][keep], Sw])
    TL = np.concatenate([zn["tag_logits"][keep], TLw])
    TG = np.concatenate([zn["tags_gt"][keep], TGw])
    TV = np.concatenate([zn["tag_valid"][keep], TVw])
    RK = np.concatenate([zn["risk"][keep], RKw])
    toks = [str(t) for t in zn["scene_tokens"][keep]] + tokw
    DOM = np.array([0] * int(keep.sum()) + [1] * len(tokw), dtype=np.int8)
    assert np.isfinite(S).all() and np.isfinite(TL).all() and np.isfinite(RK).all()
    print(f"[overnight] bank: {len(toks)} scenes "
          f"({int(keep.sum())} nuScenes + {len(tokw)} Waymo)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, emb=S.astype(np.float32), tag_logits=TL.astype(np.float32),
        tags_gt=TG.astype(np.float32), tag_valid=TV,
        risk=RK.astype(np.float32), domain=DOM,
        scene_tokens=np.array(toks))
    print(f"[overnight] wrote {args.out}")


if __name__ == "__main__":
    main()
