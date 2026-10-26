#!/usr/bin/env python3
"""Heavy training run for the REPAIRED LoRA-Grounded Filter (Phases 3-4).

Architecture repairs live in the model file (scalar-channel fix, bounded
normalised scoring, fixed temperature, LoRA r=64/alpha=128). This trainer
adds the two data-side repairs from the diagnostic:

  [P3a] DOMAIN BALANCING. Every query's positives AND conditional negatives
        are drawn half-from-nuScenes / half-from-Waymo (best effort, with
        the achieved balance logged). Kills the domain-identity shortcut
        (rain was 197/700 nuScenes vs 4/808 Waymo, so "is this nuScenes?"
        solved weather queries).

  [P3b] STRICT CONDITIONAL HARD NEGATIVES. A query is a (macro context,
        fine attribute, fine value) triple. Its hard negatives MATCH the
        anchor on EVERY macro attribute and differ ONLY on the target fine
        attribute (e.g. within night+wet: dense vs sparse). Previously the
        contradiction set was unconditional, so a night query's negatives
        were day scenes — already separable at AUC 1.000, yielding zero
        gradient. That is why 99.57% of gradient had collapsed into the
        density probes.

  [P2] VICReg variance hinge + covariance decorrelation on the PRE-norm
       fused batch (fused space had collapsed to effective rank 3.2-4.4/256).

Loss = listwise InfoNCE + w_cond * conditional-negative margin + VICReg.

CHECKPOINTING (exhaustive, resumable): trainable weights (filter + LoRA),
optimizer, scheduler, scaler, loss history, step, and ALL RNG streams
(python / numpy / torch-CPU / torch-CUDA-all / sampler generator).
Atomic writes + latest.pt. Resume with --resume.
"""
from __future__ import annotations

from paths import QWEN_VL
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scene_mining.filter.lora_grounded_filter import (LoRAGroundedFilter,
                                                      QwenVisualEncoder,
                                                      vicreg_terms)

# macro contexts (the "held fixed" part) and fine targets (the "varied" part)
MACRO_CONTEXTS = [
    ({"lighting": "night"}, "at night"),
    ({"lighting": "day"}, "in daylight"),
    ({"wet_road": True}, "on a wet road"),
    ({"weather": "rain"}, "in rain"),
    ({"lighting": "night", "wet_road": True}, "at night on a wet road"),
    ({"lighting": "day", "wet_road": True}, "in daylight on a wet road"),
]
# only positively-phrased fine concepts (CLIP handles negation poorly)
FINE_TARGETS = [
    ("density", "dense", "dense traffic", ["sparse", "moderate"]),
    ("density", "sparse", "sparse traffic", ["dense", "moderate"]),
    ("glare", True, "strong headlight glare", [False]),
    ("droplets", True, "rain droplets on the lens", [False]),
]


# Rare cells that never met min_pos but which the evaluation demands.
# Each: (positive predicate, conditional-negative predicate, text).
# The negative MATCHES the anchor on the macro axis and differs on the target.
FORCED_STRATA = [
    ({"weather": "rain", "lighting": "night"},
     {"weather": "rain", "lighting": "day"},
     "driving scene in heavy rain at night"),
    ({"weather": "rain", "lighting": "night", "wet_road": True},
     {"weather": "rain", "lighting": "day", "wet_road": True},
     "driving scene in heavy rain at night on a wet road"),
    ({"lighting": "night", "weather": "rain", "droplets": True},
     {"lighting": "day", "weather": "rain", "droplets": True},
     "driving scene with rain droplets on the lens at night"),
]


def build_stratified_queries(gt, toks, dom, min_pos=6, min_cond=6,
                             min_pos_forced=3):
    """→ list of dicts: text, pos_idx, cond_idx (macro-matched, fine-differing)."""
    def sel(pred):
        return np.array([all(gt.get(t, {}).get(k) == v for k, v in pred.items())
                         for t in toks])
    qs = []
    for mctx, mphrase in MACRO_CONTEXTS:
        for fkey, fval, fphrase, contrast_vals in FINE_TARGETS:
            if fkey in mctx:
                continue
            pos = sel({**mctx, fkey: fval})
            cond = np.zeros(len(toks), bool)
            for cv in contrast_vals:
                cond |= sel({**mctx, fkey: cv})
            cond &= ~pos
            if pos.sum() < min_pos or cond.sum() < min_cond:
                continue
            qs.append({
                "text": f"driving scene with {fphrase} {mphrase}",
                "macro": mctx, "fine": {fkey: fval},
                "pos_idx": np.where(pos)[0], "cond_idx": np.where(cond)[0],
                "n_pos": int(pos.sum()), "n_cond": int(cond.sum()),
                "pos_dom": [int((pos & (dom == 0)).sum()), int((pos & (dom == 1)).sum())],
                "cond_dom": [int((cond & (dom == 0)).sum()), int((cond & (dom == 1)).sum())],
            })
    # forced rare strata (lower threshold, explicit conditional negatives)
    for ppred, npred, text in FORCED_STRATA:
        pos = sel(ppred); cond = sel(npred) & ~pos
        if pos.sum() < min_pos_forced or cond.sum() < min_pos_forced:
            continue
        if any(q["text"] == text for q in qs):
            continue
        qs.append({"text": text, "macro": ppred, "fine": {"forced": True},
                   "pos_idx": np.where(pos)[0], "cond_idx": np.where(cond)[0],
                   "n_pos": int(pos.sum()), "n_cond": int(cond.sum()),
                   "pos_dom": [int((pos & (dom == 0)).sum()), int((pos & (dom == 1)).sum())],
                   "cond_dom": [int((cond & (dom == 0)).sum()), int((cond & (dom == 1)).sum())],
                   "forced": True})
    return qs


def build_caption_queries(cap, toks, dom, min_pos=6):
    """Ablation baseline: queries from VLM caption phrases (no exact GT)."""
    import re as _re
    from collections import Counter as _C
    pools, cnt = [], _C()
    for t in toks:
        v = cap.get(t)
        slots = (v["slots"] if isinstance(v, dict) else list(v or []))[:3]
        pools.append(frozenset(w for ph in slots
                               for w in _re.sub(r"[^a-z ]", "", ph.lower()).split()))
        for ph in slots:
            if ph:
                cnt[ph.strip().lower()] += 1
    qs = []
    for ph, _ in cnt.most_common(80):
        toks_q = frozenset(_re.sub(r"[^a-z ]", "", ph).split())
        if not toks_q:
            continue
        pos = np.array([toks_q <= p for p in pools])
        if pos.sum() < min_pos or pos.sum() > len(toks) - min_pos:
            continue
        neg = ~pos
        qs.append({"text": f"driving scene with {ph}", "macro": {}, "fine": {},
                   "pos_idx": np.where(pos)[0], "cond_idx": np.where(neg)[0],
                   "n_pos": int(pos.sum()), "n_cond": int(neg.sum()),
                   "pos_dom": [int((pos & (dom == 0)).sum()), int((pos & (dom == 1)).sum())],
                   "cond_dom": [int((neg & (dom == 0)).sum()), int((neg & (dom == 1)).sum())]})
    return qs


def balanced_draw(idx_pool, dom, k, rng):
    """Draw k indices, half from each domain where possible (best effort)."""
    d0 = idx_pool[dom[idx_pool] == 0]
    d1 = idx_pool[dom[idx_pool] == 1]
    h = k // 2
    take0 = min(h, len(d0))
    take1 = min(k - take0, len(d1))
    if take0 + take1 < k:                       # top up from whichever remains
        extra = k - take0 - take1
        if len(d0) > take0:
            take0 += min(extra, len(d0) - take0)
        elif len(d1) > take1:
            take1 += min(extra, len(d1) - take1)
    out = []
    if take0:
        out += list(rng.choice(d0, take0, replace=False))
    if take1:
        out += list(rng.choice(d1, take1, replace=False))
    while len(out) < k and len(idx_pool):        # degenerate cell fallback
        out.append(int(rng.choice(idx_pool)))
    return np.array(out[:k], dtype=int)


# ── exhaustive checkpointing ──────────────────────────────────────────
def rng_state(gen):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": torch.cuda.get_rng_state_all()
                              if torch.cuda.is_available() else None,
            "sampler_gen": gen.get_state()}


def load_rng(st, gen):
    random.setstate(st["python"]); np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch_cpu"])
    if st.get("torch_cuda_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["torch_cuda_all"])
    gen.set_state(st["sampler_gen"])


def save_ckpt(path, model, optim, sched, scaler, step, hist, gen, rank, extra):
    if rank != 0:
        return
    core = model.module if hasattr(model, "module") else model
    payload = {
        "trainable_state_dict": {n: p.detach().cpu()
                                 for n, p in core.named_parameters()
                                 if p.requires_grad},
        "buffers": {n: b.detach().cpu() for n, b in core.named_buffers()},
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": step, "loss_history": hist,
        "rng": rng_state(gen), "time": time.time(), **extra}
    tmp = Path(str(path) + ".tmp"); torch.save(payload, tmp)
    os.replace(tmp, path)
    lp = Path(path).parent / "latest.pt"
    torch.save(payload, str(lp) + ".tmp"); os.replace(str(lp) + ".tmp", lp)


def load_ckpt(path, model, optim, sched, scaler, gen, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    core = model.module if hasattr(model, "module") else model
    core.load_state_dict(ck["trainable_state_dict"], strict=False)
    optim.load_state_dict(ck["optimizer_state_dict"])
    sched.load_state_dict(ck["scheduler_state_dict"])
    scaler.load_state_dict(ck["scaler_state_dict"])
    load_rng(ck["rng"], gen)
    return ck["global_step"], ck["loss_history"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work_dirs/unified_filter/features_overnight.npz")
    ap.add_argument("--det-gt", default="work_dirs/unified_filter/deterministic_gt.json")
    ap.add_argument("--binary-flags", nargs="*", default=[
        "work_dirs/unified_filter/binary_flags_a.json",
        "work_dirs/unified_filter/binary_flags_b.json"])
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--model", default=QWEN_VL)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--queries-per-step", type=int, default=4)
    ap.add_argument("--pos-per-query", type=int, default=2)
    ap.add_argument("--cond-per-query", type=int, default=2)
    ap.add_argument("--img-side", type=int, default=336)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--tau-final", type=float, default=None,
                    help="linearly anneal tau from --tau to this value")
    ap.add_argument("--init-from", default=None,
                    help="load ONLY trainable weights (fresh optimizer/scheduler)")
    ap.add_argument("--oversample-forced", type=float, default=2.0,
                    help="relative sampling weight for forced rare strata")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=1500)
    ap.add_argument("--cond-margin", type=float, default=1.0)
    ap.add_argument("--w-cond", type=float, default=4.0)
    ap.add_argument("--w-var", type=float, default=1.0)
    ap.add_argument("--w-cov", type=float, default=0.04)
    ap.add_argument("--w-vicreg", type=float, default=1.0)
    ap.add_argument("--ckpt-every-steps", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--legacy-scoring", action="store_true",
                    help="ablation: unbounded score + learned temperature")
    ap.add_argument("--legacy-scalar", action="store_true",
                    help="ablation: (MISNOMER) only swaps (mu,sigma)->(0,1); does "
                         "NOT reproduce the LayerNorm(1) bug. Use --force-layernorm1d.")
    ap.add_argument("--unbounded-tau", action="store_true",
                    help="remove the 1e-2 floor on the learned temperature")
    ap.add_argument("--clip-logit-scale", action="store_true",
                    help="CLIP-style multiplicative logit_scale init at 1/0.07")
    ap.add_argument("--force-layernorm1d", action="store_true",
                    help="TRUE pre-repair defect: real nn.LayerNorm over the width-1 "
                         "risk scalar, which zeroes the channel for every input")
    ap.add_argument("--freeze-visual", action="store_true",
                    help="ablation: no LoRA; precompute a STATIC visual token")
    ap.add_argument("--supervision", choices=["deterministic", "caption"],
                    default="deterministic")
    ap.add_argument("--captions",
                    default="work_dirs/unified_filter/struct_captions_all.json")
    ap.add_argument("--unconditional-negs", action="store_true",
                    help="ablation: random negatives, no macro matching, no domain balance")
    ap.add_argument("--pool", choices=["xattn", "mean"], default="xattn",
                    help="R3 ablation: replace query-conditioned "
                         "cross-attention pooling with an unweighted mean")
    ap.add_argument("--fusion-layers", type=int, default=2,
                    help="R3 ablation: depth of the fusion decoder")
    ap.add_argument("--split-file", default=None,
                    help="benchmark_cvpr/split_906_602.json; restricts ALL query "
                         "positives/negatives to the train side (scene-disjoint eval)")
    ap.add_argument("--drop-attr", default=None,
                    help="leave-one-attribute-out: remove this attribute from "
                         "supervision AND zero it out of the tag logits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
    else:
        rank, world, local = 0, 1, 0
    device = torch.device(f"cuda:{local}")
    seed = args.seed + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    nprng = np.random.default_rng(seed)
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)

    z = np.load(args.features, allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]
    emb = torch.from_numpy(z["emb"]).float()
    tlog = torch.from_numpy(z["tag_logits"]).float()
    risk = torch.from_numpy(z["risk"]).float()
    # [CVPR] NOTE on LOAO and the tag vector: the 20-tag taxonomy is
    # scene_dark / scene_rainy / scene_construction_zone / ... / object_* /
    # behavior_* (see scripts/export_filter_features.py). NONE of the four
    # degradation attributes (droplets, glare, wet_road, low_contrast) has a tag
    # channel, so there is no tag-side leakage path to close for them. The
    # correlated context tags (scene_rainy, scene_dark) are deliberately LEFT IN
    # because the cached-attribute baseline also keeps its weather/lighting
    # metadata - zeroing them for us only would make the comparison unfair.
    dom = z["domain"].astype(int)

    gt = json.load(open(args.det_gt))
    for fp in args.binary_flags:
        if Path(fp).exists():
            for tk, fl in json.load(open(fp)).items():
                gt.setdefault(tk, {}).update(fl)

    # [CVPR] leave-one-attribute-out: strip the attribute from the GT the query
    # builder sees, so no query can be formed from it and it cannot appear as a
    # macro context or a fine target.
    if args.drop_attr:
        for _tk in list(gt):
            gt[_tk].pop(args.drop_attr, None)
        if rank == 0:
            print(f"[cvpr] LOAO: dropped attribute '{args.drop_attr}' from supervision")

    if args.supervision == "caption":
        _cap = json.load(open(args.captions))["captions"]
        queries = build_caption_queries(_cap, toks, dom)
    else:
        queries = build_stratified_queries(gt, toks, dom)

    # [CVPR] scene-disjoint training: keep only train-split scenes as positives
    # or conditional negatives. Evaluation scenes never contribute a gradient.
    if args.split_file:
        _sp = json.load(open(args.split_file))
        _train = set(_sp["train"])
        _keep = np.array([t in _train for t in toks], dtype=bool)
        _kept = []
        for q in queries:
            q["pos_idx"] = q["pos_idx"][_keep[q["pos_idx"]]]
            q["cond_idx"] = q["cond_idx"][_keep[q["cond_idx"]]]
            q["n_pos"], q["n_cond"] = int(len(q["pos_idx"])), int(len(q["cond_idx"]))
            q["pos_dom"] = [int((dom[q["pos_idx"]] == 0).sum()), int((dom[q["pos_idx"]] == 1).sum())]
            q["cond_dom"] = [int((dom[q["cond_idx"]] == 0).sum()), int((dom[q["cond_idx"]] == 1).sum())]
            if q["n_pos"] >= 2 and q["n_cond"] >= 2:
                _kept.append(q)
        if rank == 0:
            print(f"[cvpr] split-restricted: {len(queries)} -> {len(_kept)} queries "
                  f"({_keep.sum()} / {len(toks)} scenes trainable)")
        queries = _kept
    if rank == 0:
        print(f"[v2] {len(queries)} STRATIFIED queries "
              f"(macro-matched conditional negatives)")
        for q in queries:
            print(f"   pos {q['n_pos']:4d} (nu{q['pos_dom'][0]}/wa{q['pos_dom'][1]}) | "
                  f"cond {q['n_cond']:4d} (nu{q['cond_dom'][0]}/wa{q['cond_dom'][1]}) | "
                  f"{q['text']}")
        (work / "queries.json").write_text(json.dumps(
            [{k: (v if not isinstance(v, np.ndarray) else v.tolist())
              for k, v in q.items()} for q in queries], indent=1))

    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
    ctok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    cenc = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True).to(device).eval()
    with torch.no_grad():
        tt = ctok([q["text"] for q in queries], padding=True, truncation=True,
                  return_tensors="pt").to(device)
        qclip = F.normalize(cenc(**tt).text_embeds.float(), dim=-1).cpu()
    del cenc; torch.cuda.empty_cache()

    enc = QwenVisualEncoder(args.model,
                            lora_r=(1 if args.freeze_visual else args.lora_r),
                            lora_alpha=args.lora_alpha, device=str(device),
                            img_max_side=args.img_side)
    enc.model.to(device)
    if args.freeze_visual:                 # ablation: static visual token
        for p_ in enc.parameters():
            p_.requires_grad_(False)
    # Scalar standardisation statistics are a property of the BANK, not a
    # constant: the temporal ablation (T=1,2) shifts the risk distribution,
    # and a stale (mu, sigma) would confound "less temporal context" with
    # "mis-calibrated standardiser". On features_overnight this reproduces
    # the published constants (0.417797, 0.081112) exactly, so no existing
    # run changes.
    _rk = risk.reshape(-1).cpu().numpy()
    risk_stats = (float(_rk.mean()), float(_rk.std()))
    model = LoRAGroundedFilter(enc, tau=args.tau,
                               legacy_scoring=args.legacy_scoring,
                               legacy_scalar=args.legacy_scalar,
                               force_layernorm1d=args.force_layernorm1d,
                               unbounded_tau=args.unbounded_tau,
                               clip_logit_scale=args.clip_logit_scale,
                               pool=args.pool,
                               n_fusion_layers=args.fusion_layers,
                               risk_stats=risk_stats).to(device)

    V_static = None
    if args.freeze_visual:                 # precompute once -> fast steps
        from PIL import Image as _I
        enc.model.eval()
        V_static = torch.zeros(len(toks), enc.d_model, device=device)
        with torch.no_grad():
            for i in range(0, len(toks), 16):
                ims = [_I.open(Path(args.thumbs) / f"{t}.jpg").convert("RGB")
                       for t in toks[i:i + 16]]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    V_static[i:i + 16] = enc.encode(ims)
        if rank == 0:
            print(f"[ablation] static visual tokens precomputed for {len(toks)} scenes")
    if rank == 0:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_lora = sum(p.numel() for n_, p in model.named_parameters()
                     if p.requires_grad and "lora_" in n_)
        print(f"[v2] trainable {n_tr:,} (LoRA r={args.lora_r}: {n_lora:,}) | "
              f"fixed tau={float(model.tau):.3f}")
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=True)
    core = model.module if hasattr(model, "module") else model

    lora_p = [p for n_, p in core.named_parameters()
              if p.requires_grad and "lora_" in n_]
    filt_p = [p for n_, p in core.named_parameters()
              if p.requires_grad and "lora_" not in n_]
    optim = torch.optim.AdamW([{"params": filt_p, "lr": args.lr},
                               {"params": lora_p, "lr": args.lora_lr}],
                              weight_decay=0.01)
    sched = torch.optim.lr_scheduler.SequentialLR(
        optim,
        [torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=0.02, total_iters=args.warmup_steps),
         torch.optim.lr_scheduler.CosineAnnealingLR(
             optim, T_max=max(1, args.steps - args.warmup_steps), eta_min=1e-7)],
        milestones=[args.warmup_steps])
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    if args.init_from:
        _ck = torch.load(args.init_from, map_location=device, weights_only=False)
        core.load_state_dict(_ck["trainable_state_dict"], strict=False)
        if rank == 0:
            print(f"[init-from] weights loaded from {args.init_from} "
                  f"(step {_ck['global_step']}); optimizer/scheduler are FRESH")
    gstep, hist = 0, []
    if args.resume:
        gstep, hist = load_ckpt(args.resume, model, optim, sched, scaler,
                                gen, device)
        if rank == 0:
            print(f"[resume] {args.resume} → step {gstep}")

    qweights = np.array([args.oversample_forced if q.get("forced") else 1.0
                         for q in queries], dtype=float)
    qweights /= qweights.sum()

    from PIL import Image
    nq = args.queries_per_step
    npos, ncond = args.pos_per_query, args.cond_per_query
    model.train()
    t_start = time.time()
    while gstep < args.steps:
        qsel = nprng.choice(len(queries), min(nq, len(queries)),
                            replace=False, p=qweights)
        idx, per_q = [], []
        for qi in qsel:
            q = queries[qi]
            if args.unconditional_negs:
                p = nprng.choice(q["pos_idx"], min(npos, len(q["pos_idx"])),
                                 replace=len(q["pos_idx"]) < npos)
                c = nprng.choice(q["cond_idx"], min(ncond, len(q["cond_idx"])),
                                 replace=len(q["cond_idx"]) < ncond)
            else:
                p = balanced_draw(q["pos_idx"], dom, npos, nprng)
                c = balanced_draw(q["cond_idx"], dom, ncond, nprng)
            per_q.append((qi, p, c))
            idx += list(p) + list(c)
        idx = np.array(idx, dtype=int)
        S = len(idx)

        # relevance / conditional-negative masks over the sampled bank
        rel = np.zeros((len(qsel), S), bool)
        cnd = np.zeros((len(qsel), S), bool)
        for r_, (qi, _, _) in enumerate(per_q):
            q = queries[qi]
            pset, cset = set(q["pos_idx"].tolist()), set(q["cond_idx"].tolist())
            for j, s_ in enumerate(idx):
                if s_ in pset:
                    rel[r_, j] = True
                elif s_ in cset:
                    cnd[r_, j] = True

        images = [Image.open(Path(args.thumbs) / f"{toks[i]}.jpg").convert("RGB")
                  for i in idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = (V_static[idx] if V_static is not None
                 else core.visual.encode(images))
            e = emb[idx].to(device); t_ = tlog[idx].to(device)
            r_in = risk[idx].to(device)
            qq = qclip[qsel].to(device)
            qrep = qq.unsqueeze(1).expand(-1, S, -1).reshape(-1, qq.shape[-1])
            s, fused = core.score(
                qrep, e.repeat(len(qsel), 1), t_.repeat(len(qsel), 1),
                r_in.repeat(len(qsel), 1), v.repeat(len(qsel), 1),
                return_fused=True)
        s = s.float().view(len(qsel), S)
        m = torch.from_numpy(rel).to(device)
        c = torch.from_numpy(cnd).to(device)

        has_pos = m.any(1)
        if not has_pos.any():
            gstep += 1; sched.step(); continue
        sp, mp, cp = s[has_pos], m[has_pos], c[has_pos]
        nce = (torch.logsumexp(sp, 1)
               - torch.logsumexp(sp.masked_fill(~mp, -1e4), 1)).mean()
        worst_pos = sp.masked_fill(~mp, 1e4).min(1).values
        cond_loss = sp.sum() * 0.0
        hc = cp.any(1)
        if hc.any():
            best_cond = sp.masked_fill(~cp, -1e4).max(1).values
            cond_loss = args.w_cond * F.relu(
                args.cond_margin - (worst_pos - best_cond))[hc].mean()
        vic, v_var, v_cov = vicreg_terms(fused.float(), args.w_var, args.w_cov)
        loss = nce + cond_loss + args.w_vicreg * vic

        optim.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(core.trainable_parameters(), 1.0)
        optim.step(); sched.step()
        gstep += 1
        # tau anneal AFTER the step: mutating a buffer that the forward pass
        # consumed would bump its autograd version counter and break backward.
        if args.tau_final is not None:
            frac = min(1.0, gstep / max(1, args.steps))
            core.tau.data.fill_(
                max(args.tau + (args.tau_final - args.tau) * frac, 0.1))

        if gstep % args.log_every == 0 and rank == 0:
            with torch.no_grad():
                p_ = torch.softmax(sp, 1)
                eff = float(torch.exp(-(p_ * (p_ + 1e-12).log()).sum(1)).mean())
                dbal = float((dom[idx] == 1).mean())
            rec = {"step": gstep, "loss": float(loss), "nce": float(nce),
                   "cond": float(cond_loss), "vic_var": v_var, "vic_cov": v_cov,
                   "gnorm": float(gnorm), "eff_support": eff,
                   "waymo_frac": dbal, "tau": float(core.tau),
                   "lr": sched.get_last_lr()[0],
                   "elapsed_h": (time.time() - t_start) / 3600.0}
            hist.append(rec)
            print(f"[s{gstep}] loss={rec['loss']:.4f} nce={rec['nce']:.4f} "
                  f"cond={rec['cond']:.4f} var={v_var:.4f} cov={v_cov:.4f} "
                  f"|g|={rec['gnorm']:.3f} eff={eff:.1f}/{S} "
                  f"wa={dbal:.2f} lr={rec['lr']:.2e}", flush=True)
        if gstep % args.ckpt_every_steps == 0:
            save_ckpt(work / f"step{gstep}.pt", model, optim, sched, scaler,
                      gstep, hist, gen, rank,
                      {"args": vars(args), "n_queries": len(queries)})

    save_ckpt(work / "final.pt", model, optim, sched, scaler, gstep, hist,
              gen, rank, {"args": vars(args), "n_queries": len(queries)})
    if rank == 0:
        (work / "loss_history.json").write_text(json.dumps(hist, indent=1))
        print(f"[done] {gstep} steps in {(time.time()-t_start)/3600:.2f} h → {work}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
