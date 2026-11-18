#!/usr/bin/env python3
"""WACV multi-domain DDP training — unified nuScenes + Waymo stream.

Trains the SceneMiner heads (and optionally Qwen LoRA adapters) on a
single interleaved nuScenes+Waymo data stream, with IN-BAND MASKED
losses so each domain only contributes supervision where it actually has
ground truth.

Pipeline (who-feeds-whom):

    torchrun (N procs)
      └─ ddp_setup()                      → process group + per-rank device
      └─ build_model()                    → frozen v8 + trainable heads(+LoRA), DDP-wrapped
      └─ build_unified_loader()           → ConcatDataset[nusc,waymo]
                                            + DistributedMultinomialSampler
                                            + universal_collate
      └─ for epoch: for batch:
            forward_model(batch)          → SceneMinerOutput
            WACVMaskedLoss(out, batch)    → total + per-component + per-DOMAIN logs
            backward / optim.step
            checkpoint @ step 500 and each epoch end (rank 0)

Safe-import contract: this script only READS the frozen BMVC modules
(``scene_mining.models``, ``scene_mining.losses``); the WACV-specific
masked loss lives here so no baseline file is edited.

Launch (this node has 2× A40; use --nproc_per_node=4 on a 4-GPU node):
    torchrun --nproc_per_node=4 scripts/train_wacv.py \
        --config configs/experiment/nuscenes_scenemining_v8_unified.py \
        --waymo work_dirs/waymo_zeroshot/extracted_large \
        --nusc-root data/nuscenes --nusc-version v1.0-trainval --nusc-split train \
        --weights 0.6 0.4 --batch-size 2 --epochs 3 --use-lora
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scene_mining.data import (        # noqa: E402
    NuScenesUniversalDataset, WaymoUniversalDataset,
    DistributedMultinomialSampler, universal_collate,
)
from scene_mining.losses.losses import InfoNCELoss, MotionLoss   # read-only reuse

from paths import QWEN_VL  # noqa: E402
TOKENIZER = QWEN_VL


# ──────────────────────────────────────────────────────────────────────
# Numerically-stabilized InfoNCE (WACV-local; baseline losses.py untouched)
# ──────────────────────────────────────────────────────────────────────
class StableInfoNCELoss(InfoNCELoss):
    """InfoNCE with cosine-sim clamping + a temperature floor.

    Subclasses the frozen baseline ``InfoNCELoss`` (reusing its FIFO queue
    + DDP all-gather machinery) and overrides ONLY ``forward`` to add the
    numerical guards that prevented the step-8 NaN explosion under LoRA:

      * everything in fp32 (LoRA-perturbed bf16 embeddings overflow),
      * cosine similarities clamped to [-1+e, 1-e] BEFORE the /tau divide,
        so a degenerate row can't push a logit to +-inf,
      * tau floored (``temp_floor``) so the learnable temperature can't
        collapse toward 0 and blow logits up.

    The baseline class stays byte-identical for BMVC reproducibility.
    """

    def __init__(self, temperature=0.07, learnable=True, queue_size=256,
                 temp_floor: float = 0.05, sim_eps: float = 1e-6):
        super().__init__(temperature, learnable, queue_size)
        self.temp_floor = float(temp_floor)
        self.sim_eps = float(sim_eps)

    def _clamp_sim(self, s):
        return torch.clamp(s, min=-1.0 + self.sim_eps, max=1.0 - self.sim_eps)

    def forward(self, emb_a, emb_b):
        B, D = emb_a.shape
        tau = self.log_temp.exp().clamp(min=self.temp_floor)   # floored tau
        emb_a = F.normalize(emb_a.float(), dim=-1)             # re-normalise in fp32
        emb_b = F.normalize(emb_b.float(), dim=-1)

        if self._queue is None:
            self._init_queue(D, emb_a.device)
        elif self._queue.device != emb_a.device:
            self._queue = self._queue.to(emb_a.device)

        if B >= 2:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                world = _dist.get_world_size()
                gathered = [torch.zeros_like(emb_b) for _ in range(world)]
                _dist.all_gather(gathered, emb_b.contiguous())
                gathered[_dist.get_rank()] = emb_b
                emb_b_all = torch.cat(gathered, dim=0)
                sim = self._clamp_sim(emb_a @ emb_b_all.T)
                offset = _dist.get_rank() * B
                labels = torch.arange(offset, offset + B, device=emb_a.device)
                loss = F.cross_entropy(sim / tau, labels)
            else:
                sim = self._clamp_sim(emb_a @ emb_b.T)
                labels = torch.arange(B, device=sim.device)
                loss = (F.cross_entropy(sim / tau, labels)
                        + F.cross_entropy(sim.T / tau, labels)) / 2.0
            if self.queue_size > 0:
                self._enqueue(emb_b)
            return loss

        # B == 1: queue negatives.
        keys = torch.cat([emb_b, self._queue], dim=0)
        sim = self._clamp_sim(emb_a @ keys.T)
        labels = torch.zeros(B, dtype=torch.long, device=sim.device)
        loss = F.cross_entropy(sim / tau, labels)
        self._enqueue(emb_b)
        return loss
# Param-name substrings that define the WACV-trainable heads.
HEAD_NAME_KEYS = ("tagging_head", "risk_head", "embedding_head",
                  "velocity_head", "traj_proj")


# ──────────────────────────────────────────────────────────────────────
# Masked multi-domain loss
# ──────────────────────────────────────────────────────────────────────
class WACVMaskedLoss(nn.Module):
    """Combine the four task losses with in-band per-domain masking.

      * tag  : BCE-with-logits × ``tag_valid_mask`` (Waymo supervises 5/20)
      * risk : SmoothL1 only on rows where ``risk_valid`` (Waymo = NaN → skipped)
      * motion + InfoNCE : computed normally for BOTH domains
    Returns the scalar total AND a log dict with per-component and
    per-DOMAIN breakdowns (so we can confirm both datasets learn).
    """

    def __init__(self, w_embed=1.0, w_tag=0.5, w_risk=0.2, w_motion=0.5,
                 temperature=0.07, queue_size=256, motion_waypoint_scale=20.0,
                 temp_floor=0.05):
        super().__init__()
        self.w_embed, self.w_tag = w_embed, w_tag
        self.w_risk, self.w_motion = w_risk, w_motion
        # Stabilized InfoNCE only built when actually used (w_embed > 0).
        self.infonce = (StableInfoNCELoss(temperature, queue_size=queue_size,
                                          temp_floor=temp_floor)
                        if w_embed > 0 else None)
        self.motion = MotionLoss(waypoint_scale=motion_waypoint_scale)

    # ── masked primitives ────────────────────────────────────────────
    @staticmethod
    def _masked_tag(logits, targets, valid):
        per = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none")          # (B,20)
        v = valid.float()
        denom = v.sum().clamp(min=1.0)
        return (per * v).sum() / denom

    @staticmethod
    def _masked_risk(pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        valid = torch.isfinite(target)
        if valid.sum() == 0:                       # all-Waymo batch → no risk GT
            return pred.sum() * 0.0                # keeps graph, zero grad
        per = F.smooth_l1_loss(pred[valid], target[valid], reduction="mean")
        return per

    def forward(self, out, batch, domain_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # GT tensors arrive on CPU from the collate; move them to the
        # model-output device and compute the loss in fp32 (BCE/InfoNCE are
        # numerically happier in fp32 than under bf16 autocast).
        dev = out.tag_logits.device
        tags   = batch["semantic_tags"]["values"].to(dev)
        tvalid = batch["semantic_tags"]["valid_mask"].to(dev)
        risk_t = batch["risk_scalar"].to(dev)
        traj   = batch["trajectory_gt"]["waypoints"].to(dev)
        amask  = batch["trajectory_gt"]["agent_mask"].to(dev)
        wmask  = batch["trajectory_gt"]["waypoint_mask"].to(dev)

        tag_logits = out.tag_logits.float()
        risk_pred  = out.risk_scores.float()
        traj_pred  = out.traj_pred.float()
        emb_a, emb_b = out.embeddings.float(), out.embeddings_b.float()

        l_tag    = self._masked_tag(tag_logits, tags, tvalid)
        l_risk   = self._masked_risk(risk_pred, risk_t)
        l_motion = self.motion(traj_pred, traj, amask, wmask)
        # BYPASS embed entirely when disabled — never compute it, so a NaN
        # embedding can't reach the total via 0 * NaN = NaN.
        embed_zeroed = False
        if self.w_embed > 0 and self.infonce is not None:
            l_embed = self.infonce(emb_a, emb_b)
            # ── PER-TERM EMBED GUARD (DDP-safe) ───────────────────────
            # InfoNCE occasionally overflows to NaN/Inf in the embedding
            # head (bf16) and the all-gather spreads it.  Rather than skip
            # the WHOLE step (losing finite tag/risk/motion grads), drop
            # ONLY the embed term for this step.  The zero decision is
            # COLLECTIVE: if embed is non-finite on ANY rank, EVERY rank
            # zeroes it — so the embed head's used/unused status stays
            # identical across ranks and DDP never sees a grad mismatch.
            embed_bad = torch.tensor(
                [0.0 if torch.isfinite(l_embed) else 1.0], device=l_embed.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(embed_bad, op=dist.ReduceOp.MAX)
            if embed_bad.item() > 0:
                # Replace (not multiply — NaN*0 = NaN) with a fresh constant
                # that carries no gradient to the embed head this step.
                l_embed = torch.zeros((), device=tag_logits.device)
                embed_zeroed = True
        else:
            l_embed = torch.zeros((), device=tag_logits.device)

        total = (self.w_tag * l_tag + self.w_risk * l_risk
                 + self.w_motion * l_motion + self.w_embed * l_embed)

        log = {"total": float(total.detach()),
               "tag": float(l_tag.detach()), "risk": float(l_risk.detach()),
               "motion": float(l_motion.detach()), "embed": float(l_embed.detach()),
               "embed_zeroed": bool(embed_zeroed)}

        # ── per-domain breakdown (detached, logging only) ─────────────
        with torch.no_grad():
            for did, name in ((0, "nuscenes"), (1, "waymo")):
                sel = domain_ids == did
                if sel.sum() == 0:
                    continue
                log[f"{name}/n"] = int(sel.sum())
                log[f"{name}/tag"] = float(self._masked_tag(
                    tag_logits[sel], tags[sel], tvalid[sel]).detach())
                rk = self._masked_risk(risk_pred.view(-1)[sel], risk_t[sel])
                log[f"{name}/risk"] = float(rk.detach())
                log[f"{name}/motion"] = float(self.motion(
                    traj_pred[sel], traj[sel], amask[sel], wmask[sel]).detach())
        return total, log


# ──────────────────────────────────────────────────────────────────────
# DDP setup
# ──────────────────────────────────────────────────────────────────────
def ddp_setup():
    """Read torchrun env, init the process group, bind this rank's GPU."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world > 1
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}"
                          if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return rank, local_rank, world, distributed, device


def is_main(rank: int) -> bool:
    return rank == 0


def log0(rank, *a):
    if is_main(rank):
        print(*a, flush=True)


# ──────────────────────────────────────────────────────────────────────
# Model build + freeze
# ──────────────────────────────────────────────────────────────────────
def _chain_load(model, base_ckpt, v8_ckpt, rank):
    """Optional: warm-load v4 base then overlay v8 trainable delta."""
    for tag, path in (("base", base_ckpt), ("v8", v8_ckpt)):
        if not path:
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck.get("model", ck))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log0(rank, f"[load] {tag}: {path}  "
                   f"(loaded {len(sd)} tensors, {len(missing)} missing)")


def build_model(cfg, device, args, rank):
    from scene_mining.models.scene_miner import SCENE_MINING_MODELS  # noqa: F401
    model = SCENE_MINING_MODELS.build(cfg.model)
    _chain_load(model, args.base_checkpoint, args.v8_checkpoint, rank)

    # 1) Freeze EVERYTHING.
    for p in model.parameters():
        p.requires_grad_(False)

    # 2) Unfreeze the downstream heads (by name).  With --freeze-motion the
    #    motion params (traj_proj + velocity_head) stay frozen at the
    #    chain-loaded BMVC optimum: only the semantic heads + LoRA train,
    #    while the motion LOSS still anchors LoRA (traj_head reads the
    #    LoRA-influenced VLM tokens, so its gradient reaches the adapters).
    freeze_motion = bool(getattr(args, "freeze_motion", False))
    motion_keys = ("velocity_head", "traj_proj")
    n_head, n_motion_frozen = 0, 0
    for name, p in model.named_parameters():
        if any(k in name for k in HEAD_NAME_KEYS):
            if freeze_motion and any(k in name for k in motion_keys):
                n_motion_frozen += p.numel()
                continue
            p.requires_grad_(True)
            n_head += p.numel()

    # 3) Optionally enable Qwen LoRA adapters (if the model was built with
    #    them).  We flip only ``lora_*`` params — NOT the backbone blocks
    #    that ``freeze_for_stage3`` would also unfreeze.
    n_lora = 0
    if args.use_lora:
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
                n_lora += p.numel()
        if n_lora == 0:
            log0(rank, "[freeze] WARNING: --use-lora set but model has no "
                       "lora_ params (config did not inject LoRA). "
                       "Training heads only.")

    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log0(rank, f"[freeze] trainable: heads={n_head:,} params"
               + (f" + LoRA={n_lora:,}" if args.use_lora else "")
               + (f" | motion FROZEN ({n_motion_frozen:,} params pinned)"
                  if freeze_motion else "")
               + f"  ({sum(p.numel() for p in trainable):,} total grad params)")
    return model, trainable


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────
def build_unified_loader(cfg, args, rank, world):
    from scene_mining.datasets.nuscenes_dataset import NuScenesFrameDataset
    waymo = WaymoUniversalDataset(args.waymo, tokenizer_path=TOKENIZER)
    nusc_inner = NuScenesFrameDataset(
        dataroot=args.nusc_root, version=args.nusc_version, split=args.nusc_split,
        labels_dir=args.labels, tokenizer_path=TOKENIZER,
        n_agents=8, n_waypoints=6, max_text_len=128,
    )
    nusc = NuScenesUniversalDataset(nusc_inner, tokenizer_path=TOKENIZER)
    log0(rank, f"[data] nuScenes={len(nusc)}  Waymo={len(waymo)} frames")

    concat = ConcatDataset([nusc, waymo])         # order fixes domain ids: 0,1
    # Epoch size: if --steps-per-epoch <= 0, derive ONE effective pass over
    # the combined dataset (the honest "full-trainval" epoch).  Otherwise use
    # the explicit step budget.  Either way it's a global (all-rank) total.
    if args.steps_per_epoch and args.steps_per_epoch > 0:
        samples_per_epoch = args.steps_per_epoch * args.batch_size * world
    else:
        samples_per_epoch = len(concat)           # ≈ one pass over nusc+waymo
    sampler = DistributedMultinomialSampler(
        dataset_sizes=[len(nusc), len(waymo)], weights=args.weights,
        num_samples=samples_per_epoch, num_replicas=world, rank=rank, seed=0)
    # Per-rank batches per epoch (drop_last) — the unit global_step counts in.
    batches_per_epoch = sampler.num_samples // args.batch_size
    log0(rank, f"[data] epoch = {samples_per_epoch} samples (global) → "
               f"{sampler.num_samples}/rank → {batches_per_epoch} batches/rank "
               f"(batch_size={args.batch_size}, world={world})")
    loader = DataLoader(concat, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, collate_fn=universal_collate,
                        pin_memory=True, drop_last=True)
    # nuScenes occupies global indices [0, len(nusc)); Waymo the rest.
    return loader, sampler, len(nusc), batches_per_epoch


def domain_ids_from_batch(batch, device) -> torch.Tensor:
    """0 = nuScenes, 1 = Waymo, read from the per-item meta side-channel."""
    return torch.tensor([0 if m["dataset"] == "nuscenes" else 1
                         for m in batch["meta"]], device=device)


def derive_agent_positions(traj_gt, wp_mask):
    """First valid GT waypoint per agent → Phase-2 grounding anchor."""
    wm = wp_mask.to(traj_gt.dtype)
    first_valid = wm.argmax(dim=-1)
    any_valid = wm.sum(dim=-1) > 0
    idx = first_valid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
    anchor = torch.gather(traj_gt, 2, idx).squeeze(2)
    return anchor * any_valid.to(traj_gt.dtype).unsqueeze(-1)


def _sanitize_agent_mask(amask: torch.Tensor) -> torch.Tensor:
    """Prevent fully-masked MotionHead self-attention.

    Localised root cause of the ~14% training NaN: a frame with ZERO valid
    agents makes the MotionHead decoder's self-attention key_padding_mask
    mask EVERY key → softmax(all -inf) = NaN (in fp32 too — this is a
    masking degeneracy, not a precision issue).  ~2.5% of nuScenes frames
    are zero-agent; across 8 frames/step (4 ranks x bs 2) that is ~14% of
    steps — exactly the observed skip rate.

    Fix: for any zero-agent sample, unmask slot 0 ONLY for the model's
    attention input.  This is loss-safe: the loss uses the ORIGINAL batch
    agent_mask (still all-False for that frame), and slot 0 has no GT
    waypoints, so the mask-aware motion loss contributes 0 for it.
    """
    no_agents = ~amask.any(dim=1)                # (B,) True where 0 agents
    if no_agents.any():
        amask = amask.clone()
        amask[no_agents, 0] = True               # give attention ≥1 key
    return amask


def forward_model(model, batch, device):
    imgs = batch["camera_images"].to(device)
    intr = batch["bev_features"]["intrinsics"].to(device)
    extr = batch["bev_features"]["extrinsics"].to(device)
    ids  = batch["text_queries"]["input_ids"].to(device)
    am   = batch["text_queries"]["attention_mask"].to(device)
    amask = batch["trajectory_gt"]["agent_mask"].to(device)
    traj  = batch["trajectory_gt"]["waypoints"].to(device)
    wmask = batch["trajectory_gt"]["waypoint_mask"].to(device)
    agent_positions = derive_agent_positions(traj, wmask)
    # Sanitize the mask passed to the (frozen) MotionHead attention only.
    amask_model = _sanitize_agent_mask(amask)
    return model(images=imgs, intrinsics=intr, extrinsics=extr,
                 input_ids=ids, attention_mask=am, image_hw=None,
                 agent_mask=amask_model, temporal_images=None,
                 temporal_intrinsics=None, temporal_extrinsics=None,
                 temporal_ego_poses=None, agent_positions=agent_positions)


# ──────────────────────────────────────────────────────────────────────
# Checkpointing (rank 0, trainable params only → small files)
# ──────────────────────────────────────────────────────────────────────
def save_ckpt(model, optim, scaler, step, epoch, out_dir, tag, rank):
    """Resumable checkpoint (rank 0 only).  Stores ONLY the trainable
    (requires_grad) params — the frozen 3B backbone is identical every run,
    so this keeps files at a few MB.  Includes optimizer (Adam momentum),
    GradScaler, and the exact (epoch, global_step) position."""
    if not is_main(rank):
        return
    core = model.module if hasattr(model, "module") else model
    trainable_sd = {n: p.detach().cpu()
                    for n, p in core.named_parameters() if p.requires_grad}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # Atomic write: save to .tmp then rename, so a time-limit kill mid-write
    # can never leave a half-written (corrupt) checkpoint.
    path = out / f"wacv_{tag}.pth"
    tmp = out / f"wacv_{tag}.pth.tmp"
    torch.save({
        "model_state_dict": trainable_sd,                 # trainable params only
        "optimizer_state_dict": optim.state_dict(),       # Adam momentum
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": step,
    }, tmp)
    os.replace(tmp, path)                                  # atomic on POSIX
    # Also refresh a stable "latest" pointer for --resume convenience.
    latest = out / "wacv_latest.pth"
    ltmp = out / "wacv_latest.pth.tmp"
    torch.save(torch.load(path, map_location="cpu", weights_only=False), ltmp)
    os.replace(ltmp, latest)
    print(f"[ckpt] saved {path} + wacv_latest.pth  "
          f"({len(trainable_sd)} trainable tensors, step={step}, epoch={epoch})",
          flush=True)


def load_ckpt(model, optim, scaler, path, rank):
    """Restore trainable weights + optimizer + scaler from a checkpoint.
    Returns (resume_epoch, resume_global_step).  Frozen backbone is left as
    built (it is not in the checkpoint)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    core = model.module if hasattr(model, "module") else model
    missing, unexpected = core.load_state_dict(ck["model_state_dict"], strict=False)
    # 'missing' is expected & huge (the frozen backbone isn't in the ckpt);
    # what matters is that NONE of the trainable tensors are unexpected.
    if optim is not None and ck.get("optimizer_state_dict") is not None:
        optim.load_state_dict(ck["optimizer_state_dict"])
    if scaler is not None and ck.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ck["scaler_state_dict"])
    ep, gs = int(ck["epoch"]), int(ck["global_step"])
    log0(rank, f"[resume] loaded {path}: restored "
               f"{len(ck['model_state_dict'])} trainable tensors, optimizer, "
               f"scaler | resuming at epoch={ep} global_step={gs} "
               f"({len(unexpected)} unexpected keys)")
    return ep, gs


# ──────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--waymo", default="work_dirs/waymo_zeroshot/extracted_large")
    ap.add_argument("--nusc-root", default="data/nuscenes")
    ap.add_argument("--nusc-version", default="v1.0-trainval")
    ap.add_argument("--nusc-split", default="train")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--weights", type=float, nargs=2, default=[0.6, 0.4])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--freeze-motion", action="store_true",
                    help="Pin traj_proj + velocity_head at their chain-loaded "
                         "values; train only semantic heads + LoRA. The motion "
                         "loss stays active as a LoRA anchor.")
    ap.add_argument("--lora-lr", type=float, default=1e-5,
                    help="LoRA-group learning rate (heads use --lr).")
    ap.add_argument("--lora-warmup-steps", type=int, default=200,
                    help="Linear warmup steps for the LoRA group (0→lora-lr).")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max_norm for clip_grad_norm_ before each step.")
    ap.add_argument("--w-embed", type=float, default=None,
                    help="InfoNCE weight; 0.0 fully bypasses the embed loss. "
                         "Overrides cfg.wacv_loss.w_embed when set.")
    ap.add_argument("--temp-floor", type=float, default=0.05,
                    help="Min InfoNCE temperature (prevents logit blow-up).")
    ap.add_argument("--base-checkpoint", default=None)
    ap.add_argument("--v8-checkpoint", default=None)
    ap.add_argument("--work-dir", default="work_dirs/wacv_unified")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every-steps", type=int, default=500,
                    help="Periodic checkpoint cadence (also saves at epoch end).")
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16",
                    help="bf16 (no GradScaler) or fp16 (enables GradScaler).")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint path to resume from (restores weights, "
                         "optimizer, scaler, epoch, global_step + fast-forwards).")
    ap.add_argument("--smoke-steps", type=int, default=0,
                    help=">0: run only this many steps then exit (CI smoke).")
    args = ap.parse_args()

    rank, local_rank, world, distributed, device = ddp_setup()
    log0(rank, f"[ddp] world_size={world} device={device} "
               f"distributed={distributed}")

    from mmengine.config import Config
    cfg = Config.fromfile(args.config)

    model, trainable = build_model(cfg, device, args, rank)
    # Split trainable params into head vs LoRA groups BEFORE the DDP wrap
    # (clean names).  LoRA gets its own (lower) LR + a linear warmup so the
    # early representation shift is smooth — the bf16 LoRA explosion we saw
    # previously was amplified by full-LR cold-start adapters.
    head_params, lora_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lora_params if "lora_" in n else head_params).append(p)

    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters: a head can go grad-less on an all-other-domain
        # batch (e.g. risk_head when every row is Waymo) → DDP must tolerate it.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # Loss config: cfg.wacv_loss as base, CLI overrides on top.
    wacv_loss_cfg = dict(cfg.get("wacv_loss", {}) or {})
    if args.w_embed is not None:
        wacv_loss_cfg["w_embed"] = args.w_embed
    wacv_loss_cfg.setdefault("temp_floor", args.temp_floor)
    loss_fn = WACVMaskedLoss(**wacv_loss_cfg).to(device)
    log0(rank, f"[loss] w_embed={loss_fn.w_embed} "
               f"(InfoNCE {'ACTIVE+clamped' if loss_fn.w_embed > 0 else 'BYPASSED'}) "
               f"temp_floor={args.temp_floor}")

    param_groups = [{"params": head_params, "lr": args.lr, "name": "heads"}]
    lora_group = None
    if lora_params:
        # Start at 0 — the warmup schedule ramps it to --lora-lr by step
        # --lora-warmup-steps.
        param_groups.append({"params": lora_params, "lr": 0.0, "name": "lora"})
    optim = torch.optim.AdamW(param_groups, weight_decay=0.01)
    if lora_params:
        lora_group = optim.param_groups[1]
        log0(rank, f"[optim] heads lr={args.lr} | LoRA lr=0→{args.lora_lr} "
                   f"(linear warmup over {args.lora_warmup_steps} steps) | "
                   f"grad-clip max_norm={args.grad_clip}")
    else:
        log0(rank, f"[optim] heads lr={args.lr} | grad-clip "
                   f"max_norm={args.grad_clip} | (no LoRA params)")
    # GradScaler: enabled ONLY for fp16. Under bf16 it is a no-op (bf16 has
    # fp32 dynamic range), but we keep the object so scaler_state_dict is
    # always present in the checkpoint and fp16 is a one-flag switch.
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp_dtype == "fp16"))
    loader, sampler, _, batches_per_epoch = build_unified_loader(cfg, args, rank, world)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch, global_step, batches_to_skip = 0, 0, 0
    if args.resume:
        start_epoch_ck, global_step = load_ckpt(model, optim, scaler, args.resume, rank)
        # Map the saved global_step back to (epoch, batch-within-epoch). The
        # sampler is deterministic per (seed, epoch, rank), so re-running the
        # epoch and skipping the consumed batches lands EXACTLY where we left.
        start_epoch = global_step // batches_per_epoch
        batches_to_skip = global_step % batches_per_epoch
        log0(rank, f"[resume] fast-forward → start_epoch={start_epoch}, "
                   f"skip {batches_to_skip} batches of epoch {start_epoch} "
                   f"(batches/epoch={batches_per_epoch})")

    # Per-rank diagnostics (smoke mode only): records exactly which samples
    # THIS rank consumed, so we can prove cross-rank stream disjointness +
    # equal batch counts after the run.
    seen_keys: list = []
    rank_batches = 0
    n_skipped = 0          # WHOLE-step skips (rare full-forward NaN)
    n_embed_rescued = 0    # steps where ONLY the embed term was zeroed (kept)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        # On the resumed epoch, fast-forward past already-consumed batches.
        skip_n = batches_to_skip if epoch == start_epoch else 0
        for bi, batch in enumerate(loader):
            if bi < skip_n:
                continue          # replay-skip (sampler stream is deterministic)
            if args.smoke_steps:
                rank_batches += 1
                seen_keys.extend(
                    [f"{m['dataset']}:{m['scene_token'][:10]}:{m['frame_idx']}"
                     for m in batch["meta"]])
            domain_ids = domain_ids_from_batch(batch, device)
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=(device.type == "cuda")):
                out = forward_model(model, batch, device)
            # Loss in fp32 (outside autocast) for numerical stability.
            total, log = loss_fn(out, batch, domain_ids)
            global_step += 1
            if log.get("embed_zeroed"):
                n_embed_rescued += 1

            # ── Non-finite guard (COLLECTIVE across ranks) ────────────
            # A NaN/Inf on ANY rank must make EVERY rank skip in lockstep —
            # otherwise the healthy ranks call backward()+all-reduce while the
            # bad rank doesn't, and DDP deadlocks.  We all-reduce a bad-flag,
            # then every rank takes the same branch.
            bad = torch.tensor(
                [0.0 if torch.isfinite(total) else 1.0], device=device)
            if distributed:
                dist.all_reduce(bad, op=dist.ReduceOp.SUM)
            skipped = bad.item() > 0
            if skipped:
                optim.zero_grad(set_to_none=True)   # drop any partial grads
                n_skipped += 1
                print(f"[rank {rank}] WARN step {global_step}: NON-FINITE loss "
                      f"(local total={log['total']}, embed={log['embed']}) — "
                      f"zero_grad + SKIP (no backward/step)", flush=True)
            else:
                # LoRA linear warmup: ramp the LoRA group's LR 0 → lora_lr
                # over the first --lora-warmup-steps optimizer steps.
                if lora_group is not None:
                    warm = min(1.0, global_step / max(1, args.lora_warmup_steps))
                    lora_group["lr"] = args.lora_lr * warm
                optim.zero_grad(set_to_none=True)
                # GradScaler path: under bf16 the scaler is disabled, so
                # these calls are equivalent to plain backward()/step().
                scaler.scale(total).backward()
                # Gradient clipping (unscale first so the norm is true under
                # fp16; a no-op unscale under disabled-bf16 scaler).
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(optim)
                scaler.update()
            # Skipped or not, fall through to logging / ckpt / smoke-exit so
            # the loop still advances and can exit on a NaN-only tail.

            if global_step % args.log_every == 0:
                tag_skip = (" [SKIPPED]" if skipped
                            else " [embed-rescued]" if log.get("embed_zeroed")
                            else "")
                msg = (f"[e{epoch} s{global_step}]{tag_skip} "
                       f"total={log['total']:.4f} "
                       f"tag={log['tag']:.4f} risk={log['risk']:.4f} "
                       f"motion={log['motion']:.4f} embed={log['embed']:.4f}")
                for d in ("nuscenes", "waymo"):
                    if f"{d}/n" in log:
                        msg += (f" | {d}(n={log[f'{d}/n']}): "
                                f"tag={log[f'{d}/tag']:.3f} "
                                f"risk={log[f'{d}/risk']:.3f} "
                                f"motion={log[f'{d}/motion']:.3f}")
                log0(rank, msg)

            if (args.ckpt_every_steps and global_step > 0
                    and global_step % args.ckpt_every_steps == 0):
                save_ckpt(model, optim, scaler, global_step, epoch,
                          args.work_dir, f"step{global_step}", rank)
            if args.smoke_steps and global_step >= args.smoke_steps:
                # Each rank dumps its consumed-sample keys + counts so we can
                # verify disjoint streams + equal batch counts post-run.
                import json as _json
                Path(args.work_dir).mkdir(parents=True, exist_ok=True)
                n_nusc = sum(k.startswith("nuscenes") for k in seen_keys)
                n_way = sum(k.startswith("waymo") for k in seen_keys)
                (Path(args.work_dir) / f"rank{rank}_seen.json").write_text(
                    _json.dumps({"rank": rank, "world": world,
                                 "batches": rank_batches, "items": len(seen_keys),
                                 "n_nuscenes": n_nusc, "n_waymo": n_way,
                                 "n_skipped": n_skipped, "keys": seen_keys}))
                print(f"[rank {rank}/{world}] smoke {global_step} steps OK | "
                      f"batches={rank_batches} items={len(seen_keys)} "
                      f"nusc={n_nusc} waymo={n_way} | "
                      f"non-finite steps skipped={n_skipped}", flush=True)
                if distributed:
                    dist.barrier(); dist.destroy_process_group()
                return

        save_ckpt(model, optim, scaler, global_step, epoch,
                  args.work_dir, f"epoch{epoch}", rank)
        log0(rank, f"[epoch {epoch}] done in {time.time()-t0:.1f}s "
                   f"({global_step} steps) | whole-step skips={n_skipped} "
                   f"embed-rescued={n_embed_rescued}")

    if distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
