"""Training losses for SceneMiner.

Uses mmdet/mmengine loss implementations where they exist, and adds
InfoNCE (not in mmdet) from scratch.

Combined loss:
    L = w_embed·InfoNCE + w_tag·mmdet.CrossEntropyLoss(BCE) + w_risk·mmdet.SmoothL1Loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── InfoNCE (contrastive embedding loss — not in mmdet) ───────────────────────

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE loss for scene embeddings with a momentum queue.

    When ``batch_size >= 2`` the standard symmetric InfoNCE is used.
    When ``batch_size == 1`` (or any B < 2) a queue of past embeddings acts as
    negatives, giving a real gradient signal even for single-sample batches.

    The queue is a FIFO ring-buffer of ``queue_size`` L2-normalised embeddings
    stored as a plain tensor attribute (not a registered buffer, so it is *not*
    included in checkpoints — it is rebuilt each training run, which is fine).

    Args:
        temperature: Initial logit temperature (learnable).
        queue_size: Number of past embeddings kept as negatives.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable: bool = True,
        queue_size: int = 256,
    ) -> None:
        super().__init__()
        if learnable:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(temperature)))
        self.queue_size = queue_size
        # Lazily allocated on first forward so we know d_embed and device.
        self._queue: torch.Tensor | None = None
        self._queue_ptr: int = 0

    def _init_queue(self, d: int, device: torch.device) -> None:
        q = F.normalize(torch.randn(self.queue_size, d, device=device), dim=1)
        self._queue = q
        self._queue_ptr = 0

    def _enqueue(self, keys: torch.Tensor) -> None:
        """Insert ``keys`` (B, D) into the FIFO queue (no grad)."""
        keys = keys.detach()
        B = keys.shape[0]
        ptr = self._queue_ptr
        # Wrap-around: insert in one or two chunks
        end = min(ptr + B, self.queue_size)
        self._queue[ptr:end] = keys[: end - ptr]
        if end - ptr < B:
            remainder = B - (end - ptr)
            self._queue[:remainder] = keys[end - ptr:]
            self._queue_ptr = remainder
        else:
            self._queue_ptr = end % self.queue_size

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Args: emb_a, emb_b — (B, D) L2-normalised embeddings."""
        B, D = emb_a.shape
        temp = self.log_temp.exp().clamp(min=1e-4)

        # Always compute in fp32 — fp16 gradients overflow at temp=0.07 under AMP.
        emb_a = emb_a.float()
        emb_b = emb_b.float()

        # Ensure queue is initialised (needed for both B>=2 and B==1 paths).
        if self._queue is None:
            self._init_queue(D, emb_a.device)
        elif self._queue.device != emb_a.device:
            self._queue = self._queue.to(emb_a.device)

        if B >= 2:
            # In distributed training, gather emb_b from all ranks so every GPU
            # sees the full set of negatives (similar to SimCLR's all-gather).
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                world = _dist.get_world_size()
                gathered = [torch.zeros_like(emb_b) for _ in range(world)]
                _dist.all_gather(gathered, emb_b.contiguous())
                # Replace current rank's slice with the gradient-carrying copy.
                gathered[_dist.get_rank()] = emb_b
                emb_b_all = torch.cat(gathered, dim=0)      # (B*world, D)
                # Labels: each local emb_a[i] matches emb_b_all at offset rank*B+i
                offset = _dist.get_rank() * B
                logits_ab = emb_a @ emb_b_all.T / temp
                labels = torch.arange(offset, offset + B, device=emb_a.device)
                loss = F.cross_entropy(logits_ab, labels)
            else:
                logits = (emb_a @ emb_b.T) / temp
                labels = torch.arange(B, device=logits.device)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
            if self.queue_size > 0:
                self._enqueue(emb_b)
            return loss

        # B == 1: use queue as negatives.
        # keys = [positive (emb_b[0]), negatives (queue)]
        keys = torch.cat([emb_b, self._queue], dim=0)   # (1 + Q, D)
        logits = (emb_a @ keys.T) / temp                 # (1, 1 + Q)
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)  # positive at idx 0
        loss = F.cross_entropy(logits, labels)
        self._enqueue(emb_b)
        return loss


# ── Tagging loss — Binary CE from mmdet when available ────────────────────────

def _build_tagging_loss(pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Return mmdet CrossEntropyLoss (use_sigmoid=True) or fallback BCE."""
    try:
        from mmdet.models.losses import CrossEntropyLoss
        return CrossEntropyLoss(use_sigmoid=True, reduction="mean", loss_weight=1.0)
    except Exception:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ── Risk loss — SmoothL1 from mmdet when available ────────────────────────────

def _build_risk_loss() -> nn.Module:
    """Return mmdet SmoothL1Loss or PyTorch HuberLoss."""
    try:
        from mmdet.models.losses import SmoothL1Loss
        return SmoothL1Loss(beta=1.0, reduction="mean", loss_weight=1.0)
    except Exception:
        return nn.HuberLoss(delta=1.0)


class TaggingLoss(nn.Module):
    """Multi-label BCE tagging loss."""
    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self._loss = _build_tagging_loss(pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss(logits, targets.float())


class RiskLoss(nn.Module):
    """Huber/SmoothL1 risk regression loss."""
    def __init__(self) -> None:
        super().__init__()
        self._loss = _build_risk_loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target.float())


# ── Motion (trajectory) loss ──────────────────────────────────────────────

class MotionLoss(nn.Module):
    """Smooth L1 loss for multi-agent trajectory prediction with agent masking.

    Waypoints are in metres (ego frame), so raw Smooth L1 values can reach
    5-20 per step. That dominates the other losses (embed≈0.7, tag≈0.7,
    risk≈0.02) and caused gradient explosion + NaN corruption on the
    randomly-initialised MotionHead. We scale by 1/waypoint_scale so the
    loss sits in [0, 1] range, matching the other losses.

    Args:
        beta: Smooth L1 transition point (default 1.0).
        waypoint_scale: Typical waypoint distance (m). Predictions and GT
            are both divided by this, so loss is in "normalised" space.
    """

    def __init__(self, beta: float = 1.0, waypoint_scale: float = 20.0) -> None:
        super().__init__()
        self.beta = beta
        self.waypoint_scale = waypoint_scale

    def forward(
        self,
        traj_pred: torch.Tensor,
        traj_gt: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        waypoint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            traj_pred: (B, N_agents, T, 2) predicted waypoints (metres).
            traj_gt:   (B, N_agents, T, 2) ground truth waypoints (metres).
            agent_mask:    (B, N_agents) bool, True = valid agent.
            waypoint_mask: (B, N_agents, T) bool/uint8, True = valid future step.
                Without this, padded (0, 0) steps are scored as if the agent
                "drove to origin" — about 25% of timesteps in nuScenes have
                this padding, and including them collapses the head toward
                predicting zeros.
        Returns:
            Scalar loss.
        """
        pred_n = traj_pred / self.waypoint_scale
        gt_n   = traj_gt   / self.waypoint_scale
        loss = F.smooth_l1_loss(pred_n, gt_n, beta=self.beta, reduction="none")  # (B, N, T, 2)

        if agent_mask is None and waypoint_mask is None:
            return loss.mean()

        # Build combined mask of shape (B, N, T, 1).
        if waypoint_mask is not None:
            wp_mask = waypoint_mask.to(loss.dtype).unsqueeze(-1)            # (B, N, T, 1)
        else:
            wp_mask = torch.ones_like(loss[..., :1])
        if agent_mask is not None:
            ag_mask = agent_mask.to(loss.dtype).unsqueeze(-1).unsqueeze(-1) # (B, N, 1, 1)
        else:
            ag_mask = 1.0
        full_mask = wp_mask * ag_mask                                      # (B, N, T, 1)

        loss = loss * full_mask
        n_valid = full_mask.sum().clamp(min=1.0) * 2                       # × 2 for (x, y)
        return loss.sum() / n_valid


# ── Combined loss ─────────────────────────────────────────────────────────────

class SceneMinerLoss(nn.Module):
    """Weighted combination of all SceneMiner training losses.

    L = w_embed·InfoNCE + w_tag·BCE_tags + w_risk·SmoothL1_risk + w_occ·BCE_occ
    Caption CE is passed in externally (computed by VLM forward pass).

    Args:
        w_embed: InfoNCE weight.
        w_tag: Tagging BCE weight.
        w_risk: Risk regression weight.
        w_caption: Caption CE weight (applied when caption_loss is provided).
        w_occ: BEV occupancy BCE weight (used in stage0 pre-training).
        temperature: InfoNCE initial temperature.
        queue_size: InfoNCE queue size.
    """

    def __init__(
        self,
        w_embed: float = 1.0,
        w_tag: float = 0.5,
        w_risk: float = 0.2,
        w_caption: float = 0.3,
        w_occ: float = 0.0,
        w_motion: float = 0.0,
        temperature: float = 0.07,
        queue_size: int = 256,
        motion_waypoint_scale: float = 20.0,
        motion_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.w_embed = w_embed
        self.w_tag = w_tag
        self.w_risk = w_risk
        self.w_caption = w_caption
        self.w_occ = w_occ
        self.w_motion = w_motion
        self.infonce = InfoNCELoss(temperature, queue_size=queue_size)
        self.tagging = TaggingLoss()
        self.risk = RiskLoss()
        self.motion = MotionLoss(beta=motion_beta, waypoint_scale=motion_waypoint_scale) if w_motion > 0 else None

    def forward(
        self,
        embeddings_a: torch.Tensor | None,
        embeddings_b: torch.Tensor | None,
        tag_logits: torch.Tensor | None = None,
        tag_targets: torch.Tensor | None = None,
        risk_preds: torch.Tensor | None = None,
        risk_targets: torch.Tensor | None = None,
        caption_loss: torch.Tensor | None = None,
        occ_logits: torch.Tensor | None = None,
        occ_gt: torch.Tensor | None = None,
        traj_pred: torch.Tensor | None = None,
        traj_gt: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
        waypoint_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with keys: total, embed, tag (opt), risk (opt), caption (opt), occ (opt)."""
        losses: dict[str, torch.Tensor] = {}
        device = (embeddings_a if embeddings_a is not None else occ_logits).device
        total = torch.tensor(0.0, device=device)

        if embeddings_a is not None and embeddings_b is not None and self.w_embed > 0:
            embed_loss = self.infonce(embeddings_a, embeddings_b)
            losses["embed"] = embed_loss
            total = total + self.w_embed * embed_loss

        if tag_logits is not None and tag_targets is not None and self.w_tag > 0:
            tag_loss = self.tagging(tag_logits, tag_targets)
            losses["tag"] = tag_loss
            total = total + self.w_tag * tag_loss

        if risk_preds is not None and risk_targets is not None and self.w_risk > 0:
            risk_loss = self.risk(risk_preds, risk_targets)
            losses["risk"] = risk_loss
            total = total + self.w_risk * risk_loss

        if caption_loss is not None and self.w_caption > 0:
            losses["caption"] = caption_loss
            total = total + self.w_caption * caption_loss

        if occ_logits is not None and occ_gt is not None and self.w_occ > 0:
            if occ_logits.dim() == 3:
                # Binary mode (B, H, W) — legacy BCE with dynamic pos_weight
                gt_f = occ_gt.float().to(device)
                with torch.no_grad():
                    n_pos = gt_f.sum().clamp(min=1.0)
                    n_neg = (gt_f.numel() - n_pos).clamp(min=1.0)
                    pw = (n_neg / n_pos).clamp(max=50.0)
                occ_loss = F.binary_cross_entropy_with_logits(
                    occ_logits.float(), gt_f, pos_weight=pw,
                )
            else:
                # Semantic mode (B, n_classes, H, W) — cross-entropy with
                # dynamic inverse-frequency class weights to handle the heavy
                # background dominance (~95% of cells are class 0).
                gt_long = occ_gt.long().to(device)
                n_classes = occ_logits.shape[1]
                with torch.no_grad():
                    gt_flat = gt_long.view(-1)
                    counts = torch.bincount(gt_flat, minlength=n_classes).float().clamp(min=1.0)
                    weights = (gt_flat.numel() / (n_classes * counts)).clamp(max=50.0)
                occ_loss = F.cross_entropy(
                    occ_logits.float(), gt_long, weight=weights.to(device),
                )
            losses["occ"] = occ_loss
            total = total + self.w_occ * occ_loss

        if self.motion is not None and traj_pred is not None and traj_gt is not None:
            motion_loss = self.motion(traj_pred, traj_gt, agent_mask, waypoint_mask)
            losses["motion"] = motion_loss
            total = total + self.w_motion * motion_loss

        losses["total"] = total
        return losses
