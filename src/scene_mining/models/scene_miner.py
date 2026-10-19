"""SceneMiner: top-level model using mmengine BaseModule + registry.

All components are registered in SCENE_MINING_MODELS and resolved from
mmengine config dicts, so the full pipeline is config-driven without
any hard-coded class imports in the top-level model.

Data flow (3-stream Q-Former):
    images (B, N_cam, 3, H, W)
        → Backbone (SigLIP2 / DINOv2)     → patch_features (B, N_cam, N_patches, D)
        → SpatialEncoder (BEV / TPV)       → spatial_tokens (B, N_spatial, D_bev)
        → MotionHead (optional)            → motion_tokens  (B, N_agents, D_bev)
        → QFormer (BLIP-2, 3 streams)     → visual_tokens  (B, 32, D_vlm)
            stream 1: BEV spatial tokens
            stream 2: Camera patch features (flattened from backbone)
            stream 3: Motion tokens (from MotionHead)
        → VLM forward                      → hidden_states  (B, 32+L, D_vlm)
        → pool visual positions            → pooled         (B, D_vlm)
        → EmbeddingHead / TaggingHead / RiskHead
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# ── mmengine registry ─────────────────────────────────────────────────────────
try:
    from mmengine.model import BaseModule
    from mmengine.registry import Registry, build_from_cfg

    SCENE_MINING_MODELS = Registry("scene_mining_models", parent=None)

    def build_component(cfg: dict) -> nn.Module:
        """Build a registered component from an mmengine config dict."""
        return build_from_cfg(cfg, SCENE_MINING_MODELS)

    _HAS_MMENGINE = True

except ImportError:
    # Fallback: plain nn.Module, no registry
    BaseModule = nn.Module  # type: ignore[assignment,misc]
    SCENE_MINING_MODELS = None  # type: ignore[assignment]
    _HAS_MMENGINE = False

    def build_component(cfg: dict) -> nn.Module:  # type: ignore[misc]
        raise ImportError(
            "mmengine is required for config-driven component instantiation. "
            "pip install mmengine"
        )


# ── Register backbone wrappers ────────────────────────────────────────────────
if _HAS_MMENGINE:
    from scene_mining.backbones.siglip2 import SigLIP2Backbone
    from scene_mining.backbones.dinov2 import DINOv2Backbone

    SCENE_MINING_MODELS.register_module()(SigLIP2Backbone)
    SCENE_MINING_MODELS.register_module()(DINOv2Backbone)

    from scene_mining.spatial.bevformer.encoder import BEVFormerEncoderWrapper
    from scene_mining.spatial.tpvformer.encoder import TPVFormerEncoderWrapper

    SCENE_MINING_MODELS.register_module(name="BEVFormerEncoder")(BEVFormerEncoderWrapper)
    SCENE_MINING_MODELS.register_module(name="TPVFormerEncoder")(TPVFormerEncoderWrapper)

    from scene_mining.adapters.qformer import QFormer
    SCENE_MINING_MODELS.register_module()(QFormer)

    from scene_mining.vlm.qwen25vl import Qwen25VL
    from scene_mining.vlm.llama32vision import LLaMA32Vision
    from scene_mining.vlm.tiny_vlm import TinyVLM
    SCENE_MINING_MODELS.register_module()(Qwen25VL)
    SCENE_MINING_MODELS.register_module()(LLaMA32Vision)
    SCENE_MINING_MODELS.register_module()(TinyVLM)

    from scene_mining.backbones.tiny_backbone import TinyBackbone
    SCENE_MINING_MODELS.register_module()(TinyBackbone)

    from scene_mining.heads import OccupancyHead, BEVSemanticDecoder, TPVSemanticHead, MotionHead
    SCENE_MINING_MODELS.register_module()(OccupancyHead)
    SCENE_MINING_MODELS.register_module()(BEVSemanticDecoder)
    SCENE_MINING_MODELS.register_module()(TPVSemanticHead)
    SCENE_MINING_MODELS.register_module()(MotionHead)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class SceneMinerOutput:
    embeddings: torch.Tensor       # (B, d_embed) L2-normalised — from head_a
    embeddings_b: torch.Tensor     # (B, d_embed) L2-normalised — from head_b (for InfoNCE)
    tag_logits: torch.Tensor       # (B, n_tags)
    risk_scores: torch.Tensor      # (B, 1)
    captions: list[str] | None
    visual_tokens: torch.Tensor    # (B, n_query, d_vlm)  — for FeatureCache
    pooled: torch.Tensor           # (B, d_vlm)
    spatial_tokens: torch.Tensor | None  # (B, bev_h*bev_w, d_bev) — for OccupancyHead
    occ_logits: torch.Tensor | None      # (B, bev_h, bev_w) — from OccupancyHead
    traj_pred: torch.Tensor | None = None  # (B, N_agents, T, 2) — from MotionHead


# ── SceneMiner ─────────────────────────────────────────────────────────────────

class SceneMiner(BaseModule):
    """Config-driven scene mining model.

    Can be instantiated directly or via mmengine config dict::

        model = SceneMiner(
            backbone=dict(type="SigLIP2Backbone", proj_dim=256, freeze=True),
            spatial_encoder=dict(type="BEVFormerEncoder", bev_h=50, bev_w=50),
            qformer=dict(type="QFormer", d_spatial=256, d_vlm=3584),
            vlm=dict(type="Qwen25VL", load_in_4bit=True),
        )

    Or with direct module instances (no mmengine required).

    Args:
        backbone: Config dict or BaseBackbone instance.
        spatial_encoder: Config dict or nn.Module (BEV/TPV encoder).
        qformer: Config dict or QFormer instance.
        vlm: Config dict or BaseVLM instance.
        n_tags: Number of scene tags for TaggingHead.
        d_embed: Output embedding dimension.
        caption_prompt: Default generation prompt.
        init_cfg: mmengine init config (optional).
    """

    def __init__(
        self,
        backbone: dict | nn.Module,
        spatial_encoder: dict | nn.Module,
        qformer: dict | nn.Module | None = None,
        vlm: dict | nn.Module | None = None,
        n_tags: int = 20,
        d_embed: int = 256,
        caption_prompt: str = "Describe this driving scene:",
        occ_head: dict | nn.Module | None = None,
        traj_head: dict | nn.Module | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg) if _HAS_MMENGINE else super().__init__()

        self.backbone = _resolve(backbone)
        self.spatial_encoder = _resolve(spatial_encoder)
        self.qformer = _resolve(qformer) if qformer is not None else None
        self.vlm = _resolve(vlm) if vlm is not None else None
        self.caption_prompt = caption_prompt

        from scene_mining.heads import EmbeddingHead, TaggingHead, RiskHead, OccupancyHead
        if self.vlm is not None:
            d_vlm = self.vlm.d_model
            # Two independent projection heads — used as two "views" for InfoNCE.
            # Different initialised weights → different embeddings for the same pooled
            # state → non-trivial contrastive loss even at batch_size=1.
            self.embedding_head: EmbeddingHead | None = EmbeddingHead(d_vlm, d_embed)
            self.embedding_head_b: EmbeddingHead | None = EmbeddingHead(d_vlm, d_embed)
            self.tagging_head: TaggingHead | None = TaggingHead(d_vlm, n_tags)
            self.risk_head: RiskHead | None = RiskHead(d_vlm)
        else:
            # Occ-only model: VLM-dependent heads are not needed
            self.embedding_head = None
            self.embedding_head_b = None
            self.tagging_head = None
            self.risk_head = None

        # OccupancyHead: optional, used only in stage0 pre-training
        if occ_head is not None:
            self.occ_head: OccupancyHead | None = (
                _resolve(occ_head) if isinstance(occ_head, dict) else occ_head
            )
        else:
            self.occ_head = None

        # MotionHead: optional, transformer decoder for trajectory prediction
        if traj_head is not None:
            self.traj_head: nn.Module | None = (
                _resolve(traj_head) if isinstance(traj_head, dict) else traj_head
            )
        else:
            self.traj_head = None

        # When True, forward() skips QFormer + VLM for faster stage0 iteration
        self._occ_only_mode: bool = False

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,           # (B, N_cam, 3, H, W)
        intrinsics: torch.Tensor,        # (B, N_cam, 3, 3)
        extrinsics: torch.Tensor,        # (B, N_cam, 4, 4) ego→cam
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_hw: tuple[int, int] | None = None,
        agent_mask: torch.Tensor | None = None,
        temporal_images: torch.Tensor | None = None,        # (B, T, N_cam, 3, H, W)
        temporal_intrinsics: torch.Tensor | None = None,    # (B, T, N_cam, 3, 3)
        temporal_extrinsics: torch.Tensor | None = None,    # (B, T, N_cam, 4, 4)
        temporal_ego_poses: torch.Tensor | None = None,     # (B, T, 4, 4) ego→world
        agent_positions: torch.Tensor | None = None,        # (B, N_agents, 2) Phase-2
    ) -> SceneMinerOutput:
        B, N_cam, C, H, W = images.shape
        image_hw = image_hw or (H, W)

        patch_features = self.backbone(images)                          # (B, N_cam, N_p, D)
        spatial_tokens = self.spatial_encoder(
            patch_features, intrinsics, extrinsics, image_hw
        )                                                               # (B, N_spatial, D)

        # ── Phase-1 temporal BEV stack (only built when supplied) ───────────
        # Re-runs backbone + BEVFormer for each past frame, warps the result
        # into the current ego frame, then concatenates along channels.
        # Past frames are processed under ``torch.no_grad`` so we don't store
        # activations for them — they're context only.  The CURRENT frame is
        # already in ``spatial_tokens`` and is reused (with gradient).
        temporal_spatial_tokens = None
        if (
            temporal_images is not None
            and temporal_ego_poses is not None
            and self.traj_head is not None
            and getattr(self.traj_head, "temporal_proj", None) is not None
        ):
            from scene_mining.spatial.bev_warp import stack_temporal_bev
            T = temporal_images.shape[1]
            bev_h = bev_w = int(spatial_tokens.shape[1] ** 0.5)
            D = spatial_tokens.shape[-1]
            # Past frames: backbone + BEV in no_grad to keep memory bounded.
            past_bevs: list[torch.Tensor] = []
            for t in range(T - 1):
                with torch.no_grad():
                    pf = self.backbone(temporal_images[:, t].contiguous())
                    bev_t = self.spatial_encoder(
                        pf,
                        temporal_intrinsics[:, t].contiguous(),
                        temporal_extrinsics[:, t].contiguous(),
                        image_hw,
                    )                                                # (B, H*W, D)
                # (B, H*W, D) → (B, D, H, W) for grid_sample
                past_bevs.append(
                    bev_t.transpose(1, 2).reshape(B, D, bev_h, bev_w).contiguous()
                )
            # Current frame (with gradient).
            current_bev_chw = (
                spatial_tokens.transpose(1, 2).reshape(B, D, bev_h, bev_w).contiguous()
            )
            stacked = stack_temporal_bev(
                past_bevs + [current_bev_chw],
                temporal_ego_poses,
                self.spatial_encoder.pc_range
                if hasattr(self.spatial_encoder, "pc_range")
                else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            )                                                        # (B, T*D, H, W)
            # Back to token form (B, H*W, T*D) for the MotionHead.
            temporal_spatial_tokens = (
                stacked.flatten(2).transpose(1, 2).contiguous()
            )

        # Stage0 fast path: skip QFormer + VLM, predict occupancy only
        if self._occ_only_mode:
            occ_logits = self.occ_head(spatial_tokens) if self.occ_head is not None else None
            dummy = torch.zeros(B, 1, device=images.device)
            return SceneMinerOutput(
                embeddings=dummy,
                embeddings_b=dummy,
                tag_logits=dummy,
                risk_scores=dummy,
                captions=None,
                visual_tokens=dummy,
                pooled=dummy,
                spatial_tokens=spatial_tokens,
                occ_logits=occ_logits,
            )

        if self.qformer is None or self.vlm is None:
            raise RuntimeError(
                "qformer and vlm must be configured for the full forward pass. "
                "Set _occ_only_mode=True (via freeze_for_stage0) or provide qformer/vlm."
            )

        # ── MotionHead (runs BEFORE Q-Former so motion tokens can be fused) ──
        traj_pred = None
        motion_tokens = None
        if self.traj_head is not None:
            traj_out = self.traj_head(
                spatial_tokens,
                agent_mask=agent_mask,
                temporal_spatial_tokens=temporal_spatial_tokens,
                agent_positions=agent_positions,
            )
            traj_pred = traj_out["traj_pred"]
            motion_tokens = traj_out.get("decoded_tokens")  # (B, N_agents, d_model)

        # ── Camera features for Q-Former stream 2 ────────────────────
        # Flatten backbone patch features: (B, N_cam, N_p, D) → (B, N_cam*N_p, D)
        camera_features = None
        if hasattr(self.qformer, 'camera_proj') and self.qformer.camera_proj is not None:
            camera_features = patch_features.view(B, -1, patch_features.shape[-1])

        # ── Q-Former: fuse all 3 streams → 32 visual tokens ─────────
        visual_tokens = self.qformer(
            spatial_tokens,
            camera_features=camera_features,
            motion_tokens=motion_tokens,
        )                                                               # (B, n_q, D_vlm)

        if input_ids is not None:
            hidden = self.vlm(visual_tokens, input_ids, attention_mask)
            pooled = hidden[:, : self.qformer.n_query_tokens].mean(1)  # (B, D_vlm)
        else:
            pooled = visual_tokens.mean(1)                              # (B, D_vlm)

        occ_logits = self.occ_head(spatial_tokens) if self.occ_head is not None else None

        return SceneMinerOutput(
            embeddings=self.embedding_head(pooled),
            embeddings_b=self.embedding_head_b(pooled),
            tag_logits=self.tagging_head(pooled),
            risk_scores=self.risk_head(pooled),
            captions=None,
            visual_tokens=visual_tokens,
            pooled=pooled,
            spatial_tokens=spatial_tokens,
            occ_logits=occ_logits,
            traj_pred=traj_pred,
        )

    # ── Caption generation ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_captions(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        prompts: list[str] | None = None,
        max_new_tokens: int = 128,
        image_hw: tuple[int, int] | None = None,
    ) -> list[str]:
        B = images.shape[0]
        image_hw = image_hw or images.shape[-2:]
        patch_features = self.backbone(images)
        spatial_tokens = self.spatial_encoder(patch_features, intrinsics, extrinsics, image_hw)
        visual_tokens = self.qformer(spatial_tokens)
        return self.vlm.generate(
            visual_tokens,
            prompts or [self.caption_prompt] * B,
            max_new_tokens=max_new_tokens,
        )

    # ── Staged training helpers ───────────────────────────────────────────────

    def freeze_for_stage0(self) -> None:
        """Stage 0: train spatial encoder + OccupancyHead only (LiDAR pre-train).

        Backbone, QFormer, VLM, MotionHead, and scene heads are all frozen.
        Sets _occ_only_mode=True so forward() skips QFormer+VLM.
        """
        _set_trainable(self.backbone, False)
        for m in (self.qformer, self.vlm, self.traj_head,
                  self.embedding_head, self.embedding_head_b,
                  self.tagging_head, self.risk_head):
            if m is not None:
                _set_trainable(m, False)
        _set_trainable(self.spatial_encoder, True)
        if self.occ_head is not None:
            _set_trainable(self.occ_head, True)
        self._occ_only_mode = True

    def freeze_for_stage1(self) -> None:
        """Stage 1: train spatial encoder + Q-Former + MotionHead + heads.

        OccupancyHead is frozen — it was pre-trained in stage0.
        MotionHead is trainable: its decoded_tokens feed into Q-Former stream 3,
        so it must train jointly with Q-Former for the fusion to learn.
        """
        self._occ_only_mode = False
        _set_trainable(self.backbone, False)
        if self.vlm is not None:
            _set_trainable(self.vlm, False)
        if self.occ_head is not None:
            _set_trainable(self.occ_head, False)
        _set_trainable(self.spatial_encoder, True)
        for m in (self.qformer, self.traj_head,
                  self.embedding_head, self.embedding_head_b,
                  self.tagging_head, self.risk_head):
            if m is not None:
                _set_trainable(m, True)

    def freeze_for_stage2(self, n_backbone_blocks: int = 4) -> None:
        """Stage 2: + last N backbone transformer blocks (lr ×0.1)."""
        self.freeze_for_stage1()
        _unfreeze_last_n_blocks(self.backbone, n_backbone_blocks)

    def freeze_for_motion_residual_only(self) -> None:
        """v8: train ONLY the parts of the motion head that do NOT affect
        ``decoded_tokens`` (the tensor that feeds Q-Former stream 3).

        Frozen: backbone, spatial_encoder, qformer, vlm, tag/risk/embed/occ
                heads, and inside traj_head: agent_queries, decoder layers,
                agent_pos_mlp, temporal_proj.
        Trainable inside traj_head: velocity_head, traj_proj.

        Combined with v8 identity-init on the new decoder layers and zero-init
        on agent_pos_mlp's last linear, ``decoded_tokens`` are bit-identical
        to a v4-style 2-layer head at iter 0 *and remain so throughout
        training* — every parameter that influences them is frozen.  Tag and
        risk heads see exactly the v4 Q-Former inputs they were trained for.

        Motion can still improve via:
          - velocity_head learning the per-agent (vx, vy) prediction
          - traj_proj learning a residual on top of the kinematic baseline
        """
        self._occ_only_mode = False
        _set_trainable(self.backbone, False)
        _set_trainable(self.spatial_encoder, False)
        if self.vlm is not None:
            _set_trainable(self.vlm, False)
        if self.qformer is not None:
            _set_trainable(self.qformer, False)
        if self.occ_head is not None:
            _set_trainable(self.occ_head, False)
        for m in (self.embedding_head, self.embedding_head_b,
                  self.tagging_head, self.risk_head):
            if m is not None:
                _set_trainable(m, False)
        if self.traj_head is not None:
            # First freeze the whole head, then unfreeze ONLY the residual
            # paths that do not feed Q-Former.
            _set_trainable(self.traj_head, False)
            if getattr(self.traj_head, "velocity_head", None) is not None:
                _set_trainable(self.traj_head.velocity_head, True)
            _set_trainable(self.traj_head.traj_proj, True)

    def freeze_for_motion_only(self) -> None:
        """v7 / Task 14: train ONLY the motion head, keep everything else frozen.

        Goal: refit the motion head's Phase-2 components on top of the
        already-trained v4 representations without disturbing the Q-Former
        and the tag/risk/embed heads.  Because Q-Former gets motion_tokens
        as its third stream, even motion-head changes shift its output —
        but with both Q-Former and tag/risk frozen, the heads continue to
        produce v4-quality outputs from the shifted pooled features.

        Trainable: traj_head (incl. new agent_pos_mlp, velocity_head,
        temporal_proj, decoder layers 3+4).
        Frozen: backbone, spatial_encoder, qformer, vlm,
            embedding_head, tagging_head, risk_head, occ_head.
        """
        self._occ_only_mode = False
        _set_trainable(self.backbone, False)
        _set_trainable(self.spatial_encoder, False)
        if self.vlm is not None:
            _set_trainable(self.vlm, False)
        if self.qformer is not None:
            _set_trainable(self.qformer, False)
        if self.occ_head is not None:
            _set_trainable(self.occ_head, False)
        for m in (self.embedding_head, self.embedding_head_b,
                  self.tagging_head, self.risk_head):
            if m is not None:
                _set_trainable(m, False)
        if self.traj_head is not None:
            _set_trainable(self.traj_head, True)

    def freeze_for_stage3(self) -> None:
        """Stage 3: + VLM LoRA adapters (must be applied before this call)."""
        self.freeze_for_stage2()
        if self.vlm is not None:
            for name, p in self.vlm.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # ── mmengine training interface ───────────────────────────────────────────

    def train_step(self, data: dict, optim_wrapper) -> dict:
        """mmengine Runner training step.

        Unpacks a SyntheticDataset / NuScenesDataset batch, runs forward,
        computes losses, and calls optim_wrapper.update_params.
        """
        from scene_mining.losses import SceneMinerLoss

        # Lazily build and attach loss as a submodule so it lives on the
        # right device and its parameters (e.g. InfoNCE log_temp) are
        # included in model.parameters().
        if not hasattr(self, '_loss_fn'):
            self._loss_fn = SceneMinerLoss()

        device = next(self.parameters()).device

        def _to(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        images      = _to(data['images'])
        intrinsics  = _to(data['intrinsics'])
        extrinsics  = _to(data['extrinsics'])
        input_ids   = _to(data.get('input_ids'))
        attention_mask = _to(data.get('attention_mask'))
        tag_targets = _to(data.get('tags'))
        risk_targets = _to(data.get('risk'))

        # New supervision fields (Task 4 waypoints + Task 9 masks)
        traj_gt       = _to(data.get('waypoints'))       # (B, N_agents, T, 2)
        agent_mask    = _to(data.get('agent_mask'))      # (B, N_agents) bool
        waypoint_mask = _to(data.get('waypoint_mask'))   # (B, N_agents, T) uint8

        # Phase-1 temporal context (None if dataset is single-frame).
        temporal_images     = _to(data.get('temporal_images'))     # (B, T, N_cam, 3, H, W)
        temporal_intrinsics = _to(data.get('temporal_intrinsics'))
        temporal_extrinsics = _to(data.get('temporal_extrinsics'))
        temporal_ego_poses  = _to(data.get('temporal_ego_poses'))  # (B, T, 4, 4)

        # ── Phase-2 agent grounding: cheap proxy for "current position" ────
        # Use the first valid waypoint per agent as the anchor (closest signal
        # we have without adding bbox-extraction code to the dataset).  For
        # invalid agents (agent_mask=False) or fully-padded trajectories we
        # leave the anchor at zero — agent_pos_mlp output is masked to 0 by
        # MotionHead when agent_mask is False.
        agent_positions = None
        if traj_gt is not None and waypoint_mask is not None:
            wp_mask_float = waypoint_mask.to(traj_gt.dtype)            # (B, N, T)
            first_valid = wp_mask_float.argmax(dim=-1)                  # (B, N)
            any_valid = wp_mask_float.sum(dim=-1) > 0                   # (B, N) bool
            # Gather waypoint at first_valid index per agent.
            idx = first_valid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            anchor = torch.gather(traj_gt, dim=2, index=idx).squeeze(2) # (B, N, 2)
            agent_positions = anchor * any_valid.to(traj_gt.dtype).unsqueeze(-1)

        # image_hw collation:
        #   - (B,2) tensor  → raw_hw[0] is a (B,) tensor of heights
        #   - tuple of two (B,) tensors → raw_hw[0][0] is the scalar
        #   - plain (H,W) tuple of ints → raw_hw[0] is already a scalar
        raw_hw = data.get('image_hw')
        if isinstance(raw_hw, torch.Tensor):
            # shape (B, 2)
            image_hw = (int(raw_hw[0, 0]), int(raw_hw[0, 1]))
        elif isinstance(raw_hw, (list, tuple)) and len(raw_hw) == 2:
            h, w = raw_hw[0], raw_hw[1]
            image_hw = (
                int(h[0]) if isinstance(h, torch.Tensor) else int(h),
                int(w[0]) if isinstance(w, torch.Tensor) else int(w),
            )
        else:
            image_hw = None

        self._loss_fn = self._loss_fn.to(device)

        occ_gt = _to(data.get('occ_gt'))

        with optim_wrapper.optim_context(self):
            out = self(images, intrinsics, extrinsics,
                       input_ids=input_ids, attention_mask=attention_mask,
                       image_hw=image_hw, agent_mask=agent_mask,
                       temporal_images=temporal_images,
                       temporal_intrinsics=temporal_intrinsics,
                       temporal_extrinsics=temporal_extrinsics,
                       temporal_ego_poses=temporal_ego_poses,
                       agent_positions=agent_positions)
            loss_dict = self._loss_fn(
                embeddings_a=None if self._occ_only_mode else out.embeddings,
                embeddings_b=None if self._occ_only_mode else out.embeddings_b,
                tag_logits=None if self._occ_only_mode else out.tag_logits,
                tag_targets=None if self._occ_only_mode else tag_targets,
                risk_preds=None if self._occ_only_mode else out.risk_scores.squeeze(-1),
                risk_targets=None if self._occ_only_mode else risk_targets,
                occ_logits=out.occ_logits,
                occ_gt=occ_gt,
                traj_pred=None if self._occ_only_mode else out.traj_pred,
                traj_gt=None if self._occ_only_mode else traj_gt,
                agent_mask=None if self._occ_only_mode else agent_mask,
                waypoint_mask=None if self._occ_only_mode else waypoint_mask,
            )

        # ── NaN guard: skip update if loss is non-finite ──
        # bf16 has fp32 range so there's no loss scaler to detect overflow.
        # We manually check and skip to prevent NaN from corrupting weights.
        total_loss = loss_dict['total']
        if not torch.isfinite(total_loss):
            # Zero out the loss so .backward() doesn't touch anything meaningful,
            # and skip the optimizer step. Returning the NaN value preserves
            # visibility in the logs.
            optim_wrapper.optimizer.zero_grad(set_to_none=True)
            return {k: float('nan') if torch.isnan(v) or torch.isinf(v) else v.item()
                    for k, v in loss_dict.items()}

        optim_wrapper.update_params(total_loss)
        return {k: v.item() for k, v in loss_dict.items()}


if _HAS_MMENGINE:
    SCENE_MINING_MODELS.register_module()(SceneMiner)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve(component: dict | nn.Module) -> nn.Module:
    """Build from config dict or return module as-is."""
    if isinstance(component, dict):
        if _HAS_MMENGINE:
            return build_component(component)
        raise TypeError(
            f"Received config dict but mmengine is not installed. "
            f"Install mmengine or pass module instances directly."
        )
    return component


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(trainable)


def _unfreeze_last_n_blocks(backbone: nn.Module, n: int) -> None:
    """Try to unfreeze the last N transformer blocks of a HF ViT backbone."""
    # HF ViT/SigLIP2/DINOv2 all expose .encoder.encoder.layers or .encoder.layers
    for attr in ("encoder", "model"):
        inner = getattr(backbone, attr, None)
        if inner is None:
            continue
        for attr2 in ("encoder", ""):
            layers_container = getattr(inner, attr2, inner) if attr2 else inner
            layers = getattr(layers_container, "layers", None)
            if layers is not None and len(layers) > 0:
                for layer in layers[-n:]:
                    _set_trainable(layer, True)
                return
