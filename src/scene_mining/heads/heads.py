"""Output heads for the SceneMiner deep learning pipeline.

Six heads are available:
- EmbeddingHead: 256-d L2-normalised scene embedding (InfoNCE)
- TaggingHead: Multi-label scene tag logits (BCELoss)
- RiskHead: Scalar risk/anomaly score in [0,1] (HuberLoss)
- CaptionHead: Text captions (delegated to VLM.generate)
- OccupancyHead: BEV binary occupancy prediction from spatial tokens (BCELoss)
- MotionHead: Transformer decoder for multi-agent trajectory prediction
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    """Project pooled VLM hidden state to a 256-d L2-normalised embedding.

    Architecture: Linear → GELU → LayerNorm → Linear → L2-normalise
    """

    def __init__(self, d_vlm: int = 3584, d_embed: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_vlm, d_vlm // 4),
            nn.GELU(),
            nn.LayerNorm(d_vlm // 4),
            nn.Linear(d_vlm // 4, d_embed),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Args: pooled (B, d_vlm) → Returns: (B, d_embed) normalised."""
        return F.normalize(self.net(pooled), dim=-1)


class TaggingHead(nn.Module):
    """Multi-label scene tag classifier.

    Predicts presence/absence of up to ``n_tags`` scene attributes
    (weather conditions, time of day, event types, density, etc.).
    """

    def __init__(self, d_vlm: int = 3584, n_tags: int = 20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_vlm, d_vlm // 4),
            nn.GELU(),
            nn.Linear(d_vlm // 4, n_tags),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Args: pooled (B, d_vlm) → Returns: (B, n_tags) logits (pre-sigmoid)."""
        return self.net(pooled)


class RiskHead(nn.Module):
    """Scalar risk/anomaly score predictor.

    Output is passed through Sigmoid to produce a score in [0, 1].
    """

    def __init__(self, d_vlm: int = 3584) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_vlm, d_vlm // 8),
            nn.GELU(),
            nn.Linear(d_vlm // 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Args: pooled (B, d_vlm) → Returns: (B, 1) risk score in [0, 1]."""
        return self.net(pooled)


class OccupancyHead(nn.Module):
    """BEV occupancy prediction from flattened spatial tokens.

    For n_classes=1 (binary): outputs (B, bev_h, bev_w) logits for BCE.
    For n_classes>1 (semantic): outputs (B, n_classes, bev_h, bev_w) logits for cross-entropy.

    Architecture: Linear → GELU → Linear(n_classes) → reshape
    """

    def __init__(
        self,
        d_bev: int = 256,
        bev_h: int = 50,
        bev_w: int = 50,
        n_classes: int = 1,
    ) -> None:
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(d_bev, d_bev),
            nn.GELU(),
            nn.Linear(d_bev, n_classes),
        )

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """Args: spatial_tokens (B, bev_h*bev_w, d_bev).

        Returns:
            n_classes=1: (B, bev_h, bev_w) binary logits.
            n_classes>1: (B, n_classes, bev_h, bev_w) semantic logits.
        """
        B = spatial_tokens.shape[0]
        logits = self.net(spatial_tokens)                          # (B, H*W, n_classes)
        logits = logits.view(B, self.bev_h, self.bev_w, self.n_classes)
        logits = logits.permute(0, 3, 1, 2)                       # (B, n_classes, H, W)
        if self.n_classes == 1:
            return logits.squeeze(1)                               # (B, H, W)
        return logits                                              # (B, n_classes, H, W)


class BEVSemanticDecoder(nn.Module):
    """CNN decoder for semantic BEV segmentation from BEVFormer spatial tokens.

    Reshapes flat BEV tokens to a 2D feature map and applies a series of
    Conv2d layers to produce per-cell class logits.  Unlike the simple
    per-token MLP in OccupancyHead, the convolutions capture spatial
    context between neighbouring BEV cells, matching the inductive bias of
    TPVAggregator (which also operates in a spatially-aware manner).

    Architecture: reshape → Conv-BN-ReLU × 2 → bottleneck → class logits

    Args:
        d_bev: BEVFormer token dimension (embed_dims).
        bev_h: BEV grid height.
        bev_w: BEV grid width.
        n_classes: Number of semantic classes (including background 0).
        hidden_dim: Internal channel width.  Defaults to d_bev.
    """

    def __init__(
        self,
        d_bev: int = 256,
        bev_h: int = 50,
        bev_w: int = 50,
        n_classes: int = 17,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.n_classes = n_classes
        h = hidden_dim or d_bev
        self.decoder = nn.Sequential(
            nn.Conv2d(d_bev, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, h // 2, 1, bias=False),
            nn.BatchNorm2d(h // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(h // 2, n_classes, 1),
        )

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """Args: spatial_tokens (B, bev_h*bev_w, d_bev).

        Returns: (B, n_classes, bev_h, bev_w) semantic logits.
        """
        B, _, d = spatial_tokens.shape
        x = spatial_tokens.view(B, self.bev_h, self.bev_w, d)
        x = x.permute(0, 3, 1, 2)           # (B, d_bev, H, W)
        return self.decoder(x)               # (B, n_classes, H, W)


class TPVSemanticHead(nn.Module):
    """Semantic head for TPVFormer 3-plane spatial tokens.

    TPVFormer produces tokens for three planes:
        HW (top-down BEV):  tpv_h × tpv_w
        HZ (frontal):       tpv_h × tpv_z
        WZ (lateral):       tpv_w × tpv_z

    This head mirrors the TPVAggregator from the official TPVFormer repo:
    - Splits the concatenated token sequence into the three planes.
    - Pools height context from the HZ plane (max over Z → broadcast over W).
    - Pools width  context from the WZ plane (max over Z → broadcast over H).
    - Concatenates all three feature maps and decodes with a CNN.

    Output: (B, n_classes, tpv_h, tpv_w) top-down semantic logits.

    Args:
        tpv_h: TPV grid height (rows, corresponds to X/forward axis).
        tpv_w: TPV grid width  (columns, corresponds to Y/lateral axis).
        tpv_z: TPV grid depth  (height bins, Z axis).
        embed_dims: Token feature dimension.
        n_classes: Number of semantic classes.
        hidden_dim: Internal CNN channel width.  Defaults to embed_dims * 2.
    """

    def __init__(
        self,
        tpv_h: int = 150,
        tpv_w: int = 150,
        tpv_z: int = 16,
        embed_dims: int = 96,
        n_classes: int = 17,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.n_hw = tpv_h * tpv_w
        self.n_hz = tpv_h * tpv_z
        self.n_wz = tpv_w * tpv_z
        self.n_classes = n_classes
        h = hidden_dim or embed_dims * 2
        # 3 planes contribute embed_dims each → 3 × embed_dims input channels
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dims * 3, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, h // 2, 1, bias=False),
            nn.BatchNorm2d(h // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(h // 2, n_classes, 1),
        )

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """Args: spatial_tokens (B, n_hw + n_hz + n_wz, embed_dims).

        Returns: (B, n_classes, tpv_h, tpv_w) semantic logits.
        """
        B, _, d = spatial_tokens.shape
        hw = spatial_tokens[:, :self.n_hw]                         # (B, H*W, d)
        hz = spatial_tokens[:, self.n_hw:self.n_hw + self.n_hz]   # (B, H*Z, d)
        wz = spatial_tokens[:, self.n_hw + self.n_hz:]            # (B, W*Z, d)

        # HW plane → 2D feature map (B, d, H, W)
        hw_feat = hw.view(B, self.tpv_h, self.tpv_w, d).permute(0, 3, 1, 2)

        # HZ plane → max-pool over Z → per-row feature → (B, d, H, W)
        hz_feat = hz.view(B, self.tpv_h, self.tpv_z, d)
        hz_pooled = hz_feat.max(dim=2).values                      # (B, H, d)
        hz_pooled = hz_pooled.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, self.tpv_w)

        # WZ plane → max-pool over Z → per-col feature → (B, d, H, W)
        wz_feat = wz.view(B, self.tpv_w, self.tpv_z, d)
        wz_pooled = wz_feat.max(dim=2).values                      # (B, W, d)
        wz_pooled = wz_pooled.permute(0, 2, 1).unsqueeze(-2).expand(-1, -1, self.tpv_h, -1)

        x = torch.cat([hw_feat, hz_pooled, wz_pooled], dim=1)     # (B, 3d, H, W)
        return self.decoder(x)                                     # (B, n_classes, H, W)


class MotionHead(nn.Module):
    """Transformer decoder for multi-agent trajectory prediction.

    Uses learned agent query tokens that cross-attend to BEV spatial features
    and self-attend to each other, capturing agent–scene and agent–agent
    interactions for trajectory forecasting.

    Args:
        d_model: Spatial feature dimension from BEVFormer (default 256).
        n_agents: Max agents per scene (default 8).
        n_waypoints: Number of future waypoints (default 6, at 0.5s = 3s horizon).
        n_decoder_layers: Number of transformer decoder layers (default 2).
        nhead: Number of attention heads (default 8).
        dim_feedforward: FFN hidden dimension (default 512).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_agents: int = 8,
        n_waypoints: int = 6,
        n_decoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        d_temporal_in: int | None = None,
        # ── Phase 2 ────────────────────────────────────────────────────────
        agent_position_dim: int | None = None,    # if 2, enable grounding
        use_velocity_subhead: bool = False,        # if True, kinematic prior
        waypoint_dt: float = 0.5,                  # seconds between waypoints
        # ── v8: warm-start identity preservation ───────────────────────────
        # When set, decoder layers with index ≥ identity_init_from_layer are
        # zero-initialised on their residual projections (self_attn.out_proj,
        # multihead_attn.out_proj, linear2).  Combined with zero-init on
        # agent_pos_mlp, this makes ``decoded_tokens`` bit-identical to a
        # v4-style 2-layer head at iter 0, so frozen tag/risk heads see the
        # same Q-Former inputs they were trained for.
        identity_init_from_layer: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_agents = n_agents
        self.n_waypoints = n_waypoints
        self.use_velocity_subhead = use_velocity_subhead
        self.waypoint_dt = waypoint_dt

        self.agent_queries = nn.Embedding(n_agents, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # v8 identity-init: zero the residual contributions of new decoder
        # layers so they pass the layer-N input through unchanged.  Affects
        # the three residual paths inside each TransformerDecoderLayer:
        #   1. self_attn.out_proj    (self-attention contribution)
        #   2. multihead_attn.out_proj (cross-attention contribution)
        #   3. linear2 (feedforward last linear)
        if identity_init_from_layer is not None:
            for i in range(identity_init_from_layer, n_decoder_layers):
                layer = self.decoder.layers[i]
                with torch.no_grad():
                    layer.self_attn.out_proj.weight.zero_()
                    layer.self_attn.out_proj.bias.zero_()
                    layer.multihead_attn.out_proj.weight.zero_()
                    layer.multihead_attn.out_proj.bias.zero_()
                    layer.linear2.weight.zero_()
                    layer.linear2.bias.zero_()

        self.traj_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_waypoints * 2),
        )

        # Small-init the final projection so initial predictions are near the
        # origin (not random at ±50 m). Prevents huge first-epoch motion loss.
        nn.init.normal_(self.traj_proj[-1].weight, std=0.01)
        nn.init.zeros_(self.traj_proj[-1].bias)

        # Optional projection for temporally-stacked BEV memory.
        self.temporal_proj: nn.Linear | None = None
        if d_temporal_in is not None and d_temporal_in != d_model:
            self.temporal_proj = nn.Linear(d_temporal_in, d_model)
            with torch.no_grad():
                self.temporal_proj.weight.zero_()
                T = d_temporal_in // d_model
                eye = torch.eye(d_model)
                self.temporal_proj.weight[:, (T - 1) * d_model : T * d_model] = eye
                self.temporal_proj.bias.zero_()

        # ── Phase-2 agent grounding ────────────────────────────────────────
        # Project the per-agent (x, y) position into the query embedding space
        # and ADD to the learned agent queries.  This tells the decoder *where*
        # each agent currently is, so cross-attention can focus on the right
        # BEV region instead of inferring identity from the learned-only query.
        self.agent_pos_mlp: nn.Sequential | None = None
        if agent_position_dim is not None and agent_position_dim > 0:
            self.agent_pos_mlp = nn.Sequential(
                nn.Linear(agent_position_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            # v8 zero-init: final-layer weights + bias = 0 so the position
            # contribution to queries is EXACTLY zero at step 0.  Combined
            # with zero-init on the new decoder layers (below), this makes
            # ``decoded_tokens`` bit-identical to a v4-style head at iter 0
            # — Q-Former stream 3 stays at v4 distribution, so frozen tag
            # and risk heads keep producing v4-quality outputs.
            nn.init.zeros_(self.agent_pos_mlp[-1].weight)
            nn.init.zeros_(self.agent_pos_mlp[-1].bias)

        # ── Phase-2 velocity sub-head (kinematic prior) ───────────────────
        # Predicts a per-agent (vx, vy) in m/s from the decoded query.
        # Trajectory becomes:
        #     waypoint_t = agent_position + (t+1)*dt*velocity + residual_t
        # where residual_t comes from the existing traj_proj.  This gives the
        # head a "predict zero motion at the agent's current position" baseline
        # to work from, instead of having to discover that from scratch.
        self.velocity_head: nn.Sequential | None = None
        if use_velocity_subhead:
            self.velocity_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 2),
            )
            # Tiny init → initial velocity ≈ 0, so initial trajectory ≈ agent
            # position for every time step.  Better starting point than zero.
            nn.init.normal_(self.velocity_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.velocity_head[-1].bias)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        temporal_spatial_tokens: torch.Tensor | None = None,
        agent_positions: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            spatial_tokens: (B, H*W, d_model) BEV spatial features (current frame).
            agent_mask: (B, N_agents) bool mask, True = valid agent, False = padding.
            temporal_spatial_tokens: optional (B, H*W, T*d_model) ego-aligned
                temporal stack.  When supplied, replaces ``spatial_tokens`` as
                the attention memory after projection through ``temporal_proj``.
            agent_positions: optional (B, N_agents, 2) per-agent (x, y) anchor
                in current-ego metres.  When supplied and ``agent_pos_mlp`` is
                configured, added to the learned queries (Phase-2 grounding).
                Also used as the base position for the velocity sub-head.
        Returns:
            dict with 'traj_pred' (B, N_agents, n_waypoints, 2).
        """
        if temporal_spatial_tokens is not None and self.temporal_proj is not None:
            memory = self.temporal_proj(temporal_spatial_tokens)   # (B, H*W, d_model)
        else:
            memory = spatial_tokens

        B = memory.shape[0]
        queries = self.agent_queries.weight.unsqueeze(0).expand(B, -1, -1)

        # Phase-2 agent grounding: add per-agent position embedding to queries.
        if agent_positions is not None and self.agent_pos_mlp is not None:
            pos_emb = self.agent_pos_mlp(agent_positions.to(queries.dtype))
            if agent_mask is not None:
                pos_emb = pos_emb * agent_mask.to(queries.dtype).unsqueeze(-1)
            queries = queries + pos_emb

        # tgt_key_padding_mask expects True = ignore, so invert agent_mask
        tgt_key_padding_mask = None
        if agent_mask is not None:
            tgt_key_padding_mask = ~agent_mask  # (B, N_agents)

        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, N_agents, d_model)

        # ── Trajectory prediction ─────────────────────────────────────────
        if self.use_velocity_subhead and self.velocity_head is not None:
            # Predict per-agent velocity (m/s) and integrate to a base
            # trajectory; transformer outputs are treated as residuals.
            velocity = self.velocity_head(decoded)              # (B, N, 2)
            t_steps = torch.arange(
                1, self.n_waypoints + 1, device=decoded.device, dtype=decoded.dtype,
            ) * self.waypoint_dt                                 # (T,)
            base_traj = velocity.unsqueeze(2) * t_steps.view(1, 1, -1, 1)  # (B,N,T,2)
            if agent_positions is not None:
                base_traj = base_traj + agent_positions.to(decoded.dtype).unsqueeze(2)
            residual_flat = self.traj_proj(decoded)              # (B, N, T*2)
            residual = residual_flat.view(B, self.n_agents, self.n_waypoints, 2)
            traj_pred = base_traj + residual
        else:
            traj_flat = self.traj_proj(decoded)
            traj_pred = traj_flat.view(B, self.n_agents, self.n_waypoints, 2)

        return {
            "traj_pred": traj_pred,
            "decoded_tokens": decoded,  # (B, N_agents, d_model) — for Q-Former fusion
        }
