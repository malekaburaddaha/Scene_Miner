#!/usr/bin/env python3
"""Unified Learned Filter — query-conditioned edge-case ranking.

Replaces SceneMiner's rule-composed query path (tag-synonym lookup +
description substring mask + centroid ranking) with ONE learned scorer:

    score(query, scene) = f( CLIP_text(query);  e_scene, t_scene, r_scene )

Inputs per scene (all emitted by the frozen SceneMiner forward pass and
precomputed offline — the filter never touches images):
    e  : 256-d InfoNCE retrieval embedding      (scene-instance geometry)
    t  : 20-d tag logits                        (calibrated semantics)
    r  : scalar risk score in [0,1]             (physics difficulty prior)
Query:
    q  : 512-d frozen CLIP text embedding of an arbitrary NL query.

Design principles (fusion + representation learning, not perception):
  * LATE FUSION over heterogeneous experts: each signal is projected into a
    shared d-dim token space by its own adapter (a 1-d risk scalar and a
    20-d logit vector cannot share a projection with a 256-d embedding).
  * QUERY-CONDITIONED CROSS-ATTENTION: the query token attends over the
    three scene tokens, learning per-query modality weighting — a rain
    query should weight tags; "near-miss" should weight risk; "scenes like
    this one" should weight the embedding. This replaces the hand-written
    mask/centroid rules with learned gating.
  * FiLM MODULATION of scene tokens by the query (feature-wise affine)
    before attention, so modality features are re-scaled in a
    query-dependent way (Perez et al., 2018).
  * BILINEAR + MLP SCORING head: a low-rank bilinear term captures direct
    query–scene alignment; the MLP term captures non-linear interactions;
    a learned temperature calibrates the ranking distribution.
  * The whole filter is ~1–2 M parameters: it trains in minutes/hours, is
    dataset-agnostic (consumes only SceneMiner outputs + CLIP text), and
    leaves every upstream component frozen — consistent with the paper's
    identity-preserving philosophy.

Score is used for ranking; training uses a listwise InfoNCE over scenes
per query plus a margin term against hard negatives (see
scripts/train_unified_filter.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarStandardize(nn.Module):
    """(x - mu) / sigma with FIXED dataset statistics.

    Replaces LayerNorm for width-1 channels. LayerNorm normalises over the
    last dimension, so for d_in == 1 it computes (x - x) / sqrt(0 + eps) = 0
    for EVERY input: the channel's variance is annihilated and the adapter
    emits a constant bias vector. (Diagnosed empirically: the risk token was
    bit-identical across all 1,508 scenes despite 1,035 distinct inputs.)
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("sigma", torch.tensor(max(float(sigma), 1e-6)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sigma


class ModalityAdapter(nn.Module):
    """Project one heterogeneous signal into the shared token space.

    Scalar channels (d_in == 1) are standardised with fixed dataset
    statistics; only multi-dimensional channels use LayerNorm.
    """

    def __init__(self, d_in: int, d_model: int,
                 scalar_stats: tuple[float, float] | None = None,
                 force_layernorm1d: bool = False) -> None:
        """force_layernorm1d restores the ORIGINAL pre-repair defect: a real
        nn.LayerNorm over a width-1 axis, which returns (x-x)/sqrt(0+eps) == 0
        for every input and deletes the channel. The `scalar_stats=None` path
        does NOT reproduce this - it still builds a ScalarStandardize, only with
        (mu,sigma)=(0,1) - so Row A was never a faithful pre-repair baseline.
        This flag is the only way to regenerate Table 1's 'before' column."""
        super().__init__()
        if d_in == 1 and not force_layernorm1d:
            mu, sigma = scalar_stats if scalar_stats is not None else (0.0, 1.0)
            norm: nn.Module = ScalarStandardize(mu, sigma)
        else:
            norm = nn.LayerNorm(d_in)
        self.net = nn.Sequential(
            norm,
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, d_in) → (B, d)
        return self.net(x)


class FiLM(nn.Module):
    """Feature-wise linear modulation of scene tokens by the query.

    gamma/beta are zero-initialised so modulation starts as identity —
    the same safe-start principle as the paper's identity-preserving
    protocol (training begins from unmodulated tokens).
    """

    def __init__(self, d_query: int, d_model: int, n_tokens: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(d_query, 2 * n_tokens * d_model)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.n_tokens, self.d_model = n_tokens, d_model

    def forward(self, tokens: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # tokens (B, T, d), q (B, d_q)
        gb = self.to_gamma_beta(q).view(-1, 2, self.n_tokens, self.d_model)
        gamma, beta = gb[:, 0], gb[:, 1]
        return tokens * (1.0 + gamma) + beta


class UnifiedFilter(nn.Module):
    """score(query, scene) → real-valued edge-case ranking score."""

    N_TOKENS = 3          # embedding / tags / risk

    def __init__(
        self,
        d_clip: int = 512,          # frozen CLIP text width (ViT-B/32)
        d_embed: int = 256,         # SceneMiner InfoNCE embedding
        n_tags: int = 20,
        d_model: int = 256,
        n_heads: int = 4,
        n_fusion_layers: int = 2,
        bilinear_rank: int = 64,
        dropout: float = 0.1,
        init_tau: float = 1.0,
    ) -> None:
        super().__init__()
        # per-modality adapters (heterogeneous → shared space)
        self.adapt_embed = ModalityAdapter(d_embed, d_model)
        self.adapt_tags = ModalityAdapter(n_tags, d_model)
        self.adapt_risk = ModalityAdapter(1, d_model)
        self.adapt_query = ModalityAdapter(d_clip, d_model)
        # learned modality-type embeddings (which token is which)
        self.type_embed = nn.Parameter(torch.randn(self.N_TOKENS, d_model) * 0.02)

        self.film = FiLM(d_model, d_model, self.N_TOKENS)

        # query-conditioned fusion: query token cross-attends scene tokens,
        # then a shallow self-attention block mixes everything.
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerDecoder(layer, num_layers=n_fusion_layers)

        # low-rank bilinear alignment  s_bil = (U q)·(V c)
        self.bil_q = nn.Linear(d_model, bilinear_rank, bias=False)
        self.bil_c = nn.Linear(d_model, bilinear_rank, bias=False)
        # non-linear interaction head on [q; c; q*c; |q-c|]
        self.mlp = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # learned score temperature (calibrates the ranking softmax)
        self.log_tau = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------
    def scene_tokens(self, e: torch.Tensor, t: torch.Tensor,
                     r: torch.Tensor) -> torch.Tensor:
        """(B,256),(B,20),(B,1) → (B, 3, d) modality tokens."""
        toks = torch.stack([
            self.adapt_embed(e), self.adapt_tags(t), self.adapt_risk(r)],
            dim=1)
        return toks + self.type_embed.unsqueeze(0)

    def forward(self, q_clip: torch.Tensor, e: torch.Tensor,
                t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """q_clip (B,512); e (B,256); t (B,20); r (B,1) → score (B,)."""
        q = self.adapt_query(q_clip)                       # (B, d)
        toks = self.scene_tokens(e, t, r)                  # (B, 3, d)
        toks = self.film(toks, q)                          # query-modulated
        fused = self.fusion(tgt=q.unsqueeze(1), memory=toks)[:, 0]  # (B, d)

        s_bil = (self.bil_q(q) * self.bil_c(fused)).sum(-1)
        s_mlp = self.mlp(torch.cat(
            [q, fused, q * fused, (q - fused).abs()], dim=-1)).squeeze(-1)
        return (s_bil + s_mlp) / self.log_tau.exp().clamp(min=1e-2)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rank(self, q_clip_1: torch.Tensor, e: torch.Tensor,
             t: torch.Tensor, r: torch.Tensor, topk: int = 50):
        """One query vs a scene bank: (512,),(N,256),(N,20),(N,1) → topk idx."""
        q = q_clip_1.unsqueeze(0).expand(e.shape[0], -1)
        scores = self.forward(q, e, t, r)
        return torch.topk(scores, min(topk, e.shape[0]))


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":          # smoke test: shapes + identity start
    torch.manual_seed(0)
    f = UnifiedFilter()
    B = 6
    s = f(torch.randn(B, 512), torch.randn(B, 256),
          torch.randn(B, 20), torch.rand(B, 1))
    assert s.shape == (B,), s.shape
    print(f"UnifiedFilter OK — params: {count_parameters(f):,}, "
          f"scores: {s.detach().numpy().round(3)}")
