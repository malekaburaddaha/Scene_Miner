#!/usr/bin/env python3
"""SceneMiner v2 - identity-preserving growth over a strong zero-shot base.

v1 lost to two signals it could not see (human-GT mAP, 600 labelled scenes):
    logit VLM lookup   0.733
    zero-shot SigLIP2  0.569  (0.913 droplets, 0.843 wet_road)
    SceneMiner v1      0.420

v2 therefore does not try to beat them from scratch. It *starts* as them and
learns a residual, using the Sec. 3.4 zero-initialised-branch construction:

    s(q,x) = s_base(q,x) + tanh(gate) * s_res(q,x),     gate initialised to 0

so at step 0 the ranking is EXACTLY the base ranking, and every departure from
it is something training paid for. s_base is a convex blend of the SigLIP2
image-text cosine and the cached VLM logit margin; s_res is the multimodal
fusion head over six cached tokens (BEV embedding, tags, risk, Qwen visual,
SigLIP2 image, VLM margins).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from scene_mining.filter.unified_filter import FiLM, ModalityAdapter


class SceneMinerV2(nn.Module):
    N_TOKENS = 6   # embed / tags / risk / qwen-visual / siglip-image / vlm-margins

    def __init__(self, d_embed=256, n_tags=20, d_qwen=2048, d_siglip=768,
                 n_margins=4, d_text=768, d_model=256, n_heads=4,
                 n_fusion_layers=2, bilinear_rank=64, dropout=0.1,
                 tau=0.30, risk_stats=(0.417797, 0.081112),
                 base_blend_init=0.5):
        super().__init__()
        self.adapt_embed  = ModalityAdapter(d_embed, d_model)
        self.adapt_tags   = ModalityAdapter(n_tags, d_model)
        self.adapt_risk   = ModalityAdapter(1, d_model, scalar_stats=risk_stats)
        self.adapt_qwen   = ModalityAdapter(d_qwen, d_model)
        self.adapt_siglip = ModalityAdapter(d_siglip, d_model)
        self.adapt_margin = ModalityAdapter(n_margins, d_model)
        self.adapt_query  = ModalityAdapter(d_text, d_model)
        self.type_embed = nn.Parameter(torch.randn(self.N_TOKENS, d_model) * 0.02)
        self.film = FiLM(d_model, d_model, self.N_TOKENS)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=4*d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.fusion = nn.TransformerDecoder(layer, num_layers=n_fusion_layers)
        self.bil_q = nn.Linear(d_model, bilinear_rank, bias=False)
        self.bil_c = nn.Linear(d_model, bilinear_rank, bias=False)
        self.mlp = nn.Sequential(nn.LayerNorm(4*d_model),
                                 nn.Linear(4*d_model, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.register_buffer("tau", torch.tensor(float(tau)))
        # --- identity-preserving growth ---
        # gate starts at exactly 0 so the residual contributes nothing at step 0
        self.gate = nn.Parameter(torch.zeros(()))
        # learned blend between the two base signals; starts balanced
        self.base_blend = nn.Parameter(torch.tensor(float(base_blend_init)))
        # per-query selector over the 4 margin channels, learned from the query
        self.margin_select = nn.Linear(d_text, n_margins)
        nn.init.zeros_(self.margin_select.weight); nn.init.zeros_(self.margin_select.bias)

    # ------------------------------------------------------------------
    def base_score(self, q_siglip, siglip_img, margins_z):
        """Zero-shot base: SigLIP2 cosine blended with a query-selected margin.

        At init margin_select is zero -> the margin term is a uniform mean, and
        base_blend=0.5, so the base is a fixed, sensible combination of the two
        strongest zero-shot signals from the human-GT study.
        """
        cos = F.cosine_similarity(F.normalize(q_siglip, dim=-1),
                                  F.normalize(siglip_img, dim=-1), dim=-1)
        w = torch.softmax(self.margin_select(q_siglip), dim=-1)      # (B, n_margins)
        marg = (w * margins_z).sum(-1)
        b = torch.sigmoid(self.base_blend)
        return b * cos + (1.0 - b) * torch.tanh(marg)

    def residual_score(self, q_text, e, t, r, v, s_img, marg_z):
        q = self.adapt_query(q_text)
        toks = torch.stack([self.adapt_embed(e), self.adapt_tags(t),
                            self.adapt_risk(r), self.adapt_qwen(v),
                            self.adapt_siglip(s_img), self.adapt_margin(marg_z)],
                           dim=1) + self.type_embed.unsqueeze(0)
        toks = self.film(toks, q)
        c = self.fusion(tgt=q.unsqueeze(1), memory=toks)[:, 0]
        qn, cn = F.normalize(q, dim=-1), F.normalize(c, dim=-1)
        s_bil = F.cosine_similarity(self.bil_q(qn), self.bil_c(cn), dim=-1)
        s_mlp = torch.tanh(self.mlp(torch.cat([qn, cn, qn*cn, (qn-cn).abs()], -1)).squeeze(-1))
        return (s_bil + s_mlp) / self.tau, c

    def forward(self, q_text, q_siglip, e, t, r, v, s_img, marg_z,
                return_parts=False):
        base = self.base_score(q_siglip, s_img, marg_z)
        res, fused = self.residual_score(q_text, e, t, r, v, s_img, marg_z)
        s = base + torch.tanh(self.gate) * res
        if return_parts:
            return s, base, res, fused
        return s, fused

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    torch.manual_seed(0)
    m = SceneMinerV2()
    B = 8
    q_t = torch.randn(B, 768); q_s = torch.randn(B, 768)
    e, t, r = torch.randn(B, 256), torch.randn(B, 20), torch.rand(B, 1)
    v, si, mz = torch.randn(B, 2048), torch.randn(B, 768), torch.randn(B, 4)
    s, base, res, _ = m(q_t, q_s, e, t, r, v, si, mz, return_parts=True)
    assert torch.allclose(s, base, atol=1e-6), "gate must be identity at init"
    n = sum(p.numel() for p in m.parameters())
    print(f"SceneMinerV2 OK - {n:,} params; score == base at init "
          f"(max|s-base| = {float((s-base).abs().max()):.2e})")


class SceneMinerV21(nn.Module):
    """SceneMiner v2.1 - query-conditioned per-modality routing.

    v2 used a single global scalar blend (~0.60) between the SigLIP2 cosine and
    the VLM margins, applied to every query alike. That is wrong whenever the
    two signals disagree about which query they are good at, and they do:

        attribute      zero-shot SigLIP2   logit VLM lookup
        droplets             0.913              0.824
        low_contrast         0.105              0.552

    Forcing 60 % weight on SigLIP2 dragged the v2 base to 0.3585 on
    low_contrast. v2.1 replaces the scalar with a routing distribution computed
    FROM THE QUERY,

        w(q) = softmax(W_route q_hat / tau_r)  over three base signals
               (1) SigLIP2 image-text cosine
               (2) standardised continuous VLM margins
               (3) deterministic tag / metadata prior

    and makes the residual gate query-conditioned too. W_route and the gate head
    are zero-initialised, so at step 0 routing is exactly uniform over the three
    base signals and the residual contributes nothing - a clean, defined start.
    """

    N_TOKENS = 6
    N_BASE = 3

    def __init__(self, d_embed=256, n_tags=20, d_qwen=2048, d_siglip=768,
                 n_margins=4, n_meta=6, d_text=768, d_model=256, n_heads=4,
                 n_fusion_layers=2, bilinear_rank=64, dropout=0.1,
                 tau=0.30, tau_route=1.0, risk_stats=(0.417797, 0.081112)):
        super().__init__()
        self.adapt_embed  = ModalityAdapter(d_embed, d_model)
        self.adapt_tags   = ModalityAdapter(n_tags, d_model)
        self.adapt_risk   = ModalityAdapter(1, d_model, scalar_stats=risk_stats)
        self.adapt_qwen   = ModalityAdapter(d_qwen, d_model)
        self.adapt_siglip = ModalityAdapter(d_siglip, d_model)
        self.adapt_margin = ModalityAdapter(n_margins, d_model)
        self.adapt_query  = ModalityAdapter(d_text, d_model)
        self.type_embed = nn.Parameter(torch.randn(self.N_TOKENS, d_model) * 0.02)
        self.film = FiLM(d_model, d_model, self.N_TOKENS)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=4*d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.fusion = nn.TransformerDecoder(layer, num_layers=n_fusion_layers)
        self.bil_q = nn.Linear(d_model, bilinear_rank, bias=False)
        self.bil_c = nn.Linear(d_model, bilinear_rank, bias=False)
        self.mlp = nn.Sequential(nn.LayerNorm(4*d_model),
                                 nn.Linear(4*d_model, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.register_buffer("tau_route", torch.tensor(float(tau_route)))

        # --- query-conditioned routing over the three base signals ---
        self.W_route = nn.Linear(d_text, self.N_BASE)
        nn.init.zeros_(self.W_route.weight); nn.init.zeros_(self.W_route.bias)
        # which margin channel a query cares about
        self.margin_select = nn.Linear(d_text, n_margins)
        nn.init.zeros_(self.margin_select.weight); nn.init.zeros_(self.margin_select.bias)
        # which tag / metadata channel a query cares about
        self.meta_select = nn.Linear(d_text, n_tags + n_meta)
        nn.init.zeros_(self.meta_select.weight); nn.init.zeros_(self.meta_select.bias)
        # query-conditioned residual gate, zero-initialised
        self.gate_head = nn.Linear(d_text, 1)
        nn.init.zeros_(self.gate_head.weight); nn.init.zeros_(self.gate_head.bias)

    # ------------------------------------------------------------------
    def base_parts(self, q_text, q_siglip, siglip_img, margins_z, prior):
        """The three base signals, each a (B,) score, before routing."""
        cos = F.cosine_similarity(F.normalize(q_siglip, dim=-1),
                                  F.normalize(siglip_img, dim=-1), dim=-1)
        wm = torch.softmax(self.margin_select(q_text), dim=-1)
        s_marg = torch.tanh((wm * margins_z).sum(-1))
        wp = torch.softmax(self.meta_select(q_text), dim=-1)
        s_meta = torch.tanh((wp * prior).sum(-1))
        return torch.stack([cos, s_marg, s_meta], dim=-1)          # (B, 3)

    def base_score(self, q_text, q_siglip, siglip_img, margins_z, prior,
                   return_weights=False):
        parts = self.base_parts(q_text, q_siglip, siglip_img, margins_z, prior)
        w = torch.softmax(self.W_route(q_text) / self.tau_route, dim=-1)   # (B, 3)
        s = (w * parts).sum(-1)
        return (s, w, parts) if return_weights else s

    def residual_score(self, q_text, e, t, r, v, s_img, marg_z):
        q = self.adapt_query(q_text)
        toks = torch.stack([self.adapt_embed(e), self.adapt_tags(t),
                            self.adapt_risk(r), self.adapt_qwen(v),
                            self.adapt_siglip(s_img), self.adapt_margin(marg_z)],
                           dim=1) + self.type_embed.unsqueeze(0)
        toks = self.film(toks, q)
        c = self.fusion(tgt=q.unsqueeze(1), memory=toks)[:, 0]
        qn, cn = F.normalize(q, dim=-1), F.normalize(c, dim=-1)
        s_bil = F.cosine_similarity(self.bil_q(qn), self.bil_c(cn), dim=-1)
        s_mlp = torch.tanh(self.mlp(torch.cat([qn, cn, qn*cn, (qn-cn).abs()], -1)).squeeze(-1))
        return (s_bil + s_mlp) / self.tau

    def forward(self, q_text, q_siglip, e, t, r, v, s_img, marg_z, prior,
                return_parts=False):
        base, w, parts = self.base_score(q_text, q_siglip, s_img, marg_z, prior,
                                         return_weights=True)
        res = self.residual_score(q_text, e, t, r, v, s_img, marg_z)
        gate = torch.tanh(self.gate_head(q_text)).squeeze(-1)
        s = base + gate * res
        if return_parts:
            return s, base, res, w, parts, gate
        return s

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
