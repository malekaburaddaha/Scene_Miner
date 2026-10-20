#!/usr/bin/env python3
"""SceneMiner v2.2 - anti-shortcut routing with explicit modality availability.

Three failures diagnosed in Milestones 6 and 8 are addressed here.

(1) TEACHER-SHORTCUT COLLAPSE. v2.1's router put 96.7-99.2 % of its weight on
    the VLM margin, because the training labels are themselves VLM-derived: the
    margin is a tautological oracle for its own pseudo-labels. The fix is
    threefold - channel dropout during training (the router cannot rely on a
    channel that disappears), an entropy floor that caps any single channel at
    `max_route`, and agreement weighting so pairs where SigLIP2 and the VLM
    disagree carry less weight than pairs where they corroborate.

(2) MODALITY AVAILABILITY. On a monocular corpus there is no BEV embedding,
    tag vector or risk scalar. v2.1 was handed zero tensors for them, which is
    not the same as "absent": the router still allocated weight to a channel
    whose value was a constant, and the fusion head still attended to zero
    tokens. v2.2 takes an explicit `has_3d` mask, RENORMALISES the routing
    distribution over the channels that actually exist, and masks the missing
    tokens out of cross-attention via `memory_key_padding_mask`.

(3) SCALE INCOMMENSURABILITY. Measured on BDD100K: the VLM-margin part carries
    28.1x the spread of the SigLIP2-cosine part (std 0.5419 vs 0.0193), because
    one is a tanh of a z-scored logit and the other is a raw cosine. A convex
    routing weight over parts on such different scales is NOT an influence
    weight - a nominal 50/50 blend is ~96 % margin in practice, which is why
    v2.1's blend scored BELOW both of the signals it blended (0.4181 vs 0.6218
    and 0.5871). `PartNorm` standardises each base part with running statistics,
    so a routing weight finally means what it says.

(4) NON-TRANSFERABLE CHANNEL SELECTION. `margin_select` is a learned linear map
    of the query embedding, fit on 72 nuScenes query phrasings. On BDD100K's
    unseen phrasings it picked the wrong attribute channel on 8 of 12 queries at
    p_max ~ 0.26 - a near-uniform average of all four margins, i.e. mush. The
    selector is now PRIMED by text-text similarity between the query and the
    four attribute prompts in SigLIP2 text space, a zero-shot quantity that
    transfers by construction; the learned map is zero-initialised and only
    corrects that prior. The similarity is computed on MEAN-CENTRED embeddings:
    SigLIP2 is trained for text-image alignment, not text-text, so its text
    embeddings are strongly anisotropic and a raw cosine collapses onto one
    generic direction - it agreed with a held-out a-priori channel map on only
    6/12 BDD queries, while removing the mean direction raises that to 10/12.

(5) IDENTITY-PRESERVING START is retained: routing is uniform over available
    channels and the residual gate is zero at step 0.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from scene_mining.filter.unified_filter import FiLM, ModalityAdapter

# token order in the fusion memory; the first three need the 3D rig
TOK_3D = [True, True, True, False, False, False]   # embed, tags, risk, qwen, siglip, margin

# Prompts for the four margin channels, worded from the Stage-1 VLM QUESTIONS
# themselves (never from any label), used to prime channel selection by
# text-text similarity so it transfers to unseen query phrasings.
ATTR_PROMPT = {
    "droplets":     "water droplets or rain smear on the camera lens",
    "glare":        "strong light glare, bloom, or lens flare",
    "wet_road":     "a visibly wet or reflective road surface",
    "low_contrast": "object boundaries obscured by darkness, fog, or bloom",
}


class PartNorm(nn.Module):
    """Running standardisation of the base parts, so that a routing weight is an
    influence weight. Behaves like BatchNorm1d in spirit but is kept explicit:
    the parts are a tiny, interpretable vector and we want their statistics
    frozen at eval time even when a query is scored against a whole corpus."""

    def __init__(self, n: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.momentum, self.eps = momentum, eps

    def forward(self, parts):                      # (B, n)
        if self.training and parts.shape[0] > 1:
            with torch.no_grad():
                m, v = parts.mean(0), parts.var(0, unbiased=False)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * m)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * v)
        return (parts - self.running_mean) / torch.sqrt(self.running_var + self.eps)


class SceneMinerV22(nn.Module):
    N_TOKENS = 6
    N_BASE = 3          # siglip cosine | vlm margin | tag+metadata prior

    def __init__(self, d_embed=256, n_tags=20, d_qwen=2048, d_siglip=768,
                 n_margins=4, n_meta=6, d_text=768, d_model=256, n_heads=4,
                 n_fusion_layers=2, bilinear_rank=64, dropout=0.1,
                 tau=0.30, tau_route=1.0, max_route=0.70,
                 risk_stats=(0.417797, 0.081112)):
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
        self.max_route = float(max_route)
        self.part_norm = PartNorm(self.N_BASE)
        self.register_buffer("attr_text", torch.zeros(n_margins, d_text))
        self.register_buffer("text_mean", torch.zeros(d_text))
        self.register_buffer("tau_sel", torch.tensor(0.10))
        self.W_route = nn.Linear(d_text, self.N_BASE)
        self.margin_select = nn.Linear(d_text, n_margins)
        # signed channel combination: queries may name the ABSENCE of an
        # attribute ("bright daytime driving" = not low-contrast). Initialised
        # to +1 so the margin channel is live at step 0.
        self.margin_sign = nn.Linear(d_text, 1)
        nn.init.zeros_(self.margin_sign.weight)
        nn.init.constant_(self.margin_sign.bias, 2.0)
        self.meta_select = nn.Linear(d_text, n_tags + n_meta)
        self.gate_head = nn.Linear(d_text, 1)
        for m in (self.W_route, self.margin_select, self.meta_select, self.gate_head):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def set_attr_text(self, E, bank=None):
        """Install the frozen attribute-prompt embeddings, mean-centred.

        `bank` is any label-free set of text embeddings used to estimate the
        generic direction of SigLIP2's text space (we use the training queries
        plus the attribute prompts). Centring is what makes text-text
        similarity discriminative; see the module docstring.
        """
        E = F.normalize(E.detach().float(), dim=-1)
        B = E if bank is None else F.normalize(bank.detach().float(), dim=-1)
        mu = F.normalize(B.mean(0), dim=-1)
        self.text_mean.copy_(mu)
        self.attr_text.copy_(F.normalize(E - mu.unsqueeze(0), dim=-1))

    # ------------------------------------------------------------------
    def channel_weights(self, q_text):
        """Which margin channel does this query name? Text-text similarity to the
        attribute prompts is the zero-shot prior; the learned map (zero-init)
        only corrects it. Returns (weights, signed scale)."""
        qc = F.normalize(F.normalize(q_text, dim=-1) - self.text_mean.unsqueeze(0), dim=-1)
        prior = qc @ self.attr_text.T / self.tau_sel
        w = torch.softmax(prior + self.margin_select(q_text), dim=-1)
        sgn = torch.tanh(self.margin_sign(q_text))            # (B,1), +1 at init
        return w, sgn

    def base_parts(self, q_text, q_siglip, siglip_img, margins_z, prior,
                   normalize=True):
        cos = F.cosine_similarity(F.normalize(q_siglip, dim=-1),
                                  F.normalize(siglip_img, dim=-1), dim=-1)
        wm, sgn = self.channel_weights(q_text)
        s_marg = torch.tanh(sgn.squeeze(-1) * (wm * margins_z).sum(-1))
        wp = torch.softmax(self.meta_select(q_text), dim=-1)
        s_meta = torch.tanh((wp * prior).sum(-1))
        parts = torch.stack([cos, s_marg, s_meta], dim=-1)
        return self.part_norm(parts) if normalize else parts

    def route(self, q_text, avail):
        """Routing distribution RENORMALISED over available base channels.

        `avail` is (B, N_BASE) in {0,1}. Masked channels receive -inf logits, so
        softmax redistributes their mass rather than leaving it stranded on a
        channel whose value is a constant - the bug that sank v2.1 on BDD100K.
        """
        logits = self.W_route(q_text) / self.tau_route
        logits = logits.masked_fill(avail < 0.5, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def residual_score(self, q_text, e, t, r, v, s_img, marg_z, has_3d,
                       drop_margin=None):
        q = self.adapt_query(q_text)
        toks = torch.stack([self.adapt_embed(e), self.adapt_tags(t),
                            self.adapt_risk(r), self.adapt_qwen(v),
                            self.adapt_siglip(s_img), self.adapt_margin(marg_z)],
                           dim=1) + self.type_embed.unsqueeze(0)
        toks = self.film(toks, q)
        need3d = torch.tensor(TOK_3D, device=toks.device)
        # True = ignore this memory position
        pad = (~has_3d.bool().unsqueeze(1)) & need3d.unsqueeze(0)     # (B, 6)
        if drop_margin is not None:
            is_marg = torch.zeros(self.N_TOKENS, dtype=torch.bool, device=toks.device)
            is_marg[5] = True            # the VLM-margin token
            pad = pad | (drop_margin.bool().unsqueeze(1) & is_marg.unsqueeze(0))
        c = self.fusion(tgt=q.unsqueeze(1), memory=toks,
                        memory_key_padding_mask=pad)[:, 0]
        qn, cn = F.normalize(q, dim=-1), F.normalize(c, dim=-1)
        s_bil = F.cosine_similarity(self.bil_q(qn), self.bil_c(cn), dim=-1)
        s_mlp = torch.tanh(self.mlp(torch.cat([qn, cn, qn*cn, (qn-cn).abs()], -1)).squeeze(-1))
        return (s_bil + s_mlp) / self.tau

    def forward(self, q_text, q_siglip, e, t, r, v, s_img, marg_z,
                prior, has_3d=None, drop_margin=None, return_parts=False):
        B = q_text.shape[0]
        dev = q_text.device
        if has_3d is None: has_3d = torch.ones(B, device=dev)
        if drop_margin is None: drop_margin = torch.zeros(B, device=dev)
        parts = self.base_parts(q_text, q_siglip, siglip_img=s_img,
                                margins_z=marg_z, prior=prior)
        # channel availability: margin may be dropped; meta prior needs the rig
        avail = torch.stack([torch.ones(B, device=dev),
                             1.0 - drop_margin,
                             has_3d], dim=-1)
        w = self.route(q_text, avail)
        base = (w * parts * avail).sum(-1)
        res = self.residual_score(q_text, e, t, r, v, s_img, marg_z, has_3d,
                                  drop_margin=drop_margin)
        gate = torch.tanh(self.gate_head(q_text)).squeeze(-1)
        s = base + gate * res
        if return_parts: return s, base, res, w, parts, gate
        return s

    # --- anti-shortcut regulariser -------------------------------------
    def route_penalty(self, q_text, avail):
        """Penalise any single channel exceeding max_route. Zero when the
        distribution is already flat enough, so it costs nothing once the
        router is behaving."""
        w = self.route(q_text, avail)
        over = F.relu(w.max(dim=-1).values - self.max_route)
        return (over ** 2).mean()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    torch.manual_seed(0)
    m = SceneMinerV22(); B = 8
    m.set_attr_text(torch.randn(4, 768), bank=torch.randn(40, 768))
    m.eval()                       # frozen PartNorm stats for the init checks
    qt = torch.randn(B, 768); e = torch.randn(B, 256); t = torch.randn(B, 20)
    r = torch.rand(B, 1); v = torch.randn(B, 2048); si = torch.randn(B, 768)
    mz = torch.randn(B, 4); pr = torch.randn(B, 26)
    s, base, res, w, parts, gate = m(qt, qt, e, t, r, v, si, mz, pr, return_parts=True)
    assert torch.allclose(s, base, atol=1e-6), "gate must be identity at init"
    assert torch.allclose(w, torch.full_like(w, 1/3), atol=1e-6), "routing not uniform"
    # monocular path: no 3D rig -> meta channel removed, mass renormalised over 2
    h = torch.zeros(B)
    s2, base2, _, w2, _, _ = m(qt, qt, e, t, r, v, si, mz, pr, has_3d=h, return_parts=True)
    assert float(w2[:, 2].abs().max()) == 0.0, "meta channel must be zero without rig"
    assert torch.allclose(w2[:, :2].sum(-1), torch.ones(B), atol=1e-6), "mass not renormalised"
    print(f"  params {sum(p.numel() for p in m.parameters()):,}")
    print(f"  full rig  routing: {w[0].detach().numpy().round(4)}")
    print(f"  monocular routing: {w2[0].detach().numpy().round(4)}  (renormalised over 2)")
    print(f"  identity at init: max|s-base| = {float((s-base).abs().max()):.2e}")
    # dropping the margin must change BOTH base and residual
    dm = torch.ones(B)
    r_on  = m.residual_score(qt, e, t, r, v, si, mz, torch.ones(B))
    r_off = m.residual_score(qt, e, t, r, v, si, mz, torch.ones(B), drop_margin=dm)
    assert not torch.allclose(r_on, r_off), "margin dropout must reach the residual"
    _, b_off, _, w_off, _, _ = m(qt, qt, e, t, r, v, si, mz, pr,
                                 drop_margin=dm, return_parts=True)
    assert float(w_off[:, 1].abs().max()) == 0.0, "margin channel must be zero when dropped"
    print(f"  margin dropped routing: {w_off[0].detach().numpy().round(4)}")
    print(f"  residual responds to dropout: max|d| = {float((r_on-r_off).abs().max()):.3f}")
    # PartNorm must make the parts commensurate
    raw = m.base_parts(qt, qt, si, mz, pr, normalize=False)
    nrm = m.base_parts(qt, qt, si, mz, pr)
    print(f"  raw part std   {raw.std(0).detach().numpy().round(4)}")
    print(f"  normed part std {nrm.std(0).detach().numpy().round(4)}")
    # text-primed selection must be live at init (learned map is zero)
    w_sel, sgn = m.channel_weights(qt)
    assert float(sgn.mean()) > 0.9, "margin channel must be live at init"
    assert float(w_sel.max()) > 0.26, "text prior should not be uniform"
    print(f"  channel select p_max {float(w_sel.max(-1).values.mean()):.3f}  "
          f"sign {float(sgn.mean()):+.3f}")
    print("  v2.2 OK")
