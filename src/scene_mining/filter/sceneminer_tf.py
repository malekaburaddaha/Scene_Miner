#!/usr/bin/env python3
"""Training-free control for SceneMiner (Milestone 41, E2).

Same PartNorm-standardised base parts [cos, m_sel, pi_sel], the same training-text
factored concept gate gamma_deg and fallback s_abst, and the same inference clamp
as SceneMiner v2.6, with NO learned parameter:
  * router w(q): equal weights over the available parts (W_route at its zero
    initialisation), with the margin's share scaled by gamma_deg and renormalised,
    exactly as in v2.6;
  * residual: off (gate head at zero, so g(q) = tanh(0) = 0; no FiLM, no decoder);
  * margin sign: the initial constant tanh(2) (no learned map);
  * margin column: the concept-bank selection softmax(S'_concepts / 0.1), as in v2.6;
  * prior entry pi_sel: no learned selector. Two training-free choices:
      "bank"     the metadata columns named by the concept bank, weighted by the
                 context posterior gamma_m (the same weights s_abst uses); zero
                 when the query names no context;
      "uniform"  the zero-initialised selector, a uniform average of the 26 prior
                 columns (query-independent).
PartNorm statistics are recomputed, without labels, for the control's own parts over
the 906 training scenes and the Stage-2 training queries
(scripts/e2_build_training_free.py).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from scene_mining.filter.sceneminer_v2_6 import SceneMinerV26


class SceneMinerTrainingFree(SceneMinerV26):
    def __init__(self, *args, prior_select="bank", **kw):
        super().__init__(*args, **kw)
        assert prior_select in ("bank", "uniform")
        self.prior_select = prior_select

    def prior_weights(self, q_text):
        """(B, 26) weights over the prior columns."""
        n = self.meta_mu.shape[0]
        if self.prior_select == "uniform":
            return torch.full((q_text.shape[0], n), 1.0 / n, device=q_text.device)
        pm = self.macro_posterior(q_text)                               # (B, n_cls), zero off metadata classes
        has = self.macro_cols >= 0
        W = torch.zeros(q_text.shape[0], n, device=q_text.device)
        return W.index_add(1, self.macro_cols[has], pm[:, has])

    def base_parts(self, q_text, q_siglip, siglip_img, margins_z, prior, normalize=True):
        cos = F.cosine_similarity(F.normalize(q_siglip, dim=-1), F.normalize(siglip_img, dim=-1), dim=-1)
        wm, sgn = self.channel_weights(q_text)
        s_marg = torch.tanh(sgn.squeeze(-1) * (wm * margins_z).sum(-1))
        s_meta = torch.tanh((self.prior_weights(q_text) * prior).sum(-1))
        p = torch.stack([cos, s_marg, s_meta], dim=-1)
        if not normalize: return p
        p = self.part_norm(p)
        if self.use_rescue:                                             # the inference clamp of v2.6
            strong = p[:, 0] > self.rescue_sig
            p = torch.stack([p[:, 0], torch.where(strong, p[:, 1].clamp(min=self.rescue_floor), p[:, 1]), p[:, 2]], -1)
        return p
