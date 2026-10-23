#!/usr/bin/env python3
"""SceneMiner (v2.5) - factored concept gate: gamma_deg and gamma_macro.

v2.4 decided "does a degradation concept apply?" with ONE softmax over 4
degradation concepts, 7 macro-context classes and 12 generic phrases. That
couples two independent axes: a query can name a degradation AND a context
("slick pavement in the dark"), and the context class then steals the mass the
concept needs. And when gamma ~ 0, v2.4 abstained to the SigLIP2 cosine alone,
discarding the recorded metadata even when the query named a context the bank
recognised ("pedestrians at night").

v2.5 factors the gate, using only the same 252-phrase TRAINING bank:

  gamma_deg(q)   the same contrastive posterior, computed on embeddings from
                 which (half of) the macro-context subspace has been removed -
                 the span of the 7 macro class means. A context word no longer
                 pulls a degradation query toward a foil. The removal strength
                 (0.5) and tau were chosen on training text by leave-GROUP-out
                 (scripts/explore_factored_gate_train.py).

  gamma_macro(q) the posterior mass on each macro class (night, day, rain, dry,
                 dense, sparse, moderate) from the unprojected softmax. Each
                 class maps, by name, to the recorded-metadata column it denotes.

  score   s = gamma_deg * s_head + (1 - gamma_deg) * s_abstain,
          s_abstain = (p_cos + sum_m gamma_m z_m) / sqrt(1 + (sum_m gamma_m)^2)
  where z_m is that metadata column standardised with TRAINING-scene
  statistics. Where no metadata exists (BDD100K) the sum is empty and
  s_abstain = p_cos, exactly as before.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from scene_mining.filter.sceneminer_v2_4 import SceneMinerV24, class_scores_np, gamma_np, TAU_GRID

# macro class name -> column of the 26-d prior holding that recorded metadata
META_COL = {"macro:lighting=night": 20, "macro:lighting=day": 21, "macro:weather=rain": 22,
            "macro:weather=dry": 23, "macro:density=dense": 24, "macro:density=sparse": 25}


class SceneMinerV25(SceneMinerV24):
    def __init__(self, *args, lam_proj=0.5, **kw):
        super().__init__(*args, **kw)
        d = self.W_route.in_features
        self.lam_proj = lam_proj
        self.register_buffer("macro_basis", torch.zeros(d, 7))
        self.register_buffer("bank_proj", torch.zeros(1, d))
        self.register_buffer("macro_cols", torch.full((1,), -1, dtype=torch.long))
        self.register_buffer("meta_mu", torch.zeros(26))
        self.register_buffer("meta_sd", torch.ones(26))
        self.use_macro_prior = True

    def _proj(self, X):
        Q = self.macro_basis
        return F.normalize(X - self.lam_proj * (X @ Q) @ Q.T, dim=-1)

    def set_bank(self, E_bank, cls, is_concept, mean_bank, names, tau=0.02):
        super().set_bank(E_bank, cls, is_concept, mean_bank, tau=tau)
        B = self.bank
        macro_ids = [k for k, n in enumerate(names) if n.startswith("macro:")]
        M = torch.stack([B[torch.as_tensor(cls, device=B.device) == k].mean(0) for k in macro_ids])
        Qb, _ = torch.linalg.qr(M.T)
        self.macro_basis = Qb.detach().clone()
        self.bank_proj = self._proj(B).detach().clone()
        cols = torch.full((self.n_cls,), -1, dtype=torch.long)
        for k, n in enumerate(names):
            if n in META_COL: cols[k] = META_COL[n]
        self.macro_cols = cols.to(B.device)
        return float(self.tau_g)

    def set_meta_stats(self, prior_train):
        self.meta_mu.copy_(prior_train.mean(0)); self.meta_sd.copy_(prior_train.std(0).clamp(min=1e-3))

    def _class_scores(self, qc, bank):
        c = qc @ bank.T / self.tau_g
        m = c.max(-1, keepdim=True).values
        return self.tau_g * (torch.log((torch.exp(c - m) @ self.bank_onehot).clamp(min=1e-30)) + m
                             - torch.log(self.bank_count).unsqueeze(0))

    def concept(self, q_text):
        qc = self._centre(q_text)
        S = self._class_scores(self._proj(qc), self.bank_proj)        # gamma_deg on residuals
        gamma = torch.softmax(S / self.tau_g, -1)[:, :4].sum(-1)
        a = torch.softmax(S[:, :4] / self.T_a, -1)
        return a, gamma, S[:, :4]

    def macro_posterior(self, q_text):
        S = self._class_scores(self._centre(q_text), self.bank)       # unprojected
        p = torch.softmax(S / self.tau_g, -1)
        has = (self.macro_cols >= 0).float().unsqueeze(0)
        return p * has                                                   # (B, n_cls), zero off macro-with-column

    def abstain(self, q_text, s, parts, prior=None):
        if not self.use_abstain: return s
        _, g, _ = self.concept(q_text)
        target = parts[:, 0]
        if self.use_macro_prior and prior is not None:
            meta_present = (prior[:, 20:26].abs().sum(-1) > 0).float()
            pm = self.macro_posterior(q_text)                            # (B, n_cls)
            cols = self.macro_cols.clamp(min=0)
            z = (prior[:, cols] - self.meta_mu[cols]) / self.meta_sd[cols]   # (B, n_cls)
            mix = (pm * z).sum(-1) * meta_present
            gm = pm.sum(-1) * meta_present
            target = (target + mix) / torch.sqrt(1.0 + gm ** 2)
        return g * s + (1.0 - g) * target

    def base_score(self, q_text, siglip_img, margins_z, prior, avail):
        parts = self.base_parts(q_text, q_text, siglip_img, margins_z, prior)
        s = (self.route(q_text, avail) * parts * avail).sum(-1)
        return self.abstain(q_text, s, parts, prior)

    def forward(self, q_text, q_siglip, e, t, r, v, s_img, marg_z,
                prior, has_3d=None, drop_margin=None, return_parts=False):
        B, dev = q_text.shape[0], q_text.device
        if has_3d is None: has_3d = torch.ones(B, device=dev)
        if drop_margin is None: drop_margin = torch.zeros(B, device=dev)
        parts = self.base_parts(q_text, q_siglip, s_img, marg_z, prior)
        avail = torch.stack([torch.ones(B, device=dev), 1.0 - drop_margin, has_3d], -1)
        w = self.route(q_text, avail)
        base = (w * parts * avail).sum(-1)
        res = self.residual_score(q_text, e, t, r, v, s_img, marg_z, has_3d, drop_margin=drop_margin)
        gate = torch.tanh(self.gate_head(q_text)).squeeze(-1)
        if self.use_resnorm: res = self.alpha_res * self.res_norm(res)
        s = self.abstain(q_text, base + gate * res, parts, prior)
        if return_parts: return s, base, res, w, parts, gate
        return s

    def _load_from_state_dict(self, sd, prefix, *args, **kw):
        for k in ("bank_proj", "macro_cols"):
            if prefix + k in sd: setattr(self, k, torch.empty_like(sd[prefix + k]))
        super()._load_from_state_dict(sd, prefix, *args, **kw)
