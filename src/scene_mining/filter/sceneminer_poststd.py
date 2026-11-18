#!/usr/bin/env python3
"""SceneMiner v2.6 with s_head and s_abst standardised before the Eq. (1) mixture
(Milestone 41, E5).

For each query, s_head and s_abst are computed on a fixed reference set (the 906
training scenes, in the same mode as the scenes being ranked: full or portable),
and their mean and standard deviation there are used to standardise both terms on
the ranked scenes:
    s = gamma_deg z(s_head) + (1 - gamma_deg) z(s_abst).
Every learned weight is the trained v2.6 model's; only the mixture changes.
"""
from __future__ import annotations
import torch

from scene_mining.filter.sceneminer_v2_6 import SceneMinerV26


class SceneMinerPostStd(SceneMinerV26):
    def set_reference(self, full, portable_prior):
        """full: (e, t, r, v, s_img, marg_z, prior) of the reference scenes;
        portable_prior: their prior with the 20 tag columns zeroed (portable mode)."""
        self.ref_full, self.ref_portable_prior = full, portable_prior

    def _abst(self, q_text, parts, prior):
        target = parts[:, 0]
        if self.use_macro_prior and prior is not None:
            meta_present = (prior[:, 20:26].abs().sum(-1) > 0).float()
            pm = self.macro_posterior(q_text); cols = self.macro_cols.clamp(min=0)
            z = (prior[:, cols] - self.meta_mu[cols]) / self.meta_sd[cols]
            mix = (pm * z).sum(-1) * meta_present; gm = pm.sum(-1) * meta_present
            target = (target + mix) / torch.sqrt(1.0 + gm ** 2)
        return target

    @staticmethod
    def _z(x, ref):
        return (x - ref.mean()) / ref.std().clamp(min=1e-6)

    def _head_abst_full(self, q_text, e, t, r, v, s_img, marg_z, prior, has_3d):
        B, dev = q_text.shape[0], q_text.device
        parts = self.base_parts(q_text, q_text, s_img, marg_z, prior)
        avail = torch.stack([torch.ones(B, device=dev), torch.ones(B, device=dev), has_3d], -1)
        base = (self.route(q_text, avail) * parts * avail).sum(-1)
        res = self.residual_score(q_text, e, t, r, v, s_img, marg_z, has_3d)
        if self.use_resnorm: res = self.alpha_res * self.res_norm(res)
        head = base + torch.tanh(self.gate_head(q_text)).squeeze(-1) * res
        return head, self._abst(q_text, parts, prior)

    def forward(self, q_text, q_siglip, e, t, r, v, s_img, marg_z, prior, has_3d=None, drop_margin=None, return_parts=False):
        assert not return_parts and drop_margin is None
        B, dev = q_text.shape[0], q_text.device
        if has_3d is None: has_3d = torch.ones(B, device=dev)
        head, abst = self._head_abst_full(q_text, e, t, r, v, s_img, marg_z, prior, has_3d)
        Nr = self.ref_full[0].shape[0]; q1 = q_text[:1].expand(Nr, -1)
        hr, ar = self._head_abst_full(q1, *self.ref_full, has_3d=torch.full((Nr,), float(has_3d[0]), device=dev))
        _, g, _ = self.concept(q_text)
        return g * self._z(head, hr) + (1 - g) * self._z(abst, ar)

    def _head_abst_portable(self, q_text, siglip_img, margins_z, prior, avail):
        parts = self.base_parts(q_text, q_text, siglip_img, margins_z, prior)
        return (self.route(q_text, avail) * parts * avail).sum(-1), self._abst(q_text, parts, prior)

    def base_score(self, q_text, siglip_img, margins_z, prior, avail):
        head, abst = self._head_abst_portable(q_text, siglip_img, margins_z, prior, avail)
        e, t, r, v, s_img, marg_z, _ = self.ref_full; Nr = s_img.shape[0]
        hr, ar = self._head_abst_portable(q_text[:1].expand(Nr, -1), s_img, marg_z, self.ref_portable_prior,
                                          avail[:1].expand(Nr, -1))
        _, g, _ = self.concept(q_text)
        return g * self._z(head, hr) + (1 - g) * self._z(abst, ar)
