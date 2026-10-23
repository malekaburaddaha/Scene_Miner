#!/usr/bin/env python3
"""SceneMiner v2.3 - concept-specificity gating, bounded residual, FN rescue.

Builds on v2.2 (PartNorm, entropy floor, channel dropout, explicit modality
availability). Three changes, each tied to a measurement:

(1) CONCEPT-SPECIFICITY GATE. Only four VLM margin columns exist (droplets,
    glare, wet road, low contrast). v2.2 chose among them with a learned map
    that picked the wrong column on 8/12 BDD100K queries, and it had no way to
    say "none of the four applies". v2.3 selects the column by fixed semantic
    alignment, softmax(c_k(q)/T_a) over mean-centred SigLIP2 text cosines
    c_k(q) to four canonical attribute descriptions, and scales the margin's
    routing weight by an in-vocabulary specificity
        gamma(q) = sigmoid((max_k c_k(q) - tau_c) / T_c),
    renormalising the routing mass afterwards. For an out-of-vocabulary query
    (gamma ~ 0) the margin is switched off and the base score falls back to
    SigLIP2 and the metadata prior. tau_c and T_c are set from two label-free
    text banks at train time - the in-vocabulary training phrasings and a fixed
    bank of generic out-of-vocabulary driving phrases - and never from any
    evaluation query.

(2) RESNORM - A BOUNDED RESIDUAL. Decomposing v2.2 on human labels showed the
    gated residual's contribution has 2-3x the spread of the base score, so it
    dominated the ranking. That is PartNorm's scale problem one level up: the
    base alone scored 0.9609 vs 0.8813 (full) on droplets, and recovered 12/20
    VLM-missed glare scenes vs 5/20 for the full model. The residual is now
    standardised by running statistics and enters as
        s = s_base + alpha_res * g(q) * rho_tilde,     |g| <= 1,
    so it can correct the base by at most alpha_res base-part standard
    deviations. The spread gap itself is label-free and visible from scores
    alone; alpha_res = 0.5 is fixed a priori.

(3) ASYMMETRIC FALSE-NEGATIVE RESCUE. A confident VLM "no" should not veto a
    strong visual match: when the standardised SigLIP2 part exceeds +1 sigma,
    the standardised margin part is lower-clamped at -0.5 sigma. Thresholds are
    fixed a priori. (Measured honestly: on the 20 VLM-missed glare scenes the
    margin is not strongly negative - mean +0.80 sigma, none below -0.5 - so
    this term is expected to matter little there; it is kept because the veto
    case is real in principle and costs nothing when inactive.)

(4) OOV ABSTENTION - POST-HOC, OFF BY DEFAULT (`use_abstain`). Added in
    Milestone 15 AFTER the Stratum-5 semantic results were seen. Every learned
    part of the head (routing, residual) was trained on in-vocabulary
    degradation queries only, and on semantic queries it degraded SigLIP2 badly.
    With abstention the score interpolates toward the standardised SigLIP2 part
    by the same specificity:  s_A = gamma * s + (1 - gamma) * p_siglip.
    No new hyperparameter; at gamma ~ 1 it is v2.3 unchanged.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from scene_mining.filter.sceneminer_v2_2 import SceneMinerV22, PartNorm

# canonical descriptions of the four cached VLM margin columns, in TRAINING order
CONCEPT_PROMPT = {
    "droplets":     "water droplets on camera lens",
    "glare":        "bright headlight or sun glare",
    "wet_road":     "wet reflective road surface",
    "low_contrast": "dark low contrast poor visibility",
}
# generic driving phrases naming NONE of the four concepts; used only to place
# the specificity threshold. Fixed before any evaluation query was written, and
# deliberately free of weather words and of the Stratum-5 semantic categories.
OOV_BANK = [
    "a parking lot", "a roundabout", "a tunnel entrance", "a bridge over a river",
    "a gas station", "a suburban cul-de-sac", "a toll booth", "a railway crossing",
    "a school zone sign", "a freeway on-ramp", "a shopping mall entrance",
    "a narrow alley between buildings",
]


class ResNorm(PartNorm):
    """Running standardisation of the scalar residual."""
    def __init__(self, momentum=0.01, eps=1e-5):
        super().__init__(1, momentum, eps)

    def forward(self, r):                                   # (B,)
        return super().forward(r.unsqueeze(-1)).squeeze(-1)


class SceneMinerV23(SceneMinerV22):
    def __init__(self, *args, alpha_res=0.5, rescue_sig=1.0, rescue_floor=-0.5,
                 T_a=0.10, use_gate=True, use_rescue=True, use_resnorm=True,
                 use_abstain=False, **kw):
        super().__init__(*args, **kw)
        d_text = self.W_route.in_features
        n_m = self.margin_select.out_features
        self.res_norm = ResNorm()
        self.register_buffer("concept_text", torch.zeros(n_m, d_text))
        self.register_buffer("tau_c", torch.tensor(0.0))
        self.register_buffer("T_c", torch.tensor(1.0))
        self.register_buffer("T_a", torch.tensor(float(T_a)))
        self.alpha_res, self.rescue_sig, self.rescue_floor = alpha_res, rescue_sig, rescue_floor
        # ablation switches (inference-time; training uses all three)
        self.use_gate, self.use_rescue, self.use_resnorm = use_gate, use_rescue, use_resnorm
        self.use_abstain = use_abstain
        # semantic alignment replaces the learned selector entirely
        for p in self.margin_select.parameters(): p.requires_grad_(False)

    # ---------------------------------------------------------------- concepts
    def _centre(self, X):
        return F.normalize(F.normalize(X, dim=-1) - self.text_mean.unsqueeze(0), dim=-1)

    def set_concepts(self, E_concept, in_bank, oov_bank):
        """Install concept embeddings and place the specificity threshold midway
        between the in-vocabulary and out-of-vocabulary phrase banks."""
        allb = torch.cat([E_concept, in_bank, oov_bank], 0)
        self.text_mean.copy_(F.normalize(F.normalize(allb.float(), dim=-1).mean(0), dim=-1))
        self.concept_text.copy_(self._centre(E_concept.float()))
        self.attr_text.copy_(self.concept_text)          # keep v2.2 buffer coherent
        cin = (self._centre(in_bank.float()) @ self.concept_text.T).max(-1).values
        cout = (self._centre(oov_bank.float()) @ self.concept_text.T).max(-1).values
        mi, mo = float(cin.median()), float(cout.median())
        self.tau_c.fill_(0.5 * (mi + mo))
        self.T_c.fill_(max((mi - mo) / 6.0, 1e-3))     # medians land at sigma(+-3)
        return {"median_in": mi, "median_oov": mo, "tau_c": float(self.tau_c),
                "T_c": float(self.T_c)}

    def concept(self, q_text):
        c = self._centre(q_text) @ self.concept_text.T              # (B,4)
        a = torch.softmax(c / self.T_a, dim=-1)
        gamma = torch.sigmoid((c.max(-1).values - self.tau_c) / self.T_c)
        return a, gamma, c

    # ---------------------------------------------------------------- base
    def channel_weights(self, q_text):
        a, _, _ = self.concept(q_text)
        return a, torch.tanh(self.margin_sign(q_text))

    def base_parts(self, q_text, q_siglip, siglip_img, margins_z, prior,
                   normalize=True):
        p = super().base_parts(q_text, q_siglip, siglip_img, margins_z, prior,
                               normalize=normalize)
        if normalize and self.use_rescue:
            strong = p[:, 0] > self.rescue_sig
            m = torch.where(strong, p[:, 1].clamp(min=self.rescue_floor), p[:, 1])
            p = torch.stack([p[:, 0], m, p[:, 2]], -1)
        return p

    def route(self, q_text, avail):
        w = super().route(q_text, avail)
        if not self.use_gate: return w
        _, gamma, _ = self.concept(q_text)
        w = torch.stack([w[:, 0], w[:, 1] * gamma, w[:, 2]], -1)
        s = w.sum(-1, keepdim=True)
        # if the margin was the only channel left and gamma ~ 0, fall back to
        # the cosine rather than divide by ~0
        fallback = torch.zeros_like(w); fallback[:, 0] = 1.0
        return torch.where(s > 1e-6, w / s.clamp(min=1e-6), fallback)

    def abstain(self, q_text, s, parts):
        if not self.use_abstain: return s
        _, gamma, _ = self.concept(q_text)
        return gamma * s + (1.0 - gamma) * parts[:, 0]

    def base_score(self, q_text, siglip_img, margins_z, prior, avail):
        """Portable routed base (used where no 3D rig / tags exist)."""
        parts = self.base_parts(q_text, q_text, siglip_img, margins_z, prior)
        s = (self.route(q_text, avail) * parts * avail).sum(-1)
        return self.abstain(q_text, s, parts)

    # ---------------------------------------------------------------- forward
    def forward(self, q_text, q_siglip, e, t, r, v, s_img, marg_z,
                prior, has_3d=None, drop_margin=None, return_parts=False):
        B, dev = q_text.shape[0], q_text.device
        if has_3d is None: has_3d = torch.ones(B, device=dev)
        if drop_margin is None: drop_margin = torch.zeros(B, device=dev)
        parts = self.base_parts(q_text, q_siglip, s_img, marg_z, prior)
        avail = torch.stack([torch.ones(B, device=dev), 1.0 - drop_margin, has_3d], -1)
        w = self.route(q_text, avail)
        base = (w * parts * avail).sum(-1)
        res = self.residual_score(q_text, e, t, r, v, s_img, marg_z, has_3d,
                                  drop_margin=drop_margin)
        gate = torch.tanh(self.gate_head(q_text)).squeeze(-1)
        if self.use_resnorm:
            res = self.alpha_res * self.res_norm(res)
        s = self.abstain(q_text, base + gate * res, parts)
        if return_parts: return s, base, res, w, parts, gate
        return s


if __name__ == "__main__":
    torch.manual_seed(0)
    m = SceneMinerV23(); B = 8
    m.set_concepts(torch.randn(4, 768), torch.randn(72, 768), torch.randn(12, 768))
    m.eval()
    qt = torch.randn(B, 768); e = torch.randn(B, 256); t = torch.randn(B, 20)
    r = torch.rand(B, 1); v = torch.randn(B, 2048); si = torch.randn(B, 768)
    mz = torch.randn(B, 4); pr = torch.randn(B, 26)
    s, base, res, w, parts, gate = m(qt, qt, e, t, r, v, si, mz, pr, return_parts=True)
    assert torch.allclose(s, base, atol=1e-6), "identity at init"
    assert torch.allclose(w.sum(-1), torch.ones(B), atol=1e-5)
    # a query aligned with a concept must get high gamma; the mean direction low
    a, g, c = m.concept(m.concept_text[2:3] * 1.0 + m.text_mean)
    print(f"  aligned query: gamma {float(g):.3f}, argmax concept {int(a.argmax())}")
    # gamma -> 0 must switch the margin channel off
    m.tau_c.fill_(10.0)
    w0 = m.route(qt, torch.ones(B, 3))
    assert float(w0[:, 1].max()) < 1e-6 and torch.allclose(w0.sum(-1), torch.ones(B), atol=1e-5)
    print(f"  OOV routing (gamma~0): {w0[0].detach().numpy().round(4)}")
    print(f"  trainable params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
    m.use_abstain = True; m.tau_c.fill_(10.0)
    sA, _, _, _, pA, _ = m(qt, qt, e, t, r, v, si, mz, pr, return_parts=True)
    assert torch.allclose(sA, pA[:, 0], atol=1e-5), "gamma~0 must abstain to SigLIP2"
    print("  abstention at gamma~0 returns the SigLIP2 part exactly")
    print("  v2.3 OK")
