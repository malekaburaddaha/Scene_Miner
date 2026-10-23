#!/usr/bin/env python3
"""SceneMiner v2.4 - one model: train-derived contrastive concept gate + abstention.

v2.3 and its post-hoc variant v2.3A split the wins (Milestones 14-15). v2.4
unifies them by fixing what made abstention costly: the concept-specificity
score gamma(q).

v2.3's gamma compared a query with ONE canonical string per concept and
thresholded the best cosine. Two failures followed: an in-vocabulary
low-contrast query scored gamma = 0.409 (one prompt cannot cover a concept's
phrasings), and out-of-vocabulary weather queries scored high on lexical overlap
("bright daytime driving" -> glare, 0.99), because nothing competed with the
four concepts.

v2.4 makes gamma a CONTRASTIVE posterior over classes built ONLY from training
material, with no evaluation query or label involved:

  concept classes (4)  the 156 sensor-degradation phrasings of the 390-query
                       TRAIN split, grouped by attribute;
  foil classes         the 84 macro-environmental phrasings of the TRAIN split,
                       grouped by predicate (night / day / rain / dry / dense /
                       sparse), plus the 12 generic out-of-vocabulary phrases
                       fixed in Milestone 14 (one class each).

Each class k scores a query by a soft maximum over its members,
    S_k(q) = tau * log mean_{i in k} exp(cos(q, b_i) / tau),
on embeddings mean-centred with the training bank's own mean. Then
    p(q) = softmax(S(q) / tau),   gamma(q) = sum over the 4 concepts of p_k(q),
and the margin column is chosen by softmax over the 4 concept scores / T_a.
tau is the only free quantity; it is chosen by LEAVE-ONE-OUT on the training
bank (each phrasing scored with itself removed), maximising the separation
between degradation and macro phrasings.

The final score always abstains toward the zero-shot prior in proportion to
1 - gamma:  s = gamma * s_v2.3-style + (1 - gamma) * p_siglip.  There is no
second variant.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scene_mining.filter.sceneminer_v2_3 import SceneMinerV23, OOV_BANK

TRAIN_ORDER = ["droplets", "glare", "wet_road", "low_contrast"]
TAU_GRID = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]


def macro_group(pred: dict) -> str:
    return "+".join(f"{k}={v}" for k, v in sorted(pred.items()))


def build_bank(train_queries):
    """-> texts, class names, class index per text, is_concept per class.
    Concept classes first, in TRAIN_ORDER."""
    texts, cls = [], []
    names = list(TRAIN_ORDER)
    for q in train_queries:
        if q["stratum"] == "sensor_degradation":
            texts.append(q["text"]); cls.append(names.index(q["attribute"]))
    for q in train_queries:
        if q["stratum"] == "macro_environmental":
            g = "macro:" + macro_group(q["predicate"])
            if g not in names: names.append(g)
            texts.append(q["text"]); cls.append(names.index(g))
    for t in OOV_BANK:
        names.append("oov:" + t); texts.append(t); cls.append(len(names) - 1)
    return texts, names, np.array(cls), np.array([i < 4 for i in range(len(names))])


def class_scores_np(C, cls, n_cls, tau, exclude=None):
    """C: (Q, Nb) cosines. Soft-max-over-members per class, optional LOO."""
    Q = C.shape[0]; out = np.full((Q, n_cls), -np.inf)
    for k in range(n_cls):
        m = cls == k
        for qi in range(Q):
            mm = m.copy()
            if exclude is not None and exclude[qi] >= 0: mm[exclude[qi]] = False
            if mm.sum() == 0: continue
            x = C[qi, mm] / tau
            out[qi, k] = tau * (np.logaddexp.reduce(x) - math.log(mm.sum()))
    return out


def gamma_np(S, is_concept, tau):
    z = S / tau; z = z - z.max(1, keepdims=True)
    p = np.exp(z); p[~np.isfinite(S)] = 0; p /= p.sum(1, keepdims=True)
    return p[:, is_concept].sum(1)


class SceneMinerV24(SceneMinerV23):
    def __init__(self, *args, **kw):
        kw.setdefault("use_abstain", True)
        super().__init__(*args, **kw)
        d_text = self.W_route.in_features
        self.register_buffer("bank", torch.zeros(1, d_text))
        self.register_buffer("bank_onehot", torch.zeros(1, 4))
        self.register_buffer("bank_count", torch.ones(4))
        self.register_buffer("tau_g", torch.tensor(0.05))
        self.n_cls = 4

    def set_bank(self, E_bank, cls, is_concept, mean_bank, tau=None):
        """Install the centred training bank. If tau is None, choose it by LOO."""
        mu = F.normalize(F.normalize(mean_bank.float(), dim=-1).mean(0), dim=-1)
        self.text_mean.copy_(mu)
        B = self._centre(E_bank.float())
        n_cls = int(cls.max()) + 1
        oh = torch.zeros(B.shape[0], n_cls, device=B.device)
        oh[torch.arange(B.shape[0]), torch.as_tensor(cls, device=B.device)] = 1.0
        self.bank = B.detach().clone(); self.bank_onehot = oh
        self.bank_count = oh.sum(0); self.n_cls = n_cls
        self.concept_text.copy_(F.normalize(
            torch.stack([B[torch.as_tensor(cls) == k].mean(0) for k in range(4)]), dim=-1))
        C = (B @ B.T).cpu().numpy()
        loo = np.arange(B.shape[0])
        report = {}
        for t in TAU_GRID:
            S = class_scores_np(C, cls, n_cls, t, exclude=loo)
            g = gamma_np(S, is_concept, t)
            pos, neg = g[cls < 4], g[cls >= 4]
            report[t] = {"pos_min": float(pos.min()), "pos_p05": float(np.percentile(pos, 5)),
                         "pos_median": float(np.median(pos)), "neg_max": float(neg.max()),
                         "neg_p95": float(np.percentile(neg, 95)), "neg_median": float(np.median(neg)),
                         "sep": float(np.percentile(pos, 5) - np.percentile(neg, 95))}
        # smoothest gate that still separates the training bank (LOO sep >= 0.99);
        # the separation optimum itself sits at the edge of the grid (a hard switch)
        ok = [t for t in report if report[t]["sep"] >= 0.99]
        best = tau if tau is not None else (max(ok) if ok else max(report, key=lambda t: report[t]["sep"]))
        self.tau_g.fill_(best)
        return best, report

    def concept(self, q_text):
        qc = self._centre(q_text)
        c = qc @ self.bank.T / self.tau_g                          # (B, Nb)
        m = c.max(-1, keepdim=True).values
        lse = torch.log((torch.exp(c - m) @ self.bank_onehot).clamp(min=1e-30)) + m \
              - torch.log(self.bank_count).unsqueeze(0)          # (B, n_cls)
        S = self.tau_g * lse
        p = torch.softmax(S / self.tau_g, dim=-1)
        gamma = p[:, :4].sum(-1)
        a = torch.softmax(S[:, :4] / self.T_a, dim=-1)
        return a, gamma, S[:, :4]

    # the state dict carries a bank whose size is only known after set_bank
    def _load_from_state_dict(self, sd, prefix, *args, **kw):
        for k in ("bank", "bank_onehot", "bank_count"):
            if prefix + k in sd:
                setattr(self, k, torch.empty_like(sd[prefix + k]))
        if prefix + "bank_onehot" in sd: self.n_cls = sd[prefix + "bank_onehot"].shape[1]
        super()._load_from_state_dict(sd, prefix, *args, **kw)
