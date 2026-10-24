#!/usr/bin/env python3
"""SceneMiner (v2.6) - conditional context removal for gamma_deg.

v2.5 removed half of the macro-context subspace from EVERY query before computing
gamma_deg, including queries that name no context at all. v2.6 removes, per query,
only the directions of the context classes the query ACTIVATES: macro class m is
active when its unprojected class score S_m(q) exceeds the best generic
(out-of-vocabulary) class score. Variant, strength (0.5) and tau (0.02) were chosen
on the 252 training phrasings by the Milestone 22 leave-group-out protocol
(scripts/explore_conditional_gate_train.py). Everything else is v2.5.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from scene_mining.filter.sceneminer_v2_5 import SceneMinerV25


class SceneMinerV26(SceneMinerV25):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        d = self.W_route.in_features
        self.register_buffer("macro_means", torch.zeros(7, d))
        self.register_buffer("macro_ids", torch.zeros(7, dtype=torch.long))
        self.register_buffer("oov_ids", torch.zeros(1, dtype=torch.long))

    def set_bank(self, E_bank, cls, is_concept, mean_bank, names, tau=0.02):
        out = super().set_bank(E_bank, cls, is_concept, mean_bank, names, tau=tau)
        B = self.bank; c = torch.as_tensor(cls, device=B.device)
        mids = [k for k, n in enumerate(names) if n.startswith("macro:")]
        self.macro_ids = torch.tensor(mids, device=B.device)
        self.oov_ids = torch.tensor([k for k, n in enumerate(names) if n.startswith("oov:")], device=B.device)
        self.macro_means = torch.stack([B[c == k].mean(0) for k in mids]).detach().clone()
        return out

    def active_context(self, qc):
        S = self._class_scores(qc, self.bank)                       # unprojected
        ref = S[:, self.oov_ids].max(-1, keepdim=True).values
        return S[:, self.macro_ids] > ref                            # (B, 7) bool

    def concept(self, q_text):
        qc = self._centre(q_text)
        act = self.active_context(qc)
        S = torch.empty(qc.shape[0], self.n_cls, device=qc.device)
        act_c = act.cpu()                                            # one host sync per call
        groups = {}
        for i, row in enumerate(act_c.tolist()): groups.setdefault(tuple(row), []).append(i)
        for pat, rows in groups.items():
            rows = torch.as_tensor(rows, device=qc.device)
            if not any(pat):
                S[rows] = self._class_scores(qc[rows], self.bank)
                continue
            Q, _ = torch.linalg.qr(self.macro_means[torch.as_tensor(pat, device=qc.device)].T)
            pr = lambda X: F.normalize(X - self.lam_proj * (X @ Q) @ Q.T, dim=-1)
            S[rows] = self._class_scores(pr(qc[rows]), pr(self.bank))
        gamma = torch.softmax(S / self.tau_g, -1)[:, :4].sum(-1)
        a = torch.softmax(S[:, :4] / self.T_a, -1)
        return a, gamma, S[:, :4]

    def _load_from_state_dict(self, sd, prefix, *args, **kw):
        for k in ("macro_ids", "oov_ids"):
            if prefix + k in sd: setattr(self, k, torch.empty_like(sd[prefix + k]))
        super()._load_from_state_dict(sd, prefix, *args, **kw)
