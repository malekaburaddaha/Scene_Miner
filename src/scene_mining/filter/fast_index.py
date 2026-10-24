"""Cached scene index for SceneMiner (Milestone 27).

`SceneMinerV25.forward` takes the query expanded to one row per scene, so every
query-only quantity - the concept-bank posteriors, channel weights, routing, the
query adapter, the FiLM gamma/beta and the residual gate - is recomputed for all
1508 rows, and the six scene adapters re-run on every query although they do not
depend on it. Here the scene-only work is done once when the index is built and the
query-only work once per query; only FiLM modulation, the 2-layer decoder over the
6 scene tokens and the scorer run per scene. The result equals `forward` up to
floating-point reordering (verified by scripts/verify_fast_index.py).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from scene_mining.filter.sceneminer_v2_2 import TOK_3D


@torch.no_grad()
def precompute_scene_index(m, e, t, r, v, s_img, marg_z, prior):
    toks = torch.stack([m.adapt_embed(e), m.adapt_tags(t), m.adapt_risk(r), m.adapt_qwen(v),
                        m.adapt_siglip(s_img), m.adapt_margin(marg_z)], dim=1) + m.type_embed.unsqueeze(0)
    idx = {"toks": toks, "sn": F.normalize(s_img, dim=-1), "mz": marg_z, "prior": prior, "N": toks.shape[0],
           "pad3d": torch.tensor(TOK_3D, device=toks.device).unsqueeze(0).expand(toks.shape[0], -1).contiguous()}
    if hasattr(m, "macro_cols"):
        cols = m.macro_cols.clamp(min=0)
        idx["zmeta"] = (prior[:, cols] - m.meta_mu[cols]) / m.meta_sd[cols]           # (N, n_cls)
        idx["meta_present"] = (prior[:, 20:26].abs().sum(-1) > 0).float()
    return idx


def _decoder(m, qa, toks, pad):
    """nn.TransformerDecoder (pre-norm, ReLU, eval mode) for a length-1 target that is
    the same query for every scene: layer 1's self-attention is computed once."""
    N, T, d = toks.shape
    x = qa.view(1, 1, d)
    for L in m.fusion.layers:
        H = L.multihead_attn.num_heads; dh = d // H
        h = L.norm1(x)                                           # self-attention over one token
        Wv, bv = L.self_attn.in_proj_weight[2 * d:], L.self_attn.in_proj_bias[2 * d:]
        x = x + L.self_attn.out_proj(F.linear(h, Wv, bv))
        h = L.norm2(x)
        Wq, Wk, Wv = L.multihead_attn.in_proj_weight.split(d)
        bq, bk, bv = L.multihead_attn.in_proj_bias.split(d)
        qh = F.linear(h, Wq, bq).view(x.shape[0], 1, H, dh).transpose(1, 2)        # (1|N, H, 1, dh)
        kh = F.linear(toks, Wk, bk).view(N, T, H, dh).transpose(1, 2)               # (N, H, T, dh)
        vh = F.linear(toks, Wv, bv).view(N, T, H, dh).transpose(1, 2)
        att = (qh @ kh.transpose(-1, -2)) / dh ** 0.5                               # (N, H, 1, T)
        if pad is not None: att = att.masked_fill(pad.view(N, 1, 1, T), float("-inf"))
        o = (att.softmax(-1) @ vh).transpose(1, 2).reshape(N, 1, d)
        x = x + L.multihead_attn.out_proj(o)
        x = x + L.linear2(L.activation(L.linear1(L.norm3(x))))
    return x[:, 0]


def _decoder_folded(m, qa, toks, pad):
    """The same decoder with the key and value projections folded into the query side:
    q.(W_k z + b_k) = (W_k^T q).z + q.b_k, and since attention weights sum to one,
    sum_t a_t (W_v z_t + b_v) = W_v (sum_t a_t z_t) + b_v. The 9048 scene tokens are then
    only dotted with 256-d vectors instead of being projected by 256x256 matrices."""
    N, T, d = toks.shape
    x = qa.view(1, d)
    for L in m.fusion.layers:
        H = L.multihead_attn.num_heads; dh = d // H
        h = L.norm1(x)
        Wv_s, bv_s = L.self_attn.in_proj_weight[2 * d:], L.self_attn.in_proj_bias[2 * d:]
        x = x + L.self_attn.out_proj(F.linear(h, Wv_s, bv_s))
        h = L.norm2(x)
        Wq, Wk, Wv = L.multihead_attn.in_proj_weight.split(d)
        bq, bk, bv = L.multihead_attn.in_proj_bias.split(d)
        qh = F.linear(h, Wq, bq).view(-1, H, dh)                                  # (1|N, H, dh)
        u = torch.einsum("bhe,hed->bhd", qh, Wk.view(H, dh, d))                   # W_k^T q per head
        c = (qh * bk.view(1, H, dh)).sum(-1)                                      # q . b_k
        u, c = u.expand(N, -1, -1), c.expand(N, -1)                               # layer 1: one query for all scenes
        att = (torch.einsum("ntd,nhd->nht", toks, u) + c.unsqueeze(-1)) / dh ** 0.5   # (N, H, T)
        if pad is not None: att = att.masked_fill(pad.view(N, 1, T), float("-inf"))
        agg = torch.einsum("nht,ntd->nhd", att.softmax(-1), toks)                 # (N, H, d)
        o = torch.einsum("nhd,hed->nhe", agg, Wv.view(H, dh, d)) + bv.view(1, H, dh)
        x = x + L.multihead_attn.out_proj(o.reshape(N, d))
        x = x + L.linear2(L.activation(L.linear1(L.norm3(x))))
    return x


@torch.no_grad()
def fast_score(m, idx, q_text, has_3d=1.0, gamma_skip=0.0, explicit_decoder=True):
    """As fast_score_impl, with the concept-bank gate evaluated once per query."""
    q = q_text.reshape(1, -1)
    cached = m.concept(q)
    m.concept = lambda x, _c=cached: _c                          # same query throughout this call
    try:
        return fast_score_impl(m, idx, q_text, has_3d, gamma_skip, explicit_decoder)
    finally:
        del m.concept


@torch.no_grad()
def fast_score_impl(m, idx, q_text, has_3d=1.0, gamma_skip=0.0, explicit_decoder=True):
    """Scores of all indexed scenes for one query (q_text: (d,) raw SigLIP2 text
    feature). has_3d: 1 = full mode, 0 = monocular (3D tokens masked, meta channel
    off). gamma_skip > 0 skips the learned head when gamma_deg < gamma_skip."""
    q = q_text.reshape(1, -1)
    N, dev = idx["N"], q.device
    # ---- base parts (v2.2 base_parts + v2.3 rescue), query-side once
    cos = idx["sn"] @ F.normalize(q, dim=-1)[0]
    wm, sgn = m.channel_weights(q)
    s_marg = torch.tanh(sgn[0, 0] * (idx["mz"] @ wm[0]))
    wp = torch.softmax(m.meta_select(q), dim=-1)
    s_meta = torch.tanh(idx["prior"] @ wp[0])
    parts = m.part_norm(torch.stack([cos, s_marg, s_meta], dim=-1))
    if m.use_rescue:
        strong = parts[:, 0] > m.rescue_sig
        parts = torch.stack([parts[:, 0], torch.where(strong, parts[:, 1].clamp(min=m.rescue_floor), parts[:, 1]),
                             parts[:, 2]], -1)
    avail = torch.tensor([[1.0, 1.0, float(has_3d)]], device=dev)
    w = m.route(q, avail)                                                      # (1, 3)
    base = (parts * (w * avail)).sum(-1)
    # ---- abstention target and gate
    _, g, _ = m.concept(q); g = g[0]
    target = parts[:, 0]
    if getattr(m, "use_macro_prior", False) and "zmeta" in idx:
        pm = m.macro_posterior(q)[0]                                           # (n_cls,)
        mix = (idx["zmeta"] @ pm) * idx["meta_present"]
        gm = pm.sum() * idx["meta_present"]
        target = (target + mix) / torch.sqrt(1.0 + gm ** 2)
    if not m.use_abstain:
        g = torch.ones((), device=dev)
    if gamma_skip > 0 and float(g) < gamma_skip:
        return target if m.use_abstain else base
    # ---- residual: FiLM on cached tokens, decoder, scorer
    qa = m.adapt_query(q)                                                      # (1, d)
    gb = m.film.to_gamma_beta(qa).view(2, m.film.n_tokens, m.film.d_model)
    toks = idx["toks"] * (1.0 + gb[0]) + gb[1]
    pad = None
    if not has_3d:
        pad = torch.tensor(TOK_3D, device=dev).unsqueeze(0).expand(N, -1)
    if explicit_decoder:
        c = _decoder_folded(m, qa, toks, pad)
    else:
        c = m.fusion(tgt=qa.expand(N, -1).unsqueeze(1), memory=toks, memory_key_padding_mask=pad)[:, 0]
    qn, cn = F.normalize(qa, dim=-1), F.normalize(c, dim=-1)
    s_bil = F.cosine_similarity(m.bil_q(qn), m.bil_c(cn), dim=-1)
    qn_e = qn.expand(N, -1)
    s_mlp = torch.tanh(m.mlp(torch.cat([qn_e, cn, qn_e * cn, (qn_e - cn).abs()], -1)).squeeze(-1))
    res = (s_bil + s_mlp) / m.tau
    if m.use_resnorm: res = m.alpha_res * m.res_norm(res)
    gate = torch.tanh(m.gate_head(q))[0, 0]
    s = base + gate * res
    return g * s + (1.0 - g) * target if m.use_abstain else s


@torch.no_grad()
def portable_index(m, siglip_img, margins_z, prior):
    idx = {"sn": F.normalize(siglip_img, dim=-1), "mz": margins_z, "prior": prior, "N": siglip_img.shape[0]}
    if hasattr(m, "macro_cols"):
        cols = m.macro_cols.clamp(min=0)
        idx["zmeta"] = (prior[:, cols] - m.meta_mu[cols]) / m.meta_sd[cols]
        idx["meta_present"] = (prior[:, 20:26].abs().sum(-1) > 0).float()
    return idx


@torch.no_grad()
def fast_base_score(m, siglip_img, margins_z, prior, q_text, avail_meta=0.0, idx=None):
    """Portable base score (no 3D rig, no residual), one query, query-side once.
    Equals m.base_score(q.expand(N), siglip_img, margins_z, prior, avail). Pass a
    prebuilt `idx` (portable_index) to skip the scene-side preparation."""
    idx = idx or portable_index(m, siglip_img, margins_z, prior)
    q = q_text.reshape(1, -1)
    cached = m.concept(q)                                     # the gate, once per query
    m.concept = lambda x, _c=cached: _c
    try:
        return _fast_base_impl(m, idx, q, avail_meta)
    finally:
        del m.concept


def _fast_base_impl(m, idx, q, avail_meta):
    cos = idx["sn"] @ F.normalize(q, dim=-1)[0]
    wm, sgn = m.channel_weights(q)
    s_marg = torch.tanh(sgn[0, 0] * (idx["mz"] @ wm[0]))
    wp = torch.softmax(m.meta_select(q), dim=-1)
    s_meta = torch.tanh(idx["prior"] @ wp[0])
    parts = m.part_norm(torch.stack([cos, s_marg, s_meta], dim=-1))
    if m.use_rescue:
        strong = parts[:, 0] > m.rescue_sig
        parts = torch.stack([parts[:, 0], torch.where(strong, parts[:, 1].clamp(min=m.rescue_floor), parts[:, 1]),
                             parts[:, 2]], -1)
    avail = torch.tensor([[1.0, 1.0, float(avail_meta)]], device=q.device)
    s = (parts * (m.route(q, avail) * avail)).sum(-1)
    if not m.use_abstain: return s
    _, g, _ = m.concept(q); g = g[0]
    target = parts[:, 0]
    if getattr(m, "use_macro_prior", False):
        pm = m.macro_posterior(q)[0]
        mix = (idx["zmeta"] @ pm) * idx["meta_present"]
        gm = pm.sum() * idx["meta_present"]
        target = (target + mix) / torch.sqrt(1.0 + gm ** 2)
    return g * s + (1.0 - g) * target


# ---------------------------------------------------------------- query preparation / ranking split
@torch.no_grad()
def prepare_query(m, q_text, has_3d=1.0):
    """Every query-only quantity, computed once per query (cost independent of the
    number of scenes, like text encoding)."""
    q = q_text.reshape(1, -1); dev = q.device
    cached = m.concept(q)
    m.concept = lambda x, _c=cached: _c
    try:
        wm, sgn = m.channel_weights(q)
        avail = torch.tensor([[1.0, 1.0, float(has_3d)]], device=dev)
        w = m.route(q, avail) * avail
        g = cached[1][0] if m.use_abstain else torch.ones((), device=dev)
        pm = m.macro_posterior(q)[0] if getattr(m, "use_macro_prior", False) else None
    finally:
        del m.concept
    qa = m.adapt_query(q)
    gb = m.film.to_gamma_beta(qa).view(2, m.film.n_tokens, m.film.d_model)
    return {"qn_sig": F.normalize(q, dim=-1)[0], "wm": wm[0], "sgn": sgn[0, 0], "wp": torch.softmax(m.meta_select(q), -1)[0],
            "w": w[0], "g": g, "pm": pm, "qa": qa, "gb": gb, "gate": torch.tanh(m.gate_head(q))[0, 0],
            "qn": F.normalize(qa, dim=-1), "has_3d": float(has_3d)}


@torch.no_grad()
def rank(m, idx, P):
    """Per-scene work only: base parts, FiLM on cached tokens, decoder, scorer, mix.
    rank(m, idx, prepare_query(m, q)) == fast_score(m, idx, q)."""
    N = idx["N"]
    cos = idx["sn"] @ P["qn_sig"]
    parts = m.part_norm(torch.stack([cos, torch.tanh(P["sgn"] * (idx["mz"] @ P["wm"])),
                                     torch.tanh(idx["prior"] @ P["wp"])], -1))
    if m.use_rescue:
        strong = parts[:, 0] > m.rescue_sig
        parts = torch.stack([parts[:, 0], torch.where(strong, parts[:, 1].clamp(min=m.rescue_floor), parts[:, 1]),
                             parts[:, 2]], -1)
    base = (parts * P["w"]).sum(-1)
    target = parts[:, 0]
    if P["pm"] is not None and "zmeta" in idx:
        target = (target + (idx["zmeta"] @ P["pm"]) * idx["meta_present"]) / \
                 torch.sqrt(1.0 + (P["pm"].sum() * idx["meta_present"]) ** 2)
    toks = idx["toks"] * (1.0 + P["gb"][0]) + P["gb"][1]
    pad = None if P["has_3d"] else idx["pad3d"]
    c = _decoder_folded(m, P["qa"], toks, pad)
    cn = F.normalize(c, dim=-1); qn = P["qn"]; qe = qn.expand(N, -1)
    res = (F.cosine_similarity(m.bil_q(qn), m.bil_c(cn), dim=-1)
           + torch.tanh(m.mlp(torch.cat([qe, cn, qe * cn, (qe - cn).abs()], -1)).squeeze(-1))) / m.tau
    if m.use_resnorm: res = m.alpha_res * m.res_norm(res)
    s = base + P["gate"] * res
    return P["g"] * s + (1.0 - P["g"]) * target if m.use_abstain else s


class GraphRanker:
    """CUDA-graph replay of `rank` for a fixed index (static shapes): removes the
    per-kernel launch overhead, which dominates at this size. Numerically identical
    to `rank` (same kernels, replayed)."""
    def __init__(self, m, idx, P_example):
        self.m, self.idx = m, idx
        self.P = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_example.items()}
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): rank(m, idx, self.P)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = rank(m, idx, self.P)

    def __call__(self, P):
        for k, v in P.items():
            if torch.is_tensor(v): self.P[k].copy_(v)
        self.g.replay()
        return self.out
