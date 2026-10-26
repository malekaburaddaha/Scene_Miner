#!/usr/bin/env python3
"""TASK 4.1 / 4.3 - the collapse diagnostic protocol, as a standalone tool.

Reports, for any fused representation:
  erank(X)                spectral effective rank (Roy-Vetterli entropy)
  erank(L)                effective rank of the SUPERVISION, multi-hot over the
                          underlying binary attributes, mean-centred as X is
  erank(X)/erank(L)       the ratio that distinguishes genuine collapse from a
                          false alarm against an unreachable isotropic null
  routing mass            cross-attention mass per modality token
  distinct rows           per-token duplicate count (catches a dead channel)
  score span              max-min of the score field
Applies unchanged to (a) our filter head and (b) a BLIP-2 Q-Former index.
"""
from __future__ import annotations
from paths import QWEN_VL
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

def erank(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > eps]
    if s.size == 0: return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + eps)).sum()))

def label_matrix(attrs, toks, keys):
    """Multi-hot over the underlying binary attributes (NOT one-hot over
    configurations, which would overstate the intrinsic dimension)."""
    rows, cols = [], []
    for k in keys:
        vals = sorted({attrs.get(t, {}).get(k) for t in toks} - {None})
        if len(vals) <= 1: continue
        if len(vals) == 2 and set(vals) <= {True, False}:
            cols.append((k, True))
        else:
            cols += [(k, v) for v in vals]
    for t in toks:
        rows.append([1.0 if attrs.get(t, {}).get(k) == v else 0.0 for k, v in cols])
    return np.array(rows), cols


def _capture_xattn(core, qc, emb, tlog, risk, V):
    """Average cross-attention mass per modality token, read from the decoder's
    real multihead_attn. Each TransformerDecoderLayer calls multihead_attn with
    need_weights=False, so we wrap it to force weight return and record them."""
    import torch, torch.nn.functional as F
    grabbed = []
    patched = []
    for layer in core.fusion.layers:
        mha = layer.multihead_attn
        orig = mha.forward
        def wrap(*args, _o=orig, **kw):
            kw["need_weights"] = True
            kw["average_attn_weights"] = True
            out, w = _o(*args, **kw)
            if w is not None: grabbed.append(w.detach().float())
            return out, w
        mha.forward = wrap
        patched.append((mha, orig))
    try:
        qd = core.adapt_query(qc)
        toks_m = core.film(core.scene_tokens(emb, tlog, risk, V), qd)
        _ = core.fusion(tgt=qd.unsqueeze(1), memory=toks_m)
    finally:
        for mha, orig in patched: mha.forward = orig
    if not grabbed:
        return np.full(4, np.nan)
    # (B, 1, n_mem) per layer -> mean over layers, batch, and the single query
    W = torch.stack([g.mean(0).mean(0) for g in grabbed], 0).mean(0)
    return W.cpu().numpy()


def probe_gradients(core, qc_fn, emb, tlog, risk, V, attrs, toks, attributes):
    """Per-attribute probe gradient norm ||dL/dtheta||. For each attribute we
    build anchor/positive/conditional-negative triplets that agree on every
    other attribute and differ only on the target, then measure the gradient the
    objective actually delivers. A vanishing norm = an attribute the objective
    has stopped learning, regardless of its score."""
    import torch, torch.nn.functional as F
    out = {}
    params = [p for p in core.parameters() if p.requires_grad]
    for attr in attributes:
        pos = np.array([attrs.get(t, {}).get(attr) is True for t in toks])
        neg = np.array([attrs.get(t, {}).get(attr) is False for t in toks])
        if pos.sum() < 2 or neg.sum() < 2:
            out[attr] = float("nan"); continue
        pi = np.where(pos)[0][:8]; ni = np.where(neg)[0][:8]
        idx = np.concatenate([pi, ni])
        q = qc_fn(f"driving scene with {attr.replace('_',' ')}")
        qq = q.expand(len(idx), -1)
        s = core.score(qq, emb[idx], tlog[idx], risk[idx], V[idx])
        m = torch.zeros(len(idx), dtype=torch.bool, device=s.device); m[:len(pi)] = True
        loss = (torch.logsumexp(s.unsqueeze(0), 1)
                - torch.logsumexp(s.masked_fill(~m, -1e4).unsqueeze(0), 1)).mean()
        core.zero_grad(set_to_none=True)
        g = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        n = sum(float((x.detach()**2).sum()) for x in g if x is not None) ** 0.5
        out[attr] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--legacy-scoring", action="store_true")
    ap.add_argument("--legacy-scalar", action="store_true")
    ap.add_argument("--force-layernorm1d", action="store_true")
    ap.add_argument("--freeze-visual", action="store_true")
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--lora-r", type=int, default=1)
    ap.add_argument("--qwen", default=QWEN_VL)
    ap.add_argument("--img-side", type=int, default=336)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from scene_mining.filter.lora_grounded_filter import (LoRAGroundedFilter,
                                                          QwenVisualEncoder)
    from eval_attribute_index import load_attributes, MACRO8
    import torch.nn.functional as F
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = Path(a.work)
    z = np.load(W / "features_overnight.npz", allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]
    emb = torch.from_numpy(z["emb"]).float().to(dev)
    tlog = torch.from_numpy(z["tag_logits"]).float().to(dev)
    risk = torch.from_numpy(z["risk"]).float().to(dev)
    rk = risk.reshape(-1).cpu().numpy()
    attrs = load_attributes(W)
    # prefer the 4-attribute extraction (adds low_contrast + logit margins)
    _nf = W / "binary_flags_with_logits.json"
    if _nf.exists():
        for _t, _r in json.load(open(_nf)).items():
            attrs.setdefault(_t, {}).update(
                {k: v for k, v in _r.items() if not k.endswith(("_margin", "_logp_yes"))})

    enc = QwenVisualEncoder(a.qwen, lora_r=a.lora_r, lora_alpha=128,
                            device=str(dev), img_max_side=a.img_side)
    enc.model.to(dev)
    model = LoRAGroundedFilter(enc, tau=a.tau, legacy_scoring=a.legacy_scoring,
                               legacy_scalar=a.legacy_scalar,
                               force_layernorm1d=a.force_layernorm1d,
                               risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(ck["trainable_state_dict"], strict=False)
    model.eval()

    from PIL import Image
    V = torch.zeros(len(toks), enc.d_model, device=dev)
    with torch.no_grad():
        for i in range(0, len(toks), a.batch):
            ims = [Image.open(Path(a.thumbs) / f"{t}.jpg").convert("RGB")
                   for t in toks[i:i + a.batch]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                V[i:i + a.batch] = enc.encode(ims)

    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
    ctok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    cenc = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True).to(dev).eval()

    res = {"label": a.label, "checkpoint": a.checkpoint, "n_scenes": len(toks)}
    with torch.no_grad():
        # --- per-token statistics (dead-channel detection) ---
        core = model
        toks_t = core.scene_tokens(emb, tlog, risk, V)        # (N, 4, d)
        names = ["embedding", "tags", "risk", "visual"]
        res["tokens"] = {}
        for i, nm in enumerate(names):
            X = toks_t[:, i].float().cpu().numpy()
            res["tokens"][nm] = {
                "distinct_rows": int(len(np.unique(np.round(X, 6), axis=0))),
                "per_dim_std_mean": float(X.std(0).mean()),
                "erank": erank(X)}
        # --- fused representation + routing, averaged over the query suite ---
        fused_all, routing, spans = [], [], []
        for q, _ in MACRO8:
            tt = ctok([q], padding=True, truncation=True, return_tensors="pt").to(dev)
            qc = F.normalize(cenc(**tt).text_embeds.float(), dim=-1).expand(len(toks), -1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                s, fused = core.score(qc, emb, tlog, risk, V, return_fused=True)
            fused_all.append(fused.float().cpu().numpy())
            spans.append(float(s.float().max() - s.float().min()))
            # TRUE routing mass: the decoder's own cross-attention weights,
            # captured by forcing need_weights=True on multihead_attn and
            # averaging over heads and queries (NOT a cosine proxy).
            routing.append(_capture_xattn(core, qc, emb, tlog, risk, V))
        Xf = np.concatenate(fused_all, 0)
        res["fused"] = {"erank": erank(Xf), "dim": int(Xf.shape[1])}
        res["routing_mass"] = {nm: float(v) for nm, v in zip(names, np.mean(routing, 0))}
        res["score_span"] = float(np.mean(spans))

    # --- per-attribute probe gradient audit ---
    def _qfn(text):
        tt = ctok([text], padding=True, truncation=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            return F.normalize(cenc(**tt).text_embeds.float(), dim=-1)
    for p_ in model.parameters(): p_.requires_grad_(True)
    res["probe_gradients"] = probe_gradients(
        model, _qfn, emb, tlog, risk, V, attrs, toks,
        ["droplets", "glare", "wet_road", "low_contrast"])
    GF = 1e-8
    res["probes_below_gradient_floor"] = int(sum(
        1 for v in res["probe_gradients"].values()
        if v == v and v < GF))

    # --- label-entropy null ---
    keys = ["lighting", "weather", "density", "droplets", "glare", "wet_road"]
    L, cols = label_matrix(attrs, toks, keys)
    res["label_null"] = {"erank_L": erank(L), "n_label_cols": len(cols),
                         "n_distinct_configs": int(len(np.unique(L, axis=0)))}
    res["erank_ratio"] = res["fused"]["erank"] / max(1e-9, res["label_null"]["erank_L"])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\n===== DIAGNOSTIC: {a.label} =====")
    for nm in names:
        t = res["tokens"][nm]
        print(f"  token {nm:<10} distinct_rows={t['distinct_rows']:5d}/{len(toks)}  "
              f"erank={t['erank']:7.2f}  per-dim std={t['per_dim_std_mean']:.4f}")
    print(f"  fused erank      = {res['fused']['erank']:.2f} / {res['fused']['dim']}")
    print(f"  erank(L)         = {res['label_null']['erank_L']:.2f} "
          f"({res['label_null']['n_label_cols']} cols, "
          f"{res['label_null']['n_distinct_configs']} configs)")
    print(f"  erank(X)/erank(L)= {res['erank_ratio']:.2f}")
    print(f"  routing mass     = " + "  ".join(f"{k}:{v:.3f}" for k, v in res["routing_mass"].items()))
    print(f"  score span       = {res['score_span']:.2f}")
    pg = res.get("probe_gradients", {})
    print("  probe grad norms = " + "  ".join(f"{k}:{v:.3e}" for k, v in pg.items()))
    print(f"  probes below 1e-8 floor = {res.get('probes_below_gradient_floor')}")
    print(f"[diag] wrote {a.out}")

if __name__ == "__main__":
    main()
