#!/usr/bin/env python3
"""MILESTONE 2 - train SceneMiner v2.

Three fixes to v1's supervision, each traceable to a measured failure:
  1. all four degradation attributes are supervised (v1 never saw low_contrast,
     which scored 0.089 mAP against human labels);
  2. targets are MARGIN-WEIGHTED, not hard 0/1: a scene whose VLM margin is
     near zero contributes less, so the model stops memorising the VLM's false
     positives (the VLM's precision against human labels is 0.50-0.68);
  3. the head starts as the zero-shot base (SigLIP2 + VLM margin) via a
     zero-initialised gate, so training can only add to a strong prior.
"""
from __future__ import annotations
from paths import DEVICE
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

ATTRS = ["droplets", "glare", "wet_road", "low_contrast"]
PHRASINGS = {
 "droplets": ["water droplets smearing the camera lens",
              "rain beading on the lens obscuring the view",
              "a drizzle leaving light droplets on the lens",
              "heavy rain spatter across the camera"],
 "glare": ["blinding headlight bloom from oncoming traffic",
           "specular glare washing out the image",
           "mild light flare at night",
           "intense light bloom reflecting off the road"],
 "wet_road": ["wet reflective asphalt after rainfall",
              "standing water on the road surface",
              "a damp road surface",
              "slick wet pavement"],
 "low_contrast": ["a low contrast scene where vehicles are barely distinguishable",
                  "hazy washed out view with unclear object boundaries",
                  "poor visibility with objects blending into the background",
                  "murky scene with indistinct vehicle outlines"],
}
MACRO = [({"lighting":"night"}, "at night"), ({"lighting":"day"}, "in daylight"),
         ({"weather":"rain"}, "in rain"), ({"density":"dense"}, "in dense traffic")]

def build_queries(attrs, toks, dom, train_mask, margins, attr_idx, min_pos=6):
    """(attribute, optional macro context) -> query with margin-derived weights."""
    qs = []
    for at in ATTRS:
        j = attr_idx[at]
        conf = np.tanh(np.abs(margins[:, j]) / 1.0)     # 0..1, low for borderline
        pos_all = np.array([attrs.get(t, {}).get(at) is True for t in toks])
        neg_all = np.array([attrs.get(t, {}).get(at) is False for t in toks])
        for ph in PHRASINGS[at]:
            for mctx, mph in [({}, "")] + MACRO:
                sel = (np.array([all(attrs.get(t, {}).get(k) == v for k, v in mctx.items())
                                 for t in toks]) if mctx else np.ones(len(toks), bool))
                pos = pos_all & sel & train_mask
                neg = neg_all & sel & train_mask
                if pos.sum() < min_pos or neg.sum() < min_pos: continue
                qs.append({"text": (ph + (" " + mph if mph else "")).strip(),
                           "attr": at,
                           "pos_idx": np.where(pos)[0], "cond_idx": np.where(neg)[0],
                           "pos_w": conf[np.where(pos)[0]], "cond_w": conf[np.where(neg)[0]]})
    return qs

def balanced_draw(pool, w, dom, k, rng):
    d0 = pool[dom[pool] == 0]; d1 = pool[dom[pool] == 1]
    h = k // 2; t0 = min(h, len(d0)); t1 = min(k - t0, len(d1))
    if t0 + t1 < k:
        extra = k - t0 - t1
        if len(d0) > t0: t0 += min(extra, len(d0) - t0)
        elif len(d1) > t1: t1 += min(extra, len(d1) - t1)
    out = []
    if t0: out += list(rng.choice(d0, t0, replace=False))
    if t1: out += list(rng.choice(d1, t1, replace=False))
    while len(out) < k and len(pool): out.append(int(rng.choice(pool)))
    return np.array(out[:k], dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--split-file", default="benchmark_cvpr/split_906_602.json")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--queries-per-step", type=int, default=4)
    ap.add_argument("--pos-per-query", type=int, default=2)
    ap.add_argument("--cond-per-query", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-cond", type=float, default=4.0)
    ap.add_argument("--cond-margin", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-human", default=None,
                    help="optional human-label json to add as calibration queries")
    ap.add_argument("--human-tokens", default=None,
                    help="json list of tokens allowed from --extra-human (fold train)")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(DEVICE)
    W = Path(a.work)
    z = np.load(a.features, allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]; dom = z["domain"].astype(int)
    emb = torch.from_numpy(z["emb"]).float().to(dev)
    tlog = torch.from_numpy(z["tag_logits"]).float().to(dev)
    risk = torch.from_numpy(z["risk"]).float().to(dev)
    qwen = torch.zeros(len(toks), 2048, device=dev)      # v1 visual token cache
    vcache = W/"V_frozen_cache.pt"
    if vcache.exists(): qwen = torch.load(vcache, map_location=dev).float()
    sig = torch.from_numpy(z["siglip_img"]).float().to(dev)
    mz  = torch.from_numpy(z["vlm_margin_z"]).float().to(dev)
    margins = z["vlm_margin"]; attr_idx = {a_: i for i, a_ in enumerate(list(z["margin_attrs"]))}

    from eval_attribute_index import load_attributes
    attrs = load_attributes(W)
    for f in ("binary_flags_with_logits.json", "lowcontrast_strict.json"):
        p = W/f
        if p.exists():
            for t, r in json.load(open(p)).items():
                rec = {k: v for k, v in r.items() if not k.endswith(("_margin","_logp_yes"))}
                if "low_contrast_strict" in rec: rec["low_contrast"] = rec.pop("low_contrast_strict")
                attrs.setdefault(t, {}).update(rec)
    train = set(json.load(open(a.split_file))["train"])
    tmask = np.array([t in train for t in toks])

    # optional human-calibration queries (fold-train tokens only)
    if a.extra_human and Path(a.extra_human).exists():
        hum = json.loads(Path(a.extra_human).read_text())
        allow = set(json.loads(Path(a.human_tokens).read_text())) if a.human_tokens else set(hum)
        nover = 0
        for t, rec in hum.items():
            if t not in allow or rec.get("flag") == "unsure": continue
            if t in attrs:
                attrs[t].update({k: bool(v) for k, v in rec["labels"].items()}); nover += 1
        tmask = tmask | np.array([t in allow for t in toks])
        print(f"[v2] human calibration: {nover} scenes overridden with human labels")

    queries = build_queries(attrs, toks, dom, tmask, margins, attr_idx)
    print(f"[v2] {len(queries)} training queries over {len(ATTRS)} attributes "
          f"({tmask.sum()} trainable scenes)")

    # SigLIP2 text tower for both the base and the residual query token
    from transformers import AutoModel, GemmaTokenizerFast
    P = "google/siglip2-base-patch16-224"
    sm_ = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    tk = GemmaTokenizerFast.from_pretrained(P)
    feat = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    with torch.no_grad():
        ti = tk([q["text"] for q in queries], padding="max_length", truncation=True,
                max_length=64, return_tensors="pt").to(dev)
        QT = feat(sm_.get_text_features(**ti))
    del sm_; torch.cuda.empty_cache()

    from scene_mining.filter.sceneminer_v2 import SceneMinerV2
    rk = risk.reshape(-1).cpu().numpy()
    model = SceneMinerV2(d_qwen=qwen.shape[1], d_siglip=sig.shape[1],
                         n_margins=mz.shape[1], d_text=QT.shape[1],
                         risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.SequentialLR(opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.02, total_iters=a.warmup_steps),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.steps-a.warmup_steps), eta_min=1e-7)],
        milestones=[a.warmup_steps])
    rng = np.random.default_rng(a.seed)
    nq, npos, ncond = a.queries_per_step, a.pos_per_query, a.cond_per_query
    t0 = time.time(); hist = []
    for step in range(1, a.steps+1):
        qsel = rng.choice(len(queries), min(nq, len(queries)), replace=False)
        idx, per_q = [], []
        for qi in qsel:
            q = queries[qi]
            p = balanced_draw(q["pos_idx"], q["pos_w"], dom, npos, rng)
            c = balanced_draw(q["cond_idx"], q["cond_w"], dom, ncond, rng)
            per_q.append((qi, p, c)); idx += list(p) + list(c)
        idx = np.array(idx); S = len(idx)
        rel = np.zeros((len(qsel), S), bool); cnd = np.zeros((len(qsel), S), bool)
        wgt = np.ones((len(qsel), S), dtype=np.float32)
        for r_, (qi, p, c) in enumerate(per_q):
            q = queries[qi]
            pset = {int(v): w for v, w in zip(q["pos_idx"], q["pos_w"])}
            cset = {int(v): w for v, w in zip(q["cond_idx"], q["cond_w"])}
            for j, s_ in enumerate(idx):
                s_ = int(s_)
                if s_ in pset: rel[r_, j] = True; wgt[r_, j] = pset[s_]
                elif s_ in cset: cnd[r_, j] = True; wgt[r_, j] = cset[s_]
        qt = QT[qsel]
        qrep = qt.unsqueeze(1).expand(-1, S, -1).reshape(-1, qt.shape[-1])
        gi = torch.from_numpy(idx).long().to(dev)
        rep = lambda X: X[gi].repeat(len(qsel), 1)
        s, _ = model(qrep, qrep, rep(emb), rep(tlog), rep(risk), rep(qwen), rep(sig), rep(mz))
        s = s.view(len(qsel), S)
        m = torch.from_numpy(rel).to(dev); c_ = torch.from_numpy(cnd).to(dev)
        wt = torch.from_numpy(wgt).to(dev)
        hp = m.any(1)
        if not hp.any(): sched.step(); continue
        sp, mp, cp, wp = s[hp], m[hp], c_[hp], wt[hp]
        # margin-WEIGHTED listwise InfoNCE: confident positives dominate
        num = torch.logsumexp(sp.masked_fill(~mp, -1e4) + torch.log(wp + 1e-6), 1)
        den = torch.logsumexp(sp, 1)
        nce = (den - num).mean()
        worst = sp.masked_fill(~mp, 1e4).min(1).values
        cond = sp.sum() * 0.0
        hc = cp.any(1)
        if hc.any():
            best = sp.masked_fill(~cp, -1e4).max(1).values
            cond = a.w_cond * F.relu(a.cond_margin - (worst - best))[hc].mean()
        loss = nce + cond
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            g = float(torch.tanh(model.gate)); b = float(torch.sigmoid(model.base_blend))
            hist.append({"step": step, "loss": float(loss), "gate": g, "blend": b})
            print(f"[s{step}] loss={float(loss):.4f} nce={float(nce):.4f} "
                  f"cond={float(cond):.4f} gate={g:+.4f} blend={b:.3f}", flush=True)
    wd = Path(a.work_dir); wd.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "args": vars(a), "hist": hist, "n_queries": len(queries)}, wd/"final.pt")
    print(f"[done] {a.steps} steps in {(time.time()-t0)/60:.1f} min -> {wd}")

if __name__ == "__main__":
    main()
