#!/usr/bin/env python3
"""MILESTONE 23 - two trained baselines, same supervision as SceneMiner.

  bev    BEV-TSR-style text-to-BEV retrieval [Tang et al., AAAI 2025]: an MLP maps
         the BEVFormer scene embedding e and risk r into the SigLIP2 text space and
         is trained contrastively against the query text. (BEV-TSR also uses
         captions and an LLM; this re-implements its core alignment step on our
         cached BEV features.)
  sigft  SigLIP2 with a fine-tuned image-side residual adapter (zero-init linear).

Both use the 72 training queries, the 906 training scenes and the VLM-derived
pseudo-labels that SceneMiner trains on, listwise InfoNCE with the same
margin-confidence weights, 3000 steps, AdamW 1e-4, 5 seeds.
"""
from paths import DEVICE
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from train_sceneminer_v24 import build_queries, balanced_draw, ATTRS
from eval_attribute_index import load_attributes


class BEV(nn.Module):
    def __init__(self, d_in=257, d_out=768):
        super().__init__(); self.f = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, d_out))
        self.log_t = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
    def forward(self, x): return F.normalize(self.f(x), dim=-1)


class SigFT(nn.Module):
    def __init__(self, d=768):
        super().__init__(); self.A = nn.Linear(d, d); nn.init.zeros_(self.A.weight); nn.init.zeros_(self.A.bias)
        self.log_t = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
    def forward(self, x): return F.normalize(x + self.A(x), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["bev", "sigft"], required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device(DEVICE)
    W = Path("work_dirs/unified_filter")
    z = np.load(W/"features_v2.npz", allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]; dom = z["domain"].astype(int)
    attrs = load_attributes(W)
    for f in ("binary_flags_with_logits.json", "lowcontrast_strict.json"):
        for t, r in json.load(open(W/f)).items():
            rec = {k: v for k, v in r.items() if not k.endswith(("_margin", "_logp_yes"))}
            if "low_contrast_strict" in rec: rec["low_contrast"] = rec.pop("low_contrast_strict")
            attrs.setdefault(t, {}).update(rec)
    train = set(json.load(open("benchmark_cvpr/split_906_602.json"))["train"])
    tmask = np.array([t in train for t in toks])
    margins = z["vlm_margin"]; attr_idx = {a_: i for i, a_ in enumerate(list(z["margin_attrs"]))}
    Q = build_queries(attrs, toks, dom, tmask, margins, attr_idx)
    from eval_semantic_edge_cases import text_embed
    QT = F.normalize(text_embed([q["text"] for q in Q]).to(dev), dim=-1)
    if a.kind == "bev":
        X = torch.from_numpy(np.concatenate([z["emb"], (z["risk"] - z["risk"].mean()) / z["risk"].std()], 1)).float().to(dev)
        m = BEV(X.shape[1]).to(dev)
    else:
        X = F.normalize(torch.from_numpy(z["siglip_img"]).float().to(dev), dim=-1); m = SigFT().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
    for step in range(a.steps):
        qs = rng.choice(len(Q), 4, replace=False); idx, rel, wt = [], [], []
        for qi in qs:
            q = Q[qi]
            p = balanced_draw(q["pos_idx"], q["pos_w"], dom, 2, rng); c = balanced_draw(q["cond_idx"], q["cond_w"], dom, 2, rng)
            idx += list(p) + list(c)
        idx = np.array(idx)
        R = np.zeros((4, len(idx)), bool); Wt = np.ones((4, len(idx)), np.float32)
        for r_, qi in enumerate(qs):
            pw = dict(zip(map(int, Q[qi]["pos_idx"]), Q[qi]["pos_w"]))
            for j, s_ in enumerate(idx):
                if int(s_) in pw: R[r_, j] = True; Wt[r_, j] = pw[int(s_)]
        emb = m(X[torch.from_numpy(idx).to(dev)])
        logits = m.log_t.exp() * (QT[qs] @ emb.T)
        Rm = torch.from_numpy(R).to(dev); Wm = torch.from_numpy(Wt).to(dev)
        num = torch.logsumexp(logits.masked_fill(~Rm, -1e4) + torch.log(Wm + 1e-6), 1)
        loss = (torch.logsumexp(logits, 1) - num).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    out = Path(f"work_dirs/baseline_{a.kind}/s{a.seed}"); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "kind": a.kind}, out/"final.pt")
    print(f"[{a.kind} s{a.seed}] final loss {float(loss):.4f}")


if __name__ == "__main__":
    main()
