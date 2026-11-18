#!/usr/bin/env python3
"""MILESTONE 41 - E2: build the training-free control (src/scene_mining/filter/sceneminer_tf.py).

Starts from a freshly initialised SceneMiner v2.6 (every learned parameter at its
zero-learning initial value: router and selectors at zero, gate head at zero, margin
sign at tanh(2)), copies from the trained seed-0 checkpoint ONLY the buffers built
from training text or training-scene statistics (concept bank, tau, context basis,
metadata mean/sd, text mean, attribute prompts, risk statistics), and recomputes the
PartNorm statistics of the control's own three parts, without labels, over the 906
training scenes x the 72 Stage-2 training queries (all pairs, equal weight).
Writes five identical copies (s0..s4) so the unchanged evaluation scripts, which expect
five seeds, score it; the control is deterministic (sd = 0).
  --prior bank     -> work_dirs/sceneminer_v26tf/      (E2 row)
  --prior uniform  -> work_dirs/sceneminer_v26tfu/     (check: query-independent prior)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from paths import DEVICE
from scene_mining.filter.sceneminer_tf import SceneMinerTrainingFree

dev = torch.device(DEVICE)
COPY_BUFFERS = ("T_a", "T_c", "adapt_risk.net.0.mu", "adapt_risk.net.0.sigma", "attr_text", "bank", "bank_count",
                "bank_onehot", "bank_proj", "concept_text", "macro_basis", "macro_cols", "macro_ids", "macro_means",
                "meta_mu", "meta_sd", "oov_ids", "tau", "tau_c", "tau_g", "tau_route", "tau_sel", "text_mean")


def training_queries():
    """The 72 Stage-2 training queries, built exactly as scripts/train_sceneminer_v26.py does."""
    from train_sceneminer_v26 import build_queries
    from eval_attribute_index import load_attributes
    W = Path("work_dirs/unified_filter")
    z = np.load(W/"features_v2.npz", allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]; dom = z["domain"].astype(int)
    attrs = load_attributes(W)
    for f in ("binary_flags_with_logits.json", "lowcontrast_strict.json"):
        p = W/f
        if p.exists():
            for t, r in json.load(open(p)).items():
                rec = {k: v for k, v in r.items() if not k.endswith(("_margin", "_logp_yes"))}
                if "low_contrast_strict" in rec: rec["low_contrast"] = rec.pop("low_contrast_strict")
                attrs.setdefault(t, {}).update(rec)
    train = set(json.load(open("benchmark_cvpr/split_906_602.json"))["train"])
    tmask = np.array([t in train for t in toks])
    attr_idx = {a: i for i, a in enumerate(list(z["margin_attrs"]))}
    qs = build_queries(attrs, toks, dom, tmask, z["vlm_margin"], attr_idx)
    return [q["text"] for q in qs], tmask, z


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prior", choices=["bank", "uniform"], default="bank")
    a = ap.parse_args()
    texts, tmask, z = training_queries()
    print(f"[E2] {len(texts)} training queries, {int(tmask.sum())} training scenes", flush=True)
    from eval_semantic_edge_cases import text_embed
    QT = text_embed(texts).to(dev)
    rk = z["risk"].reshape(-1)
    d = torch.load("work_dirs/sceneminer_v26/s0/final.pt", map_location=dev, weights_only=False); sd = d["state_dict"]
    torch.manual_seed(0)
    m = SceneMinerTrainingFree(d_qwen=2048, d_siglip=sd["adapt_siglip.net.1.weight"].shape[1], n_margins=4, n_meta=6,
                               d_text=sd["W_route.weight"].shape[1], risk_stats=(float(rk.mean()), float(rk.std())),
                               prior_select=a.prior).to(dev)
    fresh = m.state_dict()
    merged = {k: (sd[k] if k in COPY_BUFFERS else v) for k, v in fresh.items()}
    m.load_state_dict(merged); m.use_abstain = True
    # learned parameters at their initial values (asserted, not assumed)
    for name in ("W_route", "meta_select", "gate_head", "margin_select"):
        for p in getattr(m, name).parameters(): assert float(p.abs().max()) == 0.0, name
    assert float(m.margin_sign.weight.abs().max()) == 0.0 and abs(float(m.margin_sign.bias) - 2.0) < 1e-6
    assert float(m.film.to_gamma_beta.weight.abs().max()) == 0.0
    # label-free PartNorm statistics over training scenes x training queries
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    tm = torch.from_numpy(tmask).to(dev)
    sig, mz, prior = T("siglip_img")[tm], T("vlm_margin_z")[tm], T("prior")[tm]
    P = []
    m.eval()
    with torch.no_grad():
        for q in QT:
            qv = q.unsqueeze(0).expand(sig.shape[0], -1)
            P.append(m.base_parts(qv, qv, sig, mz, prior, normalize=False))
    P = torch.cat(P)
    mu, var = P.mean(0), P.var(0, unbiased=False)
    m.part_norm.running_mean.copy_(mu); m.part_norm.running_var.copy_(var)
    trained = (sd["part_norm.running_mean"].tolist(), sd["part_norm.running_var"].tolist())
    print(f"[E2] PartNorm mean {mu.tolist()} var {var.tolist()}  (trained model: mean {trained[0]} var {trained[1]})", flush=True)
    out = Path("work_dirs/sceneminer_v26tf" if a.prior == "bank" else "work_dirs/sceneminer_v26tfu")
    meta = {"state_dict": m.state_dict(), "training_free": a.prior, "n_training_queries": len(texts),
            "partnorm_pairs": int(P.shape[0]), "partnorm_mean": mu.tolist(), "partnorm_var": var.tolist(),
            "trained_partnorm": {"mean": trained[0], "var": trained[1]}}
    for s in range(5):
        (out/f"s{s}").mkdir(parents=True, exist_ok=True); torch.save(meta, out/f"s{s}"/"final.pt")
    json.dump({k: v for k, v in meta.items() if k != "state_dict"}, open(out/"control.json", "w"), indent=1)
    print(f"[E2] wrote {out}/s0..s4", flush=True)


if __name__ == "__main__":
    main()
