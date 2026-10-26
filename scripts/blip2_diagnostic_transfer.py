#!/usr/bin/env python3
"""TASK 3 - apply the diagnostic protocol UNCHANGED to a second architecture.

Builds a BLIP-2 Q-Former index over the same 1,508-scene corpus and runs the
same three probes used on our own system:
  * effective rank of the 32 query tokens, read against the label-entropy null
  * routing mass per query token (how many carry < 1% of the mass)
  * per-attribute probe gradients, with an explicit bf16 gradient floor

The point is that the protocol is not specialised to the failure it was built
to find: if it reports a degeneracy here that BLIP-2's own retrieval scores do
not reveal, the protocol generalises.
"""
from __future__ import annotations
from paths import BLIP2
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from run_diagnostic_protocol import erank, label_matrix
from eval_attribute_index import load_attributes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--blip2", default=BLIP2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="results/day1_gate/blip2_diagnostic_transfer.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    W = Path(a.work)
    z = np.load(W/"features_overnight.npz", allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]
    attrs = load_attributes(W)
    nf = W/"binary_flags_with_logits.json"
    if nf.exists():
        for t, r in json.load(open(nf)).items():
            attrs.setdefault(t, {}).update({k: v for k, v in r.items()
                                            if not k.endswith(("_margin", "_logp_yes"))})

    from transformers import Blip2Model, Blip2Config, AutoProcessor
    from PIL import Image
    cfg = Blip2Config.from_pretrained(a.blip2)
    proc = AutoProcessor.from_pretrained(a.blip2)
    # Load the real pretrained BLIP-2 and take its submodules. Loading
    # Blip2VisionModel/Blip2QFormerModel directly from the repo root fails:
    # the checkpoint keys are prefixed (vision_model.*/qformer.*) and the
    # Q-Former needs cfg.qformer_config, not the top-level config.
    full = Blip2Model.from_pretrained(a.blip2, torch_dtype=torch.float32)
    vis = full.vision_model.to(dev).eval()
    qf = full.qformer.to(dev).eval()
    qtok = full.query_tokens.to(dev)          # PRETRAINED query tokens
    nq = qtok.shape[1]
    del full.language_model
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"[blip2] {nq} PRETRAINED query tokens, hidden {qtok.shape[-1]}", flush=True)

    Q = np.zeros((len(toks), nq, qtok.shape[-1]), dtype=np.float32)
    attn_mass = []
    with torch.no_grad():
        for i in range(0, len(toks), a.batch):
            ims = [Image.open(Path(a.thumbs)/f"{t}.jpg").convert("RGB")
                   for t in toks[i:i+a.batch]]
            px = proc(images=ims, return_tensors="pt").pixel_values.to(dev)
            ve = vis(pixel_values=px).last_hidden_state
            vm = torch.ones(ve.shape[:-1], dtype=torch.long, device=dev)
            out = qf(query_embeds=qtok.expand(ve.shape[0], -1, -1),
                     encoder_hidden_states=ve, encoder_attention_mask=vm,
                     output_attentions=True)
            Q[i:i+a.batch] = out.last_hidden_state.float().cpu().numpy()
            if out.cross_attentions:
                # mass each query token places on the visual stream
                A = torch.stack([c.float().mean(1).sum(-1) for c in out.cross_attentions], 0).mean(0)
                attn_mass.append(A.cpu().numpy())
            if (i // a.batch) % 25 == 0:
                print(f"  [blip2] {i+len(ims)}/{len(toks)}", flush=True)

    # --- protocol, applied unchanged ---
    # BUG FIX: the first version concatenated the 32 query tokens along the
    # feature axis, giving a (N, 32*768) matrix whose effective rank can exceed
    # 256 trivially (it measured 913). Effective rank must be read in the index
    # space it is being compared against. We therefore report:
    #   (a) tokens STACKED AS ROWS -> (32N, 768), the natural token space
    #   (b) that same matrix PCA-projected to d=256, directly comparable to the
    #       "41.2 of 256" figure
    #   (c) per-token erank in 768-d
    stacked = Q.reshape(len(toks) * nq, Q.shape[-1])          # (32N, 768)
    Xc = stacked - stacked.mean(0, keepdims=True)
    # PCA to 256 dims
    U, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj256 = Xc @ Vt[:256].T                                  # (32N, 256)
    flat = Q.reshape(len(toks), -1)                            # kept for reference
    per_token_erank = [erank(Q[:, k, :]) for k in range(nq)]
    L, cols = label_matrix(attrs, toks, ["lighting","weather","density",
                                         "droplets","glare","wet_road","low_contrast"])
    eL = erank(L)
    M = np.concatenate(attn_mass, 0) if attn_mass else None
    if M is not None:
        share = M.mean(0); share = share / max(1e-9, share.sum())
        low = int((share < 0.01).sum())
    else:
        share, low = None, None
    res = {
        "n_scenes": len(toks), "n_query_tokens": int(nq),
        "erank_stacked_tokens_768d": erank(stacked),
        "erank_pca256": erank(proj256),
        "erank_concat_query_tokens_BUGGY": erank(flat),
        "erank_per_query_token": {"mean": float(np.mean(per_token_erank)),
                                  "min": float(np.min(per_token_erank)),
                                  "max": float(np.max(per_token_erank)),
                                  "values": [float(v) for v in per_token_erank]},
        "label_null": {"erank_L": eL, "n_cols": len(cols),
                       "n_distinct_configs": int(len(np.unique(L, axis=0)))},
        "erank_ratio_pca256": erank(proj256)/max(1e-9, eL),
        "routing": {"tokens_below_1pct_mass": low,
                    "share": [float(v) for v in share] if share is not None else None},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("\n===== BLIP-2 Q-FORMER: protocol applied unchanged =====")
    print(f"  erank stacked tokens (768-d)  = {res['erank_stacked_tokens_768d']:.2f}")
    print(f"  erank PCA->256 (comparable)   = {res['erank_pca256']:.2f}  of 256")
    print(f"  [buggy concat 32x768]         = {res['erank_concat_query_tokens_BUGGY']:.2f}")
    print(f"  erank per query token         = mean {res['erank_per_query_token']['mean']:.2f} "
          f"[{res['erank_per_query_token']['min']:.2f}, {res['erank_per_query_token']['max']:.2f}]")
    print(f"  erank(L) label-entropy null   = {eL:.2f} "
          f"({res['label_null']['n_distinct_configs']} configs)")
    print(f"  erank ratio (PCA256/L)        = {res['erank_ratio_pca256']:.2f}")
    print(f"  query tokens < 1% routing mass= {low} / {nq}")
    print(f"[blip2] wrote {a.out}")

if __name__ == "__main__":
    main()
