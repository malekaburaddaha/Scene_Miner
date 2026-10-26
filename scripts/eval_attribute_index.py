#!/usr/bin/env python3
"""STEP 1 - Cached attribute-index baseline vs. SceneMiner (Reviewer ETJq).

Answers: can a cached VLM/metadata attribute lookup replace the learned index?

Three rankers over the SAME 1,508-scene corpus, queries and relevance:

  lookup_flat    Boolean predicate match over cached attributes -> 2 tiers
                 (match / no-match). Ties broken uniformly at random; we report
                 EXPECTED AP over --perms random permutations within each tier.
  lookup_graded  Strongest lookup constructible from the August assets:
                 partial credit = (# satisfied conjuncts)/(# conjuncts), with
                 the continuous veh_50m count as a within-tier tie-break for any
                 density term. NOTE: the VLM logit-margin variant of this
                 baseline is NOT constructible yet - binary_flags_{a,b} are
                 DISJOINT SHARDS of pure booleans (overlap 0), so no confidence
                 or cross-run agreement signal exists. Step 2 re-extracts with
                 Yes/No log-prob margins; this is the honest stand-in.
  sceneminer     D_final/final.pt, the deployed learned index.

Query sets: macro8 (the 8-query macro suite), degrad (droplets/glare/wet_road),
severity (GRADED relevance from continuous veh_50m - the only query family where
a flat bucket provably cannot express the ground truth).
"""
from __future__ import annotations
from paths import QWEN_VL
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

# ---------------------------------------------------------------- queries ----
MACRO8 = [
    ("camera lens occluded by heavy rain droplets during daytime",
     {"droplets": True, "lighting": "day"}),
    ("dense traffic congestion on wet asphalt at night",
     {"density": "dense", "wet_road": True, "lighting": "night"}),
    ("heavy rain at a dark unlit intersection",
     {"weather": "rain", "lighting": "night"}),
    ("strong headlight glare bloom reflecting off wet road surface at night",
     {"glare": True, "wet_road": True, "lighting": "night"}),
    ("empty highway under pitch black nighttime conditions",
     {"density": "sparse", "lighting": "night"}),
    ("overcast daytime driving with wet road surface reflections",
     {"lighting": "day", "wet_road": True}),
    ("dense bumper-to-bumper traffic under overcast skies",
     {"density": "dense", "lighting": "day"}),
    ("dark urban street with wet pavement and low ambient lighting",
     {"lighting": "night", "wet_road": True}),
]

# Preliminary sensor-degradation suite over {droplets, glare, wet_road}.
# >=3 phrasings per condition, incl. severity wording, as Step 2 will formalise.
DEGRAD = [
    ("water droplets smearing the camera lens",            {"droplets": True}),
    ("rain beading on the windshield obscuring the view",  {"droplets": True}),
    ("lens spattered with heavy rain during the day",      {"droplets": True, "lighting": "day"}),
    ("a drizzle leaving light droplets on the lens",       {"droplets": True}),
    ("blinding headlight bloom from oncoming traffic",     {"glare": True}),
    ("specular glare washing out the image",               {"glare": True}),
    ("mild light flare at night",                          {"glare": True, "lighting": "night"}),
    ("intense light bloom reflecting off the road",        {"glare": True, "wet_road": True}),
    ("wet reflective asphalt after rainfall",              {"wet_road": True}),
    ("standing water on the road surface",                 {"wet_road": True}),
    ("damp road surface in daylight",                      {"wet_road": True, "lighting": "day"}),
    ("slick wet pavement at night",                        {"wet_road": True, "lighting": "night"}),
    ("rain droplets on the lens at night",                 {"droplets": True, "lighting": "night"}),
    ("glare on a wet road in the dark",                    {"glare": True, "wet_road": True, "lighting": "night"}),
]

# GRADED relevance from the continuous veh_50m count. A flat 3-way bucket
# (dense/moderate/sparse) cannot express these orderings.
SEVERITY = [
    ("extremely heavy bumper-to-bumper traffic",  ("veh_50m", "high",   None)),
    ("a moderately busy road",                    ("veh_50m", "mid",    None)),
    ("an almost completely empty road",           ("veh_50m", "low",    None)),
    ("very heavy traffic at night",               ("veh_50m", "high",   {"lighting": "night"})),
    ("heavy congestion on a wet road",            ("veh_50m", "high",   {"wet_road": True})),
    ("a deserted street after dark",              ("veh_50m", "low",    {"lighting": "night"})),
]


# ---------------------------------------------------------------- metrics ----
def ap_binary(order, rel):
    r = rel[order].astype(float)
    if r.sum() == 0: return 0.0
    hits = np.cumsum(r)
    return float((hits[r == 1] / (np.where(r == 1)[0] + 1)).mean())

def ndcg_at(order, gain, k=10):
    g = gain[order][:k]
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
    ideal = np.sort(gain)[::-1][:k]
    idcg = float((ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))).sum())
    return float((g * disc).sum() / idcg) if idcg > 0 else 0.0

def p_at(order, rel, k=5):
    return float(rel[order][:k].mean())

def expected_metrics_tiered(tier, rel, gain, perms, rng, k_p=5, k_n=10):
    """Expected AP / P@5 / NDCG@10 when ranking by `tier` (higher = better) with
    uniformly random order WITHIN each tier. Monte Carlo over `perms` draws."""
    n = len(tier)
    aps, ps, nd = np.empty(perms), np.empty(perms), np.empty(perms)
    for i in range(perms):
        jitter = rng.random(n)
        order = np.lexsort((jitter, -tier))   # primary: tier desc; secondary: random
        aps[i] = ap_binary(order, rel)
        ps[i]  = p_at(order, rel, k_p)
        nd[i]  = ndcg_at(order, gain, k_n)
    return (float(aps.mean()), float(aps.std()),
            float(ps.mean()), float(nd.mean()))


# ------------------------------------------------------------ attributes -----
def load_attributes(W):
    det = json.load(open(W / "deterministic_gt.json"))
    attrs = {k: dict(v) for k, v in det.items()}
    for shard in ("binary_flags_a.json", "binary_flags_b.json"):
        p = W / shard
        if p.exists():
            for tk, fl in json.load(open(p)).items():
                attrs.setdefault(tk, {}).update(fl)
    return attrs

def predicate_mask(attrs, toks, pred):
    return np.array([all(attrs.get(t, {}).get(k) == v for k, v in pred.items())
                     for t in toks], dtype=bool)

def conjunct_fraction(attrs, toks, pred):
    n = max(1, len(pred))
    return np.array([sum(attrs.get(t, {}).get(k) == v for k, v in pred.items()) / n
                     for t in toks], dtype=float)

def severity_targets(attrs, toks, band, extra):
    """Graded gain in {0,1,2,3} from the continuous veh_50m count, restricted to
    scenes satisfying `extra`. Quartile bands; gain decays with distance."""
    v = np.array([float(attrs.get(t, {}).get("veh_50m", 0.0)) for t in toks])
    ok = predicate_mask(attrs, toks, extra) if extra else np.ones(len(toks), bool)
    q = np.quantile(v[ok], [0.25, 0.50, 0.75]) if ok.sum() > 4 else np.quantile(v, [.25, .5, .75])
    gain = np.zeros(len(toks))
    if band == "high":
        gain = np.where(v >= q[2], 3.0, np.where(v >= q[1], 2.0, np.where(v >= q[0], 1.0, 0.0)))
    elif band == "low":
        gain = np.where(v <= q[0], 3.0, np.where(v <= q[1], 2.0, np.where(v <= q[2], 1.0, 0.0)))
    else:
        mid = q[1]; spread = max(1e-6, q[2] - q[0])
        gain = np.clip(3.0 - 3.0 * np.abs(v - mid) / spread, 0.0, 3.0)
    gain = gain * ok
    return gain, v, ok


# ------------------------------------------------------------- SceneMiner ----
def sceneminer_scores(args, toks, queries_text, W):
    import torch
    from scene_mining.filter.lora_grounded_filter import (LoRAGroundedFilter,
                                                          QwenVisualEncoder)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(W / "features_overnight.npz", allow_pickle=False)
    emb  = torch.from_numpy(z["emb"]).float().to(dev)
    tlog = torch.from_numpy(z["tag_logits"]).float().to(dev)
    risk = torch.from_numpy(z["risk"]).float().to(dev)
    rk = risk.reshape(-1).cpu().numpy()

    enc = QwenVisualEncoder(args.qwen, lora_r=args.lora_r, lora_alpha=128,
                            device=str(dev), img_max_side=args.img_side)
    enc.model.to(dev)
    model = LoRAGroundedFilter(enc, tau=args.tau,
                               risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["trainable_state_dict"], strict=False)
    assert len(unexpected) == 0, f"architecture mismatch: {unexpected[:5]}"
    model.eval()
    print(f"  [sceneminer] loaded {args.checkpoint} (step {ck['global_step']})", flush=True)

    # ---- index build: one Qwen pass over the corpus thumbnails ----
    from PIL import Image
    t0 = time.perf_counter()
    V = torch.zeros(len(toks), enc.d_model, device=dev)
    with torch.no_grad():
        for i in range(0, len(toks), args.batch):
            ims = [Image.open(Path(args.thumbs) / f"{t}.jpg").convert("RGB")
                   for t in toks[i:i + args.batch]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                V[i:i + args.batch] = enc.encode(ims)
            if (i // args.batch) % 25 == 0:
                print(f"    [visual] {i+len(ims)}/{len(toks)}", flush=True)
    if dev.type == "cuda": torch.cuda.synchronize()
    t_index = time.perf_counter() - t0

    # ---- query encoding + scoring ----
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
    import torch.nn.functional as F
    ctok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    cenc = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True).to(dev).eval()

    scores, lat = {}, []
    with torch.no_grad():
        for q in queries_text:
            t1 = time.perf_counter()
            tt = ctok([q], padding=True, truncation=True, return_tensors="pt").to(dev)
            qc = F.normalize(cenc(**tt).text_embeds.float(), dim=-1)
            qq = qc.expand(len(toks), -1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                s = model.score(qq, emb, tlog, risk, V).float()
            if dev.type == "cuda": torch.cuda.synchronize()
            lat.append((time.perf_counter() - t1) * 1e3)
            scores[q] = s.cpu().numpy()
    return scores, t_index, float(np.median(lat))


# ------------------------------------------------------------------ main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--checkpoint", default="work_dirs/ablation/D_final/final.pt")
    ap.add_argument("--qwen", default=QWEN_VL)
    ap.add_argument("--query-set", choices=["macro8", "degrad", "severity", "all"],
                    default="all")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--img-side", type=int, default=336)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--skip-sceneminer", action="store_true")
    ap.add_argument("--output", default="results/step1_attribute_index.json")
    args = ap.parse_args()

    W = Path(args.work)
    rng = np.random.default_rng(args.seed)
    z = np.load(W / "features_overnight.npz", allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]
    dom = z["domain"].astype(int)
    attrs = load_attributes(W)
    n = len(toks)
    print(f"[step1] corpus={n} (nuScenes {int((dom==0).sum())} / Waymo {int((dom==1).sum())})")
    print(f"[step1] attribute coverage: {sum(t in attrs for t in toks)}/{n}")

    sets = {"macro8": MACRO8, "degrad": DEGRAD, "severity": SEVERITY}
    chosen = list(sets) if args.query_set == "all" else [args.query_set]

    # ---- SceneMiner scores for every query we will evaluate ----
    sm_scores, t_index_sm, q_lat_sm = {}, float("nan"), float("nan")
    if not args.skip_sceneminer:
        all_q = [q for s in chosen for q, _ in sets[s]]
        sm_scores, t_index_sm, q_lat_sm = sceneminer_scores(args, toks, all_q, W)

    # ---- lookup index build cost (attributes are already cached) ----
    t0 = time.perf_counter(); _ = load_attributes(W); t_index_lookup = time.perf_counter() - t0

    out = {"corpus": n, "perms": args.perms, "seed": args.seed,
           "latency": {"sceneminer_index_s": t_index_sm,
                       "sceneminer_query_ms": q_lat_sm,
                       "lookup_index_s": t_index_lookup},
           "sets": {}}

    for sname in chosen:
        rows, per_q = [], {}
        print(f"\n===== query set: {sname} ({len(sets[sname])} queries) =====")
        print(f"{'query':<52}{'rel':>5} {'flat':>7} {'grad':>7} {'ours':>7}")
        print("-" * 82)
        agg = {"lookup_flat": [], "lookup_graded": [], "sceneminer": []}
        aggp = {k: [] for k in agg}; aggn = {k: [] for k in agg}
        for q, spec in sets[sname]:
            if sname == "severity":
                band, extra = spec[1], spec[2]
                gain, vcount, ok = severity_targets(attrs, toks, band, extra)
                rel = gain >= 2.0                      # binary view for AP/P@5
                # flat lookup: 3-way density bucket mapped to the band
                want = {"high": "dense", "mid": "moderate", "low": "sparse"}[band]
                pred = {"density": want}
                if extra: pred.update(extra)
                tier_flat = predicate_mask(attrs, toks, pred).astype(float)
                # graded lookup: bucket tier + continuous veh_50m within it
                base = conjunct_fraction(attrs, toks, pred)
                vnorm = (vcount - vcount.min()) / (np.ptp(vcount) + 1e-9)
                tier_grad = base + (vnorm if band == "high" else
                                    (1 - vnorm) if band == "low" else
                                    1 - np.abs(vnorm - 0.5) * 2) * 0.999
            else:
                pred = spec
                rel = predicate_mask(attrs, toks, pred)
                gain = rel.astype(float)
                tier_flat = rel.astype(float)
                tier_grad = conjunct_fraction(attrs, toks, pred)

            if rel.sum() == 0 or rel.sum() == n:
                print(f"{q[:50]:<52}{int(rel.sum()):>5}   (skipped: degenerate)")
                continue

            f_ap, f_sd, f_p, f_n = expected_metrics_tiered(tier_flat, rel, gain, args.perms, rng)
            g_ap, g_sd, g_p, g_n = expected_metrics_tiered(tier_grad, rel, gain, args.perms, rng)
            if q in sm_scores:
                o = np.argsort(-sm_scores[q])
                s_ap, s_p, s_n = ap_binary(o, rel), p_at(o, rel), ndcg_at(o, gain)
            else:
                s_ap = s_p = s_n = float("nan")

            agg["lookup_flat"].append(f_ap); agg["lookup_graded"].append(g_ap); agg["sceneminer"].append(s_ap)
            aggp["lookup_flat"].append(f_p); aggp["lookup_graded"].append(g_p); aggp["sceneminer"].append(s_p)
            aggn["lookup_flat"].append(f_n); aggn["lookup_graded"].append(g_n); aggn["sceneminer"].append(s_n)
            per_q[q] = {"n_relevant": int(rel.sum()),
                        "lookup_flat": {"AP": f_ap, "AP_sd": f_sd, "P@5": f_p, "NDCG@10": f_n},
                        "lookup_graded": {"AP": g_ap, "AP_sd": g_sd, "P@5": g_p, "NDCG@10": g_n},
                        "sceneminer": {"AP": s_ap, "P@5": s_p, "NDCG@10": s_n}}
            print(f"{q[:50]:<52}{int(rel.sum()):>5} {f_ap:>7.3f} {g_ap:>7.3f} {s_ap:>7.3f}")

        summ = {m: {"mAP": float(np.mean(agg[m])),
                    "P@5": float(np.mean(aggp[m])),
                    "NDCG@10": float(np.mean(aggn[m])),
                    "n_queries": len(agg[m])} for m in agg}
        prev = float(np.mean([per_q[q]["n_relevant"] for q in per_q]) / n) if per_q else 0.0
        summ["chance_prevalence"] = prev
        out["sets"][sname] = {"summary": summ, "per_query": per_q}
        print("-" * 82)
        for m in ("lookup_flat", "lookup_graded", "sceneminer"):
            s = summ[m]
            print(f"  {m:<16} mAP={s['mAP']:.4f}  P@5={s['P@5']:.4f}  NDCG@10={s['NDCG@10']:.4f}")
        print(f"  chance prevalence = {prev:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[step1] wrote {args.output}")
    print(f"[step1] latency: lookup index {t_index_lookup*1e3:.1f} ms | "
          f"sceneminer index {t_index_sm:.1f} s | sceneminer query {q_lat_sm:.2f} ms")


if __name__ == "__main__":
    main()
