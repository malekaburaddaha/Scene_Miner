#!/usr/bin/env python3
"""MILESTONE 15 - Stratum 5 against 100 % human ground truth.

Two evaluation sets:

  CORPUS: the 258 nuScenes eval-split scenes that carry human degradation
    labels. CONTAMINATION WARNING: these scenes belong to the official nuScenes
    TRAIN split. BEVFormer (source of the BEV embedding e and risk r) was trained
    on their 3D boxes, and the BLIP-2 tag head (source of the 20 tags t) was
    supervised with nuPrompt tags for them - and those tags include
    object_pedestrian, object_bus, scene_construction_zone. Any ranker that
    reads e, r or t is therefore marked [contam.] here. The clean rankers use
    only frozen, web-trained signals: SigLIP2, the Qwen VLM margins, and
    SceneMiner v2.3 in monocular mode (3D tokens and tag prior masked).

  VAL150: the 150 official nuScenes VAL scenes, which neither BEVFormer nor
    our heads trained on. Only portable signals exist here (SigLIP2 on the
    CAM_FRONT keyframe, and the CAM_FRONT VLM margins from Milestone 11), so
    this is the fully clean test. Lens/road degradation has no human labels on
    val150, so only atomic queries and night/rain conjunctions are graded.

Rankers. A ranker with nothing to say about a query (a 4-column VLM lookup asked
about pedestrians) scores every scene equally; its AP is the expectation over
random tie-breaking, i.e. chance. For conjunctions the lookups use the a-priori
column (night -> low_contrast, rain -> wet_road, others their own column) and
the metadata ranker uses recorded lighting/weather - which, for night and rain,
is the same human scene description the ground truth uses.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, p_at, ndcg_at
from eval_human_gt_full import load_human, ATTRS

dev = torch.device("cuda")
TRAIN_ORDER = ["droplets", "glare", "wet_road", "low_contrast"]
COND_COL = {"night": "low_contrast", "rain": "wet_road", "glare": "glare",
            "wet_road": "wet_road", "droplets": "droplets", "low_contrast": "low_contrast"}
META_OF = {"night": ("lighting", "night"), "rain": ("weather", "rain"),
           "glare": ("lighting", "night"), "low_contrast": ("lighting", "night"),
           "wet_road": ("weather", "rain"), "droplets": ("weather", "rain")}
TAG = {"pedestrian_close": [8], "vru": [9, 10], "construction_zone": [2],
       "heavy_vehicle": [11, 12], "rare_or_child": [14]}


def chance_ap(rel, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([ap_binary(rng.permutation(len(rel)), rel) for _ in range(n)]))


def metrics(s, rel):
    if np.ptp(s) == 0:
        return {"mAP": chance_ap(rel), "P@5": float(rel.mean()), "NDCG@10": float("nan"), "constant": True}
    o = np.argsort(-s)
    return {"mAP": ap_binary(o, rel), "P@5": p_at(o, rel), "NDCG@10": ndcg_at(o, rel.astype(float))}


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def text_embed(texts):
    from transformers import AutoModel, GemmaTokenizerFast
    P = "google/siglip2-base-patch16-224"
    sm = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    tk = GemmaTokenizerFast.from_pretrained(P)
    ft = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    with torch.no_grad():
        E = ft(sm.get_text_features(**tk(texts, padding="max_length", truncation=True,
                                         max_length=64, return_tensors="pt").to(dev)))
    del sm; torch.cuda.empty_cache()
    return E


def val150_siglip(thumbs, cache):
    if Path(cache).exists():
        z = np.load(cache, allow_pickle=False); return [str(t) for t in z["scenes"]], z["siglip"]
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    P = "google/siglip2-base-patch16-224"
    m = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
    ip = AutoImageProcessor.from_pretrained(P)
    ft = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
    sc = sorted(p.stem for p in Path(thumbs).glob("*.jpg"))
    X = []
    with torch.no_grad():
        for i in range(0, len(sc), 32):
            ims = [Image.open(Path(thumbs)/f"{s}.jpg").convert("RGB") for s in sc[i:i+32]]
            X.append(F.normalize(ft(m.get_image_features(**ip(images=ims, return_tensors="pt").to(dev))), dim=-1).cpu().numpy())
    X = np.concatenate(X); np.savez_compressed(cache, scenes=np.array(sc), siglip=X)
    del m; torch.cuda.empty_cache()
    return sc, X


def load_v23(rk, d_qwen=2048, which="v23"):
    from scene_mining.filter.sceneminer_v2_3 import SceneMinerV23
    if which == "v24":
        from scene_mining.filter.sceneminer_v2_4 import SceneMinerV24 as SceneMinerV23
    if which == "v25":
        from scene_mining.filter.sceneminer_v2_5 import SceneMinerV25 as SceneMinerV23
    if which == "v26":
        from scene_mining.filter.sceneminer_v2_6 import SceneMinerV26 as SceneMinerV23
    M = []
    for c in sorted(glob.glob(f"work_dirs/sceneminer_{which}/s*/final.pt")):
        d = torch.load(c, map_location=dev, weights_only=False)
        m = SceneMinerV23(d_qwen=d_qwen, d_siglip=768, n_margins=4, n_meta=6, d_text=768,
                          risk_stats=(float(rk.mean()), float(rk.std()))).to(dev)
        m.load_state_dict(d["state_dict"]); m.eval(); M.append(m)
    return M


def portable(M, qv, S, MZt, abstain=False, PR=None):
    """Base score with the meta channel declared absent (no rig, no tags). PR may
    carry recorded metadata (columns 20-25) for the abstention target."""
    N = S.shape[0]
    if PR is None: PR = torch.zeros(N, 26, device=dev)
    av = torch.stack([torch.ones(N, device=dev), torch.ones(N, device=dev), torch.zeros(N, device=dev)], -1)
    prev = [m.use_abstain for m in M]
    for m in M: m.use_abstain = abstain
    with torch.no_grad():
        out = np.mean([m.base_score(qv, S, MZt, PR, av).cpu().numpy() for m in M], 0)
    for m, pv in zip(M, prev): m.use_abstain = pv
    return out


def with_abstain(M, fn):
    prev = [m.use_abstain for m in M]
    for m in M: m.use_abstain = True
    try: return fn()
    finally:
        for m, pv in zip(M, prev): m.use_abstain = pv


def summarise(rows, queries):
    out = {}
    for st in ["semantic_atomic", "semantic_compositional", "all"]:
        keep = [i for i, q in enumerate(queries) if st == "all" or q["stratum"] == st]
        out[st] = {}
        for k, L in rows.items():
            v = [L[i] for i in keep if L[i] is not None]
            if not v: continue
            out[st][k] = {m: float(np.nanmean([x[m] for x in v])) for m in ("mAP", "P@5", "NDCG@10")}
            out[st][k]["n_queries"] = len(v)
    return out


def paired(rows, queries, a, b, stratum, by_predicate=False, n=10000, seed=0):
    keep = [i for i, q in enumerate(queries) if q["stratum"] == stratum
            and rows[a][i] is not None and rows[b][i] is not None]
    if by_predicate:
        grp = {}
        for i in keep: grp.setdefault(json.dumps(queries[i]["predicate"], sort_keys=True), []).append(i)
        d = np.array([np.mean([rows[a][i]["mAP"] - rows[b][i]["mAP"] for i in g]) for g in grp.values()])
    else:
        d = np.array([rows[a][i]["mAP"] - rows[b][i]["mAP"] for i in keep])
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1, 1], size=(n, len(d)))
    p = float(((np.abs((flips * d).mean(1)) >= abs(d.mean())).sum() + 1) / (n + 1))
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return {"delta": float(d.mean()), "CI95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "perm_p": p, "n_units": int(len(d)), "unit": "predicate" if by_predicate else "query"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="benchmark_cvpr/queries_semantic_edge_cases.json")
    ap.add_argument("--gt", default="work_dirs/unified_filter/human_semantic_gt.json")
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--work", default="work_dirs/unified_filter")
    ap.add_argument("--human", default="results/day1_gate/malek_repaired.json")
    ap.add_argument("--split", default="benchmark_cvpr/split_906_602.json")
    ap.add_argument("--val-margins", default="work_dirs/val150_camfront_signals.npz")
    ap.add_argument("--val-siglip", default="work_dirs/val150_siglip.npz")
    ap.add_argument("--val-thumbs", default="work_dirs/val150_thumbs")
    ap.add_argument("--out", default="results/cvpr/milestone18_semantic_edge_cases.json")
    ap.add_argument("--final", default="v24", choices=["v24", "v25", "v26"],
                    help="checkpoint family reported in the 'v2.4_*' rows (labels kept for downstream tools)")
    a = ap.parse_args()
    Q = json.load(open(a.queries))["queries"]
    G = json.load(open(a.gt))["scenes"]
    QT = text_embed([q["text"] for q in Q]); QN = F.normalize(QT, dim=-1)

    # ------------------------------------------------------------ corpus set
    z = np.load(a.features, allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]; cidx = {t: i for i, t in enumerate(corpus)}
    hum = load_human(a.human, "B")
    ev = set(json.load(open(a.split))["eval"])
    toks = [t for t in corpus if t in ev and t in G and t in hum]
    idx = np.array([cidx[t] for t in toks])
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    emb, tlog, risk = T("emb"), T("tag_logits"), T("risk")
    sig, mz, prior = T("siglip_img"), T("vlm_margin_z"), T("prior")
    qwen = torch.load(Path(a.work)/"V_frozen_cache.pt", map_location=dev).float()
    M = load_v23(risk.reshape(-1).cpu().numpy())
    M24 = load_v23(risk.reshape(-1).cpu().numpy(), which=a.final)
    for m in M24: m.use_abstain = True
    W = Path(a.work)
    vlm = json.load(open(W/"binary_flags_with_logits.json"))
    strict = json.load(open(W/"lowcontrast_strict.json"))
    meta = json.load(open(W/"deterministic_gt.json"))
    TL = tlog.cpu().numpy()[idx]
    N = len(corpus); on = torch.ones(N, device=dev); off = torch.zeros(N, device=dev)

    def colval(t, col, margin=True):
        src, key = (strict, "low_contrast_strict") if col == "low_contrast" else (vlm, col)
        return float(src.get(t, {}).get(key + "_margin", -9.)) if margin else float(bool(src.get(t, {}).get(key, False)))

    def truth(t, k):
        return G[t][k] if k in G[t] and k not in ATTRS else hum[t][k]

    rng = np.random.default_rng(0)
    rows = {k: [] for k in ["random", "flat_VLM_lookup", "logit_VLM_lookup", "metadata_matched",
                            "zeroshot_SigLIP2", "v2.3_monocular", "v2.3_portable_base",
                            "BLIP2_tag_lookup [contam.]", "v2.3_full [contam.]",
                            "v2.3A_monocular", "v2.3A_portable_base", "v2.3A_full [contam.]",
                            "v2.4_monocular", "v2.4_portable_base", "v2.4_full [contam.]"]}
    per_q = []
    for qi, q in enumerate(Q):
        rel = np.array([all(truth(t, k) == v for k, v in q["predicate"].items()) for t in toks])
        cond = q["condition"]
        R = {"random": np.zeros(len(toks))}
        if cond is None:
            R["flat_VLM_lookup"] = R["logit_VLM_lookup"] = R["metadata_matched"] = np.zeros(len(toks))
        else:
            col = COND_COL[cond]
            R["logit_VLM_lookup"] = np.array([colval(t, col) for t in toks])
            R["flat_VLM_lookup"] = np.array([colval(t, col, False) for t in toks]) + rng.random(len(toks)) * 1e-6
            fld, val = META_OF[cond]
            R["metadata_matched"] = np.array([float(meta.get(t, {}).get(fld) == val) for t in toks]) \
                + rng.random(len(toks)) * 1e-6
        R["zeroshot_SigLIP2"] = (sig[idx] @ QN[qi]).cpu().numpy()
        tag = zs(TL[:, TAG[q["semantic"]]].max(1))
        R["BLIP2_tag_lookup [contam.]"] = tag if cond is None else tag + zs(R["logit_VLM_lookup"])
        qv = QT[qi].unsqueeze(0).expand(N, -1)
        with torch.no_grad():
            R["v2.3_full [contam.]"] = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=on)
                                                .cpu().numpy() for m in M], 0)[idx]
            R["v2.3_monocular"] = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=off)
                                           .cpu().numpy() for m in M], 0)[idx]
            _, gam, _ = M[0].concept(QT[qi:qi+1])
            _, gam24, _ = M24[0].concept(QT[qi:qi+1])
        R["v2.3_portable_base"] = portable(M, qv, sig, mz)[idx]
        # POST-HOC variant: abstention toward SigLIP2 out of vocabulary
        with torch.no_grad():
            R["v2.3A_full [contam.]"] = with_abstain(M, lambda: np.mean(
                [m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=on).cpu().numpy() for m in M], 0)[idx])
            R["v2.3A_monocular"] = with_abstain(M, lambda: np.mean(
                [m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=off).cpu().numpy() for m in M], 0)[idx])
        R["v2.3A_portable_base"] = portable(M, qv, sig, mz, abstain=True)[idx]
        with torch.no_grad():
            R["v2.4_full [contam.]"] = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=on)
                                                .cpu().numpy() for m in M24], 0)[idx]
            R["v2.4_monocular"] = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=off)
                                           .cpu().numpy() for m in M24], 0)[idx]
        R["v2.4_portable_base"] = portable(M24, qv, sig, mz, abstain=True, PR=prior)[idx]
        res = {k: metrics(v, rel) for k, v in R.items()}
        for k in rows: rows[k].append(res[k])
        per_q.append({"text": q["text"], "stratum": q["stratum"], "predicate": q["predicate"],
                      "n_pos": int(rel.sum()), "gamma": float(gam), "gamma_v24": float(gam24),
                      "mAP": {k: v["mAP"] for k, v in res.items()}})
    corpus_out = {"n_scenes": len(toks), "per_query": per_q, "summary": summarise(rows, Q),
                  "tests": {}}
    for a_, b_ in [("v2.4_monocular", "zeroshot_SigLIP2"), ("v2.4_monocular", "logit_VLM_lookup"),
                   ("v2.3A_monocular", "zeroshot_SigLIP2"), ("v2.3A_full [contam.]", "zeroshot_SigLIP2"),
                   ("v2.3A_monocular", "logit_VLM_lookup"), ("v2.3_monocular", "zeroshot_SigLIP2"), ("v2.3_full [contam.]", "zeroshot_SigLIP2"),
                   ("v2.3_monocular", "logit_VLM_lookup")]:
        for st in ["semantic_atomic", "semantic_compositional"]:
            corpus_out["tests"][f"{a_} vs {b_} | {st}"] = paired(rows, Q, a_, b_, st,
                                                                by_predicate=(st == "semantic_compositional"))

    # ------------------------------------------------------------ val150 set
    vs, VS = val150_siglip(a.val_thumbs, a.val_siglip)
    vz = np.load(a.val_margins, allow_pickle=False)
    vm_sc = [str(t) for t in vz["scenes"]]
    ASKED = ["wet_road", "glare", "droplets", "low_contrast"]       # Milestone 11 column order
    VM = vz["margin"][:, [ASKED.index(k) for k in TRAIN_ORDER]]
    order = [vm_sc.index(t) for t in vs]; VM = VM[order]
    VMZ = (VM - VM.mean(0)) / (VM.std(0) + 1e-6)
    S = torch.from_numpy(VS).float().to(dev); MZt = torch.from_numpy(VMZ).float().to(dev)
    # recorded metadata for val150, from the human scene description (same source
    # as the corpus metadata); density is not recorded there, so it stays empty
    VPR = torch.zeros(len(vs), 26, device=dev)
    for i, t in enumerate(vs):
        VPR[i, 20] = float(G[t]["night"]); VPR[i, 21] = float(not G[t]["night"])
        VPR[i, 22] = float(G[t]["rain"]); VPR[i, 23] = float(not G[t]["rain"])
    vQ = [q for q in Q if q["condition"] in (None, "night", "rain")]
    vrows = {k: [] for k in ["random", "logit_VLM_lookup", "metadata_matched", "zeroshot_SigLIP2",
                             "v2.3_portable_base", "v2.3A_portable_base", "v2.4_portable_base"]}
    vper = []
    for q in vQ:
        qi = Q.index(q)
        rel = np.array([all(G[t][k] == v for k, v in q["predicate"].items()) for t in vs])
        if rel.sum() < 3: continue
        cond = q["condition"]
        R = {"random": np.zeros(len(vs))}
        if cond is None:
            R["logit_VLM_lookup"] = R["metadata_matched"] = np.zeros(len(vs))
        else:
            R["logit_VLM_lookup"] = VM[:, TRAIN_ORDER.index(COND_COL[cond])]
            R["metadata_matched"] = np.array([float(G[t][cond]) for t in vs]) + rng.random(len(vs)) * 1e-6
        R["zeroshot_SigLIP2"] = (S @ QN[qi]).cpu().numpy()
        R["v2.3_portable_base"] = portable(M, QT[qi].unsqueeze(0).expand(len(vs), -1), S, MZt)
        R["v2.3A_portable_base"] = portable(M, QT[qi].unsqueeze(0).expand(len(vs), -1), S, MZt, abstain=True)
        R["v2.4_portable_base"] = portable(M24, QT[qi].unsqueeze(0).expand(len(vs), -1), S, MZt, abstain=True,
                                           PR=VPR if a.final in ("v25", "v26") else None)
        res = {k: metrics(v, rel) for k, v in R.items()}
        for k in vrows: vrows[k].append(res[k])
        vper.append({"text": q["text"], "stratum": q["stratum"], "predicate": q["predicate"],
                     "n_pos": int(rel.sum()), "mAP": {k: v["mAP"] for k, v in res.items()}})
    vQk = [q for q in vQ if any(p["text"] == q["text"] for p in vper)]
    val_out = {"n_scenes": len(vs), "per_query": vper, "summary": summarise(vrows, vQk),
               "tests": {"v2.3_portable_base vs zeroshot_SigLIP2 | semantic_atomic":
                         paired(vrows, vQk, "v2.3_portable_base", "zeroshot_SigLIP2", "semantic_atomic"),
                         "v2.4_portable_base vs zeroshot_SigLIP2 | semantic_atomic":
                         paired(vrows, vQk, "v2.4_portable_base", "zeroshot_SigLIP2", "semantic_atomic"),
                         "v2.4_portable_base vs zeroshot_SigLIP2 | semantic_compositional":
                         paired(vrows, vQk, "v2.4_portable_base", "zeroshot_SigLIP2", "semantic_compositional"),
                         "v2.3A_portable_base vs zeroshot_SigLIP2 | semantic_atomic":
                         paired(vrows, vQk, "v2.3A_portable_base", "zeroshot_SigLIP2", "semantic_atomic")}}
    out = {"doc": __doc__, "corpus_258": corpus_out, "val150_clean": val_out}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for name, o in [("CORPUS 258", corpus_out), ("VAL150 (clean)", val_out)]:
        print(f"\n===== {name}: {o['n_scenes']} scenes =====")
        for st, d in o["summary"].items():
            print(f"  -- {st} --")
            for k, v in sorted(d.items(), key=lambda kv: -kv[1]["mAP"]):
                print(f"    {k:<30}{v['mAP']:>8.4f}  (n={v['n_queries']})")
        for k, v in o["tests"].items():
            print(f"  {k}: delta {v['delta']:+.4f} CI [{v['CI95'][0]:+.3f},{v['CI95'][1]:+.3f}] "
                  f"p={v['perm_p']:.4f} ({v['n_units']} {v['unit']}s)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
