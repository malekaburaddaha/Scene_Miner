#!/usr/bin/env python3
"""MILESTONE 27 - cached scene index: identity with the reference forward pass, and latency.

For every seed and benchmark query, the reference scores (model.forward with the
query expanded to all scenes, or model.base_score for the portable path) are
compared with the cached path (fast_index.fast_score / fast_base_score): maximum
absolute score difference and AP difference on the benchmark's own relevance.
Short-circuiting the learned head for small gamma_deg is tested at several
thresholds and kept only if every AP is unchanged. Latency: median of 50 timed
calls, one model and the 5-seed ensemble, query embedding cached, A40.
-> results/cvpr/milestone27_fast_index.json
"""
from paths import DEVICE
from paths import EXT, HUMAN_R1
import json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, load_attributes, predicate_mask
from eval_human_gt_full import load_human, ATTRS, QUERY
from eval_semantic_edge_cases import load_v23, text_embed, TRAIN_ORDER
from eval_external_bdd100k import EVAL_Q, QUESTIONS
from scene_mining.filter.fast_index import precompute_scene_index, fast_score, fast_base_score, prepare_query, rank, GraphRanker

dev = torch.device(DEVICE); W = "work_dirs/unified_filter"
FINAL = os.environ.get("SM_FINAL", "v26")


def ap(s, rel):
    s = np.asarray(s, float)
    return ap_binary(np.argsort(-s), rel)


def main():
    z = np.load(f"{W}/features_v2.npz", allow_pickle=False)
    toks = [str(t) for t in z["scene_tokens"]]; cidx = {t: i for i, t in enumerate(toks)}
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    Fe = (T("emb"), T("tag_logits"), T("risk"), torch.load(f"{W}/V_frozen_cache.pt", map_location=dev).float(),
          T("siglip_img"), T("vlm_margin_z"), T("prior"))
    M = load_v23(z["risk"].reshape(-1), which=FINAL)
    for m in M: m.use_abstain = True
    FP64 = os.environ.get("FP64") == "1"          # identity test without float32 reordering noise
    if FP64:
        torch.set_default_dtype(torch.float64)
        M = [m.double() for m in M]; Fe = tuple(x.double() for x in Fe)
    N = len(toks)
    # benchmarks on the corpus: (name, query text, relevance over idx, idx, has_3d)
    hum = load_human(HUMAN_R1, "B")
    ht = [t for t in toks if t in hum]; hidx = np.array([cidx[t] for t in ht])
    bench = [("human", QUERY[a], np.array([hum[t][a] for t in ht]), hidx, 1.0) for a in ATTRS]
    ev = set(json.load(open("benchmark_cvpr/split_906_602.json"))["eval"])
    ST = [t for t in toks if t in ev]; sidx = np.array([cidx[t] for t in ST])
    attrs = load_attributes(__import__("pathlib").Path(W))
    for f in ("binary_flags_with_logits.json", "lowcontrast_strict.json"):
        for t, r in json.load(open(f"{W}/{f}")).items():
            rec = {k: v for k, v in r.items() if not k.endswith(("_margin", "_logp_yes"))}
            if "low_contrast_strict" in rec: rec["low_contrast"] = rec.pop("low_contrast_strict")
            attrs.setdefault(t, {}).update(rec)
    for q in json.load(open("benchmark_cvpr/queries_390.json"))["eval"]:
        rel = predicate_mask(attrs, ST, q["predicate"])
        if 0 < rel.sum() < len(rel): bench.append(("suite", q["text"], rel, sidx, 1.0))
    G = json.load(open(f"{W}/human_semantic_gt.json"))["scenes"]
    s5t = [t for t in toks if t in ev and t in G and t in hum]; s5i = np.array([cidx[t] for t in s5t])
    truth = lambda t, k: G[t][k] if (k in G[t] and k not in ATTRS) else hum[t][k]
    for q in json.load(open("benchmark_cvpr/queries_semantic_edge_cases.json"))["queries"]:
        rel = np.array([all(truth(t, k) == v for k, v in q["predicate"].items()) for t in s5t])
        bench.append(("s5_mono", q["text"], rel, s5i, 0.0))
    QE = text_embed(sorted({b[1] for b in bench} | {q for q, _ in EVAL_Q}))
    qmap = dict(zip(sorted({b[1] for b in bench} | {q for q, _ in EVAL_Q}), QE.double() if FP64 else QE))

    idxs = [precompute_scene_index(m, *Fe) for m in M]
    out = {"model": FINAL, "per_benchmark": {}, "skip_thresholds": {}}
    thr_list = [1e-6, 1e-4, 1e-3, 1e-2]
    worst = {}
    gam = {}
    for bname, qt, rel, ix, h3 in bench:
        q = qmap[qt]
        with torch.no_grad():
            slow = np.stack([m(q.expand(N, -1), q.expand(N, -1), *Fe, has_3d=torch.full((N,), h3, device=dev))
                             .cpu().numpy() for m in M])
            fast = np.stack([fast_score(m, ix_, q, has_3d=h3).cpu().numpy() for m, ix_ in zip(M, idxs)])
            g = float(M[0].concept(q[None])[1][0]); gam[qt] = g
        d = float(np.abs(slow - fast).max())
        a_s, a_f = ap(slow.mean(0)[ix], rel), ap(fast.mean(0)[ix], rel)
        a_ps = [ap(slow[k][ix], rel) for k in range(len(M))]; a_pf = [ap(fast[k][ix], rel) for k in range(len(M))]
        w = worst.setdefault(bname, {"max_abs_score_diff": 0.0, "max_abs_ap_diff": 0.0, "n_queries": 0})
        w["max_abs_score_diff"] = max(w["max_abs_score_diff"], d)
        w["max_abs_ap_diff"] = max(w["max_abs_ap_diff"], abs(a_s - a_f), *[abs(x - y) for x, y in zip(a_ps, a_pf)])
        w["n_queries"] += 1
        for thr in thr_list:
            with torch.no_grad():
                fk = np.stack([fast_score(m, ix_, q, has_3d=h3, gamma_skip=thr).cpu().numpy() for m, ix_ in zip(M, idxs)])
            st = out["skip_thresholds"].setdefault(str(thr), {"max_abs_ap_diff": 0.0, "queries_skipped": 0})
            st["max_abs_ap_diff"] = max(st["max_abs_ap_diff"], abs(ap(fk.mean(0)[ix], rel) - a_s))
            st["queries_skipped"] += int(g < thr)
    # portable path: BDD100K
    root = EXT + "/bdd100k"
    labels = json.load(open(f"{root}/labels.json")); names = sorted(labels); bz = np.load(f"{root}/signals.npz")
    ASK = [k for k, _ in QUESTIONS]; BM = bz["margin"][:, [ASK.index(k) for k in TRAIN_ORDER]]
    BM = torch.from_numpy((BM - BM.mean(0)) / (BM.std(0) + 1e-6)).float().to(dev)
    BS = torch.from_numpy(bz["siglip"]).to(dev).to(torch.get_default_dtype()); BM = BM.to(BS.dtype)
    nb = len(names); PR = torch.zeros(nb, 26, device=dev)
    av = torch.stack([torch.ones(nb, device=dev), torch.ones(nb, device=dev), torch.zeros(nb, device=dev)], -1)
    w = worst.setdefault("bdd_portable", {"max_abs_score_diff": 0.0, "max_abs_ap_diff": 0.0, "n_queries": 0})
    for qt, (field, pos) in EVAL_Q:
        rel = np.array([labels[n][field] in pos for n in names])
        if rel.sum() in (0, nb): continue
        q = qmap[qt]
        with torch.no_grad():
            slow = np.stack([m.base_score(q.expand(nb, -1), BS, BM, PR, av).cpu().numpy() for m in M])
            fast = np.stack([fast_base_score(m, BS, BM, PR, q).cpu().numpy() for m in M])
        w["max_abs_score_diff"] = max(w["max_abs_score_diff"], float(np.abs(slow - fast).max()))
        w["max_abs_ap_diff"] = max(w["max_abs_ap_diff"], abs(ap(slow.mean(0), rel) - ap(fast.mean(0), rel)))
        w["n_queries"] += 1
    out["per_benchmark"] = worst
    out["gamma_deg_distribution"] = {"n": len(gam), "lt_1e-3": int(sum(v < 1e-3 for v in gam.values()))}
    # latency
    def timeit(fn, n=50):
        fn(); torch.cuda.synchronize(); ts = []
        for _ in range(n):
            a = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(1000 * (time.perf_counter() - a))
        return float(np.median(ts))
    lat = {}
    for qt in [QUERY["droplets"], "pedestrians crossing the street after dark"]:
        q = qmap.get(qt, text_embed([qt])[0]).to(torch.get_default_dtype())
        g = float(M[0].concept(q[None])[1][0])
        r = {"gamma_deg": g}
        with torch.no_grad():
            r["reference_1model_ms"] = timeit(lambda: M[0](q.expand(N, -1), q.expand(N, -1), *Fe))
            r["cached_1model_ms"] = timeit(lambda: fast_score(M[0], idxs[0], q))
            r["reference_5seed_ms"] = timeit(lambda: [m(q.expand(N, -1), q.expand(N, -1), *Fe) for m in M], 20)
            r["cached_5seed_ms"] = timeit(lambda: [fast_score(m, i_, q) for m, i_ in zip(M, idxs)], 20)
            P = prepare_query(M[0], q)
            r["prepare_query_1model_ms"] = timeit(lambda: prepare_query(M[0], q))
            r["rank_1model_ms"] = timeit(lambda: rank(M[0], idxs[0], P))
            r["rank_vs_fast_max_abs_diff"] = float((rank(M[0], idxs[0], P) - fast_score(M[0], idxs[0], q)).abs().max())
            if not FP64:
                Ps = [prepare_query(m, q) for m in M]
                G = [GraphRanker(m, i_, p_) for m, i_, p_ in zip(M, idxs, Ps)]
                r["rank_cudagraph_1model_ms"] = timeit(lambda: G[0](Ps[0]))
                r["rank_cudagraph_5seed_ms"] = timeit(lambda: [g_(p_) for g_, p_ in zip(G, Ps)])
                r["prepare_query_5seed_ms"] = timeit(lambda: [prepare_query(m, q) for m in M], 20)
                r["cudagraph_vs_rank_max_abs_diff"] = float((G[0](Ps[0]) - rank(M[0], idxs[0], Ps[0])).abs().max())
            ok_thr = [float(t) for t, v in out["skip_thresholds"].items() if v["max_abs_ap_diff"] == 0.0]
            if ok_thr:
                t = max(ok_thr)
                r["cached_skip_1model_ms"] = timeit(lambda: fast_score(M[0], idxs[0], q, gamma_skip=t))
                r["skip_threshold"] = t
        lat[qt] = r
    with torch.no_grad():
        t0 = time.perf_counter(); [precompute_scene_index(m, *Fe) for m in M]; torch.cuda.synchronize()
    out["index_build_ms_5seed"] = 1000 * (time.perf_counter() - t0)
    out["latency"] = lat
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"results/cvpr/milestone27_fast_index{'_fp64' if FP64 else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
