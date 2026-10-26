#!/usr/bin/env python3
"""MILESTONE 34 - per-query VLM reranker baseline: SigLIP2 top-50 -> Qwen2.5-VL-3B.

The lookup baseline asks the VLM four FIXED degradation questions per image. This
baseline asks the query itself: SigLIP2 ViT-B/16 retrieves the top K=50 scenes of
the evaluated pool, and Qwen2.5-VL-3B scores each candidate with the yes/no log-odds
of "Does this driving scene show {query}? Answer Yes or No." (same model, image
files, chat template and Yes/No token sets as the cached margins). The K candidates
are re-ordered by log-odds above the rest of the pool, which keeps SigLIP2 order.
Top-K is taken within each evaluated pool (550 human-GT scenes, 1000 BDD100K images,
258 Stratum-5 scenes, val150, 2000 holdout images, 602 suite scenes), the setting
most favourable to the baseline.

Benchmarks and relevance are those of eval_related_work_baselines.py (Table 2). As a
check, the SigLIP2 per-query APs computed here must equal Milestone 28's.

--part plan                       count the (dataset, image, query) jobs
--part vlm --shard i --nshard n   log-odds for one shard -> work_dirs/vlm_rerank_shard{i}.json
--part latency                    online cost per query on one A40
--part eval                       -> results/cvpr/milestone34_per_query_vlm.json
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_related_work_baselines import (Enc, Rankers, load_corpus, load_bdd, load_val150, load_holdout, holds,
                                         ap_of, sign_perm, W, HOLD)
from eval_attribute_index import load_attributes, predicate_mask
from eval_human_gt_full import load_human, ATTRS, QUERY
from eval_external_bdd100k import EVAL_Q
from eval_domain_calibrated_v2 import mmr

K = 50
PROMPT = "Does this driving scene show {q}? Answer Yes or No."
QWEN = "/nfs/turbo/umd-srawa/hf_cache/Qwen2.5-VL-3B-Instruct"
BDD_ROOT = Path("/nfs/turbo/umd-srawa/scene_mining/datasets/external_adverse/bdd100k")
IMG = {"corpus": lambda t: f"work_dirs/unified_filter/scene_thumbs/{t}.jpg",
       "val150": lambda t: f"work_dirs/val150_thumbs/{t}.jpg",
       "bdd": lambda t: str(BDD_ROOT/"images"/t), "holdout": lambda t: str(HOLD/"images"/t)}
REF = "results/cvpr/milestone28_related_work_comparison.json"
dev = torch.device("cuda")


def prompt(q):
    return PROMPT.format(q=q.strip().rstrip("."))


def items():
    """Every benchmark query: dataset, pool tokens, SigLIP2 scores over the pool, relevance."""
    enc = Enc(); R = Rankers(enc, {}, np.zeros(1))
    G = json.load(open(W/"human_semantic_gt.json"))["scenes"]
    C, B, V = load_corpus(), load_bdd(), load_val150(G)
    H = load_holdout(with_meta=False)
    hum = load_human("benchmark_cvpr/human_rater1_600.json", "B")
    cidx = {t: i for i, t in enumerate(C["toks"])}
    ev = set(json.load(open("benchmark_cvpr/split_906_602.json"))["eval"])
    out = []

    def add(bench, D, qid, text, pool, rel, **kw):
        idx = np.array([{t: i for i, t in enumerate(D["toks"])}[t] for t in pool]) if pool is not D["toks"] else np.arange(D["n"])
        s = np.asarray(R.siglip2(D, text), float)[idx]
        out.append({"bench": bench, "data": D["name"], "id": qid, "text": text, "pool": list(pool), "sig": s,
                    "rel": np.asarray(rel, bool), **kw})

    ht = [t for t in C["toks"] if t in hum]
    for at in ATTRS:
        add("human_gt", C, at, QUERY[at], ht, [hum[t][at] for t in ht])
    for qt, (field, pos) in EVAL_Q:
        rel = np.array([B["labels"][n][field] in pos for n in B["toks"]])
        if rel.sum() in (0, len(rel)): continue
        add("bdd", B, qt, qt, B["toks"], rel)
    S5 = json.load(open("benchmark_cvpr/queries_semantic_edge_cases.json"))["queries"]
    s5t = [t for t in C["toks"] if t in ev and t in G and t in hum]
    truth = lambda t, k: G[t][k] if (k in G[t] and k not in ATTRS) else hum[t][k]
    for st in ["semantic_atomic", "semantic_compositional"]:
        tag = st.split("_")[1]
        for q in [q for q in S5 if q["stratum"] == st]:
            add(f"s5_corpus_{tag}", C, q["text"], q["text"], s5t,
                [all(truth(t, k) == x for k, x in q["predicate"].items()) for t in s5t])
        for q in [q for q in S5 if q["stratum"] == st and q["condition"] in (None, "night", "rain")]:
            rel = np.array([all(G[t][k] == x for k, x in q["predicate"].items()) for t in V["toks"]])
            if rel.sum() < 3: continue
            add(f"s5_val150_{tag}", V, q["text"], q["text"], V["toks"], rel)
    HQ = json.load(open("benchmark_cvpr/bdd_holdout_queries_36.json"))["queries"]
    for q in HQ:
        rel = np.array([holds(H["labels"][n], q["predicate"]) for n in H["toks"]])
        if rel.sum() >= 5: add("holdout", H, q["id"], q["text"], H["toks"], rel)
    attrs = load_attributes(W)
    for f in ("binary_flags_with_logits.json", "lowcontrast_strict.json"):
        for t, r in json.load(open(W/f)).items():
            rec = {k: v for k, v in r.items() if not k.endswith(("_margin", "_logp_yes"))}
            if "low_contrast_strict" in rec: rec["low_contrast"] = rec.pop("low_contrast_strict")
            attrs.setdefault(t, {}).update(rec)
    ST = [t for t in C["toks"] if t in ev]; sidx = np.array([cidx[t] for t in ST])
    dom = C["dom"][sidx]; Vn = torch.nn.functional.normalize(C["F"][4], dim=-1).cpu().numpy()[sidx]
    for j, q in enumerate(json.load(open("benchmark_cvpr/queries_390.json"))["eval"]):
        rel = predicate_mask(attrs, ST, q["predicate"])
        if rel.sum() in (0, len(rel)): continue
        add("suite", C, j, q["text"], ST, rel)
    return out, {"dom": dom, "Vn": Vn, "ST": ST}


def jobs(its):
    J = {}
    for it in its:
        for i in np.argsort(-it["sig"])[:K]:
            t = it["pool"][i]; J[f"{it['data']}|{t}|{it['text']}"] = (it["data"], t, it["text"])
    return [J[k] for k in sorted(J)]


class VLM:
    def __init__(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(QWEN)
        self.m = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN, torch_dtype=torch.bfloat16).to(dev).eval()
        tok = self.proc.tokenizer
        one = lambda ss: sorted({tok.encode(s, add_special_tokens=False)[0] for s in ss
                                 if len(tok.encode(s, add_special_tokens=False)) == 1})
        self.YES, self.NO = one(["Yes", "yes", " Yes", " yes"]), one(["No", "no", " No", " no"])

    @torch.no_grad()
    def __call__(self, imgs, texts):
        """yes/no log-odds of the first generated token for each (image, query); one batch."""
        from PIL import Image
        ims = [Image.open(p).convert("RGB") for p in imgs]
        ts = [self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": im},
                                                                          {"type": "text", "text": prompt(q)}]}],
                                            tokenize=False, add_generation_prompt=True) for im, q in zip(ims, texts)]
        self.proc.tokenizer.padding_side = "left"
        inp = self.proc(text=ts, images=ims, return_tensors="pt", padding=True).to(dev)
        lp = torch.log_softmax(self.m(**inp).logits[:, -1].float(), -1)
        return (torch.logsumexp(lp[:, self.YES], 1) - torch.logsumexp(lp[:, self.NO], 1)).cpu().numpy()


def part_vlm(shard, nshard):
    its, _ = items(); J = jobs(its)[shard::nshard]
    print(f"[vlm {shard}] {len(J)} jobs", flush=True)
    vlm = VLM(); out = {}; t0 = time.time()
    for n, (d, t, q) in enumerate(J):
        out[f"{d}|{t}|{q}"] = float(vlm([IMG[d](t)], [q])[0])       # batch 1: identical to the cached-margin recipe
        if (n + 1) % 500 == 0: print(f"[vlm {shard}] {n+1}/{len(J)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(f"work_dirs/vlm_rerank_shard{shard}.json", "w"))
    print(f"[vlm {shard}] DONE {time.time()-t0:.0f}s", flush=True)


def part_latency():
    """Online cost of one query on one A40: SigLIP2 text encoding and ranking of the 1508
    cached scenes, then K Qwen2.5-VL passes (batch 1, and batches of 10)."""
    enc = Enc(); R = Rankers(enc, {}, np.zeros(1)); C = load_corpus(); vlm = VLM()
    qs = ["water droplets smearing the camera lens", "a large truck or bus on a wet road", "pedestrians crossing the road at night",
          "blinding headlight bloom from oncoming traffic", "a construction zone with traffic cones and barriers"]
    res = {"K": K, "queries": qs, "batch1_ms": [], "batch10_ms": [], "stage_ms": []}
    vlm([IMG["corpus"](C["toks"][0])], [qs[0]]); torch.cuda.synchronize()
    for q in qs:
        a = time.perf_counter()
        s = np.asarray(R.siglip2(C, q + " "), float); top = [C["toks"][i] for i in np.argsort(-s)[:K]]   # uncached text
        torch.cuda.synchronize(); b = time.perf_counter()
        for t in top: vlm([IMG["corpus"](t)], [q])
        torch.cuda.synchronize(); c = time.perf_counter()
        for i in range(0, K, 10): vlm([IMG["corpus"](t) for t in top[i:i+10]], [q] * len(top[i:i+10]))
        torch.cuda.synchronize(); d = time.perf_counter()
        res["stage_ms"].append(1000 * (b - a)); res["batch1_ms"].append(1000 * (b - a + c - b)); res["batch10_ms"].append(1000 * (b - a + d - c))
        print(f"[lat] {q[:40]:<40} stage1 {1000*(b-a):.1f} ms  batch1 {1000*(c-a):.0f} ms  batch10 {1000*(b-a+d-c):.0f} ms", flush=True)
    for k in ("stage_ms", "batch1_ms", "batch10_ms"): res[k + "_median"] = float(np.median(res[k]))
    res["gpu"] = torch.cuda.get_device_name(0)
    json.dump(res, open("work_dirs/vlm_rerank_latency.json", "w"), indent=1)


def part_eval():
    its, aux = items(); L = {}
    for f in sorted(glob.glob("work_dirs/vlm_rerank_shard*.json")): L.update(json.load(open(f)))
    miss = [k for k in (f"{d}|{t}|{q}" for d, t, q in jobs(its)) if k not in L]
    assert not miss, f"{len(miss)} VLM jobs missing, e.g. {miss[:2]}"
    ref = json.load(open(REF)); RS, RP = ref["per_query"]["SigLIP2 ViT-B/16"], ref["per_query"]["SceneMiner"]
    per = {}; hgt = {}
    for it in its:
        s = it["sig"].copy(); top = np.argsort(-s)[:K]
        lo = np.array([L[f"{it['data']}|{it['pool'][i]}|{it['text']}"] for i in top])
        s[top] = 1e3 + lo
        if it["bench"] == "human_gt": hgt[it["id"]] = (s, it)
        rec = {"id": it["id"], "AP_rerank": ap_of(s, it["rel"]), "AP_siglip2": ap_of(it["sig"], it["rel"]),
               "n_pos": int(it["rel"].sum()), "pos_in_top50": int(it["rel"][top].sum()),
               "P@5_rerank": float(it["rel"][np.argsort(-s)[:5]].mean())}
        if it["bench"] == "suite":
            o = np.argsort(-s); dom = aux["dom"]
            rec["skew"] = float(np.mean(dom[mmr(o, aux["Vn"], aux["ST"], 5)] == 0)) - float(np.mean(dom[it["rel"]] == 0))
        per.setdefault(it["bench"], []).append(rec)
    out = {"doc": __doc__, "K": K, "prompt": PROMPT, "n_vlm_calls": len(jobs(its)), "benchmarks": {}, "check_siglip2": {}}
    for b, v in per.items():
        r = {"n_queries": len(v), "mAP_rerank": float(np.mean([x["AP_rerank"] for x in v])),
             "mAP_siglip2": float(np.mean([x["AP_siglip2"] for x in v])),
             "recall_of_positives_in_top50": float(np.sum([x["pos_in_top50"] for x in v]) / max(1, np.sum([x["n_pos"] for x in v]))),
             "per_query": v}
        if b == "suite": r["skew_pts"] = float(100 * np.mean([x["skew"] for x in v]))
        if b == "holdout": r["P@5_rerank"] = float(np.mean([x["P@5_rerank"] for x in v]))
        # SceneMiner reference per query (Table 2 run) and the SigLIP2 identity check
        if b == "human_gt":
            sm = [ref["methods"]["SceneMiner"]["human_gt_per_attr"][a] for a in ATTRS]
            sg = [ref["methods"]["SigLIP2 ViT-B/16"]["human_gt_per_attr"][a] for a in ATTRS]
        elif b == "holdout":
            H = ref["holdout"]["modes"]["metadata_free"]["methods"]
            sm = [H["SceneMiner (portable)"]["per_query"][x["id"]]["AP"] for x in v]
            sg = [H["SigLIP2 ViT-B/16"]["per_query"][x["id"]]["AP"] for x in v]
        else:
            sm, sg = RP[b], RS[b]
        out["check_siglip2"][b] = float(np.max(np.abs(np.array(sg) - np.array([x["AP_siglip2"] for x in v]))))
        d = np.array(sm) - np.array([x["AP_rerank"] for x in v])
        r["sceneminer_mAP"] = float(np.mean(sm))
        r["sceneminer_minus_rerank"] = {"delta": float(d.mean()), "perm_p": sign_perm(d) if len(d) > 1 else None, "n": len(d)}
        out["benchmarks"][b] = r
        print(f"[eval] {b:<26} n={len(v):>3}  rerank {r['mAP_rerank']:.3f}  SigLIP2 {r['mAP_siglip2']:.3f}  SceneMiner {r['sceneminer_mAP']:.3f}"
              f"  delta {d.mean():+.3f} p={r['sceneminer_minus_rerank']['perm_p']}  recall@50 {r['recall_of_positives_in_top50']:.2f}"
              f"  |siglip2 check| {out['check_siglip2'][b]:.1e}", flush=True)
    # human GT has 4 queries, so the paired test resamples the 550 scenes instead (per-seed AP protocol)
    enc = Enc(); C = load_corpus(); z = np.load(W/"features_v2.npz", allow_pickle=False)
    R = Rankers(enc, {}, z["risk"].reshape(-1)); cid = {t: i for i, t in enumerate(C["toks"])}
    pool = hgt[ATTRS[0]][1]["pool"]; ix = np.array([cid[t] for t in pool])
    SM = {a: np.asarray(R.sceneminer(C, QUERY[a]), float)[:, ix] for a in ATTRS}
    rng = np.random.default_rng(0); d = []
    macro = lambda b: np.mean([ap_of(SM[a][:, b], hgt[a][1]["rel"][b], per_seed=True) - ap_of(hgt[a][0][b], hgt[a][1]["rel"][b]) for a in ATTRS])
    point = float(macro(np.arange(len(pool))))
    for _ in range(1000): d.append(macro(rng.integers(0, len(pool), len(pool))))
    out["benchmarks"]["human_gt"]["scene_bootstrap"] = {"delta_sceneminer_minus_rerank": point, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                                        "p_one_sided": float((1 + (np.array(d) <= 0).sum()) / (len(d) + 1)),
                                                        "per_attribute": {a: {"SceneMiner": ap_of(SM[a], hgt[a][1]["rel"], per_seed=True),
                                                                              "rerank": ap_of(hgt[a][0], hgt[a][1]["rel"])} for a in ATTRS}}
    print("[eval] human GT scene bootstrap", out["benchmarks"]["human_gt"]["scene_bootstrap"], flush=True)
    lat = Path("work_dirs/vlm_rerank_latency.json")
    if lat.exists(): out["latency"] = json.load(open(lat))
    json.dump(out, open("results/cvpr/milestone34_per_query_vlm.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["plan", "vlm", "latency", "eval"], required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    if a.part == "plan":
        its, _ = items(); J = jobs(its)
        print({b: sum(1 for i in its if i["bench"] == b) for b in dict.fromkeys(i["bench"] for i in its)}, "jobs", len(J))
    elif a.part == "vlm": part_vlm(a.shard, a.nshard)
    elif a.part == "latency": part_latency()
    else: part_eval()
