#!/usr/bin/env python3
"""MILESTONE 17 - LaTeX tables for the supplementary, generated from data.

  per_query_suite.tex     per-query mAP on the 150 eval queries of the 390 suite
                          (recomputed here: SceneMiner v2.4, SigLIP2, logit lookup)
  per_query_semantic.tex  per-query mAP on the 40 Stratum-5 queries (from M18)
  train_queries.tex       the 240 TRAINING queries, listed but never scored -
                          scoring them would measure memorisation, not retrieval
  bh_table.tex            all 30 front-sector tests of Milestone 11 with BH q
"""
from __future__ import annotations
import os
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, load_attributes, predicate_mask
from eval_semantic_edge_cases import text_embed, load_v23

OUT = Path("paper/supp_tables"); OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
ESC = str.maketrans({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"})
tex = lambda s: str(s).translate(ESC)
SHORT = {"sensor_degradation": "S1", "macro_environmental": "S2", "compositional_heldout": "S3",
         "negated": "S4", "semantic_atomic": "S5a", "semantic_compositional": "S5b"}


def pred_str(p):
    return tex(", ".join(f"{k}={'T' if v is True else 'F' if v is False else v}" for k, v in p.items()))


def longtable(path, head, rows, caption, label, spec):
    L = [r"\begin{longtable}{" + spec + "}",
         r"\caption{" + caption + r"}\label{" + label + r"}\\", r"\toprule", head + r"\\", r"\midrule",
         r"\endfirsthead", r"\toprule", head + r"\\", r"\midrule", r"\endhead",
         r"\bottomrule", r"\endfoot"]
    L += [" & ".join(r) + r"\\" for r in rows]
    L += [r"\end{longtable}"]
    Path(path).write_text("\n".join(L) + "\n")


def fmt(x):
    return "--" if x is None or x != x else f"{x:.3f}"


def suite_rows():
    W = Path("work_dirs/unified_filter")
    z = np.load(W/"features_v2.npz", allow_pickle=False)
    corpus = [str(t) for t in z["scene_tokens"]]
    ev = set(json.load(open("benchmark_cvpr/split_906_602.json"))["eval"])
    idx = np.array([i for i, t in enumerate(corpus) if t in ev]); toks = [corpus[i] for i in idx]
    T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    emb, tlog, risk = T("emb"), T("tag_logits"), T("risk")
    sig, mz, prior = T("siglip_img"), T("vlm_margin_z"), T("prior")
    qwen = torch.load(W/"V_frozen_cache.pt", map_location=dev).float()
    M = load_v23(risk.reshape(-1).cpu().numpy(), which=os.environ.get("SM_FINAL", "v26"))
    for m in M: m.use_abstain = True
    attrs = load_attributes(W)
    for f in ("binary_flags_with_logits.json", "lowcontrast_strict.json"):
        for t, r in json.load(open(W/f)).items():
            rec = {k: v for k, v in r.items() if not k.endswith(("_margin", "_logp_yes"))}
            if "low_contrast_strict" in rec: rec["low_contrast"] = rec.pop("low_contrast_strict")
            attrs.setdefault(t, {}).update(rec)
    vlm = json.load(open(W/"binary_flags_with_logits.json")); strict = json.load(open(W/"lowcontrast_strict.json"))
    QS = json.load(open("benchmark_cvpr/queries_390.json"))["eval"]
    QT = text_embed([q["text"] for q in QS]); N = len(corpus); on = torch.ones(N, device=dev)
    rows, raw = [], []
    for qi, q in enumerate(QS):
        rel = predicate_mask(attrs, toks, q["predicate"])
        qv = QT[qi].unsqueeze(0).expand(N, -1)
        if rel.sum() in (0, len(rel)):
            vals = [None] * 3
        else:
            with torch.no_grad():
                s = np.mean([m(qv, qv, emb, tlog, risk, qwen, sig, mz, prior, has_3d=on).cpu().numpy() for m in M], 0)[idx]
                c = F.cosine_similarity(F.normalize(qv, -1), F.normalize(sig, -1)).cpu().numpy()[idx]
            at = next((k for k in ("droplets", "glare", "wet_road", "low_contrast") if k in q["predicate"]), None)
            lk = None
            if at:
                src, key = (strict, "low_contrast_strict") if at == "low_contrast" else (vlm, at)
                lk = ap_binary(np.argsort(-np.array([float(src.get(t, {}).get(key+"_margin", -9.)) for t in toks])), rel)
            vals = [ap_binary(np.argsort(-s), rel), ap_binary(np.argsort(-c), rel), lk]
        raw.append({"text": q["text"], "stratum": q["stratum"], "n_pos": int(rel.sum()),
                    "v2.5": vals[0], "SigLIP2": vals[1], "lookup": vals[2]})
        rows.append([str(qi + 1), SHORT[q["stratum"]], tex(q["text"]), pred_str(q["predicate"]),
                     str(int(rel.sum()))] + [fmt(v) for v in vals])
    json.dump(raw, open("results/cvpr/milestone27_per_query_suite150_v26.json", "w"), indent=1)
    return rows


def main():
    head = r"\# & S & query & predicate & pos & SceneMiner & SigLIP2 & lookup"
    spec = r"r l p{5.0cm} p{3.4cm} r r r r"
    longtable(OUT/"per_query_suite.tex", head, suite_rows(),
              r"Per-query mAP, 150 evaluation queries of the 390-query suite, 602 eval-split scenes. "
              r"Relevance is a predicate over cached VLM attributes (circular; \S\ref{sec:supp-circ}). "
              r"Lookup reads the column of the attribute the predicate names; -- where it names none.",
              "tab:pq-suite", spec)

    d = json.load(open("results/cvpr/milestone27_semantic_edge_cases.json"))["corpus_258"]["per_query"]
    rows = [[str(i + 1), SHORT[q["stratum"]], tex(q["text"]), pred_str(q["predicate"]), str(q["n_pos"]),
             f"{q['gamma_v24']:.2f}"] + [fmt(q["mAP"][k]) for k in
             ["v2.4_monocular", "zeroshot_SigLIP2", "logit_VLM_lookup", "v2.4_full [contam.]"]]
            for i, q in enumerate(d)]
    longtable(OUT/"per_query_semantic.tex",
              r"\# & S & query & predicate & pos & $\gamma$ & SceneMiner & SigLIP2 & lookup & full$^\dagger$",
              rows, r"Per-query mAP on Stratum 5, 258 human-labelled nuScenes eval scenes, 100\,\% human ground "
              r"truth. SceneMiner is run monocular (BEV and tag channels masked). $^\dagger$Full model; its BEV "
              r"and tag channels were supervised on these scenes and are contaminated for object queries.",
              "tab:pq-sem", r"r l p{4.0cm} p{3.0cm} r r r r r r")

    tr = json.load(open("benchmark_cvpr/queries_390.json"))["train"]
    rows = [[str(i + 1), SHORT.get(q["stratum"], tex(q["stratum"])), tex(q["text"]), pred_str(q["predicate"])]
            for i, q in enumerate(tr)]
    longtable(OUT/"train_queries.tex", r"\# & S & training query & predicate", rows,
              r"The 240 training queries of the 390-query suite. They are listed for completeness and "
              r"\emph{not scored}: SceneMiner's head saw their text during training.",
              "tab:train-q", r"r l p{8.6cm} p{5.6cm}")

    m11 = json.load(open("results/cvpr/milestone11_camfront_errors.json"))
    R = [(p, mt, m) for p, rec in m11["partitions"].items() for mt, m in rec["metrics"].items()]
    pv = np.array([m["perm_p"] for _, _, m in R]); n = len(pv); o = np.argsort(pv)
    q = np.empty(n); q[o] = np.clip(np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1], 0, 1)
    rows = []
    for i in o:
        p, mt, m = R[i]; nd = m11["partitions"][p]["n_degraded"]
        star = r"$^{**}$" if q[i] < 0.05 else (r"$^{*}$" if q[i] < 0.10 else "")
        rows.append([tex(p), str(nd), tex(mt), f"{m['degraded']:.4f}", f"{m['clean']:.4f}",
                     f"{m['delta']:+.4f}", f"{m['delta_sigma']:+.2f}",
                     f"[{m['delta_CI95'][0]:+.3f}, {m['delta_CI95'][1]:+.3f}]",
                     f"{m['perm_p']:.3f}", f"{q[i]:.3f}{star}"])
    longtable(OUT/"bh_table.tex",
              r"partition & $n_{deg}$ & metric & degraded & clean & $\Delta$ & $\sigma$ & 95\,\% CI & $p$ & $q$",
              rows, r"All 30 CAM\_FRONT-sector tests, sorted by $p$. 150 uncontaminated "
              r"val scenes; bootstrap CI over scenes (1000); permutation $p$ (1000); Benjamini--Hochberg $q$ "
              r"over the full family. $^{**}q<0.05$, $^{*}q<0.10$.", "tab:bh",
              r"l r l r r r r l r l")
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
