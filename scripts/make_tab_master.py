#!/usr/bin/env python3
"""Write the literature comparison tables from
results/cvpr/milestone28_related_work_comparison.json, with the per-query VLM reranker
(milestone34_per_query_vlm.json) and five-seed sds of the trained rows' human GT
(milestone34_seed_sd.json) added in Milestone 34.

  paper/tab_master_full.tex  every column (supplementary)
  paper/tab_master.tex       main paper (Milestone 39): the four human-graded or external
                             benchmarks, their unweighted mean and the online cost; the mean
                             is written to results/cvpr/milestone39_main_table.json
"""
import json
from pathlib import Path
import numpy as np

R = json.load(open("results/cvpr/milestone28_related_work_comparison.json"))
LAT = json.load(open("results/cvpr/milestone27_latency.json"))          # cached-index timings for SceneMiner
VR = json.load(open("results/cvpr/milestone34_per_query_vlm.json"))       # Milestone 34: per-query VLM reranker
SD = json.load(open("results/cvpr/milestone34_seed_sd.json"))["rows"]    # Milestone 34: five-seed sd (human GT)
M, PQ = R["methods"], R["per_query"]
ROWS = [  # (label in JSON, label in paper, citation, modality string)
    ("BEV-TSR-style text-to-BEV", "BEV-TSR-style", "tang2025bevtsr", "BEV"),
    ("RefAV-style programmatic filter", "RefAV-style (LLM program)", "davidson2026refav", "VLM flags, tags, meta"),
    ("CARIM-style caption retrieval", "CARIM-style (neutral captions)", "ki2025carim", "VLM captions"),
    ("CARIM-style (enumerating prompt, M23)", "CARIM-style (listing captions)", "ki2025carim", "VLM captions"),
    ("Tag-map (Talk2BEV/nuPrompt-style)", "Tag map", "choudhary2024talk2bev,wu2023nuprompt", "BEV tags"),
    ("ReferGPT-style state + CLIP", "ReferGPT-style", "chamiti2025refergpt", "image, tags, meta"),
    ("CLIP ViT-B/32", "CLIP ViT-B/32", "radford2021clip", "image"),
    ("SigLIP ViT-B/16", "SigLIP ViT-B/16", "zhai2023siglip", "image"),
    ("SigLIP2 ViT-B/16", "SigLIP2 ViT-B/16", "tschannen2025siglip2", "image"),
    ("SigLIP2 + fine-tuned adapter", "SigLIP2 + adapter (906 scenes)", None, "image"),
    ("Qwen2.5-VL lookup", "Qwen2.5-VL lookup", "bai2025qwen25vl", "VLM margins"),
    ("VLM_RERANK", "SigLIP2 top-50 $+$ VLM rerank", None, "image, VLM"),
    ("Naive uncalibrated fusion", "Naive convex router", None, "all"),
    ("SceneMiner (portable)", "\\textbf{SceneMiner} (portable)", None, "image, VLM, meta"),
    ("SceneMiner", "\\textbf{SceneMiner} (full)", None, "all"),
]
CONTAM = lambda k: M.get(k, {}).get("reads_bev_or_tags_on_corpus", False)
CIRC = {"RefAV-style programmatic filter", "Qwen2.5-VL lookup"}      # read the very attributes that define suite relevance


def s5(k, part):
    a, c = PQ[k].get(f"s5_{part}_atomic"), PQ[k].get(f"s5_{part}_compositional")
    if not a or None in a or None in c: return None
    return float(np.mean(a + c))


def vr_cols():
    B = VR["benchmarks"]; pooled = lambda a, b: float(np.mean([x["AP_rerank"] for x in B[a]["per_query"] + B[b]["per_query"]]))
    return [B["human_gt"]["mAP_rerank"], B["bdd"]["mAP_rerank"], pooled("s5_corpus_atomic", "s5_corpus_compositional"),
            pooled("s5_val150_atomic", "s5_val150_compositional"), B["suite"]["mAP_rerank"], B["holdout"]["mAP_rerank"]]


cols = {}
for k, *_ in ROWS:
    if k == "VLM_RERANK": cols[k] = vr_cols(); continue
    m = M[k]
    ho = R["holdout"]["modes"]["metadata_free"]["methods"].get(k)
    cols[k] = [m["human_gt"], m["bdd"], s5(k, "corpus"), s5(k, "val150"), m["suite"], ho["mAP_all"] if ho else None]
# rows marked circular on the suite do not compete for "best" in that column
best = [max(v[i] for k, v in cols.items() if v[i] is not None and not (i == 4 and k in CIRC)) for i in range(6)]
lines = []
for k, lab, cit, mods in ROWS:
    m = M.get(k, {}); v = cols[k]
    cell = []
    for i, x in enumerate(v):
        if x is None: cell.append("n/a"); continue
        t = f"{x:.3f}"
        if round(x, 3) == round(best[i], 3): t = f"\\best{{{t}}}"
        if i == 0 and f"T2|{k}" in SD: t += f"$_{{\\pm{SD[f'T2|{k}']['human_gt']['sd']:.3f}}}$".replace("0.", ".", 1)
        if (i in (0, 2, 4) and CONTAM(k)) or (i in (0, 4) and k == "SceneMiner"): t += "\\cont"   # SceneMiner: 3D tokens on; S5 is monocular
        if i == 4 and k in CIRC: t += "$^{\\ddagger}$"
        cell.append(t)
    if k == "VLM_RERANK":
        skew = f"${VR['benchmarks']['suite']['skew_pts']:+.1f}$"; lat = VR["latency"]["batch10_ms_median"]
    else:
        skew = "n/a" if m["suite_skew_pts"] is None else f"${m['suite_skew_pts']:+.1f}$"
        lat = LAT[k]["ms"] if k in LAT else m.get("rank_latency_ms_1508_single_model", m["rank_latency_ms_1508"])
    name = lab + (f"~\\cite{{{cit}}}" if cit else "")
    lt = f"${lat:,.0f}$".replace(",", "{,}") if lat >= 100 else f"${lat:.2f}$"
    lines.append(f"{name} & {mods} & {cell[0]} & {cell[1]} & {cell[5]} & {cell[2]} & {cell[3]} & {cell[4]} & {skew} & {lt}\\\\")
    if k == "VLM_RERANK": lines.append("\\midrule")
lines.insert(6, "\\midrule")
tex = ["\\setlength{\\tabcolsep}{3.3pt}", "\\begin{tabular}{llcccccccc}", "\\toprule",
       " & & human GT & \\multicolumn{2}{c}{BDD100K} & \\multicolumn{2}{c}{Stratum 5} & \\multicolumn{2}{c}{150Q suite} & ms\\\\",
       "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
       "method & modalities & ($550$) & $1000$ & holdout & $258$ & val150 & mAP & skew & ($1508$)\\\\", "\\midrule",
       *lines, "\\bottomrule", "\\end{tabular}"]
Path("paper/tab_master_full.tex").write_text("\n".join(tex) + "\n")
# the supplementary has no bibliography: same table without citations
import re
Path("paper/supp_tables/master_full_table.tex").write_text(re.sub(r"~\\cite\{[^}]*\}", "", "\n".join(tex)) + "\n")

# ---------------------------------------------------------------- main-paper table (Milestone 39)
MAIN = [("Driving-retrieval paradigms (re-implemented)", ["BEV-TSR-style text-to-BEV", "RefAV-style programmatic filter",
         "CARIM-style caption retrieval", "Tag-map (Talk2BEV/nuPrompt-style)", "ReferGPT-style state + CLIP"]),
        ("Foundation models and adapted baselines", ["CLIP ViT-B/32", "SigLIP2 ViT-B/16", "SigLIP2 + fine-tuned adapter",
         "Qwen2.5-VL lookup", "Naive uncalibrated fusion", "VLM_RERANK"]),
        ("Ours", ["SceneMiner (portable)", "SceneMiner"])]
SHORT = {"BEV-TSR-style text-to-BEV": "BEV-TSR-style", "RefAV-style programmatic filter": "RefAV-style",
         "CARIM-style caption retrieval": "CARIM-style", "Tag-map (Talk2BEV/nuPrompt-style)": "Talk2BEV/nuPrompt tags",
         "ReferGPT-style state + CLIP": "ReferGPT-style", "SigLIP2 + fine-tuned adapter": "SigLIP2 $+$ trained adapter",
         "Qwen2.5-VL lookup": "Qwen2.5-VL attribute lookup", "Naive uncalibrated fusion": "Naive convex router",
         "VLM_RERANK": "SigLIP2 $+$ Qwen2.5-VL rerank", "SceneMiner (portable)": "\\textbf{SceneMiner} (portable)",
         "SceneMiner": "\\textbf{SceneMiner} (full)"}
CITE = {k: c for k, _, c, _ in ROWS}
SIG = {"BEV-TSR-style text-to-BEV": "BEV", "RefAV-style programmatic filter": "VLM, tags, meta",
       "CARIM-style caption retrieval": "captions", "Tag-map (Talk2BEV/nuPrompt-style)": "BEV tags",
       "ReferGPT-style state + CLIP": "image, tags, meta", "CLIP ViT-B/32": "image", "SigLIP2 ViT-B/16": "image",
       "SigLIP2 + fine-tuned adapter": "image", "Qwen2.5-VL lookup": "VLM", "Naive uncalibrated fusion": "all",
       "VLM_RERANK": "image, VLM", "SceneMiner (portable)": "image, VLM, meta", "SceneMiner": "all"}
main_cols, mean = {}, {}
for _, ks in MAIN:
    for k in ks:
        v = cols[k]; main_cols[k] = [v[0], v[1], v[5], v[2]]          # human GT, BDD100K, holdout, Stratum 5 (258)
        mean[k] = None if None in main_cols[k] else float(np.mean(main_cols[k]))
mbest = [max(main_cols[k][i] for k in main_cols if main_cols[k][i] is not None) for i in range(4)]
bmean = max(v for v in mean.values() if v is not None)
L = []
for g, ks in MAIN:
    L += ["\\midrule", f"\\multicolumn{{8}}{{l}}{{\\emph{{{g}}}}}\\\\"]
    for k in ks:
        cell = []
        for i, x in enumerate(main_cols[k]):
            if x is None: cell.append("--"); continue
            t = f"{x:.3f}"
            if round(x, 3) == round(mbest[i], 3): t = f"\\best{{{t}}}"
            if i == 0 and f"T2|{k}" in SD: t += f"$_{{\\pm{SD[f'T2|{k}']['human_gt']['sd']:.3f}}}$".replace("0.", ".", 1)
            if (i in (0, 3) and CONTAM(k)) or (i == 0 and k == "SceneMiner"): t += "\\cont"
            cell.append(t)
        mv = "--" if mean[k] is None else (f"\\best{{{mean[k]:.3f}}}" if round(mean[k], 3) == round(bmean, 3) else f"{mean[k]:.3f}")
        if k == "VLM_RERANK": lat = VR["latency"]["batch10_ms_median"]
        else: lat = LAT[k]["ms"] if k in LAT else M[k].get("rank_latency_ms_1508_single_model", M[k]["rank_latency_ms_1508"])
        lt = f"${lat:,.0f}$".replace(",", "{,}") if lat >= 100 else (f"${lat:.1f}$" if lat >= 1 else f"${lat:.2f}$")
        name = SHORT.get(k, k) + (f"~\\cite{{{CITE[k]}}}" if CITE.get(k) else "")
        L.append(f"{name} & {SIG[k]} & " + " & ".join(cell) + f" & {mv} & {lt}\\\\")
tex = ["\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{llcccc|cr}", "\\toprule",
       " & & degradation & \\multicolumn{2}{c}{BDD100K (external)} & semantic & & ms /\\\\",
       "\\cmidrule(lr){4-5}",
       "method & signals & (human, $550$) & $1000$ & holdout & (3D GT, $258$) & mean & query\\\\",
       *L, "\\bottomrule", "\\end{tabular}"]
Path("paper/tab_master.tex").write_text("\n".join(tex) + "\n")
json.dump({"doc": "Milestone 39 main-paper comparison: human GT (550), BDD100K (1000), holdout (2000), Stratum 5 (258) and their unweighted mean",
           "columns": ["human_gt", "bdd", "holdout", "s5_corpus"], "rows": main_cols, "mean": mean},
          open("results/cvpr/milestone39_main_table.json", "w"), indent=1)
print("\n".join(L))
