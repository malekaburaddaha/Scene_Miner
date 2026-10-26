#!/usr/bin/env python3
"""Write paper/tab_master.tex, the literature comparison table, from
results/cvpr/milestone28_related_work_comparison.json, with the per-query VLM reranker
(milestone34_per_query_vlm.json) and five-seed sds of the trained rows' human GT
(milestone34_seed_sd.json) added in Milestone 34."""
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
    ("RefAV-style programmatic filter", "RefAV-style (LLM program)", "davidson2025refav", "VLM flags, tags, meta"),
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
Path("paper/tab_master.tex").write_text("\n".join(tex) + "\n")
print("\n".join(lines))
