#!/usr/bin/env python3
"""Write paper/tab_master.tex, the literature comparison table, from
results/cvpr/milestone23_related_work_comparison.json (Milestone 23)."""
import json
from pathlib import Path
import numpy as np

R = json.load(open("results/cvpr/milestone28_related_work_comparison.json"))
LAT = json.load(open("results/cvpr/milestone27_latency.json"))          # cached-index timings for SceneMiner
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
    ("Naive uncalibrated fusion", "Naive convex router", None, "all"),
    ("SceneMiner (portable)", "\\textbf{SceneMiner} (portable)", None, "image, VLM, meta"),
    ("SceneMiner", "\\textbf{SceneMiner} (full)", None, "all"),
]
CONTAM = lambda k: M[k]["reads_bev_or_tags_on_corpus"]
CIRC = {"RefAV-style programmatic filter", "Qwen2.5-VL lookup"}      # read the very attributes that define suite relevance


def s5(k, part):
    a, c = PQ[k].get(f"s5_{part}_atomic"), PQ[k].get(f"s5_{part}_compositional")
    if not a or None in a or None in c: return None
    return float(np.mean(a + c))


cols = {}
for k, *_ in ROWS:
    m = M[k]
    cols[k] = [m["human_gt"], m["bdd"], s5(k, "corpus"), s5(k, "val150"), m["suite"]]
# rows marked circular on the suite do not compete for "best" in that column
best = [max(v[i] for k, v in cols.items() if v[i] is not None and not (i == 4 and k in CIRC)) for i in range(5)]
lines = []
for k, lab, cit, mods in ROWS:
    m = M[k]; v = cols[k]
    cell = []
    for i, x in enumerate(v):
        if x is None: cell.append("n/a"); continue
        t = f"{x:.3f}"
        if round(x, 3) == round(best[i], 3): t = f"\\best{{{t}}}"
        if (i in (0, 2, 4) and CONTAM(k)) or (i in (0, 4) and k == "SceneMiner"): t += "\\cont"   # SceneMiner: 3D tokens on; S5 is monocular
        if i == 4 and k in CIRC: t += "$^{\\ddagger}$"
        cell.append(t)
    skew = "n/a" if m["suite_skew_pts"] is None else f"${m['suite_skew_pts']:+.1f}$"
    lat = LAT[k]["ms"] if k in LAT else m.get("rank_latency_ms_1508_single_model", m["rank_latency_ms_1508"])
    name = lab + (f"~\\cite{{{cit}}}" if cit else "")
    lines.append(f"{name} & {mods} & {cell[0]} & {cell[1]} & {cell[2]} & {cell[3]} & {cell[4]} & {skew} & ${lat:.2f}$\\\\")
    if k == "Qwen2.5-VL lookup": lines.append("\\midrule")
lines.insert(6, "\\midrule")
tex = ["\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{llccccccc}", "\\toprule",
       " & & human GT & BDD100K & \\multicolumn{2}{c}{Stratum 5} & \\multicolumn{2}{c}{150Q suite} & ms\\\\",
       "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
       "method & modalities & ($550$) & ($1000$) & $258$ & val150 & mAP & skew & ($1508$)\\\\", "\\midrule",
       *lines, "\\bottomrule", "\\end{tabular}"]
Path("paper/tab_master.tex").write_text("\n".join(tex) + "\n")
print("\n".join(lines))
