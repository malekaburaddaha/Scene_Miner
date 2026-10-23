#!/usr/bin/env python3
"""Write paper/tab_bddq.tex (BDD100K per query) from the corrected BDD100K run
(Milestone 23: raw query features, results/cvpr/milestone23_bdd_rawquery_v25.json)."""
import json
from pathlib import Path

SHORT = {"a low contrast scene where vehicles are barely distinguishable from the background":
         "a low contrast scene, vehicles barely distinguishable"}
d = json.load(open("results/cvpr/milestone23_bdd_rawquery_v25.json"))["B_bdd100k"]["per_query"]
d = sorted(d, key=lambda q: (q["category"] != "in_vocab", -q["gamma"]))
rows = []
for q in d:
    m = q["mAP"]; v = {"SceneMiner": m["v2.3_full"], "SigLIP2": m["zeroshot_SigLIP2"], "lookup": m["logit_VLM_lookup"]}
    top = max(round(x, 3) for x in v.values())
    cell = lambda x: f"\\best{{{x:.3f}}}" if round(x, 3) == top else f"{x:.3f}"
    col = q["concept_argmax"].replace("_", "\\_") if q["gamma"] > 0.05 else "--"
    rows.append(f"{'I' if q['category'] == 'in_vocab' else 'O'} & {SHORT.get(q['query'], q['query'])} & "
                f"{q['gamma']:.3f} & {col} & {cell(v['SceneMiner'])} & {cell(v['SigLIP2'])} & {cell(v['lookup'])}\\\\")
tex = ["\\setlength{\\tabcolsep}{6pt}", "\\begin{tabular}{llclccc}", "\\toprule",
       " & query & $\\gamma_{\\mathrm{deg}}$ & column & SceneMiner & SigLIP2 & lookup\\\\", "\\midrule",
       *rows, "\\bottomrule", "\\end{tabular}"]
Path("paper/tab_bddq.tex").write_text("\n".join(tex) + "\n")
print("\n".join(rows))
