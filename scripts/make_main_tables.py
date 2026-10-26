#!/usr/bin/env python3
"""MILESTONE 31 - the main-paper tables, generated from the result files (3 decimals).

Writes paper/tab_design.tex, tab_abl.tex, tab_bdd.tex, tab_suite.tex, tab_regimes.tex,
tab_loop.tex, tab_rater2.tex and paper/supp_tables/{holdout,rater,rater2_methods}_table.tex.
Sources (all under results/cvpr/):
  design   v2.1 milestone6_sceneminer_v2_1.json (human GT, mode B), milestone10_query_suite_150.json,
                milestone11_domain_calibration.json (skew), milestone23_bdd_rawquery_routing.json (BDD100K)
           v2.2 milestone10_sceneminer_v2_2.json, same suite/skew/BDD files
           v2.3 milestone14_v2_3.json + milestone23_bdd_rawquery_v23.json
           v2.4 milestone18_v2_4.json + milestone23_bdd_rawquery_v24.json
           v2.5 milestone22_factored_gate.json + milestone23_bdd_rawquery_v25.json
           v2.6 milestone27_v26.json (final model)
  ablation, BDD100K, suite: milestone27_v26.json (+ milestone10 suite file for the index ceiling and lookup)
  regimes  milestone27_rater_sensitivity.json, milestone27_semantic_edge_cases.json,
           milestone27_compositional_synergy.json
"""
import json, re
from pathlib import Path
import numpy as np

R = Path("results/cvpr"); P = Path("paper")
J = lambda f: json.load(open(R/f))
f3 = lambda x: "--" if x is None else f"{x:.3f}"
sk = lambda x: "--" if x is None else f"${x:+.1f}$"
ATTRS = ["droplets", "glare", "wet_road", "low_contrast"]


def macro(A, key):
    return float(np.mean([A[a]["rankers"][key]["mAP"] for a in ATTRS]))


def sm_key(d, pats=(r"^SceneMiner_v2\.\d$", r"^v2\.\d_monocular$")):
    for p in pats:
        for k in d:
            if re.match(p, k): return k
    raise KeyError(f"no SceneMiner key in {list(d)}")


def design():
    rows = []
    m6 = J("milestone6_sceneminer_v2_1.json")["mode_B"]["attributes"]
    m10 = J("milestone10_sceneminer_v2_2.json")["mode_B"]["attributes"]
    s10 = J("milestone10_query_suite_150.json")["overall"]
    s11 = J("milestone11_domain_calibration.json")["rankers"]
    rq = J("milestone23_bdd_rawquery_routing.json")["rows"]
    rows.append(("naive convex router", macro(m6, "SceneMiner_v2.1_zeroleak"), macro(m6, "SceneMiner_v2.1_human_calibrated_5fold"),
                 rq["v21|raw"]["mAP"], s10["SceneMiner_v2.1"]["mAP"], s11["SceneMiner_v2.1"]["raw"]["skew_pts"]))
    rows.append(("$+$ anti-collapse routing, \\textsc{PartNorm}", macro(m10, "SceneMiner_v2.2_zeroleak"),
                 macro(m10, "SceneMiner_v2.2_human_calibrated_5fold"), rq["v22|raw"]["mAP"], s10["SceneMiner_v2.2"]["mAP"],
                 s11["SceneMiner_v2.2"]["raw"]["skew_pts"]))
    for lab, f, b, tag in [("$+$ \\textsc{ResNorm}, single-prompt gate", "milestone14_v2_3.json", "milestone23_bdd_rawquery_v23.json", "v2.3"),
                           ("$+$ contrastive gate, abstention", "milestone18_v2_4.json", "milestone23_bdd_rawquery_v24.json", "v2.4"),
                           ("$+$ factored gate", "milestone22_factored_gate.json", "milestone23_bdd_rawquery_v25.json", "v2.5"),
                           ("\\textbf{SceneMiner} (conditional context removal)", "milestone27_v26.json", None, "v2.3")]:
        d = J(f); A = d["A_human_gt"]["macro_mAP"]; C = d["C_suite150"]
        bd = J(b)["B_bdd100k"]["rankers"]["v2.3_full"]["mAP"] if b else d["B_bdd100k"]["rankers"]["v2.3_full"]["mAP"]
        rows.append((lab, A[f"{tag}_full"], A[f"{tag}_human_calibrated_5fold"], bd, C["overall"][f"{tag}_full"]["mAP"],
                     C["skew_pts"][f"{tag}_full"]["mean"]))
    best = [max(r[i] for r in rows) for i in (1, 2, 3, 4)]
    bsk = min(abs(r[5]) for r in rows)
    b = lambda x, i: f"\\best{{{x:.3f}}}" if round(x, 3) == round(best[i], 3) else f"${x:.3f}$"
    bs = lambda x: ("\\best{" + sk(x) + "}") if abs(x) == bsk else sk(x)
    lines = [f"{r[0]} & {b(r[1], 0)} / {b(r[2], 1)} & {b(r[3], 2)} & {b(r[4], 3)} / {bs(r[5])}" + "\\\\" for r in rows]
    tex = ["\\begin{tabular}{lccc}", "\\toprule", "design & human GT & BDD100K & suite / skew\\\\", "\\midrule",
           *lines, "\\bottomrule", "\\end{tabular}"]
    (P/"tab_design.tex").write_text("\n".join(tex) + "\n"); return rows


def ablation():
    d = J("milestone27_v26.json"); A = d["A_human_gt"]["macro_mAP"]; B = d["B_bdd100k"]["rankers"]; C = d["C_suite150"]
    rows = [("\\textbf{SceneMiner}", "full"), ("without metadata in $s_{\\mathrm{abst}}$", "no_macro_prior"),
            ("without abstention", "no_abstain"), ("without $\\gdeg$ on routing", "no_gate"),
            ("without \\textsc{ResNorm}", "no_resnorm")]
    lines = []
    for lab, k in rows:
        h = A.get(f"v2.3_{k}"); bd = B.get(f"v2.3_{k}", {}).get("mAP"); s = C["overall"].get(f"v2.3_{k}", {}).get("mAP")
        skw = C["skew_pts"].get(f"v2.3_{k}", {}).get("mean")
        lines.append(f"{lab} & {f3(h)} & {f3(bd)} & {f3(s)} / {sk(skw)}\\\\")
    tex = ["\\begin{tabular}{lccc}", "\\toprule", "variant & human GT & BDD100K & suite / skew\\\\", "\\midrule",
           *lines, "\\bottomrule", "\\end{tabular}"]
    (P/"tab_abl.tex").write_text("\n".join(tex) + "\n")


def bdd():
    B = J("milestone27_v26.json")["B_bdd100k"]["rankers"]
    rows = [("\\textbf{SceneMiner}", "v2.3_full"), ("zero-shot SigLIP2", "zeroshot_SigLIP2"), ("logit VLM lookup", "logit_VLM_lookup")]
    cols = ["mAP", "mAP_in_vocab", "mAP_out_of_vocab", "P@5", "NDCG@10"]
    best = {c: max(B[k][c] for _, k in rows) for c in cols}
    cell = lambda k, c: f"\\best{{{B[k][c]:.3f}}}" if round(B[k][c], 3) == round(best[c], 3) else f"{B[k][c]:.3f}"
    lines = [f"{lab} & " + " & ".join(cell(k, c) for c in cols) + "\\\\" for lab, k in rows]
    tex = ["\\begin{tabular}{lccccc}", "\\toprule", "ranker & all & in-voc & OOV & P@5 & NDCG@10\\\\", "\\midrule",
           *lines, "\\bottomrule", "\\end{tabular}"]
    (P/"tab_bdd.tex").write_text("\n".join(tex) + "\n")


def suite():
    C = J("milestone27_v26.json")["C_suite150"]; S10 = J("milestone10_query_suite_150.json")
    st = ["sensor_degradation", "macro_environmental", "compositional_heldout", "negated"]
    rows = [("attribute index (ceiling)", None, "attribute_index(TAUTOLOGY)"), ("\\textbf{SceneMiner}", "v2.3_full", None),
            ("\\enspace without metadata in $s_{\\mathrm{abst}}$", "v2.3_no_macro_prior", None),
            ("\\enspace without abstention", "v2.3_no_abstain", None), ("zero-shot SigLIP2", "zeroshot_SigLIP2", None),
            ("logit VLM lookup", None, "logit_VLM_lookup")]
    vals = []
    for lab, k, k10 in rows:
        if k is not None:
            v = [C["per_stratum"][s][k] for s in st] + [C["overall"][k]["mAP"]]; s_ = C["skew_pts"][k]["mean"]
        else:
            v = [S10["per_stratum"][s][k10]["mAP"] if isinstance(S10["per_stratum"][s][k10], dict) else S10["per_stratum"][s][k10]
                 for s in st] + [S10["overall"][k10]["mAP"]]
            s_ = J("milestone11_domain_calibration.json")["rankers"].get(k10, {}).get("raw", {}).get("skew_pts")
        vals.append((lab, v, s_, k10 == "attribute_index(TAUTOLOGY)"))
    best = [max(v[i] for _, v, _, ceil in vals if not ceil) for i in range(5)]
    lines = []
    for lab, v, s_, ceil in vals:
        cs = [f"{x:.3f}" if ceil or round(x, 3) != round(best[i], 3) else f"\\best{{{x:.3f}}}" for i, x in enumerate(v)]
        lines.append(f"{lab} & " + " & ".join(cs) + f" & {'--' if ceil else sk(s_)}\\\\")
    tex = ["\\begin{tabular}{lcccccc}", "\\toprule", "ranker & S1 & S2 & S3 & S4 & all & skew\\\\", "\\midrule",
           *lines, "\\bottomrule", "\\end{tabular}"]
    (P/"tab_suite.tex").write_text("\n".join(tex) + "\n")


def regimes():
    rat = J("milestone27_rater_sensitivity.json")["full_550"]
    sem = J("milestone27_semantic_edge_cases.json")
    syn = J("milestone27_compositional_synergy.json")["summary"]["all"]
    sc = sem["corpus_258"]["summary"]["semantic_atomic"]; sv = sem["val150_clean"]["summary"]["semantic_atomic"]
    # the final model's rows: the evaluation scripts keep their historical labels ("v2.4_*",
    # "SceneMiner_v2.4") for whichever checkpoint family SM_FINAL / --final selects (v2.6 here);
    # "v2.3_*" rows in the semantic files are the older v2.3 model, kept for reference
    kr, ks, kv, ky = "SceneMiner_v2.4", "v2.4_monocular", "v2.4_portable_base", "v2.4_monocular"
    for d_, k_ in ((rat, kr), (sc, ks), (sv, kv), (syn, ky)): assert k_ in d_, k_
    g = lambda d, k: (d[k].get("mAP", d[k].get("macro")) if isinstance(d[k], dict) else d[k]) if k in d else None
    reg = {"atomic_degradation_550": {"SceneMiner": g(rat, kr), **{k: g(rat, k) for k in rat if k != kr}},
           "atomic_semantics_corpus258": {"SceneMiner": g(sc, ks), "zeroshot_SigLIP2": g(sc, "zeroshot_SigLIP2"),
                                          "logit_VLM_lookup": g(sc, "logit_VLM_lookup")},
           "atomic_semantics_val150": {"SceneMiner": g(sv, kv), "zeroshot_SigLIP2": g(sv, "zeroshot_SigLIP2"),
                                       "logit_VLM_lookup": g(sv, "logit_VLM_lookup")},
           "conjunctions_corpus258": {"SceneMiner": g(syn, ky), **{k: g(syn, k) for k in syn if k != ky}}}
    # on object-only queries the lookup and the metadata are constant, so both late fusions
    # reduce exactly to SigLIP2 (shown as "= SigLIP2" in the table)
    for r_ in ("atomic_semantics_corpus258", "atomic_semantics_val150"):
        for k_ in ("late_fusion SigLIP2+lookup", "late_fusion SigLIP2+metadata"): reg[r_][k_] = reg[r_]["zeroshot_SigLIP2"]
    json.dump({"doc": "Three human-graded regimes, assembled by scripts/make_main_tables.py", "keys": [kr, ks, kv, ky],
               "regimes": reg}, open(R/"milestone27_regimes.json", "w"), indent=1)
    rows = [("\\textbf{SceneMiner}", "SceneMiner"), ("zero-shot SigLIP2", "zeroshot_SigLIP2"),
            ("logit VLM lookup", "logit_VLM_lookup"), ("fusion SigLIP2 $+$ lookup", "late_fusion SigLIP2+lookup"),
            ("fusion SigLIP2 $+$ metadata", "late_fusion SigLIP2+metadata")]
    col = lambda r, k: reg[r].get(k)
    D, SC, SV, CJ = "atomic_degradation_550", "atomic_semantics_corpus258", "atomic_semantics_val150", "conjunctions_corpus258"
    bd = max(v for v in reg[D].values() if v is not None); bc = max(v for v in reg[SC].values() if v is not None)
    bv = max(v for v in reg[SV].values() if v is not None); bj = max(v for v in reg[CJ].values() if v is not None)
    B = lambda x, b: "--" if x is None else (f"\\best{{{x:.3f}}}" if round(x, 3) == round(b, 3) else f"{x:.3f}")
    lines = []
    for lab, k in rows:
        sem_c = "$=$ SigLIP2" if k.startswith("late_fusion") else f"{B(col(SC, k), bc)} / {B(col(SV, k), bv)}"
        lines.append(f"{lab} & {B(col(D, k), bd)} & {sem_c} & {B(col(CJ, k), bj)}\\\\")
    tex = ["\\begin{tabular}{lccc}", "\\toprule", "ranker & degradation & semantics & sem.$\\times$cond.\\\\",
           " & ($550$) & ($258$ / val150) & ($258$)\\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]
    (P/"tab_regimes.tex").write_text("\n".join(tex) + "\n"); return reg


def loop():
    """paper/tab_loop.tex: top-25 front-sector metric minus the val150 mean, 62 degradation queries."""
    T = J("milestone27_val150_retrieval_v26.json")["topk"]["degradation"]
    rows = [("\\textbf{SceneMiner}", "SceneMiner_v2.4"), ("VLM margin lookup", "logit_VLM_lookup"),
            ("night/rain metadata", "metadata_matched"), ("zero-shot SigLIP2", "zeroshot_SigLIP2"), ("random", "random")]
    cols = ["mAP_front", "fp_rate", "recall_far", "vru_recall"]
    cell = lambda r: f"${r['delta']:+.3f}$" + ("$^{*}$" if r.get("p_worse", 1) < 0.05 else "")
    lines = [f"{lab} & " + " & ".join(cell(T[k][c]) for c in cols) + "\\\\" for lab, k in rows]
    tex = ["\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{lcccc}", "\\toprule",
           "ranker & mAP & ghost FP & rec.\\,$>30$\\,m & VRU rec.\\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]
    (P/"tab_loop.tex").write_text("\n".join(tex) + "\n")


def holdout_supp():
    """paper/supp_tables/holdout_table.tex: the untouched BDD100K holdout, per family, both modes."""
    H = J("milestone28_related_work_comparison.json")["holdout"]["modes"]
    rows = [("SceneMiner", "SceneMiner (portable)"), ("SigLIP2 ViT-B/16", "SigLIP2 ViT-B/16"),
            ("SigLIP ViT-B/16", "SigLIP ViT-B/16"), ("CLIP ViT-B/32", "CLIP ViT-B/32"),
            ("ReferGPT-style", "ReferGPT-style state + CLIP"), ("CARIM-style (neutral)", "CARIM-style caption retrieval"),
            ("SigLIP2 + adapter", "SigLIP2 + fine-tuned adapter"), ("Qwen2.5-VL lookup", "Qwen2.5-VL lookup"),
            ("naive convex router", "Naive uncalibrated fusion"), ("RefAV-style (LLM)", "RefAV-style programmatic filter")]
    out = ["\\begin{tabular}{lcccccc}", "\\toprule",
           " & all & (a) cond. & (b) object & (c) obj.$\\times$cond. & P@5 & $\\Delta$ vs.\\ SceneMiner ($p$)\\\\"]
    for mode, title in [("metadata_free", "metadata-free (main)"), ("metadata_equipped_circular", "metadata-equipped (circular)")]:
        M = H[mode]["methods"]; PV = H[mode]["paired_vs_sceneminer_portable"]
        out += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}}\\\\"]
        for lab, k in rows:
            v = M.get(k)
            if not v: continue
            pp = PV[k]["all"]["perm_p"] if k != "SceneMiner (portable)" else None
            pv = "--" if pp is None else (f"${-PV[k]['all']['delta']:+.3f}$ " + ("($<0.001$)" if pp < 0.001 else f"(${pp:.3f}$)"))
            out.append(f"{lab} & {v['mAP_all']:.3f} & {v['mAP_a']:.3f} & {v['mAP_b']:.3f} & {v['mAP_c']:.3f} & "
                       f"{v['P@5_all']:.3f} & {pv}\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (P/"supp_tables"/"holdout_table.tex").write_text("\n".join(out) + "\n")


def rater_supp():
    """paper/supp_tables/rater_table.tex and rater_attr_table.tex (single-rater sensitivity, v2.6)."""
    r = J("milestone27_rater_sensitivity.json"); S = "SceneMiner_v2.4"
    full, be, mc = r["full_550"], r["boundary_excluded"], r["monte_carlo"]
    m = lambda d, k: d[k]["macro"]
    L = ["\\begin{tabular}{lccc}", "\\toprule", "subset & SceneMiner & lookup & SigLIP2\\\\", "\\midrule"]
    for lab, d in [("all $550$", full), ("drop $10\\%$ most ambiguous", be["drop_10pct"]), ("drop $20\\%$ most ambiguous", be["drop_20pct"])]:
        L.append(f"{lab} & ${m(d, S):.3f}$ & ${m(d, 'logit_VLM_lookup'):.3f}$ & ${m(d, 'zeroshot_SigLIP2'):.3f}$\\\\")
    L += ["\\midrule", "second rater: Cohen's $\\kappa$ & $>$lookup & $>$SigLIP2 & $>$fusion$^{\\ast}$\\\\", "\\midrule"]
    for kp in ("0.9", "0.85", "0.75"):
        w = lambda c, mode: mc[f"kappa_{kp}_{mode}"]["win_rate"][f"{S} > {c}"]["macro"]
        cells = [f"${w(c, 'uniform'):.2f}$ / ${w(c, 'clip_boundary'):.2f}$" for c in ("logit_VLM_lookup", "zeroshot_SigLIP2", "late_fusion SigLIP2+lookup")]
        L.append(f"$\\kappa={kp}$ (uniform / ambiguity) & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule", "\\multicolumn{4}{l}{\\footnotesize $^{\\ast}$late fusion SigLIP2 $+$ lookup.}", "\\end{tabular}"]
    (P/"supp_tables"/"rater_table.tex").write_text("\n".join(L) + "\n")
    A = ["\\begin{tabular}{lcccc}", "\\toprule",
         " & \\multicolumn{2}{c}{vs.\\ lookup} & \\multicolumn{2}{c}{vs.\\ SigLIP2 $+$ lookup}\\\\",
         "attribute & $\\kappa=0.85$ & $\\kappa=0.75$ & $\\kappa=0.85$ & $\\kappa=0.75$\\\\", "\\midrule"]
    for at, lab in [("droplets", "droplets"), ("glare", "glare"), ("wet_road", "wet road"), ("low_contrast", "low contrast"), ("macro", "macro")]:
        g = lambda kp, mode, c: mc[f"kappa_{kp}_{mode}"]["win_rate"][f"{S} > {c}"][at]
        A.append(f"{lab} & ${g('0.85','uniform','logit_VLM_lookup'):.2f}$ & ${g('0.75','clip_boundary','logit_VLM_lookup'):.2f}$ & "
                 f"${g('0.85','uniform','late_fusion SigLIP2+lookup'):.2f}$ & ${g('0.75','clip_boundary','late_fusion SigLIP2+lookup'):.2f}$\\\\")
    A += ["\\bottomrule", "\\end{tabular}"]
    (P/"supp_tables"/"rater_attr_table.tex").write_text("\n".join(A) + "\n")


def rater2():
    """paper/tab_rater2.tex (per attribute: two-rater agreement, VLM-human agreement, mAP
    graded by rater 2) and paper/supp_tables/rater2_methods_table.tex (every method under
    each grading of the 160 double-labelled scenes). Source: milestone33_rater2_results.json."""
    d = J("milestone33_rater2_results.json"); A, M = d["attributes"], d["macro"]
    cols = [("SceneMiner", "SceneMiner"), ("Qwen2.5-VL lookup", "lookup"),
            ("Late fusion SigLIP2 + lookup", "fusion$^{\\ast}$"), ("SigLIP2 ViT-B/16", "SigLIP2")]
    L = ["\\begin{tabular}{lccc|cccc}", "\\toprule",
         " & \\multicolumn{3}{c|}{Cohen's $\\kappa$} & \\multicolumn{4}{c}{mAP graded by rater 2}\\\\",
         "attribute & r1--r2 & VLM--r1 & VLM--r2 & " + " & ".join(c for _, c in cols) + "\\\\", "\\midrule"]
    for at, lab in [("droplets", "droplets"), ("glare", "glare"), ("wet_road", "wet road"), ("low_contrast", "low contrast")]:
        r = A[at]; best = max(r["mAP"][k]["rater2"] for k, _ in cols)
        cell = lambda k: (f"\\best{{{r['mAP'][k]['rater2']:.3f}}}" if r["mAP"][k]["rater2"] == best else f"{r['mAP'][k]['rater2']:.3f}")
        L.append(f"{lab} & {r['human_human']['kappa']:.3f} & {r['vlm_vs_rater1']['kappa']:.3f} & {r['vlm_vs_rater2']['kappa']:.3f} & "
                 + " & ".join(cell(k) for k, _ in cols) + "\\\\")
    best = max(M["mAP"][k]["rater2"] for k, _ in cols)
    L += ["\\midrule", f"macro & {M['human_human']['kappa']:.3f} & {M['vlm_vs_rater1_kappa']:.3f} & {M['vlm_vs_rater2_kappa']:.3f} & "
          + " & ".join((f"\\best{{{M['mAP'][k]['rater2']:.3f}}}" if M["mAP"][k]["rater2"] == best else f"{M['mAP'][k]['rater2']:.3f}") for k, _ in cols)
          + "\\\\", "\\bottomrule", "\\end{tabular}"]
    (P/"tab_rater2.tex").write_text("\n".join(L) + "\n")
    rows = [("BEV-TSR-style", "BEV-TSR-style text-to-BEV"), ("RefAV-style (LLM)", "RefAV-style programmatic filter"),
            ("CARIM-style (neutral)", "CARIM-style caption retrieval"), ("CARIM-style (listing)", "CARIM-style (enumerating prompt, M23)"),
            ("Tag map", "Tag-map (Talk2BEV/nuPrompt-style)"), ("ReferGPT-style", "ReferGPT-style state + CLIP"),
            ("CLIP ViT-B/32", "CLIP ViT-B/32"), ("SigLIP ViT-B/16", "SigLIP ViT-B/16"), ("SigLIP2 ViT-B/16", "SigLIP2 ViT-B/16"),
            ("SigLIP2 + adapter", "SigLIP2 + fine-tuned adapter"), ("Qwen2.5-VL lookup", "Qwen2.5-VL lookup"),
            ("naive convex router", "Naive uncalibrated fusion"), ("late fusion SigLIP2 + lookup", "Late fusion SigLIP2 + lookup"),
            ("SceneMiner (portable)", "SceneMiner (portable)"), ("SceneMiner", "SceneMiner")]
    G = ["rater2", "rater1", "consensus", "union", "intersection"]
    best = {g: max(M["mAP"][k][g] for _, k in rows) for g in G}
    T = ["\\begin{tabular}{lccccc}", "\\toprule",
         "method & rater 2 & rater 1 & consensus & union & intersection\\\\", "\\midrule"]
    for lab, k in rows:
        v = M["mAP"][k]
        T.append(f"{lab} & " + " & ".join((f"\\best{{{v[g]:.3f}}}" if v[g] == best[g] else f"{v[g]:.3f}") for g in G) + "\\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (P/"supp_tables"/"rater2_methods_table.tex").write_text("\n".join(T) + "\n")


if __name__ == "__main__":
    for f in (design, ablation, bdd, suite, regimes, loop, holdout_supp, rater_supp, rater2):
        try: f(); print("ok", f.__name__)
        except Exception as e: print("FAILED", f.__name__, repr(e))
