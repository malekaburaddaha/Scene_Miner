#!/usr/bin/env python3
"""MILESTONE 31 - check the numbers printed in the paper against the result files.

Two layers:
 1. REGISTRY - headline claims tied to an exact JSON path. The value is formatted at
    the paper's precision and must appear in the named LaTeX file; any mismatch
    fails loudly (exit 1).
 2. PROVENANCE SCAN - every other decimal number in paper/main_cvpr.tex and the
    paper/tab_*.tex tables is looked up among all numeric values in
    results/cvpr/*.json, at its displayed precision (also as a percentage /
    "points" value). Numbers with no source are listed as UNSOURCED; this is a
    report, since a coincidental match cannot be excluded, but an unsourced number
    is always a bug to look at.
Limitation: registry values are matched as substrings of the LaTeX text, so a value
can also be "found" inside a longer number elsewhere (e.g. 0.070 inside 0.0701); the
milestone notes list the claims that were checked by hand for this.
Writes results/cvpr/milestone31_number_check.json.
"""
import glob, json, re, sys
from pathlib import Path

R = Path("results/cvpr")


def get(f, path):  # noqa: E302
    d = json.load(open(R/f))
    for k in path: d = d[k]
    return d


def mean_of(f, path, keys):
    d = get(f, path); return sum(d[k] for k in keys) / len(keys)


# (tex file, formatted string that must appear, json file, key path, format) ----------------
def registry():
    """(tex file, value-producing function, format, note). Each formatted value must
    appear in the tex file; derived values are computed from the JSON here."""
    M = lambda *p: get("milestone28_related_work_comparison.json", list(p))
    V = lambda *p: get("milestone27_v26.json", list(p))
    L = lambda *p: get("milestone27_latency.json", list(p))
    S = lambda *p: get("milestone27_semantic_edge_cases.json", list(p))
    Y = lambda *p: get("milestone27_compositional_synergy.json", list(p))
    RG = lambda *p: get("milestone27_regimes.json", list(p))
    RT = lambda *p: get("milestone27_rater_sensitivity.json", list(p))
    LP = lambda *p: get("milestone27_val150_retrieval_v26.json", list(p))
    OC = lambda *p: get("milestone32_object_cap_train.json", list(p))
    R2 = lambda *p: get("milestone33_rater2_results.json", list(p))
    VR = lambda *p: get("milestone34_per_query_vlm.json", list(p))
    BB = lambda *p: get("milestone34_larger_backbone.json", list(p))
    SDV = lambda *p: get("milestone34_seed_sd.json", ["rows"] + list(p))
    ATTR4 = ("droplets", "glare", "wet_road", "low_contrast")
    HP = lambda k, sc="all": M("holdout", "modes", "metadata_free", "paired_vs_sceneminer_portable", k, sc)
    T, MT, SP = "paper/main_cvpr.tex", "paper/tab_master.tex", "paper/supp_cvpr.tex"
    f3, p2, p3, s3 = "{:.3f}", "{:.2f}", "{:.3f}", "{:+.3f}"
    reg = [
        (T, lambda: V("A_human_gt", "macro_mAP", "v2.3_full"), f3, "human GT zero-leakage"),
        (T, lambda: V("A_human_gt", "macro_mAP", "v2.3_human_calibrated_5fold"), f3, "human GT calibrated"),
        (T, lambda: M("methods", "SceneMiner (portable)", "human_gt"), f3, "human GT portable"),
        (T, lambda: M("methods", "Naive uncalibrated fusion", "human_gt"), f3, "best alternative human GT"),
        (T, lambda: V("B_bdd100k", "rankers", "v2.3_full", "mAP"), f3, "BDD100K SceneMiner"),
        (T, lambda: V("B_bdd100k", "rankers", "zeroshot_SigLIP2", "mAP"), f3, "BDD100K SigLIP2"),
        (T, lambda: V("B_bdd100k", "rankers", "v2.3_full", "mAP_in_vocab") - V("B_bdd100k", "rankers", "zeroshot_SigLIP2", "mAP_in_vocab"), s3, "BDD in-vocab gain"),
        (T, lambda: -(V("B_bdd100k", "rankers", "v2.3_full", "mAP_out_of_vocab") - V("B_bdd100k", "rankers", "zeroshot_SigLIP2", "mAP_out_of_vocab")), f3, "BDD OOV loss"),
        (T, lambda: M("holdout", "modes", "metadata_free", "methods", "SceneMiner (portable)", "mAP_all"), f3, "holdout SceneMiner"),
        (T, lambda: M("holdout", "modes", "metadata_free", "methods", "SigLIP2 ViT-B/16", "mAP_all"), f3, "holdout SigLIP2"),
        (T, lambda: HP("SigLIP2 ViT-B/16")["perm_p"], p2, "holdout p vs SigLIP2"),
        (T, lambda: HP("CLIP ViT-B/32")["delta"], s3, "holdout vs CLIP"),
        (T, lambda: HP("CARIM-style caption retrieval")["delta"], s3, "holdout vs CARIM"),
        (T, lambda: HP("Qwen2.5-VL lookup")["delta"], s3, "holdout vs lookup"),
        (T, lambda: HP("Naive uncalibrated fusion")["delta"], s3, "holdout vs naive"),
        (T, lambda: HP("RefAV-style programmatic filter")["delta"], s3, "holdout vs RefAV"),
        (T, lambda: L("online", "rank_graph_1"), "{:.1f}", "ranking ms"),
        (T, lambda: L("online", "prepare_1"), "{:.1f}", "query preparation ms"),
        (T, lambda: L("online", "full_1"), "{:.1f}", "total ms"),
        (T, lambda: L("online", "reference_1"), "{:.1f}", "reference ms"),
        (T, lambda: V("A_human_gt", "attributes", "droplets", "rankers", "v2.3_full", "mAP"), f3, "droplets"),
        (T, lambda: V("A_human_gt", "attributes", "glare", "rankers", "v2.3_full", "mAP"), f3, "glare"),
        (T, lambda: V("A_human_gt", "attributes", "wet_road", "rankers", "v2.3_full", "mAP"), f3, "wet road"),
        (T, lambda: V("A_human_gt", "attributes", "low_contrast", "rankers", "v2.3_full", "mAP"), f3, "low contrast"),
        (T, lambda: V("A_human_gt", "macro_mAP", "v2.3_full") - V("A_human_gt", "macro_mAP", "v2.3_no_resnorm"), f3, "ResNorm worth"),
        (T, lambda: V("A_human_gt", "macro_mAP", "v2.3_full") - M("methods", "SceneMiner (portable)", "human_gt"), f3, "BEV/tags over portable"),
        (T, lambda: V("C_suite150", "overall", "v2.3_full", "mAP"), f3, "suite"),
        ("paper/tab_abl.tex", lambda: V("C_suite150", "skew_pts", "v2.3_full", "mean"), "{:+.1f}", "suite skew"),
        ("paper/tab_suite.tex", lambda: V("C_suite150", "per_stratum", "macro_environmental", "v2.3_full"), f3, "S2"),
        (T, lambda: V("C_suite150", "per_stratum", "compositional_heldout", "v2.3_full"), f3, "S3"),
        ("paper/tab_suite.tex", lambda: V("C_suite150", "per_stratum", "compositional_heldout", "v2.3_no_abstain"), f3, "S3 un-gated"),
        (T, lambda: RG("regimes", "atomic_degradation_550", "SceneMiner") - RG("regimes", "atomic_degradation_550", "late_fusion SigLIP2+lookup"), s3, "regime degradation lead"),
        (T, lambda: S("corpus_258", "tests", "v2.4_monocular vs zeroshot_SigLIP2 | semantic_atomic", "delta"), s3, "semantics vs SigLIP2"),
        (T, lambda: S("corpus_258", "tests", "v2.4_monocular vs logit_VLM_lookup | semantic_atomic", "delta"), s3, "semantics vs lookup"),
        (T, lambda: Y("tests", "v2.4_monocular - late_fusion SigLIP2+metadata", "delta"), s3, "conjunctions vs SigLIP2+metadata"),
        (T, lambda: Y("tests", "v2.4_monocular - late_fusion SigLIP2+metadata", "perm_p"), p3, "conjunctions p"),
        (T, lambda: -LP("topk", "degradation", "SceneMiner_v2.4", "mAP_front", "delta"), f3, "loop mAP"),
        (T, lambda: -LP("topk", "degradation", "SceneMiner_v2.4", "vru_recall", "delta"), f3, "loop VRU recall"),
        (T, lambda: LP("pairwise", "SceneMiner_v2.4 - zeroshot_SigLIP2 | mAP_front", "delta"), "{:.3f}", "loop vs SigLIP2"),
        (T, lambda: get("milestone23_bdd_rawquery_v24.json", ["B_bdd100k", "rankers", "v2.3_full", "mAP"]) - V("B_bdd100k", "rankers", "v2.3_full", "mAP"), f3, "design BDD trade"),
        (T, lambda: V("C_suite150", "overall", "v2.3_full", "mAP") - get("milestone18_v2_4.json", ["C_suite150", "overall", "v2.4_full", "mAP"]), s3, "design suite gain"),
        (T, lambda: Y("tests", "v2.4_monocular - logit_VLM_lookup", "delta"), s3, "conjunctions vs lookup"),
        (T, lambda: Y("tests", "v2.4_monocular - zeroshot_SigLIP2", "delta"), s3, "conjunctions vs SigLIP2"),
        (SP, lambda: OC("detectors", OC("selected"), "recall_synthetic"), p2, "object cap recall (rejected)"),
        (SP, lambda: 100 * OC("detectors", OC("selected"), "false_flag_degradation"), "{:.1f}", "object cap false-flag %"),
        (T, lambda: R2("macro", "human_human", "kappa"), f3, "rater2 kappa"),
        (T, lambda: R2("macro", "human_human", "kappa"), p2, "rater2 kappa (abstract)"),
        (T, lambda: min(R2("attributes", a, "human_human", "kappa") for a in ATTR4), f3, "rater2 kappa min attr"),
        (T, lambda: max(R2("attributes", a, "human_human", "kappa") for a in ATTR4), f3, "rater2 kappa max attr"),
        (T, lambda: R2("macro", "by_stratum_kappa", "vlm_boundary"), f3, "rater2 kappa boundary"),
        (T, lambda: R2("macro", "vlm_vs_rater1_kappa"), f3, "VLM-rater1 kappa"),
        (T, lambda: R2("macro", "vlm_vs_rater2_kappa"), f3, "VLM-rater2 kappa"),
        (T, lambda: R2("macro", "mAP", "SceneMiner", "rater2"), f3, "SceneMiner graded by rater 2"),
        (T, lambda: R2("paired_bootstrap_rater2", "SigLIP2 ViT-B/16", "delta"), s3, "rater2 lead vs SigLIP2"),
        (T, lambda: R2("paired_bootstrap_rater2", "Late fusion SigLIP2 + lookup", "delta"), s3, "rater2 lead vs fusion"),
        (T, lambda: R2("paired_bootstrap_rater2", "Qwen2.5-VL lookup", "delta"), s3, "rater2 lead vs lookup"),
        (T, lambda: VR("benchmarks", "bdd", "mAP_rerank"), f3, "reranker BDD100K"),
        (T, lambda: -VR("benchmarks", "holdout", "sceneminer_minus_rerank", "delta"), s3, "reranker holdout lead"),
        (T, lambda: VR("benchmarks", "holdout", "sceneminer_minus_rerank", "perm_p"), p3, "reranker holdout p"),
        (T, lambda: -VR("benchmarks", "s5_corpus_atomic", "sceneminer_minus_rerank", "delta"), s3, "reranker S5 atomic lead"),
        (T, lambda: max(VR("benchmarks", "s5_corpus_atomic", "sceneminer_minus_rerank", "perm_p"), 0.001), p3, "reranker S5 atomic p"),
        (T, lambda: VR("latency", "batch10_ms_median") / 1000, "{:.1f}", "reranker seconds per query"),
        (T, lambda: round(VR("latency", "batch10_ms_median") / L("online", "full_1"), -2), "{:.0f}", "reranker cost ratio"),
        (T, lambda: VR("benchmarks", "human_gt", "scene_bootstrap", "delta_sceneminer_minus_rerank"), s3, "SceneMiner lead over reranker, human GT"),
        (T, lambda: VR("benchmarks", "human_gt", "scene_bootstrap", "ci95")[0], f3, "reranker bootstrap CI low"),
        (T, lambda: VR("benchmarks", "human_gt", "scene_bootstrap", "ci95")[1], f3, "reranker bootstrap CI high"),
        (T, lambda: VR("benchmarks", "human_gt", "scene_bootstrap", "per_attribute", "low_contrast", "rerank"), f3, "reranker low contrast"),
        (T, lambda: VR("benchmarks", "human_gt", "scene_bootstrap", "per_attribute", "low_contrast", "SceneMiner"), f3, "SceneMiner low contrast"),
        (T, lambda: [x for x in VR("benchmarks", "human_gt", "per_query") if x["id"] == "low_contrast"][0]["pos_in_top50"], "{:d}", "low-contrast positives in top 50"),
        (T, lambda: [x for x in VR("benchmarks", "human_gt", "per_query") if x["id"] == "low_contrast"][0]["n_pos"], "{:d}", "low-contrast positives"),
        ("paper/tab_design.tex", lambda: f"0.794}}_{{\\pm{SDV('T1|v2.6', 'human_gt', 'sd'):.3f}".replace("pm0.", "pm."), "{}", "SceneMiner human GT seed sd"),
        ("paper/tab_master.tex", lambda: f"0.794}}_{{\\pm{SDV('T2|SceneMiner', 'human_gt', 'sd'):.3f}".replace("pm0.", "pm."), "{}", "Table 2 SceneMiner seed sd"),
        (MT, lambda: BB("rows", "SigLIP2 ViT-B/16", "human_gt"), f3, "zero-shot B/16 human GT"),
        (T, lambda: BB("rows", "SigLIP2 So400m/14", "human_gt"), f3, "zero-shot So400m human GT"),
        (T, lambda: BB("rows", "SigLIP2 ViT-B/16", "holdout"), f3, "zero-shot B/16 holdout"),
        ("paper/supp_tables/backbone_table.tex", lambda: BB("rows", "SigLIP2 So400m/14", "holdout"), f3, "zero-shot So400m holdout"),
        (T, lambda: BB("rows", "SceneMiner, So400m/14", "human_gt"), f3, "SceneMiner So400m human GT"),
        (T, lambda: BB("rows", "SceneMiner, So400m/14", "holdout"), f3, "SceneMiner So400m holdout"),
        (T, lambda: BB("rows", "SceneMiner, So400m/14", "bdd"), f3, "SceneMiner So400m BDD100K"),
        (T, lambda: RG("regimes", "atomic_degradation_550", "late_fusion SigLIP2+lookup"), f3, "best no-learning control (abstract)"),
        (T, lambda: M("n", "bdd_images") + M("holdout", "modes", "metadata_free", "n_images"), "{:d}", "external images (BDD100K + holdout)"),
        (MT, lambda: M("methods", "SceneMiner", "human_gt"), f3, "Table 2 full human GT"),
        ("paper/tab_master_full.tex", lambda: M("methods", "SceneMiner (portable)", "suite"), f3, "Table 2 portable suite"),
        # Milestone 39: main-table means and concept-gate examples
        (MT, lambda: get("milestone39_main_table.json", ["mean", "SceneMiner"]), f3, "Table 2 mean SceneMiner"),
        (MT, lambda: get("milestone39_main_table.json", ["mean", "VLM_RERANK"]), f3, "Table 2 mean reranker"),
        (T, lambda: get("milestone39_main_table.json", ["mean", "SceneMiner"]), f3, "text mean SceneMiner"),
        (T, lambda: get("milestone39_main_table.json", ["mean", "SigLIP2 ViT-B/16"]), f3, "text mean SigLIP2"),
        (T, lambda: [r for r in get("milestone39_gate_examples.json", ["rows"]) if r["query"] == "slick pavement in the dark"][0]["gamma_deg"], f3, "gate example slick pavement"),
        ("paper/tab_gate.tex", lambda: [r for r in get("milestone39_gate_examples.json", ["rows"]) if r["query"] == "snow covering the road"][0]["gamma_deg"], f3, "gate example snow"),
    ]
    return reg


def all_json_values():
    vals = []
    for f in glob.glob(str(R/"*.json")):
        try: d = json.load(open(f))
        except Exception: continue
        stack = [(d, f)]
        while stack:
            x, src = stack.pop()
            if isinstance(x, dict): stack += [(v, src) for v in x.values()]
            elif isinstance(x, list): stack += [(v, src) for v in x[:5000]]
            elif isinstance(x, (int, float)) and not isinstance(x, bool): vals.append((float(x), Path(src).name))
    return vals


NUM = re.compile(r"(?<![\w.])[-+−]?\d+\.\d+")


def strip_tex(t):
    t = re.sub(r"(?m)%.*$", "", t)
    t = re.sub(r"\\(cite|cref|Cref|ref|label|includegraphics|input|usepackage|documentclass)\*?(\[[^\]]*\])?\{[^}]*\}", "", t)
    return t


def found(s, txt):
    """s occurs as a whole number: not preceded by a digit or point, not followed by a digit."""
    return re.search(r"(?<![\d.])" + re.escape(s) + r"(?!\d)", txt) is not None


def main():
    bad = []; rep = {"registry": [], "scan": {}}
    for tex, fn, fmt, note in registry():
        try: v = fn()
        except Exception as e:
            bad.append(f"{note}: source missing ({e!r})"); continue
        s = fmt.format(v)
        txt = Path(tex).read_text().replace("$", "").replace("\\,", "")
        ok = any(found(v_, txt) for v_ in (s, s.lstrip("+"), s.replace("-", "")))
        rep["registry"].append({"claim": note, "value": s, "tex": tex, "found": ok})
        if not ok: bad.append(f"{note}: {s} not found in {tex}")
    vals = all_json_values()
    files = [Path("paper/main_cvpr.tex")] + sorted(Path("paper").glob("tab_*.tex"))
    for p in files:
        nums = NUM.findall(strip_tex(p.read_text()))
        uns = []
        for n in nums:
            x = float(n.replace("−", "-")); dp = len(n.split(".")[1])
            tol = 0.5 * 10 ** (-dp) + 1e-12
            hit = any(abs(v - x) <= tol or abs(abs(v) - abs(x)) <= tol or abs(100 * v - x) <= tol or abs(100 * abs(v) - abs(x)) <= tol
                      for v, _ in vals)
            if not hit: uns.append(n)
        rep["scan"][str(p)] = {"n_numbers": len(nums), "n_unsourced": len(uns), "unsourced": sorted(set(uns))}
        print(f"{p}: {len(nums)} numbers, {len(uns)} without a source in results/cvpr/*.json"
              + (f": {sorted(set(uns))}" if uns else ""))
    for r in rep["registry"]:
        print(("ok   " if r["found"] else "FAIL ") + f"{r['claim']:<32} {r['value']}  ({r['tex']})")
    json.dump(rep, open(R/"milestone31_number_check.json", "w"), indent=1)
    if bad:
        print("\nREGISTRY FAILURES:\n  " + "\n  ".join(bad)); sys.exit(1)
    print("REGISTRY PASSED")


if __name__ == "__main__":
    main()
