#!/usr/bin/env python3
"""MILESTONE 41 - ready-to-paste tables for the [TODO]s of the v2 paper, generated from the
result files only (no number is typed):
  -> results/cvpr/MILESTONE41_tables.tex (LaTeX tabular bodies) and MILESTONE41_tables.md
"""
import json
from pathlib import Path
import numpy as np

R = Path("results/cvpr")
J = lambda f: json.load(open(R/f))
f3 = lambda x: "--" if x is None else f"{x:.3f}"
ATTRS = ["droplets", "glare", "wet_road", "low_contrast"]
AL = ["droplets", "glare", "wet road", "low contr."]
TEX, MD = [], []


def table(name, header, rows, note=""):
    TEX.append(f"% ---- {name}\n" + (f"% {note}\n" if note else "") + " & ".join(header) + r" \\ \midrule" + "\n" +
               "\n".join(" & ".join(r) + r" \\" for r in rows) + "\n")
    MD.append(f"### {name}\n\n" + (note + "\n\n" if note else "") + "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" +
              "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n")


def p_str(p): return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    # ---------------- Table 2 (main comparison), late-fusion row added
    M28, LF, VR, SD = J("milestone28_related_work_comparison.json"), J("milestone41_late_fusion.json"), \
        J("milestone34_per_query_vlm.json"), J("milestone34_seed_sd.json")["rows"]
    LAT = J("milestone27_latency.json")

    def cols(M, k, per_query):
        m = M["methods"][k]; ho = M["holdout"]["modes"]["metadata_free"]["methods"].get(k)
        a, c = per_query[k].get("s5_corpus_atomic"), per_query[k].get("s5_corpus_compositional")
        s5 = None if (not a or None in a or None in c) else float(np.mean(a + c))
        return [m["human_gt"], m["bdd"], ho["mAP_all"] if ho else None, s5]
    rows = [("BEV-TSR-style [44]", "BEV", "BEV-TSR-style text-to-BEV", True), ("RefAV-style [10]", "VLM, tags, meta", "RefAV-style programmatic filter", True),
            ("CARIM-style [22]", "captions", "CARIM-style caption retrieval", False),
            ("Talk2BEV/nuPrompt tags [9, 48]", "BEV tags", "Tag-map (Talk2BEV/nuPrompt-style)", True),
            ("ReferGPT-style [6]", "image, tags, meta", "ReferGPT-style state + CLIP", True), ("CLIP ViT-B/32 [35]", "image", "CLIP ViT-B/32", False),
            ("SigLIP ViT-B/16 [54]", "image", "SigLIP ViT-B/16", False), ("SigLIP2 ViT-B/16 [45]", "image", "SigLIP2 ViT-B/16", False),
            ("SigLIP2 + trained adapter", "image", "SigLIP2 + fine-tuned adapter", False), ("Qwen2.5-VL attribute lookup [2]", "VLM", "Qwen2.5-VL lookup", False),
            ("Naive convex router", "all", "Naive uncalibrated fusion", True), ("Late fusion (SigLIP2 + lookup)", "image, VLM", "LF", False),
            ("SigLIP2 + Qwen2.5-VL rerank", "image, VLM", "RR", False), ("SceneMiner (portable)", "image, VLM, meta", "SceneMiner (portable)", False),
            ("SceneMiner (full)", "all", "SceneMiner", True)]
    vals, lat = {}, {}
    for lab, _, k, _ in rows:
        if k == "LF": vals[lab] = cols(LF, "Late fusion (SigLIP2 + lookup)", LF["per_query"]); continue
        if k == "RR":
            B = VR["benchmarks"]; pooled = float(np.mean([x["AP_rerank"] for x in B["s5_corpus_atomic"]["per_query"] + B["s5_corpus_compositional"]["per_query"]]))
            vals[lab] = [B["human_gt"]["mAP_rerank"], B["bdd"]["mAP_rerank"], B["holdout"]["mAP_rerank"], pooled]; lat[lab] = VR["latency"]["batch10_ms_median"]; continue
        vals[lab] = cols(M28, k, M28["per_query"])
        lat[lab] = LAT[k]["ms"] if k in LAT else M28["methods"][k].get("rank_latency_ms_1508_single_model", M28["methods"][k]["rank_latency_ms_1508"])
    a40 = M28["methods"]["SigLIP2 ViT-B/16"]["rank_latency_ms_1508"] + M28["methods"]["Qwen2.5-VL lookup"]["rank_latency_ms_1508"]
    fuse = LF["methods"]["Late fusion (SigLIP2 + lookup)"]["rank_latency_ms_1508"] - LF["methods"]["SigLIP2 ViT-B/16"]["rank_latency_ms_1508"] \
        - LF["methods"]["Qwen2.5-VL lookup"]["rank_latency_ms_1508"]
    lat["Late fusion (SigLIP2 + lookup)"] = a40 + fuse
    order = [sorted([v[i] for v in vals.values() if v[i] is not None], reverse=True) for i in range(4)]
    body = []
    for lab, sig, k, dag in rows:
        cell = []
        for i, x in enumerate(vals[lab]):
            if x is None: cell.append("--"); continue
            t = f"{x:.3f}"
            if round(x, 3) == round(order[i][0], 3): t = rf"\textbf{{{t}}}"
            elif round(x, 3) == round(order[i][1], 3): t = rf"\underline{{{t}}}"
            if i == 0 and f"T2|{k}" in SD: t += rf"$_{{\pm{SD[f'T2|{k}']['human_gt']['sd']:.3f}}}$".replace("0.", ".", 1)
            if dag and i in (0, 3): t += r"$^\dagger$"
            cell.append(t)
        L = lat[lab]; ms = f"{L:,.0f}" if L >= 100 else (f"{L:.1f}" if L >= 1 else f"{L:.2f}")
        if k == "LF": ms += r"$^\ast$"
        body.append([lab, sig] + cell + [ms])
    table("Table 2 (main comparison) - complete, late-fusion row filled", ["method", "signals", "degradation (human, 550)", "BDD100K 1000",
          "holdout", "semantic (3D GT, 258)", "ms / query"], body,
          f"Late fusion = z(SigLIP2 cosine) + z(a-priori VLM margin), each z-scored over the ranked set. *ms: A40 component sum "
          f"(SigLIP2 {M28['methods']['SigLIP2 ViT-B/16']['rank_latency_ms_1508']:.3f} + lookup {M28['methods']['Qwen2.5-VL lookup']['rank_latency_ms_1508']:.3f} ms) "
          f"+ {fuse:.3f} ms host-side z-scoring and sum (measured on CPU; it runs on the host on any machine). Bold/underline recomputed.")
    # ---------------- E1
    E1 = J("milestone41_e1_bootstrap.json")
    for setname, title in [("rater1_550", "E1 (a) rater 1, 550 scenes"), ("rater2_160", "E1 (b) rater 2, 160 double-labelled scenes")]:
        S = E1["sets"][setname]; rows_ = []
        for c, v in S["paired"].items():
            a, b = c.split(" - ")
            rows_.append([f"{a.replace('SceneMiner ', 'SM ')} vs {b.replace(' (SigLIP2 + lookup)', '').replace('Qwen2.5-VL attribute ', '')}"] +
                         [f"{v[k]['delta']:+.3f} [{v[k]['ci95'][0]:+.3f}, {v[k]['ci95'][1]:+.3f}] p={p_str(v[k]['p_two_sided'])}" for k in ATTRS + ["macro"]])
        mp = S["mAP"]
        rows_ = [[m] + [f3(mp[m][k]) for k in ATTRS + ["macro"]] for m in ("SceneMiner (full)", "SceneMiner (portable)", "Late fusion (SigLIP2 + lookup)",
                                                                             "Qwen2.5-VL attribute lookup")] + rows_
        table(title, ["", *AL, "macro"], rows_, f"Top: mAP. Bottom: paired difference, scene-bootstrap 95% CI ({E1['n_boot']} resamples), "
              "two-sided bootstrap p. Trained models: per-seed AP averaged over five seeds.")
    # ---------------- Table 5 + E2 + E5
    D = (Path("paper/tab_design.tex").read_text().split(r"\midrule")[1].split(r"\bottomrule")[0]).strip().splitlines()
    tf = J("milestone41_e2_v26tf.json"); t = tf["methods"]["SceneMiner"]
    rows5 = [[x.strip() for x in l.rstrip("\\").split("&")] for l in D if "&" in l]
    rows5.append(["training-free control (E2)", f"{t['human_gt']:.3f}", f"{t['bdd']:.3f}", f"{t['suite']:.3f}"])
    table("Table 5 (design progression) + E2 row", ["design step", "degradation", "BDD100K", "suite"], rows5,
          "Rows 1-6 as generated from the result files (paper/tab_design.tex), with Row 2/3 BDD100K = 0.570 / 0.589 and Row 6 = 0.794+-.008. "
          "E2: same PartNorm parts, gate, fallback and clamp, equal weights, no learned parameter; deterministic (sd 0).")
    B2 = J("milestone41_e2_bootstrap.json")
    rowsE2 = []
    for lab, f, key in [("SceneMiner (full)", "milestone41_late_fusion.json", "SceneMiner"), ("SceneMiner (portable)", "milestone41_late_fusion.json", "SceneMiner (portable)"),
                        ("training-free control", "milestone41_e2_v26tf.json", "SceneMiner")]:
        d = J(f); m = d["methods"][key]; ho = d["holdout"]["modes"]["metadata_free"]["methods"][key]["mAP_all"]
        r2 = B2["sets"]["rater2_160"]["mAP"]["Training-free control" if lab.startswith("training") else lab]["macro"]
        rowsE2.append([lab, f3(m["human_gt"]), f3(r2), f3(m["bdd"]), f3(ho), f"{(m['s5_corpus_atomic'] + m['s5_corpus_compositional']) / 2:.3f}",
                       f"{m['s5_corpus_atomic']:.3f} / {m['s5_val150_atomic']:.3f}", f3(m["suite"])])
    table("E2 - learned components vs the training-free control, all benchmarks", ["", "degradation r1 (550)", "degradation r2 (160)", "BDD100K 1000",
          "holdout", "semantic 258", "atomic semantics 258 / val150", "suite"], rowsE2)
    rowsE2b = []
    for setname in ("rater1_550", "rater2_160"):
        S = B2["sets"][setname]
        for c in ("SceneMiner (full) - Training-free control", "SceneMiner (portable) - Training-free control"):
            v = S["paired"][c]
            rowsE2b.append([f"{setname}: {c.split(' - ')[0]} - control"] + [f"{v[k]['delta']:+.3f} [{v[k]['ci95'][0]:+.3f}, {v[k]['ci95'][1]:+.3f}] p={p_str(v[k]['p_two_sided'])}"
                                                                              for k in ATTRS + ["macro"]])
        rowsE2b.append([f"{setname}: control mAP"] + [f3(S["mAP"]["Training-free control"][k]) for k in ATTRS + ["macro"]])
    table("E2 - paired bootstrap, learned minus training-free", ["", *AL, "macro"], rowsE2b)
    # ---------------- Table 4 check
    T4 = J("milestone41_table4_check.json")["sets"]
    rows4 = []
    for k in ("SceneMiner (5-seed ensemble)", "SceneMiner (per-seed mean)", "zero-shot SigLIP2", "logit VLM lookup", "fusion SigLIP2 + lookup", "fusion SigLIP2 + metadata"):
        r = [k, f3(T4["s5_20"]["mAP"][k]), f3(T4["synergy_30"]["mAP"][k])]
        tq = T4["s5_20"]["tests_by_query"].get(k); tp = T4["synergy_30"].get("tests_by_predicate", {}).get(k)
        r += [f"{tq['delta']:+.3f}, p={p_str(tq['perm_p'])}" if tq else "", f"{tp['delta']:+.3f}, p={p_str(tp['perm_p'])} ({tp['wins']}/{tp['n_predicates']} won)" if tp else ""]
        rows4.append(r)
    table("Table 4 conjunction column - both query sets", ["ranker", "20 Stratum-5 conjunctions (Table 2 set)", "30 phrasings / 11 predicates (published column)",
          "SceneMiner - ranker, by query (20)", "SceneMiner - ranker, by predicate (11)"], rows4)
    # ---------------- E3
    E3 = J("milestone41_e3_val150_stratified.json"); MET = ["mAP_front", "fp_rate", "recall_far", "vru_recall"]
    rk = ["SceneMiner", "VLM margin lookup", "night/rain metadata", "zero-shot SigLIP2"]
    rows6 = [[r] + [f"{E3['A_global_null'][r][m]['delta']:+.3f} (p={p_str(E3['A_global_null'][r][m]['p_worse'])})" for m in MET] for r in rk]
    rows6.append(["random (50 draws)"] + [f"{E3['random_50_draw_delta'][m]:+.3f}" for m in MET])
    table("Table 6 (detection error of the top 25, val150) with exact p", ["ranker", "mAP", "ghost FP", "rec. > 30 m", "VRU rec."], rows6,
          f"Top-25 front-sector metric minus the val150 mean; one-sided scene-permutation p ({E3['n_perm']} permutations).")
    rows6b = [[r, f"{E3['top25_night_or_rain_share'][r]:.2f}"] + [f"{E3['B_stratum_null'][r][m]['excess']:+.3f} (p={p_str(E3['B_stratum_null'][r][m]['p_worse'])})" for m in MET] for r in rk]
    table("E3 (a) controlled for night/rain: excess over a stratum-matched null", ["ranker", "night/rain share of top 25", "mAP", "ghost FP", "rec. > 30 m", "VRU rec."], rows6b,
          f"Strata: day-dry {E3['strata']['day-dry']}, night-or-rain {E3['strata']['night-or-rain']} (val150 share 0.26). Metrics permuted within strata; "
          "the excess is the top-25 mean minus its stratum-matched expectation.")
    C = E3["C_within_stratum"]["day-dry"]
    rows6c = [[r] + [f"{C['rankers'][r][m]['delta']:+.3f} (p={p_str(C['rankers'][r][m]['p_worse'])})" for m in MET] for r in rk]
    table(f"E3 (b) within day-dry scenes only (top {C['k']} of {C['n']})", ["ranker", "mAP", "ghost FP", "rec. > 30 m", "VRU rec."], rows6c)
    rows6d = []
    for comp, scopes in E3["D_paired"].items():
        for sc in ("global", "stratum-preserving", "within day-dry", "within night-or-rain"):
            rows6d.append([f"{comp} [{sc}]"] + [f"{scopes[sc][m]['delta']:+.3f} (p={p_str(scopes[sc][m]['p_two_sided'])})" for m in MET])
    table("E3 (c) paired tests between rankers", ["comparison [null]", "mAP", "ghost FP", "rec. > 30 m", "VRU rec."], rows6d, "Two-sided.")
    # ---------------- E4
    E4 = J("milestone41_e4_val150_flags.json"); A = E4["partitions"]["any_degradation"]
    rows7 = [[lab, f"{A['metrics'][m]['degraded']:.4f}", f"{A['metrics'][m]['clean']:.4f}", f"{A['metrics'][m]['delta']:+.4f}",
              f"{A['metrics'][m]['published_q_BH']:.3f}", f"{A['metrics'][m]['q_BH']:.3f}"]
             for lab, m in [("mAP", "mAP_front"), ("ghost FP @ s>0.3", "fp_rate"), ("mATE (m)", "mATE_front"), ("recall > 30 m", "recall_far"), ("mAOE (rad)", "mAOE_front")]]
    table(f"E4 - CAM_FRONT degradation vs 3D error (val150, {A['n_degraded']} degraded vs {A['n_clean']} clean)", ["metric", "degraded", "clean", "Delta",
          "q (BH, published p, 1000 perm.)", "q (BH, 10000 perm.)"], rows7, "BH over all 30 tests (5 partitions x 6 metrics). Per-attribute positives: "
          + ", ".join(f"{k} {v}" for k, v in E4["per_attribute_positive"].items()) + ".")
    # ---------------- E5
    E5, PS = J("milestone41_e5_scale.json"), J("milestone41_e5_v26ps.json")
    rows8 = [[n.replace("_", " "), f"{v['s_head']['mean']:+.3f}", f"{v['s_head']['sd_pooled']:.3f}", f"{v['s_head']['sd_within_query_mean']:.3f}",
              f"{v['s_abst']['mean']:+.3f}", f"{v['s_abst']['sd_pooled']:.3f}", f"{v['s_abst']['sd_within_query_mean']:.3f}"] for n, v in E5["sets"].items()]
    table("E5 (a) empirical scale over the 906 training scenes (five seeds)", ["queries", "s_head mean", "s_head sd", "s_head sd within query",
          "s_abst mean", "s_abst sd", "s_abst sd within query"], rows8, f"Decomposition check: max |error| {E5['decomposition_max_abs_error']:.1e}.")
    rows9 = []
    for lab, d in [("Eq. (1) as published", J("milestone41_late_fusion.json")), ("s_head, s_abst standardised per query on the 906 training scenes", PS)]:
        for key in ("SceneMiner", "SceneMiner (portable)"):
            m = d["methods"][key]; ho = d["holdout"]["modes"]["metadata_free"]["methods"][key]["mAP_all"]
            rows9.append([f"{lab} - {key}", f3(m["human_gt"]), f3(m["bdd"]), f3(ho), f"{(m['s5_corpus_atomic'] + m['s5_corpus_compositional']) / 2:.3f}",
                          f"{m['s5_val150_atomic']:.3f}", f3(m["suite"])])
    table("E5 (b) mAP with and without post-standardisation", ["", "degradation (550)", "BDD100K 1000", "holdout", "semantic 258", "val150 atomic", "suite"], rows9)
    (R/"MILESTONE41_tables.tex").write_text("% Generated by scripts/make_m41_tables.py from results/cvpr/*.json\n" + "\n".join(TEX))
    (R/"MILESTONE41_tables.md").write_text("# Milestone 41 tables (generated by scripts/make_m41_tables.py)\n\n" + "\n".join(MD))
    print("\n".join(MD))


if __name__ == "__main__":
    main()
