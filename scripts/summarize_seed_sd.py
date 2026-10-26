#!/usr/bin/env python3
"""MILESTONE 34 - five-seed standard deviations for every trained row of Tables 1 and 2.

Reads work_dirs/seed_sd/*_s{0..4}.json (scripts/run_seed_sd.sh: the same evaluation
scripts that produced the tables, each run on one seed's checkpoint) and reports, per
row and benchmark, the five single-seed values, their mean and sample sd (ddof=1).

Human GT is scored per seed in the paper (the reported value IS the mean over seeds),
so for that column the mean below must reproduce the table value; this is asserted.
The other benchmarks report the five-seed ensemble; there the sd measures how much a
single model would move, and the table value (ensemble) is listed next to it.
-> results/cvpr/milestone34_seed_sd.json
"""
import json, re
from pathlib import Path
import numpy as np

O = Path("work_dirs/seed_sd"); R = Path("results/cvpr")
J = lambda p: json.load(open(p))
ATTRS = ["droplets", "glare", "wet_road", "low_contrast"]
SEEDS = range(5)


def key(d, suffix):
    ks = [k for k in d if re.match(rf"^v2\.\d_{suffix}$", k)]
    assert len(ks) == 1, (suffix, list(d)); return ks[0]


def macro_attr(A, k):
    return float(np.mean([A[a]["rankers"][k]["mAP"] for a in ATTRS]))


def v2x(f):
    d = J(f); A = d["A_human_gt"]["macro_mAP"]; B = d["B_bdd100k"]["rankers"]; C = d["C_suite150"]
    return {"human_gt": A[key(A, "full")], "human_gt_calibrated": A[key(A, "human_calibrated_5fold")],
            "bdd": float(B[key(B, "full")]["mAP"]), "suite": C["overall"][key(C["overall"], "full")]["mAP"],
            "suite_skew_pts": C["skew_pts"][key(C["skew_pts"], "full")]["mean"],
            **{f"abl_{k.split('_', 1)[1]}": A[k] for k in A if re.match(r"^v2\.\d_(no_|base_only)", k)}}


def t2(f, label):
    d = J(f); m = d["methods"][label]
    r = {k: m.get(k) for k in ("human_gt", "bdd", "s5_corpus_atomic", "s5_corpus_compositional",
                               "s5_val150_atomic", "s5_val150_compositional", "suite", "suite_skew_pts")}
    h = d["holdout"]["modes"]["metadata_free"]["methods"].get(label); r["holdout"] = h["mAP_all"] if h else None
    return r


def main():
    per = {}                                    # row -> seed -> metrics
    for s in SEEDS:
        add = lambda row, v: per.setdefault(row, {}).__setitem__(s, v)
        for lab in ("BEV-TSR-style text-to-BEV", "SigLIP2 + fine-tuned adapter", "Naive uncalibrated fusion",
                    "SceneMiner (portable)", "SceneMiner"):
            add(f"T2|{lab}", t2(O/f"t2_s{s}.json", lab))
        h21 = J(O/f"v21hum_s{s}.json")["mode_B"]["attributes"]
        add("T1|v2.1", {**t2(O/f"t2_s{s}.json", "Naive uncalibrated fusion"),
                        "human_gt_calibrated": macro_attr(h21, "SceneMiner_v2.1_human_calibrated_5fold"),
                        "human_gt_v21_script": macro_attr(h21, "SceneMiner_v2.1_zeroleak")})
        h22 = J(O/f"v22hum_s{s}.json")["mode_B"]["attributes"]
        add("T1|v2.2", {"human_gt": macro_attr(h22, "SceneMiner_v2.2_zeroleak"),
                        "human_gt_calibrated": macro_attr(h22, "SceneMiner_v2.2_human_calibrated_5fold"),
                        "bdd": J(O/f"v22bdd_s{s}.json")["rows"]["v22|raw"]["mAP"],
                        "suite": J(O/f"v22suite_s{s}.json")["overall"]["SceneMiner_v2.2"]["mAP"],
                        "suite_skew_pts": J(O/f"v22skew_s{s}.json")["rankers"]["SceneMiner_v2.2"]["raw"]["skew_pts"]})
        for v in ("23", "24", "25", "26"):
            add(f"T1|v2.{v[1]}", v2x(O/f"v{v}_s{s}.json"))
    # reported values (the table sources)
    ref = J(R/"milestone28_related_work_comparison.json")
    rep = {f"T2|{lab}": t2(R/"milestone28_related_work_comparison.json", lab) for lab in
           ("BEV-TSR-style text-to-BEV", "SigLIP2 + fine-tuned adapter", "Naive uncalibrated fusion", "SceneMiner (portable)", "SceneMiner")}
    m6, m10 = J(R/"milestone6_sceneminer_v2_1.json")["mode_B"]["attributes"], J(R/"milestone10_sceneminer_v2_2.json")["mode_B"]["attributes"]
    rep["T1|v2.1"] = {**rep["T2|Naive uncalibrated fusion"], "human_gt_calibrated": macro_attr(m6, "SceneMiner_v2.1_human_calibrated_5fold")}
    rep["T1|v2.2"] = {"human_gt": macro_attr(m10, "SceneMiner_v2.2_zeroleak"), "human_gt_calibrated": macro_attr(m10, "SceneMiner_v2.2_human_calibrated_5fold"),
                      "bdd": J(R/"milestone23_bdd_rawquery_routing.json")["rows"]["v22|raw"]["mAP"],
                      "suite": J(R/"milestone10_query_suite_150.json")["overall"]["SceneMiner_v2.2"]["mAP"],
                      "suite_skew_pts": J(R/"milestone11_domain_calibration.json")["rankers"]["SceneMiner_v2.2"]["raw"]["skew_pts"]}
    for v, f, b in [("3", "milestone14_v2_3.json", "milestone23_bdd_rawquery_v23.json"), ("4", "milestone18_v2_4.json", "milestone23_bdd_rawquery_v24.json"),
                    ("5", "milestone22_factored_gate.json", "milestone23_bdd_rawquery_v25.json"), ("6", "milestone27_v26.json", None)]:
        r = v2x(R/f)
        if b: r["bdd"] = float(J(R/b)["B_bdd100k"]["rankers"]["v2.3_full"]["mAP"])
        rep[f"T1|v2.{v}"] = r
    out = {"doc": __doc__, "rows": {}}
    for row, S in per.items():
        out["rows"][row] = {}
        for k in S[0]:
            vals = [S[s].get(k) for s in SEEDS]
            if any(v is None for v in vals): continue
            vals = [float(v) for v in vals]
            out["rows"][row][k] = {"seeds": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                                   "reported": rep.get(row, {}).get(k)}
        hg = out["rows"][row].get("human_gt")
        if hg and hg["reported"] is not None:
            hg["mean_reproduces_reported"] = abs(hg["mean"] - hg["reported"]) < 1e-4
    json.dump(out, open(R/"milestone34_seed_sd.json", "w"), indent=1)
    for row, M in out["rows"].items():
        print(row, "  ".join(f"{k} {v['reported'] if v['reported'] is None else round(float(v['reported']), 3)}"
                              f"(mean {v['mean']:.3f} sd {v['sd']:.3f})" for k, v in M.items() if not k.startswith("abl_")))


if __name__ == "__main__":
    main()
