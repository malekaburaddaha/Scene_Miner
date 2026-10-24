#!/usr/bin/env python3
"""MILESTONE 30 - the 160-scene second-rater subset (benchmark_cvpr/rater2_subset_160.json).

Drawn from the 550 scenes rater 1 labelled. Per attribute, 40 scenes: 20 that rater
1 marked positive (uniform draw) and 20 boundary cases, the scenes whose VLM margin
for that attribute is closest to zero (chosen without rater-1 labels). Scenes already
taken for an earlier attribute are skipped, so the 160 are distinct. Seed 0. The
subset over-represents positives and hard cases on purpose, so agreement on it is a
conservative estimate of agreement on the full sample; incorporate_rater2.py reports
it per stratum.
"""
import json, random, sys
sys.path.insert(0, "scripts")
from eval_human_gt_full import load_human, ATTRS

W = "work_dirs/unified_filter"
hum = load_human("results/day1_gate/malek_repaired.json", "B")
vlm = json.load(open(f"{W}/binary_flags_with_logits.json")); strict = json.load(open(f"{W}/lowcontrast_strict.json"))
def marg(t, a):
    src, key = (strict, "low_contrast_strict") if a == "low_contrast" else (vlm, a)
    return float(src.get(t, {}).get(key + "_margin", 9e9))


rng = random.Random(0); chosen, strata = [], {}
for a in ATTRS:
    pos = sorted(t for t in hum if hum[t][a] and t not in chosen); rng.shuffle(pos)
    take = pos[:20]; chosen += take; strata.update({t: f"{a}:rater1_positive" for t in take})
    bord = sorted((t for t in hum if t not in chosen), key=lambda t: abs(marg(t, a)))[:20]
    chosen += bord; strata.update({t: f"{a}:vlm_boundary" for t in bord})
order = chosen[:]; rng.shuffle(order)          # rater 2 sees them in random order
json.dump({"doc": __doc__, "n": len(order), "tokens": order, "stratum": strata},
          open("benchmark_cvpr/rater2_subset_160.json", "w"), indent=1)
print(len(order), len(set(order)))
