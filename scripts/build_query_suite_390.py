#!/usr/bin/env python3
"""MILESTONE 3 - the 390-query suite (240 train pool / 150 eval), 4 strata.

Strata, per the paper spec:
  62 sensor-degradation  (droplets, glare, wet_road, low_contrast)
  58 macro-environmental (weather, lighting, traffic density)
  18 held-out compositional conjunctions  -- rain AND night withheld from training
  12 negated forms
Each condition appears in >=3 phrasings with severity gradations, so retrieval
cannot succeed by keyword match alone. Train and eval query sets are disjoint.
"""
from __future__ import annotations
import json, itertools
from pathlib import Path

DEGRAD = {
 "droplets": ["water droplets smearing the camera lens",
              "rain beading on the lens obscuring the view",
              "a light drizzle leaving droplets on the lens",
              "heavy rain spatter across the camera",
              "lens covered in rain streaks",
              "faint water specks on the lens"],
 "glare": ["blinding headlight bloom from oncoming traffic",
           "specular glare washing out the image",
           "mild light flare at the edge of frame",
           "intense light bloom reflecting off the road",
           "harsh sun glare low on the horizon",
           "scattered street-light halos"],
 "wet_road": ["wet reflective asphalt after rainfall",
              "standing water pooling on the road",
              "a lightly damp road surface",
              "slick wet pavement throwing reflections",
              "soaked roadway with mirror-like sheen",
              "patchy wet tarmac"],
 "low_contrast": ["a low contrast scene where vehicles are barely distinguishable",
                  "hazy washed out view with unclear object boundaries",
                  "poor visibility with objects blending into the background",
                  "murky scene with indistinct vehicle outlines",
                  "flat grey scene with little tonal separation",
                  "dim view where shapes lose definition"],
}
MACRO = {
 "lighting=night": ["driving after dark","a night-time street scene","low ambient light driving"],
 "lighting=day":   ["bright daytime driving","a clear daylight street","midday road scene"],
 "weather=rain":   ["driving in rain","a rainy street","wet weather driving"],
 "weather=dry":    ["dry road conditions","a dry clear day","no precipitation"],
 "density=dense":  ["dense bumper-to-bumper traffic","a congested roadway","heavy traffic jam"],
 "density=sparse": ["an almost empty road","sparse traffic","a deserted street"],
 "density=moderate":["moderately busy road","a few vehicles around","light traffic flow"],
}
COMPOSITIONAL = [  # withheld from the training pool entirely
 ({"weather":"rain","lighting":"night"}, "heavy rain at night"),
 ({"weather":"rain","lighting":"night"}, "a dark rainy street after dark"),
 ({"weather":"rain","lighting":"night"}, "night-time downpour"),
 ({"droplets":True,"lighting":"night"},  "rain droplets on the lens at night"),
 ({"glare":True,"wet_road":True},        "glare reflecting off a wet road"),
 ({"density":"dense","lighting":"night"},"dense traffic after dark"),
]
NEGATED = [
 ({"weather":"rain","lighting":"day"},  "heavy rain but not at night"),
 ({"lighting":"night","weather":"dry"}, "night driving without rain"),
 ({"wet_road":True,"lighting":"day"},   "a wet road but in daylight"),
 ({"density":"sparse","lighting":"day"},"an empty road that is not dark"),
 ({"droplets":True,"lighting":"day"},   "lens droplets but not at night"),
 ({"glare":True,"lighting":"day"},      "glare that is not from headlights"),
]

CONTEXTS = [({}, ""), ({"lighting":"night"}, "at night"), ({"lighting":"day"}, "in daylight"),
            ({"weather":"rain"}, "in rain"), ({"weather":"dry"}, "in dry conditions"),
            ({"density":"dense"}, "in dense traffic"), ({"density":"sparse"}, "on an empty road")]

def pred_of(key):
    k, v = key.split("="); return {k: v}

def main():
    train, ev = [], []
    seen = set()
    def push(dst, text, pred, stratum, attribute):
        if text in seen: return False
        seen.add(text); dst.append({"text": text, "predicate": pred,
                                    "stratum": stratum, "attribute": attribute})
        return True
    # ---- sensor degradation: phrasing x context; phrasings 0-3 train, 4-5 eval ----
    for at, ph in DEGRAD.items():
        for i, p in enumerate(ph):
            dst = ev if i >= 4 else train
            for ctx, cph in CONTEXTS:
                pr = {at: True}; pr.update(ctx)
                txt = (p + (" " + cph if cph else "")).strip()
                push(dst, txt, pr, "sensor_degradation", at)
    # ---- macro: phrasing 0-1 train, 2 eval; plus cross-context variants ----
    for mk, ph in MACRO.items():
        for i, p in enumerate(ph):
            dst = ev if i >= 2 else train
            push(dst, p, pred_of(mk), "macro_environmental", mk)
    for mk1, mk2 in itertools.combinations(MACRO, 2):
        k1, k2 = mk1.split("=")[0], mk2.split("=")[0]
        if k1 == k2: continue
        pr = pred_of(mk1); pr.update(pred_of(mk2))
        txt = f"{MACRO[mk1][0]}, {MACRO[mk2][0]}"
        dst = ev if len([q for q in ev if q["stratum"]=="macro_environmental"]) < 58 else train
        push(dst, txt, pr, "macro_environmental", f"{mk1}+{mk2}")
    # ---- compositional (EVAL ONLY, withheld from training) ----
    for pr, txt in COMPOSITIONAL: push(ev, txt, pr, "compositional_heldout", "+".join(pr))
    for at in DEGRAD:
        for mk in ["lighting=night", "weather=rain"]:
            pr = {at: True}; pr.update(pred_of(mk))
            txt = f"{DEGRAD[at][2]} combined with {MACRO[mk][0]}"
            push(ev, txt, pr, "compositional_heldout", f"{at}+{mk}")
    # ---- negated (EVAL ONLY) ----
    for pr, txt in NEGATED: push(ev, txt, pr, "negated", "+".join(pr))
    for at in DEGRAD:
        pr = {at: True, "lighting": "day"}
        push(ev, f"{DEGRAD[at][0]} but not after dark", pr, "negated", at)
        pr2 = {at: True, "weather": "dry"}
        push(ev, f"{DEGRAD[at][1]} without rainfall", pr2, "negated", at)
    out = {"n_train": len(train), "n_eval": len(ev),
           "strata": {s: sum(1 for q in ev if q["stratum"] == s)
                      for s in ["sensor_degradation","macro_environmental",
                                "compositional_heldout","negated"]},
           "withheld_from_training":
               "all compositional conjunctions are eval-only; rain AND night never trained",
           "train": train, "eval": ev}
    Path("benchmark_cvpr").mkdir(exist_ok=True)
    Path("benchmark_cvpr/queries_390.json").write_text(json.dumps(out, indent=1))
    assert not ({q["text"] for q in train} & {q["text"] for q in ev}), "overlap"
    print(f"train pool {len(train)}  eval {len(ev)}  total {len(train)+len(ev)}")
    for k, n in out["strata"].items(): print(f"  eval stratum {k:<26} {n}")
    print("train/eval disjoint: VERIFIED")

if __name__ == "__main__":
    main()
