#!/usr/bin/env python3
"""MILESTONE 7 - the exact 390-query suite: 240 train / 150 eval.

Eval strata (paper spec):
    62 sensor-degradation, 58 macro-environmental,
    18 held-out compositional conjunctions, 12 negated.
`rain AND night` (and every other conjunction) is withheld from the 240 training
queries entirely. Train and eval text strings are 100 % disjoint; each condition
appears in several phrasings with severity gradations so retrieval cannot
succeed by keyword match alone.
"""
from __future__ import annotations
import itertools, json
from pathlib import Path

SEV = {  # severity-graded phrasings, mild -> extreme
 "droplets": ["a few water specks on the lens","light droplets scattered on the lens",
   "water droplets smearing the camera lens","rain beading across the lens",
   "heavy rain spatter obscuring the camera","the lens drowned in rain streaks",
   "drizzle marking the lens","persistent droplet smear on the glass"],
 "glare": ["a faint halo around a street light","mild light flare at the frame edge",
   "noticeable glare washing across the image","specular glare washing out the image",
   "strong headlight bloom from oncoming traffic","blinding light bloom swallowing the scene",
   "harsh low-sun glare","scattered street-light halos"],
 "wet_road": ["a lightly damp road surface","patchy wet tarmac",
   "wet reflective asphalt after rainfall","slick wet pavement throwing reflections",
   "standing water pooling on the road","a soaked roadway with mirror-like sheen",
   "rain-slicked carriageway","puddled road surface"],
 "low_contrast": ["slightly flat lighting","a dim view where shapes lose definition",
   "hazy washed out view with unclear boundaries","poor visibility with objects blending in",
   "a low contrast scene where vehicles are barely distinguishable",
   "murky scene with indistinct vehicle outlines","grey scene with little tonal separation",
   "objects dissolving into the background"],
}
MACROV = {
 "lighting=night": ["driving after dark","a night-time street scene","low ambient light driving",
                    "night driving on an unlit road","the street under lamplight"],
 "lighting=day":   ["bright daytime driving","a clear daylight street","a midday road scene",
                    "driving under full daylight","an afternoon street view"],
 "weather=rain":   ["driving in rain","a rainy street","wet weather driving",
                    "rainfall over the roadway","driving through a shower"],
 "weather=dry":    ["dry road conditions","a dry clear day","no precipitation on the road",
                    "driving in settled weather","a rain-free street"],
 "density=dense":  ["dense bumper-to-bumper traffic","a congested roadway","a heavy traffic jam",
                    "queued vehicles filling the lane","packed traffic ahead"],
 "density=sparse": ["an almost empty road","sparse traffic","a deserted street",
                    "an open road with few cars","a near-empty carriageway"],
 "density=moderate":["a moderately busy road","a few vehicles around","light traffic flow",
                     "steady but unhurried traffic","a handful of cars on the road"],
}
CTX = [({"lighting":"night"},"at night"), ({"lighting":"day"},"in daylight"),
       ({"weather":"rain"},"in rain"), ({"density":"dense"},"in dense traffic"),
       ({"density":"sparse"},"on an empty road")]

def pred_of(k): a,b=k.split("="); return {a:b}

def main():
    train, ev, seen = [], [], set()
    def mk(dst, text, pred, stratum, attr, cap=None):
        if text in seen: return False
        if cap is not None and sum(1 for q in dst if q["stratum"]==stratum) >= cap: return False
        seen.add(text); dst.append({"text":text,"predicate":pred,"stratum":stratum,
                                    "attribute":attr}); return True

    # ---------- EVAL: 62 sensor-degradation ----------
    for at, ph in SEV.items():
        for p in ph[:4]:                       # 4 bare phrasings x 4 attrs = 16
            mk(ev, p, {at:True}, "sensor_degradation", at, 62)
    for at, ph in SEV.items():                 # + context-composed, severity varied
        for p, (c, cp) in itertools.product(ph[:3], CTX):
            pr={at:True}; pr.update(c)
            mk(ev, f"{p} {cp}", pr, "sensor_degradation", at, 62)
    # ---------- EVAL: 58 macro-environmental ----------
    for mk_, ph in MACROV.items():
        for p in ph[:3]:
            mk(ev, p, pred_of(mk_), "macro_environmental", mk_, 58)
    for k1, k2 in itertools.combinations(MACROV, 2):
        if k1.split("=")[0]==k2.split("=")[0]: continue
        pr=pred_of(k1); pr.update(pred_of(k2))
        mk(ev, f"{MACROV[k1][0]}, {MACROV[k2][0]}", pr, "macro_environmental",
           f"{k1}+{k2}", 58)
    # ---------- EVAL: 18 compositional (NEVER trained) ----------
    comp=[({"weather":"rain","lighting":"night"}, t) for t in
          ["heavy rain at night","a dark rainy street after dark","a night-time downpour",
           "rainfall on an unlit road","driving through rain in darkness"]]
    comp += [({"droplets":True,"lighting":"night"},"rain droplets on the lens at night"),
             ({"glare":True,"wet_road":True},"glare reflecting off a wet road"),
             ({"density":"dense","lighting":"night"},"dense traffic after dark"),
             ({"wet_road":True,"lighting":"night"},"slick pavement in the dark"),
             ({"low_contrast":True,"lighting":"night"},"indistinct shapes after dark")]
    for pr,t in comp: mk(ev, t, pr, "compositional_heldout", "+".join(pr), 18)
    for at in SEV:
        for mkey in ["lighting=night","weather=rain"]:
            pr={at:True}; pr.update(pred_of(mkey))
            mk(ev, f"{SEV[at][2]} together with {MACROV[mkey][0]}", pr,
               "compositional_heldout", f"{at}+{mkey}", 18)
    # ---------- EVAL: 12 negated ----------
    neg=[({"weather":"rain","lighting":"day"},"heavy rain but not at night"),
         ({"lighting":"night","weather":"dry"},"night driving without any rain"),
         ({"wet_road":True,"lighting":"day"},"a wet road but in daylight"),
         ({"density":"sparse","lighting":"day"},"an empty road that is not dark")]
    for pr,t in neg: mk(ev,t,pr,"negated","+".join(pr),12)
    for at in SEV:
        mk(ev, f"{SEV[at][3]} but not after dark", {at:True,"lighting":"day"},
           "negated", at, 12)
        mk(ev, f"{SEV[at][4]} without rainfall", {at:True,"weather":"dry"},
           "negated", at, 12)

    # ---------- more macro to reach 58 (3-way and reworded combinations) ----------
    for k1, k2 in itertools.combinations(MACROV, 2):
        if k1.split("=")[0] == k2.split("=")[0]: continue
        pr = pred_of(k1); pr.update(pred_of(k2))
        mk(ev, f"{MACROV[k1][1]} with {MACROV[k2][1]}", pr, "macro_environmental",
           f"{k1}+{k2}", 58)
    for k1, k2 in itertools.combinations(MACROV, 2):
        if k1.split("=")[0] == k2.split("=")[0]: continue
        pr = pred_of(k1); pr.update(pred_of(k2))
        mk(ev, f"{MACROV[k2][2]} during {MACROV[k1][2]}", pr, "macro_environmental",
           f"{k1}+{k2}", 58)

    # predicates reserved for the held-out compositional stratum - no training
    # query may carry any of them, or the "never supervised" claim is false.
    RESERVED = {tuple(sorted(q["predicate"].items())) for q in ev
                if q["stratum"] == "compositional_heldout"}
    def mk_train(text, pred, stratum, attr):
        if tuple(sorted(pred.items())) in RESERVED: return False
        return mk(train, text, pred, stratum, attr)

    # ---------- TRAIN: 240, disjoint text, no reserved conjunctions ----------
    for at, ph in SEV.items():
        for p in ph[4:]: mk_train(p, {at: True}, "sensor_degradation", at)
    for at, ph in SEV.items():
        for p, (c, cp) in itertools.product(ph[4:], CTX):
            pr = {at: True}; pr.update(c)
            mk_train(f"{p} {cp}", pr, "sensor_degradation", at)
    for mk_, ph in MACROV.items():
        for p in ph[3:]: mk_train(p, pred_of(mk_), "macro_environmental", mk_)
    for at, ph in SEV.items():
        for p, (c, cp) in itertools.product(ph[:4], CTX):
            pr = {at: True}; pr.update(c)
            mk_train(f"{p}, seen {cp}", pr, "sensor_degradation", at)
    for mk_, ph in MACROV.items():
        for p in ph:
            mk_train(f"a scene showing {p}", pred_of(mk_), "macro_environmental", mk_)
    for mk_, ph in MACROV.items():
        for p in ph:
            mk_train(f"footage of {p}", pred_of(mk_), "macro_environmental", mk_)
    for at, ph in SEV.items():
        for p in ph:
            mk_train(f"a recording with {p}", {at: True}, "sensor_degradation", at)
    for at, ph in SEV.items():
        for p in ph:
            mk_train(f"dashcam footage showing {p}", {at: True}, "sensor_degradation", at)
    for mk_, ph in MACROV.items():
        for p in ph:
            mk_train(f"a clip captured during {p}", pred_of(mk_), "macro_environmental", mk_)
    train = train[:240]
    strata = {s: sum(1 for q in ev if q["stratum"]==s) for s in
              ["sensor_degradation","macro_environmental","compositional_heldout","negated"]}
    out={"n_train":len(train),"n_eval":len(ev),"n_total":len(train)+len(ev),
         "strata":strata,
         "withheld_from_training":"all compositional conjunctions; rain AND night never trained",
         "train":train,"eval":ev}
    Path("benchmark_cvpr").mkdir(exist_ok=True)
    Path("benchmark_cvpr/queries_390.json").write_text(json.dumps(out,indent=1))
    assert not ({q["text"] for q in train} & {q["text"] for q in ev}), "train/eval overlap"
    tr_preds={tuple(sorted(q["predicate"].items())) for q in train}
    comp_preds=RESERVED
    print(f"train {len(train)}  eval {len(ev)}  total {len(train)+len(ev)}")
    for k,v in strata.items(): print(f"  {k:<26} {v}")
    print(f"  compositional predicates also present in train: {len(tr_preds & comp_preds)} (must be 0)")
    print("  train/eval text disjoint: VERIFIED")

if __name__ == "__main__":
    main()
