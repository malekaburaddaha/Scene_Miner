#!/usr/bin/env python3
"""MILESTONE 15 - Stratum 5: 40 safety-critical semantic edge-case queries.

20 atomic queries: four phrasings for each of the five human-annotated semantic
flags. 20 compositional queries over 12 semantic x condition predicates. A
predicate qualifies only if it has >= 5 positives among the 258 human-labelled
nuScenes eval scenes - a rule fixed from label COUNTS, before any ranker was
run - and 8 of the 12 carry two phrasings to reach 20 queries. Several
conjunctions one would like to test (any VRU conjunction, heavy vehicle in low
contrast, pedestrian at night with glare) have 1-4 positives and are excluded
rather than evaluated on noise.

Ground truth for every predicate is human: semantic flags from nuScenes 3D
annotation, night/rain from the human-written scene description, and
glare/wet_road/droplets/low_contrast from the human rater.
"""
import json
from pathlib import Path

A = [
 ("pedestrian_close", ["pedestrians walking near the roadway",
                       "a person crossing the street in front of the car",
                       "people on foot close to the vehicle",
                       "pedestrians at a crosswalk ahead"]),
 ("vru", ["bicyclist or motorcyclist in traffic",
          "a cyclist riding on the road ahead",
          "a motorcycle rider sharing the lane",
          "vulnerable road users close to the car"]),
 ("construction_zone", ["road construction zone with traffic cones and barriers",
                        "traffic cones blocking part of the lane",
                        "construction workers and equipment on the road",
                        "a work zone with orange cones"]),
 ("heavy_vehicle", ["large truck or bus ahead",
                    "a city bus right in front of the car",
                    "a heavy truck driving close by",
                    "a trailer or construction vehicle nearby"]),
 ("rare_or_child", ["a child, stroller or wheelchair user near the road",
                    "rare or unusual road users near the vehicle",
                    "a police officer, emergency vehicle or debris on the street",
                    "unexpected objects or vulnerable people near the lane"]),
]
C = [
 ({"pedestrian_close": True, "night": True}, ["pedestrians walking near the road at night",
                                              "people crossing the street after dark"]),
 ({"pedestrian_close": True, "rain": True}, ["pedestrians walking in the rain",
                                             "people on foot near the road on a rainy day"]),
 ({"pedestrian_close": True, "wet_road": True}, ["pedestrians near a wet reflective road"]),
 ({"pedestrian_close": True, "droplets": True}, ["pedestrians visible through raindrops on the camera lens",
                                                 "a person on the street seen through a wet, spotted lens"]),
 ({"pedestrian_close": True, "low_contrast": True}, ["pedestrians barely visible in a low contrast dark scene"]),
 ({"construction_zone": True, "rain": True}, ["construction zone with cones on a rainy day",
                                              "road works with traffic cones in the rain"]),
 ({"construction_zone": True, "wet_road": True}, ["construction barriers and cones on a wet reflective road",
                                                  "traffic cones on wet glistening pavement"]),
 ({"construction_zone": True, "droplets": True}, ["traffic cones seen through water droplets on the lens"]),
 ({"heavy_vehicle": True, "rain": True}, ["a bus or truck ahead in the rain",
                                          "heavy truck driving in rainy weather"]),
 ({"heavy_vehicle": True, "wet_road": True}, ["a large truck on a wet reflective road",
                                              "a bus driving on wet pavement"]),
 ({"heavy_vehicle": True, "droplets": True}, ["a bus or truck seen through rain droplets on the lens",
                                              "a large vehicle ahead behind a rain-spattered lens"]),
 ({"rare_or_child": True, "wet_road": True}, ["unusual road users near a wet road"]),
]
qs = []
for flag, ph in A:
    for t in ph:
        qs.append({"text": t, "predicate": {flag: True}, "stratum": "semantic_atomic",
                   "semantic": flag, "condition": None})
for pred, ph in C:
    sem = next(k for k in pred if k in dict(A)); cond = next(k for k in pred if k != sem)
    for t in ph:
        qs.append({"text": t, "predicate": pred, "stratum": "semantic_compositional",
                   "semantic": sem, "condition": cond})
assert len(qs) == 40 and sum(q["stratum"] == "semantic_atomic" for q in qs) == 20
Path("benchmark_cvpr/queries_semantic_edge_cases.json").write_text(json.dumps(
    {"n": len(qs), "doc": __doc__, "queries": qs}, indent=1))
print(f"wrote 40 queries ({len(C)} compositional predicates)")
