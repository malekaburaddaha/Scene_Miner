#!/usr/bin/env python3
"""TASK 3.3 - deterministic, frozen 906/602 split with Waymo segment disjointness.

Target (paper spec): 906 train = 422 nuScenes + 484 Waymo chunks from 121 segments
                     602 eval  = 278 nuScenes + 324 Waymo chunks from  81 segments
Waymo scene ids are '<segment>#c<k>'; the split is drawn at SEGMENT granularity so
no 20 s source segment contributes chunks to both sides (near-duplicate leakage).
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

SEED = 1234
z = np.load("work_dirs/unified_filter/features_overnight.npz", allow_pickle=False)
toks = [str(t) for t in z["scene_tokens"]]; dom = z["domain"].astype(int)
nusc = [t for t, d in zip(toks, dom) if d == 0]
waymo = [t for t, d in zip(toks, dom) if d == 1]

seg = defaultdict(list)
for t in waymo: seg[re.sub(r"#c\d+$", "", t)].append(t)
sizes = sorted({len(v) for v in seg.values()})
print(f"nuScenes={len(nusc)}  Waymo={len(waymo)}  segments={len(seg)}  chunks/segment={sizes}")

rng = np.random.default_rng(SEED)
# --- nuScenes: straight random 422/278 ---
ns = list(nusc); rng.shuffle(ns)
ns_tr, ns_ev = sorted(ns[:422]), sorted(ns[422:])

# --- Waymo: choose 121 TRAIN segments whose chunk total is exactly 484 ---
segs = sorted(seg); rng.shuffle(segs)
target_tr_seg, target_tr_chunk = 121, 484
best = None
for _ in range(20000):
    rng.shuffle(segs)
    cand = segs[:target_tr_seg]
    n = sum(len(seg[s]) for s in cand)
    if n == target_tr_chunk: best = list(cand); break
    if best is None or abs(n - target_tr_chunk) < abs(best[1] - target_tr_chunk):
        best = (list(cand), n) if not isinstance(best, list) else best
if not isinstance(best, list):
    best = best[0]; print(f"  WARNING: exact chunk target not hit, using closest")
tr_seg = sorted(best); ev_seg = sorted(set(segs) - set(tr_seg))
wa_tr = sorted(t for s in tr_seg for t in seg[s])
wa_ev = sorted(t for s in ev_seg for t in seg[s])

train = sorted(ns_tr + wa_tr); ev = sorted(ns_ev + wa_ev)
assert not (set(train) & set(ev)), "train/eval overlap!"
assert not (set(tr_seg) & set(ev_seg)), "segment leakage!"
assert len(train) + len(ev) == len(toks)

out = {
    "seed": SEED, "spec": "906 train / 602 eval, Waymo split at segment granularity",
    "counts": {"train": len(train), "eval": len(ev),
               "train_nuscenes": len(ns_tr), "train_waymo": len(wa_tr),
               "eval_nuscenes": len(ns_ev), "eval_waymo": len(wa_ev),
               "train_segments": len(tr_seg), "eval_segments": len(ev_seg)},
    "train": train, "eval": ev,
    "train_segments": tr_seg, "eval_segments": ev_seg,
}
Path("benchmark_cvpr").mkdir(exist_ok=True)
Path("benchmark_cvpr/split_906_602.json").write_text(json.dumps(out, indent=1))
c = out["counts"]
print(f"TRAIN {c['train']:4d} = {c['train_nuscenes']} nuScenes + {c['train_waymo']} Waymo "
      f"({c['train_segments']} segments)")
print(f"EVAL  {c['eval']:4d} = {c['eval_nuscenes']} nuScenes + {c['eval_waymo']} Waymo "
      f"({c['eval_segments']} segments)")
print("segment-disjoint: VERIFIED   train/eval overlap: NONE")
print("wrote benchmark_cvpr/split_906_602.json")
