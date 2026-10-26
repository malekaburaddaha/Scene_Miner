#!/usr/bin/env python3
"""Dump SceneMiner (D_final) scores over the full corpus for every TASK-1 query."""
from paths import QWEN_VL
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import MACRO8, DEGRAD, SEVERITY, sceneminer_scores

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="work_dirs/unified_filter")
ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
ap.add_argument("--checkpoint", default="work_dirs/ablation/D_final/final.pt")
ap.add_argument("--qwen", default=QWEN_VL)
ap.add_argument("--tau", type=float, default=0.30)
ap.add_argument("--lora-r", type=int, default=64)
ap.add_argument("--img-side", type=int, default=336)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--out", default="results/overnight_oct18/sm_scores.json")
a = ap.parse_args()
W = Path(a.work)
z = np.load(W/"features_overnight.npz", allow_pickle=False)
toks = [str(t) for t in z["scene_tokens"]]
qs = [q for q, _ in MACRO8] + [q for q, _ in DEGRAD] + [q for q, _ in SEVERITY]
sc, t_idx, lat = sceneminer_scores(a, toks, qs, W)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({k: v.tolist() for k, v in sc.items()}))
json.dump({"index_build_s": t_idx, "query_ms_median": lat, "n_queries": len(qs)},
          open(str(a.out).replace(".json", "_latency.json"), "w"), indent=1)
print(f"[dump] {len(sc)} queries -> {a.out} | index {t_idx:.1f}s | query {lat:.2f}ms")
