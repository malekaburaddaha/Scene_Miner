#!/usr/bin/env python3
"""Stage 1 - the 26-d prior pi of every scene.

    pi = [ z(t_1..t_20) , night, day, rain, dry, dense, sparse ]

  * t: the 20 BLIP-2 Q-Former tag logits of the multi-camera model
    (features_overnight.npz, scripts/export_filter_features_overnight.py), each
    column z-scored over the scene bank (sd + 1e-6), without labels;
  * six one-hot indicators of recorded metadata (scripts/build_deterministic_gt.py:
    nuScenes scene description / Waymo time_of_day and weather, vehicle density from
    3D boxes; a scene can be neither day nor night, e.g. dusk).
Adds `prior` and `prior_cols` to the feature store. With --check it compares the
result with an existing store instead of writing (Milestone 41: max |diff| = 0 on the
released features_v2.npz).
"""
import argparse, json
import numpy as np

COLS = [f"tag{i}" for i in range(20)] + ["lighting=night", "lighting=day", "weather=rain", "weather=dry",
                                         "density=dense", "density=sparse"]


def build_prior(tag_logits, toks, meta):
    T = np.asarray(tag_logits, np.float32)
    Z = (T - T.mean(0, keepdims=True)) / (T.std(0, keepdims=True) + 1e-6)
    M = np.array([[float(meta[t]["lighting"] == "night"), float(meta[t]["lighting"] == "day"),
                   float(meta[t]["weather"] == "rain"), float(meta[t]["weather"] == "dry"),
                   float(meta[t]["density"] == "dense"), float(meta[t]["density"] == "sparse")] for t in toks], np.float32)
    return np.concatenate([Z, M], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--meta", default="work_dirs/unified_filter/deterministic_gt.json")
    ap.add_argument("--check", action="store_true", help="compare with the store's prior instead of writing")
    a = ap.parse_args()
    z = np.load(a.features, allow_pickle=False); toks = [str(t) for t in z["scene_tokens"]]
    P = build_prior(z["tag_logits"], toks, json.load(open(a.meta)))
    if a.check:
        d = float(np.abs(P - z["prior"]).max())
        print(f"[prior] max |rebuilt - stored| = {d:.3g} over {P.shape}")
        raise SystemExit(0 if d < 1e-5 else 1)
    out = {k: z[k] for k in z.files}; out["prior"] = P; out["prior_cols"] = np.array(COLS)
    np.savez_compressed(a.features, **out)
    print(f"[prior] wrote prior {P.shape} into {a.features}")


if __name__ == "__main__":
    main()
