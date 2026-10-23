#!/usr/bin/env python3
"""MILESTONE 15 - human semantic ground truth from the official nuScenes tables.

Every flag here comes from nuScenes `sample_annotation` / `instance` /
`category` / `attribute` / `visibility` (human 3D annotation) or from the
human-written `scene.description`. No model output is used anywhere.

Geometry: an object counts only if it is inside the CAM_FRONT field of view
(|azimuth| <= 32.6 deg in the ego frame, the camera's own half-FOV) and at
least partly visible (visibility level >= 2, i.e. > 40 %). A SCENE flag is set
when the condition holds in at least 25 % of the scene's annotated keyframes, so
a single passing object does not label a 20-second scene; the per-scene
fractions are stored so other thresholds can be checked.

Definitions (fixed before any count was looked at):
  pedestrian_close   adult / child / construction_worker / police_officer
                     pedestrian within 25 m.
  vru                bicycle or motorcycle WITH RIDER (attribute
                     cycle.with_rider) within 40 m, or any pedestrian within 15 m.
  construction_zone  >= 3 traffic cones, or cones together with a barrier, or a
                     construction worker, or a construction vehicle, within 40 m.
                     Deviation from the brief: barriers ALONE do not count -
                     they line ordinary roads throughout nuScenes and would mark
                     most scenes as construction zones.
  heavy_vehicle      rigid or bendy bus, truck, trailer or construction vehicle
                     within 25 m.
  rare_or_child      child, stroller, wheelchair, police officer, emergency
                     vehicle, debris or pushable/pullable object within 40 m.
  night, rain        'night' / 'rain' in the human-written scene description.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

PED = {"human.pedestrian.adult", "human.pedestrian.child",
       "human.pedestrian.construction_worker", "human.pedestrian.police_officer"}
HEAVY = {"vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.truck",
         "vehicle.trailer", "vehicle.construction"}
RARE = {"human.pedestrian.child", "human.pedestrian.stroller",
        "human.pedestrian.wheelchair", "human.pedestrian.police_officer",
        "vehicle.emergency.ambulance", "vehicle.emergency.police",
        "movable_object.debris", "movable_object.pushable_pullable"}
FLAGS = ["pedestrian_close", "vru", "construction_zone", "heavy_vehicle", "rare_or_child"]


def yaw_inv(q):
    w, x, y, z = q
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, -s], [s, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--features", default="work_dirs/unified_filter/features_v2.npz")
    ap.add_argument("--val-thumbs", default="work_dirs/val150_thumbs")
    ap.add_argument("--min-frac", type=float, default=0.25)
    ap.add_argument("--out", default="work_dirs/unified_filter/human_semantic_gt.json")
    a = ap.parse_args()
    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=a.dataroot, verbose=False)
    z = np.load(a.features, allow_pickle=False)
    corpus_nus = [str(t) for t, d in zip(z["scene_tokens"], z["domain"]) if int(d) == 0]
    val150 = sorted(p.stem for p in Path(a.val_thumbs).glob("*.jpg"))
    want = {t: "corpus" for t in corpus_nus}; want.update({t: "val150" for t in val150})

    sd = nusc.get("sample_data", nusc.get("sample", nusc.scene[0]["first_sample_token"])["data"]["CAM_FRONT"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    half = float(np.arctan(sd["width"] / (2 * np.array(cs["camera_intrinsic"])[0, 0])))
    attr_name = {r["token"]: r["name"] for r in nusc.attribute}

    out = {}
    for sc in nusc.scene:
        tok = sc["token"]
        if tok not in want: continue
        desc = sc["description"].lower()
        per_frame = defaultdict(list); counts = defaultdict(int)
        st = sc["first_sample_token"]; nf = 0
        while st:
            s = nusc.get("sample", st); nf += 1
            pose = nusc.get("ego_pose", nusc.get("sample_data", s["data"]["LIDAR_TOP"])["ego_pose_token"])
            R = yaw_inv(pose["rotation"]); P0 = np.array(pose["translation"][:2])
            f = {k: False for k in FLAGS}; cones = barrier = 0
            for at in s["anns"]:
                ann = nusc.get("sample_annotation", at)
                if int(ann["visibility_token"]) < 2: continue
                v = R @ (np.array(ann["translation"][:2]) - P0)
                if v[0] <= 0 or abs(np.arctan2(v[1], v[0])) > half: continue
                d = float(np.hypot(*v)); cat = ann["category_name"]
                attrs = {attr_name[x] for x in ann["attribute_tokens"]}
                if cat in PED and d <= 25: f["pedestrian_close"] = True; counts["pedestrian_close"] += 1
                if (cat in ("vehicle.bicycle", "vehicle.motorcycle") and "cycle.with_rider" in attrs and d <= 40) \
                        or (cat.startswith("human.pedestrian") and d <= 15):
                    f["vru"] = True; counts["vru"] += 1
                if d <= 40:
                    if cat == "movable_object.trafficcone": cones += 1
                    if cat == "movable_object.barrier": barrier += 1
                    if cat in ("human.pedestrian.construction_worker", "vehicle.construction"):
                        f["construction_zone"] = True
                    if cat in RARE: f["rare_or_child"] = True; counts["rare_or_child"] += 1
                if cat in HEAVY and d <= 25: f["heavy_vehicle"] = True; counts["heavy_vehicle"] += 1
            if cones >= 3 or (cones >= 1 and barrier >= 1): f["construction_zone"] = True
            counts["cones"] += cones
            for k in FLAGS: per_frame[k].append(f[k])
            st = s["next"]
        frac = {k: float(np.mean(per_frame[k])) for k in FLAGS}
        out[tok] = {"source": want[tok], "n_keyframes": nf,
                    **{k: frac[k] >= a.min_frac for k in FLAGS},
                    "frac": frac, "counts": dict(counts),
                    "night": "night" in desc, "rain": "rain" in desc,
                    "description": sc["description"]}
    meta = {"definition_doc": __doc__, "camfront_half_fov_deg": float(np.degrees(half)),
            "min_keyframe_frac": a.min_frac, "n_scenes": len(out)}
    Path(a.out).write_text(json.dumps({"meta": meta, "scenes": out}, indent=1))
    for src in ("corpus", "val150"):
        S = [v for v in out.values() if v["source"] == src]
        print(f"[{src}] {len(S)} scenes: " + "  ".join(
            f"{k} {sum(v[k] for v in S)}" for k in FLAGS + ["night", "rain"]))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
