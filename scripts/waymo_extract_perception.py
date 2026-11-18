#!/usr/bin/env python3
"""Stage A — Waymo Open *Perception* (v1.4.3) → neutral PNG/npz dump.

Runs in the standalone ``waymo-extract`` conda env (TensorFlow 2.12 +
waymo-open-dataset-tf-2-12-0).  It NEVER imports torch and writes only
neutral artefacts (PNG + npz + json), so the downstream zero-shot eval
can run inside the untouched ``scene_mining`` env.

For each selected (segment, frame) it dumps:

    <out>/<segment_id>/<frame_idx>/
        cam_FRONT.png  cam_FRONT_LEFT.png  cam_FRONT_RIGHT.png
        cam_SIDE_LEFT.png  cam_SIDE_RIGHT.png          (5 Waymo cameras)
        calib.npz      # intrinsics (5,3,3), extrinsics cam→vehicle (5,4,4),
                       #   ego2world (4,4), image_hw (5,2)
        boxes.npz      # ids, types, centers (N,3), dims (N,3), headings (N,),
                       #   velocities (N,2)  — vehicle frame, this frame
        meta.json      # time_of_day, weather, location, object counts, ts

A top-level ``manifest.json`` lists every extracted frame plus the
forward-looking motion horizon (future agent centres) used later for
ADE/FDE.  Waymo perception is 10 Hz; our model predicts 6 future steps
at 0.5 s ⇒ stride 5 frames, horizon 30 frames.

Usage (inside waymo-extract env):
    python scripts/waymo_extract_perception.py \
        --waymo-root <WAYMO_OPEN_DATASET_v1.4.3>/individual_files/validation \
        --out work_dirs/waymo_zeroshot/extracted \
        --num-segments 4 --frames-per-segment 8 --frame-stride 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset


# Waymo CameraName enum (1-indexed) → canonical name.
WAYMO_CAM_ID_TO_NAME = {
    1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT",
    4: "SIDE_LEFT", 5: "SIDE_RIGHT",
}
# Waymo Label.Type enum.
WAYMO_TYPE_NAME = {0: "UNKNOWN", 1: "VEHICLE", 2: "PEDESTRIAN",
                   3: "SIGN", 4: "CYCLIST"}

# Future-horizon parameters (match our 6×0.5 s = 3 s motion head).
N_FUTURE_STEPS = 6
STEP_STRIDE_FRAMES = 5          # 0.5 s at 10 Hz


def _intrinsic_to_K(intrinsic) -> np.ndarray:
    """Waymo intrinsic = [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3] → 3×3 K."""
    f_u, f_v, c_u, c_v = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    return np.array([[f_u, 0.0, c_u],
                     [0.0, f_v, c_v],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def parse_frame(data) -> open_dataset.Frame:
    # protobuf >= 4.x ParseFromString requires bytes (not bytearray).
    frame = open_dataset.Frame()
    frame.ParseFromString(bytes(data))
    return frame


def extract_cameras(frame) -> dict[str, np.ndarray]:
    """name → (H, W, 3) uint8 RGB."""
    out: dict[str, np.ndarray] = {}
    for img in frame.images:
        name = WAYMO_CAM_ID_TO_NAME.get(img.name)
        if name is None:
            continue
        bgr = cv2.imdecode(np.frombuffer(img.image, np.uint8), cv2.IMREAD_COLOR)
        out[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out


def extract_calib(frame) -> dict[str, np.ndarray]:
    """Stacked intrinsics/extrinsics in canonical camera order."""
    names = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    calib_by_id = {c.name: c for c in frame.context.camera_calibrations}
    Ks, exts, hws = [], [], []
    name_to_id = {v: k for k, v in WAYMO_CAM_ID_TO_NAME.items()}
    for nm in names:
        cid = name_to_id[nm]
        cal = calib_by_id.get(cid)
        if cal is None:
            Ks.append(np.eye(3, dtype=np.float32))
            exts.append(np.eye(4, dtype=np.float32))
            hws.append((0, 0))
            continue
        Ks.append(_intrinsic_to_K(cal.intrinsic))
        # extrinsic.transform is a row-major 4×4 camera→vehicle matrix.
        ext = np.array(cal.extrinsic.transform, dtype=np.float32).reshape(4, 4)
        exts.append(ext)
        hws.append((int(cal.height), int(cal.width)))
    return {
        "intrinsics": np.stack(Ks, axis=0),
        "extrinsics_cam2veh": np.stack(exts, axis=0),
        "image_hw": np.asarray(hws, dtype=np.int32),
        "camera_order": np.asarray(names),
    }


def extract_boxes(frame) -> dict[str, np.ndarray]:
    ids, types, centers, dims, headings, vels = [], [], [], [], [], []
    for lab in frame.laser_labels:
        b = lab.box
        ids.append(lab.id)
        types.append(int(lab.type))
        centers.append([b.center_x, b.center_y, b.center_z])
        dims.append([b.length, b.width, b.height])
        headings.append(b.heading)
        md = lab.metadata
        vels.append([getattr(md, "speed_x", 0.0), getattr(md, "speed_y", 0.0)])
    n = len(ids)
    return {
        "ids":       np.asarray(ids),
        "types":     np.asarray(types, dtype=np.int32),
        "centers":   np.asarray(centers, dtype=np.float32).reshape(n, 3),
        "dims":      np.asarray(dims, dtype=np.float32).reshape(n, 3),
        "headings":  np.asarray(headings, dtype=np.float32).reshape(n),
        "velocities": np.asarray(vels, dtype=np.float32).reshape(n, 2),
    }


def extract_meta(frame) -> dict:
    st = frame.context.stats
    laser_counts = {WAYMO_TYPE_NAME.get(c.type, str(c.type)): int(c.count)
                    for c in st.laser_object_counts}
    return {
        "time_of_day": st.time_of_day,     # "Day" / "Night" / "Dawn/Dusk"
        "weather":     st.weather,         # "sunny" / "rain"
        "location":    st.location,        # "location_sf" / ...
        "laser_object_counts": laser_counts,
        "timestamp_micros":    int(frame.timestamp_micros),
        "ego2world": np.array(frame.pose.transform,
                              dtype=np.float64).reshape(4, 4).tolist(),
    }


def scan_condition(seg_path: Path) -> tuple[str, str]:
    """Read ONLY the first frame of a segment → (time_of_day, weather).

    ``take(1)`` reads a single ~5 MB record, so scanning is cheap even
    over NFS — no full-segment iteration."""
    ds = tf.data.TFRecordDataset(str(seg_path), compression_type="").take(1)
    for data in ds:
        fr = parse_frame(data.numpy())
        st = fr.context.stats
        return st.time_of_day, st.weather
    return "", ""


def select_balanced_segments(all_segs: list[Path], n_target: int,
                             scan_limit: int) -> list[Path]:
    """Scan first-frame conditions and pick a set balanced across
    {night, rain, day-clear}, prioritising night + rain coverage."""
    cands = all_segs[:scan_limit]
    print(f"[extract] scanning conditions of {len(cands)} segments …")
    buckets: dict[str, list[Path]] = {"night": [], "rain": [], "day_clear": []}
    for i, sp in enumerate(cands):
        try:
            tod, weather = scan_condition(sp)
        except Exception as e:        # noqa: BLE001
            print(f"  scan fail {sp.name[:40]}: {e}")
            continue
        key = ("rain" if "rain" in weather.lower()
               else "night" if "night" in tod.lower()
               else "day_clear")
        buckets[key].append(sp)
        if (i + 1) % 40 == 0:
            print(f"  scanned {i+1}/{len(cands)}  "
                  f"(night={len(buckets['night'])} rain={len(buckets['rain'])} "
                  f"day={len(buckets['day_clear'])})")
    print(f"[extract] condition buckets: "
          f"night={len(buckets['night'])} rain={len(buckets['rain'])} "
          f"day_clear={len(buckets['day_clear'])}")

    # Round-robin across buckets so night/rain are represented even though
    # they are the minority of the val split.
    selected: list[Path] = []
    order = ["rain", "night", "day_clear"]   # scarcest first
    idxs = {k: 0 for k in order}
    while len(selected) < n_target and any(
            idxs[k] < len(buckets[k]) for k in order):
        for k in order:
            if idxs[k] < len(buckets[k]) and len(selected) < n_target:
                selected.append(buckets[k][idxs[k]])
                idxs[k] += 1
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waymo-root", required=True,
                    help="…/waymo_open_dataset_v_1_4_3/individual_files/validation")
    ap.add_argument("--out", default="work_dirs/waymo_zeroshot/extracted")
    ap.add_argument("--num-segments", type=int, default=4)
    ap.add_argument("--frames-per-segment", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=10,
                    help="Keyframe stride within a segment (in 10 Hz frames)")
    ap.add_argument("--balance-conditions", action="store_true",
                    help="Scan first-frame conditions and pick a night/rain/"
                         "day-balanced segment set")
    ap.add_argument("--scan-limit", type=int, default=202,
                    help="Max segments to scan when balancing conditions")
    args = ap.parse_args()

    root = Path(args.waymo_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_segs = sorted(root.glob("segment-*.tfrecord"))
    if args.balance_conditions:
        segments = select_balanced_segments(
            all_segs, args.num_segments, args.scan_limit)
    else:
        segments = all_segs[: args.num_segments]
    print(f"[extract] {len(segments)} segments selected from {root}")

    # Frames we actually need to parse per segment: the candidate keyframes
    # plus their forward motion-horizon frames.  Parsing only these (instead
    # of all ~200 frames) makes large multi-segment runs tractable.
    kf_candidates = list(range(0, args.frames_per_segment * args.frame_stride,
                               args.frame_stride))
    needed_idx = set(kf_candidates)
    for kf in kf_candidates:
        for s in range(1, N_FUTURE_STEPS + 1):
            needed_idx.add(kf + s * STEP_STRIDE_FRAMES)
    max_needed = max(needed_idx)

    manifest: list[dict] = []
    for seg_idx, seg_path in enumerate(segments):
        seg_id = seg_path.stem.replace("segment-", "").split("_with_camera")[0]
        print(f"[extract] [{seg_idx+1}/{len(segments)}] {seg_id}")
        ds = tf.data.TFRecordDataset(str(seg_path), compression_type="")

        # Parse-optimised first pass: only the needed frames are decoded.
        frame_protos: dict[int, object] = {}     # keyframe protos (for images)
        frame_pose: dict[int, np.ndarray] = {}
        agent_track: dict[str, dict[int, list[float]]] = {}
        n_frames = 0
        for fi, data in enumerate(ds):
            n_frames = fi + 1
            if fi > max_needed:
                break
            if fi not in needed_idx:
                continue
            frame = parse_frame(data.numpy())
            frame_pose[fi] = np.array(frame.pose.transform,
                                      dtype=np.float64).reshape(4, 4)
            for lab in frame.laser_labels:
                agent_track.setdefault(lab.id, {})[fi] = [
                    lab.box.center_x, lab.box.center_y, lab.box.center_z]
            if fi in kf_candidates:
                frame_protos[fi] = frame

        def _future_in_capture_ego(aid: str, cap_fi: int) -> list:
            P_cap_inv = np.linalg.inv(frame_pose[cap_fi])
            track = agent_track.get(aid, {})
            steps = []
            for s in range(1, N_FUTURE_STEPS + 1):
                tf_idx = cap_fi + s * STEP_STRIDE_FRAMES
                c = track.get(tf_idx)
                if c is None:
                    steps.append(None)
                    continue
                world = frame_pose[tf_idx] @ np.array([c[0], c[1], c[2], 1.0])
                ego = P_cap_inv @ world
                steps.append([float(ego[0]), float(ego[1])])
            return steps

        # Valid keyframes = those actually present in the segment.
        kf_idxs = [kf for kf in kf_candidates if kf in frame_protos]
        if not kf_idxs:
            print(f"[extract]   segment too short ({n_frames} frames) — skip")
            continue

        for fi in kf_idxs:
            frame = frame_protos[fi]
            fdir = out_root / seg_id / f"{fi:04d}"
            fdir.mkdir(parents=True, exist_ok=True)

            cams = extract_cameras(frame)
            for nm, rgb in cams.items():
                cv2.imwrite(str(fdir / f"cam_{nm}.png"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.savez(fdir / "calib.npz", **extract_calib(frame))
            boxes = extract_boxes(frame)
            np.savez(fdir / "boxes.npz", **boxes)
            meta = extract_meta(frame)
            (fdir / "meta.json").write_text(json.dumps(meta, indent=2))

            future = {aid: _future_in_capture_ego(aid, fi)
                      for aid in boxes["ids"].tolist()}
            (fdir / "future.json").write_text(json.dumps(future))

            manifest.append({
                "segment": seg_id,
                "frame_idx": fi,
                "rel_dir": str((fdir).relative_to(out_root)),
                "n_cameras": len(cams),
                "n_agents": int(len(boxes["ids"])),
                "time_of_day": meta["time_of_day"],
                "weather": meta["weather"],
                "location": meta["location"],
            })
        print(f"[extract]   wrote {len(kf_idxs)} keyframes "
              f"(segment ≥{n_frames} frames, "
              f"tod={meta['time_of_day']!r} weather={meta['weather']!r})")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[extract] DONE — {len(manifest)} frames → {out_root}/manifest.json")


if __name__ == "__main__":
    main()
