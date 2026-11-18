"""Prepare NuScenes annotation pkl files in mmdet3d-compatible format.

Generates ``nuscenes_infos_{split}.pkl`` files that contain pre-computed
camera parameters and 3D bounding box annotations, following the mmdet3d
NuScenes info convention.  These pkl files allow fast dataset loading without
repeated nuscenes-devkit queries at training time.

Usage::

    python scripts/prepare_nuscenes_data.py \\
        --dataroot ./data/nuscenes \\
        --version  v1.0-mini \\
        --out-dir  ./data/nuscenes \\
        --splits   train val

Output files (one per split)::

    {out_dir}/nuscenes_infos_{split}.pkl

Each pkl is a list of dicts with keys::

    token              str   — nuScenes sample token
    timestamp          int   — UNIX microseconds
    ego2global         ndarray (4,4) float32 — ego→global SE3
    cams               dict[cam_name → cam_info_dict]
    lidar_path         str   — path to LIDAR_TOP .pcd.bin relative to dataroot
    gt_boxes           ndarray (N,9) — [cx,cy,cz,dx,dy,dz,yaw,vx,vy] in ego frame
    gt_names           list[str] — N category name strings (detection classes only)
    num_lidar_pts      list[int] — points inside each box
    valid_flag         list[bool] — True if num_lidar_pts > 0

cam_info_dict keys::

    data_path          str   — path relative to dataroot
    sensor2ego         ndarray (4,4) — camera→ego SE3
    ego2global         ndarray (4,4) — ego→global SE3 at camera capture time
    cam_intrinsic      ndarray (3,3)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


# NuScenes 10-class detection set (mirrors NUSCENES_DETECTION_CLASSES in dataset)
DETECTION_CLASSES = {
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone", "barrier",
}

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


def _quat_to_rot(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y**2+z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2+z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2+y**2)],
    ], dtype=np.float32)


def _make_se3(rotation_quat, translation) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _quat_to_rot(rotation_quat)
    T[:3, 3] = np.array(translation, dtype=np.float32)
    return T


def _category_to_detection(category_name: str) -> str | None:
    """Map a nuScenes category string to one of the 10 detection class names."""
    for part in category_name.split("."):
        if part in DETECTION_CLASSES:
            return part
    return None


def _get_box_velocity(nusc: NuScenes, sample_annotation_token: str) -> np.ndarray:
    """Return (vx, vy) velocity in ego frame, or (nan, nan) if unavailable."""
    ann = nusc.get("sample_annotation", sample_annotation_token)
    if ann["prev"] == "":
        return np.array([np.nan, np.nan], dtype=np.float32)
    prev = nusc.get("sample_annotation", ann["prev"])
    # Time delta in seconds
    curr_sd = nusc.get("sample_data",
                       nusc.get("sample", ann["sample_token"])["data"]["LIDAR_TOP"])
    prev_sd = nusc.get("sample_data",
                       nusc.get("sample", prev["sample_token"])["data"]["LIDAR_TOP"])
    dt = (curr_sd["timestamp"] - prev_sd["timestamp"]) * 1e-6
    if abs(dt) < 1e-6:
        return np.array([np.nan, np.nan], dtype=np.float32)

    diff = np.array(ann["translation"]) - np.array(prev["translation"])
    vel_global = diff[:2] / dt

    # Rotate to ego frame at current sample
    ego_pose = nusc.get("ego_pose", curr_sd["ego_pose_token"])
    R_e2g = _quat_to_rot(ego_pose["rotation"])[:2, :2]
    vel_ego = R_e2g.T @ vel_global
    return vel_ego.astype(np.float32)


def process_sample(nusc: NuScenes, sample_token: str) -> dict:
    """Build an info dict for one nuScenes sample."""
    sample = nusc.get("sample", sample_token)

    # Ego pose (from LIDAR_TOP)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    ego2global = _make_se3(ego_pose["rotation"], ego_pose["translation"])
    R_e2g = ego2global[:3, :3].astype(np.float64)
    t_e2g = ego2global[:3, 3].astype(np.float64)
    R_g2e = R_e2g.T

    # Camera info
    cams: dict = {}
    for cam_name in CAMERAS:
        cam_token = sample["data"].get(cam_name)
        if cam_token is None:
            continue
        cam_sd = nusc.get("sample_data", cam_token)
        cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_ego_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
        sensor2ego = _make_se3(cam_cs["rotation"], cam_cs["translation"])
        cam_ego2global = _make_se3(cam_ego_pose["rotation"], cam_ego_pose["translation"])
        cams[cam_name] = dict(
            data_path=cam_sd["filename"],
            sensor2ego=sensor2ego,
            ego2global=cam_ego2global,
            cam_intrinsic=np.array(cam_cs["camera_intrinsic"], dtype=np.float32),
            timestamp=cam_sd["timestamp"],
        )

    # 3D bounding boxes in ego frame
    gt_boxes_list, gt_names_list, num_pts_list, valid_list = [], [], [], []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det_name = _category_to_detection(ann["category_name"])
        if det_name is None:
            continue

        # Centre in ego frame
        c_global = np.array(ann["translation"], dtype=np.float64)
        c_ego = R_g2e @ (c_global - t_e2g)

        # Box size [dx, dy, dz]
        w, l, h = ann["size"]   # nuScenes: [width, length, height]

        # Yaw in ego frame
        R_box_global = _quat_to_rot(ann["rotation"]).astype(np.float64)
        R_box_ego = R_g2e @ R_box_global
        yaw = float(np.arctan2(R_box_ego[1, 0], R_box_ego[0, 0]))

        # Velocity
        vel = _get_box_velocity(nusc, ann_token)

        gt_boxes_list.append([
            c_ego[0], c_ego[1], c_ego[2],
            l, w, h,         # dx=length, dy=width, dz=height
            yaw,
            vel[0], vel[1],
        ])
        gt_names_list.append(det_name)
        num_pts_list.append(ann.get("num_lidar_pts", 0))
        valid_list.append(ann.get("num_lidar_pts", 0) > 0)

    gt_boxes = np.array(gt_boxes_list, dtype=np.float32).reshape(-1, 9)

    return dict(
        token=sample_token,
        timestamp=sample["timestamp"],
        ego2global=ego2global,
        cams=cams,
        lidar_path=lidar_sd["filename"],
        gt_boxes=gt_boxes,
        gt_names=gt_names_list,
        num_lidar_pts=num_pts_list,
        valid_flag=valid_list,
    )


def create_infos(
    dataroot: str,
    version: str,
    out_dir: str,
    splits: list[str],
) -> None:
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    splits_scenes = create_splits_scenes()

    for split in splits:
        split_key = split
        # v1.0-mini uses "mini_train" / "mini_val" in create_splits_scenes
        if "mini" in version:
            split_key = f"mini_{split}" if f"mini_{split}" in splits_scenes else split
        scene_names = set(splits_scenes.get(split_key, []))

        sample_tokens = []
        for scene in nusc.scene:
            if scene["name"] not in scene_names:
                continue
            # Walk the linked-list of samples in this scene
            token = scene["first_sample_token"]
            while token != "":
                sample_tokens.append(token)
                token = nusc.get("sample", token)["next"]

        print(f"Processing split={split!r}: {len(sample_tokens)} samples")
        infos = []
        for i, tok in enumerate(sample_tokens):
            if i % 50 == 0:
                print(f"  {i}/{len(sample_tokens)}")
            infos.append(process_sample(nusc, tok))

        out_path = Path(out_dir) / f"nuscenes_infos_{split}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(infos, f)
        print(f"Saved {len(infos)} infos → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NuScenes mmdet3d-style pkl files")
    parser.add_argument("--dataroot", default="./data/nuscenes")
    parser.add_argument("--version",  default="v1.0-mini")
    parser.add_argument("--out-dir",  default="./data/nuscenes")
    parser.add_argument("--splits",   nargs="+", default=["train", "val"])
    args = parser.parse_args()

    create_infos(
        dataroot=args.dataroot,
        version=args.version,
        out_dir=args.out_dir,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
