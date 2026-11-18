"""Compute per-scene GT interaction risk scores from nuScenes annotations.

For each scene, samples multiple keyframes and computes a physics-based risk
score from three components:

1. Proximity — inverse-exponential distance of closest agent to ego vehicle.
   Closer agents = higher risk. Scale: 5m → 0.37, 2m → 0.67, 0m → 1.0.

2. TTC (Time-To-Collision) — estimated time until an agent would collide with
   ego if both maintain current velocities. Low TTC = high risk.
   Uses simplified linear projection: TTC = -dot(Δpos, Δvel) / |Δvel|².

3. VRU density — fraction of agents within 15m that are pedestrians, cyclists,
   or motorcyclists. More vulnerable road users = higher risk.

Each component is in [0, 1]. The per-keyframe score is a weighted average:
   score = w_prox × proximity + w_ttc × ttc_risk + w_vru × vru_density

The per-scene score is the maximum across all keyframes (the "most dangerous
moment" in the scene determines the scene difficulty).

Output: data/labels/gt_risk_scores.json → {scene_token: float}
"""

import json
import os
import math
from pathlib import Path

import numpy as np


# ── VRU categories ───────────────────────────────────────────────────────────
VRU_CATEGORIES = {
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.police_officer",
    "human.pedestrian.personal_mobility",
    "vehicle.bicycle",
    "vehicle.motorcycle",
}

# Ignore static objects that aren't real traffic agents
IGNORE_CATEGORIES = {
    "movable_object.trafficcone",
    "movable_object.barrier",
    "movable_object.pushable_pullable",
    "movable_object.debris",
    "static_object.bicycle_rack",
}

# ── Scoring weights ──────────────────────────────────────────────────────────
W_PROXIMITY = 0.4
W_TTC = 0.3
W_VRU = 0.3

PROXIMITY_SCALE = 5.0       # metres — exponential decay scale
TTC_THRESHOLD = 4.0         # seconds — below this is "dangerous"
NEARBY_RADIUS = 15.0        # metres — agents within this radius are "nearby"


def compute_keyframe_score(nusc, sample_token: str) -> float:
    """Compute risk score for a single keyframe."""
    sample = nusc.get("sample", sample_token)

    # Get ego pose
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_pos = np.array(ego_pose["translation"][:2])  # (x, y) in global frame

    # Compute ego velocity from consecutive ego poses
    ego_vel = np.array([0.0, 0.0])
    if sd["prev"]:
        sd_prev = nusc.get("sample_data", sd["prev"])
        ego_prev = nusc.get("ego_pose", sd_prev["ego_pose_token"])
        dt = (ego_pose["timestamp"] - ego_prev["timestamp"]) / 1e6  # seconds
        if dt > 0:
            ego_vel = (np.array(ego_pose["translation"][:2]) -
                       np.array(ego_prev["translation"][:2])) / dt

    # Process all annotations
    ann_tokens = sample["anns"]
    if not ann_tokens:
        return 0.0

    distances = []
    ttc_values = []
    n_nearby = 0
    n_vru_nearby = 0

    for ann_token in ann_tokens:
        ann = nusc.get("sample_annotation", ann_token)
        cat = ann["category_name"]

        # Skip static objects
        if cat in IGNORE_CATEGORIES:
            continue

        # Agent position relative to ego
        agent_pos = np.array(ann["translation"][:2])
        dist = np.linalg.norm(agent_pos - ego_pos)
        distances.append(dist)

        # Count nearby agents and VRUs
        if dist < NEARBY_RADIUS:
            n_nearby += 1
            if cat in VRU_CATEGORIES:
                n_vru_nearby += 1

        # Compute TTC using nusc.box_velocity
        try:
            agent_vel = nusc.box_velocity(ann_token)[:2]  # (vx, vy) global
        except Exception:
            agent_vel = np.array([0.0, 0.0])

        # Relative position and velocity
        delta_pos = agent_pos - ego_pos  # agent relative to ego
        delta_vel = agent_vel - ego_vel  # agent velocity relative to ego

        # TTC: time until closest approach
        # TTC = -dot(Δpos, Δvel) / |Δvel|²  (only valid if agents are approaching)
        speed_sq = np.dot(delta_vel, delta_vel)
        if speed_sq > 0.1:  # avoid division by near-zero
            ttc = -np.dot(delta_pos, delta_vel) / speed_sq
            if 0 < ttc < TTC_THRESHOLD:
                # Check if closest approach distance is small enough to be dangerous
                closest_dist = np.linalg.norm(delta_pos + delta_vel * ttc)
                if closest_dist < 3.0:  # within 3 metres at closest approach
                    ttc_values.append(ttc)

    if not distances:
        return 0.0

    # ── Component 1: Proximity ───────────────────────────────────────────
    min_dist = min(distances)
    proximity = math.exp(-min_dist / PROXIMITY_SCALE)  # [0, 1]

    # ── Component 2: TTC risk ────────────────────────────────────────────
    if ttc_values:
        min_ttc = min(ttc_values)
        # Map: 0s → 1.0, 2s → 0.5, 4s → 0.0
        ttc_risk = max(0.0, 1.0 - min_ttc / TTC_THRESHOLD)
    else:
        ttc_risk = 0.0

    # ── Component 3: VRU density ─────────────────────────────────────────
    vru_density = n_vru_nearby / max(n_nearby, 1)

    # ── Combined ─────────────────────────────────────────────────────────
    score = W_PROXIMITY * proximity + W_TTC * ttc_risk + W_VRU * vru_density
    return min(score, 1.0)


def compute_scene_score(nusc, scene: dict) -> float:
    """Compute risk score for a full scene (max across keyframes)."""
    sample_token = scene["first_sample_token"]
    scores = []

    while sample_token:
        score = compute_keyframe_score(nusc, sample_token)
        scores.append(score)
        sample = nusc.get("sample", sample_token)
        sample_token = sample["next"] if sample["next"] else None

    return max(scores) if scores else 0.0


def main():
    from nuscenes.nuscenes import NuScenes

    output_path = Path("data/labels/gt_risk_scores.json")

    print("[risk] Loading nuScenes...")
    nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes", verbose=False)
    print(f"[risk] {len(nusc.scene)} scenes loaded")

    results = {}
    for i, scene in enumerate(nusc.scene):
        score = compute_scene_score(nusc, scene)
        results[scene["token"]] = round(score, 6)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(nusc.scene)}] scored")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[risk] Saved {len(results)} scores → {output_path}")

    # Statistics
    scores = list(results.values())
    print(f"\n  min={min(scores):.4f}  max={max(scores):.4f}  "
          f"mean={sum(scores)/len(scores):.4f}  "
          f"median={sorted(scores)[len(scores)//2]:.4f}")

    # Distribution
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = sum(1 for s in scores if lo <= s < hi)
        bar = "█" * (count // 5)
        print(f"  [{lo:.1f}-{hi:.1f}) {count:>4d} {bar}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
