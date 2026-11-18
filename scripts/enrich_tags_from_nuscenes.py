"""Enrich nuprompt_tags.json with environmental tags from nuScenes scene descriptions.

nuPrompt is strong for object/action tags but weak for weather/lighting/environment.
nuScenes scene.json has expert-written descriptions covering rain, night, construction,
etc. This script keyword-matches those descriptions and merges into existing tags
using OR logic (never flips a 1 back to 0).

Run after extract_nuprompt_tags.py.
"""

import json
import os
from pathlib import Path

# Tags to enrich from scene.json — index must match TAG_DEFINITIONS in extract_nuprompt_tags.py
SCENE_DESC_TAGS = {
    0:  ("night",         ["night"]),
    1:  ("rain",          ["rain"]),
    2:  ("highway",       ["highway", "freeway", "expressway"]),
    3:  ("intersection",  ["intersection", "cross intersection"]),
    4:  ("turning",       ["turn left", "turn right", "turning"]),
    5:  ("heavy_traffic",  ["traffic", "busy"]),
    8:  ("construction",  ["construction"]),
    9:  ("parking",       ["parking", "parked"]),
    15: ("lane_change",   ["lane change", "overtake", "overtaken", "overtaking"]),
    16: ("sidewalk",      ["sidewalk", "pedestrian path"]),
}


def main():
    scene_json_path = Path("data/nuscenes/v1.0-trainval/scene.json")
    tags_path = Path("data/labels/nuprompt_tags.json")

    with open(scene_json_path) as f:
        scenes = json.load(f)
    with open(tags_path) as f:
        tags = json.load(f)

    print(f"[enrich] Loaded {len(scenes)} scene descriptions, {len(tags)} existing tag entries")

    # Build lookup: scene_token → description
    desc_map = {s["token"]: s["description"].lower() for s in scenes}

    # Track before/after counts per enriched tag
    before_counts = {}
    after_counts = {}
    for idx in SCENE_DESC_TAGS:
        before_counts[idx] = sum(1 for v in tags.values() if v[idx] == 1)

    # Enrich
    enriched = 0
    for scene_token, tag_vec in tags.items():
        desc = desc_map.get(scene_token, "")
        if not desc:
            continue
        for idx, (name, keywords) in SCENE_DESC_TAGS.items():
            if tag_vec[idx] == 1:
                continue  # already set by nuPrompt, keep it
            for kw in keywords:
                if kw in desc:
                    tag_vec[idx] = 1
                    enriched += 1
                    break

    # After counts
    for idx in SCENE_DESC_TAGS:
        after_counts[idx] = sum(1 for v in tags.values() if v[idx] == 1)

    # Save
    with open(tags_path, "w") as f:
        json.dump(tags, f, indent=2)
    print(f"[enrich] Flipped {enriched} tag bits (0→1). Saved → {tags_path}")

    # Report
    print(f"\n{'Tag':<25s} {'Before':>7s} {'After':>7s} {'Added':>7s}")
    print("-" * 50)
    for idx, (name, _) in SCENE_DESC_TAGS.items():
        b, a = before_counts[idx], after_counts[idx]
        print(f"  {name:<23s} {b:>6d}  {a:>6d}  +{a - b:>5d}")

    # Overall stats
    avg_tags = sum(sum(v) for v in tags.values()) / len(tags)
    zero_scenes = sum(1 for v in tags.values() if sum(v) == 0)
    print(f"\nAverage tags per scene: {avg_tags:.1f}")
    print(f"Scenes with zero tags: {zero_scenes}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
