"""Extract per-scene binary tag labels from nuPrompt annotations.

For each of the 850 nuScenes scenes, this script:
1. Loads all nuPrompt JSON files in the scene folder
2. Collects every prompt string (lowercased)
3. Keyword-matches against 20 predefined tags
4. Outputs a JSON mapping scene_token → 20-dim binary list

Output: data/labels/nuprompt_tags.json
"""

import json
import os
import re
from pathlib import Path

# ── 20 tags with keyword lists ───────────────────────────────────────────────
# Each tag has a name and a list of keywords/phrases to search for in prompts.
# A tag fires (=1) if ANY keyword is found in ANY prompt for that scene.

TAG_DEFINITIONS = [
    # idx  name                keywords
    (0,  "night",             ["night", "dark", "nighttime", "night time"]),
    (1,  "rain",              ["rain", "wet", "rainy"]),
    (2,  "highway",           ["highway", "freeway", "expressway", "motorway"]),
    (3,  "intersection",      ["intersection", "crossroad", "junction"]),
    (4,  "turning",           ["turning", "turn ", "curving", "bending",
                               "right turn", "left turn", "in the process of turning",
                               "in the middle of turning", "making a turn"]),
    (5,  "heavy_traffic",     ["traffic", "congestion", "congested", "crowded"]),
    (6,  "pedestrian",        ["pedestrian", "person ", "people", "walking",
                               "crosses the street", "crossing the road",
                               "crossing the street", "on foot", "someone walking",
                               "someone on foot"]),
    (7,  "cyclist",           ["bicycle", "cyclist", "bike", "cycling"]),
    (8,  "construction",      ["construction", "cone", "barrier", "work zone"]),
    (9,  "parking",           ["parking", "parked", "stationary", "stopped",
                               "stopping by the side", "not in motion", "not moving"]),
    (10, "low_visibility",    ["fog", "visibility", "glare", "obscured"]),
    (11, "bus",               ["bus"]),
    (12, "truck",             ["truck", "trailer"]),
    (13, "motorcycle",        ["motorcycle", "motorbike"]),
    (14, "opposite_direction",["opposite direction", "oncoming",
                               "reverse direction", "against"]),
    (15, "lane_change",       ["lane change", "merging", "merge"]),
    (16, "sidewalk",          ["sidewalk", "pavement", "walkway",
                               "pedestrian path"]),
    (17, "same_direction",    ["same direction", "same way"]),
    (18, "front_vehicle",     ["front car", "front vehicle", "vehicle ahead",
                               "car ahead"]),
    (19, "moving",            ["moving", "in motion", "mobile"]),
]

N_TAGS = len(TAG_DEFINITIONS)
TAG_NAMES = [t[1] for t in TAG_DEFINITIONS]


def extract_tags_for_scene(scene_dir: Path) -> list[int]:
    """Load all JSON prompts in a scene folder, return 20-dim binary vector."""
    # Collect all prompt strings from all JSON files
    all_prompts = []
    for jf in scene_dir.glob("*.json"):
        with open(jf) as f:
            data = json.load(f)
        prompts = data.get("prompt", [])
        if isinstance(prompts, str):
            prompts = [prompts]
        all_prompts.extend(prompts)

    # Also include the original_prompt if present
    for jf in scene_dir.glob("*.json"):
        with open(jf) as f:
            data = json.load(f)
        orig = data.get("original_prompt")
        if orig and isinstance(orig, str):
            all_prompts.append(orig)

    # Lowercase everything into one big searchable blob
    blob = " ".join(p.lower().replace("_", " ") for p in all_prompts)

    # Match each tag
    tags = [0] * N_TAGS
    for idx, name, keywords in TAG_DEFINITIONS:
        for kw in keywords:
            if kw.lower() in blob:
                tags[idx] = 1
                break

    return tags


def main():
    nuprompt_dir = Path("data/nuprompt_v1.0")
    output_path = Path("data/labels/nuprompt_tags.json")

    if not nuprompt_dir.exists():
        print(f"[error] nuPrompt directory not found: {nuprompt_dir}")
        return

    scene_dirs = sorted(
        p for p in nuprompt_dir.iterdir() if p.is_dir()
    )
    print(f"[tags] Found {len(scene_dirs)} scenes in {nuprompt_dir}")

    results = {}
    tag_counts = [0] * N_TAGS  # track how often each tag fires

    for scene_dir in scene_dirs:
        scene_token = scene_dir.name
        tags = extract_tags_for_scene(scene_dir)
        results[scene_token] = tags
        for i, v in enumerate(tags):
            tag_counts[i] += v

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[tags] Saved {len(results)} scenes → {output_path}")

    # Print tag distribution
    print(f"\n{'Tag':<25s} {'Count':>6s} {'%':>7s}")
    print("-" * 40)
    for i, name in enumerate(TAG_NAMES):
        pct = 100.0 * tag_counts[i] / len(results)
        print(f"  {name:<23s} {tag_counts[i]:>5d}  {pct:>5.1f}%")

    # Scenes with zero tags
    zero_tag_scenes = sum(1 for v in results.values() if sum(v) == 0)
    print(f"\nScenes with zero tags: {zero_tag_scenes}")
    avg_tags = sum(sum(v) for v in results.values()) / len(results)
    print(f"Average tags per scene: {avg_tags:.1f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
