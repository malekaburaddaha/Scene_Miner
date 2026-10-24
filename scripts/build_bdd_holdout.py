#!/usr/bin/env python3
"""MILESTONE 28 - an untouched BDD100K holdout with native human labels.

Source: the ungated `dgural/bdd100k` mirror (10,000 images with BDD100K's own human
weather / timeofday / scene tags and 2D boxes). The 1,000 images used since
Milestone 8 are excluded; from the remaining 9,000 we draw 2,000 with a fixed seed,
oversampling adverse conditions as in Milestone 8 so rare classes are testable.
Also writes the box labels of the original 1,000 images (for figure callouts).
"""
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/nfs/turbo/umd-srawa/scene_mining/datasets/external_adverse")


def lab(r, k):
    v = r.get(k)
    return v.get("label") if isinstance(v, dict) else v


def boxes(r):
    d = r.get("detections") or {}
    out = []
    for b in d.get("detections", []) if isinstance(d, dict) else []:
        out.append({"label": b["label"], "bbox": [round(x, 5) for x in b["bounding_box"]],
                    "occluded": bool(b.get("occluded", False)), "truncated": bool(b.get("truncated", False))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="dgural/bdd100k")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    recs = json.load(open(hf_hub_download(a.repo, "samples.json", repo_type="dataset")))["samples"]
    used = set(json.load(open(ROOT/"bdd100k"/"labels.json")))
    # box labels of the images already in use (for Figure 3 callouts only)
    old = {Path(r["filepath"]).name: boxes(r) for r in recs if Path(r["filepath"]).name in used}
    (ROOT/"bdd100k"/"boxes.json").write_text(json.dumps(old))
    pool = [r for r in recs if Path(r["filepath"]).name not in used
            and lab(r, "weather") not in (None, "undefined") and lab(r, "timeofday") not in (None, "undefined")]
    rng = random.Random(a.seed)
    by = defaultdict(list)
    for r in pool:
        adverse = lab(r, "weather") in ("rainy", "snowy", "foggy") or lab(r, "timeofday") == "night"
        by["adverse" if adverse else "benign"].append(r)
    for v in by.values(): rng.shuffle(v)
    n_adv = min(len(by["adverse"]), int(a.n * 0.6))
    sel = by["adverse"][:n_adv] + by["benign"][:a.n - n_adv]
    rng.shuffle(sel)
    out = ROOT/"bdd100k_holdout"; (out/"images").mkdir(parents=True, exist_ok=True)
    meta = {}
    for i, r in enumerate(sel):
        fn = Path(r["filepath"]).name
        dst = out/"images"/fn
        if not dst.exists():
            dst.write_bytes(Path(hf_hub_download(a.repo, r["filepath"], repo_type="dataset")).read_bytes())
        meta[fn] = {"weather": lab(r, "weather"), "timeofday": lab(r, "timeofday"), "scene": lab(r, "scene"),
                    "boxes": boxes(r)}
        if (i + 1) % 250 == 0: print(f"  [holdout] {i+1}/{len(sel)}", flush=True)
    assert not (set(meta) & used)
    (out/"labels.json").write_text(json.dumps(meta))
    print(f"pool {len(pool)}; selected {len(meta)} ({n_adv} adverse); boxes for {len(old)} original images")
    print("weather:", dict(Counter(v["weather"] for v in meta.values()).most_common()))
    print("timeofday:", dict(Counter(v["timeofday"] for v in meta.values()).most_common()))
    print("box labels:", dict(Counter(b["label"] for v in meta.values() for b in v["boxes"]).most_common()))


if __name__ == "__main__":
    main()
