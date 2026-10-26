#!/usr/bin/env python3
"""MILESTONE 8 - fetch a stratified BDD100K sample with native human labels.

BDD100K carries per-image human annotations for `weather`
(clear / overcast / rainy / snowy / foggy / partly cloudy) and `timeofday`
(daytime / night / dawn-dusk). No portal login: the `dgural/bdd100k` mirror on
the HuggingFace Hub is ungated. We oversample the adverse conditions so the
rare classes are testable.
"""
from __future__ import annotations
from paths import EXT
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="dgural/bdd100k")
    ap.add_argument("--out", default=EXT + "/bdd100k")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    out = Path(a.out); (out/"images").mkdir(parents=True, exist_ok=True)
    recs = json.load(open(hf_hub_download(a.repo, "samples.json", repo_type="dataset")))["samples"]
    def lab(r, k):
        v = r.get(k)
        if isinstance(v, dict): v = v.get("label")
        return v
    keep = [r for r in recs if lab(r,"weather") not in (None, "undefined")
            and lab(r,"timeofday") not in (None, "undefined")]
    rng = random.Random(a.seed)
    # oversample adverse: rainy / snowy / foggy, and night
    by = defaultdict(list)
    for r in keep:
        adverse = (lab(r,"weather") in ("rainy","snowy","foggy")) or (lab(r,"timeofday") == "night")
        by["adverse" if adverse else "benign"].append(r)
    for v in by.values(): rng.shuffle(v)
    n_adv = min(len(by["adverse"]), int(a.n*0.6))
    sel = by["adverse"][:n_adv] + by["benign"][:a.n - n_adv]
    rng.shuffle(sel)
    print(f"  pool {len(keep)} labelled; selected {len(sel)} "
          f"({n_adv} adverse / {len(sel)-n_adv} benign)")
    meta = {}
    for i, r in enumerate(sel):
        fn = Path(r["filepath"]).name
        dst = out/"images"/fn
        if not dst.exists():
            p = hf_hub_download(a.repo, r["filepath"], repo_type="dataset")
            dst.write_bytes(Path(p).read_bytes())
        meta[fn] = {"weather": lab(r,"weather"), "timeofday": lab(r,"timeofday"),
                    "scene": lab(r,"scene")}
        if (i+1) % 200 == 0: print(f"  [fetch] {i+1}/{len(sel)}", flush=True)
    (out/"labels.json").write_text(json.dumps(meta, indent=1))
    print(f"  wrote {len(meta)} images + labels.json -> {out}")
    print("  weather :", dict(Counter(v["weather"] for v in meta.values()).most_common()))
    print("  timeofday:", dict(Counter(v["timeofday"] for v in meta.values()).most_common()))

if __name__ == "__main__":
    main()
