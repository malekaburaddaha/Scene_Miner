#!/usr/bin/env python3
"""Stage 1 - merge the sharded VLM margin outputs into one per-scene file.

  python scripts/stage1_merge_shards.py work_dirs/unified_filter/logits_shard{0,1,2,3}.json \
      --out work_dirs/unified_filter/binary_flags_with_logits.json
  python scripts/stage1_merge_shards.py work_dirs/unified_filter/lowcontrast_strict_shard{0,1,2,3}.json \
      --out work_dirs/unified_filter/lowcontrast_strict.json
--check compares the union with an existing --out instead of writing it.
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("shards", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    merged = {}
    for s in a.shards:
        d = json.load(open(s)); assert not set(d) & set(merged), f"{s} overlaps an earlier shard"; merged.update(d)
    if a.check:
        ref = json.load(open(a.out)); same = set(ref) == set(merged) and all(ref[k] == merged[k] for k in ref)
        print(f"[merge] {len(merged)} scenes from {len(a.shards)} shards; identical to {a.out}: {same}")
        raise SystemExit(0 if same else 1)
    Path(a.out).write_text(json.dumps(merged, indent=1)); print(f"[merge] wrote {a.out} ({len(merged)} scenes)")


if __name__ == "__main__":
    main()
