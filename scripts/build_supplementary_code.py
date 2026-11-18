#!/usr/bin/env python3
"""MILESTONE 36 - the anonymised supplementary code bundle, paper/supplementary_code.zip.

Run from the repository root. Packs the tracked src/, scripts/, configs/, pipeline/,
benchmark_cvpr/, results/cvpr/*.json and the environment files, with the repository README as
ANONYMOUS_README.md (Milestone 41; the README constant below is the Milestone 36 text, kept for
reference). Excluded: scripts/check_anonymity.py
(its token list names the affiliation it searches for) and this builder. Member
timestamps are fixed, so the archive depends only on the files. Then
`python scripts/check_anonymity.py --zip paper/supplementary_code.zip` scans every member.
"""
import subprocess, zipfile
from pathlib import Path

EXCLUDE = {"scripts/check_anonymity.py", "scripts/build_supplementary_code.py"}
OUT = Path("paper/supplementary_code.zip")
README = """# SceneMiner: anonymous supplementary code

Code, benchmark definitions, anonymised human labels and every result file behind the
tables of the submission.

## Contents

| path | what |
|---|---|
| `src/scene_mining/filter/` | SceneMiner (`sceneminer_v2_6.py`, the final model) and its predecessors; the cached index (`fast_index.py`) |
| `scripts/` | training, evaluation, baselines, table and figure generators |
| `benchmark_cvpr/` | splits, the 390-query suite, the 40 Stratum-5 queries, the pre-registered 36-query BDD100K holdout, the 160-scene second-rater subset |
| `benchmark_cvpr/human_rater1_600.json` | rater 1, 600 scenes (550 after unsure/skip), four degradation attributes |
| `benchmark_cvpr/human_rater2_160.json` | rater 2, 160 scenes |
| `results/cvpr/*.json` | every number in the paper; `scripts/verify_paper_numbers.py` maps each claim to a JSON path |

## Environment

Python 3.11, PyTorch 2, `transformers`, `pypdf`, `numpy`, `matplotlib`.
Machine-specific locations are read from environment variables (`scripts/paths.py`):

| variable | default |
|---|---|
| `SCENEMINER_DATA_ROOT` | `data` (holds `external_adverse/bdd100k`, `external_adverse/bdd100k_holdout`) |
| `QWEN_VL` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `BLIP2` | `Salesforce/blip2-opt-2.7b` |
| `SIGLIP2_SO400M` | `google/siglip2-so400m-patch14-384` |
| `PY` (shell scripts) | `python` |

Datasets are not redistributed: nuScenes, Waymo Open Dataset and BDD100K must be
obtained under their own licences. Cached Stage-1 features and checkpoints are not
included for size.

## Reproducing

```
export PYTHONPATH=src
python scripts/train_sceneminer_v26.py --work-dir work_dirs/sceneminer_v26/s0 --seed 0   # x5 seeds
SM_FINAL=v26 python scripts/eval_related_work_baselines.py      # Table 2
SM_FINAL=v26 python scripts/incorporate_rater2.py               # second rater
python scripts/eval_per_query_vlm_reranker.py --part eval       # per-query VLM reranker
python scripts/make_main_tables.py && python scripts/make_tab_master.py   # tables from results/cvpr
```

The table generators read only `results/cvpr/*.json`, so all tables can be rebuilt
without data or GPUs.
"""


def main():
    files = subprocess.run(["git", "ls-files", "src", "scripts", "benchmark_cvpr", "results/cvpr", "configs", "pipeline",
                            "requirements.txt", "environment.yml", "environment_bevformer.yml"],
                           capture_output=True, text=True, check=True).stdout.split()
    keep = [f for f in files if f not in EXCLUDE and (not f.startswith("results/cvpr/") or f.endswith(".json"))
            and "/logs/" not in f]
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in ["ANONYMOUS_README.md"] + keep:
            zi = zipfile.ZipInfo(f"sceneminer/{f}", date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
            z.writestr(zi, Path("README.md").read_text() if f == "ANONYMOUS_README.md" else Path(f).read_bytes())
    print(f"{OUT}: {len(keep) + 1} files, {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
