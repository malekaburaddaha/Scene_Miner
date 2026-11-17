# SceneMiner

Code for **SceneMiner: Language-Guided Retrieval of Degraded Driving Scenes That Break
Perception** (anonymous submission).

SceneMiner is a two-stage, text-queryable retrieval system over driving logs:

* **Stage 1 (build time, once per corpus)** caches per-scene signals: a SigLIP2 image
  embedding, Qwen2.5-VL yes/no log-prob margins for four degradation attributes
  (droplets, glare, wet road, low contrast), and optionally a BEV embedding, tags and a
  risk scalar.
* **Stage 2 (per query)** is a small scale-aware fusion head with modality adapters,
  FiLM conditioning, a cross-attention decoder and a scorer. A factored concept gate and
  safe abstention decide how far it may depart from the zero-shot SigLIP2 score.

The repository also contains the de-contaminated evaluation protocol: frozen splits,
the 390-query suite, the human labels from two raters, and every result file behind the
tables in the paper.

## Layout

| path | contents |
|---|---|
| `src/scene_mining/filter/sceneminer_v2_6.py` | **the final model** (the paper's SceneMiner) |
| `src/scene_mining/filter/fast_index.py` | pre-projected cached index used for the latency numbers |
| `src/scene_mining/filter/unified_filter.py` | modality adapters, FiLM, scalar standardisation |
| `src/scene_mining/filter/sceneminer_v2*.py` | earlier versions used in the design-progression ablation |
| `src/scene_mining/` (other) | backbones, VLM wrappers, dataset adapters, diagnostic hooks, losses |
| `scripts/` | Stage-1 caching, training, evaluation, baselines, table and figure generators, annotation UI |
| `benchmark_cvpr/` | 906/602 split, 390-query suite, semantic edge-case queries, 36-query BDD100K holdout, human labels (rater 1: 600 scenes, rater 2: 160 scenes) |
| `results/cvpr/*.json` | the metric files every table in the paper is generated from |

## Installation

```bash
pip install -r requirements.txt
export PYTHONPATH=src:$PYTHONPATH
```

The 3D detection study (BEVFormer) needs a separate, older stack (PyTorch 1.10,
mmdet3d 0.17.1) and the official BEVFormer code base; it is only used for the downstream
perception experiments.

Model and data locations are read from environment variables (`scripts/paths.py`):

| variable | default |
|---|---|
| `SCENEMINER_DATA_ROOT` | `data` (holds `external_adverse/bdd100k`, `external_adverse/bdd100k_holdout`) |
| `QWEN_VL` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `BLIP2` | `Salesforce/blip2-opt-2.7b` |
| `SIGLIP2_SO400M` | `google/siglip2-so400m-patch14-384` |
| `PY` (shell scripts) | `python` |

## Data

Datasets are not redistributed. nuScenes, the Waymo Open Dataset and BDD100K must be
obtained under their own licences (`scripts/fetch_bdd100k.py` selects the 1,000-image
stratified BDD100K subset). Cached Stage-1 features and checkpoints are not included
because of their size; they are written to `work_dirs/` by the scripts below.
The base scene store `work_dirs/unified_filter/features_overnight.npz` (BEV embedding,
tags, risk scalar and scene thumbnails) is read by the scripts below but is produced by the
Stage-1 BEVFormer/Q-Former pipeline, which is not part of this release.

## Rebuilding the tables without data or GPUs

The table generators read only `results/cvpr/*.json`:

```bash
mkdir -p paper/supp_tables paper/figures
python scripts/make_main_tables.py
python scripts/make_tab_master.py
python scripts/make_tab_bddq.py
python scripts/make_supp_tables.py
```

## Full pipeline

```bash
# Stage 1: frozen split, query suite and cached scene signals
python scripts/build_split_906_602.py
python scripts/build_query_suite_390.py
python scripts/extract_vlm_logits.py --shard 0 --nshards 1 --out work_dirs/unified_filter/logits_shard0.json
python scripts/extract_lowcontrast_strict.py --shard 0 --nshards 1 --out work_dirs/unified_filter/lc_shard0.json
python scripts/cache_v2_signals.py

# Stage 2: train the final model (5 seeds)
for s in 0 1 2 3 4; do
  python scripts/train_sceneminer_v26.py --work-dir work_dirs/sceneminer_v26/s$s --seed $s --steps 3000
done

# Evaluation: human labels, BDD100K, holdout, query suite, val150 3D study, baselines
SM_FINAL=v26 scripts/run_m27_evals.sh
SM_FINAL=v26 python scripts/incorporate_rater2.py
FP64=1 python scripts/verify_fast_index.py && python scripts/time_sceneminer_latency.py

# Collapse diagnostic protocol
python scripts/run_diagnostic_protocol.py --help
python scripts/label_entropy_null.py
```

`scripts/run_m27_evals.sh` runs four lanes in parallel, one GPU each; edit the
`CUDA_VISIBLE_DEVICES` values for your machine.

## Human annotation

`scripts/annotate_ui.py --rater <name> --port 8800` serves a dependency-free local
labelling UI (keys `1`–`4` toggle the attributes, `Enter` saves and advances, `u` marks
unsure, `s` skips). It autosaves after every scene and can be resumed.
