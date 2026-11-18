# SceneMiner: Language-Guided Retrieval of Degraded Driving Scenes

Code, benchmark definitions and anonymised human labels for the CVPR 2027 submission.
The repository covers the whole pipeline: Stage 1 (frozen per-scene signals, computed once
and cached), Stage 2 (the 4.68M-parameter fusion model), and every table and analysis.

```
query ─► SigLIP2 text ─┐
                       ▼
Stage 1 (once per scene, cached)                    Stage 2 (3.5 ms per query over 1508 scenes)
  multi-camera: BEVFormer 50x50 + BLIP-2 Q-Former    PartNorm base (cosine, VLM margin, prior), routed
    + Qwen2.5-VL-3B LoRA r=16  -> e, t, r            + ResNorm-bounded cross-attention correction
  single-camera: SigLIP2 s, Qwen2.5-VL margins m,    concept gate (252 training phrasings) mixes the
    VLM token v; recorded metadata -> prior pi        head with the SigLIP2 + metadata fallback
```

## Repository layout

| path | contents |
|---|---|
| `pipeline/` | the four entry points: `stage1_multicam.sh`, `stage1_single_camera.sh`, `stage2_train.sh`, `eval_paper.sh` |
| `configs/experiment/` | Stage-1 multi-camera model configs (v2 → v4 motion mask → v8 unified) and their bases |
| `src/scene_mining/` | Stage-1 model (`models/`, `spatial/bevformer/`, `adapters/qformer.py`, `vlm/qwen25vl.py`, `heads/`), datasets incl. Waymo 5→6 camera padding (`data/universal_dataset.py`), Stage-2 models (`filter/`) |
| `src/scene_mining/filter/` | `sceneminer_v2_6.py` (final model) and its predecessors v2.1–v2.5 (Table 5), `fast_index.py` (cached index), `sceneminer_tf.py` (E2 training-free control), `sceneminer_poststd.py` (E5) |
| `scripts/` | every Stage-1 extraction, training, evaluation, baseline, table and figure script |
| `benchmark_cvpr/` | splits, queries, human labels and external benchmark definitions (below) |
| `results/cvpr/` | every result file behind the paper; `scripts/verify_paper_numbers.py` maps each claim to one |
| `docs/development_log.md` | the milestone-by-milestone development record |

## Setup

```bash
conda env create -f environment.yml && conda activate scene_mining   # Python 3.11, torch 2.4.1, transformers 5.2
conda env create -f environment_bevformer.yml                        # only for BEVFormer detector inference
export PYTHONPATH=src
```
Machine-specific locations are environment variables (`scripts/paths.py`):

| variable | default | meaning |
|---|---|---|
| `SM_DEVICE` | `cuda` if available, else `cpu` | device of every script |
| `SCENEMINER_DATA_ROOT` | `data` | holds `external_adverse/bdd100k{,_holdout}` |
| `QWEN_VL` | `Qwen/Qwen2.5-VL-3B-Instruct` | hub id or local path |
| `BLIP2` | `Salesforce/blip2-opt-2.7b` | hub id or local path |
| `SIGLIP2_SO400M` | `google/siglip2-so400m-patch14-384` | backbone study |

Data (not redistributed; obtain under each licence): nuScenes v1.0-trainval in `data/nuscenes`
(+ nuPrompt for the Stage-1 tag targets), Waymo Open Dataset v1.4.3 perception, and BDD100K
(fetched by `scripts/fetch_bdd100k.py` / `build_bdd_holdout.py` from the public mirror).
Feature caches and checkpoints are not tracked (`.gitignore`); the pipeline regenerates them
under `work_dirs/`.

## Stage 1: per-scene signals (`pipeline/stage1_multicam.sh`, `pipeline/stage1_single_camera.sh`)

**Multi-camera path** (nuScenes, Waymo; 4 GPUs). Four phases, each initialised from the previous:
`configs/experiment/nuscenes_scenemining_v2.py` (BEVFormer encoder over frozen SigLIP2 patches,
BLIP-2 Q-Former with 32 queries, Qwen2.5-VL-3B with LoRA r=16, embedding/tag/risk heads) →
`..._v4_motion_mask.py` → `..._v8_unified.py` → `scripts/train_wacv.py` (interleaved nuScenes + Waymo
stream, LoRA trainable, motion frozen). Waymo's five cameras are padded to six slots with a
validity mask. `scripts/export_filter_features_{scaled,overnight}.py` export the 256-d embedding
e, 20 tag logits t and risk r for the 1508 scenes (700 nuScenes, 808 Waymo 4-frame chunks; frame
list in `benchmark_cvpr/waymo_frames.json`); `scripts/build_deterministic_gt.py` the recorded
metadata. Stage-1 targets: `extract_nuprompt_tags.py`, `enrich_tags_from_nuscenes.py`,
`compute_gt_risk_scores.py`.

**Single-camera path** (any front camera; also BDD100K and val150). SigLIP2 ViT-B/16 image
embedding s (`cache_v2_signals.py`); four zero-shot Qwen2.5-VL-3B yes/no log-odds m (wet road,
glare, droplets, low contrast; `extract_vlm_logits.py`, `extract_lowcontrast_strict.py`, merged by
`stage1_merge_shards.py`); 2048-d VLM token v (`stage1_vlm_token.py`); 26-d prior π = z-scored tag
logits + six metadata indicators (`stage1_build_prior.py`).

**Checking a cache.** Each single-camera signal can be recomputed and compared:
`stage1_build_prior.py --check` (exact), `stage1_merge_shards.py ... --check` (exact),
`stage1_check_cache.py -n 4` (SigLIP2: cosine 1.00000; margins: same sign, within bf16 precision)
and `stage1_vlm_token.py --check 4` (cosine ≥ 0.9997 on CPU against the GPU cache).

## Stage 2: SceneMiner (`pipeline/stage2_train.sh`)

`scripts/train_sceneminer_v26.py`, five seeds × 3000 AdamW steps on the 906 training scenes with
VLM pseudo-labels (no human label is used for training). The model is
`src/scene_mining/filter/sceneminer_v2_6.py`; the design steps of Table 5 are `train_sceneminer_v2{1..5}.py`.

## Evaluation (`pipeline/eval_paper.sh`)

| paper | command (from the repository root) | result file |
|---|---|---|
| Fig. 2a, Sec. 3.1 lookup ceiling | `eval_attribute_index.py`, `eval_v22_query_suite.py` | `milestone10_*.json` |
| Fig. 2b, Sec. 3.2 detector overlap | `per_scene_official_metrics.py` (train vs val) | `bevformer_ours278_per_scene_official.json`, `bevformer_val_per_scene_official.json` |
| Sec. 3.3 scale mismatch | `diag_bdd_blend.py`, `diag_v22_decomposition.py` | logs, `milestone14_*.json` |
| Table 1 gate decisions | `gate_examples.py` | `milestone39_gate_examples.json` |
| Table 2 main comparison | `eval_related_work_baselines.py`; late fusion: `SM_METHODS=siglip2,lookup,late_fusion,sceneminer_portable,sceneminer`; reranker: `eval_per_query_vlm_reranker.py --part eval` | `milestone28_related_work_comparison.json`, `milestone41_late_fusion.json`, `milestone34_per_query_vlm.json` |
| Table 3 second annotator | `incorporate_rater2.py` | `milestone33_rater2_results.json` |
| Table 4 query types | `eval_rater_sensitivity.py`, `eval_semantic_edge_cases.py --final v26`, `eval_compositional_synergy.py`, `table4_conjunctions.py` | `milestone27_*.json`, `milestone41_table4_check.json` |
| Table 5 design progression | `eval_v23.py --model v2X`, `run_seed_sd.sh`, `summarize_seed_sd.py` | `milestone27_v26.json`, `milestone34_seed_sd.json` |
| Table 6, Sec. 7 detection error | `val150_camfront_errors.py`, `val150_ped_vru_metrics.py`, `val150_retrieval_to_3d.py --model v26` | `milestone11_*.json`, `milestone27_val150_retrieval_v26.json` |
| E1 paired bootstrap | `e1_paired_bootstrap.py` | `milestone41_e1_bootstrap.json` |
| E2 training-free control | `e2_build_training_free.py`, then Table 2 script with `SM_FINAL=v26tf`; `SM_CONTROL=v26tf e1_paired_bootstrap.py` | `milestone41_e2_*.json` |
| E3 night/rain-controlled detection study | `e3_val150_stratified.py` | `milestone41_e3_val150_stratified.json` |
| E4 val150 degradation flags | `e4_val150_degradation_flags.py` | `milestone41_e4_val150_flags.json` |
| E5 scale of s_head / s_abst | `e5_scale.py`, then Table 2 script with `SM_FINAL=v26ps` | `milestone41_e5_*.json` |
| all LaTeX tables / figures | `make_main_tables.py`, `make_tab_master.py`, `make_supp_tables.py`, `make_figures.py` | `paper/` |

Every number in the paper is checked against these files by `scripts/verify_paper_numbers.py`.
All Milestone 41 numbers were computed on CPU (`SM_DEVICE=cpu`).

## Benchmarks and annotations (`benchmark_cvpr/`)

| file | contents |
|---|---|
| `split_906_602.json` | driving-log-disjoint split of the 1508 corpus scenes (906 train / 602 evaluation) |
| `human_rater1_600.json` | annotator 1: four degradations on 600 evaluation scenes (550 after unsure/skip) |
| `human_rater2_160.json`, `rater2_subset_160.json` | annotator 2 on 160 of them, and how the subset was drawn |
| `queries_390.json` | the 390-query suite (240 training / 150 evaluation; attribute-derived relevance) |
| `queries_semantic_edge_cases.json` | the 40 object queries (20 atomic, 20 conjunctions) |
| `human_semantic_gt.json` | object flags from official nuScenes 3D annotation and the scene description |
| `bdd100k_1000_queries_12.json` | BDD100K-1000: 12 queries, their BDD100K label predicates, VLM channel, image list |
| `bdd_holdout_queries_36.json`, `bdd100k_holdout_2000_images.json` | the pre-registered holdout: 36 queries, predicates, protocol, image list |
| `waymo_frames.json` | Waymo (segment, frame) identifiers of the corpus and the Stage-1 training stream |

Human labels are anonymised (`"rater": "rater1" / "rater2"`) and never used for training.

## Licences

nuScenes, the Waymo Open Dataset and BDD100K are used under their non-commercial research
licences; no data, images or labels of these datasets are redistributed here, only identifiers,
query definitions and our own annotations.
