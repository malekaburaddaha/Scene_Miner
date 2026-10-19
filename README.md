# SceneMiner

A text-queryable index over driving logs for the sensor-degradation and
macro-environmental conditions that degrade perception — water on the lens,
headlight bloom on wet asphalt, loss of contrast at night — together with the
collapse diagnostic that validates the fusion behind it.

All scene signals are cached at build time, so a query costs one text encoding
plus a single pass of a small head over the cached corpus.

## Installation

```bash
# retrieval / diagnostics
conda activate /nfs/turbo/umd-srawa/conda_envs/envs/scene_mining   # torch 2.4.1, transformers 5.2
# 3D detector evaluation (BEVFormer needs the older mm-stack)
conda activate /nfs/turbo/umd-srawa/conda_envs/envs/bevformer      # torch 1.10, mmdet3d 0.17.1
export PYTHONPATH=src:$PYTHONPATH
```

GPUs on this cluster run in `Exclusive_Process` mode: **one process per GPU**.
Use `scripts/orchestrate_overnight.py`, which holds exactly one job per GPU and
retries once.

## Architecture

A frozen stack produces four cached signals per scene — a 256-d BEVFormer-style
retrieval embedding `e`, 20 tag logits `t`, a scalar risk `r`, and a Qwen2.5-VL
visual token `v`. **v2** adds a SigLIP2 image embedding and four continuous
VLM Yes/No log-prob margins. A query is embedded by a frozen text tower; each
signal passes a private adapter into a shared 256-d token space, a
zero-initialised FiLM layer conditions the tokens on the query, and the query
token cross-attends them.

v2 uses identity-preserving growth: the score is a zero-shot base plus a
residual behind a zero-initialised gate, so at step 0 the ranking **is** the
base ranking (verified exactly) and every departure is something training paid for.

Scalar channels are standardised with fixed dataset statistics, never
`LayerNorm` — a `LayerNorm` over a width-1 axis returns 0 for every input and
deletes the channel. `--force-layernorm1d` restores that defect on purpose to
regenerate the pre-repair baseline.

## Collapse diagnostic protocol

`scripts/run_diagnostic_protocol.py` reports, for any fused representation:
spectral effective rank read against a **label-entropy null**, true
cross-attention routing mass per modality, per-token distinct-row counts
(which catch a dead channel), score span, and per-attribute probe gradients.

Note on the null: a multi-hot matrix over *k* atomic binary attributes has
rank ≤ *k*, so `erank(L) ≤ k` by arithmetic. Both the atomic and the
query-predicate constructions are reported.

## Reproducing every table

```bash
# frozen split and cached signals
python scripts/build_split_906_602.py
python scripts/extract_vlm_logits.py --shard i --nshards 4 --out .../logits_shard$i.json
python scripts/extract_lowcontrast_strict.py --shard i --nshards 4 --out .../lc_shard$i.json
python scripts/cache_v2_signals.py

# Table 2 (de-confounded ablation, 5 seeds) and the fair leave-one-attribute-out
python scripts/orchestrate_overnight.py
python scripts/eval_cvpr_suites.py --mode ablation --out results/cvpr/table2.json
python scripts/eval_cvpr_suites.py --mode loao     --out results/cvpr/loao.json

# cached-attribute baseline (the tautology check) and the diagnostic protocol
python scripts/eval_attribute_index.py --output results/cvpr/attribute_index.json
python scripts/run_diagnostic_protocol.py --checkpoint <ckpt> --out results/cvpr/diag.json
python scripts/blip2_diagnostic_transfer.py
python scripts/label_entropy_null.py

# SceneMiner v2 and the human-ground-truth evaluation
python scripts/train_sceneminer_v2.py --work-dir work_dirs/sceneminer_v2/s0 --seed 0
python scripts/eval_v2_human_gt.py
python scripts/eval_human_gt_full.py

# 3D gate (bevformer env)
python tools/test.py <cfg> ckpts/bevformer_base.pth --launcher pytorch --out val_results.pkl
python scripts/per_scene_official_metrics.py --results-json <results_nusc.json> --out <per_scene.json>
python scripts/run_3d_gate.py --per-scene <per_scene.json> --out results/cvpr/gate.json

# human annotation (two raters, same frozen 600-scene sample)
python scripts/annotate_ui.py --rater <name> --port 8800
```

## Results

`results/cvpr/MASTER_REPORT.md` summarises every table. Headline: against 600
human-labelled scenes SceneMiner v2 reaches **0.7587** mean mAP, ahead of the
continuous VLM lookup at 0.7325 and v1 at 0.4202, with zero train/eval leakage.

## Layout

```
src/        model, adapters, losses, diagnostic hooks
scripts/    training, evaluation, baselines, diagnostics, annotation UI
benchmark_cvpr/  frozen split, query suite, human annotations
results/cvpr/    result tables and milestone reports
paper/      LaTeX source
```

Checkpoints, feature stores and datasets are not tracked; see `.gitignore`.
