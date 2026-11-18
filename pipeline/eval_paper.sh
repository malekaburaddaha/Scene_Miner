#!/usr/bin/env bash
# Every table, diagnostic and analysis of the paper, from the Stage-1 cache and the Stage-2
# checkpoints. Run from the repository root. All steps also run on CPU with SM_DEVICE=cpu
# (every number of Milestone 41 was produced that way); the per-query VLM reranker and the
# detector inference need a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}" SM_FINAL=v26
PY=${PY:-python}
R=results/cvpr

# --- 1. diagnostics (Sec. 3) --------------------------------------------------------------
$PY scripts/eval_attribute_index.py                      # attribute-derived relevance: the lookup ceiling (1.000 mAP)
$PY scripts/eval_v22_query_suite.py                      # 150-query suite with the index ceiling (Fig. 2a)
# train-test overlap of the upstream detector: per-scene official nuScenes mAP of the public
# bevformer_base checkpoint on corpus (train-split) scenes vs val150 (Fig. 2b). Predictions come
# from the BEVFormer repository's test script (bevformer env, environment_bevformer.yml).
#   corpus: predictions on the 278 nuScenes evaluation-split corpus scenes (official train split)
$PY scripts/per_scene_official_metrics.py --results-json work_dirs/bevformer_out/ours278_fmt/pts_bbox/results_nusc.json \
    --eval-set train --out $R/bevformer_ours278_per_scene_official.json
#   val150: predictions on the official val split
$PY scripts/per_scene_official_metrics.py --results-json work_dirs/bevformer_out/official_eval/pts_bbox/results_nusc.json \
    --eval-set val --out $R/bevformer_val_per_scene_official.json
$PY scripts/diag_bdd_blend.py                            # scale mismatch: the convex blend below both inputs
$PY scripts/diag_v22_decomposition.py                    # residual dominance (base 0.961 vs 0.881 with residual)

# --- 2. main evaluation of the final model -------------------------------------------------
$PY scripts/eval_v23.py --model v26 --ckpts "work_dirs/sceneminer_v26/s*/final.pt" --bench all --out $R/milestone27_v26.json
$PY scripts/eval_semantic_edge_cases.py --final v26 --out $R/milestone27_semantic_edge_cases.json
SM_OUT=$R/milestone27_compositional_synergy.json $PY scripts/eval_compositional_synergy.py
SM_OUT=$R/milestone27_rater_sensitivity.json $PY scripts/eval_rater_sensitivity.py

# --- 3. Table 2: every method on identical splits, queries and metrics ---------------------
SM_OUT=$R/milestone28_related_work_comparison.json SM_CAL=$R/milestone27_v26.json $PY scripts/eval_related_work_baselines.py
SM_METHODS=siglip2,lookup,late_fusion,sceneminer_portable,sceneminer SM_OUT=$R/milestone41_late_fusion.json \
    $PY scripts/eval_related_work_baselines.py          # the late-fusion row (Milestone 41)
$PY scripts/eval_per_query_vlm_reranker.py --part eval   # SigLIP2 top 50 + Qwen2.5-VL per query (GPU)
$PY scripts/time_sceneminer_latency.py                   # ms per query (A40)

# --- 4. Tables 1, 3, 4 ----------------------------------------------------------------------
$PY scripts/gate_examples.py                             # Table 1: concept-gate decisions
$PY scripts/incorporate_rater2.py                        # Table 3: second annotator
$PY scripts/table4_conjunctions.py                       # Table 4: conjunction column, both query sets

# --- 5. Table 5 and the analyses E1, E2, E5 ------------------------------------------------
for s in 0 1 2 3 4; do bash scripts/run_seed_sd.sh $s 0; done && $PY scripts/summarize_seed_sd.py   # five-seed sds
$PY scripts/e1_paired_bootstrap.py                       # E1: paired scene-bootstrap CIs, both annotators
for v in v26tf v26tfu v26ps; do                          # E2 control and its check, E5 post-standardised mixture
  SM_FINAL=$v SM_CAL=/nonexistent SM_METHODS=sceneminer_portable,sceneminer SM_OUT=$R/milestone41_$([ $v = v26ps ] && echo e5 || echo e2)_$v.json \
      $PY scripts/eval_related_work_baselines.py
done
SM_CONTROL=v26tf SM_OUT=$R/milestone41_e2_bootstrap.json $PY scripts/e1_paired_bootstrap.py

# --- 6. Sec. 7: detector error on retrieved scenes (val150) --------------------------------
$PY scripts/val150_camfront_errors.py                    # front-sector metrics; caches the val150 CAM_FRONT margins
$PY scripts/val150_ped_vru_metrics.py                    # adds VRU recall
$PY scripts/val150_retrieval_to_3d.py --model v26 --out $R/milestone27_val150_retrieval_v26.json   # Table 6
$PY scripts/e4_val150_degradation_flags.py               # E4: the 42/108 flag rule and the BH table
$PY scripts/e3_val150_stratified.py                      # E3: night/rain-controlled nulls and paired tests

# --- 7. tables and figures from the result files (no data or GPU needed) -------------------
$PY scripts/make_main_tables.py && $PY scripts/make_tab_master.py && $PY scripts/make_supp_tables.py
$PY scripts/make_figures.py && $PY scripts/make_m41_tables.py
echo "evaluation done"
