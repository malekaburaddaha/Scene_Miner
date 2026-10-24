#!/bin/bash
# Milestones 27-28: re-run every evaluation with the final model (v2.6) on 4 GPUs,
# one job per GPU (GPUs are Exclusive_Process).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}" SM_FINAL=v26
PY=/nfs/turbo/umd-srawa/conda_envs/envs/scene_mining/bin/python
L=results/m28_logs; mkdir -p $L
lane0 () { CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_v23.py --model v26 --ckpts "work_dirs/sceneminer_v26/s*/final.pt" --bench all --out results/cvpr/milestone27_v26.json > $L/e_v23.log 2>&1
           CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_semantic_edge_cases.py --final v26 --out results/cvpr/milestone27_semantic_edge_cases.json > $L/e_sem.log 2>&1; }
lane1 () { CUDA_VISIBLE_DEVICES=1 $PY scripts/holdout_signals.py --part merge > $L/e_merge.log 2>&1
           CUDA_VISIBLE_DEVICES=1 FP64=0 $PY scripts/verify_fast_index.py > $L/e_fast0.log 2>&1
           CUDA_VISIBLE_DEVICES=1 FP64=1 $PY scripts/verify_fast_index.py > $L/e_fast1.log 2>&1; }
lane2 () { CUDA_VISIBLE_DEVICES=2 $PY scripts/refav_llm_compile.py > $L/e_refav.log 2>&1
           CUDA_VISIBLE_DEVICES=2 SM_OUT=results/cvpr/milestone27_compositional_synergy.json $PY scripts/eval_compositional_synergy.py > $L/e_syn.log 2>&1; }
lane3 () { CUDA_VISIBLE_DEVICES=3 $PY scripts/val150_retrieval_to_3d.py --model v26 --out results/cvpr/milestone27_val150_retrieval_v26.json > $L/e_val150.log 2>&1
           CUDA_VISIBLE_DEVICES=3 SM_OUT=results/cvpr/milestone27_rater_sensitivity.json $PY scripts/eval_rater_sensitivity.py > $L/e_rater.log 2>&1; }
lane0 & lane1 & lane2 & lane3 & wait
# the comparison needs the holdout signals, the LLM programs and the v2.6 calibration file
CUDA_VISIBLE_DEVICES=0 SM_OUT=results/cvpr/milestone28_related_work_comparison.json SM_CAL=results/cvpr/milestone27_v26.json \
    $PY scripts/eval_related_work_baselines.py > $L/e_rw.log 2>&1
echo "ALL_DONE"
