#!/usr/bin/env bash
# Stage 2: SceneMiner (v2.6) and every trained comparison, five seeds each, from the Stage-1
# cache (work_dirs/unified_filter/features_v2.npz + V_frozen_cache.pt). Under a minute per seed
# on one A40; SM_DEVICE=cpu also works (slower). Run from the repository root.
#
# SceneMiner v2.6 = src/scene_mining/filter/sceneminer_v2_6.py: PartNorm / ResNorm, two-layer MLP
# token adapters, FiLM, two-layer four-head cross-attention decoder, factored concept gate built
# from the 252 training phrasings with conditional half-strength context removal; Eq. (4) loss
# (weighted listwise InfoNCE + conditional hinge + routing cap) in scripts/train_sceneminer_v26.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
PY=${PY:-python}
seeds () {   # seeds () <trainer> <work_root> [extra args]: five seeds, one after another
  local tr=$1 root=$2; shift 2
  for s in 0 1 2 3 4; do $PY "$tr" --work-dir "$root/s$s" --seed $s --steps 3000 "$@"; done
}
# (on a 4-GPU node, scripts/launch_seeds.sh <trainer> <work_root> <log_prefix> runs the five seeds in parallel)

# final model
seeds scripts/train_sceneminer_v26.py work_dirs/sceneminer_v26
# design progression (Table 5): naive convex router, + PartNorm, + ResNorm, + contrastive gate, + factored gate
seeds scripts/train_sceneminer_v21.py work_dirs/sceneminer_v21
for v in 22 23 24 25; do seeds scripts/train_sceneminer_v$v.py work_dirs/sceneminer_v$v; done
# trained baselines (Table 2): BEV-TSR-style projection and the SigLIP2 adapter
for s in 0 1 2 3 4; do
  $PY scripts/train_projection_baselines.py --kind bev --seed $s
  $PY scripts/train_projection_baselines.py --kind sigft --seed $s
done
# controls built without training (Milestone 41)
$PY scripts/e2_build_training_free.py --prior bank       # E2: training-free control  -> work_dirs/sceneminer_v26tf
$PY scripts/e2_build_training_free.py --prior uniform    # E2 check (query-independent prior) -> work_dirs/sceneminer_v26tfu
$PY scripts/e5_scale.py                                  # E5: scale of s_head / s_abst; writes work_dirs/sceneminer_v26ps
echo "Stage 2 done"
