#!/bin/bash
# Milestone 34: score every trained row of Tables 1 and 2 one seed at a time, with the
# same scripts that produced the tables, so +/- sd sits next to the reported value.
#   usage: scripts/run_seed_sd.sh <seed> <gpu>      (then: python scripts/summarize_seed_sd.py)
set -u
cd "$(dirname "$0")/.."
S=$1; export CUDA_VISIBLE_DEVICES=$2 PYTHONPATH="src:${PYTHONPATH:-}"
PY=${PY:-python}
O=work_dirs/seed_sd; mkdir -p $O
# Table 2 trained rows (BEV-TSR-style, adapter, naive router = Table 1 row 1, SceneMiner and its portable mode = row 6)
SM_FINAL=v26 SM_SEED=$S SM_METHODS=bev_tsr,siglip2_ft,naive,sceneminer_portable,sceneminer SM_OUT=$O/t2_s$S.json \
  SM_CAL=/nonexistent $PY scripts/eval_related_work_baselines.py
# Table 1 rows 3-6 (v26 also gives the calibrated human GT and the ablations per seed)
for v in 23 24 25 26; do
  $PY scripts/eval_v23.py --model v$v --ckpts "work_dirs/sceneminer_v$v/s$S/final.pt" --out $O/v${v}_s$S.json
done
# Table 1 row 1 (v2.1): calibrated human GT
SM_SEED_GLOB=s$S $PY scripts/eval_v21_human_gt.py --out $O/v21hum_s$S.json
# Table 1 row 2 (v2.2): human GT, BDD100K with raw query features, suite, top-5 skew
SM_SEED_GLOB=s$S $PY scripts/eval_v22_human_gt.py --out $O/v22hum_s$S.json
SM_SEED_GLOB=s$S SM_OUT=$O/v22bdd_s$S.json $PY scripts/bdd_routing_rawquery_check.py
SM_SEED_GLOB=s$S $PY scripts/eval_v22_query_suite.py --out $O/v22suite_s$S.json
SM_SEED_GLOB=s$S $PY scripts/eval_domain_calibrated_v2.py --out $O/v22skew_s$S.json
echo "SEED_DONE $S"
