#!/bin/bash
# Train 5 seeds on 4 GPUs. GPUs are Exclusive_Process: one job per GPU, so
# slot 0 runs seeds 0 and 4 back to back.
#   usage: scripts/launch_seeds.sh <trainer.py> <work_root> <log_prefix> [extra args]
set -u
cd "$(dirname "$0")/.."
TR=$1; ROOT=$2; LOG=$3; shift 3
EXTRA=("$@")
export PYTHONPATH="src:${PYTHONPATH:-}"
PY=${PY:-python}
mkdir -p "$ROOT" results/day1_gate/logs
run () { CUDA_VISIBLE_DEVICES=$1 $PY "$TR" --work-dir "$ROOT/s$2" --seed $2 --steps 3000 "${EXTRA[@]}" \
           > "results/day1_gate/logs/${LOG}_s$2.log" 2>&1; }
( run 0 0; run 0 4 ) &
run 1 1 &
run 2 2 &
run 3 3 &
wait
echo "ALL_DONE $LOG"
