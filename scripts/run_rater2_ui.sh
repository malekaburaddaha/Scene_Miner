#!/bin/bash
# Second-rater annotation, 160 scenes (Milestone 30).
#   usage: scripts/run_rater2_ui.sh <rater_name> [port]     (default port 8802)
# Open the printed URL, label each scene with keys 1-4 (droplets, glare, wet road,
# low contrast), Enter saves and advances, b goes back, u marks unsure. Progress is
# saved after every scene and resumes on restart. Then run:
#   python scripts/incorporate_rater2.py --rater <rater_name>
set -eu
cd "$(dirname "$0")/.."
NAME=${1:?usage: run_rater2_ui.sh <rater_name> [port]}; PORT=${2:-8802}
PY=${PY:-/nfs/turbo/umd-srawa/conda_envs/envs/scene_mining/bin/python}
exec $PY scripts/annotate_ui.py --rater "$NAME" --port "$PORT" \
     --sample benchmark_cvpr/rater2_subset_160.json --outdir results/rater2
