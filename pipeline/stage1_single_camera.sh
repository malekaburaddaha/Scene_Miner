#!/usr/bin/env bash
# Stage 1, single-camera ("portable") path: SigLIP2 ViT-B/16 image embedding s, four zero-shot
# Qwen2.5-VL-3B yes/no margins m, 2048-d VLM token v, and the 26-d prior pi; for the corpus
# (CAM_FRONT thumbnails written by the multi-camera export), BDD100K-1000, the BDD100K holdout
# and nuScenes val150. Run from the repository root after pipeline/stage1_multicam.sh.
# GPU recommended (about 1 s per image for the four margins on an A40); SM_DEVICE=cpu works.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
PY=${PY:-python}; NS=${NSHARD:-4}
U=work_dirs/unified_filter

# --- corpus (1508 scenes) ---------------------------------------------------------------
for k in $(seq 0 $((NS - 1))); do      # shards can run on separate GPUs
  $PY scripts/extract_vlm_logits.py --out $U/logits_shard$k.json --shard $k --nshards $NS
  $PY scripts/extract_lowcontrast_strict.py --out $U/lowcontrast_strict_shard$k.json --shard $k --nshards $NS
done
$PY scripts/stage1_merge_shards.py $(for k in $(seq 0 $((NS - 1))); do echo $U/logits_shard$k.json; done) --out $U/binary_flags_with_logits.json
$PY scripts/stage1_merge_shards.py $(for k in $(seq 0 $((NS - 1))); do echo $U/lowcontrast_strict_shard$k.json; done) --out $U/lowcontrast_strict.json
$PY scripts/cache_v2_signals.py           # SigLIP2 image embeddings + the four margins -> features_v2.npz
$PY scripts/stage1_build_prior.py         # 26-d prior pi (z-scored tag logits + six metadata indicators)
$PY scripts/stage1_vlm_token.py           # 2048-d VLM token v -> V_frozen_cache.pt
# checks against an existing cache (see README): exact prior, SigLIP2 and margins on a sample, VLM token
$PY scripts/stage1_build_prior.py --check && $PY scripts/stage1_check_cache.py -n 4 && $PY scripts/stage1_vlm_token.py --check 4

# --- BDD100K-1000 and the pre-registered holdout (2000 disjoint images) -----------------
$PY scripts/fetch_bdd100k.py              # 1000 images + BDD100K's own labels (image list: benchmark_cvpr/bdd100k_1000_queries_12.json)
$PY scripts/eval_external_bdd100k.py      # SigLIP2 + margins -> data/external_adverse/bdd100k/signals.npz
$PY scripts/build_bdd_holdout.py          # 2000 images (list: benchmark_cvpr/bdd100k_holdout_2000_images.json)
for k in $(seq 0 $((NS - 1))); do $PY scripts/holdout_signals.py --part margins --shard $k --nshard $NS; done
$PY scripts/holdout_signals.py --part merge

# --- nuScenes val150 (unseen by every nuScenes-trained model) ---------------------------
$PY scripts/val150_3d_study.py            # CAM_FRONT thumbnails -> work_dirs/val150_thumbs
$PY scripts/extract_human_semantic_gt.py  # object flags from nuScenes 3D annotation (copy: benchmark_cvpr/human_semantic_gt.json)
# val150 margins are cached by the detector study (pipeline/eval_paper.sh, step 6)

# --- baseline signals --------------------------------------------------------------------
$PY scripts/embed_images_baselines.py     # CLIP ViT-B/32 and SigLIP ViT-B/16 image embeddings
for k in $(seq 0 $((NS - 1))); do $PY scripts/caption_scenes_qwen.py --shard $k --nshard $NS; done   # CARIM-style captions
$PY scripts/refav_llm_compile.py          # RefAV-style programs (local Qwen2.5-VL as compiler; no external API)
echo "single-camera Stage 1 done"
