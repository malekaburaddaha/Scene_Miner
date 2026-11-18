#!/usr/bin/env bash
# Stage 1, multi-camera path (nuScenes, Waymo): BEV embedding e (256-d), 20 tag logits t,
# risk scalar r. GPU, 4 x A40 for the training phases; run from the repository root.
#
#   Waymo has 5 cameras: scene_mining.data.universal_dataset pads every frame to 6 camera
#   slots and masks the padded view (camera_valid = False), so one model serves both datasets.
#
# Environment: PY (default python) in the `scene_mining` env (environment.yml);
# WAYMO_PY in a separate TensorFlow env for the Waymo dump (waymo-open-dataset-tf-2-12-0);
# WAYMO_ROOT = Waymo Open Dataset v1.4.3 (perception) tfrecords.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
PY=${PY:-python}; WAYMO_PY=${WAYMO_PY:-python}; NPROC=${NPROC:-4}
DIST="$PY -m torch.distributed.run --nproc_per_node=$NPROC"

# --- 0. data and Stage-1 supervision ----------------------------------------------------
# nuScenes v1.0-trainval in data/nuscenes; nuPrompt annotations as expected by the tag script.
$PY scripts/prepare_nuscenes_data.py --dataroot data/nuscenes --version v1.0-trainval --out-dir data/nuscenes --splits train val
$PY scripts/extract_nuprompt_tags.py           # -> data/labels/nuprompt_tags.json (20 binary tags per scene)
$PY scripts/enrich_tags_from_nuscenes.py       # adds environmental tags from scene descriptions
$PY scripts/compute_gt_risk_scores.py          # -> per-scene risk target (proximity, TTC, VRU density)
# Waymo frames: dump with the TF env; the exact (segment, frame) pairs of the paper are listed in
# benchmark_cvpr/waymo_frames.json (check the generated manifest.json files against it).
$WAYMO_PY scripts/waymo_extract_perception.py --waymo-root "$WAYMO_ROOT/individual_files/training" \
    --out work_dirs/waymo_zeroshot/extracted_large
for k in 0 1 2; do
  $WAYMO_PY scripts/waymo_extract_perception.py --waymo-root "$WAYMO_ROOT/individual_files/training" \
      --out work_dirs/waymo_zeroshot/extracted_all_$k
done

# --- 1. multi-camera model, four phases (each phase initialises from the previous one) ---
# v2: BEVFormer 50x50 encoder over frozen SigLIP2 ViT-B/16 patches, BLIP-2 Q-Former (32 queries),
#     Qwen2.5-VL-3B with LoRA r=16, embedding / tag / risk heads; from the public foundation weights.
$DIST scripts/train.py --config configs/experiment/nuscenes_scenemining_v2.py \
    --work-dir work_dirs/nuscenes_scenemining_v2
# v4: motion head with agent masking.
$DIST scripts/train.py --config configs/experiment/nuscenes_scenemining_v4_motion_mask.py \
    --work-dir work_dirs/nuscenes_scenemining_v4_motion_mask --load-from work_dirs/nuscenes_scenemining_v2/epoch_15.pth
# v8: unified tag / risk / motion heads (decoded tokens locked to v4's).
$DIST scripts/train.py --config configs/experiment/nuscenes_scenemining_v8_unified.py \
    --work-dir work_dirs/nuscenes_scenemining_v8_unified --load-from work_dirs/nuscenes_scenemining_v4_motion_mask/epoch_1.pth
# WACV LoRA-FM: one interleaved nuScenes (0.6) + Waymo (0.4) stream, masked per-domain losses,
# Qwen LoRA trainable, motion head frozen.
$DIST scripts/train_wacv.py --config configs/experiment/nuscenes_scenemining_v8_unified.py \
    --base-checkpoint work_dirs/nuscenes_scenemining_v4_motion_mask/epoch_1.pth \
    --v8-checkpoint work_dirs/nuscenes_scenemining_v8_unified/epoch_1_trainable.pth \
    --waymo work_dirs/waymo_zeroshot/extracted_large --nusc-root data/nuscenes --nusc-version v1.0-trainval \
    --nusc-split train --labels data/labels --weights 0.6 0.4 --batch-size 2 --epochs 3 --steps-per-epoch 0 \
    --lr 1e-4 --num-workers 4 --use-lora --freeze-motion --lora-lr 1e-5 --lora-warmup-steps 200 --grad-clip 1.0 \
    --w-embed 1.0 --temp-floor 0.05 --amp-dtype bf16 --ckpt-every-steps 500 --log-every 1 \
    --work-dir work_dirs/wacv_unified_lora_fm

# --- 2. export e, t, r for the 1508-scene corpus ------------------------------------------
# 700 nuScenes train scenes (one CAM_FRONT thumbnail per scene is saved for the single-camera path)
$PY scripts/export_filter_features_scaled.py --wacv-checkpoint work_dirs/wacv_unified_lora_fm/wacv_epoch2.pth \
    --out work_dirs/unified_filter/features_scaled.npz --thumbs work_dirs/unified_filter/scene_thumbs
# + 808 Waymo scenes (202 segments x four 4-frame chunks), merged with the nuScenes features
$PY scripts/export_filter_features_overnight.py \
    --extracted-roots work_dirs/waymo_zeroshot/extracted_all_0 work_dirs/waymo_zeroshot/extracted_all_1 work_dirs/waymo_zeroshot/extracted_all_2 \
    --nusc-features work_dirs/unified_filter/features_scaled.npz --wacv-checkpoint work_dirs/wacv_unified_lora_fm/wacv_epoch2.pth \
    --out work_dirs/unified_filter/features_overnight.npz --thumbs work_dirs/unified_filter/scene_thumbs --chunk 4
# recorded metadata (lighting, weather, density) for the prior and the night/rain fallback
$PY scripts/build_deterministic_gt.py --features work_dirs/unified_filter/features_overnight.npz \
    --nusc-root data/nuscenes --waymo-roots work_dirs/waymo_zeroshot/extracted_all_0 work_dirs/waymo_zeroshot/extracted_all_1 work_dirs/waymo_zeroshot/extracted_all_2
echo "multi-camera Stage 1 done: work_dirs/unified_filter/features_overnight.npz"
