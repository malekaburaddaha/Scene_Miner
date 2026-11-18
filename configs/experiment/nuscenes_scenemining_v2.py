# SceneMiner v2 — full pipeline with 3-stream Q-Former + MotionHead supervision
#
# What's new vs. nuscenes_hybrid_bev_qwen.py:
#   - 3-stream Q-Former: fuses BEV spatial + camera features + motion tokens
#   - MotionHead with GT future waypoints as supervision (Task 4 output)
#   - Dataset loads real labels from data/labels/:
#       * nuprompt_tags.json         (20-dim binary tags — Task 1+2)
#       * gt_risk_scores.json        (physics-based risk — Task 3)
#       * gt_future_waypoints.pkl    (per-agent future trajectories — Task 4)
#       * scene_descriptions.json    (tokenized for VLM — Task 5)
#   - Checkpoint every 5 epochs with full optimizer + scheduler state (resumable)
#
# Qwen3.5-9B is deferred: it needs transformers>=4.57.0.dev0 which would
# require either an environment rebuild or in-place upgrade of the main
# scene_mining env. For now we keep Qwen2.5-VL-3B which is known to work.
#
# Run:
#   python scripts/train.py --config configs/experiment/nuscenes_scenemining_v2.py
#
# Resume:
#   python scripts/train.py --config configs/experiment/nuscenes_scenemining_v2.py --resume

_base_ = [
    "../_base_/datasets/nuscenes_train.py",
    "../_base_/training/staged_3stage.py",
    "../_base_/loss/default.py",
]

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="SceneMiner",
    backbone=dict(
        type="SigLIP2Backbone",
        model_name="google/siglip2-base-patch16-224",
        proj_dim=256,
        freeze=True,
    ),
    spatial_encoder=dict(
        type="BEVFormerEncoder",
        bev_h=50,
        bev_w=50,
        num_feature_levels=1,
        num_cams=6,
        embed_dims=256,
        num_layers=3,
        num_points_in_pillar=4,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    ),
    # ── 3-stream Q-Former (Task 7) ─────────────────────────────────────
    # Fuses BEV tokens + camera patch features + motion tokens before
    # cross-attention. Each stream is projected to encoder_hidden_d.
    qformer=dict(
        type="QFormer",
        d_spatial=256,          # BEVFormer output dim
        d_camera=256,           # SigLIP2 patch features after proj_dim=256
        d_motion=256,           # MotionHead d_model
        d_vlm=2048,             # Qwen2.5-VL-3B hidden_size
        n_query_tokens=32,
        pretrained_name="Salesforce/blip2-opt-2.7b",
    ),
    vlm=dict(
        type="Qwen25VL",
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        load_in_4bit=False,
        lora_r=16,
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj"],
    ),
    # ── MotionHead (Task 8, now with GT supervision) ──────────────────
    # Predicts (N_agents=8, T=6, 2) future waypoints in ego frame.
    # Also supplies decoded_tokens to Q-Former stream 3.
    traj_head=dict(
        type="MotionHead",
        d_model=256,
        n_agents=8,
        n_waypoints=6,
        n_decoder_layers=2,
        nhead=8,
        dim_feedforward=512,
        dropout=0.1,
    ),
    n_tags=20,
    d_embed=256,
)

# ── Data (Task 9) ─────────────────────────────────────────────────────────────
# Points to the new label files produced by Tasks 1-5.
# tokenizer_path is the Qwen tokenizer used to encode scene_descriptions.json.
data = dict(
    type="NuScenesDataset",
    dataroot="./data/nuscenes",
    version="v1.0-trainval",
    split="train",
    val_split="val",
    num_cams=6,
    image_size=(448, 800),
    verbose=False,
    load_occ_gt=False,
    # ── New label sources ─────────────────────────────────────────────
    labels_dir="./data/labels",
    tokenizer_path="Qwen/Qwen2.5-VL-3B-Instruct",
    n_agents=8,
    n_waypoints=6,
    max_text_len=128,
)

# ── Loss weights ──────────────────────────────────────────────────────────────
# Override default.py to enable MotionLoss.
# w_motion=0.1 with normalised MotionLoss (divided by 20m waypoint scale)
# keeps motion contribution in the same range as the other losses.
# Prior run with w_motion=0.3 on un-normalised loss caused NaN collapse.
loss = dict(
    w_embed=1.0,
    w_tag=0.5,
    w_risk=0.2,
    w_caption=0.0,     # caption CE disabled until stage 3 wiring is finalized
    w_motion=0.1,      # enable MotionLoss for waypoint supervision
    temperature=0.07,
)

# ── Optimiser override: lower base LR + longer warmup for stability ─────────
# Initial run with base_lr=1e-4 caused grad_norm=148 early, which tipped
# bf16 into NaN. Halve it and extend warmup.
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",
    optimizer=dict(
        type="AdamW",
        lr=5e-5,              # halved from 1e-4
        weight_decay=0.01,
    ),
    clip_grad=dict(max_norm=1.0),
)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.05,    # 5% of base_lr at step 0 (was 0.1)
        by_epoch=False,
        begin=0,
        end=1000,             # longer warmup: 1000 steps (was 500)
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=30,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=30,
    ),
]

# ── Checkpoint every 5 epochs ─────────────────────────────────────────────────
# Overrides the base config's interval=1. Saves full state (model + optimizer
# + scheduler) so `--resume` fully restores training.
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=5,
        max_keep_ckpts=6,      # keep 6 × 5 epochs = 30 epochs of history
        save_optimizer=True,
        save_param_scheduler=True,
        save_last=True,         # always keep a last_checkpoint pointer
    ),
)

work_dir = "work_dirs/nuscenes_scenemining_v2"
