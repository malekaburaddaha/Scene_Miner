# SceneMiner v4 — motion-head fix retrain (with waypoint_mask)
#
# v3 vs. v4 diagnosis (see PAPER_RESULTS.md):
#   v3 raised w_motion to 0.5 and removed the 20× normalisation, but motion
#   train loss stayed at ~9 m across all 5 epochs.  Root cause discovered by
#   `scripts/sanity_check_waypoints.py`:
#     * ~25 % of GT trajectories contain padded (0, 0) steps for "agent left
#       the scene" — these were being supervised as real targets.
#     * MotionLoss only used agent_mask, not waypoint_mask (per-step).
#     * The model learned a constant-zero-after-some-prefix solution that
#       minimises loss on padding ~equally well as predicting real motion.
#
# v4 fix:
#   - MotionLoss now multiplies in waypoint_mask (B, N, T) before summing.
#   - SceneMinerLoss and SceneMiner.train_step plumb waypoint_mask through.
#   - Eval (scripts/evaluate_nuscenes.py) likewise uses waypoint_mask so
#     ADE/FDE are computed over real future steps only.
#
# Everything else matches v3 (w_motion=0.5, waypoint_scale=1.0, LR=2e-5,
# warm-start from v2 epoch_15.pth).
#
# Run:
#   python -m torch.distributed.run --nproc_per_node=4 scripts/train.py --config configs/experiment/nuscenes_scenemining_v4_motion_mask.py --work-dir work_dirs/nuscenes_scenemining_v4_motion_mask  (see pipeline/stage1_multicam.sh)

_base_ = [
    "../_base_/datasets/nuscenes_train.py",
    "../_base_/training/staged_3stage.py",
    "../_base_/loss/default.py",
]

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
        bev_h=50, bev_w=50,
        num_feature_levels=1, num_cams=6,
        embed_dims=256, num_layers=3, num_points_in_pillar=4,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    ),
    qformer=dict(
        type="QFormer",
        d_spatial=256, d_camera=256, d_motion=256, d_vlm=2048,
        n_query_tokens=32,
        pretrained_name="Salesforce/blip2-opt-2.7b",
    ),
    vlm=dict(
        type="Qwen25VL",
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        load_in_4bit=False,
        lora_r=16, lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj"],
    ),
    traj_head=dict(
        type="MotionHead",
        d_model=256, n_agents=8, n_waypoints=6,
        n_decoder_layers=2, nhead=8,
        dim_feedforward=512, dropout=0.1,
    ),
    n_tags=20,
    d_embed=256,
)

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
    labels_dir="./data/labels",
    tokenizer_path="Qwen/Qwen2.5-VL-3B-Instruct",
    n_agents=8, n_waypoints=6, max_text_len=128,
)

loss = dict(
    w_embed=1.0,
    w_tag=0.5,
    w_risk=0.2,
    w_caption=0.0,
    w_motion=0.5,
    motion_waypoint_scale=1.0,
    motion_beta=1.0,
    temperature=0.07,
)

train = dict(
    stage1_epochs=5,
    stage2_epochs=0,
    stage3_epochs=0,
    base_lr=2e-5,
    stage2_backbone_lr=1e-6,
    stage3_lora_lr=1e-5,
    n_backbone_blocks=4,
    grad_clip_norm=1.0,
    batch_size=2,
    gradient_accumulation=32,
    warmup_iters=200,
    warmup_ratio=0.1,
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",
    optimizer=dict(type="AdamW", lr=2e-5, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(type="CosineAnnealingLR", T_max=5, eta_min=1e-7, by_epoch=True, begin=0, end=5),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        # Save weights only — earlier run OOM-killed during full-state save
        # right at the epoch 1 boundary.  We're only 5 epochs and won't resume,
        # so optimizer + scheduler state is disposable.
        save_optimizer=False,
        save_param_scheduler=False,
        save_last=True,
    ),
)

work_dir = "work_dirs/nuscenes_scenemining_v4_motion_mask"
