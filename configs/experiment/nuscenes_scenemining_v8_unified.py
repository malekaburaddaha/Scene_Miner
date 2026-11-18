# SceneMiner v8 — Unified pipeline, true tag/risk preservation.
#
# Why v7 failed (PAPER_RESULTS Task 14):
#   v7 froze Q-Former + tag/risk hoping their forward output would stay v4's.
#   It didn't: the new MotionHead architecture (random-init decoder layers 3+4,
#   agent_pos_mlp) shifted ``decoded_tokens`` → Q-Former stream-3 input →
#   pooled features → frozen tag/risk heads saw a different distribution and
#   regressed (macro F1 0.260 → 0.146; risk Pearson +0.39 → −0.19).
#
# v8 fix: lock decoded_tokens to v4's at iter 0 AND throughout training.
#
# 1. ``identity_init_from_layer=2`` on MotionHead → new decoder layers 2+3
#    have zeroed residual projections (self_attn.out_proj, multihead_attn.
#    out_proj, linear2).  Each new layer becomes the identity function on
#    its input, so the layer-1 output flows through unchanged.
# 2. ``agent_pos_mlp`` last linear is zero-init (weight=0, bias=0), so
#    position embeddings contribute 0 to queries at iter 0.
# 3. Smoke test verified ``decoded_tokens`` bit-identical to a v4-style
#    2-layer head (diff 2.6e-6, numerical noise).
# 4. ``freeze_for_motion_residual_only`` (in scene_miner.py) freezes
#    EVERY parameter that influences ``decoded_tokens``: agent_queries,
#    full decoder, agent_pos_mlp, temporal_proj.  Q-Former, tag/risk/embed,
#    backbone, VLM all stay frozen too.
# 5. Trainable: velocity_head (~32k params) + traj_proj (~70k params).
#    Both affect ``traj_pred`` only, never ``decoded_tokens``.
#
# Expected outcome:
#   - Tag micro-F1 ≈ 0.554, macro-F1 ≈ 0.260, risk Pearson ≈ +0.393
#     (identical to v4 because the heads see v4-identical inputs).
#   - Motion ADE/FDE driven by velocity_head learning the kinematic prior
#     and traj_proj learning a residual.  Should approach v6's ~2 m ADE
#     since the Phase-2 mechanisms are still active in the prediction path.
#
# Run:
#   python scripts/train_wacv.py --config configs/experiment/nuscenes_scenemining_v8_unified.py
# Checkpoints are hub ids; point them at local copies with --cfg-options if needed.

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
        proj_dim=256, freeze=True,
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
        n_decoder_layers=4,
        nhead=8, dim_feedforward=512, dropout=0.1,
        d_temporal_in=768,
        agent_position_dim=2,
        use_velocity_subhead=True,
        waypoint_dt=0.5,
        identity_init_from_layer=2,             # ← v8 key change
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
    n_temporal_frames=3,
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

# 2-epoch test.  Trainable param count is tiny (~100K) so optimisation
# converges quickly.  If motion ADE drops < 4 m on val, we keep training
# more epochs; otherwise we have the answer fast.
train = dict(
    stage1_epochs=2,
    stage2_epochs=0,
    stage3_epochs=0,
    base_lr=5e-5,                                # bigger LR — fewer params
    stage2_backbone_lr=1e-6,
    stage3_lora_lr=1e-5,
    n_backbone_blocks=4,
    grad_clip_norm=1.0,
    batch_size=2,
    gradient_accumulation=32,
    warmup_iters=200,
    warmup_ratio=0.1,
    motion_residual_only_mode=True,              # ← v8 switch
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",
    optimizer=dict(type="AdamW", lr=5e-5, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(type="CosineAnnealingLR", T_max=2, eta_min=1e-7, by_epoch=True, begin=0, end=2),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=10**9,
        save_optimizer=False,
        save_param_scheduler=False,
        save_last=False,
        save_best=None,
    ),
)
custom_hooks = [
    dict(type="TrainableParamCheckpointHook", interval=1, max_keep_ckpts=2),
]

work_dir = "work_dirs/nuscenes_scenemining_v8_unified"
