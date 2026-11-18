# Standard 3-stage training schedule (30 epochs total, effective batch = 16).
#
#   Stage 1 (10 ep): backbone + VLM frozen → train spatial encoder + Q-Former + heads
#   Stage 2 (10 ep): unfreeze last 4 backbone blocks at 10x reduced LR
#   Stage 3 (10 ep): enable VLM LoRA adapters

train = dict(
    stage1_epochs=10,
    stage2_epochs=10,
    stage3_epochs=10,
    base_lr=1e-4,
    stage2_backbone_lr=1e-5,
    stage3_lora_lr=1e-4,
    n_backbone_blocks=4,
    grad_clip_norm=1.0,
    batch_size=2,
    gradient_accumulation=32,   # effective batch = 2 × 4 GPUs × 32 = 256
    warmup_iters=500,           # linear LR warmup for first N iterations
    warmup_ratio=0.1,           # start LR = base_lr * warmup_ratio
)

# Standard mmengine optim_wrapper config.
# accumulative_counts is set at runtime in train.py to account for world_size.
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",           # bf16 has fp32 value range, safe for InfoNCE gradients
    optimizer=dict(
        type="AdamW",
        lr=1e-4,                # == train.base_lr; duplicated for Runner compatibility
        weight_decay=0.01,
    ),
    clip_grad=dict(max_norm=1.0),
)

# LinearLR warmup for first warmup_iters steps, then CosineAnnealing to end.
# T_max == total epochs (stage1 + stage2 + stage3 = 30).
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,       # == train.warmup_ratio
        by_epoch=False,
        begin=0,
        end=500,                # == train.warmup_iters
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

# Hooks registered in the mmengine HOOKS registry (see src/scene_mining/hooks/).
# Stage/LR params are read from runner.cfg.train at before_run time.
custom_hooks = [
    dict(type="StageSwitchHook", priority="HIGH"),
    dict(type="OccEvalHook", n_vis_frames=5, priority="LOW"),
]

# Full-state CheckpointHook — saves model + optimizer + scheduler for --resume.
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True,
        save_param_scheduler=True,
    ),
)
