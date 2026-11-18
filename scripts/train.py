"""Training entry point — uses mmengine Runner.

The three custom hooks (TrainableParamCheckpointHook, StageSwitchHook,
OccEvalHook) are proper mmengine-registered Hook classes defined in
src/scene_mining/hooks/ and configured via ``custom_hooks`` in the
training config files.  This script is a thin runner setup — all hook
behaviour lives in the hook classes themselves.

Usage:
    python scripts/train.py --config configs/experiment/av2_hybrid_bev_qwen.py
    python scripts/train.py --config ... --resume
    torchrun --nproc_per_node=4 scripts/train.py --config ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True,  help="Path to mmengine config .py file")
    p.add_argument("--work-dir",  default="work_dirs/scene_miner", help="Output directory")
    p.add_argument("--resume",    action="store_true", help="Resume from latest checkpoint")
    p.add_argument("--load-from", default=None,
                   help="Path to trainable-param checkpoint (.pth) to warm-start from")
    p.add_argument("--no-mmengine", action="store_true",
                   help="Validate config/model, then raise (mmengine required for training)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for python/numpy/torch RNGs and mmengine.randomness. "
                        "Default None = current non-deterministic behaviour.")
    return p.parse_args()


def _stub_mmcv_ops() -> None:
    """Patch mmcv/mmengine so a broken mmcv compiled extension doesn't crash.

    After a PyTorch upgrade the mmcv._ext .so may have undefined symbols.
    mmengine checks for it via mmcv_full_available() (pkgutil.find_loader) and
    then imports mmcv.ops.SyncBatchNorm.  We short-circuit both checks.

    Long-term fix: upgrade mmcv to a version compiled against the current
    PyTorch wheel (see pyproject.toml).
    """
    import sys
    import types
    import importlib.machinery
    import torch

    try:
        import mmengine.utils.dl_utils.misc as _misc
        _misc.mmcv_full_available = lambda: False
    except Exception:
        pass

    if "mmcv.ops" not in sys.modules:
        try:
            import mmcv.ops  # noqa: F401
        except (ImportError, OSError):
            stub = types.ModuleType("mmcv.ops")
            stub.__spec__ = importlib.machinery.ModuleSpec("mmcv.ops", loader=None)
            stub.SyncBatchNorm = torch.nn.SyncBatchNorm
            sys.modules["mmcv.ops"] = stub


def _build_optim_wrapper_cfg(cfg, tr, grad_accum: int) -> dict:
    """Merge cfg.optim_wrapper with the runtime-computed accumulative_counts.

    Reads the top-level ``optim_wrapper`` from the config file (set in
    _base_/training/*.py) and overrides ``accumulative_counts`` with the
    value computed after accounting for DDP world_size.  Falls back to
    sensible defaults if the config key is absent.
    """
    weight_decay = getattr(tr, "weight_decay", 0.01)

    defaults = dict(
        type="AmpOptimWrapper",
        dtype="bfloat16",
        optimizer=dict(
            type="AdamW",
            lr=tr.base_lr,
            weight_decay=weight_decay,
        ),
        accumulative_counts=grad_accum,
        clip_grad=dict(max_norm=tr.grad_clip_norm),
    )

    cfg_optim = getattr(cfg, "optim_wrapper", None)
    if cfg_optim is not None:
        merged = dict(cfg_optim) if hasattr(cfg_optim, "keys") else dict(cfg_optim)
        merged["accumulative_counts"] = grad_accum   # always runtime value
        defaults.update(merged)

    return defaults


def _find_latest_checkpoint(work_dir: str) -> str | None:
    """Find the latest epoch_N.pth checkpoint in work_dir."""
    from pathlib import Path as _P
    ckpts = sorted(_P(work_dir).glob("epoch_*.pth"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return str(ckpts[-1]) if ckpts else None


def _determine_stage(epoch: int, tr) -> str:
    """Return the stage name for a given epoch number."""
    stage0 = getattr(tr, "stage0_epochs", 0)
    end1 = stage0 + tr.stage1_epochs
    end2 = end1 + tr.stage2_epochs
    if epoch <= stage0:
        return "stage0"
    elif epoch <= end1:
        return "stage1"
    elif epoch <= end2:
        return "stage2"
    return "stage3"


def train_with_mmengine(cfg_path: str, work_dir: str, resume: bool,
                        load_from: str | None = None,
                        seed: int | None = None) -> None:
    """Build model, dataloader, and Runner; delegate everything else to hooks."""
    import os, random, torch, torch.distributed as dist
    import numpy as np

    # Pin each DDP rank to its own GPU *before* any CUDA call.
    # Required when GPUs are in Exclusive Process compute mode.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # ── v8 seed plumbing (Phase 4) ─────────────────────────────────────────
    # When --seed is provided, set every common RNG source before importing
    # mmengine (which has its own init_seed paths).  Offset by local_rank so
    # the four DDP workers don't all start in the same dropout state.
    if seed is not None:
        effective_seed = int(seed) + local_rank
        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)
        print(f"[seed] rank {local_rank}: seed={effective_seed} "
              f"(base={seed} + rank={local_rank})")

    _stub_mmcv_ops()

    from mmengine.config import Config
    from mmengine.runner import Runner

    # Register custom hooks so mmengine can instantiate them from config dicts.
    from scene_mining.hooks import (
        TrainableParamCheckpointHook,
        StageSwitchHook,
        OccEvalHook,
    )

    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = work_dir
    if seed is not None:
        # mmengine 0.10+ honours cfg.randomness.seed for internal RNGs.
        cfg.randomness = dict(seed=int(seed), diff_rank_seed=False)

    # ── Model ──────────────────────────────────────────────────────────────────
    from scene_mining.models.scene_miner import SCENE_MINING_MODELS
    from mmengine.registry import build_from_cfg

    model = build_from_cfg(cfg.model, SCENE_MINING_MODELS)

    # Pre-build loss so log_temp (learnable InfoNCE temperature) is part of the
    # model's parameter tree before the optimizer is constructed.
    from scene_mining.losses import SceneMinerLoss
    model._loss_fn = SceneMinerLoss(**{k: v for k, v in cfg.loss.items()})

    # ── Load trainable-param checkpoint (warm-start) ──────────────────────────
    if load_from is not None:
        ckpt = torch.load(load_from, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded_epoch = ckpt.get("epoch", "?")
        print(f"[load-from] Loaded {len(state)} params from {load_from} "
              f"(epoch {loaded_epoch})")
        if missing:
            print(f"[load-from]   missing keys: {len(missing)}")
        if unexpected:
            print(f"[load-from]   unexpected keys: {len(unexpected)}")

    # ── Detect resume stage (if applicable) ──────────────────────────────────
    tr = cfg.train
    stage0_epochs = getattr(tr, "stage0_epochs", 0)
    resume_stage = None

    if resume:
        ckpt_path = _find_latest_checkpoint(work_dir)
        if ckpt_path is not None:
            _meta = torch.load(ckpt_path, map_location="cpu",
                               weights_only=False)["meta"]
            resume_stage = _determine_stage(_meta["epoch"], tr)
            print(f"[resume] Detected epoch {_meta['epoch']} → {resume_stage}")
            del _meta

    # ── Initial freeze state ───────────────────────────────────────────────────
    n_bb = getattr(tr, "n_backbone_blocks", 4)
    if resume_stage == "stage2":
        model.freeze_for_stage2(n_bb)
    elif resume_stage == "stage3":
        model.freeze_for_stage3()
    elif resume_stage == "stage0":
        model.freeze_for_stage0()
    elif stage0_epochs > 0 and resume_stage is None:
        model.freeze_for_stage0()
        print(f"[stage] Starting stage0 ({stage0_epochs} ep) — "
              "spatial encoder + occ_head only")
    elif getattr(tr, "motion_residual_only_mode", False):
        # v8: train only velocity_head + traj_proj. Everything that affects
        # decoded_tokens (the Q-Former stream-3 input) stays frozen, so tag/
        # risk heads see exactly the v4 Q-Former inputs they were trained for.
        model.freeze_for_motion_residual_only()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[stage] motion_residual_only_mode — {n_trainable:,} trainable "
              "params (velocity_head + traj_proj only; decoded_tokens locked)")
    elif getattr(tr, "motion_only_mode", False):
        # v7: train motion head only, keep Q-Former + tag/risk/embed frozen
        # so they continue producing v4-quality outputs.
        model.freeze_for_motion_only()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[stage] motion_only_mode — {n_trainable:,} trainable params "
              "(traj_head only; Q-Former, heads, encoder, backbone, VLM all frozen)")
    else:
        model.freeze_for_stage1()

    max_epochs = (stage0_epochs
                  + tr.stage1_epochs + tr.stage2_epochs + tr.stage3_epochs)

    # ── Dataloader ─────────────────────────────────────────────────────────────
    from scene_mining.datasets.builder import build_train_dataloader
    train_dataloader = build_train_dataloader(cfg)

    # Scale gradient accumulation inversely with world_size so effective
    # batch size is constant across single-GPU and DDP runs.
    world_size = (dist.get_world_size()
                  if (dist.is_available() and dist.is_initialized()) else 1)
    grad_accum = max(1, tr.gradient_accumulation // world_size)
    if world_size > 1:
        print(f"[dist] world_size={world_size}, grad_accum "
              f"{tr.gradient_accumulation} → {grad_accum}")

    # ── Runner ─────────────────────────────────────────────────────────────────
    runner = Runner(
        model=model,
        work_dir=work_dir,
        train_dataloader=train_dataloader,
        train_cfg=dict(
            type="EpochBasedTrainLoop",
            max_epochs=max_epochs,
            val_interval=max_epochs,  # OccEvalHook handles per-epoch eval
        ),
        optim_wrapper=_build_optim_wrapper_cfg(cfg, tr, grad_accum),
        param_scheduler=getattr(cfg, "param_scheduler", None),
        default_hooks=getattr(cfg, "default_hooks", dict(
            checkpoint=dict(type="CheckpointHook", interval=0),
        )),
        custom_hooks=getattr(cfg, "custom_hooks", []),
        resume=resume,
        cfg=cfg,
    )

    # ── Resume fix for mid-stage checkpoints ─────────────────────────────────
    # mmengine's resume requires the optimizer to have the same number of
    # param groups as the saved checkpoint.  Stage 2/3 checkpoints have 2
    # groups (backbone/LoRA + other) but the default optimizer has 1.
    # We monkeypatch load_checkpoint to rebuild param groups right before
    # the checkpoint's optimizer state is loaded.
    if resume and resume_stage in ("stage2", "stage3"):
        _orig_load_ckpt = runner.load_checkpoint

        def _load_ckpt_rebuild_groups(filename, **kwargs):
            module = getattr(runner.model, "module", runner.model)
            opt = runner.optim_wrapper.optimizer

            if resume_stage == "stage2":
                bb_params = [p for _, p in module.backbone.named_parameters()
                             if p.requires_grad]
                other = [p for nm, p in module.named_parameters()
                         if p.requires_grad and not nm.startswith("backbone.")]
                groups = [
                    dict(params=other, lr=tr.base_lr),
                    dict(params=bb_params, lr=tr.stage2_backbone_lr),
                ]
            else:  # stage3
                lora = [p for _, p in module.vlm.named_parameters()
                        if p.requires_grad]
                other = [p for nm, p in module.named_parameters()
                         if p.requires_grad and not nm.startswith("vlm.")]
                groups = [
                    dict(params=other, lr=tr.base_lr),
                    dict(params=lora, lr=tr.stage3_lora_lr),
                ]

            opt.param_groups.clear()
            opt.state.clear()
            for g in groups:
                opt.add_param_group(g)
            print(f"[resume] Rebuilt optimizer with {len(groups)} param groups "
                  f"for {resume_stage}")

            return _orig_load_ckpt(filename, **kwargs)

        runner.load_checkpoint = _load_ckpt_rebuild_groups

    runner.train()


def train_fallback(cfg_path: str, work_dir: str) -> None:
    """Validate config + model construction, then raise NotImplementedError.

    The mmengine-based path is the only fully-implemented training loop.
    This stub confirms config and model construction work before failing.
    """
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location("cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod

    from scene_mining.models.scene_miner import SceneMiner

    print(f"[fallback] Building model from config: {cfg_path}")
    model = SceneMiner(**cfg.model)
    model.freeze_for_stage1()
    print(f"[fallback] Model OK — trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    _Path(work_dir).mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "train_fallback() is not fully implemented. "
        "Install mmengine and use train_with_mmengine() instead.\n"
        f"  Config:   {cfg_path}\n"
        f"  Work dir: {work_dir}"
    )


if __name__ == "__main__":
    args = parse_args()
    if args.no_mmengine:
        train_fallback(args.config, args.work_dir)
    else:
        try:
            import mmengine  # noqa: F401
        except ImportError:
            print("mmengine not installed.")
            train_fallback(args.config, args.work_dir)
        else:
            train_with_mmengine(args.config, args.work_dir, args.resume,
                                load_from=args.load_from, seed=args.seed)
