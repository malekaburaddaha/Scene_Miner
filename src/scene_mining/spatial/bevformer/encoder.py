"""BEVFormer spatial encoder — wraps mmdet3d's BEVFormerEncoder.

Uses mmdet3d's production BEVFormerEncoder with MSDeformableAttention3D /
TemporalSelfAttention. Falls back to a lightweight F.grid_sample stub when
mmdet3d is not installed (CPU / unit-test environments).

Install mmdet3d for full functionality:
    pip install mmengine mmcv>=2.1.0 mmdet>=3.3.0 mmdet3d>=1.4.0
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn


def build_bevformer_encoder(
    bev_h: int = 50,
    bev_w: int = 50,
    num_feature_levels: int = 1,
    num_cams: int = 7,
    embed_dims: int = 256,
    num_layers: int = 3,
    num_points_in_pillar: int = 4,
    pc_range: list[float] | None = None,
) -> "BEVFormerEncoderWrapper":
    """Convenience factory — see BEVFormerEncoderWrapper for args."""
    return BEVFormerEncoderWrapper(
        bev_h=bev_h,
        bev_w=bev_w,
        num_feature_levels=num_feature_levels,
        num_cams=num_cams,
        embed_dims=embed_dims,
        num_layers=num_layers,
        num_points_in_pillar=num_points_in_pillar,
        pc_range=pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    )


class BEVFormerEncoderWrapper(nn.Module):
    """Adapter that delegates to mmdet3d's BEVFormerEncoder when available.

    Interface::

        forward(patch_features, intrinsics, extrinsics, image_hw)
            → (B, bev_h * bev_w, embed_dims)

    Args:
        bev_h/bev_w: BEV grid size (cells).
        num_feature_levels: FPN levels fed to deformable attention (1 for ViT).
        num_cams: Number of ring cameras.
        embed_dims: Spatial token dimension.
        num_layers: BEVFormerEncoder transformer layers.
        num_points_in_pillar: Vertical sampling points per BEV pillar.
        pc_range: [xmin, ymin, zmin, xmax, ymax, zmax] in metres.
    """

    def __init__(
        self,
        bev_h: int,
        bev_w: int,
        num_feature_levels: int,
        num_cams: int,
        embed_dims: int,
        num_layers: int,
        num_points_in_pillar: int,
        pc_range: list[float],
    ) -> None:
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self._encoder = _try_build_mmdet3d(
            bev_h, bev_w, num_feature_levels, num_cams,
            embed_dims, num_layers, num_points_in_pillar, pc_range,
        )

    def forward(
        self,
        patch_features: torch.Tensor,   # (B, N_cam, N_patches, embed_dims)
        intrinsics: torch.Tensor,        # (B, N_cam, 3, 3)
        extrinsics: torch.Tensor,        # (B, N_cam, 4, 4) ego→cam
        image_hw: tuple[int, int],       # (H, W) of the patch grid
    ) -> torch.Tensor:
        """Returns (B, bev_h * bev_w, embed_dims)."""
        return self._encoder(patch_features, intrinsics, extrinsics, image_hw)


# ── mmdet3d-backed encoder ────────────────────────────────────────────────────

class _MMDet3dBEVEncoder(nn.Module):
    """Wraps mmdet3d BEVFormerEncoder in SceneMiner's calling convention."""

    def __init__(self, encoder: nn.Module, bev_h: int, bev_w: int, embed_dims: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.bev_h = bev_h
        self.bev_w = bev_w
        # Learnable BEV query & positional embeddings
        self.bev_queries = nn.Embedding(bev_h * bev_w, embed_dims)
        self.bev_pos = nn.Embedding(bev_h * bev_w, embed_dims)
        nn.init.normal_(self.bev_queries.weight, std=0.02)
        nn.init.normal_(self.bev_pos.weight, std=0.02)

    def forward(self, patch_features, intrinsics, extrinsics, image_hw):
        B, N_cam, N_p, D = patch_features.shape
        H, W = image_hw
        ph = pw = int(N_p ** 0.5)

        # Reshape to (B, N_cam, D, ph, pw) — mmdet3d mlvl_feats format
        feats = patch_features.view(B, N_cam, ph, pw, D).permute(0, 1, 4, 2, 3)
        mlvl_feats = [feats]

        idx = torch.arange(self.bev_h * self.bev_w, device=patch_features.device)
        bev_q = self.bev_queries(idx).unsqueeze(0).expand(B, -1, -1)
        bev_p = self.bev_pos(idx).unsqueeze(0).expand(B, -1, -1)

        img_metas = _make_img_metas(B, N_cam, intrinsics, extrinsics, H, W)
        spatial_shapes = torch.tensor([[ph, pw]], dtype=torch.long, device=feats.device)
        level_start_index = torch.zeros(1, dtype=torch.long, device=feats.device)

        bev = self.encoder(
            bev_q,
            mlvl_feats,
            bev_pos=bev_p,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=torch.zeros(B, 2, device=bev_q.device),
            img_metas=img_metas,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
        )
        return bev  # (B, bev_h*bev_w, D)


# ── Pure-PyTorch fallback ──────────────────────────────────────────────────────

class _FallbackBEVEncoder(nn.Module):
    """F.grid_sample fallback when mmdet3d is not installed.

    Produces correct output shapes for CPU unit tests.
    No custom CUDA — not suitable for production training.
    """

    def __init__(self, bev_h: int, bev_w: int, n_cams: int, d: int, n_layers: int) -> None:
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.bev_queries = nn.Embedding(bev_h * bev_w, d)
        nn.init.normal_(self.bev_queries.weight, std=0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=d * 4,
                                       batch_first=True, dropout=0.1),
            num_layers=n_layers,
        )
        self.cam_attn = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
        self.cam_norm = nn.LayerNorm(d)

    def forward(self, patch_features, intrinsics, extrinsics, image_hw):
        B = patch_features.shape[0]
        # Simple cross-attend BEV queries → pooled camera context
        cam_ctx = patch_features.mean(dim=1)  # (B, N_patches, D)
        idx = torch.arange(self.bev_h * self.bev_w, device=patch_features.device)
        bev = self.bev_queries(idx).unsqueeze(0).expand(B, -1, -1)
        bev_ca, _ = self.cam_attn(bev, cam_ctx, cam_ctx)
        bev = self.cam_norm(bev + bev_ca)
        return self.encoder(bev)  # (B, bev_h*bev_w, D)


# ── Factory helper ─────────────────────────────────────────────────────────────

def _try_build_mmdet3d(
    bev_h, bev_w, num_feature_levels, num_cams,
    embed_dims, num_layers, num_points_in_pillar, pc_range,
) -> nn.Module:
    try:
        from mmdet3d.registry import MODELS  # noqa: F401

        encoder_cfg = dict(
            type="BEVFormerEncoder",
            num_layers=num_layers,
            pc_range=pc_range,
            num_points_in_pillar=num_points_in_pillar,
            return_intermediate=False,
            transformerlayers=dict(
                type="BEVFormerLayer",
                attn_cfgs=[
                    dict(
                        type="TemporalSelfAttention",
                        embed_dims=embed_dims,
                        num_levels=1,
                    ),
                    dict(
                        type="SpatialCrossAttention",
                        pc_range=pc_range,
                        deformable_attention=dict(
                            type="MSDeformableAttention3D",
                            embed_dims=embed_dims,
                            num_points=num_points_in_pillar,
                            num_levels=num_feature_levels,
                        ),
                        embed_dims=embed_dims,
                    ),
                ],
                feedforward_channels=embed_dims * 4,
                ffn_dropout=0.1,
                operation_order=(
                    "self_attn", "norm", "cross_attn", "norm", "ffn", "norm"
                ),
            ),
        )
        encoder = MODELS.build(encoder_cfg)
        return _MMDet3dBEVEncoder(encoder, bev_h, bev_w, embed_dims)

    except (ImportError, Exception) as exc:
        warnings.warn(
            f"mmdet3d not available ({exc}). Using fallback BEV encoder. "
            "Run: pip install mmcv mmdet mmdet3d for production use.",
            ImportWarning,
            stacklevel=4,
        )
        return _FallbackBEVEncoder(bev_h, bev_w, num_cams, embed_dims, num_layers)


# Alias exported by __init__.py and spatial/__init__.py
BEVFormerEncoder = BEVFormerEncoderWrapper


def _make_img_metas(B, N_cam, intrinsics, extrinsics, H, W):
    """Convert calibration tensors to mmdet3d img_metas list."""
    metas = []
    for b in range(B):
        lidar2img = []
        for c in range(N_cam):
            K = intrinsics[b, c].detach().cpu().numpy()
            E = extrinsics[b, c].detach().cpu().numpy()
            K4 = np.eye(4, dtype=np.float32)
            K4[:3, :3] = K
            lidar2img.append((K4 @ E).tolist())
        metas.append({
            "img_shape": [(H, W, 3)] * N_cam,
            "lidar2img": lidar2img,
        })
    return metas
