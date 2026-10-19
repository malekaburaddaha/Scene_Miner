"""TPVFormer spatial encoder — wraps mmdet3d's TPVFormer.

TPVFormer (arXiv:2302.07817) represents the 3D scene as three orthogonal
planes (HW, ZH, ZW) and is available in mmdet3d>=1.1.0.

When mmdet3d is not installed, falls back to a minimal 3-plane transformer
using standard PyTorch nn.MultiheadAttention.

Install mmdet3d for full functionality:
    pip install mmengine mmcv>=2.1.0 mmdet>=3.3.0 mmdet3d>=1.4.0
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn


def build_tpvformer_encoder(
    tpv_h: int = 200,
    tpv_w: int = 200,
    tpv_z: int = 16,
    num_cams: int = 7,
    embed_dims: int = 256,
    num_layers: int = 3,
    pc_range: list[float] | None = None,
) -> "TPVFormerEncoderWrapper":
    """Convenience factory. See TPVFormerEncoderWrapper for args."""
    return TPVFormerEncoderWrapper(
        tpv_h=tpv_h, tpv_w=tpv_w, tpv_z=tpv_z,
        num_cams=num_cams, embed_dims=embed_dims,
        num_layers=num_layers,
        pc_range=pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    )


class TPVFormerEncoderWrapper(nn.Module):
    """Adapter delegating to mmdet3d's TPVFormerEncoder when available.

    Interface::

        forward(patch_features, intrinsics, extrinsics, image_hw)
            → (B, tpv_h*tpv_w + tpv_h*tpv_z + tpv_w*tpv_z, embed_dims)
    """

    def __init__(
        self,
        tpv_h: int,
        tpv_w: int,
        tpv_z: int,
        num_cams: int,
        embed_dims: int,
        num_layers: int,
        pc_range: list[float],
    ) -> None:
        super().__init__()
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.embed_dims = embed_dims
        self._encoder = _try_build_mmdet3d_tpv(
            tpv_h, tpv_w, tpv_z, num_cams, embed_dims, num_layers, pc_range
        )

    @property
    def n_tokens(self) -> int:
        return (self.tpv_h * self.tpv_w
                + self.tpv_h * self.tpv_z
                + self.tpv_w * self.tpv_z)

    def forward(
        self,
        patch_features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> torch.Tensor:
        return self._encoder(patch_features, intrinsics, extrinsics, image_hw)


# ── mmdet3d-backed TPVFormer ──────────────────────────────────────────────────

class _MMDet3dTPVEncoder(nn.Module):
    def __init__(self, encoder, tpv_h, tpv_w, tpv_z, embed_dims):
        super().__init__()
        self.encoder = encoder
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.q_hw = nn.Embedding(tpv_h * tpv_w, embed_dims)
        self.q_zh = nn.Embedding(tpv_z * tpv_h, embed_dims)
        self.q_zw = nn.Embedding(tpv_z * tpv_w, embed_dims)
        for emb in (self.q_hw, self.q_zh, self.q_zw):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, patch_features, intrinsics, extrinsics, image_hw):
        from scene_mining.spatial.bevformer.encoder import _make_img_metas
        B, N_cam, N_p, D = patch_features.shape
        H, W = image_hw
        ph = pw = int(N_p ** 0.5)
        feats = patch_features.view(B, N_cam, ph, pw, D).permute(0, 1, 4, 2, 3)
        img_metas = _make_img_metas(B, N_cam, intrinsics, extrinsics, H, W)

        hw_q = self.q_hw.weight.unsqueeze(0).expand(B, -1, -1)
        zh_q = self.q_zh.weight.unsqueeze(0).expand(B, -1, -1)
        zw_q = self.q_zw.weight.unsqueeze(0).expand(B, -1, -1)

        hw_out, zh_out, zw_out = self.encoder(
            hw_q, zh_q, zw_q, [feats],
            img_metas=img_metas,
            tpv_h=self.tpv_h, tpv_w=self.tpv_w, tpv_z=self.tpv_z,
        )
        return torch.cat([hw_out, zh_out, zw_out], dim=1)


# ── Fallback ──────────────────────────────────────────────────────────────────

class _FallbackTPVEncoder(nn.Module):
    def __init__(self, tpv_h, tpv_w, tpv_z, n_cams, d, n_layers):
        super().__init__()
        self.q_hw = nn.Embedding(tpv_h * tpv_w, d)
        self.q_zh = nn.Embedding(tpv_z * tpv_h, d)
        self.q_zw = nn.Embedding(tpv_z * tpv_w, d)
        for emb in (self.q_hw, self.q_zh, self.q_zw):
            nn.init.normal_(emb.weight, std=0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=d * 4,
                                       batch_first=True, dropout=0.1),
            num_layers=n_layers,
        )
        self.cross = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, patch_features, intrinsics, extrinsics, image_hw):
        B = patch_features.shape[0]
        cam_ctx = patch_features.mean(dim=1)
        planes = []
        for emb in (self.q_hw, self.q_zh, self.q_zw):
            q = emb.weight.unsqueeze(0).expand(B, -1, -1)
            ca, _ = self.cross(q, cam_ctx, cam_ctx)
            q = self.norm(q + ca)
            planes.append(self.encoder(q))
        return torch.cat(planes, dim=1)


def _try_build_mmdet3d_tpv(tpv_h, tpv_w, tpv_z, num_cams, embed_dims, num_layers, pc_range):
    try:
        from mmdet3d.registry import MODELS  # noqa: F401

        encoder_cfg = dict(
            type="TPVFormerEncoder",
            tpv_h=tpv_h, tpv_w=tpv_w, tpv_z=tpv_z,
            num_layers=num_layers,
            pc_range=pc_range,
            num_points_in_pillar=[4, 32, 32],
            num_points_in_pillar_cross_view=[32, 32, 32],
            return_intermediate=False,
            transformerlayers=dict(
                type="TPVFormerLayer",
                attn_cfgs=[
                    dict(
                        type="TPVCrossViewHybridAttention",
                        tpv_h=tpv_h, tpv_w=tpv_w, tpv_z=tpv_z,
                        num_anchors=[4, 32, 32],
                        embed_dims=embed_dims, num_heads=8,
                        num_levels=1, num_points=4,
                    ),
                    dict(
                        type="TPVImageCrossAttention",
                        pc_range=pc_range, num_cams=num_cams,
                        deformable_attention=dict(
                            type="MSDeformableAttention3D",
                            embed_dims=embed_dims, num_points=4, num_levels=1,
                        ),
                        embed_dims=embed_dims,
                    ),
                ],
                feedforward_channels=embed_dims * 4,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        )
        encoder = MODELS.build(encoder_cfg)
        return _MMDet3dTPVEncoder(encoder, tpv_h, tpv_w, tpv_z, embed_dims)

    except (ImportError, Exception) as exc:
        warnings.warn(
            f"mmdet3d TPVFormerEncoder not available ({exc}). Using fallback.",
            ImportWarning, stacklevel=4,
        )
        return _FallbackTPVEncoder(tpv_h, tpv_w, tpv_z, num_cams, embed_dims, num_layers)


# Alias exported by __init__.py and spatial/__init__.py
TPVFormerEncoder = TPVFormerEncoderWrapper
