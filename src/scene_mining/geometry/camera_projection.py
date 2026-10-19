"""Camera projection utilities for BEV/TPV reference point generation.

All operations are pure PyTorch — no MMCV or custom CUDA kernels required.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_bev_reference_points(
    bev_h: int,
    bev_w: int,
    bev_z_heights: list[float],
    intrinsics: torch.Tensor,
    extrinsics_ego2cam: torch.Tensor,
) -> torch.Tensor:
    """Project 3D BEV grid points onto each camera image plane.

    For each BEV grid cell (at one or more Z heights), compute its
    normalised UV coordinates in each camera image.

    Args:
        bev_h: Number of BEV rows (X-axis, forward).
        bev_w: Number of BEV columns (Y-axis, left).
        bev_z_heights: List of Z heights to sample per cell (metres).
            Typical: [-1.0, 0.0, 1.0, 2.0] for multi-height sampling.
        intrinsics: (B, N_cam, 3, 3) camera K matrices.
        extrinsics_ego2cam: (B, N_cam, 4, 4) ego-to-camera transforms.

    Returns:
        ref_pts: (B, bev_h*bev_w, N_cam, 2) normalised UV in [0,1].
            Points outside the image field are clamped (use valid mask).
        valid_mask: (B, bev_h*bev_w, N_cam) bool True where the point
            projects inside the image.
    """
    B, N_cam = intrinsics.shape[:2]
    device = intrinsics.device
    dtype = intrinsics.dtype

    # BEV grid in ego frame — centred at origin
    # X range: [-bev_h/2, bev_h/2] * resolution_m (assumed 0.5m)
    # Y range: [-bev_w/2, bev_w/2] * resolution_m
    bev_range_x = bev_h * 0.5  # total half-extent in metres (1 cell = 1m)
    bev_range_y = bev_w * 0.5

    xs = torch.linspace(-bev_range_x, bev_range_x, bev_h, device=device, dtype=dtype)
    ys = torch.linspace(-bev_range_y, bev_range_y, bev_w, device=device, dtype=dtype)

    # Grid: (bev_h, bev_w, 2)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="xy")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (H*W, 2)

    N_pts = bev_h * bev_w
    N_z = len(bev_z_heights)
    z_tensor = torch.tensor(bev_z_heights, device=device, dtype=dtype)  # (N_z,)

    # Build 3D points in ego frame: (N_pts*N_z, 3)
    pts_xy = grid_xy.unsqueeze(1).expand(-1, N_z, -1).reshape(-1, 2)
    pts_z = z_tensor.unsqueeze(0).expand(N_pts, -1).reshape(-1, 1)
    pts_3d = torch.cat([pts_xy, pts_z, torch.ones_like(pts_z)], dim=-1)  # (N_pts*N_z, 4)

    # Project into each camera
    # extrinsics: (B, N_cam, 4, 4)
    # pts_3d: (B, N_pts*N_z, 4) after broadcast
    pts_3d_b = pts_3d.unsqueeze(0).expand(B, -1, -1)  # (B, N_pts*N_z, 4)

    # Camera coordinates: (B, N_cam, N_pts*N_z, 4)
    pts_cam = torch.einsum(
        "bcij,bnj->bcni",
        extrinsics_ego2cam,
        pts_3d_b,
    )  # (B, N_cam, N_pts*N_z, 4)

    # Pixel coordinates: (B, N_cam, N_pts*N_z, 3) = K @ xyz_cam
    xyz_cam = pts_cam[..., :3]  # (B, N_cam, N_pts*N_z, 3)
    uv_hom = torch.einsum("bcij,bcnj->bcni", intrinsics, xyz_cam)  # (B, N_cam, N, 3)

    depth = uv_hom[..., 2:3].clamp(min=1e-5)
    uv = uv_hom[..., :2] / depth  # (B, N_cam, N, 2)

    # Normalise to [0,1] — need image size
    # Approximation: normalise by the max expected pixel value
    # (This will be adjusted by the spatial encoder with actual H, W)
    # We return raw pixel UV here; normalisation happens in BEVFormer

    # Take mean across Z heights for each BEV cell
    uv_mean = uv.view(B, N_cam, N_pts, N_z, 2).mean(dim=3)  # (B, N_cam, N_pts, 2)
    depth_mean = depth.view(B, N_cam, N_pts, N_z, 1).mean(dim=3).squeeze(-1)

    # Valid mask: depth > 0 (in front of camera)
    valid = depth_mean > 0.1  # (B, N_cam, N_pts)

    # Rearrange to (B, N_pts, N_cam, 2)
    ref_pts = uv_mean.permute(0, 2, 1, 3)   # (B, N_pts, N_cam, 2)
    valid = valid.permute(0, 2, 1)           # (B, N_pts, N_cam)

    return ref_pts, valid


def normalize_uv(
    uv: torch.Tensor,
    image_h: int,
    image_w: int,
) -> torch.Tensor:
    """Normalise pixel UV coordinates to [-1, 1] for F.grid_sample.

    Args:
        uv: (..., 2) pixel coordinates (u=col, v=row).
        image_h: Image height in pixels.
        image_w: Image width in pixels.

    Returns:
        Normalised UV in [-1, 1] range.
    """
    uv_norm = uv.clone()
    uv_norm[..., 0] = (uv[..., 0] / (image_w - 1)) * 2 - 1
    uv_norm[..., 1] = (uv[..., 1] / (image_h - 1)) * 2 - 1
    return uv_norm


def sample_features_at_points(
    patch_features: torch.Tensor,
    ref_pts_norm: torch.Tensor,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    """Sample patch features at 2D reference points using bilinear interpolation.

    Uses F.grid_sample (pure PyTorch, no CUDA custom ops).

    Args:
        patch_features: (B, N_cam, N_patches, D) — reshaped to spatial grid.
        ref_pts_norm: (B, N_pts, N_cam, 2) normalised UV in [-1, 1].
        patch_h: Spatial height of the patch grid (sqrt(N_patches) for ViTs).
        patch_w: Spatial width of the patch grid.

    Returns:
        sampled: (B, N_pts, N_cam, D) sampled features.
    """
    B, N_cam, N_p, D = patch_features.shape
    assert N_p == patch_h * patch_w, f"N_patches {N_p} != {patch_h}*{patch_w}"

    # Reshape to spatial: (B*N_cam, D, patch_h, patch_w)
    feat_spatial = patch_features.view(B * N_cam, patch_h, patch_w, D)
    feat_spatial = feat_spatial.permute(0, 3, 1, 2).contiguous()

    # Reshape ref_pts: (B*N_cam, N_pts, 1, 2)
    N_pts = ref_pts_norm.shape[1]
    ref = ref_pts_norm.permute(0, 2, 1, 3)  # (B, N_cam, N_pts, 2)
    ref = ref.reshape(B * N_cam, N_pts, 1, 2)

    # Sample: (B*N_cam, D, N_pts, 1)
    sampled = F.grid_sample(
        feat_spatial, ref,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (B*N_cam, D, N_pts, 1)

    sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B*N_cam, N_pts, D)
    return sampled.view(B, N_cam, N_pts, D).permute(0, 2, 1, 3)  # (B, N_pts, N_cam, D)
