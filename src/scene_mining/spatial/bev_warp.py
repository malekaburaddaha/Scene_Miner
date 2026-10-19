"""Ego-motion compensation for BEV feature warping.

When stacking BEV features from multiple time steps, each past BEV was computed
in the ego frame of *its own* timestamp.  Between t-1 and t the ego vehicle
moves a few metres and rotates a few degrees, so the same world point sits at
different BEV cells in the two grids.  Stacking them naively would smear motion
features across cells that don't correspond to the same physical location.

This module computes the rigid 2-D xy-plane transform from one ego frame to
another and uses ``F.grid_sample`` to resample the past BEV grid into the
current ego frame.  Bilinear sampling, ``padding_mode="zeros"`` so cells that
fall outside the past grid (because the ego vehicle moved forward and reveals
new territory in front) are filled with zeros.

The transform stays inside ``torch`` so it's fully differentiable and runs
on the GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _build_normalized_grid(
    bev_h: int, bev_w: int, pc_range: list[float],
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Return BEV cell centres in metres, shape (bev_h, bev_w, 2)  -> (x, y).

    Convention: ``bev_w`` columns index the x axis (ego-forward in BEVFormer
    convention is +x at the centre), ``bev_h`` rows index the y axis.  Cell
    (0, 0) is the (x_min, y_min) corner.
    """
    x_min, y_min, _, x_max, y_max, _ = pc_range
    xs = torch.linspace(
        x_min + (x_max - x_min) / (2 * bev_w),
        x_max - (x_max - x_min) / (2 * bev_w),
        bev_w, device=device, dtype=dtype,
    )                                       # (bev_w,)
    ys = torch.linspace(
        y_min + (y_max - y_min) / (2 * bev_h),
        y_max - (y_max - y_min) / (2 * bev_h),
        bev_h, device=device, dtype=dtype,
    )                                       # (bev_h,)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (bev_h, bev_w)
    return torch.stack([xx, yy], dim=-1)            # (bev_h, bev_w, 2)


def warp_bev(
    past_bev: torch.Tensor,           # (B, D, bev_h, bev_w) channels-first
    ego_past_world: torch.Tensor,     # (B, 4, 4) past ego → world (homogeneous)
    ego_current_world: torch.Tensor,  # (B, 4, 4) current ego → world
    pc_range: list[float],
) -> torch.Tensor:
    """Resample ``past_bev`` into the *current* ego frame.

    Returns a tensor of the same shape as ``past_bev`` whose cell at
    current-ego coordinate ``(x, y)`` holds the past BEV value at that same
    world point.
    """
    B, D, bev_h, bev_w = past_bev.shape
    device, dtype = past_bev.device, past_bev.dtype

    # Output grid: BEV cell centres in CURRENT ego coords (metres).
    grid_xy = _build_normalized_grid(bev_h, bev_w, pc_range, device, dtype)  # (H, W, 2)
    grid_xy = grid_xy.unsqueeze(0).expand(B, -1, -1, -1)                     # (B, H, W, 2)

    # We want, for each output cell (in current ego coords), the corresponding
    # location in past ego coords:
    #   p_past = T_world_to_past @ T_current_to_world @ p_current
    #          = inv(T_past_world) @ T_current_world @ p_current
    # Extract the 2D rigid part (x, y, yaw).
    T_current_to_past = torch.linalg.solve(ego_past_world, ego_current_world)  # (B, 4, 4)
    R = T_current_to_past[:, :2, :2]                                            # (B, 2, 2)
    t = T_current_to_past[:, :2, 3]                                             # (B, 2)

    # Apply: past_xy = R @ current_xy + t.  Use einsum for clean batched matmul.
    past_xy = torch.einsum("bij,bhwj->bhwi", R, grid_xy) + t[:, None, None, :]  # (B, H, W, 2)

    # Normalise to [-1, 1] for grid_sample.  pc_range gives the metric extent.
    x_min, y_min, _, x_max, y_max, _ = pc_range
    span_x = x_max - x_min
    span_y = y_max - y_min
    u = (past_xy[..., 0] - x_min) / span_x * 2 - 1                          # x → width axis
    v = (past_xy[..., 1] - y_min) / span_y * 2 - 1                          # y → height axis
    sample_grid = torch.stack([u, v], dim=-1)                                # (B, H, W, 2)

    # grid_sample expects (B, C, H, W) input and (B, H_out, W_out, 2) grid,
    # last dim being (x_norm, y_norm) where x is column (width).
    warped = F.grid_sample(
        past_bev, sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )                                                                        # (B, D, H, W)
    return warped


def stack_temporal_bev(
    bev_per_frame: list[torch.Tensor],          # each (B, D, bev_h, bev_w), oldest first
    ego_poses: torch.Tensor,                    # (B, T, 4, 4), oldest first
    pc_range: list[float],
) -> torch.Tensor:
    """Warp every past BEV into the current ego frame and concat along channels.

    The *last* entry in ``bev_per_frame`` is the current frame and is left
    untouched.  Earlier entries are warped using the corresponding
    ``ego_poses[:, t]`` → ``ego_poses[:, -1]`` transform.

    Returns:
        (B, T * D, bev_h, bev_w) channels-first stacked tensor.
    """
    T = len(bev_per_frame)
    assert ego_poses.shape[1] == T, f"ego_poses time dim {ego_poses.shape[1]} != bev list {T}"
    current_pose = ego_poses[:, -1]   # (B, 4, 4)
    out: list[torch.Tensor] = []
    for t in range(T):
        if t == T - 1:
            out.append(bev_per_frame[t])
        else:
            out.append(warp_bev(
                bev_per_frame[t],
                ego_past_world=ego_poses[:, t],
                ego_current_world=current_pose,
                pc_range=pc_range,
            ))
    return torch.cat(out, dim=1)      # (B, T*D, H, W)
