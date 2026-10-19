"""TrackCropExtractor: project 3D bbox onto camera images and pool patch features.

No re-encoding — pools SigLIP2 patch features already computed by the
backbone. ~0 extra FLOPs after backbone forward pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class TrackCropExtractor:
    """Extract per-track crop embeddings from pre-computed patch features.

    For each track record, projects its 3D bounding box corners onto each
    camera image, then mean-pools the SigLIP2 patch features that fall
    within the projected bounding box.

    Args:
        av2_dataset: AV2Dataset instance (for calibration).
        log_id: AV2 log ID.
    """

    def __init__(self, av2_dataset: Any, log_id: str) -> None:
        self.av2_dataset = av2_dataset
        self.log_id = log_id
        self._calibs_cache: dict[str, Any] | None = None

    def extract(
        self,
        rec: Any,
        patch_features: torch.Tensor,  # (N_cam, N_patches, D)
        timestamp_ns: int,
        patch_size: int = 16,
    ) -> tuple[list[torch.Tensor | None], list[float]]:
        """Extract crop embeddings for a single TrackRecord.

        Args:
            rec: TrackRecord for the track.
            patch_features: (N_cam, N_patches, D) patch features from backbone.
            timestamp_ns: Current timestamp.
            patch_size: ViT patch size in pixels (default 16 for ViT-B/16).

        Returns:
            Tuple of:
            - crop_embeddings: list of N_cam tensors (D,) or None
            - visibility_scores: list of N_cam floats [0,1]
        """
        calibs = self._get_calibs()
        N_cam = patch_features.shape[0]
        N_patches = patch_features.shape[1]
        patch_h = patch_w = int(N_patches ** 0.5)

        crop_embeddings: list[torch.Tensor | None] = []
        visibility_scores: list[float] = []

        for cam_idx, (cam_name, calib) in enumerate(calibs.items()):
            if cam_idx >= N_cam:
                break

            # Get image dimensions
            img_h = calib.image_height
            img_w = calib.image_width

            # Project 3D bbox corners onto this camera
            uv_pts, depths = self._project_bbox(rec, calib)
            if uv_pts is None or depths is None or depths.min() < 0.1:
                crop_embeddings.append(None)
                visibility_scores.append(0.0)
                continue

            # Compute 2D bbox in pixel space
            u_min, v_min = uv_pts[:, 0].min(), uv_pts[:, 1].min()
            u_max, v_max = uv_pts[:, 0].max(), uv_pts[:, 1].max()

            # Clip to image bounds
            u_min = max(0.0, float(u_min))
            v_min = max(0.0, float(v_min))
            u_max = min(float(img_w - 1), float(u_max))
            v_max = min(float(img_h - 1), float(v_max))

            if u_max <= u_min or v_max <= v_min:
                crop_embeddings.append(None)
                visibility_scores.append(0.0)
                continue

            # Compute visibility score (fraction of bbox in image)
            bbox_area = (u_max - u_min) * (v_max - v_min)
            full_area = (
                (uv_pts[:, 0].max() - uv_pts[:, 0].min()) *
                (uv_pts[:, 1].max() - uv_pts[:, 1].min())
            )
            vis = float(bbox_area / (full_area + 1e-6))
            vis = min(1.0, max(0.0, vis))

            # Map pixel bbox to patch grid indices
            scale_h = patch_h / img_h
            scale_w = patch_w / img_w
            pi_min = int(v_min * scale_h)
            pi_max = min(patch_h, int(v_max * scale_h) + 1)
            pj_min = int(u_min * scale_w)
            pj_max = min(patch_w, int(u_max * scale_w) + 1)

            if pi_max <= pi_min or pj_max <= pj_min:
                crop_embeddings.append(None)
                visibility_scores.append(0.0)
                continue

            # Reshape patch features to (patch_h, patch_w, D)
            feats_2d = patch_features[cam_idx].view(patch_h, patch_w, -1)
            crop = feats_2d[pi_min:pi_max, pj_min:pj_max, :]  # (rh, rw, D)
            crop_emb = crop.reshape(-1, crop.shape[-1]).mean(dim=0)  # (D,)
            crop_emb = F.normalize(crop_emb, dim=0)

            crop_embeddings.append(crop_emb)
            visibility_scores.append(vis)

        # Pad to N_cam if calibs has fewer cameras
        while len(crop_embeddings) < N_cam:
            crop_embeddings.append(None)
            visibility_scores.append(0.0)

        return crop_embeddings, visibility_scores

    def _project_bbox(
        self, rec: Any, calib: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Project 3D bbox corners into camera image coordinates.

        Returns:
            uv_pts: (8, 2) pixel coordinates of 8 bbox corners.
            depths: (8,) depth values (positive = in front of camera).
        """
        corners_3d = _get_3d_corners(rec)  # (8, 3) in ego frame
        # Homogeneous: (8, 4)
        corners_h = np.concatenate(
            [corners_3d, np.ones((8, 1))], axis=-1
        )

        # Transform to camera frame
        ego2cam = calib.extrinsics_ego2cam  # (4, 4)
        corners_cam = (ego2cam @ corners_h.T).T  # (8, 4)
        depths = corners_cam[:, 2]

        if depths.max() < 0.1:
            return None, None

        # Project with intrinsics
        K = calib.intrinsics  # (3, 3)
        xyz = corners_cam[:, :3]  # (8, 3)
        uv_h = (K @ xyz.T).T  # (8, 3)
        uv = uv_h[:, :2] / (uv_h[:, 2:3] + 1e-6)

        return uv, depths

    def _get_calibs(self) -> dict[str, Any]:
        if self._calibs_cache is None:
            self._calibs_cache = self.av2_dataset.get_all_calibrations(self.log_id)
        return self._calibs_cache


def _get_3d_corners(rec: Any) -> np.ndarray:
    """Compute 8 corners of a 3D bounding box from a TrackRecord.

    Returns:
        corners: (8, 3) in the ego-vehicle coordinate frame.
    """
    l, w, h = rec.size[0] / 2, rec.size[1] / 2, rec.size[2] / 2
    yaw = rec.yaw
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # 8 corners in object frame: ±l, ±w, ±h
    local = np.array([
        [ l,  w,  h], [ l, -w,  h], [-l, -w,  h], [-l,  w,  h],
        [ l,  w, -h], [ l, -w, -h], [-l, -w, -h], [-l,  w, -h],
    ])

    # Rotation matrix (yaw only — z-axis rotation)
    R = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0,      0,     1],
    ])

    return (R @ local.T).T + rec.translation_m
