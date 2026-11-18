"""SyntheticDataset — random SceneMiner-compatible batches, no real data.

Used for CPU smoke-testing and end-to-end training demos when no AV2 /
nuScenes data is available.  Each sample is reproducibly seeded from its
index, so results are stable across DataLoader restarts.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """Generates random (images, calibration, labels) matching AV2Dataset format.

    ``__getitem__`` returns a dict that a standard ``DataLoader`` will collate
    into the batch format expected by ``SceneMiner.forward``:

    .. code-block:: python

        {
            "images":     (N_cam, 3, H, W),   # float32
            "intrinsics": (N_cam, 3, 3),       # float32 identity + focal noise
            "extrinsics": (N_cam, 4, 4),       # float32 identity
            "image_hw":   (H, W),              # tuple[int, int]
            "tags":       (n_tags,),           # float32 multi-hot
            "risk":       (),                  # float32 scalar in [0, 1]
            "input_ids":  (seq_len,),          # long  dummy text tokens
        }

    Args:
        n_samples: Number of synthetic samples.
        n_cams: Cameras per sample (7 for AV2, 6 for nuScenes).
        image_h, image_w: Spatial resolution (use small values for CPU).
        n_tags: Tag-head output size.
        seq_len: Text token sequence length fed to the VLM.
        vocab_size: Token vocabulary ceiling for random ``input_ids``.
    """

    def __init__(
        self,
        n_samples: int = 16,
        n_cams: int = 7,
        image_h: int = 64,
        image_w: int = 64,
        n_tags: int = 20,
        seq_len: int = 8,
        vocab_size: int = 512,
    ) -> None:
        self.n_samples = n_samples
        self.n_cams = n_cams
        self.image_h = image_h
        self.image_w = image_w
        self.n_tags = n_tags
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator()
        g.manual_seed(idx)

        # Slightly randomised intrinsics (focal ≈ 500 px, principal = centre)
        fx = 500.0 + torch.randn(1, generator=g).item() * 10
        intrinsics = torch.eye(3).unsqueeze(0).repeat(self.n_cams, 1, 1)
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fx
        intrinsics[:, 0, 2] = self.image_w / 2
        intrinsics[:, 1, 2] = self.image_h / 2

        return {
            "images": torch.randn(self.n_cams, 3, self.image_h, self.image_w, generator=g),
            "intrinsics": intrinsics,
            "extrinsics": torch.eye(4).unsqueeze(0).repeat(self.n_cams, 1, 1),
            "image_hw": (self.image_h, self.image_w),
            "tags": (torch.rand(self.n_tags, generator=g) > 0.8).float(),
            "risk": torch.rand(1, generator=g).squeeze(),
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,), generator=g),
        }
