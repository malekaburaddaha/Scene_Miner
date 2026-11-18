#!/usr/bin/env python3
"""Domain-balancing samplers for unified multi-dataset training.

Problem: nuScenes (~28 k keyframes) and our Waymo subset (hundreds→
thousands of frames) differ in size by orders of magnitude.  A plain
``ConcatDataset`` + shuffle would draw samples in proportion to size, so
the model would see almost only the larger domain and overfit to it —
the smaller domain gets starved.

``MultinomialDatasetSampler`` decouples sampling probability from dataset
size: you specify a target mixture (e.g. 60 % nuScenes / 40 % Waymo) and
it emits global indices into a ``ConcatDataset`` that realise that
mixture every epoch, sampling WITH replacement within each domain so the
smaller domain is effectively up-sampled to hit its quota.
"""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class MultinomialDatasetSampler(Sampler[int]):
    """Yield ``ConcatDataset`` indices with a fixed per-domain mixture.

    Args:
        dataset_sizes: length of each constituent dataset, in the SAME
            order they were passed to ``ConcatDataset``.
        weights: target sampling probability per dataset (need not sum to
            1 — normalised internally).  e.g. ``[0.6, 0.4]``.
        num_samples: indices emitted per epoch.  Defaults to the total
            size of all datasets (one "virtual epoch").
        seed: base RNG seed; the epoch counter is mixed in so successive
            epochs differ while staying reproducible.
        replacement: sample within a chosen domain WITH replacement
            (default True) — required to up-sample the minority domain
            beyond its real size.

    The two-stage draw per index:
        1. pick a domain  d ~ Multinomial(normalised weights)
        2. pick a local index uniformly in [0, size_d)
        3. map to the global index  offset_d + local
    """

    def __init__(self, dataset_sizes, weights, num_samples=None,
                 seed: int = 0, replacement: bool = True):
        if len(dataset_sizes) != len(weights):
            raise ValueError("dataset_sizes and weights must align in length")
        if any(s <= 0 for s in dataset_sizes):
            raise ValueError(f"every dataset must be non-empty: {dataset_sizes}")

        self.dataset_sizes = list(dataset_sizes)
        w = torch.tensor(weights, dtype=torch.double)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"weights must be non-negative & sum>0: {weights}")
        self.weights = w / w.sum()                      # normalised mixture
        self.num_samples = int(num_samples) if num_samples is not None \
            else int(sum(dataset_sizes))
        self.seed = int(seed)
        self.replacement = replacement
        self._epoch = 0

        # Global-index offset of each dataset inside the ConcatDataset.
        self.offsets = [0]
        for s in self.dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

    def set_epoch(self, epoch: int) -> None:
        """Mirror DistributedSampler so reshuffling differs per epoch."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)

        # Stage 1: how many samples each domain contributes this epoch.
        domain_choices = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=g)

        indices = []
        for d in range(len(self.dataset_sizes)):
            n_d = int((domain_choices == d).sum().item())
            if n_d == 0:
                continue
            size_d = self.dataset_sizes[d]
            if self.replacement:
                local = torch.randint(0, size_d, (n_d,), generator=g)
            else:
                # Tile a permutation if we need more than the domain has.
                reps = (n_d + size_d - 1) // size_d
                local = torch.cat([torch.randperm(size_d, generator=g)
                                   for _ in range(reps)])[:n_d]
            indices.append(local + self.offsets[d])

        all_idx = torch.cat(indices)
        # Shuffle so domains are interleaved within the epoch, not blocked.
        all_idx = all_idx[torch.randperm(all_idx.numel(), generator=g)]
        return iter(all_idx.tolist())


class DistributedMultinomialSampler(Sampler[int]):
    """DDP-aware weighted interleaving sampler.

    Plain ``MultinomialDatasetSampler`` under DDP is WRONG: every rank
    would draw the same global stream, so 4 GPUs train on 4 identical
    copies of the data (no speed-up, inflated effective epoch).  The
    fix is the same contract ``DistributedSampler`` provides — each rank
    must emit a DISJOINT shard — but here the shard is drawn from the
    weighted mixture rather than a flat permutation.

    Strategy: each rank runs its OWN independent multinomial draw of
    ``num_samples // world_size`` indices, seeded by
    ``seed + epoch*world_size + rank``.  Because each rank's draw
    individually realises the target mixture, their union does too, and
    distinct per-rank seeds make the shards (statistically) disjoint —
    which is all DDP needs, since with-replacement sampling already
    implies repeats exist within the global stream anyway.

    Args mirror ``MultinomialDatasetSampler`` plus:
        num_replicas: world size (defaults to dist.get_world_size()).
        rank:         this process's rank (defaults to dist.get_rank()).
        num_samples:  TOTAL indices across all ranks per epoch; each rank
                      emits ``num_samples // num_replicas``.
    """

    def __init__(self, dataset_sizes, weights, num_samples=None,
                 num_replicas: int | None = None, rank: int | None = None,
                 seed: int = 0, replacement: bool = True):
        import torch.distributed as dist
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset_sizes = list(dataset_sizes)
        w = torch.tensor(weights, dtype=torch.double)
        self.weights = w / w.sum()
        total = int(num_samples) if num_samples is not None else int(sum(dataset_sizes))
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        # Per-rank quota (rounded so every rank emits the same count → no
        # DDP hang from uneven batch counts across ranks).
        self.num_samples = total // self.num_replicas
        self.seed = int(seed)
        self.replacement = replacement
        self._epoch = 0

        self.offsets = [0]
        for s in self.dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        # Distinct seed per (epoch, rank) → disjoint shards that each
        # realise the mixture.
        g.manual_seed(self.seed + self._epoch * self.num_replicas + self.rank)

        domain_choices = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=g)
        indices = []
        for d in range(len(self.dataset_sizes)):
            n_d = int((domain_choices == d).sum().item())
            if n_d == 0:
                continue
            size_d = self.dataset_sizes[d]
            if self.replacement:
                local = torch.randint(0, size_d, (n_d,), generator=g)
            else:
                reps = (n_d + size_d - 1) // size_d
                local = torch.cat([torch.randperm(size_d, generator=g)
                                   for _ in range(reps)])[:n_d]
            indices.append(local + self.offsets[d])
        all_idx = torch.cat(indices)
        all_idx = all_idx[torch.randperm(all_idx.numel(), generator=g)]
        return iter(all_idx.tolist())
