"""Retrieval evaluation metrics for scene embedding space.

Functions:
- compute_map_at_k: Mean Average Precision @ k
- compute_recall_at_k: Recall @ k

Both functions operate on pre-computed embedding matrices and string labels.
Two embeddings share a label if and only if they are considered a positive
pair (e.g. same scenario class, same query description).

Usage::

    import torch
    from scene_mining.evaluation.retrieval import compute_map_at_k, compute_recall_at_k

    embeddings = torch.randn(100, 256)   # [N, D]
    labels = ["a"] * 10 + ["b"] * 90    # same label = positive pair

    map10  = compute_map_at_k(embeddings, labels, k=10)
    rec10  = compute_recall_at_k(embeddings, labels, k=10)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_map_at_k(
    embeddings: torch.Tensor,
    labels: list[str],
    k: int = 10,
) -> float:
    """Compute Mean Average Precision @ k (mAP@k).

    For each query, rank all other items by cosine similarity, compute
    Average Precision over the top-k retrieved items, then average over
    all queries.

    Args:
        embeddings: ``[N, D]`` float tensor of L2-normalised embeddings.
            Will be L2-normalised internally if not already.
        labels: Length-N list of label strings.  Items with the same
            label are positive pairs.
        k: Number of top retrieved items to consider.

    Returns:
        mAP@k as a float in [0, 1].
    """
    embeddings = F.normalize(embeddings.float(), dim=1)
    N = embeddings.shape[0]
    if N < 2:
        return 0.0

    # Pairwise cosine similarity matrix [N, N]
    sim = embeddings @ embeddings.T  # [N, N]

    avg_precisions = []

    for i in range(N):
        # Mask out self
        scores = sim[i].clone()
        scores[i] = -2.0  # exclude self

        # Sort descending
        sorted_indices = scores.argsort(descending=True)
        top_k_indices = sorted_indices[:k]

        # Relevance: 1 if same label, else 0
        n_pos_total = sum(1 for j, lbl in enumerate(labels) if lbl == labels[i] and j != i)
        if n_pos_total == 0:
            continue

        n_hits = 0
        precisions = []
        for rank, idx in enumerate(top_k_indices.tolist(), start=1):
            if labels[idx] == labels[i]:
                n_hits += 1
                precisions.append(n_hits / rank)

        ap = sum(precisions) / min(n_pos_total, k) if precisions else 0.0
        avg_precisions.append(ap)

    return float(sum(avg_precisions) / len(avg_precisions)) if avg_precisions else 0.0


def compute_recall_at_k(
    embeddings: torch.Tensor,
    labels: list[str],
    k: int = 10,
) -> float:
    """Compute Recall @ k (R@k).

    For each query, compute the fraction of its positives that appear
    in the top-k retrieved items, then average over all queries.

    Args:
        embeddings: ``[N, D]`` float tensor. Will be L2-normalised internally.
        labels: Length-N list of label strings.
        k: Number of top retrieved items to consider.

    Returns:
        R@k as a float in [0, 1].
    """
    embeddings = F.normalize(embeddings.float(), dim=1)
    N = embeddings.shape[0]
    if N < 2:
        return 0.0

    sim = embeddings @ embeddings.T  # [N, N]
    recalls = []

    for i in range(N):
        scores = sim[i].clone()
        scores[i] = -2.0

        top_k_labels = [labels[j] for j in scores.argsort(descending=True)[:k].tolist()]
        n_pos_total = sum(1 for j, lbl in enumerate(labels) if lbl == labels[i] and j != i)
        if n_pos_total == 0:
            continue

        n_retrieved = sum(1 for lbl in top_k_labels if lbl == labels[i])
        recalls.append(n_retrieved / min(n_pos_total, k))

    return float(sum(recalls) / len(recalls)) if recalls else 0.0
