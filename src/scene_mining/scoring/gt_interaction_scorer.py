"""GTInteractionScorer — ground-truth interaction difficulty from nuScenes annotations.

Provides an alternative difficulty signal based on object counts, agent
proximity, and annotation-derived risk, usable for evaluation and ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class GTInteractionScorer:
    """Score scene difficulty from ground-truth nuScenes annotations.

    Args:
        proximity_threshold_m: Distance (metres) within which two agents
            are considered in close proximity.
        vru_categories: nuScenes annotation categories considered as
            Vulnerable Road Users.
        w_density: Weight for agent density component.
        w_proximity: Weight for close-interaction component.
        w_vru: Weight for VRU presence component.
    """

    proximity_threshold_m: float = 5.0
    vru_categories: list[str] = field(default_factory=lambda: [
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.wheelchair",
        "human.pedestrian.stroller",
        "vehicle.bicycle",
        "vehicle.motorcycle",
    ])
    w_density: float = 0.4
    w_proximity: float = 0.4
    w_vru: float = 0.2

    def __post_init__(self) -> None:
        total = self.w_density + self.w_proximity + self.w_vru
        if total <= 0:
            raise ValueError("At least one weight must be positive.")
        self.w_density /= total
        self.w_proximity /= total
        self.w_vru /= total

    def score_from_annotations(
        self,
        agent_positions: torch.Tensor,          # (N, 2) xy in ego frame
        agent_categories: list[str],
        ego_speed_ms: float = 0.0,
        max_agents_norm: float = 20.0,
    ) -> float:
        """Compute a single difficulty score in [0, 1] from annotation data.

        Args:
            agent_positions: (N, 2) tensor of agent xy positions in ego frame.
            agent_categories: List of N nuScenes category strings.
            ego_speed_ms: Ego vehicle speed in m/s (unused currently, reserved).
            max_agents_norm: Agent count normalisation ceiling.

        Returns:
            Scalar float difficulty score in [0, 1].
        """
        N = agent_positions.shape[0]
        if N == 0:
            return 0.0

        # ── Density component ─────────────────────────────────────────────
        density = min(N / max_agents_norm, 1.0)

        # ── Proximity component ───────────────────────────────────────────
        # Fraction of agents within proximity_threshold of ego (at origin)
        dists = torch.norm(agent_positions, dim=1)  # (N,)
        n_close = (dists < self.proximity_threshold_m).sum().item()
        proximity = min(n_close / max(N, 1), 1.0)

        # ── VRU component ─────────────────────────────────────────────────
        n_vru = sum(1 for c in agent_categories if c in self.vru_categories)
        vru = min(n_vru / max(N, 1), 1.0)

        score = self.w_density * density + self.w_proximity * proximity + self.w_vru * vru
        return float(min(max(score, 0.0), 1.0))
