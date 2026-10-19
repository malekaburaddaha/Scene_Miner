"""SceneDifficultyScorer — combines RiskHead output, tag complexity,
and motion uncertainty into a single difficulty score per scene.

This is the BASELINE scorer (Option B): RiskHead uses BEV features only.
Option A (BEV + CLIP language features in RiskHead) is future work.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch


class SceneDifficultyScorer:
    """Weighted combination of risk, tag complexity, and motion uncertainty.

    Args:
        w1: Weight for risk_score (from RiskHead sigmoid output).
        w2: Weight for tag_complexity (mean sigmoid of danger tag subset).
        w3: Weight for motion_uncertainty (from U-Net ensemble).
        danger_tag_indices: Indices of tags considered dangerous.
            Defaults to 0–9 (the first 10 semantically labelled tags).
        uncertainty_scores_path: Optional path to JSON file mapping
            scene_token → normalised uncertainty float [0, 1].
            If None or missing, motion uncertainty is omitted from scoring.

    Weights are auto-normalised so they always sum to 1.
    """

    def __init__(
        self,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
        danger_tag_indices: list[int] | None = None,
        uncertainty_scores_path: str | Path | None = None,
    ) -> None:
        self.danger_tag_indices = danger_tag_indices if danger_tag_indices is not None \
            else list(range(10))

        # Normalise weights so they sum to 1
        total = w1 + w2 + w3
        if total <= 0:
            raise ValueError("At least one weight must be positive.")
        self.w1 = w1 / total
        self.w2 = w2 / total
        self.w3 = w3 / total

        # Load uncertainty lookup table
        self.uncertainty_lookup: dict[str, float] | None = None
        if uncertainty_scores_path is not None:
            unc_path = Path(uncertainty_scores_path)
            if unc_path.exists():
                with open(unc_path) as f:
                    self.uncertainty_lookup = json.load(f)
            else:
                warnings.warn(
                    f"Uncertainty scores file not found: {unc_path}. "
                    "Motion uncertainty will be set to 0.0.",
                    stacklevel=2,
                )

    def compute_tag_complexity(self, tag_logits: torch.Tensor) -> torch.Tensor:
        """Mean sigmoid activation over danger tags.

        Args:
            tag_logits: (B, n_tags) raw logits from TaggingHead.

        Returns:
            (B, 1) float tensor in [0, 1].
        """
        danger = tag_logits[:, self.danger_tag_indices]  # (B, K)
        return torch.sigmoid(danger).mean(dim=1, keepdim=True)  # (B, 1)

    def lookup_uncertainty(self, scene_tokens: list[str]) -> torch.Tensor:
        """Fetch motion uncertainty for a batch of scene tokens.

        Args:
            scene_tokens: List of B scene token strings.

        Returns:
            (B, 1) float tensor, 0.0 for unknown tokens.
        """
        vals = [
            self.uncertainty_lookup.get(t, 0.0)
            if self.uncertainty_lookup is not None else 0.0
            for t in scene_tokens
        ]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(1)

    def score(
        self,
        risk_score: torch.Tensor,
        tag_logits: torch.Tensor,
        scene_tokens: list[str] | None = None,
    ) -> torch.Tensor:
        """Compute per-scene difficulty score.

        Args:
            risk_score: (B, 1) sigmoid risk output from RiskHead.
            tag_logits: (B, n_tags) raw logits from TaggingHead.
            scene_tokens: Optional list of B scene token strings for
                uncertainty lookup. Required when w3 > 0.

        Returns:
            (B, 1) difficulty score clamped to [0, 1].
        """
        tag_complexity = self.compute_tag_complexity(tag_logits)  # (B, 1)

        # Ensure risk_score is (B, 1)
        if risk_score.dim() == 1:
            risk_score = risk_score.unsqueeze(1)

        # Move tag_complexity to same device as risk_score
        tag_complexity = tag_complexity.to(risk_score.device)

        has_uncertainty = (
            self.uncertainty_lookup is not None
            and scene_tokens is not None
            and self.w3 > 0
        )

        if has_uncertainty:
            motion_unc = self.lookup_uncertainty(scene_tokens).to(risk_score.device)
            s = self.w1 * risk_score + self.w2 * tag_complexity + self.w3 * motion_unc
        else:
            # Renormalise to w1 + w2 only
            denom = self.w1 + self.w2
            if denom <= 0:
                denom = 1.0
            s = (self.w1 * risk_score + self.w2 * tag_complexity) / denom

        return s.clamp(0.0, 1.0)
