"""Base VLM wrapper interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseVLM(nn.Module, ABC):
    """Abstract VLM backend.

    Subclasses wrap Qwen2.5-VL, LLaMA-3.2-Vision, etc. They expose:
    - ``forward``: inject visual tokens + text → hidden states
    - ``generate``: produce text captions
    - ``d_model``: hidden state dimension
    """

    @property
    @abstractmethod
    def d_model(self) -> int:
        """Hidden state dimension of this VLM."""
        ...

    @abstractmethod
    def forward(
        self,
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inject visual tokens and run forward pass.

        Args:
            visual_tokens: (B, N_vis, d_vlm) from Q-Former.
            input_ids: (B, L_text) tokenised text.
            attention_mask: (B, N_vis + L_text) optional mask.

        Returns:
            hidden_states: (B, N_vis + L_text, d_vlm)
        """
        ...

    @abstractmethod
    def generate(
        self,
        visual_tokens: torch.Tensor,
        prompts: list[str],
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Generate text captions conditioned on visual tokens.

        Args:
            visual_tokens: (B, N_vis, d_vlm).
            prompts: List of B text prompts.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            List of B generated strings.
        """
        ...
