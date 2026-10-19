"""Q-Former adapter — uses HuggingFace BLIP-2's Blip2QFormerModel.

Reuses the production BLIP-2 Q-Former from ``transformers`` instead of
reimplementing it. Only addition: a projection head that maps the Q-Former
output to the target VLM's embedding dimension.

Supports three input streams (concatenated before cross-attention):
  1. BEV spatial tokens  (from BEVFormer / TPVFormer)
  2. Camera features     (from SigLIP2 / DINOv2 backbone)  — optional
  3. Motion tokens       (from MotionHead transformer)      — optional

When only spatial_tokens are provided, behaviour is identical to the
original single-stream Q-Former.

Reference: BLIP-2 (Li et al., 2023) — transformers.Blip2QFormerModel.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QFormer(nn.Module):
    """BLIP-2 Q-Former adapter — 3-stream fusion between scene features and a VLM.

    Compresses multi-source scene tokens (BEV, camera, motion) to
    ``n_query_tokens`` visual tokens in the VLM input space.

    The three input streams capture complementary information:
      - BEV tokens: spatial layout — *where* things are.
      - Camera features: fine-grained visual detail — texture, pose, signals.
      - Motion tokens: agent dynamics — *how* things are moving.

    All streams are projected to a shared encoder_hidden_dim, concatenated
    along the sequence dimension, and fed to the Q-Former's cross-attention.
    The learned query tokens attend to all streams simultaneously.

    Args:
        d_spatial: BEV/TPV spatial token dimension (from spatial encoder).
        d_camera: Camera patch feature dimension (from backbone).
            Set to 0 or None to disable the camera stream.
        d_motion: Motion token dimension (from MotionHead).
            Set to 0 or None to disable the motion stream.
        d_vlm: VLM hidden dimension (output projection target).
        n_query_tokens: Learned query count (default 32, same as BLIP-2).
        pretrained_name: HuggingFace BLIP-2 model ID to initialise weights
            from. ``None`` initialises the Q-Former from scratch.
    """

    def __init__(
        self,
        d_spatial: int = 256,
        d_camera: int | None = None,
        d_motion: int | None = None,
        d_vlm: int = 3584,
        n_query_tokens: int = 32,
        pretrained_name: str | None = "Salesforce/blip2-opt-2.7b",
    ) -> None:
        super().__init__()
        self.n_query_tokens = n_query_tokens
        self.d_spatial = d_spatial
        self.d_camera = d_camera or 0
        self.d_motion = d_motion or 0
        self.d_vlm = d_vlm

        # Load/init BLIP-2 Q-Former
        # qformer_d        = Q-Former self-attention hidden size (768)
        # encoder_hidden_d = size expected by cross-attention key/value (1408 for blip2-opt-2.7b)
        self._qformer, qformer_d, encoder_hidden_d = _load_blip2_qformer(
            pretrained_name, n_query_tokens
        )
        self._encoder_hidden_d = encoder_hidden_d

        # Learned query embeddings (analogous to BLIP-2's query_tokens)
        self.query_tokens = nn.Parameter(torch.randn(1, n_query_tokens, qformer_d) * 0.02)

        # ── Stream 1: BEV spatial tokens (always present) ────────────
        # Project to encoder_hidden_d (NOT qformer_d) because the
        # cross-attention key/value weights have in_features=encoder_hidden_d.
        self.spatial_proj = (
            nn.Linear(d_spatial, encoder_hidden_d)
            if d_spatial != encoder_hidden_d else nn.Identity()
        )

        # ── Stream 2: Camera features (optional) ─────────────────────
        if self.d_camera > 0:
            self.camera_proj = nn.Sequential(
                nn.Linear(self.d_camera, encoder_hidden_d),
                nn.LayerNorm(encoder_hidden_d),
            )
        else:
            self.camera_proj = None

        # ── Stream 3: Motion tokens (optional) ───────────────────────
        if self.d_motion > 0:
            self.motion_proj = nn.Sequential(
                nn.Linear(self.d_motion, encoder_hidden_d),
                nn.LayerNorm(encoder_hidden_d),
            )
        else:
            self.motion_proj = None

        # Project Q-Former output → VLM input dimension
        self.output_proj = nn.Sequential(
            nn.Linear(qformer_d, d_vlm),
            nn.LayerNorm(d_vlm),
        )

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        camera_features: torch.Tensor | None = None,
        motion_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress multi-stream scene tokens to VLM-compatible visual tokens.

        Args:
            spatial_tokens: (B, N_spatial, d_spatial) BEV or TPV tokens.
            camera_features: (B, N_cam_patches, d_camera) backbone patch features.
                Typically N_cam_patches = N_cam × N_patches_per_cam.
                Pass None to skip this stream.
            motion_tokens: (B, N_agents, d_motion) MotionHead decoded tokens.
                Pass None to skip this stream.

        Returns:
            visual_tokens: (B, n_query_tokens, d_vlm)
        """
        B = spatial_tokens.shape[0]

        # ── Build the encoder hidden states from all available streams ──
        # Each stream is projected to encoder_hidden_d, then concatenated
        # along the sequence dimension (dim=1).
        streams = []

        # Stream 1: BEV spatial (always)
        streams.append(self.spatial_proj(spatial_tokens))     # (B, N_s, enc_d)

        # Stream 2: Camera features (optional)
        if camera_features is not None and self.camera_proj is not None:
            streams.append(self.camera_proj(camera_features))  # (B, N_c, enc_d)

        # Stream 3: Motion tokens (optional)
        if motion_tokens is not None and self.motion_proj is not None:
            streams.append(self.motion_proj(motion_tokens))    # (B, N_m, enc_d)

        # Concatenate: (B, N_s + N_c + N_m, encoder_hidden_d)
        encoder_hidden = torch.cat(streams, dim=1)

        # ── Q-Former cross-attention ───────────────────────────────────
        query_embeds = self.query_tokens.expand(B, -1, -1)    # (B, n_q, qformer_d)

        qformer_out = self._qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden,
            return_dict=True,
        )
        # last_hidden_state: (B, n_query_tokens, qformer_d)
        query_output = qformer_out.last_hidden_state[:, : self.n_query_tokens]

        return self.output_proj(query_output)                  # (B, n_q, d_vlm)


# ── Factory ────────────────────────────────────────────────────────────────────

def _load_blip2_qformer(
    pretrained_name: str | None,
    n_query_tokens: int,
) -> tuple[nn.Module, int, int]:
    """Load BLIP-2 Q-Former from HuggingFace or init from scratch.

    Falls back to a pure-PyTorch cross-attention Q-Former when
    ``transformers`` / BLIP-2 is unavailable (e.g. CPU-only environments
    with an incompatible transformers version).

    Returns:
        (qformer_module, qformer_hidden_dim, encoder_hidden_dim)

        qformer_hidden_dim  — hidden size of the Q-Former's own self-attention
        encoder_hidden_dim  — expected dim of encoder_hidden_states passed to
                              the cross-attention key/value projections
    """
    HIDDEN = 768

    if pretrained_name is not None:
        try:
            from transformers import Blip2Model
            full = Blip2Model.from_pretrained(pretrained_name)
            qformer = full.qformer
            enc_hidden = getattr(qformer.config, "encoder_hidden_size", HIDDEN)
            return qformer, qformer.config.hidden_size, enc_hidden
        except Exception:
            pass  # Fall through to scratch init

    # Try BLIP-2 scratch init; fall back to pure-PyTorch if unavailable
    try:
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        config = Blip2QFormerConfig(
            hidden_size=HIDDEN,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            cross_attention_frequency=2,
            encoder_hidden_size=HIDDEN,
        )
        return Blip2QFormerModel(config), HIDDEN, HIDDEN
    except Exception:
        import warnings
        warnings.warn(
            "Blip2QFormerModel unavailable (transformers/PyTorch version mismatch). "
            "Using lightweight pure-PyTorch fallback Q-Former.",
            ImportWarning,
            stacklevel=4,
        )
        return _FallbackQFormer(hidden_size=HIDDEN), HIDDEN, HIDDEN


class _FallbackQFormer(nn.Module):
    """Pure-PyTorch cross-attention Q-Former (no transformers dependency).

    Mirrors the interface expected by ``QFormer.forward``:
    accepts ``query_embeds``, ``encoder_hidden_states``, ``return_dict``
    and returns a namespace with ``.last_hidden_state``.
    """

    def __init__(self, hidden_size: int = 768, n_layers: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def forward(
        self,
        query_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
        **_kwargs,
    ) -> object:
        # query_embeds         : (B, n_q, hidden_size)
        # encoder_hidden_states: (B, N_spatial, hidden_size)
        out = self.decoder(query_embeds, encoder_hidden_states)
        if return_dict:
            from types import SimpleNamespace
            return SimpleNamespace(last_hidden_state=out)
        return out
