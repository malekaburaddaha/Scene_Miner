"""Abstract base class for scene dataset adapters.

All concrete dataset implementations (AV2, nuScenes, Waymo) should
subclass BaseSceneDataset and implement its three abstract methods so
that the rest of the pipeline (feature extraction, evaluation, scripts)
can treat every dataset uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scene_mining.datasets.av2_dataset import BackboneInput
    from scene_mining.tracks.track import TrackCollection


class BaseSceneDataset(ABC):
    """Abstract dataset adapter for scene mining.

    Concrete subclasses must implement:
    - ``get_log_ids()``
    - ``get_annotations(log_id)``
    - ``adapt_to_backbone_input(log_id, timestamp_ns)``
    """

    @abstractmethod
    def get_log_ids(self) -> list[str]:
        """Return all log (scene) IDs available in this split."""

    @abstractmethod
    def get_annotations(self, log_id: str) -> "TrackCollection":
        """Load ground-truth 3-D track annotations for *log_id*.

        Returns:
            TrackCollection with all annotated objects across all
            lidar timestamps in the log.
        """

    @abstractmethod
    def adapt_to_backbone_input(
        self,
        log_id: str,
        timestamp_ns: int,
        frame: dict[str, Any] | None = None,
    ) -> "BackboneInput":
        """Build a BackboneInput for the given log / timestamp.

        Args:
            log_id: Dataset-specific scene/log identifier.
            timestamp_ns: Timestamp in nanoseconds.
            frame: Optional pre-loaded frame dict (camera_name → image).
                   If None, the adapter should load images itself.

        Returns:
            BackboneInput ready for the SceneMiner backbone.
        """

    # ── Optional helpers (may be overridden) ──────────────────────────────────

    def get_hd_map(self, log_id: str) -> Any:
        """Return the HD map for *log_id*, or None if not available."""
        return None

    def get_lidar_timestamps(self, log_id: str) -> list[int]:
        """Return sorted lidar timestamps (ns) for *log_id*.

        Default: infer from the track collection timestamps.
        """
        tracks = self.get_annotations(log_id)
        if hasattr(tracks, "timestamps"):
            return sorted(tracks.timestamps)
        return []
