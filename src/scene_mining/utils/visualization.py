"""BEV visualization for scene mining results.

Draws track bounding-box footprints in bird's-eye-view (BEV) with
referred tracks highlighted in green, related tracks in blue.
No interactive display — uses the Agg backend so it runs on headless
cluster nodes.

Usage::

    from scene_mining.utils.visualization import draw_bev_scenario
    img = draw_bev_scenario(track_collection, scenario_dict)
    img.save("bev.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def draw_bev_scenario(
    track_collection: Any,
    scenario_dict: dict[str, dict[str, list[int]]],
    timestamp_ns: int | None = None,
    figsize: tuple[float, float] = (12.0, 12.0),
    range_m: float = 80.0,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> "PIL.Image.Image":  # noqa: F821
    """Draw a BEV scenario visualisation.

    Args:
        track_collection: ``TrackCollection`` with BEV bbox data.
        scenario_dict: ``ScenarioDict`` mapping
            ``track_uuid → {related_uuid → [timestamp_ns]}``.
        timestamp_ns: If given, draw only tracks at this timestamp.
            Otherwise draws all track footprints across time.
        figsize: Matplotlib figure size in inches.
        range_m: Half-width of the BEV window in metres.
        save_path: If given, save the image to this path.
        title: Optional figure title.

    Returns:
        PIL.Image with the rendered BEV plot.
    """
    from PIL import Image
    import io

    referred_ids = set(scenario_dict.keys())
    related_ids: set[str] = set()
    for inner in scenario_dict.values():
        related_ids.update(inner.keys())
    related_ids -= referred_ids  # referred takes priority in colour

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Collect records to draw
    records = _collect_records(track_collection, timestamp_ns)

    drawn_ids: set[str] = set()
    for rec in records:
        tid = rec.track_id
        if tid in drawn_ids:
            continue
        drawn_ids.add(tid)

        color = _track_color(tid, referred_ids, related_ids)
        alpha = 0.85 if (tid in referred_ids or tid in related_ids) else 0.3
        _draw_box(ax, rec, color=color, alpha=alpha)

    # Overlay timestamp annotation
    if timestamp_ns is not None:
        ts_s = timestamp_ns * 1e-9
        ax.text(
            -range_m + 1, range_m - 2,
            f"t = {ts_s:.3f} s",
            color="white", fontsize=9, va="top",
        )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#00ff88", label="Referred track"),
        mpatches.Patch(color="#4488ff", label="Related track"),
        mpatches.Patch(color="#888888", label="Background track"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              facecolor="#333355", edgecolor="white", labelcolor="white")

    if title:
        ax.set_title(title, color="white", fontsize=11)

    # Render to PIL
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    buf.close()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        pil_image.save(save_path)

    return pil_image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_records(track_collection: Any, timestamp_ns: int | None) -> list[Any]:
    """Return a flat list of TrackRecord objects to draw."""
    if timestamp_ns is not None and hasattr(track_collection, "get_at_timestamp"):
        return track_collection.get_at_timestamp(timestamp_ns)
    # Fallback: iterate all records
    if hasattr(track_collection, "records"):
        return list(track_collection.records)
    if hasattr(track_collection, "__iter__"):
        return list(track_collection)
    return []


def _track_color(
    track_id: str,
    referred_ids: set[str],
    related_ids: set[str],
) -> str:
    if track_id in referred_ids:
        return "#00ff88"   # green
    if track_id in related_ids:
        return "#4488ff"   # blue
    return "#888888"       # grey


def _draw_box(ax: plt.Axes, rec: Any, color: str, alpha: float) -> None:
    """Draw a 2-D rotated bounding box for *rec* in the BEV plane."""
    tx, ty = float(rec.translation_m[0]), float(rec.translation_m[1])
    length = float(rec.size[0])   # x-extent
    width = float(rec.size[1])    # y-extent
    yaw = float(rec.yaw)

    # Corner offsets in object frame (front, back × left, right)
    dx = length / 2.0
    dy = width / 2.0
    corners_local = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy],
    ])

    # Rotate by yaw
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    corners_world = corners_local @ rot.T + np.array([tx, ty])

    polygon = plt.Polygon(corners_world, closed=True,
                          edgecolor=color, facecolor=color,
                          linewidth=1.0, alpha=alpha)
    ax.add_patch(polygon)

    # Forward arrow
    ax.annotate(
        "",
        xy=(tx + dx * cos_y, ty + dx * sin_y),
        xytext=(tx, ty),
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
    )


__all__ = ["draw_bev_scenario"]
