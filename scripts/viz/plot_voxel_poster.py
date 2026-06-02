"""Create a clean poster figure for event voxels and voxel clips.

The script uses real precomputed event voxels from this repository. It creates
one academic-style figure with:
  1. a 3D event-volume view of one voxel grid, where nonzero voxel activations
     are plotted inside the x-y-time cuboid;
  2. a clip view showing consecutive voxel grids as signed image projections.

Example:
    python scripts/viz/plot_voxel_poster.py --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


POS_COLOR = "#b2182b"
NEG_COLOR = "#2166ac"
NEUTRAL = "#252525"
GRID_COLOR = "#b7b7b7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a publication/poster-ready event voxel and a consecutive "
            "voxel clip using real precomputed event voxels."
        )
    )
    parser.add_argument(
        "--voxel-dir",
        type=Path,
        default=REPO_ROOT / "data" / "tartanair" / "precomputed_test" / "competition_Test_ME000",
        help="Directory containing derotated_voxels.npy, metadata.json, and relative_motions.txt.",
    )
    parser.add_argument(
        "--voxel-file",
        default="derotated_voxels.npy",
        help="Name of the precomputed voxel file inside --voxel-dir.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Voxel index to visualize. If omitted, an active real voxel is selected automatically.",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=4,
        help="Number of consecutive voxel grids shown in the clip panel.",
    )
    parser.add_argument(
        "--search-samples",
        type=int,
        default=96,
        help="Number of candidate voxels inspected when --index is omitted.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=14000,
        help="Maximum nonzero voxel activations plotted in the 3D volume.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Absolute threshold used to decide whether a voxel cell is active.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used only for plotting subsampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "plots" / "poster_voxel_events.png",
        help="Output figure path. Use .png, .pdf, or .svg.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Export resolution for raster outputs.")
    parser.add_argument("--show", action="store_true", help="Open the figure window after saving.")
    parser.add_argument("--view-elev", type=float, default=23.0, help="3D view elevation.")
    parser.add_argument("--view-azim", type=float, default=-57.0, help="3D view azimuth.")
    return parser.parse_args()


def load_metadata(voxel_dir: Path) -> dict:
    metadata_path = voxel_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_anchor_times_us(voxel_dir: Path, expected_count: int) -> np.ndarray | None:
    rel_path = voxel_dir / "relative_motions.txt"
    if not rel_path.exists():
        return None
    rel = np.atleast_2d(np.loadtxt(rel_path, dtype=np.float64, skiprows=1))
    if rel.shape[1] < 2:
        return None
    anchors = np.concatenate([rel[:1, 0], rel[:, 1]], axis=0).astype(np.int64)
    if len(anchors) != expected_count:
        return None
    return anchors


def choose_active_index(
    voxels: np.ndarray,
    clip_len: int,
    eps: float,
    search_samples: int,
) -> tuple[int, int]:
    max_start = voxels.shape[0] - clip_len
    if max_start < 0:
        raise ValueError(f"clip_len={clip_len} is longer than the sequence length {voxels.shape[0]}.")
    if search_samples <= 0:
        idx = max_start // 2
        activity = int(np.count_nonzero(np.abs(voxels[idx]) > eps))
        return idx, activity

    candidates = np.unique(
        np.linspace(0, max_start, min(search_samples, max_start + 1), dtype=np.int64)
    )
    best_idx = int(candidates[0])
    best_activity = -1
    for idx in candidates:
        activity = int(np.count_nonzero(np.abs(voxels[int(idx)]) > eps))
        if activity > best_activity:
            best_idx = int(idx)
            best_activity = activity
    return best_idx, best_activity


def voxel_to_event_points(
    voxel: np.ndarray,
    duration_ms: float,
    eps: float,
    max_events: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    channels = voxel.shape[0]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for bin_idx in range(channels):
        plane = voxel[bin_idx]
        y_idx, x_idx = np.nonzero(np.abs(plane) > eps)
        if len(x_idx) == 0:
            continue
        xs.append(x_idx.astype(np.float32))
        ys.append(y_idx.astype(np.float32))
        t_ms = (bin_idx + 0.5) * duration_ms / channels
        ts.append(np.full(len(x_idx), t_ms, dtype=np.float32))
        values.append(plane[y_idx, x_idx].astype(np.float32))

    if not xs:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty, 0

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    t = np.concatenate(ts)
    value = np.concatenate(values)
    total = len(value)

    if max_events > 0 and total > max_events:
        keep = rng.choice(total, size=max_events, replace=False)
        keep.sort()
        x, y, t, value = x[keep], y[keep], t[keep], value[keep]

    return x, y, t, value, total


def signed_projection(voxel: np.ndarray) -> np.ndarray:
    """Collapse temporal bins to one signed image for compact clip display."""
    return np.sum(voxel, axis=0)


def robust_symmetric_limit(images: list[np.ndarray], percentile: float = 99.4) -> float:
    values = np.concatenate([np.abs(img).ravel() for img in images])
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if len(values) == 0:
        return 1.0
    return float(max(np.percentile(values, percentile), 1e-6))


def set_academic_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": NEUTRAL,
            "xtick.color": NEUTRAL,
            "ytick.color": NEUTRAL,
            "text.color": NEUTRAL,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_volume_box(ax, width: int, height: int, duration_ms: float, channels: int) -> None:
    corners = np.array(
        [
            [0, 0, 0],
            [duration_ms, 0, 0],
            [duration_ms, width, 0],
            [0, width, 0],
            [0, 0, height],
            [duration_ms, 0, height],
            [duration_ms, width, height],
            [0, width, height],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i0, i1 in edges:
        ax.plot(
            [corners[i0, 0], corners[i1, 0]],
            [corners[i0, 1], corners[i1, 1]],
            [corners[i0, 2], corners[i1, 2]],
            color=GRID_COLOR,
            linewidth=0.8,
            alpha=0.8,
        )

    for bin_idx in range(1, channels):
        t = bin_idx * duration_ms / channels
        ax.plot([t, t], [0, width], [0, 0], color=GRID_COLOR, linewidth=0.45, alpha=0.45)
        ax.plot([t, t], [0, width], [height, height], color=GRID_COLOR, linewidth=0.45, alpha=0.45)
        ax.plot([t, t], [0, 0], [0, height], color=GRID_COLOR, linewidth=0.45, alpha=0.45)
        ax.plot([t, t], [width, width], [0, height], color=GRID_COLOR, linewidth=0.45, alpha=0.45)


def format_volume_axis(ax, width: int, height: int, duration_ms: float, elev: float, azim: float) -> None:
    ax.set_title("Single event voxel: nonzero activations in x-y-time")
    ax.set_xlabel("time $t$ [ms]", labelpad=7)
    ax.set_ylabel("image $x$ [px]", labelpad=7)
    ax.set_zlabel("image $y$ [px]", labelpad=7)
    ax.set_xlim(0, duration_ms)
    ax.set_ylim(0, width)
    ax.set_zlim(height, 0)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.set_box_aspect((1.0, 1.25, 0.85))


def make_figure(
    voxel: np.ndarray,
    clip: np.ndarray,
    index: int,
    anchors_us: np.ndarray | None,
    metadata: dict,
    eps: float,
    max_events: int,
    seed: int,
    view_elev: float,
    view_azim: float,
) -> plt.Figure:
    duration_ms = float(metadata.get("delta_t_ms", 50.0))
    channels, height, width = voxel.shape
    clip_len = clip.shape[0]
    rng = np.random.default_rng(seed)

    x, y, t, value, total_events = voxel_to_event_points(
        voxel=voxel,
        duration_ms=duration_ms,
        eps=eps,
        max_events=max_events,
        rng=rng,
    )
    colors = np.where(value >= 0.0, POS_COLOR, NEG_COLOR)
    size = 2.0 + 7.0 * np.clip(np.abs(value), 0.0, 1.0)

    projections = [signed_projection(clip_i) for clip_i in clip]
    clim = robust_symmetric_limit(projections)

    fig = plt.figure(figsize=(13.2, 6.7), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.64])
    ax_3d = fig.add_subplot(outer[0], projection="3d")
    right = outer[1].subgridspec(2, 1, height_ratios=[1.0, 1.0])
    clip_grid = right[0].subgridspec(1, clip_len + 1, width_ratios=[1.0] * clip_len + [0.055])
    bin_grid = right[1].subgridspec(1, channels + 1, width_ratios=[1.0] * channels + [0.055])

    draw_volume_box(ax_3d, width=width, height=height, duration_ms=duration_ms, channels=channels)
    ax_3d.scatter(
        t,
        x,
        y,
        c=colors,
        s=size,
        alpha=0.46,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    format_volume_axis(ax_3d, width=width, height=height, duration_ms=duration_ms, elev=view_elev, azim=view_azim)
    ax_3d.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=POS_COLOR, markersize=6, label="positive event"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=NEG_COLOR, markersize=6, label="negative event"),
        ],
        loc="upper left",
        frameon=False,
        borderpad=0.1,
        handletextpad=0.4,
    )

    clip_images = []
    for j in range(clip_len):
        ax = fig.add_subplot(clip_grid[j])
        image = ax.imshow(
            projections[j],
            cmap="RdBu_r",
            vmin=-clim,
            vmax=clim,
            origin="upper",
            interpolation="nearest",
            rasterized=True,
        )
        clip_images.append(image)
        ax.set_title(f"$V_{{{index + j}}}$")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#666666")

        if anchors_us is not None:
            end_ms = anchors_us[index + j] / 1000.0
            start_ms = end_ms - duration_ms
            subtitle = f"{start_ms:.1f}-{end_ms:.1f} ms"
        else:
            subtitle = f"{duration_ms:.0f} ms window"
        ax.text(
            0.5,
            -0.08,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
        )

    bin_clim = robust_symmetric_limit([voxel[c] for c in range(channels)])
    bin_images = []
    for c in range(channels):
        ax = fig.add_subplot(bin_grid[c])
        image = ax.imshow(
            voxel[c],
            cmap="RdBu_r",
            vmin=-bin_clim,
            vmax=bin_clim,
            origin="upper",
            interpolation="nearest",
            rasterized=True,
        )
        bin_images.append(image)
        t0 = c * duration_ms / channels
        t1 = (c + 1) * duration_ms / channels
        ax.set_title(f"bin {c}\n{t0:.0f}-{t1:.0f} ms")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#666666")

    cbar = fig.colorbar(clip_images[0], cax=fig.add_subplot(clip_grid[-1]))
    cbar.set_label("signed voxel sum")
    bin_cbar = fig.colorbar(bin_images[0], cax=fig.add_subplot(bin_grid[-1]))
    bin_cbar.set_label("signed voxel value")

    sequence_name = Path(metadata.get("source_sequence", "")).name or "precomputed event sequence"
    figure_title = (
        f"Event voxel representation from real data: {sequence_name} "
        f"({channels} temporal bins, {height} x {width} px)"
    )
    fig.suptitle(figure_title, fontsize=13, y=1.015)

    caption = (
        f"Left: nonzero cells of voxel $V_{{{index}}}$ plotted in the event volume "
        f"(shown {len(x):,} of {total_events:,} active cells). "
        "Top right: consecutive voxel grids collapsed over temporal bins. "
        f"Bottom right: the {channels} temporal bins that form $V_{{{index}}}$."
    )
    fig.text(0.012, -0.015, caption, ha="left", va="bottom", fontsize=8.8, color="#333333")

    return fig


def main() -> None:
    args = parse_args()
    set_academic_style()

    voxel_dir = args.voxel_dir.resolve()
    voxel_path = voxel_dir / args.voxel_file
    if not voxel_path.exists():
        raise FileNotFoundError(f"Missing voxel file: {voxel_path}")

    metadata = load_metadata(voxel_dir)
    voxels = np.load(voxel_path, mmap_mode="r")
    if voxels.ndim != 4:
        raise ValueError(f"Expected voxel shape [N, C, H, W], got {voxels.shape}.")

    if args.clip_len < 1:
        raise ValueError("--clip-len must be at least 1.")

    max_start = voxels.shape[0] - args.clip_len
    if max_start < 0:
        raise ValueError(f"--clip-len {args.clip_len} exceeds sequence length {voxels.shape[0]}.")

    if args.index is None:
        index, activity = choose_active_index(
            voxels=voxels,
            clip_len=args.clip_len,
            eps=args.eps,
            search_samples=args.search_samples,
        )
        print(f"Selected active voxel index {index} ({activity:,} active cells).")
    else:
        index = int(args.index)
        if not 0 <= index <= max_start:
            raise IndexError(f"--index must be in [0, {max_start}] for clip_len={args.clip_len}.")

    voxel = np.asarray(voxels[index], dtype=np.float32)
    clip = np.asarray(voxels[index : index + args.clip_len], dtype=np.float32)
    anchors_us = load_anchor_times_us(voxel_dir, expected_count=voxels.shape[0])

    fig = make_figure(
        voxel=voxel,
        clip=clip,
        index=index,
        anchors_us=anchors_us,
        metadata=metadata,
        eps=args.eps,
        max_events=args.max_events,
        seed=args.seed,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved poster figure: {args.output}")
    print(f"Voxel source: {voxel_path}")
    print(f"Voxel shape: {tuple(voxels.shape)}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
