"""Generate a formal poster asset for a single event voxel and a voxel clip.

The figure is intentionally limited to the input representation: one event
voxel shown as temporal-bin planes, and one clip shown as consecutive voxel
frames. Both panels use real precomputed event voxels from this repository.

Example:
    python scripts/viz/plot_voxel_clip_asset.py --output plots/voxel_clip_asset.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


INK = "#181818"
MUTED = "#5f5f5f"
FRAME = "#5a5a5a"
LIGHT_FRAME = "#9a9a9a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot only the event voxel and voxel clip as a paper/poster asset."
    )
    parser.add_argument(
        "--voxel-dir",
        type=Path,
        default=REPO_ROOT / "data" / "tartanair" / "precomputed_test" / "competition_Test_ME000",
        help="Directory containing derotated_voxels.npy and metadata.json.",
    )
    parser.add_argument("--voxel-file", default="derotated_voxels.npy")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="First voxel index in the clip. If omitted, an active real clip is selected.",
    )
    parser.add_argument("--clip-len", type=int, default=6, help="Number of voxel frames in the clip, N_f.")
    parser.add_argument("--search-samples", type=int, default=96)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "plots" / "voxel_clip_asset.png",
        help="Output path. Use .png, .pdf, or .svg.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_metadata(voxel_dir: Path) -> dict:
    metadata_path = voxel_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def choose_active_index(voxels: np.ndarray, clip_len: int, eps: float, search_samples: int) -> tuple[int, int]:
    max_start = voxels.shape[0] - clip_len
    if max_start < 0:
        raise ValueError(f"clip_len={clip_len} exceeds sequence length {voxels.shape[0]}.")

    candidates = np.unique(
        np.linspace(0, max_start, min(search_samples, max_start + 1), dtype=np.int64)
    )
    best_idx = int(candidates[0])
    best_activity = -1
    for idx in candidates:
        activity = int(np.count_nonzero(np.abs(voxels[int(idx) : int(idx) + clip_len]) > eps))
        if activity > best_activity:
            best_idx = int(idx)
            best_activity = activity
    return best_idx, best_activity


def signed_projection(voxel: np.ndarray) -> np.ndarray:
    return np.sum(voxel, axis=0)


def robust_limit(images: list[np.ndarray], percentile: float = 99.2) -> float:
    values = np.concatenate([np.abs(image).ravel() for image in images])
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if len(values) == 0:
        return 1.0
    return float(max(np.percentile(values, percentile), 1e-6))


def rgba_signed(image: np.ndarray, clim: float, alpha: float = 1.0, grayscale: bool = False) -> np.ndarray:
    if grayscale:
        magnitude = np.abs(image)
        scale = np.percentile(magnitude[magnitude > 0], 99.0) if np.any(magnitude > 0) else 1.0
        rgba = mpl.colormaps["Greys"](0.12 + 0.72 * np.clip(magnitude / max(float(scale), 1e-6), 0.0, 1.0))
    else:
        rgba = mpl.colormaps["RdBu_r"](Normalize(vmin=-clim, vmax=clim)(image))
    rgba[..., 3] = alpha
    return rgba


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], lw: float = 1.25) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=INK, mutation_scale=9, shrinkA=0, shrinkB=0),
        zorder=50,
    )


def draw_dots(ax: plt.Axes, x: float, y: float, angle: float = 0.0) -> None:
    offsets = np.array([[-0.014, 0.0], [0.0, 0.0], [0.014, 0.0]])
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    pts = offsets @ rot.T
    for dx, dy in pts:
        ax.plot(x + dx, y + dy, marker=".", color=INK, markersize=4, zorder=55)


def draw_bracket(
    ax: plt.Axes,
    x: float,
    y0: float,
    y1: float,
    label: str,
    side: str = "right",
    fontsize: float = 11.0,
) -> None:
    tick = 0.018 if side == "right" else -0.018
    ax.plot([x, x + tick], [y0, y0], color=INK, lw=1.0, zorder=60)
    ax.plot([x + tick, x + tick], [y0, y1], color=INK, lw=1.0, zorder=60)
    ax.plot([x, x + tick], [y1, y1], color=INK, lw=1.0, zorder=60)
    ax.text(
        x + tick + (0.01 if side == "right" else -0.01),
        (y0 + y1) / 2,
        label,
        ha="left",
        va="center",
        fontsize=fontsize,
    )


def draw_stacked_planes(
    ax: plt.Axes,
    images: list[np.ndarray],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    dx: float,
    dy: float,
    label: str,
    bracket_label: str,
    time_label: str,
    front_count: int = 3,
) -> None:
    n = len(images)
    for j in range(n - 1, -1, -1):
        x0 = x + j * dx
        y0 = y + j * dy
        alpha = 0.54 + 0.46 * (n - 1 - j) / max(n - 1, 1)
        image = images[j].copy()
        image[..., 3] *= alpha
        ax.imshow(image, extent=(x0, x0 + w, y0, y0 + h), interpolation="bilinear", zorder=10 + n - j)
        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                w,
                h,
                edgecolor=FRAME if j < front_count else LIGHT_FRAME,
                facecolor="none",
                linewidth=0.95 if j < front_count else 0.75,
                zorder=25 + n - j,
            )
        )

    draw_arrow(ax, (x - 0.12, y - 0.035), (x - 0.025, y + 0.12), lw=1.35)
    ax.text(x - 0.082, y + 0.038, time_label, rotation=57, ha="center", va="center", fontsize=10, color=INK)
    draw_dots(ax, x - 0.136, y + 0.135, angle=0.35)

    bx = x + w + 0.015
    draw_bracket(
        ax,
        bx,
        y + 0.015,
        y + h + (n - 1) * dy - 0.004,
        bracket_label,
        fontsize=11.5,
    )
    ax.text(x + w / 2 - 0.005, y - 0.055, label, ha="center", va="center", fontsize=8.8, color=MUTED)


def make_figure(clip: np.ndarray, metadata: dict, index: int) -> plt.Figure:
    channels = clip.shape[1]
    projections = [signed_projection(voxel) for voxel in clip]
    clip_clim = robust_limit(projections)
    bin_clim = robust_limit([clip[0, c] for c in range(channels)])

    voxel_bin_images = [
        rgba_signed(clip[0, c], bin_clim, alpha=0.96 if c < 2 else 0.68, grayscale=c >= 2)
        for c in range(channels)
    ]
    clip_images = [
        rgba_signed(projection, clip_clim, alpha=0.96 if j < 3 else 0.72, grayscale=j >= 3)
        for j, projection in enumerate(projections)
    ]

    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    draw_stacked_planes(
        ax,
        voxel_bin_images,
        x=0.25,
        y=0.35,
        w=0.17,
        h=0.13,
        dx=-0.025,
        dy=0.034,
        label=rf"single event voxel $V_{{{index}}}$",
        bracket_label=rf"$C={channels}$",
        time_label=r"$\tau$",
        front_count=2,
    )
    draw_stacked_planes(
        ax,
        clip_images,
        x=0.72,
        y=0.35,
        w=0.17,
        h=0.13,
        dx=-0.025,
        dy=0.034,
        label=r"voxel clip $\{V_i\}_{i=1}^{N_f}$",
        bracket_label=r"$N_f$",
        time_label="time",
        front_count=3,
    )

    duration = metadata.get("delta_t_ms", None)
    source = Path(metadata.get("source_sequence", "")).name or "real event data"
    detail = f"Real precomputed event voxels from {source}"
    if duration is not None:
        detail += f"; each voxel integrates {duration} ms"
    ax.text(0.5, 0.08, detail, ha="center", va="center", fontsize=7.6, color=MUTED)
    ax.set_aspect("auto")

    return fig


def main() -> None:
    args = parse_args()
    set_style()

    voxel_dir = args.voxel_dir.resolve()
    voxel_path = voxel_dir / args.voxel_file
    if not voxel_path.exists():
        raise FileNotFoundError(f"Missing voxel file: {voxel_path}")

    metadata = load_metadata(voxel_dir)
    voxels = np.load(voxel_path, mmap_mode="r")
    if voxels.ndim != 4:
        raise ValueError(f"Expected voxel shape [N, C, H, W], got {voxels.shape}.")
    if args.clip_len < 2:
        raise ValueError("--clip-len must be at least 2.")

    max_start = voxels.shape[0] - args.clip_len
    if max_start < 0:
        raise ValueError(f"--clip-len {args.clip_len} exceeds sequence length {voxels.shape[0]}.")

    if args.index is None:
        index, activity = choose_active_index(voxels, args.clip_len, args.eps, args.search_samples)
        print(f"Selected active clip index {index} ({activity:,} active cells across the clip).")
    else:
        index = int(args.index)
        if not 0 <= index <= max_start:
            raise IndexError(f"--index must be in [0, {max_start}] for clip_len={args.clip_len}.")

    clip = np.asarray(voxels[index : index + args.clip_len], dtype=np.float32)
    fig = make_figure(clip, metadata=metadata, index=index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved voxel asset: {args.output}")
    print(f"Voxel source: {voxel_path}")
    print(f"Clip shape: {tuple(clip.shape)}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
