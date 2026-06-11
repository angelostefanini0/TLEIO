import argparse
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.learning.dataloader.events_to_voxel.utils import (
    build_derotation_context,
    load_event_camera_matrix,
)
from src.learning.dataloader.representation.event_derotation import (
    derotate_events_in_slices,
    raw_events_to_fixed_window_voxel,
    resolve_derotation_slices,
)
from src.learning.dataloader.representation.event_slicer import EventSlicer
from src.spatial_math import normalize_quaternions
from scripts.utils.config import default_config_path, parse_args_with_config

try:
    import hdf5plugin  # noqa: F401  # Registers external HDF5 filters when installed.
except ImportError:
    hdf5plugin = None


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes", "y"}:
        return True
    if v.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize raw and de-rotated event slices in x/y/time. "
            "Events are de-rotated in small temporal slices, then voxelized separately."
        )
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=None,
        help=(
            "Sequence directory containing events.h5, stamped_groundtruth.txt, and K.yaml. "
            "Processed sequences with ms_to_idx and raw EDS-style sequences are both supported."
        ),
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=50.0,
        help="Temporal window size to visualize.",
    )
    parser.add_argument(
        "--anchor-index",
        type=int,
        default=None,
        help=(
            "Anchor index from anchor_poses.txt. The window is "
            "[anchor - duration, anchor]. Defaults to the middle anchor if possible."
        ),
    )
    parser.add_argument(
        "--start-us",
        type=int,
        default=None,
        help="Explicit window start timestamp in the processed sequence time frame.",
    )
    parser.add_argument(
        "--end-us",
        type=int,
        default=None,
        help="Explicit window end timestamp in the processed sequence time frame.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of temporal bins used for final voxelization after event de-rotation.",
    )
    parser.add_argument(
        "--derotation-slices",
        type=int,
        default=None,
        help=(
            "Number of small temporal windows used for event-space de-rotation. "
            "Overrides --derotation-slice-ms when set."
        ),
    )
    parser.add_argument(
        "--derotation-slice-ms",
        type=float,
        default=0.5,
        help="Approximate duration of each event de-rotation slice in milliseconds.",
    )
    parser.add_argument("--height", type=int, default=480, help="Original event image height.")
    parser.add_argument("--width", type=int, default=640, help="Original event image width.")
    parser.add_argument(
        "--downsampling-factor",
        type=float,
        default=1.0,
        help="Apply the same event-coordinate downsampling used during training.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size used only to validate the downsampled dimensions.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=60000,
        help="Maximum number of nonzero voxel pixels to scatter per panel.",
    )
    parser.add_argument(
        "--voxel-eps",
        type=float,
        default=1e-6,
        help="Absolute voxel value threshold used when extracting scatter points from warped rasters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure, for example outputs/derotation_debug.png.",
    )
    parser.add_argument(
        "--show",
        type=str2bool,
        default=True,
        help="Show the matplotlib window after generating the figure.",
    )
    parser.add_argument("--view-elev", type=float, default=24.0, help="3D view elevation.")
    parser.add_argument("--view-azim", type=float, default=-58.0, help="3D view azimuth.")
    return parse_args_with_config(
        parser,
        default_config_path("visualize_event_derotation"),
        required=("sequence_dir",),
    )


def load_table(path: Path, *, skiprows: int = 0) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    data = np.loadtxt(path, dtype=np.float64, skiprows=skiprows)
    return np.atleast_2d(data)


def timestamps_to_us(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) >= 2:
        diffs = np.diff(timestamps)
        diffs = diffs[diffs > 0]
        median_dt = float(np.median(diffs)) if len(diffs) else np.inf
    else:
        median_dt = np.inf

    # Raw EDS stamped_groundtruth.txt stores seconds; processed files store integer us.
    if median_dt < 1000.0:
        return np.rint(timestamps * 1e6).astype(np.int64)
    return np.rint(timestamps).astype(np.int64)


def get_downsampled_size(
    original_height: int,
    original_width: int,
    downsampling_factor: float,
    patch_size: int,
) -> tuple[int, int]:
    if downsampling_factor <= 0:
        raise ValueError("--downsampling-factor must be > 0.")
    if patch_size <= 0:
        raise ValueError("--patch-size must be > 0.")

    new_height = int(round(original_height * downsampling_factor))
    new_width = int(round(original_width * downsampling_factor))
    if new_height <= 0 or new_width <= 0:
        raise ValueError(f"Downsampled size must stay positive, got {new_height}x{new_width}.")
    if new_height % patch_size != 0 or new_width % patch_size != 0:
        raise ValueError(
            "Downsampled size must be divisible by patch size. "
            f"factor={downsampling_factor} gives {new_height}x{new_width} with patch_size={patch_size}."
        )
    return new_height, new_width


def choose_window_us(args: argparse.Namespace, gt_timestamps_us: np.ndarray) -> tuple[int, int, str]:
    duration_us = int(round(args.duration_ms * 1000.0))
    if duration_us <= 0:
        raise ValueError("--duration-ms must be positive.")

    if args.start_us is not None or args.end_us is not None:
        if args.start_us is None and args.end_us is None:
            raise ValueError("Internal timestamp window selection error.")
        if args.start_us is None:
            end_us = int(args.end_us)
            start_us = end_us - duration_us
        elif args.end_us is None:
            start_us = int(args.start_us)
            end_us = start_us + duration_us
        else:
            start_us = int(args.start_us)
            end_us = int(args.end_us)
        if start_us >= end_us:
            raise ValueError("Window start must be smaller than window end.")
        return start_us, end_us, "explicit"

    anchor_path = args.sequence_dir / "anchor_poses.txt"
    if anchor_path.exists():
        anchors = timestamps_to_us(load_table(anchor_path, skiprows=1)[:, 0])
        if len(anchors) == 0:
            raise ValueError(f"No anchors found in {anchor_path}")

        anchor_idx = args.anchor_index
        if anchor_idx is None:
            anchor_idx = len(anchors) // 2
        if not 0 <= anchor_idx < len(anchors):
            raise IndexError(f"--anchor-index {anchor_idx} out of range [0, {len(anchors) - 1}]")

        end_us = int(anchors[anchor_idx])
        start_us = end_us - duration_us
        return start_us, end_us, f"anchor {anchor_idx}"

    if args.anchor_index is not None:
        raise FileNotFoundError(f"--anchor-index requires {anchor_path}")

    end_us = int(gt_timestamps_us[len(gt_timestamps_us) // 2])
    start_us = end_us - duration_us
    return start_us, end_us, "middle GT timestamp"


def build_sequence_info(
    sequence_dir: Path,
    new_height: int,
    new_width: int,
    original_height: int,
    original_width: int,
) -> dict:
    gt_full = load_table(sequence_dir / "stamped_groundtruth.txt")
    if gt_full.shape[1] < 8:
        raise ValueError(
            f"{sequence_dir / 'stamped_groundtruth.txt'}: expected at least 8 columns, got {gt_full.shape[1]}."
        )

    scale_y = new_height / original_height
    scale_x = new_width / original_width
    return {
        "seq_path": sequence_dir,
        "gt_timestamps_us": timestamps_to_us(gt_full[:, 0]),
        "gt_quat_xyzw": normalize_quaternions(gt_full[:, 4:8].astype(np.float64)),
        "camera_matrix": load_event_camera_matrix(
            root_path=sequence_dir.parent,
            seq_path=sequence_dir,
            scale_x=scale_x,
            scale_y=scale_y,
        ),
    }

def subsample_indices(num_events: int, max_events: int) -> np.ndarray:
    if max_events <= 0 or num_events <= max_events:
        return np.arange(num_events)
    return np.linspace(0, num_events - 1, max_events, dtype=np.int64)


def polarity_colors(p: np.ndarray) -> np.ndarray:
    return np.where(p > 0, "#d62728", "#1f77b4")


def voxel_to_points(
    voxel: torch.Tensor,
    duration_ms: float,
    max_points: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    voxel_np = voxel.detach().cpu().numpy()
    xs = []
    ys = []
    ts = []
    vals = []
    num_bins = voxel_np.shape[0]

    for bin_idx in range(num_bins):
        plane = voxel_np[bin_idx]
        y_idx, x_idx = np.nonzero(np.abs(plane) > eps)
        if len(x_idx) == 0:
            continue

        xs.append(x_idx.astype(np.float64))
        ys.append(y_idx.astype(np.float64))
        ts.append(np.full(len(x_idx), (bin_idx + 0.5) * duration_ms / num_bins, dtype=np.float64))
        vals.append(plane[y_idx, x_idx].astype(np.float64))

    if not xs:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty, 0

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    t = np.concatenate(ts)
    value = np.concatenate(vals)
    total = len(value)
    keep = subsample_indices(total, max_points)
    return x[keep], y[keep], t[keep], value[keep], total


def count_nonzero_voxel_pixels(voxel: torch.Tensor, eps: float) -> int:
    voxel_np = voxel.detach().cpu().numpy()
    return int(np.count_nonzero(np.abs(voxel_np) > eps))


def configure_3d_axis(ax, title: str, width: int, height: int, duration_ms: float, elev: float, azim: float) -> None:
    ax.set_title(title)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("x [px]")
    ax.set_zlabel("y [px]")
    ax.xaxis.labelpad = 18
    ax.set_xlim(0, duration_ms)
    ax.set_ylim(0, width)
    ax.set_zlim(height, 0)
    ax.view_init(elev=elev, azim=azim)


def configure_2d_axis(ax, title: str, width: int, height: int) -> None:
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def make_raw_event_stream_plot(
    x: np.ndarray,
    y: np.ndarray,
    t_ms: np.ndarray,
    polarity: np.ndarray,
    width: int,
    height: int,
    duration_ms: float,
    title: str,
    view_elev: float,
    view_azim: float,
    max_events: int,
) -> plt.Figure:
    keep = subsample_indices(len(t_ms), max_events)
    x_plot = x[keep]
    y_plot = y[keep]
    t_plot = t_ms[keep]
    colors = polarity_colors(polarity[keep])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    scatter_kwargs = {
        "s": 1.5,
        "alpha": 0.35,
        "linewidths": 0,
        "depthshade": False,
    }
    ax.scatter(t_plot, x_plot, y_plot, c=colors, **scatter_kwargs)
    configure_3d_axis(ax, "Raw sliced event stream", width, height, duration_ms, view_elev, view_azim)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.text2D(
        0.02,
        0.95,
        f"events shown: {len(t_plot)}/{len(t_ms)}",
        transform=ax.transAxes,
        fontsize=10,
    )
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _voxel_image_stride(height: int, width: int) -> int:
    return max(1, int(np.ceil(max(height, width) / 400.0)))



def _robust_voxel_scale(*voxels: np.ndarray) -> float:
    values = [np.abs(voxel[np.abs(voxel) > 0.0]).reshape(-1) for voxel in voxels]
    values = [value for value in values if len(value) > 0]
    if not values:
        return 1.0
    scale = float(np.percentile(np.concatenate(values), 99.0))
    return scale if scale > 0.0 else 1.0


def _plot_voxel_bin_planes(
    ax,
    voxel_np: np.ndarray,
    duration_ms: float,
    max_abs: float,
    title: str,
    view_elev: float,
    view_azim: float,
) -> None:
    num_bins, height, width = voxel_np.shape
    stride = _voxel_image_stride(height, width)
    y_grid, x_grid = np.mgrid[0:height:stride, 0:width:stride]
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-max_abs, vmax=max_abs)
    bin_ms = duration_ms / num_bins

    for bin_idx in range(num_bins):
        plane = voxel_np[bin_idx, ::stride, ::stride]
        rgba = cmap(norm(plane))
        rgba[..., :3] = 0.78 * rgba[..., :3] + 0.22
        signal = np.clip(np.abs(plane) / max_abs, 0.0, 1.0)
        rgba[..., 3] = np.where(signal > 0.015, 0.35 + 0.65 * np.sqrt(signal), 0.0)

        t_center = (bin_idx + 0.5) * bin_ms
        t_plane = np.full_like(x_grid, t_center, dtype=np.float64)
        ax.plot_surface(
            t_plane,
            x_grid,
            y_grid,
            facecolors=rgba,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        ax.plot(
            [t_center, t_center, t_center, t_center, t_center],
            [0, width, width, 0, 0],
            [0, 0, height, height, 0],
            color="#444444",
            linewidth=0.8,
            alpha=0.65,
        )

    configure_3d_axis(ax, title, width, height, duration_ms, view_elev, view_azim)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def make_voxel_bin_3d_plot(
    raw_voxel: torch.Tensor,
    derot_voxel: torch.Tensor,
    duration_ms: float,
    title: str,
    view_elev: float,
    view_azim: float,
) -> plt.Figure:
    raw_np = raw_voxel.detach().cpu().numpy()
    derot_np = derot_voxel.detach().cpu().numpy()
    if raw_np.shape != derot_np.shape:
        raise ValueError(
            f"Raw and de-rotated voxels must have the same shape, got {raw_np.shape} and {derot_np.shape}."
        )

    max_abs = _robust_voxel_scale(raw_np, derot_np)

    fig = plt.figure(figsize=(15, 7))
    ax_raw = fig.add_subplot(1, 2, 1, projection="3d")
    ax_derot = fig.add_subplot(1, 2, 2, projection="3d")

    _plot_voxel_bin_planes(
        ax_raw,
        raw_np,
        duration_ms,
        max_abs,
        "Raw voxel bins",
        view_elev,
        view_azim,
    )
    _plot_voxel_bin_planes(
        ax_derot,
        derot_np,
        duration_ms,
        max_abs,
        "De-rotated voxel bins",
        view_elev,
        view_azim,
    )

    fig.suptitle(title)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08, wspace=0.02)
    return fig


def make_voxel_bin_plot(
    raw_voxel: torch.Tensor,
    derot_voxel: torch.Tensor,
    duration_ms: float,
    title: str,
) -> plt.Figure:
    raw_np = raw_voxel.detach().cpu().numpy()
    derot_np = derot_voxel.detach().cpu().numpy()
    if raw_np.shape != derot_np.shape:
        raise ValueError(
            f"Raw and de-rotated voxels must have the same shape, got {raw_np.shape} and {derot_np.shape}."
        )

    num_bins = raw_np.shape[0]
    max_abs = float(max(np.max(np.abs(raw_np)), np.max(np.abs(derot_np))))
    if max_abs == 0.0:
        max_abs = 1.0

    fig_width = max(12.0, 2.6 * num_bins)
    fig, axes = plt.subplots(
        2,
        num_bins,
        figsize=(fig_width, 6.0),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    bin_ms = duration_ms / num_bins
    image = None
    for bin_idx in range(num_bins):
        t0 = bin_idx * bin_ms
        t1 = (bin_idx + 1) * bin_ms
        for row, (label, voxel_np) in enumerate(
            (("Raw", raw_np), ("De-rotated", derot_np))
        ):
            ax = axes[row, bin_idx]
            image = ax.imshow(
                voxel_np[bin_idx],
                cmap="seismic",
                vmin=-max_abs,
                vmax=max_abs,
                origin="upper",
                interpolation="nearest",
            )
            ax.set_title(f"bin {bin_idx}\n{t0:.1f}-{t1:.1f} ms")
            if bin_idx == 0:
                ax.set_ylabel(label)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="voxel value")
    return fig


def h5_get_dataset(h5f: h5py.File, name: str) -> h5py.Dataset:
    candidates = [name, f"events/{name}"]
    for candidate in candidates:
        if candidate in h5f:
            return h5f[candidate]
    raise KeyError(f"Could not find any of {candidates} in {h5f.filename}")


def h5_binary_search_left(dataset: h5py.Dataset, value: int) -> int:
    left = 0
    right = int(dataset.shape[0])
    while left < right:
        mid = (left + right) // 2
        if int(dataset[mid]) < value:
            left = mid + 1
        else:
            right = mid
    return left


def load_raw_events_by_time(h5f: h5py.File, start_us: int, end_us: int) -> dict[str, np.ndarray]:
    t_ds = h5_get_dataset(h5f, "t")
    x_ds = h5_get_dataset(h5f, "x")
    y_ds = h5_get_dataset(h5f, "y")
    p_ds = h5_get_dataset(h5f, "p")

    start_idx = h5_binary_search_left(t_ds, start_us)
    end_idx = h5_binary_search_left(t_ds, end_us)
    return {
        "t": np.asarray(t_ds[start_idx:end_idx], dtype=np.int64),
        "x": np.asarray(x_ds[start_idx:end_idx]),
        "y": np.asarray(y_ds[start_idx:end_idx]),
        "p": np.asarray(p_ds[start_idx:end_idx]),
    }


def has_processed_index(h5f: h5py.File, metadata_h5f: h5py.File | None) -> bool:
    source = metadata_h5f if metadata_h5f is not None else h5f
    return "ms_to_idx" in source and ("events/t" in source or "t" in source)


def load_events(sequence_dir: Path, start_us: int, end_us: int) -> dict[str, np.ndarray] | None:
    events_file = sequence_dir / "events.h5"
    metadata_file = sequence_dir / "events_meta.h5"
    if not events_file.exists():
        raise FileNotFoundError(f"Missing required file: {events_file}")

    try:
        with h5py.File(events_file, "r") as h5f:
            if metadata_file.exists():
                with h5py.File(metadata_file, "r") as metadata_h5f:
                    if has_processed_index(h5f, metadata_h5f):
                        return EventSlicer(h5f, metadata_h5f).get_events(start_us, end_us)
                    return load_raw_events_by_time(h5f, start_us, end_us)
            if has_processed_index(h5f, None):
                return EventSlicer(h5f).get_events(start_us, end_us)
            return load_raw_events_by_time(h5f, start_us, end_us)
    except OSError as exc:
        if hdf5plugin is None:
            raise RuntimeError(
                "Failed to read the HDF5 event file. This environment does not have "
                "hdf5plugin installed, which may be required for compressed event files."
            ) from exc
        raise


def main() -> None:
    """
    Visualize event-space de-rotation followed by fixed-window voxelization.
    """
    args = parse_args()
    sequence_dir = args.sequence_dir.resolve()
    if not sequence_dir.is_dir():
        raise NotADirectoryError(f"Not a sequence directory: {sequence_dir}")
    

    new_height, new_width = get_downsampled_size(
        original_height=args.height,
        original_width=args.width,
        downsampling_factor=args.downsampling_factor,
        patch_size=args.patch_size,
    )

    seq_info = build_sequence_info(
        sequence_dir=sequence_dir,
        new_height=new_height,
        new_width=new_width,
        original_height=args.height,
        original_width=args.width,
    )
    start_us, end_us, window_source = choose_window_us(args, seq_info["gt_timestamps_us"])
    duration_ms = (end_us - start_us) / 1000.0
    num_derotation_slices = resolve_derotation_slices(
        duration_ms=duration_ms,
        derotation_slices=args.derotation_slices,
        derotation_slice_ms=args.derotation_slice_ms,
    )
    derotation_context = build_derotation_context(
        seq_info=seq_info,
        ts_start_us=start_us,
        ts_end_us=end_us,
        num_bins=num_derotation_slices,
    )

    events = load_events(sequence_dir, start_us, end_us)
    if events is None or len(events["t"]) == 0:
        raise RuntimeError(f"No events found in window [{start_us}, {end_us}) us.")

    scale_x = new_width / args.width
    scale_y = new_height / args.height
    raw_x = events["x"].astype(np.float64) * scale_x
    raw_y = events["y"].astype(np.float64) * scale_y
    t_us = events["t"].astype(np.int64)
    polarity = events["p"]

    derotation_start = perf_counter()
    derot_x, derot_y, derot_valid, homographies = derotate_events_in_slices(
        x=raw_x,
        y=raw_y,
        t_us=t_us,
        ts_start_us=start_us,
        ts_end_us=end_us,
        context=derotation_context,
        width=new_width,
        height=new_height,
    )
    derotation_warp_s = perf_counter() - derotation_start

    raw_voxel_start = perf_counter()
    raw_voxel = raw_events_to_fixed_window_voxel(
        x=raw_x,
        y=raw_y,
        p=polarity,
        t_us=t_us,
        ts_start_us=start_us,
        ts_end_us=end_us,
        num_bins=args.num_bins,
        height=new_height,
        width=new_width,
    )
    raw_voxel_s = perf_counter() - raw_voxel_start

    derot_voxel_start = perf_counter()
    derot_voxel = raw_events_to_fixed_window_voxel(
        x=derot_x[derot_valid],
        y=derot_y[derot_valid],
        p=polarity[derot_valid],
        t_us=t_us[derot_valid],
        ts_start_us=start_us,
        ts_end_us=end_us,
        num_bins=args.num_bins,
        height=new_height,
        width=new_width,
    )
    derotation_voxel_s = perf_counter() - derot_voxel_start
    derotation_total_s = derotation_warp_s + derotation_voxel_s

    raw_total = count_nonzero_voxel_pixels(raw_voxel, args.voxel_eps)
    derot_total = count_nonzero_voxel_pixels(derot_voxel, args.voxel_eps)

    print(f"Sequence: {sequence_dir}")
    print(f"Window source: {window_source}")
    print(f"Window: [{start_us}, {end_us}) us ({duration_ms:.3f} ms)")
    print(f"Events loaded: {len(events['t'])}")
    print(
        "Event-space derotation: "
        f"{num_derotation_slices} slices "
        f"({duration_ms / num_derotation_slices:.3f} ms/slice), "
        f"valid warped events: {int(np.count_nonzero(derot_valid))}"
    )
    print(f"Final voxel bins: {args.num_bins}")
    print(
        "Timing: "
        f"raw voxel {raw_voxel_s * 1000.0:.2f} ms | "
        f"derotation warp {derotation_warp_s * 1000.0:.2f} ms | "
        f"derot voxel {derotation_voxel_s * 1000.0:.2f} ms | "
        f"derotation path total {derotation_total_s * 1000.0:.2f} ms"
    )
    print(
        "Timing per loaded event: "
        f"derotation path {derotation_total_s * 1e6 / len(events['t']):.3f} us/event"
    )
    print(f"Raw nonzero voxel pixels: {raw_total}")
    print(f"De-rotated nonzero voxel pixels: {derot_total}")
    print(f"Downsampled size: {new_height}x{new_width}")
    print("First derotation-slice homography:")
    print(np.array2string(homographies[0], precision=6, suppress_small=True))

    title = (
        f"{sequence_dir.name} | {duration_ms:.1f} ms | "
        f"{num_derotation_slices} derotation slices -> {args.num_bins} voxel bins"
    )
    raw_t_ms = (t_us.astype(np.float64) - float(start_us)) / 1000.0
    fig = make_raw_event_stream_plot(
        x=raw_x,
        y=raw_y,
        t_ms=raw_t_ms,
        polarity=polarity,
        width=new_width,
        height=new_height,
        duration_ms=duration_ms,
        title=f"{title} | raw sliced event stream",
        view_elev=args.view_elev,
        view_azim=args.view_azim,
        max_events=args.max_events,
    )
    bin_fig = make_voxel_bin_3d_plot(
        raw_voxel=raw_voxel,
        derot_voxel=derot_voxel,
        duration_ms=duration_ms,
        title=f"{title} | voxel bins as 3D planes",
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"Saved figure: {args.output}")
        bin_output = args.output.with_name(f"{args.output.stem}_bins{args.output.suffix}")
        bin_fig.savefig(bin_output, dpi=180)
        print(f"Saved voxel-bin figure: {bin_output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(bin_fig)


if __name__ == "__main__":
    main()
