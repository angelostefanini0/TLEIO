import argparse
from pathlib import Path
import sys

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
    normalize_quaternions,
)
from src.learning.dataloader.representation.event_slicer import EventSlicer
from src.learning.dataloader.representation.voxel_grid import VoxelGrid, quat_xyzw_to_rotmat

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
            "The de-rotation uses the same per-bin rotation homographies as voxelization."
        )
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help=(
            "Sequence directory containing events.h5, stamped_groundtruth.txt, and K.yaml. "
            "Processed sequences with ms_to_idx and raw EDS-style sequences are both supported."
        ),
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=100.0,
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
        help="Number of temporal bins used for the same de-rotation discretization as voxelization.",
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
    return parser.parse_args()


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


def homography_from_bin_to_ref(camera_matrix: np.ndarray, bin_quat_xyzw: np.ndarray, ref_quat_xyzw: np.ndarray) -> np.ndarray:
    K = np.asarray(camera_matrix, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    R_ref = quat_xyzw_to_rotmat(np.asarray(ref_quat_xyzw, dtype=np.float64))
    R_bin = quat_xyzw_to_rotmat(np.asarray(bin_quat_xyzw, dtype=np.float64))
    R_ref_from_bin = R_ref.T @ R_bin
    H_ref_from_bin = K @ R_ref_from_bin @ K_inv
    H_ref_from_bin /= H_ref_from_bin[2, 2]
    return H_ref_from_bin


def raw_events_to_fixed_window_voxel(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t_us: np.ndarray,
    ts_start_us: int,
    ts_end_us: int,
    num_bins: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Build the same pre-warp voxel raster that VoxelGrid.convert builds when
    derotation is enabled, using fixed window-relative time binning.
    """
    window_duration_us = float(ts_end_us - ts_start_us)
    if window_duration_us <= 0:
        raise ValueError("Window duration must be positive.")

    x_t = torch.from_numpy(x.astype(np.float32, copy=False))
    y_t = torch.from_numpy(y.astype(np.float32, copy=False))
    p_t = torch.from_numpy(p.astype(np.float32, copy=False))
    time_t = torch.from_numpy((t_us.astype(np.float32) - np.float32(ts_start_us)))

    with torch.no_grad():
        voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float32)
        t_norm = (num_bins - 1) * time_t / window_duration_us

        x0 = x_t.int()
        y0 = y_t.int()
        t0 = t_norm.int()
        value = 2 * p_t - 1

        for dx in [0, 1]:
            for dy in [0, 1]:
                for dt in [0, 1]:
                    xlim = x0 + dx
                    ylim = y0 + dy
                    tlim = t0 + dt

                    mask = (
                        (xlim < width)
                        & (xlim >= 0)
                        & (ylim < height)
                        & (ylim >= 0)
                        & (tlim < num_bins)
                        & (tlim >= 0)
                    )

                    interp_weights = (
                        value
                        * (1 - (xlim.float() - x_t).abs())
                        * (1 - (ylim.float() - y_t).abs())
                        * (1 - (tlim.float() - t_norm).abs())
                    )

                    index = height * width * tlim.long() + width * ylim.long() + xlim.long()
                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

    return voxel_grid


def derotate_voxel_with_training_path(raw_voxel: torch.Tensor, context: dict) -> torch.Tensor:
    """
    Call the same method used during training. This method internally uses
    cv2.warpPerspective for each temporal bin.
    """
    derotator = VoxelGrid(
        channels=raw_voxel.shape[0],
        height=raw_voxel.shape[1],
        width=raw_voxel.shape[2],
        derotate=True,
    )
    return derotator.derotate_voxel_grid(
        voxel_grid=raw_voxel,
        camera_matrix=np.asarray(context["camera_matrix"], dtype=np.float64),
        bin_quat_xyzw=np.asarray(context["bin_quat_xyzw"], dtype=np.float64),
        ref_quat_xyzw=np.asarray(context["ref_quat_xyzw"], dtype=np.float64),
    )


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


def configure_3d_axis(ax, title: str, width: int, height: int, duration_ms: float, elev: float, azim: float) -> None:
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_zlabel("time [ms]")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_zlim(0, duration_ms)
    ax.view_init(elev=elev, azim=azim)


def configure_2d_axis(ax, title: str, width: int, height: int) -> None:
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def make_plot(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    raw_t_ms: np.ndarray,
    raw_value: np.ndarray,
    derot_x: np.ndarray,
    derot_y: np.ndarray,
    derot_t_ms: np.ndarray,
    derot_value: np.ndarray,
    width: int,
    height: int,
    duration_ms: float,
    title: str,
    view_elev: float,
    view_azim: float,
) -> plt.Figure:
    raw_colors = polarity_colors(raw_value)
    derot_colors = polarity_colors(derot_value)
    fig = plt.figure(figsize=(14, 10))
    ax_raw_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_derot_3d = fig.add_subplot(2, 2, 2, projection="3d")
    ax_raw_2d = fig.add_subplot(2, 2, 3)
    ax_derot_2d = fig.add_subplot(2, 2, 4)

    scatter_kwargs = {
        "s": 1.5,
        "alpha": 0.35,
        "linewidths": 0,
        "depthshade": False,
    }
    ax_raw_3d.scatter(raw_x, raw_y, raw_t_ms, c=raw_colors, **scatter_kwargs)
    ax_derot_3d.scatter(derot_x, derot_y, derot_t_ms, c=derot_colors, **scatter_kwargs)

    configure_3d_axis(ax_raw_3d, "Raw events", width, height, duration_ms, view_elev, view_azim)
    configure_3d_axis(ax_derot_3d, "De-rotated events", width, height, duration_ms, view_elev, view_azim)

    ax_raw_2d.scatter(raw_x, raw_y, c=raw_colors, s=1.5, alpha=0.35, linewidths=0)
    ax_derot_2d.scatter(derot_x, derot_y, c=derot_colors, s=1.5, alpha=0.35, linewidths=0)
    configure_2d_axis(ax_raw_2d, "Raw x/y projection", width, height)
    configure_2d_axis(ax_derot_2d, "De-rotated x/y projection", width, height)

    fig.suptitle(title)
    fig.tight_layout()
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
    Script to match the training path to visualize the exact same derotation
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
    context = build_derotation_context(
        seq_info=seq_info,
        ts_start_us=start_us,
        ts_end_us=end_us,
        num_bins=args.num_bins,
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
    derot_voxel = derotate_voxel_with_training_path(raw_voxel, context)

    duration_ms = (end_us - start_us) / 1000.0
    raw_plot_x, raw_plot_y, raw_plot_t, raw_plot_value, raw_total = voxel_to_points(
        raw_voxel,
        duration_ms=duration_ms,
        max_points=args.max_events,
        eps=args.voxel_eps,
    )
    derot_plot_x, derot_plot_y, derot_plot_t, derot_plot_value, derot_total = voxel_to_points(
        derot_voxel,
        duration_ms=duration_ms,
        max_points=args.max_events,
        eps=args.voxel_eps,
    )
    homographies = [
        homography_from_bin_to_ref(
            camera_matrix=context["camera_matrix"],
            bin_quat_xyzw=context["bin_quat_xyzw"][idx],
            ref_quat_xyzw=context["ref_quat_xyzw"],
        )
        for idx in range(args.num_bins)
    ]

    print(f"Sequence: {sequence_dir}")
    print(f"Window source: {window_source}")
    print(f"Window: [{start_us}, {end_us}) us ({duration_ms:.3f} ms)")
    print(f"Events loaded: {len(events['t'])}")
    print(f"Raw nonzero voxel pixels: {raw_total} (plotted {len(raw_plot_x)})")
    print(f"De-rotated nonzero voxel pixels: {derot_total} (plotted {len(derot_plot_x)})")
    print(f"Downsampled size: {new_height}x{new_width}")
    print("First bin homography:")
    print(np.array2string(homographies[0], precision=6, suppress_small=True))

    title = (
        f"{sequence_dir.name} | {duration_ms:.1f} ms | "
        f"{args.num_bins} bins | cv2.warpPerspective derotation"
    )
    fig = make_plot(
        raw_x=raw_plot_x,
        raw_y=raw_plot_y,
        raw_t_ms=raw_plot_t,
        raw_value=raw_plot_value,
        derot_x=derot_plot_x,
        derot_y=derot_plot_y,
        derot_t_ms=derot_plot_t,
        derot_value=derot_plot_value,
        width=new_width,
        height=new_height,
        duration_ms=duration_ms,
        title=title,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"Saved figure: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
