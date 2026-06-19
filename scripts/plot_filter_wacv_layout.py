"""Generate WACV-style GT/TLEIO trajectory plots directly from trajectory txt files."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_SRC = ROOT / "evaluation" / "rpg_trajectory_evaluation" / "src" / "rpg_trajectory_evaluation"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))


def infer_time_scale_to_seconds(timestamps: np.ndarray) -> float:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    diffs = np.diff(np.sort(timestamps))
    diffs = diffs[diffs > 0.0]
    median_dt = float(np.median(diffs)) if len(diffs) else 0.0
    if median_dt > 1e7:
        return 1e-9
    if median_dt > 1e1:
        return 1e-6
    return 1.0


def load_trajectory(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue

    table = np.asarray(rows, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 8:
        raise ValueError(f"{path} has shape {table.shape}, expected N x 8: t px py pz qx qy qz qw.")
    table = table[np.argsort(table[:, 0])]
    table[:, 0] *= infer_time_scale_to_seconds(table[:, 0])
    return table


def interpolate_positions(reference: np.ndarray, query_times_s: np.ndarray) -> np.ndarray:
    ref_t = reference[:, 0]
    ref_p = reference[:, 1:4]
    return np.column_stack([np.interp(query_times_s, ref_t, ref_p[:, axis]) for axis in range(3)])


def rpg_ate_aligned_positions(ground_truth: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    from trajectory import Trajectory

    with tempfile.TemporaryDirectory(prefix="wacv_rpg_align_", dir=ROOT) as temp_dir:
        eval_dir = Path(temp_dir)
        np.savetxt(eval_dir / "stamped_groundtruth.txt", ground_truth, fmt="%.9f")
        np.savetxt(eval_dir / "stamped_traj_estimate.txt", estimate, fmt="%.9f")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            traj = Trajectory(str(eval_dir), est_type="traj_est")
            if not traj.data_loaded:
                raise RuntimeError("RPG trajectory loader failed.")
            traj.compute_absolute_error()
            return np.asarray(traj.p_es_aligned, dtype=np.float64)


def short_sequence_name(sequence: str) -> str:
    return sequence.replace("competition_Test_", "")


def configure_style(font_scale: float, line_width: float) -> None:
    title_size = 10.5 * font_scale
    label_size = 11.5 * font_scale
    tick_size = 9.0 * font_scale
    legend_size = 10.0 * font_scale
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "lines.linewidth": line_width,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def set_equal_xy(ax, gt_xy: np.ndarray, est_xy: np.ndarray) -> None:
    values = np.vstack([gt_xy, est_xy])
    x_min, y_min = np.min(values, axis=0)
    x_max, y_max = np.max(values, axis=0)
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    radius = 0.55 * max(x_max - x_min, y_max - y_min, 1e-6)
    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_aspect("equal", adjustable="box")


def plot_sequence(
    sequence: str,
    gt_table: np.ndarray,
    est_table: np.ndarray,
    out_path: Path,
    ate_aligned: bool,
    dpi: int,
    line_width: float,
) -> None:
    est_times = est_table[:, 0]
    gt_positions = interpolate_positions(gt_table, est_times)
    if ate_aligned:
        est_positions = rpg_ate_aligned_positions(gt_table, est_table)
        min_len = min(len(est_times), len(gt_positions), len(est_positions))
        est_times = est_times[:min_len]
        gt_positions = gt_positions[:min_len]
        est_positions = est_positions[:min_len]
    else:
        est_positions = est_table[:, 1:4]

    t_rel = est_times - est_times[0]

    fig = plt.figure(figsize=(12.0, 4.55))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=(1.58, 1.0),
        left=0.070,
        right=0.985,
        bottom=0.120,
        top=0.940,
        hspace=0.34,
        wspace=0.205,
    )
    axes = [fig.add_subplot(gs[row, 0]) for row in range(3)]
    xy_ax = fig.add_subplot(gs[:, 1])

    colors = {"gt": "tab:blue", "tleio": "tab:green"}
    labels = ("X", "Y", "Z")
    for axis_idx, label in enumerate(labels):
        axes[axis_idx].plot(
            t_rel,
            gt_positions[:, axis_idx],
            color=colors["gt"],
            linewidth=line_width,
            label="Ground Truth",
        )
        axes[axis_idx].plot(
            t_rel,
            est_positions[:, axis_idx],
            color=colors["tleio"],
            linewidth=line_width,
            label="TLEIO",
        )
        axes[axis_idx].set_title(f"{label} Position", pad=2)
        axes[axis_idx].set_ylabel(f"{label} [m]", fontweight="bold", labelpad=4)
        axes[axis_idx].grid(True)
        axes[axis_idx].margins(x=0.01)
        axes[axis_idx].tick_params(axis="both", pad=2.5, width=0.8)
        if axis_idx < 2:
            axes[axis_idx].tick_params(labelbottom=False)
        else:
            axes[axis_idx].set_xlabel("Time [s]", fontweight="bold", labelpad=4)

    axes[0].legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        borderpad=0.35,
        handlelength=2.1,
        columnspacing=1.4,
    )
    fig.align_ylabels(axes)
    xy_ax.plot(gt_positions[:, 0], gt_positions[:, 1], color=colors["gt"], linewidth=line_width, label="Ground Truth")
    xy_ax.plot(est_positions[:, 0], est_positions[:, 1], color=colors["tleio"], linewidth=line_width, label="TLEIO")
    xy_ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], color="red", marker="x", s=28, linewidths=1.4, zorder=5)
    xy_ax.set_title(f"XY Projection ({short_sequence_name(sequence)})", pad=3)
    xy_ax.set_xlabel("X [m]", fontweight="bold", labelpad=4)
    xy_ax.set_ylabel("Y [m]", fontweight="bold", labelpad=4)
    xy_ax.grid(True)
    xy_ax.tick_params(axis="both", pad=2.5, width=0.8)
    set_equal_xy(xy_ax, gt_positions[:, :2], est_positions[:, :2])
    xy_ax.legend(loc="upper right", frameon=True, borderpad=0.45, handlelength=2.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_sequence_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def find_sequences(est_root: Path, sequence_arg: str | None) -> list[str]:
    requested = parse_sequence_arg(sequence_arg)
    if requested is not None:
        return requested
    return sorted(
        path.name
        for path in est_root.iterdir()
        if path.is_dir() and (path / "stamped_traj_estimate.txt").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True, help="Root with <sequence>/stamped_groundtruth.txt.")
    parser.add_argument("--est-root", type=Path, required=True, help="Root with <sequence>/stamped_traj_estimate.txt.")
    parser.add_argument("--out-root", type=Path, default=None, help="Output root. Defaults to --est-root.")
    parser.add_argument("--sequence", type=str, default=None, help="Optional comma-separated sequence list.")
    parser.add_argument("--suffix", default="_wacv_direct_ate", help="Output filename suffix.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--no-ate", action="store_true", help="Plot raw TLEIO instead of RPG ATE-aligned TLEIO.")
    parser.add_argument("--font-scale", type=float, default=1.0, help="Scale all plot fonts.")
    parser.add_argument("--line-width", type=float, default=1.8, help="Trajectory line width.")
    args = parser.parse_args()

    configure_style(args.font_scale, args.line_width)
    out_root = args.out_root if args.out_root is not None else args.est_root
    sequences = find_sequences(args.est_root, args.sequence)
    if not sequences:
        raise FileNotFoundError(f"No stamped_traj_estimate.txt files found under {args.est_root}")

    for sequence in sequences:
        gt_path = args.gt_root / sequence / "stamped_groundtruth.txt"
        est_path = args.est_root / sequence / "stamped_traj_estimate.txt"
        if not gt_path.exists():
            print(f"skip {sequence}: missing {gt_path}")
            continue
        if not est_path.exists():
            print(f"skip {sequence}: missing {est_path}")
            continue

        out_path = out_root / sequence / f"{sequence}{args.suffix}.{args.format}"
        plot_sequence(
            sequence=sequence,
            gt_table=load_trajectory(gt_path),
            est_table=load_trajectory(est_path),
            out_path=out_path,
            ate_aligned=not args.no_ate,
            dpi=args.dpi,
            line_width=args.line_width,
        )
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
