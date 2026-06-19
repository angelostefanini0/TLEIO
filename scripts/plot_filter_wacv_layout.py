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


def load_anchor_poses(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = load_trajectory(path)
    return table[:, 0], table[:, 1:4], table[:, 4:8]


def load_relative_motion_table(path: Path) -> np.ndarray:
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
    if table.ndim != 2 or table.shape[1] < 5:
        raise ValueError(f"{path} has shape {table.shape}, expected at least N x 5.")
    return table


def quat_to_matrix_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat / np.linalg.norm(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def compute_net_only_trajectory(sequence: str, gt_root: Path, net_root: Path | None) -> np.ndarray | None:
    sequence_dir = gt_root / sequence
    anchor_path = sequence_dir / "anchor_poses.txt"
    if not anchor_path.exists():
        anchor_path = sequence_dir / "stamped_groundtruth.txt"
    if net_root is not None:
        rel_path = net_root / f"{sequence}.txt"
    else:
        candidates = [
            sequence_dir / f"{sequence}.txt",
            ROOT / "data" / "tartanair" / "processed" / sequence / f"{sequence}.txt",
        ]
        rel_path = next((path for path in candidates if path.exists()), candidates[0])
    if not anchor_path.exists() or not rel_path.exists():
        print(f"skip net-only for {sequence}: missing {anchor_path if not anchor_path.exists() else rel_path}")
        return None

    anchor_times_s, anchor_positions, anchor_quats = load_anchor_poses(anchor_path)
    rel_table = load_relative_motion_table(rel_path)
    rel_dp = rel_table[:, 2:5]
    limit = min(len(rel_dp), len(anchor_times_s) - 1)
    if limit <= 0:
        return None

    positions = [anchor_positions[0].astype(np.float64)]
    quats = [anchor_quats[0].astype(np.float64)]
    for idx in range(limit):
        positions.append(positions[-1] + quat_to_matrix_xyzw(quats[-1]) @ rel_dp[idx])
        quats.append(anchor_quats[idx + 1].astype(np.float64))

    return np.column_stack(
        [
            anchor_times_s[: limit + 1],
            np.asarray(positions, dtype=np.float64),
            np.asarray(quats, dtype=np.float64),
        ]
    )


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
    title_size = 9.8 * font_scale
    label_size = 10.8 * font_scale
    tick_size = 8.4 * font_scale
    legend_size = 9.8 * font_scale
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.5,
            "lines.linewidth": line_width,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def set_equal_3d(ax, *position_sets: np.ndarray) -> None:
    values = np.vstack([positions for positions in position_sets if positions is not None])
    mins = np.min(values, axis=0)
    maxs = np.max(values, axis=0)
    mids = 0.5 * (mins + maxs)
    radius = 0.55 * max(float(np.max(maxs - mins)), 1e-6)
    ax.set_xlim(mids[0] - radius, mids[0] + radius)
    ax.set_ylim(mids[1] - radius, mids[1] + radius)
    ax.set_zlim(mids[2] - radius, mids[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def plot_sequence(
    sequence: str,
    gt_table: np.ndarray,
    est_table: np.ndarray,
    out_path: Path,
    ate_aligned: bool,
    dpi: int,
    line_width: float,
    net_table: np.ndarray | None = None,
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

    net_positions = None
    if net_table is not None:
        if ate_aligned:
            net_positions = rpg_ate_aligned_positions(gt_table, net_table)
            net_times = net_table[:, 0]
            min_len = min(len(net_times), len(net_positions))
            net_times = net_times[:min_len]
            net_positions = net_positions[:min_len]
        else:
            net_times = net_table[:, 0]
            net_positions = net_table[:, 1:4]
        net_positions = np.column_stack(
            [np.interp(est_times, net_times, net_positions[:, axis]) for axis in range(3)]
        )

    t_rel = est_times - est_times[0]

    fig = plt.figure(figsize=(9.9, 3.85))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=(1.58, 1.0),
        left=0.066,
        right=0.990,
        bottom=0.125,
        top=0.930,
        hspace=0.34,
        wspace=0.205,
    )
    axes = [fig.add_subplot(gs[row, 0]) for row in range(3)]
    traj3d_ax = fig.add_subplot(gs[:, 1], projection="3d")

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
        if net_positions is not None:
            axes[axis_idx].plot(
                t_rel,
                net_positions[:, axis_idx],
                color="tab:red",
                linewidth=line_width,
                label="EventsFormer",
            )
        axes[axis_idx].set_ylabel(f"{label} [m]", fontweight="bold", labelpad=3)
        axes[axis_idx].grid(True)
        axes[axis_idx].margins(x=0.01)
        axes[axis_idx].tick_params(axis="both", pad=2.0, width=0.8)
        if axis_idx < 2:
            axes[axis_idx].tick_params(labelbottom=False)
        else:
            axes[axis_idx].set_xlabel("Time [s]", fontweight="bold", labelpad=4)

    fig.align_ylabels(axes)
    traj3d_ax.plot(
        gt_positions[:, 0],
        gt_positions[:, 1],
        gt_positions[:, 2],
        color=colors["gt"],
        linewidth=line_width,
        label="Ground Truth",
    )
    traj3d_ax.plot(
        est_positions[:, 0],
        est_positions[:, 1],
        est_positions[:, 2],
        color=colors["tleio"],
        linewidth=line_width,
        label="TLEIO",
    )
    if net_positions is not None:
        traj3d_ax.plot(
            net_positions[:, 0],
            net_positions[:, 1],
            net_positions[:, 2],
            color="tab:red",
            linewidth=line_width,
            label="EventsFormer",
        )
    traj3d_ax.scatter(
        gt_positions[-1, 0],
        gt_positions[-1, 1],
        gt_positions[-1, 2],
        color="red",
        marker="x",
        s=28,
        linewidths=1.4,
        zorder=5,
    )
    traj3d_ax.set_title(f"3D Trajectory ({short_sequence_name(sequence)})", pad=3)
    traj3d_ax.set_xlabel("X [m]", fontweight="bold", labelpad=4)
    traj3d_ax.set_ylabel("Y [m]", fontweight="bold", labelpad=4)
    traj3d_ax.set_zlabel("Z [m]", fontweight="bold", labelpad=4)
    traj3d_ax.grid(True)
    traj3d_ax.tick_params(axis="both", pad=2.0, width=0.8)
    set_equal_3d(traj3d_ax, gt_positions, est_positions, net_positions)
    traj3d_ax.view_init(elev=28, azim=-62)
    traj3d_ax.legend(loc="upper right", frameon=True, framealpha=0.92, borderpad=0.45, handlelength=2.2)

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
    parser.add_argument("--net-root", type=Path, default=None, help="Optional root with <sequence>.txt net predictions.")
    parser.add_argument("--plot-net-only", action="store_true", help="Overlay net-only trajectory in red.")
    parser.add_argument("--no-net-only", action="store_true", help="Disable the default net-only red overlay.")
    parser.add_argument("--font-scale", type=float, default=1.0, help="Scale all plot fonts.")
    parser.add_argument("--line-width", type=float, default=2.1, help="Trajectory line width.")
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
        net_table = (
            compute_net_only_trajectory(sequence, args.gt_root, args.net_root)
            if args.plot_net_only or not args.no_net_only
            else None
        )
        plot_sequence(
            sequence=sequence,
            gt_table=load_trajectory(gt_path),
            est_table=load_trajectory(est_path),
            out_path=out_path,
            ate_aligned=not args.no_ate,
            dpi=args.dpi,
            line_width=args.line_width,
            net_table=net_table,
        )
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
