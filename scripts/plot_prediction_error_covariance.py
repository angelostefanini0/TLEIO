"""Plot EventsFormer translation error against predicted covariance/sigma.

The script compares network relative-motion predictions with the ground-truth
relative motions used by the precomputed TartanAir competition split. Prediction
files are expected to contain:

    t0_us t1_us px py pz sigma_x sigma_y sigma_z

It writes a compact 4x4 summary grid for all sequences plus optional detailed
per-sequence figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEQUENCES = [
    "competition_Test_ME000",
    "competition_Test_ME001",
    "competition_Test_ME002",
    "competition_Test_ME003",
    "competition_Test_ME004",
    "competition_Test_ME005",
    "competition_Test_ME006",
    "competition_Test_ME007",
    "competition_Test_MH000",
    "competition_Test_MH001",
    "competition_Test_MH002",
    "competition_Test_MH003",
    "competition_Test_MH004",
    "competition_Test_MH005",
    "competition_Test_MH006",
    "competition_Test_MH007",
]

COLORS = {
    "error": "#2F78D4",
    "sigma": "#BBD7F0",
    "cone": "#8A2BE2",
    "x": "#1F77B4",
    "y": "#2CA02C",
    "z": "#D99A00",
    "grid": "0.82",
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_table(path: Path, min_cols: int) -> np.ndarray:
    rows = []
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
    if table.ndim != 2 or table.shape[1] < min_cols:
        raise ValueError(f"{path} has shape {table.shape}, expected at least {min_cols} columns.")
    return table


def align_by_timestamps(gt_table: np.ndarray, pred_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_lookup = {
        (int(round(row[0])), int(round(row[1]))): row
        for row in gt_table
    }

    gt_rows = []
    pred_rows = []
    for row in pred_table:
        key = (int(round(row[0])), int(round(row[1])))
        gt_row = gt_lookup.get(key)
        if gt_row is None:
            continue
        gt_rows.append(gt_row)
        pred_rows.append(row)

    if not gt_rows:
        raise ValueError("No matching timestamp pairs between GT and predictions.")

    return np.asarray(gt_rows, dtype=np.float64), np.asarray(pred_rows, dtype=np.float64)


def sequence_stats(gt_table: np.ndarray, pred_table: np.ndarray) -> dict[str, np.ndarray | float]:
    gt, pred = align_by_timestamps(gt_table, pred_table)
    time_s = (pred[:, 1] - pred[0, 0]) * 1e-6
    error_xyz = pred[:, 2:5] - gt[:, 2:5]
    abs_error_xyz = np.abs(error_xyz)
    error_norm = np.linalg.norm(error_xyz, axis=1)
    sigma_xyz = pred[:, 5:8]
    sigma_norm = np.linalg.norm(sigma_xyz, axis=1)
    rmse = float(np.sqrt(np.mean(error_norm**2)))
    mean_sigma = float(np.mean(sigma_norm))
    corr = float(np.corrcoef(error_norm, sigma_norm)[0, 1]) if len(error_norm) > 1 else np.nan
    coverage_3sigma = float(np.mean(np.abs(error_xyz) <= 3.0 * np.maximum(sigma_xyz, 1e-12)))
    normalized_abs_error = np.abs(error_xyz) / np.maximum(sigma_xyz, 1e-12)
    median_normalized_error = float(np.median(normalized_abs_error))
    return {
        "time_s": time_s,
        "error_xyz": error_xyz,
        "abs_error_xyz": abs_error_xyz,
        "error_norm": error_norm,
        "sigma_xyz": sigma_xyz,
        "sigma_norm": sigma_norm,
        "rmse": rmse,
        "mean_sigma": mean_sigma,
        "corr": corr,
        "coverage_3sigma": coverage_3sigma,
        "median_normalized_error": median_normalized_error,
    }


def short_name(sequence: str) -> str:
    return sequence.replace("competition_Test_", "")


def style_axis(ax) -> None:
    ax.grid(True, color=COLORS["grid"], linewidth=0.55, alpha=0.85)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_linewidth(0.8)


def plot_summary_grid(stats_by_sequence: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(13.2, 8.4), sharex=False, sharey=False)
    handles = None
    labels = None

    for ax, sequence in zip(axes.flat, SEQUENCES):
        stats = stats_by_sequence[sequence]
        time_s = stats["time_s"]
        error_norm = stats["error_norm"]
        sigma_norm = stats["sigma_norm"]

        err_line, = ax.plot(
            time_s,
            error_norm,
            color=COLORS["error"],
            linewidth=1.55,
            label="Error norm",
        )
        sig_line, = ax.plot(
            time_s,
            sigma_norm,
            color=COLORS["sigma"],
            linewidth=1.45,
            label="Pred. sigma norm",
        )
        ax.set_title(
            f"{short_name(sequence)}  RMSE={stats['rmse']:.2f} m",
            fontsize=10.5,
            pad=3,
        )
        style_axis(ax)
        if handles is None:
            handles = [err_line, sig_line]
            labels = [line.get_label() for line in handles]

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]", fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Magnitude [m]", fontweight="bold")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="0.7",
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=0.8, w_pad=0.75)
    save_all(fig, out_path)


def plot_sequence_detail(sequence: str, stats: dict, out_path: Path, n_sigma: float) -> None:
    time_s = stats["time_s"]
    error_xyz = stats["error_xyz"]
    sigma_xyz = stats["sigma_xyz"]
    labels = ("x", "y", "z")

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 7.1), sharex=True)
    fig.suptitle(
        f"Translation Error with Predicted {n_sigma:g}$\\sigma$ Uncertainty vs GT relative motions ({short_name(sequence)})",
        fontsize=13,
        y=0.985,
    )

    for axis_idx, (ax, label) in enumerate(zip(axes, labels)):
        error = error_xyz[:, axis_idx]
        sigma = n_sigma * sigma_xyz[:, axis_idx]
        line_color = "#2F78D4"
        band_color = "#BBD7F0"

        ax.fill_between(
            time_s,
            -sigma,
            sigma,
            color=band_color,
            alpha=0.72,
            linewidth=0.0,
            label=fr"$\pm {n_sigma:g}\sigma_{label}$",
        )
        ax.plot(
            time_s,
            error,
            color=line_color,
            linewidth=1.25,
            label=f"error_{label}",
            zorder=3,
        )
        ax.axhline(0.0, color="0.35", linewidth=0.7, alpha=0.75)
        ax.set_ylabel(f"e{label} [m]")
        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.88,
            facecolor="white",
            edgecolor="0.8",
            borderpad=0.3,
            handlelength=1.8,
        )
        style_axis(ax)

        max_abs = float(np.nanmax(np.abs(np.concatenate([error, sigma]))))
        if np.isfinite(max_abs) and max_abs > 0:
            limit = 1.12 * max_abs
            ax.set_ylim(-limit, limit)

    axes[-1].set_xlabel("time [s]")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=0.75)
    save_all(fig, out_path)


def plot_sequence_cones(sequence: str, stats: dict, out_path: Path, n_sigma: float) -> None:
    error_xyz = stats["error_xyz"]
    sigma_xyz = stats["sigma_xyz"]
    labels = ("X", "Y", "Z")
    outside_rates = []

    fig, axes = plt.subplots(1, 3, figsize=(8.7, 2.75), sharey=True)
    for axis_idx, (ax, label) in enumerate(zip(axes, labels)):
        error = error_xyz[:, axis_idx]
        sigma = sigma_xyz[:, axis_idx]
        outside = np.abs(error) > n_sigma * np.maximum(sigma, 1e-12)
        outside_rates.append(100.0 * float(np.mean(outside)))

        ax.scatter(
            error,
            sigma,
            s=3.0,
            color="#1F77B4",
            alpha=0.55,
            linewidths=0.0,
            rasterized=True,
        )

        max_abs_error = max(float(np.nanmax(np.abs(error))), 1e-6)
        max_sigma = max(float(np.nanmax(sigma)), max_abs_error / n_sigma, 1e-6)
        x_lim = 1.08 * max_abs_error
        y_lim = 1.10 * max_sigma
        cone_x = np.asarray([-x_lim, 0.0, x_lim])
        cone_y = np.abs(cone_x) / n_sigma
        ax.plot(cone_x[:2], cone_y[:2], color="red", linestyle="--", linewidth=0.85, alpha=0.85)
        ax.plot(cone_x[1:], cone_y[1:], color="red", linestyle="--", linewidth=0.85, alpha=0.85)

        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(0.0, y_lim)
        ax.set_xlabel(f"Error {label}", fontsize=12)
        ax.set_title(f"{short_name(sequence)} {label}", fontsize=10.5, pad=3)
        style_axis(ax)

    axes[0].set_ylabel("Sigmas", fontsize=12)
    fig.suptitle(
        (
            f"Network uncertainty cones ({short_name(sequence)}): "
            f"outside {n_sigma:g}$\\sigma$ = "
            f"x {outside_rates[0]:.2f}%, y {outside_rates[1]:.2f}%, z {outside_rates[2]:.2f}%"
        ),
        fontsize=11.5,
        y=1.02,
    )

    fig.tight_layout(pad=0.35, w_pad=0.6)
    save_all(fig, out_path)


def choose_cherry_pick_sequence(stats_by_sequence: dict[str, dict]) -> str:
    """Pick a sequence where uncertainty is visually meaningful and calibrated."""

    best_sequence = None
    best_score = -np.inf
    for sequence, stats in stats_by_sequence.items():
        corr = stats["corr"]
        if not np.isfinite(corr):
            corr = 0.0
        coverage = stats["coverage_3sigma"]
        median_norm = stats["median_normalized_error"]
        rmse = stats["rmse"]
        mean_sigma = stats["mean_sigma"]

        # Prefer calibrated bands with visible dynamics, not trivially huge sigma.
        score = (
            2.0 * min(coverage, 0.99)
            + 0.8 * max(corr, 0.0)
            - 0.35 * abs(median_norm - 1.0)
            - 0.08 * mean_sigma
            + 0.03 * rmse
        )
        if score > best_score:
            best_score = score
            best_sequence = sequence

    assert best_sequence is not None
    return best_sequence


def plot_calibration_scatter(stats_by_sequence: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.7, 3.6))
    rmse = np.asarray([stats_by_sequence[seq]["rmse"] for seq in SEQUENCES])
    mean_sigma = np.asarray([stats_by_sequence[seq]["mean_sigma"] for seq in SEQUENCES])
    colors = ["#D99A00" if "_ME" in seq else "#8A2BE2" for seq in SEQUENCES]

    ax.scatter(mean_sigma, rmse, c=colors, s=42, edgecolor="black", linewidth=0.45, zorder=3)
    for seq, x_val, y_val in zip(SEQUENCES, mean_sigma, rmse):
        ax.annotate(short_name(seq).replace("00", "0"), (x_val, y_val), fontsize=7.8, xytext=(3, 2), textcoords="offset points")

    ax.set_xlabel("Mean predicted sigma norm [m]", fontweight="bold")
    ax.set_ylabel("Prediction RMSE [m]", fontweight="bold")
    style_axis(ax)
    fig.tight_layout(pad=0.35)
    save_all(fig, out_path)


def save_all(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("data/tartanair/predicted_relative_motions/vggt_massive_v6_1_e70_covariance_tartan_test"),
    )
    parser.add_argument("--gt-root", type=Path, default=Path("data/tartanair/precomputed_test"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/tartanair_prediction_error_covariance"))
    parser.add_argument("--no-details", action="store_true", help="Only save the all-sequence summary figures.")
    parser.add_argument("--n-sigma", type=float, default=3.0)
    parser.add_argument(
        "--cherry-pick",
        action="store_true",
        help="Save only the best sequence for a paper figure, selected from calibration/coverage.",
    )
    parser.add_argument("--sequence", type=str, default=None, help="Force a specific sequence instead of auto cherry-pick.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    stats_by_sequence = {}
    for sequence in SEQUENCES:
        gt_path = args.gt_root / sequence / "relative_motions.txt"
        pred_path = args.pred_root / f"{sequence}.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing GT relative motions: {gt_path}")
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")

        gt_table = load_table(gt_path, min_cols=5)
        pred_table = load_table(pred_path, min_cols=8)
        stats_by_sequence[sequence] = sequence_stats(gt_table, pred_table)

    plot_summary_grid(stats_by_sequence, args.out_dir / "all_sequences_error_vs_covariance")
    plot_calibration_scatter(stats_by_sequence, args.out_dir / "sequence_rmse_vs_mean_sigma")

    selected_sequence = args.sequence or choose_cherry_pick_sequence(stats_by_sequence)
    selected_stats = stats_by_sequence[selected_sequence]
    plot_sequence_detail(
        selected_sequence,
        selected_stats,
        args.out_dir / f"paper_{selected_sequence}_error_vs_{args.n_sigma:g}sigma",
        n_sigma=args.n_sigma,
    )
    plot_sequence_cones(
        selected_sequence,
        selected_stats,
        args.out_dir / f"paper_{selected_sequence}_error_cones_{args.n_sigma:g}sigma",
        n_sigma=args.n_sigma,
    )

    if not args.no_details and not args.cherry_pick:
        for sequence, stats in stats_by_sequence.items():
            plot_sequence_detail(
                sequence,
                stats,
                args.out_dir / "per_sequence" / f"{sequence}_error_vs_{args.n_sigma:g}sigma",
                n_sigma=args.n_sigma,
            )

    summary_path = args.out_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("sequence,rmse_m,mean_sigma_norm_m,error_sigma_corr,coverage_3sigma,median_normalized_error\n")
        for sequence in SEQUENCES:
            stats = stats_by_sequence[sequence]
            handle.write(
                f"{sequence},{stats['rmse']:.9f},{stats['mean_sigma']:.9f},{stats['corr']:.9f},"
                f"{stats['coverage_3sigma']:.9f},{stats['median_normalized_error']:.9f}\n"
            )

    print(f"Saved plots to: {args.out_dir.resolve()}")
    print(f"Saved summary to: {summary_path.resolve()}")
    print(f"Cherry-picked sequence: {selected_sequence}")


if __name__ == "__main__":
    main()
