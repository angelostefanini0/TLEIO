"""Run TartanAir competition filters and plot GT vs ATE-aligned estimates only."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from filter_diagnostics import get_rpg_ate_and_aligned_trajectory
from main_filter import CONFIG, RunnerConfig, run_filter


def load_best_params(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data.get("best", {}).get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"No best.params object in {path}")
    valid_fields = {field.name for field in fields(RunnerConfig)}
    return {key: value for key, value in params.items() if key in valid_fields}


def save_gt_filter_ate_plot(
    path: Path,
    ground_truth: np.ndarray,
    aligned_est_positions: np.ndarray,
    ate_rmse_m: float,
) -> Path:
    gt = np.asarray(ground_truth, dtype=np.float64)
    min_len = min(len(gt), len(aligned_est_positions))
    gt_positions = gt[:min_len, 1:4]
    est_positions = aligned_est_positions[:min_len]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        gt_positions[:, 0],
        gt_positions[:, 1],
        gt_positions[:, 2],
        label="Ground Truth",
        color="tab:blue",
        linewidth=2.0,
    )
    ax.plot(
        est_positions[:, 0],
        est_positions[:, 1],
        est_positions[:, 2],
        label=f"Filter ATE Aligned (RMSE {ate_rmse_m:.3f} m)",
        color="tab:red",
        linestyle="-.",
        linewidth=2.0,
    )
    ax.scatter(*gt_positions[0], color="black", marker="o", s=45, label="Start")
    ax.scatter(*gt_positions[-1], color="red", marker="x", s=45, label="End")
    ax.set_title("GT vs Filter ATE Aligned")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()

    max_range = np.ptp(gt_positions, axis=0).max() / 2.0
    mid = (gt_positions.max(axis=0) + gt_positions.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TartanAir competition filter with optimized params and plot GT vs ATE-aligned filter."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "tartanair" / "processed_test",
    )
    parser.add_argument(
        "--params-root",
        type=Path,
        default=ROOT / "outputs" / "tuning_tartanair_v2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "tartanair_competition_optimized_ate",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Sequence to run. Can be provided multiple times. Defaults to all competition_Test_* sequences.",
    )
    parser.add_argument(
        "--relative-motions-file",
        type=str,
        default=None,
        help="Optional prediction path relative to each processed sequence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = (
        args.sequence
        if args.sequence
        else sorted(path.name for path in args.processed_dir.glob("competition_Test_*") if path.is_dir())
    )
    if not sequences:
        raise RuntimeError(f"No competition_Test_* sequences found in {args.processed_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequence in sequences:
        params_path = args.params_root / sequence / "best_filter_params.json"
        if not params_path.is_file():
            print(f"Skipping {sequence}: missing {params_path}")
            continue

        print(f"Running {sequence}")
        params = load_best_params(params_path)
        sequence_out = args.output_root / sequence
        config = replace(
            CONFIG,
            dataset="tartanair",
            sequence=sequence,
            processed_dir=args.processed_dir,
            relative_motion_filename=args.relative_motions_file,
            plot_transformer=False,
            plot_imu=False,
            plot_projections=False,
            plot_ate=False,
            plot_aa_transformer=False,
            interactive_plot=False,
            save_trajectory_file=True,
            save_diagnostic_plots=False,
            **params,
        )
        results = run_filter(config)
        ate_rmse_m, aligned_positions = get_rpg_ate_and_aligned_trajectory(
            results["ground_truth"],
            results["trajectory"],
        )
        trajectory_path = sequence_out / "stamped_traj_estimate.txt"
        sequence_out.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            trajectory_path,
            results["trajectory"],
            fmt="%.9f",
            header="timestamp_s px py pz qx qy qz qw",
            comments="",
        )
        plot_path = save_gt_filter_ate_plot(
            sequence_out / f"{sequence}_gt_filter_ate_aligned.png",
            results["ground_truth"],
            aligned_positions,
            ate_rmse_m,
        )
        rows.append(
            {
                "sequence": sequence,
                "ate_rmse_m": ate_rmse_m,
                "num_anchors": results["num_anchors"],
                "num_updates_attempted": results["num_updates_attempted"],
                "num_updates_rejected": results["num_updates_rejected"],
                "trajectory_path": str(trajectory_path),
                "plot_path": str(plot_path),
                "params_path": str(params_path),
            }
        )
        print(f"  ATE RMSE: {ate_rmse_m:.6f} m")
        print(f"  Plot: {plot_path}")

    summary_path = args.output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "ate_rmse_m",
                "num_anchors",
                "num_updates_attempted",
                "num_updates_rejected",
                "trajectory_path",
                "plot_path",
                "params_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
