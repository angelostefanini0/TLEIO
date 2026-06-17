"""Tune the downstream filter with the DAVIS240C DEIO MPE protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

try:
    from main_filter import CONFIG, RunnerConfig, run_filter
    from optimum_search import COARSE_LOG10_RANGES, REFINE_LOG10_HALF_WIDTH
except ImportError:
    from .main_filter import CONFIG, RunnerConfig, run_filter
    from .optimum_search import COARSE_LOG10_RANGES, REFINE_LOG10_HALF_WIDTH


FILTER_SEARCH_KEYS = (
    "sigma_na",
    "sigma_ng",
    "sigma_nba",
    "sigma_nbg",
    "assumed_sigma_rel_x_t",
    "assumed_sigma_rel_y_t",
    "assumed_sigma_rel_z_t",
    "meas_cov_scale",
    "initial_attitude_sigma_deg",
    "initial_velocity_sigma_mps",
    "initial_position_sigma_m",
    "initial_z_sigma_m",
    "initial_bg_sigma_rps",
    "initial_ba_sigma_mps2",
)

# processing_davis240c.py uses R_world_model = R_world_davis @ R_davis_model.
# Body-frame IMU vectors therefore require R_davis_model.T:
# [x_model, y_model, z_model] = [z_davis, x_davis, y_davis].
DAVIS_IMU_AXIS_MATRIX = (
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
)


def infer_time_scale_to_seconds(timestamps: np.ndarray) -> float:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    positive = np.diff(timestamps)
    positive = positive[positive > 0]
    median_dt = float(np.median(positive)) if len(positive) else 0.0
    if median_dt > 1e7:
        return 1e-9
    if median_dt > 1e1:
        return 1e-6
    return 1.0


def associate_positions(
    ground_truth: np.ndarray,
    estimate: np.ndarray,
    max_diff_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = np.asarray(ground_truth, dtype=np.float64).copy()
    est = np.asarray(estimate, dtype=np.float64).copy()
    gt[:, 0] *= infer_time_scale_to_seconds(gt[:, 0])
    est[:, 0] *= infer_time_scale_to_seconds(est[:, 0])

    gt_t = gt[:, 0]
    matched_gt = []
    matched_est = []
    matched_t = []
    last_gt_idx = -1
    for row in est:
        idx = int(np.searchsorted(gt_t, row[0]))
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(gt_t) and i > last_gt_idx]
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda i: abs(gt_t[i] - row[0]))
        if abs(gt_t[best_idx] - row[0]) > max_diff_s:
            continue
        matched_gt.append(gt[best_idx, 1:4])
        matched_est.append(row[1:4])
        matched_t.append(row[0])
        last_gt_idx = best_idx

    if len(matched_gt) < 3:
        raise ValueError("Fewer than three timestamp-associated trajectory poses.")
    return (
        np.asarray(matched_t),
        np.asarray(matched_gt),
        np.asarray(matched_est),
    )


def align_sim3(
    estimated: np.ndarray,
    reference: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    source = estimated[fit_mask]
    target = reference[fit_mask]
    if len(source) < 3:
        raise ValueError("DEIO prefix alignment needs at least three poses.")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    if variance <= 1e-12:
        raise ValueError("Estimated prefix has near-zero positional variance.")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ estimated.T)).T + translation
    return aligned, scale


def compute_deio_metrics(
    ground_truth: np.ndarray,
    estimate: np.ndarray,
    align_first_seconds: float,
    max_diff_s: float,
    align_first_poses: int | None = None,
) -> dict[str, float]:
    timestamps, gt_pos, est_pos = associate_positions(
        ground_truth,
        estimate,
        max_diff_s=max_diff_s,
    )
    if align_first_poses is not None:
        n_to_align = min(int(align_first_poses), len(timestamps))
        fit_mask = np.zeros(len(timestamps), dtype=bool)
        fit_mask[:n_to_align] = True
    else:
        elapsed = timestamps - timestamps[0]
        fit_mask = elapsed <= align_first_seconds
    aligned, scale = align_sim3(est_pos, gt_pos, fit_mask)
    errors = np.linalg.norm(aligned - gt_pos, axis=1)

    gt_full = np.asarray(ground_truth, dtype=np.float64)
    path_length = float(np.sum(np.linalg.norm(np.diff(gt_full[:, 1:4], axis=0), axis=1)))
    if path_length <= 1e-12:
        raise ValueError("Ground-truth path length is zero.")

    return {
        "ate_rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "ape_mean_m": float(np.mean(errors)),
        "path_length_m": path_length,
        "mpe_percent": float(100.0 * np.mean(errors) / path_length),
        "alignment_scale": scale,
        "matched_poses": int(len(errors)),
        "alignment_poses": int(np.count_nonzero(fit_mask)),
    }


def sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    return 10.0 ** rng.uniform(low, high)


def sample_coarse(rng: random.Random) -> dict[str, float]:
    return {
        key: sample_log_uniform(rng, *COARSE_LOG10_RANGES[key])
        for key in FILTER_SEARCH_KEYS
    }


def sample_refined(rng: random.Random, center: dict[str, float]) -> dict[str, float]:
    result = {}
    for key in FILTER_SEARCH_KEYS:
        center_log = math.log10(center[key])
        width = REFINE_LOG10_HALF_WIDTH[key]
        result[key] = sample_log_uniform(rng, center_log - width, center_log + width)
    return result


def evaluate(
    base_config: RunnerConfig,
    params: dict[str, float],
    align_first_seconds: float,
    max_diff_s: float,
    align_first_poses: int | None = None,
) -> dict | None:
    config = replace(
        base_config,
        **params,
        save_trajectory_file=False,
        save_diagnostic_plots=False,
        plot_transformer=False,
        plot_imu=False,
        plot_projections=False,
        plot_ate=False,
        interactive_plot=False,
    )
    try:
        results = run_filter(config)
        metrics = compute_deio_metrics(
            results["ground_truth"],
            results["trajectory"],
            align_first_seconds=align_first_seconds,
            max_diff_s=max_diff_s,
            align_first_poses=align_first_poses,
        )
    except Exception as exc:
        print(f"  trial failed: {type(exc).__name__}: {exc}")
        return None
    return {"params": params, "metrics": metrics}


def update_best(best: dict | None, candidate: dict | None) -> dict | None:
    if candidate is None:
        return best
    if best is None or candidate["metrics"]["mpe_percent"] < best["metrics"]["mpe_percent"]:
        return candidate
    return best


def print_trial(stage: str, index: int, trial: dict) -> None:
    metrics = trial["metrics"]
    print(
        f"[{stage} {index:04d}] "
        f"MPE={metrics['mpe_percent']:.6f}% "
        f"ATE={metrics['ate_rmse_m']:.6f} m "
        f"path={metrics['path_length_m']:.3f} m "
        f"align_scale={metrics['alignment_scale']:.6f}"
    )


def save_best_trajectory(
    base_config: RunnerConfig,
    best: dict,
    sequence_dir: Path,
) -> None:
    final_config = replace(
        base_config,
        **best["params"],
        save_trajectory_file=False,
        save_diagnostic_plots=False,
    )
    results = run_filter(final_config)
    np.savetxt(
        sequence_dir / "best_stamped_traj_estimate.txt",
        results["trajectory"],
        fmt="%.9f",
    )


def tune_sequence(args: argparse.Namespace, sequence: str) -> dict:
    rng = random.Random(args.seed)
    base_config = replace(
        CONFIG,
        dataset="davis240c",
        sequence=sequence,
        processed_dir=args.processed_dir,
        relative_motion_filename=args.relative_motions_file,
        network_scale=1.0,
        network_scale_x=1.0,
        network_scale_y=1.0,
        network_scale_z=1.0,
        oracle_scale_window=None,
        imu_axis_matrix=DAVIS_IMU_AXIS_MATRIX,
        gravity_world_mps2=(0.0, 0.0, -9.80665),
        save_trajectory_file=False,
        save_diagnostic_plots=False,
    )

    default_params = {
        key: float(getattr(base_config, key))
        for key in FILTER_SEARCH_KEYS
    }
    best = evaluate(
        base_config,
        default_params,
        args.align_first_seconds,
        args.max_diff_seconds,
        args.align_first_poses,
    )
    if best is not None:
        print_trial("default", 0, best)

    for idx in range(args.coarse_trials):
        trial = evaluate(
            base_config,
            sample_coarse(rng),
            args.align_first_seconds,
            args.max_diff_seconds,
            args.align_first_poses,
        )
        if trial is not None:
            print_trial("coarse", idx, trial)
            previous = best
            best = update_best(best, trial)
            if best is not previous:
                print("  new best coarse candidate")

    if best is None:
        raise RuntimeError(f"No valid trial completed for {sequence}.")

    center = best["params"]
    for idx in range(args.refine_trials):
        trial = evaluate(
            base_config,
            sample_refined(rng, center),
            args.align_first_seconds,
            args.max_diff_seconds,
            args.align_first_poses,
        )
        if trial is not None:
            print_trial("refine", idx, trial)
            previous = best
            best = update_best(best, trial)
            if best is not previous:
                center = best["params"]
                print("  new best refined candidate")

    sequence_dir = args.output_dir / sequence
    sequence_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "davis240c",
        "sequence": sequence,
        "objective": "deio_mpe_percent",
        "protocol": {
            "alignment": "sim3",
            "correct_scale": True,
            "align_first_seconds": args.align_first_seconds,
            "align_first_poses": args.align_first_poses,
            "max_timestamp_difference_seconds": args.max_diff_seconds,
            "mpe_definition": "100 * mean translation APE / full GT path length",
        },
        "prediction_file": args.relative_motions_file.format(sequence=sequence),
        "best": best,
        "base_config": asdict(base_config),
    }
    (sequence_dir / "best_filter_params.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    save_best_trajectory(base_config, best, sequence_dir)
    return summary


def collect_summaries(output_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(output_dir.glob("*/best_filter_params.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_csv(output_dir: Path, summaries: list[dict]) -> Path:
    csv_path = output_dir / "summary.csv"
    fields = [
        "sequence",
        "ate_rmse_m",
        "ape_mean_m",
        "path_length_m",
        "mpe_percent",
        "alignment_scale",
        *FILTER_SEARCH_KEYS,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            best = summary["best"]
            metrics = best["metrics"]
            params = best["params"]
            writer.writerow({
                "sequence": summary["sequence"],
                **{key: metrics[key] for key in fields if key in metrics},
                **{key: params[key] for key in FILTER_SEARCH_KEYS},
            })
    return csv_path


def print_dataset_summary(output_dir: Path) -> None:
    summaries = collect_summaries(output_dir)
    if not summaries:
        raise RuntimeError(f"No best_filter_params.json files found under {output_dir}.")
    csv_path = write_csv(output_dir, summaries)
    print("\nBest parameter sets")
    for summary in summaries:
        best = summary["best"]
        print(
            f"{summary['sequence']}: "
            f"MPE={best['metrics']['mpe_percent']:.6f}% "
            f"ATE={best['metrics']['ate_rmse_m']:.6f} m "
            f"params={json.dumps(best['params'], sort_keys=True)}"
        )
    average_mpe = float(np.mean([
        summary["best"]["metrics"]["mpe_percent"]
        for summary in summaries
    ]))
    print(f"\nAverage MPE over {len(summaries)} sequences: {average_mpe:.6f}%")
    print(f"CSV summary: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DAVIS240C filter parameters using the DEIO MPE protocol."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/davis240c/processed_checkpoint_compatible"),
    )
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument(
        "--relative-motions-file",
        type=str,
        default=(
            "../../predicted_relative_motions/"
            "checkpoint_compatible_windowed_scale_oracle/"
            "local_linear_no_bias_w25/{sequence}.txt"
        ),
    )
    parser.add_argument("--coarse-trials", type=int, default=800)
    parser.add_argument("--refine-trials", type=int, default=1200)
    parser.add_argument("--align-first-seconds", type=float, default=5.0)
    parser.add_argument(
        "--align-first-poses",
        type=int,
        default=None,
        help="If set, align with the first N timestamp-associated poses instead of first seconds.",
    )
    parser.add_argument("--max-diff-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/davis240c/windowed_scale_w25_filter_opt_mpe_deio"),
    )
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print_dataset_summary(args.output_dir)
        return

    if not args.processed_dir.is_dir():
        raise FileNotFoundError(f"Processed DAVIS root not found: {args.processed_dir}")
    sequences = (
        [args.sequence]
        if args.sequence
        else sorted(path.name for path in args.processed_dir.iterdir() if path.is_dir())
    )
    for sequence in sequences:
        prediction = Path(args.relative_motions_file.format(sequence=sequence))
        if not prediction.is_absolute():
            prediction = args.processed_dir / sequence / prediction
        if not prediction.is_file():
            print(f"Skipping {sequence}: prediction not found at {prediction.resolve()}")
            continue
        print("\n" + "=" * 72)
        print(f"Optimizing {sequence}")
        print("=" * 72)
        summary = tune_sequence(args, sequence)
        best = summary["best"]
        print(
            f"Best {sequence}: MPE={best['metrics']['mpe_percent']:.6f}% "
            f"ATE={best['metrics']['ate_rmse_m']:.6f} m"
        )

    print_dataset_summary(args.output_dir)


if __name__ == "__main__":
    main()
