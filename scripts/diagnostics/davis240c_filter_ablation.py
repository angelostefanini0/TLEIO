"""Ablate DAVIS240C filter geometry and weighting with DEIO MPE."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import permutations, product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main_filter import CONFIG, run_filter
from src.optimum_search_deio_mpe import compute_deio_metrics


def proper_signed_permutation_matrices() -> list[tuple[str, tuple[float, ...]]]:
    matrices = []
    axes = ("x", "y", "z")
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for output_axis, input_axis in enumerate(permutation):
                matrix[output_axis, input_axis] = signs[output_axis]
            if np.linalg.det(matrix) < 0.5:
                continue
            label = ",".join(
                f"{'+' if signs[i] > 0 else '-'}{axes[permutation[i]]}"
                for i in range(3)
            )
            matrices.append((label, tuple(matrix.reshape(-1).tolist())))
    return matrices


def gravity_hypotheses() -> list[tuple[str, tuple[float, float, float]]]:
    value = 9.80665
    return [
        ("+x", (value, 0.0, 0.0)),
        ("-x", (-value, 0.0, 0.0)),
        ("+y", (0.0, value, 0.0)),
        ("-y", (0.0, -value, 0.0)),
        ("+z", (0.0, 0.0, value)),
        ("-z", (0.0, 0.0, -value)),
    ]


def parse_float_list(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("Expected comma-separated positive values.")
    return result


def evaluate(config, align_first_seconds: float, max_diff_seconds: float) -> dict | None:
    try:
        results = run_filter(config)
        metrics = compute_deio_metrics(
            results["ground_truth"],
            results["trajectory"],
            align_first_seconds=align_first_seconds,
            max_diff_s=max_diff_seconds,
        )
        metrics["num_updates_rejected"] = int(results["num_updates_rejected"])
        return metrics
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return None


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_base_config(args: argparse.Namespace):
    return replace(
        CONFIG,
        dataset="davis240c",
        sequence=args.sequence,
        processed_dir=args.processed_dir,
        relative_motion_filename=args.relative_motions_file,
        network_scale=1.0,
        network_scale_x=1.0,
        network_scale_y=1.0,
        network_scale_z=1.0,
        oracle_scale_window=None,
        save_trajectory_file=False,
        save_diagnostic_plots=False,
        plot_transformer=False,
        plot_imu=False,
        plot_projections=False,
        plot_ate=False,
        interactive_plot=False,
    )


def geometry_stage(args: argparse.Namespace, base_config) -> tuple[dict, list[dict]]:
    rows = []
    best = None
    matrices = proper_signed_permutation_matrices()
    gravities = gravity_hypotheses()
    total = len(matrices) * len(gravities)
    index = 0
    for axis_label, axis_matrix in matrices:
        for gravity_label, gravity in gravities:
            index += 1
            print(
                f"[geometry {index:03d}/{total}] "
                f"imu=[{axis_label}] gravity={gravity_label}"
            )
            config = replace(
                base_config,
                imu_axis_matrix=axis_matrix,
                gravity_world_mps2=gravity,
            )
            metrics = evaluate(
                config,
                args.align_first_seconds,
                args.max_diff_seconds,
            )
            if metrics is None:
                continue
            row = {
                "stage": "geometry",
                "sequence": args.sequence,
                "imu_axes": axis_label,
                "gravity": gravity_label,
                "use_prediction_covariance": base_config.use_prediction_covariance,
                "meas_cov_scale": base_config.meas_cov_scale,
                "imu_noise_scale": 1.0,
                **metrics,
            }
            rows.append(row)
            if best is None or row["mpe_percent"] < best["mpe_percent"]:
                best = row
                print(f"  NEW BEST MPE={row['mpe_percent']:.6f}%")
    if best is None:
        raise RuntimeError("No valid geometry hypothesis completed.")
    return best, rows


def weighting_stage(
    args: argparse.Namespace,
    base_config,
    geometry_best: dict,
) -> tuple[dict, list[dict]]:
    axis_lookup = dict(proper_signed_permutation_matrices())
    gravity_lookup = dict(gravity_hypotheses())
    cov_modes = (False, True)
    combinations = list(product(
        cov_modes,
        args.covariance_scales,
        args.imu_noise_scales,
    ))
    rows = []
    best = None
    for index, (use_prediction_covariance, cov_scale, imu_noise_scale) in enumerate(
        combinations,
        start=1,
    ):
        print(
            f"[weighting {index:03d}/{len(combinations)}] "
            f"pred_cov={use_prediction_covariance} "
            f"cov_scale={cov_scale:g} imu_noise_scale={imu_noise_scale:g}"
        )
        config = replace(
            base_config,
            imu_axis_matrix=axis_lookup[geometry_best["imu_axes"]],
            gravity_world_mps2=gravity_lookup[geometry_best["gravity"]],
            use_prediction_covariance=use_prediction_covariance,
            meas_cov_scale=cov_scale,
            sigma_na=base_config.sigma_na * imu_noise_scale,
            sigma_ng=base_config.sigma_ng * imu_noise_scale,
            sigma_nba=base_config.sigma_nba * imu_noise_scale,
            sigma_nbg=base_config.sigma_nbg * imu_noise_scale,
        )
        metrics = evaluate(
            config,
            args.align_first_seconds,
            args.max_diff_seconds,
        )
        if metrics is None:
            continue
        row = {
            "stage": "weighting",
            "sequence": args.sequence,
            "imu_axes": geometry_best["imu_axes"],
            "gravity": geometry_best["gravity"],
            "use_prediction_covariance": use_prediction_covariance,
            "meas_cov_scale": cov_scale,
            "imu_noise_scale": imu_noise_scale,
            **metrics,
        }
        rows.append(row)
        if best is None or row["mpe_percent"] < best["mpe_percent"]:
            best = row
            print(f"  NEW BEST MPE={row['mpe_percent']:.6f}%")
    if best is None:
        raise RuntimeError("No valid weighting hypothesis completed.")
    return best, rows


def save_best_trajectory(args: argparse.Namespace, base_config, best: dict) -> Path:
    axis_lookup = dict(proper_signed_permutation_matrices())
    gravity_lookup = dict(gravity_hypotheses())
    noise_scale = float(best["imu_noise_scale"])
    config = replace(
        base_config,
        imu_axis_matrix=axis_lookup[best["imu_axes"]],
        gravity_world_mps2=gravity_lookup[best["gravity"]],
        use_prediction_covariance=bool(best["use_prediction_covariance"]),
        meas_cov_scale=float(best["meas_cov_scale"]),
        sigma_na=base_config.sigma_na * noise_scale,
        sigma_ng=base_config.sigma_ng * noise_scale,
        sigma_nba=base_config.sigma_nba * noise_scale,
        sigma_nbg=base_config.sigma_nbg * noise_scale,
    )
    results = run_filter(config)
    path = args.output_dir / args.sequence / "best_stamped_traj_estimate.txt"
    np.savetxt(path, results["trajectory"], fmt="%.9f")
    return path


def summarize_results(output_dir: Path) -> None:
    rows = []
    for summary_path in sorted(output_dir.glob("*/best_ablation.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best = summary["best"]
        rows.append({
            "sequence": summary["sequence"],
            "ate_rmse_m": best["ate_rmse_m"],
            "path_length_m": best["path_length_m"],
            "mpe_percent": best["mpe_percent"],
            "alignment_scale": best["alignment_scale"],
            "imu_axes": best["imu_axes"],
            "gravity": best["gravity"],
            "use_prediction_covariance": best["use_prediction_covariance"],
            "meas_cov_scale": best["meas_cov_scale"],
            "imu_noise_scale": best["imu_noise_scale"],
            "num_updates_rejected": best["num_updates_rejected"],
        })
    if not rows:
        raise FileNotFoundError(
            f"No best_ablation.json files found below {output_dir}."
        )

    summary_csv = output_dir / "summary.csv"
    write_rows(summary_csv, rows)
    average_mpe = float(np.mean([row["mpe_percent"] for row in rows]))
    print("\nBEST FILTER COMBINATIONS")
    for row in rows:
        print(
            f"{row['sequence']}: MPE={row['mpe_percent']:.6f}% "
            f"ATE={row['ate_rmse_m']:.6f} m "
            f"imu=[{row['imu_axes']}] gravity={row['gravity']} "
            f"pred_cov={row['use_prediction_covariance']} "
            f"cov_scale={row['meas_cov_scale']} "
            f"imu_noise_scale={row['imu_noise_scale']}"
        )
    print(f"\nAverage MPE over {len(rows)} sequences: {average_mpe:.6f}%")
    print(f"CSV summary: {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test all proper IMU axis rotations and gravity directions, then "
            "ablate covariance and IMU weighting using DEIO MPE."
        )
    )
    parser.add_argument("--sequence")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/davis240c/processed_checkpoint_compatible"),
    )
    parser.add_argument(
        "--relative-motions-file",
        type=str,
        default=(
            "../../predicted_relative_motions/"
            "checkpoint_compatible_windowed_scale_oracle/"
            "local_linear_no_bias_w25/{sequence}.txt"
        ),
    )
    parser.add_argument(
        "--covariance-scales",
        type=parse_float_list,
        default=parse_float_list("0.03,0.1,0.3,1,3,10,30,100"),
    )
    parser.add_argument(
        "--imu-noise-scales",
        type=parse_float_list,
        default=parse_float_list("0.03,0.1,0.3,1,3,10,30"),
    )
    parser.add_argument("--align-first-seconds", type=float, default=5.0)
    parser.add_argument("--max-diff-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/davis240c/windowed_scale_w25_filter_ablation_deio"),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Aggregate completed per-sequence JSON files into summary.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_only:
        summarize_results(args.output_dir)
        return
    if not args.sequence:
        raise ValueError("--sequence is required unless --summary-only is used.")

    sequence_dir = args.output_dir / args.sequence
    sequence_dir.mkdir(parents=True, exist_ok=True)
    base_config = make_base_config(args)

    geometry_best, geometry_rows = geometry_stage(args, base_config)
    write_rows(sequence_dir / "geometry_trials.csv", geometry_rows)

    weighting_best, weighting_rows = weighting_stage(
        args,
        base_config,
        geometry_best,
    )
    write_rows(sequence_dir / "weighting_trials.csv", weighting_rows)
    trajectory_path = save_best_trajectory(args, base_config, weighting_best)

    summary = {
        "sequence": args.sequence,
        "protocol": {
            "metric": "DEIO MPE",
            "alignment": "Sim(3), including scale",
            "align_first_seconds": args.align_first_seconds,
            "mpe": "100 * mean translation APE / GT path length",
        },
        "geometry_best": geometry_best,
        "best": weighting_best,
        "trajectory": str(trajectory_path),
    }
    summary_path = sequence_dir / "best_ablation.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBEST COMBINATION")
    print(json.dumps(weighting_best, indent=2))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
