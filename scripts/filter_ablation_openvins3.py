"""Run the OpenVINS-3 ME004 covariance-scale ablation grid."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from compute_ate_metrics import compute_metrics_for_paths


REQUIRED_SUMMARY_COLUMNS = [
    "run",
    "flags",
    "status",
    "runtime_s",
    "raw_ate_rmse_m",
    "se3_aligned_ate_rmse_m",
    "sim3_aligned_scaled_ate_rmse_m",
    "sim3_scale",
    "raw_rotation_rmse_deg",
    "rejected_updates",
    "median_chi2_ratio",
    "p95_chi2_ratio",
    "mean_correction_norm",
]


def parse_run_log(path: Path) -> dict[str, float]:
    """Parse a main_filter run log for scalar metrics."""

    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Updates rejected:" in line:
            metrics["rejected_updates"] = float(line.rsplit(maxsplit=1)[-1])
        elif "Mean correction norm:" in line:
            metrics["mean_correction_norm"] = float(line.rsplit(maxsplit=1)[-1])
    return metrics


def parse_chi2_ratios(path: Path) -> dict[str, float]:
    """Parse update diagnostics for chi-square summary statistics."""

    with path.open("r", encoding="utf-8") as handle:
        ratios = sorted(float(row["chi2_ratio"]) for row in csv.DictReader(handle))
    if not ratios:
        return {"median_chi2_ratio": float("nan"), "p95_chi2_ratio": float("nan")}
    mid = len(ratios) // 2
    median = ratios[mid] if len(ratios) % 2 else 0.5 * (ratios[mid - 1] + ratios[mid])
    p95 = ratios[int(round(0.95 * (len(ratios) - 1)))]
    return {"median_chi2_ratio": median, "p95_chi2_ratio": p95}


def copy_run_artifacts(source_dir: Path, destination_dir: Path) -> None:
    """Copy standard main_filter output artifacts into one ablation folder."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    filenames = [
        "stamped_traj_estimate.txt",
        "update_diagnostics.csv",
        "competition_Test_ME004_trajectory_comparison.png",
        "competition_Test_ME004_rotation_comparison.png",
        "competition_Test_ME004_trajectory_3d.png",
    ]
    for filename in filenames:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, destination_dir / filename)


def run_scale_grid(
    scales: list[float],
    output_root: Path,
    python_executable: str = sys.executable,
) -> list[dict[str, float | str]]:
    """Run ME004 for each global measurement covariance scale."""

    run_rows: list[dict[str, float | str]] = []
    main_output_dir = ROOT / "outputs" / "main_filter" / "tartanair" / "competition_Test_ME004"
    ground_truth_path = ROOT / "data" / "tartanair" / "processed" / "competition_Test_ME004" / "anchor_poses.txt"

    for scale in scales:
        run_name = f"meas_cov_scale_{scale:g}".replace(".", "p")
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        flags = [
            "--dataset",
            "tartanair",
            "--sequence",
            "competition_Test_ME004",
            "--imu_interval_mode",
            "paired_samples",
            "--nominal_integration_method",
            "midpoint",
            "--meas_cov_scale",
            str(scale),
        ]
        command = [python_executable, str(ROOT / "src" / "main_filter.py"), *flags]
        start = time.monotonic()
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        runtime_s = time.monotonic() - start
        status = "ok" if completed.returncode == 0 else "failed"
        if status == "ok":
            copy_run_artifacts(main_output_dir, run_dir)
            ate_rows = compute_metrics_for_paths([run_dir / "stamped_traj_estimate.txt"], ground_truth_path)
            ate_metrics = ate_rows[0]
            log_metrics = parse_run_log(run_dir / "run.log")
            chi2_metrics = parse_chi2_ratios(run_dir / "update_diagnostics.csv")
        else:
            ate_metrics = {}
            log_metrics = {}
            chi2_metrics = {}
        run_rows.append(
            {
                "run": run_name,
                "flags": " ".join(flags),
                "status": status,
                "runtime_s": runtime_s,
                "raw_ate_rmse_m": ate_metrics.get("raw_ate_rmse_m", ""),
                "se3_aligned_ate_rmse_m": ate_metrics.get("se3_aligned_ate_rmse_m", ""),
                "sim3_aligned_scaled_ate_rmse_m": ate_metrics.get("sim3_aligned_scaled_ate_rmse_m", ""),
                "sim3_scale": ate_metrics.get("sim3_scale", ""),
                "raw_rotation_rmse_deg": ate_metrics.get("raw_rotation_rmse_deg", ""),
                "rejected_updates": log_metrics.get("rejected_updates", ""),
                "median_chi2_ratio": chi2_metrics.get("median_chi2_ratio", ""),
                "p95_chi2_ratio": chi2_metrics.get("p95_chi2_ratio", ""),
                "mean_correction_norm": log_metrics.get("mean_correction_norm", ""),
            }
        )
    return run_rows


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> Path:
    """Write the ablation summary CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in REQUIRED_SUMMARY_COLUMNS})
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenVINS-3 covariance scale ablations.")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.25],
        help="Global measurement covariance scales to evaluate.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=ROOT / "outputs" / "comparison_ME004" / "openvins_3" / "cov_scale_grid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_scale_grid(args.scales, args.output_root)
    summary_path = write_summary(args.output_root / "summary.csv", rows)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
