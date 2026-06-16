"""Run the TLEIO filter on a batch of EDS sequences with network outputs."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compute_ate_metrics import (  # noqa: E402
    apply_alignment,
    ate_rmse,
    compute_metrics_for_paths,
    interpolate_ground_truth,
    load_trajectory,
    umeyama_alignment,
)


DEFAULT_SEQUENCES = [
    "01_peanuts_light",
    "02_rocket_earth_light",
    "03_rocket_earth_dark",
    "07_ziggy_and_fuzz_hdr",
    "08_peanuts_running",
    "09_ziggy_flying_pieces",
    "11_all_characters",
]

PRESET_FLAGS = {
    "openvins_best": [
        "--imu_interval_mode",
        "paired_samples",
        "--nominal_integration_method",
        "midpoint",
    ],
    "v3_cov_scale_2p5": [
        "--imu_interval_mode",
        "paired_samples",
        "--nominal_integration_method",
        "midpoint",
        "--meas_cov_scale",
        "2.5",
    ],
    "v3_midpoint_half_R": [
        "--imu_interval_mode",
        "paired_samples",
        "--nominal_integration_method",
        "midpoint_half_R",
    ],
}

SUMMARY_FIELDS = [
    "sequence",
    "status",
    "runtime_s",
    "flags",
    "raw_ate_rmse_m",
    "se3_aligned_ate_rmse_m",
    "sim3_aligned_scaled_ate_rmse_m",
    "sim3_scale",
    "raw_rotation_rmse_deg",
    "first20_raw_ate_rmse_m",
    "first20_sim3_ate_rmse_m",
    "first20_sim3_scale",
    "printed_position_rmse_m",
    "printed_rotation_rmse_deg",
    "max_position_error_m",
    "max_rotation_error_deg",
    "updates_rejected",
    "mean_residual_norm",
    "mean_correction_norm",
    "median_chi2_ratio",
    "p95_chi2_ratio",
    "max_chi2_ratio",
    "max_edge_chi2_ratio",
]


def _sequence_dir(sequence: str) -> Path:
    return ROOT / "data" / "eds" / "processed" / sequence


def validate_sequence_inputs(sequences: list[str]) -> None:
    """Fail early if any sequence is missing required processed/network files."""

    missing: list[str] = []
    for sequence in sequences:
        sequence_dir = _sequence_dir(sequence)
        for filename in (
            "anchor_poses.txt",
            "imu.csv",
            f"{sequence}.txt",
        ):
            path = sequence_dir / filename
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required EDS batch input files:\n" + "\n".join(missing))


def parse_run_log(path: Path) -> dict[str, float]:
    """Extract scalar values printed by main_filter.py."""

    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "printed_position_rmse_m": r"Position RMSE \[m\]:\s+([0-9.eE+-]+)",
        "printed_rotation_rmse_deg": r"Rotation RMSE \[deg\]:\s+([0-9.eE+-]+)",
        "max_position_error_m": r"MAX position error \[m\]:\s+([0-9.eE+-]+)",
        "max_rotation_error_deg": r"MAX rotation error \[deg\]:\s+([0-9.eE+-]+)",
        "updates_rejected": r"Updates rejected:\s+([0-9.eE+-]+)",
        "mean_residual_norm": r"Mean residual norm:\s+([0-9.eE+-]+)",
        "mean_correction_norm": r"Mean correction norm:\s+([0-9.eE+-]+)",
    }
    values: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        values[key] = float(match.group(1)) if match else float("nan")
    return values


def parse_update_diagnostics(path: Path) -> dict[str, float]:
    """Summarize chi-square ratios from update_diagnostics.csv."""

    if not path.exists():
        return {
            "median_chi2_ratio": float("nan"),
            "p95_chi2_ratio": float("nan"),
            "max_chi2_ratio": float("nan"),
            "max_edge_chi2_ratio": float("nan"),
        }
    chi2_ratios: list[float] = []
    edge_ratios: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("chi2_ratio", ""):
                chi2_ratios.append(float(row["chi2_ratio"]))
            if row.get("max_edge_chi2_ratio", ""):
                edge_ratios.append(float(row["max_edge_chi2_ratio"]))
    if not chi2_ratios:
        return {
            "median_chi2_ratio": float("nan"),
            "p95_chi2_ratio": float("nan"),
            "max_chi2_ratio": float("nan"),
            "max_edge_chi2_ratio": float("nan"),
        }
    chi2 = np.asarray(chi2_ratios, dtype=np.float64)
    edge = np.asarray(edge_ratios, dtype=np.float64) if edge_ratios else np.array([np.nan])
    return {
        "median_chi2_ratio": float(np.median(chi2)),
        "p95_chi2_ratio": float(np.percentile(chi2, 95)),
        "max_chi2_ratio": float(np.max(chi2)),
        "max_edge_chi2_ratio": float(np.nanmax(edge)),
    }


def first_window_metrics(estimated_path: Path, ground_truth_path: Path, seconds: float = 20.0) -> dict[str, float]:
    """Compute raw and Sim3 ATE on the first `seconds` of one trajectory."""

    estimated = load_trajectory(estimated_path)
    ground_truth = load_trajectory(ground_truth_path)
    gt_positions, _ = interpolate_ground_truth(ground_truth, estimated[:, 0])
    estimated_positions = estimated[:, 1:4]
    relative_time = estimated[:, 0] - estimated[0, 0]
    mask = relative_time <= seconds
    if int(np.sum(mask)) < 3:
        return {
            "first20_raw_ate_rmse_m": float("nan"),
            "first20_sim3_ate_rmse_m": float("nan"),
            "first20_sim3_scale": float("nan"),
        }
    scale, rotation, translation = umeyama_alignment(
        estimated_positions[mask],
        gt_positions[mask],
        with_scale=True,
    )
    aligned = apply_alignment(estimated_positions[mask], scale, rotation, translation)
    return {
        "first20_raw_ate_rmse_m": ate_rmse(estimated_positions[mask], gt_positions[mask]),
        "first20_sim3_ate_rmse_m": ate_rmse(aligned, gt_positions[mask]),
        "first20_sim3_scale": scale,
    }


def copy_artifacts(sequence: str, destination: Path) -> None:
    """Copy standard main_filter artifacts for one sequence."""

    source = ROOT / "outputs" / "main_filter" / "eds" / sequence
    filenames = [
        "stamped_traj_estimate.txt",
        "update_diagnostics.csv",
        "consistency_diagnostics.csv",
        f"{sequence}_trajectory_comparison.png",
        f"{sequence}_rotation_comparison.png",
        f"{sequence}_trajectory_3d.png",
    ]
    destination.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        src = source / filename
        if src.exists():
            shutil.copy2(src, destination / filename)


def run_one_sequence(
    sequence: str,
    run_dir: Path,
    flags: list[str],
    python_executable: str = sys.executable,
) -> dict[str, float | str]:
    """Run main_filter.py on one EDS sequence and return one summary row."""

    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(ROOT / "src" / "main_filter.py"),
        "--dataset",
        "eds",
        "--sequence",
        sequence,
        *flags,
    ]
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
    row: dict[str, float | str] = {
        "sequence": sequence,
        "status": "ok" if completed.returncode == 0 else f"failed:{completed.returncode}",
        "runtime_s": runtime_s,
        "flags": " ".join(flags),
    }
    if completed.returncode != 0:
        return row

    copy_artifacts(sequence, run_dir)
    estimated_path = run_dir / "stamped_traj_estimate.txt"
    ground_truth_path = _sequence_dir(sequence) / "anchor_poses.txt"
    ate_metrics = compute_metrics_for_paths([estimated_path], ground_truth_path)[0]
    row.update(
        {
            "raw_ate_rmse_m": ate_metrics["raw_ate_rmse_m"],
            "se3_aligned_ate_rmse_m": ate_metrics["se3_aligned_ate_rmse_m"],
            "sim3_aligned_scaled_ate_rmse_m": ate_metrics["sim3_aligned_scaled_ate_rmse_m"],
            "sim3_scale": ate_metrics["sim3_scale"],
            "raw_rotation_rmse_deg": ate_metrics["raw_rotation_rmse_deg"],
        }
    )
    row.update(first_window_metrics(estimated_path, ground_truth_path))
    row.update(parse_run_log(run_dir / "run.log"))
    row.update(parse_update_diagnostics(run_dir / "update_diagnostics.csv"))
    return row


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> Path:
    """Write batch summary CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return path


def write_summary_markdown(path: Path, rows: list[dict[str, float | str]]) -> Path:
    """Write a compact Markdown table."""

    lines = [
        "# EDS Batch Summary",
        "",
        "| sequence | status | raw ATE m | Sim3 ATE m | Sim3 scale | first20 Sim3 ATE m | rot RMSE deg | rejected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key, "")
            return f"{value:.6f}" if isinstance(value, float) and np.isfinite(value) else str(value)

        lines.append(
            "| {sequence} | {status} | {raw} | {sim3} | {scale} | {first20} | {rot} | {rejected} |".format(
                sequence=row.get("sequence", ""),
                status=row.get("status", ""),
                raw=fmt("raw_ate_rmse_m"),
                sim3=fmt("sim3_aligned_scaled_ate_rmse_m"),
                scale=fmt("sim3_scale"),
                first20=fmt("first20_sim3_ate_rmse_m"),
                rot=fmt("raw_rotation_rmse_deg"),
                rejected=fmt("updates_rejected"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_eds_batch(
    sequences: list[str],
    preset: str = "openvins_best",
    run_name: str | None = None,
    extra_filter_args: list[str] | None = None,
    output_root: Path | None = None,
) -> list[dict[str, float | str]]:
    """Run the filter on an EDS sequence batch and save a comparison summary."""

    if preset not in PRESET_FLAGS:
        raise ValueError(f"Unknown preset {preset!r}. Available presets: {sorted(PRESET_FLAGS)}")
    validate_sequence_inputs(sequences)
    flags = [*PRESET_FLAGS[preset]]
    if extra_filter_args:
        flags.extend(extra_filter_args)
    run_name = run_name or preset
    output_root = output_root or (ROOT / "outputs" / "comparison_eds_batch" / run_name)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for sequence in sequences:
        print(f"Running {sequence} with {run_name}...", flush=True)
        rows.append(run_one_sequence(sequence, output_root / sequence, flags))
    write_summary_csv(output_root / "summary.csv", rows)
    write_summary_markdown(output_root / "summary.md", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TLEIO on a batch of EDS sequences.")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES, help="EDS sequence names.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_FLAGS),
        default="openvins_best",
        help="Filter preset to run.",
    )
    parser.add_argument("--run_name", default=None, help="Output folder name. Defaults to preset.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Output root. Defaults to outputs/comparison_eds_batch/<run_name>.",
    )
    parser.add_argument(
        "--extra_filter_arg",
        action="append",
        default=[],
        help="Append one extra raw main_filter.py argument. Repeat for each token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_eds_batch(
        sequences=args.sequences,
        preset=args.preset,
        run_name=args.run_name,
        extra_filter_args=args.extra_filter_arg,
        output_root=args.output_root,
    )
    output_root = args.output_root or (ROOT / "outputs" / "comparison_eds_batch" / (args.run_name or args.preset))
    print(f"Saved summary: {output_root / 'summary.csv'}")
    for row in rows:
        print(
            "{sequence}: {status} raw={raw} sim3={sim3} first20={first20}".format(
                sequence=row.get("sequence"),
                status=row.get("status"),
                raw=row.get("raw_ate_rmse_m", ""),
                sim3=row.get("sim3_aligned_scaled_ate_rmse_m", ""),
                first20=row.get("first20_sim3_ate_rmse_m", ""),
            )
        )


if __name__ == "__main__":
    main()
