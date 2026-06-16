"""Summarize TLEIO measurement covariance calibration from update diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


AXES = ("x", "y", "z")
NUM_EDGES = 4
COMPONENT_DIM = 12


def _read_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def load_update_diagnostics(path: Path, accepted_only: bool = False) -> list[dict[str, str]]:
    """Load update diagnostics rows, optionally keeping accepted updates only."""

    with Path(path).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if accepted_only:
        rows = [row for row in rows if int(float(row.get("accepted", "1") or 1)) == 1]
    if not rows:
        raise ValueError(f"No diagnostics rows available in {path}.")
    return rows


def diagnostics_to_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract residuals, sigmas, and chi-square ratios from diagnostics rows."""

    residuals = np.asarray(
        [[_read_float(row, f"residual_{idx}") for idx in range(COMPONENT_DIM)] for row in rows],
        dtype=np.float64,
    )
    sigmas = np.asarray(
        [[_read_float(row, f"sigma_{idx}") for idx in range(COMPONENT_DIM)] for row in rows],
        dtype=np.float64,
    )
    chi2_ratios = np.asarray([_read_float(row, "chi2_ratio") for row in rows], dtype=np.float64)
    if not np.isfinite(residuals).all():
        raise ValueError("Diagnostics contain non-finite residual components.")
    if not np.isfinite(sigmas).all() or np.any(sigmas <= 0.0):
        raise ValueError("Diagnostics contain invalid sigma components.")
    return residuals, sigmas, chi2_ratios


def _mad(values: np.ndarray) -> float:
    """Return median absolute deviation."""

    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def compute_calibration_summary(rows: list[dict[str, str]]) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    """Compute edge/axis covariance calibration summary rows."""

    residuals, sigmas, chi2_ratios = diagnostics_to_arrays(rows)
    normalized = residuals / sigmas

    summary_rows: list[dict[str, float | int | str]] = []
    for edge_idx in range(NUM_EDGES):
        for axis_idx, axis_name in enumerate(AXES):
            component_idx = edge_idx * 3 + axis_idx
            residual_component = residuals[:, component_idx]
            sigma_component = sigmas[:, component_idx]
            normalized_component = normalized[:, component_idx]
            summary_rows.append(
                {
                    "edge": edge_idx,
                    "axis": axis_name,
                    "component": component_idx,
                    "num_samples": int(len(rows)),
                    "empirical_residual_std": float(np.std(residual_component, ddof=1)) if len(rows) > 1 else 0.0,
                    "mean_predicted_sigma": float(np.mean(sigma_component)),
                    "median_predicted_sigma": float(np.median(sigma_component)),
                    "normalized_residual_rms": float(np.sqrt(np.mean(normalized_component**2))),
                    "normalized_residual_mad": _mad(normalized_component),
                }
            )

    finite_chi2 = chi2_ratios[np.isfinite(chi2_ratios)]
    global_normalized_rms = float(np.sqrt(np.mean(normalized**2)))
    global_summary = {
        "num_updates": float(len(rows)),
        "global_normalized_residual_rms": global_normalized_rms,
        "recommended_meas_cov_scale_multiplier": global_normalized_rms**2,
        "median_chi2_ratio": float(np.median(finite_chi2)) if len(finite_chi2) else np.nan,
        "p95_chi2_ratio": float(np.percentile(finite_chi2, 95)) if len(finite_chi2) else np.nan,
        "max_chi2_ratio": float(np.max(finite_chi2)) if len(finite_chi2) else np.nan,
    }
    return summary_rows, global_summary


def write_summary_csv(path: Path, summary_rows: list[dict[str, float | int | str]]) -> Path:
    """Write edge/axis calibration rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return path


def write_summary_markdown(
    path: Path,
    summary_rows: list[dict[str, float | int | str]],
    global_summary: dict[str, float],
) -> Path:
    """Write a Markdown calibration report."""

    lines = [
        "# Measurement Covariance Calibration",
        "",
        "This file is diagnostic only. It does not change filter parameters automatically.",
        "",
        "## Global Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in global_summary.items():
        lines.append(f"| {key} | {value:.9f} |")

    lines.extend(
        [
            "",
            "## Edge And Axis Summary",
            "",
            "| edge | axis | empirical residual std | mean sigma | median sigma | normalized RMS | normalized MAD |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {edge} | {axis} | {std:.9f} | {mean:.9f} | {median:.9f} | {rms:.9f} | {mad:.9f} |".format(
                edge=row["edge"],
                axis=row["axis"],
                std=row["empirical_residual_std"],
                mean=row["mean_predicted_sigma"],
                median=row["median_predicted_sigma"],
                rms=row["normalized_residual_rms"],
                mad=row["normalized_residual_mad"],
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_calibration(
    diagnostics_path: Path,
    output_dir: Path | None = None,
    accepted_only: bool = False,
) -> tuple[Path, Path, list[dict[str, float | int | str]], dict[str, float]]:
    """Run covariance calibration and write CSV/Markdown summaries."""

    rows = load_update_diagnostics(diagnostics_path, accepted_only=accepted_only)
    summary_rows, global_summary = compute_calibration_summary(rows)
    output_dir = Path(output_dir) if output_dir is not None else Path(diagnostics_path).parent
    csv_path = write_summary_csv(output_dir / "covariance_calibration_summary.csv", summary_rows)
    md_path = write_summary_markdown(output_dir / "covariance_calibration_summary.md", summary_rows, global_summary)
    return csv_path, md_path, summary_rows, global_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze measurement covariance calibration from update diagnostics.")
    parser.add_argument("--diagnostics", type=Path, required=True, help="Path to update_diagnostics.csv.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for calibration summaries.")
    parser.add_argument("--accepted_only", action="store_true", help="Use accepted updates only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path, md_path, _, global_summary = run_calibration(
        args.diagnostics,
        output_dir=args.output_dir,
        accepted_only=args.accepted_only,
    )
    print(f"Saved CSV:      {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(
        "Recommended meas_cov_scale multiplier: "
        f"{global_summary['recommended_meas_cov_scale_multiplier']:.9f}"
    )


if __name__ == "__main__":
    main()
