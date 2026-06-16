import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from filter_covariance_calibration import compute_calibration_summary, run_calibration
from filter_ablation_openvins3 import REQUIRED_SUMMARY_COLUMNS, write_summary


def _write_synthetic_diagnostics(path: Path) -> None:
    fieldnames = ["accepted", "rejected", "chi2_ratio"]
    fieldnames += [f"residual_{idx}" for idx in range(12)]
    fieldnames += [f"sigma_{idx}" for idx in range(12)]
    rows = []
    for row_idx in range(3):
        row = {"accepted": 1, "rejected": 0, "chi2_ratio": 0.1 * (row_idx + 1)}
        for component_idx in range(12):
            row[f"residual_{component_idx}"] = float(component_idx + 1)
            row[f"sigma_{component_idx}"] = 2.0
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_covariance_calibration_script_runs_on_synthetic_csv(tmp_path):
    diagnostics_path = tmp_path / "update_diagnostics.csv"
    _write_synthetic_diagnostics(diagnostics_path)

    csv_path, md_path, rows, global_summary = run_calibration(diagnostics_path, output_dir=tmp_path)

    assert csv_path.exists()
    assert md_path.exists()
    assert len(rows) == 12
    assert global_summary["num_updates"] == pytest.approx(3.0)


def test_covariance_calibration_groups_edges_and_axes_correctly(tmp_path):
    diagnostics_path = tmp_path / "update_diagnostics.csv"
    _write_synthetic_diagnostics(diagnostics_path)

    _, _, rows, _ = run_calibration(diagnostics_path, output_dir=tmp_path)

    assert rows[0]["edge"] == 0
    assert rows[0]["axis"] == "x"
    assert rows[3]["edge"] == 1
    assert rows[3]["axis"] == "x"
    assert rows[-1]["edge"] == 3
    assert rows[-1]["axis"] == "z"


def test_covariance_calibration_recommends_scale_from_normalized_rms():
    rows = []
    for _ in range(2):
        row = {"accepted": "1", "rejected": "0", "chi2_ratio": "1.0"}
        for component_idx in range(12):
            row[f"residual_{component_idx}"] = "2.0"
            row[f"sigma_{component_idx}"] = "1.0"
        rows.append(row)

    _, global_summary = compute_calibration_summary(rows)

    assert global_summary["global_normalized_residual_rms"] == pytest.approx(2.0)
    assert global_summary["recommended_meas_cov_scale_multiplier"] == pytest.approx(4.0)


def test_covariance_calibration_handles_missing_rejections(tmp_path):
    diagnostics_path = tmp_path / "update_diagnostics.csv"
    _write_synthetic_diagnostics(diagnostics_path)

    _, _, rows, _ = run_calibration(diagnostics_path, output_dir=tmp_path, accepted_only=True)

    assert len(rows) == 12


def test_scale_grid_summary_has_required_columns(tmp_path):
    summary_path = write_summary(
        tmp_path / "summary.csv",
        [
            {
                "run": "scale_1",
                "flags": "--meas_cov_scale 1.0",
                "status": "ok",
                "runtime_s": 1.0,
            }
        ],
    )

    with summary_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")

    assert header == REQUIRED_SUMMARY_COLUMNS
