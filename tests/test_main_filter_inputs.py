from pathlib import Path

import numpy as np
import pytest

from src.main_filter import (
    RunnerConfig,
    _build_anchor_times_from_relative_motions,
    _build_exact_imu_intervals,
    _build_exact_imu_segment,
    _build_joint_covariance_for_window,
    _load_relative_motion_table,
    _sanitize_relative_sigmas,
    _save_update_diagnostics,
    _summarize_chi2_ratios,
)
from filter.measurement_triplet import make_default_joint_covariance


def _write_relative_file(sequence_path: Path, rows: list[str]) -> None:
    sequence_path.mkdir(parents=True, exist_ok=True)
    rel_path = sequence_path / f"{sequence_path.name}.txt"
    rel_path.write_text("header line\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_load_relative_motion_table_accepts_5_and_8_columns(tmp_path):
    seq5 = tmp_path / "seq5"
    _write_relative_file(
        seq5,
        [
            "0 1 0.1 0.0 0.0",
            "1 2 0.1 0.0 0.0",
            "2 3 0.1 0.0 0.0",
            "3 4 0.1 0.0 0.0",
        ],
    )
    assert _load_relative_motion_table(seq5, use_gt=False).shape == (4, 5)

    seq8 = tmp_path / "seq8"
    _write_relative_file(
        seq8,
        [
            "0 1 0.1 0.0 0.0 0.01 0.02 0.03",
            "1 2 0.1 0.0 0.0 0.01 0.02 0.03",
            "2 3 0.1 0.0 0.0 0.01 0.02 0.03",
            "3 4 0.1 0.0 0.0 0.01 0.02 0.03",
        ],
    )
    assert _load_relative_motion_table(seq8, use_gt=False).shape == (4, 8)


def test_build_anchor_times_extracts_sigmas_for_8_columns():
    table = np.array(
        [
            [0, 1, 1, 0, 0, 0.1, 0.2, 0.3],
            [1, 2, 1, 0, 0, 0.4, 0.5, 0.6],
            [2, 3, 1, 0, 0, 0.7, 0.8, 0.9],
            [3, 4, 1, 0, 0, 1.0, 1.1, 1.2],
        ],
        dtype=float,
    )
    anchor_times, measurements, sigmas = _build_anchor_times_from_relative_motions(table)

    np.testing.assert_allclose(anchor_times, np.array([0, 1, 2, 3, 4], dtype=float))
    np.testing.assert_allclose(measurements, table[:, 2:5])
    np.testing.assert_allclose(sigmas, table[:, 5:8])


def test_regressed_sigmas_fill_12d_covariance_diagonal():
    base = make_default_joint_covariance(0.5)
    sigmas = np.arange(1, 16, dtype=float).reshape(5, 3) * 0.01
    covariance, used_regressed = _build_joint_covariance_for_window(base, sigmas, 1)

    assert used_regressed
    np.testing.assert_allclose(np.diag(covariance), sigmas[1:5].reshape(-1) ** 2)


def test_negative_or_nan_sigmas_are_rejected():
    config = RunnerConfig()
    with pytest.raises(ValueError, match="non-finite"):
        _sanitize_relative_sigmas(np.array([[np.nan, 0.1, 0.1]]), config)
    with pytest.raises(ValueError, match="non-negative"):
        _sanitize_relative_sigmas(np.array([[-0.1, 0.1, 0.1]]), config)


def test_covariance_flag_can_disable_regressed_sigmas():
    config = RunnerConfig(use_regressed_covariance=False)
    sigmas = np.ones((4, 3))
    assert _sanitize_relative_sigmas(sigmas, config) is None


def test_exact_imu_segment_hits_requested_end_time():
    raw_times = np.array([0.0, 0.5, 1.0])
    raw_gyro = np.column_stack([raw_times, raw_times + 1.0, raw_times + 2.0])
    raw_accel = np.column_stack([raw_times + 3.0, raw_times + 4.0, raw_times + 5.0])

    segment = _build_exact_imu_segment(raw_times, raw_gyro, raw_accel, 0.25, 0.75)

    assert segment[-1].timestamp == pytest.approx(0.75)
    assert [m.dt for m in segment] == pytest.approx([0.25, 0.25])
    np.testing.assert_allclose(segment[-1].gyro, [0.75, 1.75, 2.75])


def test_exact_imu_segment_rejects_zero_dt():
    raw_times = np.array([0.0, 0.5, 0.5, 1.0])
    raw_gyro = np.zeros((4, 3))
    raw_accel = np.zeros((4, 3))

    with pytest.raises(ValueError, match="duplicate|non-increasing"):
        _build_exact_imu_segment(raw_times, raw_gyro, raw_accel, 0.25, 0.75)


def test_exact_imu_intervals_include_requested_start_and_end():
    raw_times = np.array([0.0, 0.5, 1.0])
    raw_gyro = np.column_stack([raw_times, raw_times + 1.0, raw_times + 2.0])
    raw_accel = np.column_stack([raw_times + 3.0, raw_times + 4.0, raw_times + 5.0])

    intervals = _build_exact_imu_intervals(raw_times, raw_gyro, raw_accel, 0.25, 0.75)

    assert intervals[0].t0 == pytest.approx(0.25)
    assert intervals[-1].t1 == pytest.approx(0.75)
    assert [interval.dt for interval in intervals] == pytest.approx([0.25, 0.25])
    np.testing.assert_allclose(intervals[0].gyro0, [0.25, 1.25, 2.25])
    np.testing.assert_allclose(intervals[-1].gyro1, [0.75, 1.75, 2.75])


def test_exact_imu_intervals_reject_duplicate_raw_timestamps():
    raw_times = np.array([0.0, 0.5, 0.5, 1.0])
    raw_gyro = np.zeros((4, 3))
    raw_accel = np.zeros((4, 3))

    with pytest.raises(ValueError, match="strictly increasing"):
        _build_exact_imu_intervals(raw_times, raw_gyro, raw_accel, 0.25, 0.75)


def test_update_diagnostics_csv_is_written(tmp_path):
    path = _save_update_diagnostics(
        tmp_path / "update_diagnostics.csv",
        [
            {
                "anchor_idx": 4,
                "timestamp_s": 1.0,
                "accepted": 1,
                "rejected": 0,
                "mahalanobis_sq": 2.0,
                "chi2_threshold": 4.0,
                "chi2_ratio": 0.5,
                "residual_norm": 0.1,
                "correction_norm": 0.2,
            }
        ],
    )

    text = path.read_text(encoding="utf-8")
    assert "anchor_idx,timestamp_s,accepted" in text
    assert "4,1.0,1" in text


def test_chi2_ratio_summary_matches_raw_values():
    summary = _summarize_chi2_ratios([1.0, 2.0, 3.0])

    assert summary["median_chi2_ratio"] == pytest.approx(2.0)
    assert summary["p95_chi2_ratio"] == pytest.approx(2.9)
    assert summary["max_chi2_ratio"] == pytest.approx(3.0)
