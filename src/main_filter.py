"""Run the EKF directly on processed `relative_motions.txt` measurements.

1. load one processed sequence (`anchor_poses.txt`, `relative_motions.txt`, `imu.csv`);
2. initialize the EKF from the first two anchor poses;
3. propagate the IMU exactly from anchor to anchor;
4. update the EKF with overlapping triplets of relative translations from the transformer;
5. marginalize a single oldest clone after each attempted update.

The transformer's output is assumed to already be available on disk.
Thus, it is an asynchronous implementation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from filter.imu_buffer import ImuInterval, ImuMeasurement
from filter.measurement_triplet import make_default_joint_covariance
from filter.scekf import ImuMSCKF


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from filter_diagnostics import compute_filter_diagnostics, print_filter_run_summary, show_interactive_3d_plot


@dataclass(frozen=True)
class RunnerConfig:
    """User-editable configuration for the minimal relative-motion filter runner."""

    # Paths
    data_root: Path = ROOT / "data"
    dataset: str = "eds"
    sequence: str = "00_peanuts_dark"

    @property
    def processed_root(self) -> Path:
        return self.data_root / self.dataset / "processed"
    
    @property
    def out_dir(self) -> Path:
        return ROOT / "outputs" / "main_filter" / self.dataset

    # Execution modes
    use_gt: bool = False  # Set via CLI argument
    plot_transformer: bool = False 
    plot_imu: bool = False
    interactive_plot: bool = False 
    plot_projections: bool = False

    # Optional sequence truncation
    max_frames: int | None = None

    # # IMU preprocessing
    imu_axis_multipliers: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # IMU process noise
    sigma_na: float = 0.011065875226523246
    sigma_ng: float = 0.01251528557615725
    sigma_nba: float = 6.536078678232154e-05
    sigma_nbg: float = 2.1514640261497524e-05

    # EKF assumed measurement covariance
    assumed_sigma_rel_t: float = 0.02194332115673975
    assumed_sigma_rel_r_deg: float = 2.0
    meas_cov_scale: float = 1.2649054158337365
    meas_cov_axis_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_regressed_covariance: bool = True
    min_regressed_sigma_m: float = 1e-4
    max_regressed_sigma_m: float = 1.0
    chi2_confidence: float = 0.95
    chi2_multiplier: float = 1.0
    enable_chi2_gating: bool = True
    use_fej: bool = True
    use_block_update: bool = False
    covariance_repair_mode: str = "jitter"
    imu_noise_model: str = "discrete"
    imu_interval_mode: str = "sample_dt"
    covariance_propagation_mode: str = "per_sample"
    nominal_integration_method: str = "euler"
    update_solve_method: str = "innovation"
    gating_mode: str = "global"
    fej_scope: str = "clone_update"
    edge_robust_mode: str = "off"
    edge_inflation_factor: float = 100.0
    edge_chi2_multiplier: float = 1.0

    # Optional extra synthetic noise added on top of measurements
    extra_measurement_noise_t: float = 0.0
    seed: int = 7

    # Initialization offsets applied on top of the first anchor pose/velocity
    initial_position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_velocity_offset_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_euler_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_bg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_ba: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity_world_mps2: tuple[float, float, float] = (0.0, 0.0, 9.80665)
    initial_attitude_sigma_deg: float = 0.11534784349262132
    initial_velocity_sigma_mps: float = 1.8658950002457901
    initial_position_sigma_m: float = 0.04181564546764053
    initial_z_sigma_m: float = 0.006867502596918262
    initial_bg_sigma_rps: float = 0.00033573143221825514
    initial_ba_sigma_mps2: float = 0.1779266257977154

# Global instance of default configuration
CONFIG = RunnerConfig()


def _sequence_path(config: RunnerConfig) -> Path:
    """Resolve the processed sequence directory."""

    sequence_path = config.processed_root / config.sequence
    if not sequence_path.exists():
        raise FileNotFoundError(f"Processed sequence folder does not exist: {sequence_path}")
    return sequence_path


def _load_anchor_poses(sequence_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load processed anchor poses from the text file."""

    anchor_path = sequence_path / "anchor_poses.txt"
    # Skip the header and ensure it is a 2D array
    anchor_table = np.atleast_2d(np.loadtxt(anchor_path, dtype=np.float64, skiprows=1))
    if anchor_table.shape[1] != 8:
        raise ValueError(
            f"{anchor_path} has {anchor_table.shape[1]} columns, expected 8: "
            "timestamp px py pz qx qy qz qw."
        )
    # Extract timestamps,positions and quaternions 
    timestamps_us = anchor_table[:, 0].astype(np.int64)
    positions = anchor_table[:, 1:4].astype(np.float64)
    quaternions = anchor_table[:, 4:8].astype(np.float64)
    return timestamps_us, positions, quaternions


def _load_relative_motion_table(sequence_path: Path, use_gt: bool) -> np.ndarray:
    """Load processed relative motions and skip stale non-numeric headers.

    Supported formats:
    - `t0 t1 px py pz`
    - `t0 t1 px py pz sigma_x sigma_y sigma_z`
    - `t0 t1 px py pz qx qy qz qw` for legacy rotation-bearing files. The
      current EKF uses only the translation columns from this format.
    """
    #Chooses file based on configuration
    filename = "relative_motions.txt" if use_gt else f"{sequence_path.name}.txt"
    rel_path = sequence_path / filename
    
    rows: list[list[float]] = []
    with rel_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                # Try to convert the row to numbers
                rows.append([float(value) for value in parts])
            except ValueError:
                continue

    relative_motions = np.asarray(rows, dtype=np.float64)
    supported_columns = {5, 8, 9}
    if relative_motions.ndim != 2 or relative_motions.shape[1] not in supported_columns:
        raise ValueError(
            f"{rel_path} has shape {relative_motions.shape}, expected N x 5, N x 8, or N x 9: "
            "t0 t1 px py pz [sigma_x sigma_y sigma_z | qx qy qz qw]."
        )
    if relative_motions.shape[0] < 4:
        raise ValueError(f"{rel_path} needs at least four rows to form one triplet update.")
    return relative_motions


def _load_sequence_imu(sequence_path: Path) -> np.ndarray:
    """Load one processed IMU table with columns `timestamp gx gy gz ax ay az`."""

    imu_path = sequence_path / "imu.csv"
    imu = np.loadtxt(imu_path, delimiter=",", comments="#", ndmin=2)
    if imu.shape[1] != 7:
        raise ValueError(
            f"{imu_path} has {imu.shape[1]} columns, expected 7: timestamp gx gy gz ax ay az."
        )
    # Sort data by timestamp to ensure causality
    return imu[np.argsort(imu[:, 0])]


def _infer_time_scale_to_seconds(timestamps: np.ndarray) -> float:
    """Infer whether timestamps are in seconds, microseconds, or nanoseconds."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    positive_diffs = np.diff(timestamps)
    positive_diffs = positive_diffs[positive_diffs > 0]
    median_dt = float(np.median(positive_diffs)) if len(positive_diffs) > 0 else 0.0

    if median_dt > 1e7:
        return 1e-9
    if median_dt > 1e1:
        return 1e-6
    return 1.0


def _build_anchor_times_from_relative_motions(relative_motion_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Recover anchor timestamps, translations, and optional diagonal sigmas."""

    raw_times = relative_motion_table[:, :2]
    time_scale = _infer_time_scale_to_seconds(raw_times.reshape(-1))
    # COnvert timestamps in seconds
    edge_start_times_s = raw_times[:, 0].astype(np.float64) * time_scale
    edge_end_times_s = raw_times[:, 1].astype(np.float64) * time_scale
    # Temporal validity check
    if np.any(edge_end_times_s <= edge_start_times_s):
        raise ValueError("Found a non-positive interval in relative motions table.")

    continuity_error = np.max(np.abs(edge_end_times_s[:-1] - edge_start_times_s[1:]))
    if continuity_error > 1e-9:
        raise ValueError(
            "Consecutive relative-motion rows are not time-continuous; "
            f"max discontinuity is {continuity_error:.3e} s."
        )
    # Build a unique array of anchor timestamps by joining the start of the first row and all endings
    anchor_times_s = np.concatenate([edge_start_times_s[:1], edge_end_times_s], axis=0)
    # Extract the translation measurements (px, py, pz)
    relative_measurements = relative_motion_table[:, 2:5].astype(np.float64)
    relative_sigmas = None
    if relative_motion_table.shape[1] == 8:
        relative_sigmas = relative_motion_table[:, 5:8].astype(np.float64)

    return anchor_times_s, relative_measurements, relative_sigmas


def _sanitize_relative_sigmas(
    relative_sigmas: np.ndarray | None,
    config: RunnerConfig,
) -> np.ndarray | None:
    """Validate and clip regressed translation sigmas according to runner config."""

    if relative_sigmas is None or not config.use_regressed_covariance:
        return None

    sigmas = np.asarray(relative_sigmas, dtype=np.float64)
    if sigmas.ndim != 2 or sigmas.shape[1] != 3:
        raise ValueError(f"Expected relative sigmas with shape N x 3, got {sigmas.shape}.")
    if not np.isfinite(sigmas).all():
        raise ValueError("Regressed covariance sigmas contain non-finite values.")
    if np.any(sigmas < 0.0):
        raise ValueError("Regressed covariance sigmas must be non-negative.")
    if config.min_regressed_sigma_m <= 0.0:
        raise ValueError("min_regressed_sigma_m must be positive.")
    if config.max_regressed_sigma_m < config.min_regressed_sigma_m:
        raise ValueError("max_regressed_sigma_m must be >= min_regressed_sigma_m.")

    return np.clip(sigmas, config.min_regressed_sigma_m, config.max_regressed_sigma_m)


def _build_joint_covariance_for_window(
    base_joint_covariance: np.ndarray,
    relative_sigmas: np.ndarray | None,
    start_idx: int,
    axis_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, bool]:
    """Build the 12x12 measurement covariance for one four-edge update window."""

    covariance = np.asarray(base_joint_covariance, dtype=np.float64).copy()
    axis_scale_arr = np.asarray(axis_scale, dtype=np.float64)
    if axis_scale_arr.shape != (3,):
        raise ValueError(f"Expected three axis covariance scale values, got shape {axis_scale_arr.shape}.")
    if not np.isfinite(axis_scale_arr).all() or np.any(axis_scale_arr <= 0.0):
        raise ValueError("Measurement covariance axis scale values must be finite and positive.")
    if relative_sigmas is None:
        return covariance, False

    sigmas = np.asarray(relative_sigmas[start_idx : start_idx + 4], dtype=np.float64)
    if sigmas.shape != (4, 3):
        raise ValueError(f"Expected four rows of 3D sigmas for update covariance, got {sigmas.shape}.")
    sigmas = sigmas * axis_scale_arr[None, :]
    np.fill_diagonal(covariance, sigmas.reshape(-1) ** 2)
    return covariance, True


def _validate_anchor_alignment(
    anchor_timestamps_us: np.ndarray,
    relative_anchor_times_s: np.ndarray,
    relative_measurements: np.ndarray,
    relative_sigmas: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Ensure anchor_poses.txt and relative_motions.txt describe the same timeline."""

    anchor_times_s = anchor_timestamps_us.astype(np.float64) * 1e-6
    if len(anchor_times_s) != len(relative_anchor_times_s):
        raise ValueError(
            "anchor_poses.txt and relative motions disagree on the number of anchors: "
            f"{len(anchor_times_s)} vs {len(relative_anchor_times_s)}."
        )

    max_error = float(np.max(np.abs(anchor_times_s - relative_anchor_times_s)))
    if max_error > 1e-9:
        raise ValueError(
            "anchor_poses.txt and relative motions disagree on anchor timestamps; "
            f"maximum mismatch is {max_error:.3e} s."
        )

    if relative_measurements.shape[0] != len(anchor_times_s) - 1:
        raise ValueError(
            "Expected one relative-motion row per consecutive anchor pair, got "
            f"{relative_measurements.shape[0]} rows for {len(anchor_times_s)} anchors."
        )

    return anchor_timestamps_us, anchor_times_s, relative_measurements, relative_sigmas


def _truncate_sequence(
    anchor_timestamps_us: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_quaternions: np.ndarray,
    relative_measurements: np.ndarray,
    relative_sigmas: np.ndarray | None,
    max_frames: int | None,
):
    """Optionally keep only the first `max_frames` anchors."""

    if max_frames is None:
        return anchor_timestamps_us, anchor_positions, anchor_quaternions, relative_measurements, relative_sigmas

    if max_frames < 5:
        raise ValueError("`max_frames` must be at least 5 to run triplet updates.")
    # Truncate all related arrays
    anchor_timestamps_us = anchor_timestamps_us[:max_frames]
    anchor_positions = anchor_positions[:max_frames]
    anchor_quaternions = anchor_quaternions[:max_frames]
    relative_measurements = relative_measurements[: max(0, max_frames - 1)]
    if relative_sigmas is not None:
        relative_sigmas = relative_sigmas[: max(0, max_frames - 1)]
        
    return anchor_timestamps_us, anchor_positions, anchor_quaternions, relative_measurements, relative_sigmas


def _build_exact_imu_segment(
    raw_times_s: np.ndarray,
    raw_gyro: np.ndarray,
    raw_accel: np.ndarray,
    start_time_s: float,
    end_time_s: float,
) -> list[ImuMeasurement]:
    """Resample the IMU stream so propagation lands exactly on `end_time_s`."""

    if end_time_s <= start_time_s:
        return []

    if start_time_s < raw_times_s[0] or end_time_s > raw_times_s[-1]:
        raise ValueError("Requested IMU propagation interval falls outside the IMU time range.")
    if not np.isfinite(raw_times_s).all() or not np.isfinite(raw_gyro).all() or not np.isfinite(raw_accel).all():
        raise ValueError("IMU stream contains non-finite values.")
    # Mask to take only IMU readings between the start and end of the interval
    interior_mask = (raw_times_s > start_time_s) & (raw_times_s < end_time_s)
    segment_times = list(raw_times_s[interior_mask])
    segment_times.append(float(end_time_s))
    if len(segment_times) == 0:
        raise ValueError("Exact IMU segment construction produced no samples.")
    segment_times_arr = np.asarray(segment_times, dtype=np.float64)
    if not np.isfinite(segment_times_arr).all():
        raise ValueError("Exact IMU segment contains non-finite timestamps.")
    if np.any(np.diff(segment_times_arr) <= 0.0):
        raise ValueError("Exact IMU segment contains duplicate or non-increasing timestamps.")
    if not np.isclose(segment_times_arr[-1], end_time_s, rtol=0.0, atol=1e-12):
        raise ValueError("Exact IMU segment does not end at the requested timestamp.")
    # Interpolate gyroscope and accelerometer for the exact final instant
    gyro_interp = np.column_stack(
        [np.interp(segment_times, raw_times_s, raw_gyro[:, axis]) for axis in range(3)]
    )
    accel_interp = np.column_stack(
        [np.interp(segment_times, raw_times_s, raw_accel[:, axis]) for axis in range(3)]
    )
    # Build the list of ImuMeasurement objects by calculating time deltas (dt)
    measurements: list[ImuMeasurement] = []
    prev_time_s = float(start_time_s)
    for sample_idx, timestamp_s in enumerate(segment_times):
        timestamp_s = float(timestamp_s)
        dt = timestamp_s - prev_time_s
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Exact IMU segment produced non-positive dt: {dt}.")
        accel_sample = accel_interp[sample_idx].astype(np.float64)
        gyro_sample = gyro_interp[sample_idx].astype(np.float64)
        if not np.isfinite(accel_sample).all() or not np.isfinite(gyro_sample).all():
            raise ValueError("Exact IMU segment interpolation produced non-finite values.")
        measurements.append(
            ImuMeasurement(
                timestamp=timestamp_s,
                dt=dt,
                accel=accel_sample,
                gyro=gyro_sample,
            )
        )
        prev_time_s = timestamp_s

    return measurements


def _build_exact_imu_intervals(
    raw_times_s: np.ndarray,
    raw_gyro: np.ndarray,
    raw_accel: np.ndarray,
    start_time_s: float,
    end_time_s: float,
) -> list[ImuInterval]:
    """Build explicit IMU sample intervals that include exact start and end times."""

    if end_time_s <= start_time_s:
        return []

    if start_time_s < raw_times_s[0] or end_time_s > raw_times_s[-1]:
        raise ValueError("Requested IMU propagation interval falls outside the IMU time range.")
    if not np.isfinite(raw_times_s).all() or not np.isfinite(raw_gyro).all() or not np.isfinite(raw_accel).all():
        raise ValueError("IMU stream contains non-finite values.")
    if np.any(np.diff(raw_times_s) <= 0.0):
        raise ValueError("IMU timestamps must be strictly increasing.")

    interior_mask = (raw_times_s > start_time_s) & (raw_times_s < end_time_s)
    interval_times = np.concatenate(
        [
            np.array([float(start_time_s)], dtype=np.float64),
            raw_times_s[interior_mask].astype(np.float64),
            np.array([float(end_time_s)], dtype=np.float64),
        ]
    )
    if not np.isfinite(interval_times).all():
        raise ValueError("Exact IMU intervals contain non-finite timestamps.")
    if np.any(np.diff(interval_times) <= 0.0):
        raise ValueError("Exact IMU intervals contain duplicate or non-increasing timestamps.")

    gyro_interp = np.column_stack(
        [np.interp(interval_times, raw_times_s, raw_gyro[:, axis]) for axis in range(3)]
    )
    accel_interp = np.column_stack(
        [np.interp(interval_times, raw_times_s, raw_accel[:, axis]) for axis in range(3)]
    )
    if not np.isfinite(gyro_interp).all() or not np.isfinite(accel_interp).all():
        raise ValueError("Exact IMU interval interpolation produced non-finite values.")

    intervals: list[ImuInterval] = []
    for sample_idx in range(len(interval_times) - 1):
        t0 = float(interval_times[sample_idx])
        t1 = float(interval_times[sample_idx + 1])
        dt = t1 - t0
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Exact IMU interval produced non-positive dt: {dt}.")
        intervals.append(
            ImuInterval(
                t0=t0,
                t1=t1,
                accel0=accel_interp[sample_idx].astype(np.float64),
                gyro0=gyro_interp[sample_idx].astype(np.float64),
                accel1=accel_interp[sample_idx + 1].astype(np.float64),
                gyro1=gyro_interp[sample_idx + 1].astype(np.float64),
            )
        )

    return intervals


def _build_anchor_imu_segments(
    imu_table: np.ndarray,
    anchor_timestamps_us: np.ndarray,
    axis_multipliers: tuple[float, float, float],
) -> list[list[ImuMeasurement]]:
    """Precompute one exact propagation segment for each consecutive anchor pair."""

    time_scale = _infer_time_scale_to_seconds(imu_table[:, 0])
    raw_times_s = imu_table[:, 0].astype(np.float64) * time_scale
    if np.any(np.diff(raw_times_s) <= 0.0):
        raise ValueError("IMU timestamps must be strictly increasing.")
    raw_gyro = imu_table[:, 1:4].astype(np.float64)
    raw_accel = imu_table[:, 4:7].astype(np.float64)
    # Axis multipliers (to invert axes if necessary)
    axis_multipliers_arr = np.asarray(axis_multipliers, dtype=np.float64)
    raw_gyro = raw_gyro * axis_multipliers_arr
    raw_accel = raw_accel * axis_multipliers_arr
    anchor_times_s = anchor_timestamps_us.astype(np.float64) * 1e-6

    if anchor_times_s[0] < raw_times_s[0] or anchor_times_s[-1] > raw_times_s[-1]:
        raise ValueError("Anchor timestamps fall outside the IMU stream.")
    # Create a list of segments, each containing IMU readings to go from anchor i to i+1
    segments: list[list[ImuMeasurement]] = []
    for idx in range(len(anchor_times_s) - 1):
        segments.append(
            _build_exact_imu_segment(
                raw_times_s,
                raw_gyro,
                raw_accel,
                anchor_times_s[idx],
                anchor_times_s[idx + 1],
            )
        )
    return segments


def _build_anchor_imu_intervals(
    imu_table: np.ndarray,
    anchor_timestamps_us: np.ndarray,
    axis_multipliers: tuple[float, float, float],
) -> list[list[ImuInterval]]:
    """Precompute one explicit interval list for each consecutive anchor pair."""

    time_scale = _infer_time_scale_to_seconds(imu_table[:, 0])
    raw_times_s = imu_table[:, 0].astype(np.float64) * time_scale
    if np.any(np.diff(raw_times_s) <= 0.0):
        raise ValueError("IMU timestamps must be strictly increasing.")
    raw_gyro = imu_table[:, 1:4].astype(np.float64)
    raw_accel = imu_table[:, 4:7].astype(np.float64)
    axis_multipliers_arr = np.asarray(axis_multipliers, dtype=np.float64)
    raw_gyro = raw_gyro * axis_multipliers_arr
    raw_accel = raw_accel * axis_multipliers_arr
    anchor_times_s = anchor_timestamps_us.astype(np.float64) * 1e-6

    if anchor_times_s[0] < raw_times_s[0] or anchor_times_s[-1] > raw_times_s[-1]:
        raise ValueError("Anchor timestamps fall outside the IMU stream.")

    segments: list[list[ImuInterval]] = []
    for idx in range(len(anchor_times_s) - 1):
        segments.append(
            _build_exact_imu_intervals(
                raw_times_s,
                raw_gyro,
                raw_accel,
                anchor_times_s[idx],
                anchor_times_s[idx + 1],
            )
        )
    return segments


def _make_filter_args(config: RunnerConfig) -> SimpleNamespace:
    """Create the args namespace consumed by `ImuMSCKF`."""

    return SimpleNamespace(
        sigma_na=float(config.sigma_na),
        sigma_ng=float(config.sigma_ng),
        sigma_nba=float(config.sigma_nba),
        sigma_nbg=float(config.sigma_nbg),
        sigma_rel_t=float(config.assumed_sigma_rel_t),
        sigma_rel_r=float(np.deg2rad(config.assumed_sigma_rel_r_deg)),
        meas_cov_scale=float(config.meas_cov_scale),
        initial_attitude_sigma_rad=float(np.deg2rad(config.initial_attitude_sigma_deg)),
        initial_velocity_sigma_mps=float(config.initial_velocity_sigma_mps),
        initial_position_sigma_m=float(config.initial_position_sigma_m),
        initial_z_sigma_m=float(config.initial_z_sigma_m),
        initial_bg_sigma_rps=float(config.initial_bg_sigma_rps),
        initial_ba_sigma_mps2=float(config.initial_ba_sigma_mps2),
        chi2_confidence=float(config.chi2_confidence),
        chi2_multiplier=float(config.chi2_multiplier),
        enable_chi2_gating=bool(config.enable_chi2_gating),
        use_fej=bool(config.use_fej),
        use_block_update=bool(config.use_block_update),
        covariance_repair_mode=str(config.covariance_repair_mode),
        imu_noise_model=str(config.imu_noise_model),
        covariance_propagation_mode=str(config.covariance_propagation_mode),
        nominal_integration_method=str(config.nominal_integration_method),
        update_solve_method=str(config.update_solve_method),
        gating_mode=str(config.gating_mode),
        fej_scope=str(config.fej_scope),
        edge_robust_mode=str(config.edge_robust_mode),
        edge_inflation_factor=float(config.edge_inflation_factor),
        edge_chi2_multiplier=float(config.edge_chi2_multiplier),
    )


def _apply_initial_offsets(
    R0: np.ndarray,
    v0: np.ndarray,
    p0: np.ndarray,
    config: RunnerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply user-configurable initialization offsets on top of the anchor state."""

    euler_offset_rad = np.deg2rad(np.asarray(config.initial_euler_offset_deg, dtype=np.float64))
    R_offset = Rotation.from_euler("xyz", euler_offset_rad).as_matrix()
    p_offset = np.asarray(config.initial_position_offset_m, dtype=np.float64)
    v_offset = np.asarray(config.initial_velocity_offset_mps, dtype=np.float64)
    return R0 @ R_offset, v0 + v_offset, p0 + p_offset


def _state_to_row(timestamp_s: float, ekf_state) -> np.ndarray:
    """Convert one EKF state into a text-friendly row."""

    quaternion_xyzw = Rotation.from_matrix(ekf_state.R).as_quat()
    return np.concatenate(
        [
            np.array([float(timestamp_s)], dtype=np.float64),
            ekf_state.p.astype(np.float64),
            quaternion_xyzw.astype(np.float64),
        ]
    )


def _save_trajectory(path: Path, trajectory_table: np.ndarray) -> Path:
    """Save one trajectory table with timestamp, position, and quaternion."""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "timestamp_s px py pz qx qy qz qw"
    np.savetxt(path, trajectory_table, fmt="%.9f", header=header, comments="")
    return path


def _save_update_diagnostics(path: Path, rows: list[dict]) -> Path:
    """Save per-update innovation and covariance diagnostics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "anchor_idx",
        "timestamp_s",
        "accepted",
        "rejected",
        "mahalanobis_sq",
        "chi2_threshold",
        "chi2_ratio",
        "residual_norm",
        "correction_norm",
        "sigma_min",
        "sigma_max",
        "sigma_mean",
        "update_solve_method",
        "condition_number_R",
        "condition_number_S",
        "whitening_applied",
        "whitening_repaired_R",
        "edge_robust_mode",
        "num_inflated_edges",
        "inflated_edge_indices",
        "edge_rejected",
        "edge0_chi2_ratio",
        "edge1_chi2_ratio",
        "edge2_chi2_ratio",
        "edge3_chi2_ratio",
        "max_edge_chi2_ratio",
    ]
    for component_idx in range(12):
        fieldnames.append(f"residual_{component_idx}")
    for component_idx in range(12):
        fieldnames.append(f"sigma_{component_idx}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _propagate_anchor_segment(
    ekf: ImuMSCKF,
    sample_segment: list[ImuMeasurement],
    interval_segment: list[ImuInterval] | None,
    config: RunnerConfig,
) -> None:
    """Dispatch propagation through the selected IMU input representation."""

    if config.imu_interval_mode == "sample_dt":
        ekf.propagate(sample_segment)
    elif config.imu_interval_mode == "paired_samples":
        if interval_segment is None:
            raise ValueError("Paired-sample propagation requires precomputed IMU intervals.")
        ekf.propagate_intervals(interval_segment)
    else:
        raise ValueError(f"Unknown IMU interval mode {config.imu_interval_mode!r}.")


def _summarize_chi2_ratios(chi2_ratios: list[float]) -> dict[str, float | None]:
    """Summarize normalized innovation gating ratios."""

    if not chi2_ratios:
        return {
            "median_chi2_ratio": None,
            "p95_chi2_ratio": None,
            "max_chi2_ratio": None,
        }
    values = np.asarray(chi2_ratios, dtype=np.float64)
    return {
        "median_chi2_ratio": float(np.median(values)),
        "p95_chi2_ratio": float(np.percentile(values, 95)),
        "max_chi2_ratio": float(np.max(values)),
    }


def _build_ground_truth_trajectory(
    anchor_timestamps_us: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_quaternions: np.ndarray,
) -> np.ndarray:
    """Pack the anchor poses into the same trajectory-table format as the estimate."""

    return np.column_stack(
        [
            anchor_timestamps_us.astype(np.float64) * 1e-6,
            anchor_positions.astype(np.float64),
            anchor_quaternions.astype(np.float64),
        ]
    )

def _compute_transformer_trajectory(
    sequence_path: Path,
    anchor_timestamps_us: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_quaternions: np.ndarray,
    max_frames: int | None
) -> np.ndarray | None:
    """
    Computes the trajectory based solely on the Transformer's
    predictions, without any EKF filtering. 
    Useful for visualizing the raw network performance in the 3D plot.
    """
    try:
        # Reload the Transformer's predictions 
        regressed_table = _load_relative_motion_table(sequence_path, use_gt=False)
        
        reg_dp = regressed_table[:, 2:5]
        
        # Apply truncation if requested
        if max_frames is not None:
            reg_dp = reg_dp[: max(0, max_frames - 1)]
                
        # Ensure the lengths match
        if len(reg_dp) == len(anchor_timestamps_us) - 1:
            regr_positions = [anchor_positions[0].astype(np.float64)]
            regr_quaternions = [anchor_quaternions[0].astype(np.float64)]
            
            # Integrate the poses 
            for i in range(len(reg_dp)):
                R_curr = Rotation.from_quat(regr_quaternions[-1])
                
                # Translate into the global frame
                p_next = regr_positions[-1] + R_curr.as_matrix() @ reg_dp[i]
                regr_positions.append(p_next)

                regr_quaternions.append(anchor_quaternions[i + 1])
                    
            return _build_ground_truth_trajectory(
                anchor_timestamps_us,
                np.array(regr_positions),
                np.array(regr_quaternions), 
            )
            
    except Exception as e:
        print(f"Warning: Failed to generate the regressed trajectory for the plot: {e}")
        
    return None


def run_filter(config: RunnerConfig) -> dict:
    """Run the relative-motion EKF on one processed sequence."""

    if config.imu_interval_mode != "paired_samples":
        if config.covariance_propagation_mode != "per_sample":
            raise ValueError("Summed covariance propagation requires --imu_interval_mode paired_samples.")
        if config.nominal_integration_method != "euler":
            raise ValueError("Midpoint nominal integration requires --imu_interval_mode paired_samples.")
    axis_scale_arr = np.asarray(config.meas_cov_axis_scale, dtype=np.float64)
    if axis_scale_arr.shape != (3,) or not np.isfinite(axis_scale_arr).all() or np.any(axis_scale_arr <= 0.0):
        raise ValueError("meas_cov_axis_scale must contain three finite positive values.")
    if config.edge_robust_mode not in {"off", "inflate", "reject"}:
        raise ValueError(f"Unknown edge robust mode {config.edge_robust_mode!r}.")
    if not np.isfinite(config.edge_inflation_factor) or config.edge_inflation_factor <= 0.0:
        raise ValueError("edge_inflation_factor must be finite and positive.")
    if not np.isfinite(config.edge_chi2_multiplier) or config.edge_chi2_multiplier <= 0.0:
        raise ValueError("edge_chi2_multiplier must be finite and positive.")

    # Directory path for the given dataset sequence
    sequence_path = _sequence_path(config)

    # Load foundational data: anchor ground truths and IMU inputs
    anchor_timestamps_us, anchor_positions, anchor_quaternions = _load_anchor_poses(sequence_path)
    # Pass `config.use_gt` to determine which table to load
    relative_motion_table = _load_relative_motion_table(sequence_path, config.use_gt)
    imu_table = _load_sequence_imu(sequence_path)

    # Extract EKF measurement bounds (times & translations) from the relative motion table
    relative_anchor_times_s, relative_measurements, relative_sigmas = _build_anchor_times_from_relative_motions(
        relative_motion_table
    )
    relative_sigmas = _sanitize_relative_sigmas(relative_sigmas, config)

    # Ensure that anchor_poses timeline perfectly matches the relative_motions timeline
    anchor_timestamps_us, _, relative_measurements, relative_sigmas = _validate_anchor_alignment(
        anchor_timestamps_us,
        relative_anchor_times_s,
        relative_measurements,
        relative_sigmas
    )
    anchor_timestamps_us, anchor_positions, anchor_quaternions, relative_measurements, relative_sigmas = _truncate_sequence(
        anchor_timestamps_us,
        anchor_positions,
        anchor_quaternions,
        relative_measurements,
        relative_sigmas,
        config.max_frames,
    )

    if len(anchor_timestamps_us) < 3:
        raise ValueError("Need at least three anchors to run the triplet EKF update.")

    # Slice the IMU data stream to exactly match the durations between anchors
    anchor_imu_segments = _build_anchor_imu_segments(
        imu_table,
        anchor_timestamps_us,
        config.imu_axis_multipliers,
    )
    anchor_imu_intervals = None
    if config.imu_interval_mode == "paired_samples":
        anchor_imu_intervals = _build_anchor_imu_intervals(
            imu_table,
            anchor_timestamps_us,
            config.imu_axis_multipliers,
        )

    # Determine absolute starting states from the first two available ground truth anchors
    anchor_times_s = anchor_timestamps_us.astype(np.float64) * 1e-6
    p0 = anchor_positions[0].astype(np.float64)
    R0 = Rotation.from_quat(anchor_quaternions[0]).as_matrix()
    dt0 = max(anchor_times_s[1] - anchor_times_s[0], 1e-9)
    v0 = (anchor_positions[1] - anchor_positions[0]) / dt0
    R0, v0, p0 = _apply_initial_offsets(R0, v0.astype(np.float64), p0, config)
    bg0 = np.asarray(config.initial_bg, dtype=np.float64)
    ba0 = np.asarray(config.initial_ba, dtype=np.float64)
    # Initialize the MSCKF
    ekf = ImuMSCKF(_make_filter_args(config))
    ekf.g = np.asarray(config.gravity_world_mps2, dtype=np.float64)
    ekf.initialize_with_state(anchor_times_s[0], R0, v0, p0, bg0, ba0)
    # Setting for IMU plot
    imu_trajectory_rows = []
    if config.plot_imu:
        imu_ekf = ImuMSCKF(_make_filter_args(config))
        imu_ekf.g = np.asarray(config.gravity_world_mps2, dtype=np.float64)
        imu_ekf.initialize_with_state(anchor_times_s[0], R0.copy(), v0.copy(), p0.copy(), bg0.copy(), ba0.copy())
        imu_trajectory_rows.append(_state_to_row(anchor_times_s[0], imu_ekf.state))
    # Pre-configure random noise generator and joint covariance matrices for measurements
    rng = np.random.default_rng(config.seed)
    joint_covariance = make_default_joint_covariance(float(config.assumed_sigma_rel_t))
    # Diagnostics setup
    trajectory_rows = [_state_to_row(anchor_times_s[0], ekf.state)]
    residual_norms: list[float] = []
    delta_norms: list[float] = []
    mahalanobis_values: list[float] = []
    chi2_ratios: list[float] = []
    update_diagnostic_rows: list[dict] = []
    rejected_updates = 0
    accepted_updates = 0
    used_regressed_covariance = False
    # The first state clone (anchor 0) is recorded before propagating
    ekf.augment_clone()
    # Propagate EKF state forward using IMU segment 0 and augment clone (Initialization)
    for anchor_idx in range(1, 4):
        _propagate_anchor_segment(
            ekf,
            anchor_imu_segments[anchor_idx - 1],
            anchor_imu_intervals[anchor_idx - 1] if anchor_imu_intervals is not None else None,
            config,
        )
        ekf.augment_clone()
        trajectory_rows.append(_state_to_row(anchor_times_s[anchor_idx], ekf.state))

        if config.plot_imu:
            _propagate_anchor_segment(
                imu_ekf,
                anchor_imu_segments[anchor_idx - 1],
                anchor_imu_intervals[anchor_idx - 1] if anchor_imu_intervals is not None else None,
                config,
            )
            imu_trajectory_rows.append(_state_to_row(anchor_times_s[anchor_idx], imu_ekf.state))

    # MAIN FILTER LOOP (Triplets)
    # Iterating starting from anchor 2 ensures we have a triplet window available
    for anchor_idx in range(4, len(anchor_times_s)):
        # Prediction Step (IMU Integration)
        _propagate_anchor_segment(
            ekf,
            anchor_imu_segments[anchor_idx - 1],
            anchor_imu_intervals[anchor_idx - 1] if anchor_imu_intervals is not None else None,
            config,
        )
        ekf.augment_clone()

        if config.plot_imu:
            _propagate_anchor_segment(
                imu_ekf,
                anchor_imu_segments[anchor_idx - 1],
                anchor_imu_intervals[anchor_idx - 1] if anchor_imu_intervals is not None else None,
                config,
            )
            imu_trajectory_rows.append(_state_to_row(anchor_times_s[anchor_idx], imu_ekf.state))

        # Measurement Extraction
        measurement = relative_measurements[anchor_idx - 4 : anchor_idx].copy()
        # Optionally inject extra noise into the measurement for testing
        if config.extra_measurement_noise_t > 0.0:
            measurement += rng.normal(
                scale=float(config.extra_measurement_noise_t),
                size=measurement.shape,
            )
        current_joint_covariance, used_regressed_for_update = _build_joint_covariance_for_window(
            joint_covariance,
            relative_sigmas,
            anchor_idx - 4,
            axis_scale=config.meas_cov_axis_scale,
        )
        used_regressed_covariance = used_regressed_covariance or used_regressed_for_update
        # Measurement update
        update_info = ekf.update(
            {
                "relative_pose": measurement,
                "joint_covariance": current_joint_covariance,
            }
        )
        # Diagnostics
        if update_info.get("rejected", False):
            rejected_updates += 1
        else:
            accepted_updates += 1
            residual_norms.append(float(np.linalg.norm(update_info["residual"])))
            delta_norms.append(float(np.linalg.norm(update_info["delta_x"])))
        if update_info.get("mahalanobis_sq") is not None:
            mahalanobis = float(update_info["mahalanobis_sq"])
            mahalanobis_values.append(mahalanobis)
            threshold = float(update_info["chi2_threshold"])
            if threshold > 0.0 and np.isfinite(threshold):
                chi2_ratios.append(mahalanobis / threshold)
        residual = np.asarray(update_info["residual"], dtype=np.float64)
        sigma_vector = np.sqrt(np.clip(np.diag(current_joint_covariance), 0.0, np.inf))
        chi2_threshold = float(update_info["chi2_threshold"])
        chi2_ratio = (
            float(update_info["mahalanobis_sq"]) / chi2_threshold
            if chi2_threshold > 0.0 and np.isfinite(chi2_threshold)
            else np.nan
        )
        diagnostic_row = {
            "anchor_idx": int(anchor_idx),
            "timestamp_s": float(anchor_times_s[anchor_idx]),
            "accepted": int(not update_info.get("rejected", False)),
            "rejected": int(update_info.get("rejected", False)),
            "mahalanobis_sq": float(update_info["mahalanobis_sq"]),
            "chi2_threshold": chi2_threshold,
            "chi2_ratio": chi2_ratio,
            "residual_norm": float(np.linalg.norm(residual)),
            "correction_norm": (
                float(np.linalg.norm(update_info["delta_x"]))
                if update_info.get("delta_x") is not None
                else np.nan
            ),
            "sigma_min": float(np.min(sigma_vector)),
            "sigma_max": float(np.max(sigma_vector)),
            "sigma_mean": float(np.mean(sigma_vector)),
            "update_solve_method": str(update_info.get("update_solve_method", config.update_solve_method)),
            "condition_number_R": float(update_info.get("condition_number_R", np.nan)),
            "condition_number_S": float(update_info.get("condition_number_S", np.nan)),
            "whitening_applied": int(bool(update_info.get("whitening_applied", False))),
            "whitening_repaired_R": int(bool(update_info.get("whitening_repaired_R", False))),
            "edge_robust_mode": str(update_info.get("edge_robust_mode", config.edge_robust_mode)),
            "num_inflated_edges": int(update_info.get("num_inflated_edges", 0)),
            "inflated_edge_indices": " ".join(str(idx) for idx in update_info.get("inflated_edge_indices", [])),
            "edge_rejected": int(bool(update_info.get("edge_rejected", False))),
        }
        edge_ratios = list(update_info.get("edge_chi2_ratios", [np.nan, np.nan, np.nan, np.nan]))
        for edge_idx in range(4):
            diagnostic_row[f"edge{edge_idx}_chi2_ratio"] = float(edge_ratios[edge_idx])
        diagnostic_row["max_edge_chi2_ratio"] = float(np.nanmax(edge_ratios))
        for component_idx, value in enumerate(residual.reshape(-1)):
            diagnostic_row[f"residual_{component_idx}"] = float(value)
        for component_idx, value in enumerate(sigma_vector.reshape(-1)):
            diagnostic_row[f"sigma_{component_idx}"] = float(value)
        update_diagnostic_rows.append(diagnostic_row)
        # Log current state after correction
        trajectory_rows.append(_state_to_row(anchor_times_s[anchor_idx], ekf.state))
        # Drop the oldest historical clone from the sliding window
        ekf.marginalize_oldest_clone()

    # Save the EKF's finalized estimate
    sequence_out_dir = config.out_dir / config.sequence
    trajectory_table = np.asarray(trajectory_rows, dtype=np.float64)
    saved_path = _save_trajectory(
        sequence_out_dir / f"stamped_traj_estimate.txt",
        trajectory_table,
    )
    update_diagnostics_path = _save_update_diagnostics(
        sequence_out_dir / "update_diagnostics.csv",
        update_diagnostic_rows,
    )
    ground_truth_trajectory = _build_ground_truth_trajectory(
        anchor_timestamps_us,
        anchor_positions,
        anchor_quaternions,
    )

    regressed_trajectory = None
    #Adds a plot for the transformer output before EKF
    if config.plot_transformer:
        regressed_trajectory = _compute_transformer_trajectory(
            sequence_path,
            anchor_timestamps_us,
            anchor_positions,
            anchor_quaternions,
            config.max_frames
        )
    imu_trajectory = np.asarray(imu_trajectory_rows, dtype=np.float64) if config.plot_imu else None
        
    # Diagnostics: compute error metrics
    diagnostics = compute_filter_diagnostics(
        trajectory_table,
        ground_truth_trajectory,
        regressed_trajectory=regressed_trajectory,
        imu_trajectory= imu_trajectory,
        output_dir=sequence_out_dir,
        file_prefix=config.sequence,
        plot_projections=config.plot_projections,
    )

    chi2_summary = _summarize_chi2_ratios(chi2_ratios)

    return {
        "dataset": config.dataset,
        "sequence": config.sequence,
        "num_anchors": int(len(anchor_times_s)),
        "num_updates_attempted": int(len(anchor_times_s) - 4),
        "num_updates_accepted": int(accepted_updates),
        "num_updates_rejected": int(rejected_updates),
        "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else None,
        "mean_delta_norm": float(np.mean(delta_norms)) if delta_norms else None,
        "mean_mahalanobis_sq": float(np.mean(mahalanobis_values)) if mahalanobis_values else None,
        "max_mahalanobis_sq": float(np.max(mahalanobis_values)) if mahalanobis_values else None,
        "mean_chi2_ratio": float(np.mean(chi2_ratios)) if chi2_ratios else None,
        "median_chi2_ratio": chi2_summary["median_chi2_ratio"],
        "p95_chi2_ratio": chi2_summary["p95_chi2_ratio"],
        "max_chi2_ratio": chi2_summary["max_chi2_ratio"],
        "used_regressed_covariance": bool(used_regressed_covariance),
        "use_fej": bool(config.use_fej),
        "use_block_update": bool(config.use_block_update),
        "covariance_repair_events": [
            diagnostic for diagnostic in ekf.covariance_diagnostics if diagnostic.get("repair_applied", False)
        ],
        "trajectory": trajectory_table,
        "ground_truth": ground_truth_trajectory,
        "regressed": regressed_trajectory,
        "imu_only": imu_trajectory,
        "saved_file": str(saved_path),
        "update_diagnostics_file": str(update_diagnostics_path),
        "diagnostics": diagnostics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument(
        "--gt", 
        action="store_true",
        help="Uses the ground truth for the updates, default regressed_relative_motions.txt (Transformer output)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=CONFIG.dataset, 
        help="Dataset folder name to process (e.g., 'eds', 'tartanair')"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=CONFIG.sequence, 
        help="Sequence folder name to process (e.g., '00_peanuts_dark', '01_peanuts_light', '03_rocket_earth_dark')"
    )
    parser.add_argument(
        "--plot_transformer",
        action="store_true",
        help="Plots the regressed trajectory from the Transformer."
    )
    parser.add_argument(
        "--plot_imu",
        action="store_true",
        help="Plots the open-loop IMU integration trajectory."
    )
    parser.add_argument(
        "--interactive_plot",
        action="store_true",
        help="Opens the interactive 3D plot window at the end of the run."
    )
    parser.add_argument(
        "--plot_projections",
        action="store_true",
        help="Saves the 2D projections plots."
    )
    parser.add_argument(
        "--fixed_covariance",
        action="store_true",
        help="Ignore regressed sigma columns and use the fixed assumed translation covariance.",
    )
    parser.add_argument(
        "--meas_cov_scale",
        type=float,
        default=CONFIG.meas_cov_scale,
        help="Global multiplier applied to the 12D measurement covariance inside the EKF update.",
    )
    parser.add_argument(
        "--meas_cov_axis_scale",
        type=float,
        nargs=3,
        default=CONFIG.meas_cov_axis_scale,
        metavar=("SX", "SY", "SZ"),
        help="Per-axis sigma scale applied to regressed translation covariance before building each 12D window.",
    )
    parser.add_argument(
        "--use_fej",
        action="store_true",
        default=CONFIG.use_fej,
        help="Use first-estimate clone poses for translation-update Jacobians. Enabled by default.",
    )
    parser.add_argument(
        "--disable_fej",
        action="store_true",
        help="Disable first-estimate Jacobians for an ablation run.",
    )
    parser.add_argument(
        "--block_update",
        action="store_true",
        help="Use the block Kalman update path instead of the dense path.",
    )
    parser.add_argument(
        "--disable_chi2_gating",
        action="store_true",
        help="Compute chi-square diagnostics but do not reject updates.",
    )
    parser.add_argument(
        "--covariance_repair_mode",
        choices=("strict", "jitter", "clip"),
        default=CONFIG.covariance_repair_mode,
        help="Covariance repair policy used by the EKF.",
    )
    parser.add_argument(
        "--imu_noise_model",
        choices=("discrete", "continuous"),
        default=CONFIG.imu_noise_model,
        help="Interpretation of IMU noise parameters for covariance propagation.",
    )
    parser.add_argument(
        "--imu_interval_mode",
        choices=("sample_dt", "paired_samples"),
        default=CONFIG.imu_interval_mode,
        help="Select the original sample-dt propagation or explicit paired IMU intervals.",
    )
    parser.add_argument(
        "--covariance_propagation_mode",
        choices=("per_sample", "summed"),
        default=CONFIG.covariance_propagation_mode,
        help="Apply IMU covariance each interval or accumulate one summed transition/noise pair.",
    )
    parser.add_argument(
        "--nominal_integration_method",
        choices=("euler", "midpoint", "midpoint_half_R"),
        default=CONFIG.nominal_integration_method,
        help="Nominal IMU integration rule for paired-sample propagation.",
    )
    parser.add_argument(
        "--update_solve_method",
        choices=("innovation", "whitened", "qr"),
        default=CONFIG.update_solve_method,
        help="Measurement update conditioning method.",
    )
    parser.add_argument(
        "--gating_mode",
        choices=("global",),
        default=CONFIG.gating_mode,
        help="Chi-square gating mode. Only global 12D gating is implemented in this pass.",
    )
    parser.add_argument(
        "--fej_scope",
        choices=("clone_update",),
        default=CONFIG.fej_scope,
        help="FEJ scope. Full propagation FEJ is intentionally not enabled in this pass.",
    )
    parser.add_argument(
        "--edge_robust_mode",
        choices=("off", "inflate", "reject"),
        default=CONFIG.edge_robust_mode,
        help="Per-edge robust handling mode for the four 3D relative-translation residuals.",
    )
    parser.add_argument(
        "--edge_inflation_factor",
        type=float,
        default=CONFIG.edge_inflation_factor,
        help="Covariance multiplier for failed edges when edge robust mode is inflate.",
    )
    parser.add_argument(
        "--edge_chi2_multiplier",
        type=float,
        default=CONFIG.edge_chi2_multiplier,
        help="Additional multiplier for 3D per-edge chi-square thresholds.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the minimal relative-motion filter with the top-of-file configuration."""
    args = parse_args()
    # Deals with different axis directions of EDS dataset
    dataset_params = {}
    dataset_name = args.dataset.lower()
    
    if dataset_name == "eds":
        dataset_params = {
            "imu_axis_multipliers": (-1.0, -1.0, 1.0),
            "gravity_world_mps2": (0.0, 0.0, -9.80665)
        }

    # Create a new config based on CONFIG but overriding attributes from args
    active_config = replace(
        CONFIG, 
        use_gt=args.gt, 
        dataset=args.dataset,
        sequence=args.sequence,
        plot_transformer=args.plot_transformer,
        plot_imu=args.plot_imu,
        interactive_plot=args.interactive_plot,
        plot_projections=args.plot_projections,
        use_regressed_covariance=not args.fixed_covariance,
        meas_cov_scale=args.meas_cov_scale,
        meas_cov_axis_scale=tuple(args.meas_cov_axis_scale),
        use_fej=bool(args.use_fej and not args.disable_fej),
        use_block_update=args.block_update,
        enable_chi2_gating=not args.disable_chi2_gating,
        covariance_repair_mode=args.covariance_repair_mode,
        imu_noise_model=args.imu_noise_model,
        imu_interval_mode=args.imu_interval_mode,
        covariance_propagation_mode=args.covariance_propagation_mode,
        nominal_integration_method=args.nominal_integration_method,
        update_solve_method=args.update_solve_method,
        gating_mode=args.gating_mode,
        fej_scope=args.fej_scope,
        edge_robust_mode=args.edge_robust_mode,
        edge_inflation_factor=args.edge_inflation_factor,
        edge_chi2_multiplier=args.edge_chi2_multiplier,
        **dataset_params
    )
    # Execute EKF processing
    results = run_filter(active_config)
    #Print summary of results
    print_filter_run_summary(
        dataset=results["dataset"],
        sequence=results["sequence"],
        num_anchors=results["num_anchors"],
        num_updates_attempted=results["num_updates_attempted"],
        num_updates_rejected=results["num_updates_rejected"],
        diagnostics=results["diagnostics"],
        mean_residual_norm=results["mean_residual_norm"],
        mean_delta_norm=results["mean_delta_norm"],
        saved_trajectory_path=results["saved_file"],
    )

    #Plots (if requested)
    if active_config.interactive_plot:
        if results["regressed"] is not None:
            show_interactive_3d_plot(
                estimated_trajectory=results["trajectory"],
                ground_truth_trajectory=results["ground_truth"],
                regressed_trajectory=results["regressed"],
                imu_trajectory=results.get("imu_only"),
            )
        else:
            show_interactive_3d_plot(
                estimated_trajectory=results["trajectory"],
                ground_truth_trajectory=results["ground_truth"],
                imu_trajectory=results.get("imu_only"),
            )


if __name__ == "__main__":
    main()
