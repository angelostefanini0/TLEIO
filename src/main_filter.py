"""Run the EKF directly on processed relative-motion measurements.

1. load one processed sequence (`anchor_poses.txt`, relative-motion file, `imu.csv`);
2. initialize the EKF from the first two anchor poses;
3. propagate the IMU exactly from anchor to anchor;
4. update the EKF with overlapping triplets of relative translations from
   the selected relative-motion file;
5. marginalize a single oldest clone after each attempted update.

The transformer's output is assumed to already be available on disk either as
`relative_motions.txt` or `regressed_relative_motions.txt`, so this is an
asynchronous implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from filter.imu_buffer import ImuMeasurement
from filter.measurement_triplet import make_default_joint_covariance
from filter.scekf import ImuMSCKF


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from filter_diagnostics import compute_filter_diagnostics, print_filter_run_summary


@dataclass(frozen=True)
class RunnerConfig:
    """User-editable configuration for the minimal relative-motion filter runner."""

    # Paths
    processed_root: Path = ROOT / "data" / "eds" / "processed_train"
    sequence: str = "03_rocket_earth_dark"
    out_dir: Path = ROOT / "outputs" / "main_filter"
    use_regressed_relative_motions: bool = False

    # Optional sequence truncation
    max_frames: int | None = None

    # IMU preprocessing
    imu_axis_multipliers: tuple[float, float, float] = (-1.0, -1.0, 1.0)

    # IMU process noise
    sigma_na: float = 5.90e-03
    sigma_ng: float = 9.57e-03
    sigma_nba: float = 8.81e-05
    sigma_nbg: float = 3.99e-05

    # EKF assumed measurement covariance
    assumed_sigma_rel_t: float = 0.03
    assumed_sigma_rel_r_deg: float = 2.0
    meas_cov_scale: float = 1.0

    # Optional extra synthetic noise added on top of the selected relative-motion file
    extra_measurement_noise_t: float = 0.0
    seed: int = 7

    # Initialization offsets applied on top of the first anchor pose/velocity
    initial_velocity_window_size: int = 2
    initial_position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_velocity_offset_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_euler_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_bg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_ba: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity_world_mps2: tuple[float, float, float] = (0.0, 0.0, -9.80665)


CONFIG = RunnerConfig()


def _sequence_path(config: RunnerConfig) -> Path:
    """Resolve the processed sequence directory."""

    sequence_path = config.processed_root / config.sequence
    if not sequence_path.exists():
        raise FileNotFoundError(f"Processed sequence folder does not exist: {sequence_path}")
    return sequence_path


def _load_anchor_poses(sequence_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load processed anchor poses."""

    anchor_path = sequence_path / "anchor_poses.txt"
    anchor_table = np.atleast_2d(np.loadtxt(anchor_path, dtype=np.float64, skiprows=1))
    if anchor_table.shape[1] != 8:
        raise ValueError(
            f"{anchor_path} has {anchor_table.shape[1]} columns, expected 8: "
            "timestamp px py pz qx qy qz qw."
        )

    timestamps_us = anchor_table[:, 0].astype(np.int64)
    positions = anchor_table[:, 1:4].astype(np.float64)
    quaternions = anchor_table[:, 4:8].astype(np.float64)
    return timestamps_us, positions, quaternions


def _load_relative_motion_table(sequence_path: Path, use_regressed_relative_motions: bool) -> np.ndarray:
    """Load processed relative motions and skip any stale non-numeric header lines."""

    rel_filename = (
        "regressed_relative_motions.txt"
        if use_regressed_relative_motions
        else "relative_motions.txt"
    )
    rel_path = sequence_path / rel_filename
    rows: list[list[float]] = []
    with rel_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue

    relative_motions = np.asarray(rows, dtype=np.float64)
    if relative_motions.ndim != 2 or relative_motions.shape[1] not in (8, 9):
        raise ValueError(
            f"{rel_path} has shape {relative_motions.shape}, expected either N x 8 "
            "(t0 t1 px py pz rx ry rz) or N x 9 (t0 t1 px py pz qx qy qz qw)."
        )
    if relative_motions.shape[0] < 2:
        raise ValueError(f"{rel_path} needs at least two rows to form one triplet update.")
    return relative_motions


def _load_sequence_imu(sequence_path: Path) -> np.ndarray:
    """Load one processed IMU table with columns `timestamp gx gy gz ax ay az`."""

    imu_path = sequence_path / "imu.csv"
    imu = np.loadtxt(imu_path, delimiter=",", comments="#", ndmin=2)
    if imu.shape[1] != 7:
        raise ValueError(
            f"{imu_path} has {imu.shape[1]} columns, expected 7: timestamp gx gy gz ax ay az."
        )
    return imu[np.argsort(imu[:, 0])]


def _infer_time_scale_to_seconds(timestamps: np.ndarray) -> float:
    """Infer whether timestamps are in seconds, microseconds, or nanoseconds."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    positive_diffs = np.diff(timestamps)
    positive_diffs = positive_diffs[positive_diffs > 0]
    median_dt = float(np.median(positive_diffs)) if len(positive_diffs) > 0 else 0.0

    if median_dt > 1e5:
        return 1e-9
    if median_dt > 1e1:
        return 1e-6
    return 1.0


def _build_anchor_times_from_relative_motions(relative_motion_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover anchor timestamps and the translation-only measurements used by the EKF.

    The current EKF update uses only the translation component, so both
    `t0 t1 px py pz rx ry rz` and `t0 t1 px py pz qx qy qz qw` layouts are
    supported.
    """

    raw_times = relative_motion_table[:, :2]
    time_scale = _infer_time_scale_to_seconds(raw_times.reshape(-1))
    edge_start_times_s = raw_times[:, 0].astype(np.float64) * time_scale
    edge_end_times_s = raw_times[:, 1].astype(np.float64) * time_scale

    if np.any(edge_end_times_s <= edge_start_times_s):
        raise ValueError("Found a non-positive interval in `relative_motions.txt`.")

    continuity_error = np.max(np.abs(edge_end_times_s[:-1] - edge_start_times_s[1:]))
    if continuity_error > 1e-9:
        raise ValueError(
            "Consecutive relative-motion rows are not time-continuous; "
            f"max discontinuity is {continuity_error:.3e} s."
        )

    anchor_times_s = np.concatenate([edge_start_times_s[:1], edge_end_times_s], axis=0)
    relative_measurements = relative_motion_table[:, 2:5].astype(np.float64)
    return anchor_times_s, relative_measurements


def _validate_anchor_alignment(
    anchor_timestamps_us: np.ndarray,
    relative_anchor_times_s: np.ndarray,
    relative_measurements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure anchor_poses.txt and relative_motions.txt describe the same timeline."""

    anchor_times_s = anchor_timestamps_us.astype(np.float64) * 1e-6
    if len(anchor_times_s) != len(relative_anchor_times_s):
        raise ValueError(
            "anchor_poses.txt and relative_motions.txt disagree on the number of anchors: "
            f"{len(anchor_times_s)} vs {len(relative_anchor_times_s)}."
        )

    max_error = float(np.max(np.abs(anchor_times_s - relative_anchor_times_s)))
    if max_error > 1e-9:
        raise ValueError(
            "anchor_poses.txt and relative_motions.txt disagree on anchor timestamps; "
            f"maximum mismatch is {max_error:.3e} s."
        )

    if relative_measurements.shape[0] != len(anchor_times_s) - 1:
        raise ValueError(
            "Expected one relative-motion row per consecutive anchor pair, got "
            f"{relative_measurements.shape[0]} rows for {len(anchor_times_s)} anchors."
        )

    return anchor_timestamps_us, anchor_times_s, relative_measurements


def _truncate_sequence(
    anchor_timestamps_us: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_quaternions: np.ndarray,
    relative_measurements: np.ndarray,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optionally keep only the first `max_frames` anchors."""

    if max_frames is None:
        return anchor_timestamps_us, anchor_positions, anchor_quaternions, relative_measurements

    if max_frames < 3:
        raise ValueError("`max_frames` must be at least 3 to run triplet updates.")

    anchor_timestamps_us = anchor_timestamps_us[:max_frames]
    anchor_positions = anchor_positions[:max_frames]
    anchor_quaternions = anchor_quaternions[:max_frames]
    relative_measurements = relative_measurements[: max(0, max_frames - 1)]
    return anchor_timestamps_us, anchor_positions, anchor_quaternions, relative_measurements


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

    interior_mask = (raw_times_s > start_time_s) & (raw_times_s < end_time_s)
    segment_times = list(raw_times_s[interior_mask])
    segment_times.append(float(end_time_s))

    gyro_interp = np.column_stack(
        [np.interp(segment_times, raw_times_s, raw_gyro[:, axis]) for axis in range(3)]
    )
    accel_interp = np.column_stack(
        [np.interp(segment_times, raw_times_s, raw_accel[:, axis]) for axis in range(3)]
    )

    measurements: list[ImuMeasurement] = []
    prev_time_s = float(start_time_s)
    for sample_idx, timestamp_s in enumerate(segment_times):
        timestamp_s = float(timestamp_s)
        measurements.append(
            ImuMeasurement(
                timestamp=timestamp_s,
                dt=max(timestamp_s - prev_time_s, 0.0),
                accel=accel_interp[sample_idx].astype(np.float64),
                gyro=gyro_interp[sample_idx].astype(np.float64),
            )
        )
        prev_time_s = timestamp_s

    return measurements


def _build_anchor_imu_segments(
    imu_table: np.ndarray,
    anchor_timestamps_us: np.ndarray,
    axis_multipliers: tuple[float, float, float],
) -> list[list[ImuMeasurement]]:
    """Precompute one exact propagation segment for each consecutive anchor pair."""

    time_scale = _infer_time_scale_to_seconds(imu_table[:, 0])
    raw_times_s = imu_table[:, 0].astype(np.float64) * time_scale
    raw_gyro = imu_table[:, 1:4].astype(np.float64)
    raw_accel = imu_table[:, 4:7].astype(np.float64)
    axis_multipliers_arr = np.asarray(axis_multipliers, dtype=np.float64)
    raw_gyro = raw_gyro * axis_multipliers_arr
    raw_accel = raw_accel * axis_multipliers_arr
    anchor_times_s = anchor_timestamps_us.astype(np.float64) * 1e-6

    if anchor_times_s[0] < raw_times_s[0] or anchor_times_s[-1] > raw_times_s[-1]:
        raise ValueError("Anchor timestamps fall outside the IMU stream.")

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


def _make_filter_args(config: RunnerConfig) -> SimpleNamespace:
    """Create the small args namespace consumed by `ImuMSCKF`."""

    return SimpleNamespace(
        sigma_na=float(config.sigma_na),
        sigma_ng=float(config.sigma_ng),
        sigma_nba=float(config.sigma_nba),
        sigma_nbg=float(config.sigma_nbg),
        sigma_rel_t=float(config.assumed_sigma_rel_t),
        sigma_rel_r=float(np.deg2rad(config.assumed_sigma_rel_r_deg)),
        meas_cov_scale=float(config.meas_cov_scale),
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


def _estimate_initial_velocity(
    anchor_times_s: np.ndarray,
    anchor_positions: np.ndarray,
    window_size: int = 2,
) -> np.ndarray:
    """Estimate the initial velocity from the first `window_size` anchors.

    When enough anchors are available, this uses the displacement from the first
    to the fourth anchor, which corresponds to 200 ms for the default 50 ms
    anchor spacing. If fewer anchors exist, it falls back to the furthest
    available anchor.
    """

    if len(anchor_times_s) < 2:
        return np.zeros(3, dtype=np.float64)

    last_idx = min(window_size - 1, len(anchor_times_s) - 1)
    dt = max(float(anchor_times_s[last_idx] - anchor_times_s[0]), 1e-9)
    return (anchor_positions[last_idx] - anchor_positions[0]).astype(np.float64) / dt


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


def run_filter(config: RunnerConfig = CONFIG) -> dict:
    """Run the relative-motion EKF on one processed sequence."""

    sequence_path = _sequence_path(config)
    anchor_timestamps_us, anchor_positions, anchor_quaternions = _load_anchor_poses(sequence_path)
    relative_motion_table = _load_relative_motion_table(
        sequence_path,
        config.use_regressed_relative_motions,
    )
    imu_table = _load_sequence_imu(sequence_path)

    relative_anchor_times_s, relative_measurements = _build_anchor_times_from_relative_motions(
        relative_motion_table
    )
    anchor_timestamps_us, _, relative_measurements = _validate_anchor_alignment(
        anchor_timestamps_us,
        relative_anchor_times_s,
        relative_measurements,
    )
    anchor_timestamps_us, anchor_positions, anchor_quaternions, relative_measurements = _truncate_sequence(
        anchor_timestamps_us,
        anchor_positions,
        anchor_quaternions,
        relative_measurements,
        config.max_frames,
    )

    if len(anchor_timestamps_us) < 3:
        raise ValueError("Need at least three anchors to run the triplet EKF update.")

    anchor_imu_segments = _build_anchor_imu_segments(
        imu_table,
        anchor_timestamps_us,
        config.imu_axis_multipliers,
    )

    anchor_times_s = anchor_timestamps_us.astype(np.float64) * 1e-6
    p0 = anchor_positions[0].astype(np.float64)
    R0 = Rotation.from_quat(anchor_quaternions[0]).as_matrix()
    v0 = _estimate_initial_velocity(
        anchor_times_s,
        anchor_positions,
        window_size=config.initial_velocity_window_size,
    )
    R0, v0, p0 = _apply_initial_offsets(R0, v0.astype(np.float64), p0, config)
    bg0 = np.asarray(config.initial_bg, dtype=np.float64)
    ba0 = np.asarray(config.initial_ba, dtype=np.float64)

    ekf = ImuMSCKF(_make_filter_args(config))
    ekf.g = np.asarray(config.gravity_world_mps2, dtype=np.float64)
    ekf.initialize_with_state(anchor_times_s[0], R0, v0, p0, bg0, ba0)

    rng = np.random.default_rng(config.seed)
    joint_covariance = make_default_joint_covariance(float(config.assumed_sigma_rel_t))

    trajectory_rows = [_state_to_row(anchor_times_s[0], ekf.state)]
    residual_norms: list[float] = []
    delta_norms: list[float] = []
    rejected_updates = 0

    ekf.augment_clone()
    ekf.propagate(anchor_imu_segments[0])
    trajectory_rows.append(_state_to_row(anchor_times_s[1], ekf.state))
    ekf.augment_clone()

    for anchor_idx in range(2, len(anchor_times_s)):
        ekf.propagate(anchor_imu_segments[anchor_idx - 1])
        ekf.augment_clone()

        measurement = relative_measurements[anchor_idx - 2 : anchor_idx].copy()
        if config.extra_measurement_noise_t > 0.0:
            measurement += rng.normal(
                scale=float(config.extra_measurement_noise_t),
                size=measurement.shape,
            )

        update_info = ekf.update(
            {
                "relative_pose": measurement,
                "joint_covariance": joint_covariance,
            }
        )
        if update_info.get("rejected", False):
            rejected_updates += 1
        else:
            residual_norms.append(float(np.linalg.norm(update_info["residual"])))
            delta_norms.append(float(np.linalg.norm(update_info["delta_x"])))

        trajectory_rows.append(_state_to_row(anchor_times_s[anchor_idx], ekf.state))
        ekf.marginalize_oldest_clone()

    sequence_out_dir = config.out_dir / config.sequence
    trajectory_table = np.asarray(trajectory_rows, dtype=np.float64)
    saved_path = _save_trajectory(
        sequence_out_dir / f"{config.sequence}_trajectory.txt",
        trajectory_table,
    )
    ground_truth_trajectory = _build_ground_truth_trajectory(
        anchor_timestamps_us,
        anchor_positions,
        anchor_quaternions,
    )
    diagnostics = compute_filter_diagnostics(
        trajectory_table,
        ground_truth_trajectory,
        output_dir=sequence_out_dir,
        file_prefix=config.sequence,
    )

    return {
        "sequence": config.sequence,
        "num_anchors": int(len(anchor_times_s)),
        "num_updates_attempted": int(len(anchor_times_s) - 2),
        "num_updates_rejected": int(rejected_updates),
        "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else None,
        "mean_delta_norm": float(np.mean(delta_norms)) if delta_norms else None,
        "trajectory": trajectory_table,
        "saved_file": str(saved_path),
        "diagnostics": diagnostics,
    }


def main() -> None:
    """Run the minimal relative-motion filter with the top-of-file configuration."""

    results = run_filter(CONFIG)
    print_filter_run_summary(
        sequence=results["sequence"],
        num_anchors=results["num_anchors"],
        num_updates_attempted=results["num_updates_attempted"],
        num_updates_rejected=results["num_updates_rejected"],
        mean_residual_norm=results["mean_residual_norm"],
        mean_delta_norm=results["mean_delta_norm"],
        diagnostics=results["diagnostics"],
        saved_trajectory_path=results["saved_file"],
    )


if __name__ == "__main__":
    main()
