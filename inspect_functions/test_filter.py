"""Run a GT-driven smoke test for the TLEIO filter on raw EDS IMU data.

This inspection script is meant to answer one concrete question: does the
current filter implementation behave sensibly when we feed it realistic IMU
propagation data and synthetic relative-pose measurements built from ground
truth. By default the synthetic measurements are scheduled every 75 ms so that
each triplet spans 150 ms, matching the intended three-voxel timing semantics.
The script:
1. loads raw IMU, ground-truth poses, and frame timestamps;
2. builds realistic measurement timestamps and interpolates GT poses there;
3. converts consecutive GT frame poses into the transformer's `2 x 7` relative
   measurements over triplets;
4. perturbs those measurements with configurable translation/rotation noise;
5. runs the clone-based EKF update and reports trajectory errors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from filter.imu_buffer import ImuMeasurement
from filter.measurement_triplet import make_default_joint_covariance
from filter.scekf import ImuMSCKF


def load_imu_table(path: Path) -> np.ndarray:
    """Load the raw IMU csv and validate the expected `timestamp gx gy gz ax ay az` layout."""

    imu = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if imu.shape[1] != 7:
        raise ValueError(
            f"{path} has {imu.shape[1]} columns, expected 7: timestamp gx gy gz ax ay az."
        )
    imu = imu[np.argsort(imu[:, 0])]
    return imu


def load_pose_table(path: Path) -> np.ndarray:
    """Load the stamped GT pose table and validate the expected 8-column layout."""

    poses = np.loadtxt(path, comments="#", ndmin=2)
    if poses.shape[1] != 8:
        raise ValueError(
            f"{path} has {poses.shape[1]} columns, expected 8: timestamp px py pz qx qy qz qw."
        )
    return poses


def load_frame_times(path: Path) -> np.ndarray:
    """Load image timestamps for the optional image-aligned measurement mode."""

    frame_times = np.loadtxt(path, ndmin=1)
    if frame_times.ndim != 1:
        frame_times = frame_times.reshape(-1)
    if frame_times.size < 3:
        raise ValueError(f"{path} needs at least three frame timestamps to build triplet updates.")
    return np.sort(frame_times.astype(np.float64))


def build_measurement_anchor_times(start_time_s: float, end_time_s: float, measurement_dt_s: float) -> np.ndarray:
    """Build evenly spaced synthetic measurement timestamps across the valid interval."""

    if measurement_dt_s <= 0.0:
        raise ValueError("Measurement spacing must be strictly positive.")
    if end_time_s <= start_time_s:
        return np.array([], dtype=np.float64)

    count = int(np.floor((end_time_s - start_time_s) / measurement_dt_s)) + 1
    times = start_time_s + measurement_dt_s * np.arange(count, dtype=np.float64)
    times = times[times <= end_time_s + 1e-12]
    return times


def infer_time_scale_to_seconds(timestamps: np.ndarray) -> float:
    """Infer whether timestamps are already in seconds, microseconds, or nanoseconds."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    positive_diffs = np.diff(timestamps)
    positive_diffs = positive_diffs[positive_diffs > 0]
    median_dt = float(np.median(positive_diffs)) if len(positive_diffs) > 0 else 0.0

    if median_dt > 1e5:
        return 1e-9
    if median_dt > 1e1:
        return 1e-6
    return 1.0


def normalize_quaternions(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Normalize a batch of xyzw quaternions and fail loudly on zero-norm inputs."""

    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
    norms = np.linalg.norm(quaternions_xyzw, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Found a near-zero quaternion while normalizing GT poses.")
    return quaternions_xyzw / norms


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two xyzw quaternions with sign-corrected spherical interpolation."""

    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(dot)
    theta = alpha * theta_0
    s0 = np.sin(theta_0 - theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


def interpolate_poses(gt_times_s: np.ndarray, gt_positions: np.ndarray, gt_quaternions: np.ndarray, query_times_s: np.ndarray):
    """Interpolate GT positions linearly and orientations with quaternion slerp."""

    right = np.searchsorted(gt_times_s, query_times_s, side="left")
    right = np.clip(right, 1, len(gt_times_s) - 1)
    left = right - 1

    t0 = gt_times_s[left]
    t1 = gt_times_s[right]
    alpha = (query_times_s - t0) / np.maximum(t1 - t0, 1e-12)
    alpha = np.clip(alpha, 0.0, 1.0)

    p0 = gt_positions[left]
    p1 = gt_positions[right]
    positions = (1.0 - alpha[:, None]) * p0 + alpha[:, None] * p1

    q0 = gt_quaternions[left]
    q1 = gt_quaternions[right]
    quaternions = np.stack([slerp(a, b, w) for a, b, w in zip(q0, q1, alpha)], axis=0)
    return positions, quaternions


def compute_relative_pose(position_i: np.ndarray, rotation_i: np.ndarray, position_j: np.ndarray, rotation_j: np.ndarray) -> np.ndarray:
    """Convert two world-frame poses into one `7D` body-frame relative pose `t_ij, q_ij`."""

    translation = rotation_i.T @ (position_j - position_i)
    relative_rotation = rotation_i.T @ rotation_j
    quaternion = Rotation.from_matrix(relative_rotation).as_quat()
    return np.concatenate([translation, quaternion], axis=0)


def build_triplet_measurement(positions: np.ndarray, quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Build the transformer's stacked `2 x 7` measurement from three GT poses."""

    rotations = Rotation.from_quat(quaternions_xyzw).as_matrix()
    rel_12 = compute_relative_pose(positions[0], rotations[0], positions[1], rotations[1])
    rel_23 = compute_relative_pose(positions[1], rotations[1], positions[2], rotations[2])
    return np.stack([rel_12, rel_23], axis=0)


def perturb_triplet_measurement(clean_measurement_2x7: np.ndarray, rng: np.random.Generator, sigma_translation: float, sigma_rotation_rad: float) -> np.ndarray:
    """Add Gaussian translation noise and small-angle rotation noise to a `2 x 7` triplet."""

    noisy = np.asarray(clean_measurement_2x7, dtype=np.float64).copy()
    for edge_idx in range(2):
        noisy[edge_idx, :3] += rng.normal(scale=sigma_translation, size=3)

        base_rotation = Rotation.from_quat(noisy[edge_idx, 3:7])
        perturbation = Rotation.from_rotvec(rng.normal(scale=sigma_rotation_rad, size=3))
        noisy[edge_idx, 3:7] = (perturbation * base_rotation).as_quat()

    return noisy


def build_segment_measurements(raw_times_s: np.ndarray, raw_gyro: np.ndarray, raw_accel: np.ndarray, start_time_s: float, end_time_s: float) -> list[ImuMeasurement]:
    """Resample the raw IMU stream onto one exact propagation segment ending at `end_time_s`."""

    if end_time_s <= start_time_s:
        return []

    interior_mask = (raw_times_s > start_time_s) & (raw_times_s < end_time_s)
    segment_times = list(raw_times_s[interior_mask])
    segment_times.append(float(end_time_s))

    gyro_interp = np.column_stack(
        [np.interp(segment_times, raw_times_s, raw_gyro[:, axis]) for axis in range(3)]
    )
    accel_interp = np.column_stack(
        [np.interp(segment_times, raw_times_s, raw_accel[:, axis]) for axis in range(3)]
    )

    measurements = []
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


def rotation_error_deg(reference_quaternion_xyzw: np.ndarray, estimate_quaternion_xyzw: np.ndarray) -> float:
    """Compute the geodesic angle between two xyzw quaternions in degrees."""

    q_ref = reference_quaternion_xyzw / np.linalg.norm(reference_quaternion_xyzw)
    q_est = estimate_quaternion_xyzw / np.linalg.norm(estimate_quaternion_xyzw)
    dot = np.clip(abs(np.dot(q_ref, q_est)), -1.0, 1.0)
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def make_filter_args(sigma_rel_t: float, sigma_rel_r_rad: float) -> SimpleNamespace:
    """Create the small argument namespace needed by the current EKF implementation."""

    return SimpleNamespace(
        sigma_na=0.01,
        sigma_ng=0.001,
        sigma_nba=1e-4,
        sigma_nbg=1e-5,
        sigma_rel_t=sigma_rel_t,
        sigma_rel_r=sigma_rel_r_rad,
        meas_cov_scale=1.0,
    )


def build_trajectory_table(times_s: np.ndarray, positions: np.ndarray, quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Pack one trajectory into a text-friendly table with time, position, and quaternion columns."""

    return np.column_stack([times_s, positions, quaternions_xyzw])


def save_trajectory_table(path: Path, trajectory_table: np.ndarray) -> Path:
    """Save one trajectory table with a human-readable header for later inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "timestamp_s px py pz qx qy qz qw"
    np.savetxt(path, trajectory_table, fmt="%.9f", header=header, comments="")
    return path


def save_trajectory_comparison_plot(path: Path, times_s: np.ndarray, gt_positions: np.ndarray, estimated_positions: np.ndarray) -> Path:
    """Save a compact trajectory plot comparing the EKF estimate against GT."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    t_rel = times_s - times_s[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(gt_positions[:, 0], gt_positions[:, 1], label="GT")
    axes[0, 0].plot(estimated_positions[:, 0], estimated_positions[:, 1], label="EKF")
    axes[0, 0].set_title("XY trajectory")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].grid(True)
    axes[0, 0].axis("equal")
    axes[0, 0].legend()

    for axis_idx, label in enumerate(("x", "y", "z")):
        row = 0 if axis_idx < 2 else 1
        col = 1 if axis_idx < 2 else 0
        axis = axes[row, col]
        axis.plot(t_rel, gt_positions[:, axis_idx], label=f"GT {label}")
        axis.plot(t_rel, estimated_positions[:, axis_idx], label=f"EKF {label}")
        axis.set_xlabel("time [s]")
        axis.set_ylabel(f"{label} [m]")
        axis.grid(True)
        axis.legend()

    position_error = np.linalg.norm(estimated_positions - gt_positions, axis=1)
    axes[1, 1].plot(t_rel, position_error, color="tab:red")
    axes[1, 1].set_title("Position error")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("||p_est - p_gt|| [m]")
    axes[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def test_filter(
    imu_path: Path = ROOT / "data/eds/imu.csv",
    gt_path: Path = ROOT / "data/eds/stamped_groundtruth.txt",
    frame_timestamps_path: Path | None = ROOT / "data/eds/images_timestamps.txt",
    measurement_dt_ms: float = 75.0,
    use_frame_timestamps: bool = False,
    sigma_rel_t: float = 0.03,
    sigma_rel_r_deg: float = 2.0,
    assumed_sigma_rel_t: float | None = None,
    assumed_sigma_rel_r_deg: float | None = None,
    zero_measurement_noise: bool = False,
    seed: int = 7,
    max_frames: int | None = None,
    output_dir: Path | None = ROOT / "inspect_functions" / "outputs" / "test_filter",
) -> dict:
    """Run a GT-driven noisy-measurement filter test and return trajectory/error diagnostics.

    The returned dictionary contains two trajectory views:
    1. anchor-time states at the measurement timestamps, which are useful for
       judging update behavior;
    2. a dense propagated trajectory sampled at each IMU step plus each
       measurement time, which is what gets written to disk by default.

    The summary metrics are still computed on the anchor-time states. By
    default the function also writes the dense estimated trajectory, dense GT
    trajectory, and a comparison plot to `inspect_functions/outputs/test_filter`.

    The synthetic measurement corruption and the EKF's assumed covariance are
    configurable separately. This is useful when we want to test whether the
    filter is under-trusting or over-trusting the relative-pose update. By
    default the measurement anchors are synthetic timestamps separated by
    `measurement_dt_ms`, so each triplet spans `2 * measurement_dt_ms`.
    """

    imu_table = load_imu_table(Path(imu_path))
    gt_table = load_pose_table(Path(gt_path))

    imu_times_s = imu_table[:, 0].astype(np.float64) * infer_time_scale_to_seconds(imu_table[:, 0])
    gt_times_s = gt_table[:, 0].astype(np.float64)

    overlap_start = max(imu_times_s[0], gt_times_s[0])
    overlap_end = min(imu_times_s[-1], gt_times_s[-1])

    measurement_dt_s = float(measurement_dt_ms) * 1e-3
    if use_frame_timestamps:
        if frame_timestamps_path is None:
            raise ValueError("`frame_timestamps_path` is required when `use_frame_timestamps=True`.")
        frame_times = load_frame_times(Path(frame_timestamps_path))
        frame_times_s = frame_times.astype(np.float64) * infer_time_scale_to_seconds(frame_times)
        overlap_start = max(overlap_start, frame_times_s[0])
        overlap_end = min(overlap_end, frame_times_s[-1])
        measurement_times_s = frame_times_s[
            (frame_times_s >= overlap_start) & (frame_times_s <= overlap_end)
        ]
    else:
        measurement_times_s = build_measurement_anchor_times(
            overlap_start,
            overlap_end,
            measurement_dt_s,
        )

    if max_frames is not None:
        measurement_times_s = measurement_times_s[:max_frames]
    if measurement_times_s.size < 3:
        raise ValueError("Need at least three overlapping frame times to test the triplet update.")

    gt_positions = gt_table[:, 1:4].astype(np.float64)
    gt_quaternions = normalize_quaternions(gt_table[:, 4:8].astype(np.float64))
    anchor_positions, anchor_quaternions = interpolate_poses(
        gt_times_s,
        gt_positions,
        gt_quaternions,
        measurement_times_s,
    )

    measurement_sigma_rel_t = 0.0 if zero_measurement_noise else float(sigma_rel_t)
    measurement_sigma_rel_r_deg = 0.0 if zero_measurement_noise else float(sigma_rel_r_deg)
    measurement_sigma_rel_r_rad = float(np.deg2rad(measurement_sigma_rel_r_deg))

    if assumed_sigma_rel_t is None:
        assumed_sigma_rel_t = float(sigma_rel_t)
    else:
        assumed_sigma_rel_t = float(assumed_sigma_rel_t)

    if assumed_sigma_rel_r_deg is None:
        assumed_sigma_rel_r_deg = float(sigma_rel_r_deg)
    else:
        assumed_sigma_rel_r_deg = float(assumed_sigma_rel_r_deg)

    assumed_sigma_rel_r_rad = float(np.deg2rad(assumed_sigma_rel_r_deg))
    rng = np.random.default_rng(seed)

    filter_args = make_filter_args(assumed_sigma_rel_t, assumed_sigma_rel_r_rad)
    ekf = ImuMSCKF(filter_args)

    initial_rotation = Rotation.from_quat(anchor_quaternions[0]).as_matrix()
    initial_position = anchor_positions[0]
    initial_velocity = (anchor_positions[1] - anchor_positions[0]) / max(
        measurement_times_s[1] - measurement_times_s[0], 1e-9
    )
    ekf.initialize_with_state(
        measurement_times_s[0],
        initial_rotation,
        initial_velocity,
        initial_position,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    ekf.augment_clone()

    raw_gyro = imu_table[:, 1:4].astype(np.float64)
    raw_accel = imu_table[:, 4:7].astype(np.float64)
    joint_covariance = make_default_joint_covariance(assumed_sigma_rel_t, assumed_sigma_rel_r_rad)

    estimated_positions = [ekf.state.p.copy()]
    estimated_quaternions = [Rotation.from_matrix(ekf.state.R).as_quat()]
    dense_times_s = [measurement_times_s[0]]
    dense_estimated_positions = [ekf.state.p.copy()]
    dense_estimated_quaternions = [Rotation.from_matrix(ekf.state.R).as_quat()]
    propagated_position_errors = [0.0]
    corrected_position_errors = [0.0]
    propagated_rotation_errors = [0.0]
    corrected_rotation_errors = [0.0]
    residual_norms = []
    delta_norms = []

    current_time_s = float(measurement_times_s[0])
    for frame_idx in range(1, measurement_times_s.size):
        target_time_s = float(measurement_times_s[frame_idx])
        imu_segment = build_segment_measurements(
            imu_times_s,
            raw_gyro,
            raw_accel,
            current_time_s,
            target_time_s,
        )
        for measurement_idx, measurement in enumerate(imu_segment):
            ekf.propagate([measurement])
            current_time_s = measurement.timestamp
            if measurement_idx < len(imu_segment) - 1:
                dense_times_s.append(current_time_s)
                dense_estimated_positions.append(ekf.state.p.copy())
                dense_estimated_quaternions.append(Rotation.from_matrix(ekf.state.R).as_quat())

        gt_quaternion = anchor_quaternions[frame_idx]
        propagated_position_errors.append(
            float(np.linalg.norm(ekf.state.p - anchor_positions[frame_idx]))
        )
        propagated_rotation_errors.append(
            rotation_error_deg(gt_quaternion, Rotation.from_matrix(ekf.state.R).as_quat())
        )

        ekf.augment_clone()

        if ekf.state.get_clone_count() == 3:
            clean_measurement = build_triplet_measurement(
                anchor_positions[frame_idx - 2 : frame_idx + 1],
                anchor_quaternions[frame_idx - 2 : frame_idx + 1],
            )
            noisy_measurement = perturb_triplet_measurement(
                clean_measurement,
                rng,
                measurement_sigma_rel_t,
                measurement_sigma_rel_r_rad,
            )
            update_info = ekf.update(
                {
                    "relative_pose": noisy_measurement,
                    "joint_covariance": joint_covariance,
                }
            )
            residual_norms.append(float(np.linalg.norm(update_info["residual"])))
            delta_norms.append(float(np.linalg.norm(update_info["delta_x"])))
            ekf.marginalize_oldest_clone()

        corrected_position_errors.append(
            float(np.linalg.norm(ekf.state.p - anchor_positions[frame_idx]))
        )
        corrected_rotation_errors.append(
            rotation_error_deg(gt_quaternion, Rotation.from_matrix(ekf.state.R).as_quat())
        )
        estimated_positions.append(ekf.state.p.copy())
        estimated_quaternions.append(Rotation.from_matrix(ekf.state.R).as_quat())
        dense_times_s.append(target_time_s)
        dense_estimated_positions.append(ekf.state.p.copy())
        dense_estimated_quaternions.append(Rotation.from_matrix(ekf.state.R).as_quat())

    estimated_positions = np.asarray(estimated_positions)
    estimated_quaternions = np.asarray(estimated_quaternions)
    dense_times_s = np.asarray(dense_times_s)
    dense_estimated_positions = np.asarray(dense_estimated_positions)
    dense_estimated_quaternions = np.asarray(dense_estimated_quaternions)
    corrected_position_errors = np.asarray(corrected_position_errors)
    corrected_rotation_errors = np.asarray(corrected_rotation_errors)
    propagated_position_errors = np.asarray(propagated_position_errors)
    propagated_rotation_errors = np.asarray(propagated_rotation_errors)

    dense_gt_positions, dense_gt_quaternions = interpolate_poses(
        gt_times_s,
        gt_positions,
        gt_quaternions,
        dense_times_s,
    )

    anchor_estimated_trajectory = build_trajectory_table(
        measurement_times_s,
        estimated_positions,
        estimated_quaternions,
    )
    anchor_ground_truth_trajectory = build_trajectory_table(
        measurement_times_s,
        anchor_positions,
        anchor_quaternions,
    )
    estimated_trajectory = build_trajectory_table(
        dense_times_s,
        dense_estimated_positions,
        dense_estimated_quaternions,
    )
    ground_truth_trajectory = build_trajectory_table(
        dense_times_s,
        dense_gt_positions,
        dense_gt_quaternions,
    )

    saved_files = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        saved_files["estimated_trajectory"] = save_trajectory_table(
            output_dir / "estimated_trajectory.txt",
            estimated_trajectory,
        )
        saved_files["ground_truth_trajectory"] = save_trajectory_table(
            output_dir / "ground_truth_trajectory.txt",
            ground_truth_trajectory,
        )
        saved_files["anchor_estimated_trajectory"] = save_trajectory_table(
            output_dir / "anchor_estimated_trajectory.txt",
            anchor_estimated_trajectory,
        )
        saved_files["anchor_ground_truth_trajectory"] = save_trajectory_table(
            output_dir / "anchor_ground_truth_trajectory.txt",
            anchor_ground_truth_trajectory,
        )
        saved_files["trajectory_plot"] = save_trajectory_comparison_plot(
            output_dir / "trajectory_comparison.png",
            dense_times_s,
            dense_gt_positions,
            dense_estimated_positions,
        )

    return {
        "times_s": measurement_times_s,
        "dense_times_s": dense_times_s,
        "gt_positions": anchor_positions,
        "gt_quaternions": anchor_quaternions,
        "estimated_positions": estimated_positions,
        "estimated_quaternions": estimated_quaternions,
        "dense_gt_positions": dense_gt_positions,
        "dense_gt_quaternions": dense_gt_quaternions,
        "dense_estimated_positions": dense_estimated_positions,
        "dense_estimated_quaternions": dense_estimated_quaternions,
        "anchor_estimated_trajectory": anchor_estimated_trajectory,
        "anchor_ground_truth_trajectory": anchor_ground_truth_trajectory,
        "estimated_trajectory": estimated_trajectory,
        "ground_truth_trajectory": ground_truth_trajectory,
        "propagated_position_errors": propagated_position_errors,
        "corrected_position_errors": corrected_position_errors,
        "propagated_rotation_errors_deg": propagated_rotation_errors,
        "corrected_rotation_errors_deg": corrected_rotation_errors,
        "propagated_position_rmse_m": float(np.sqrt(np.mean(propagated_position_errors**2))),
        "position_rmse_m": float(np.sqrt(np.mean(corrected_position_errors**2))),
        "propagated_position_final_error_m": float(propagated_position_errors[-1]),
        "position_final_error_m": float(corrected_position_errors[-1]),
        "propagated_rotation_rmse_deg": float(np.sqrt(np.mean(propagated_rotation_errors**2))),
        "rotation_rmse_deg": float(np.sqrt(np.mean(corrected_rotation_errors**2))),
        "propagated_rotation_final_error_deg": float(propagated_rotation_errors[-1]),
        "rotation_final_error_deg": float(corrected_rotation_errors[-1]),
        "mean_position_improvement_m": float(np.mean(propagated_position_errors - corrected_position_errors)),
        "mean_rotation_improvement_deg": float(np.mean(propagated_rotation_errors - corrected_rotation_errors)),
        "num_frames": int(measurement_times_s.size),
        "num_measurement_times": int(measurement_times_s.size),
        "num_trajectory_samples": int(dense_times_s.size),
        "num_updates": int(len(residual_norms)),
        "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
        "mean_delta_norm": float(np.mean(delta_norms)) if delta_norms else 0.0,
        "seed": int(seed),
        "sigma_rel_t": float(sigma_rel_t),
        "sigma_rel_r_deg": float(sigma_rel_r_deg),
        "measurement_dt_ms": float(measurement_dt_ms),
        "use_frame_timestamps": bool(use_frame_timestamps),
        "measurement_sigma_rel_t": measurement_sigma_rel_t,
        "measurement_sigma_rel_r_deg": measurement_sigma_rel_r_deg,
        "assumed_sigma_rel_t": assumed_sigma_rel_t,
        "assumed_sigma_rel_r_deg": assumed_sigma_rel_r_deg,
        "zero_measurement_noise": bool(zero_measurement_noise),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "saved_files": {key: str(value) for key, value in saved_files.items()},
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the GT-driven filter smoke test."""

    parser = argparse.ArgumentParser(description="Run a noisy GT-driven test of the TLEIO filter.")
    parser.add_argument("--imu", type=Path, default=ROOT / "data/eds/imu.csv")
    parser.add_argument("--gt", type=Path, default=ROOT / "data/eds/stamped_groundtruth.txt")
    parser.add_argument("--frames", type=Path, default=ROOT / "data/eds/images_timestamps.txt")
    parser.add_argument(
        "--measurement_dt_ms",
        type=float,
        default=75.0,
        help="Synthetic measurement-anchor spacing in milliseconds. The default 75 ms makes each triplet span 150 ms.",
    )
    parser.add_argument(
        "--use_frame_timestamps",
        action="store_true",
        help="Use image timestamps as measurement anchors instead of synthetic 75 ms spacing.",
    )
    parser.add_argument("--sigma_rel_t", type=float, default=0.03)
    parser.add_argument("--sigma_rel_r_deg", type=float, default=2.0)
    parser.add_argument(
        "--assumed_sigma_rel_t",
        type=float,
        default=None,
        help="EKF assumed translation sigma [m]. Defaults to --sigma_rel_t.",
    )
    parser.add_argument(
        "--assumed_sigma_rel_r_deg",
        type=float,
        default=None,
        help="EKF assumed rotation sigma [deg]. Defaults to --sigma_rel_r_deg.",
    )
    parser.add_argument(
        "--zero_measurement_noise",
        action="store_true",
        help="Use perfect GT-derived relative-pose measurements while keeping the EKF covariance configurable.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "inspect_functions" / "outputs" / "test_filter",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI wrapper and print a compact summary of the filter test results."""

    args = parse_args()
    results = test_filter(
        imu_path=args.imu,
        gt_path=args.gt,
        frame_timestamps_path=args.frames,
        measurement_dt_ms=args.measurement_dt_ms,
        use_frame_timestamps=args.use_frame_timestamps,
        sigma_rel_t=args.sigma_rel_t,
        sigma_rel_r_deg=args.sigma_rel_r_deg,
        assumed_sigma_rel_t=args.assumed_sigma_rel_t,
        assumed_sigma_rel_r_deg=args.assumed_sigma_rel_r_deg,
        zero_measurement_noise=args.zero_measurement_noise,
        seed=args.seed,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
    )

    print(f"Measurement anchors used:    {results['num_measurement_times']}")
    print(f"Dense trajectory samples:    {results['num_trajectory_samples']}")
    print(f"Triplet updates applied:     {results['num_updates']}")
    print(f"Measurement dt [ms]:         {results['measurement_dt_ms']:.3f}")
    print(f"Using frame timestamps:      {results['use_frame_timestamps']}")
    print(f"Propagated pos RMSE [m]:     {results['propagated_position_rmse_m']:.6f}")
    print(f"Position RMSE [m]:           {results['position_rmse_m']:.6f}")
    print(f"Propagated final pos [m]:    {results['propagated_position_final_error_m']:.6f}")
    print(f"Final position error [m]:    {results['position_final_error_m']:.6f}")
    print(f"Propagated rot RMSE [deg]:   {results['propagated_rotation_rmse_deg']:.6f}")
    print(f"Rotation RMSE [deg]:         {results['rotation_rmse_deg']:.6f}")
    print(f"Propagated final rot [deg]:  {results['propagated_rotation_final_error_deg']:.6f}")
    print(f"Final rotation error [deg]:  {results['rotation_final_error_deg']:.6f}")
    print(f"Mean pos improvement [m]:    {results['mean_position_improvement_m']:.6f}")
    print(f"Mean rot improvement [deg]:  {results['mean_rotation_improvement_deg']:.6f}")
    print(f"Mean residual norm:          {results['mean_residual_norm']:.6f}")
    print(f"Mean correction norm:        {results['mean_delta_norm']:.6f}")
    print(f"Measurement sigma t [m]:     {results['measurement_sigma_rel_t']:.6f}")
    print(f"Measurement sigma r [deg]:   {results['measurement_sigma_rel_r_deg']:.6f}")
    print(f"Assumed EKF sigma t [m]:     {results['assumed_sigma_rel_t']:.6f}")
    print(f"Assumed EKF sigma r [deg]:   {results['assumed_sigma_rel_r_deg']:.6f}")
    print(f"Zero measurement noise:      {results['zero_measurement_noise']}")
    if results["saved_files"]:
        print(f"Estimated trajectory:        {results['saved_files']['estimated_trajectory']}")
        print(f"Ground-truth trajectory:     {results['saved_files']['ground_truth_trajectory']}")
        print(f"Anchor est trajectory:       {results['saved_files']['anchor_estimated_trajectory']}")
        print(f"Anchor GT trajectory:        {results['saved_files']['anchor_ground_truth_trajectory']}")
        print(f"Trajectory plot:             {results['saved_files']['trajectory_plot']}")


if __name__ == "__main__":
    main()
