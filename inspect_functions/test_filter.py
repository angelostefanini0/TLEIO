"""Run a processed-sequence smoke test for the TLEIO filter.

This inspection script is meant to answer one concrete question: does the
current filter implementation behave sensibly when we feed it realistic IMU
propagation data together with the same `2 x 7` relative-pose targets that the
transformer is trained to predict. In the processed-data mode used by default,
the script:
1. loads zero-based IMU data from a processed sequence;
2. loads precomputed adjacent-anchor relative poses from `relative_motions.txt`;
3. slides a triplet window over those rows so each EKF update consumes two
   consecutive `7D` measurements, exactly like theon/rotation noise;
5. uses stamped GT only for initialization, evaluation, and network output;
4. perturbs those measurements with configurable translati plotting.
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


def load_relative_motion_table(path: Path) -> np.ndarray:
    """Load processed relative motions and validate the expected 9-column layout.

    The processed files may contain a stale text header, so this parser skips
    non-numeric lines and keeps only the numeric `t0 t1 px py pz qx qy qz qw`
    rows that match the transformer's per-edge output contract.
    """

    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue

    relative_motions = np.asarray(rows, dtype=np.float64)
    if relative_motions.ndim != 2 or relative_motions.shape[1] != 9:
        raise ValueError(
            f"{path} has shape {relative_motions.shape}, expected N x 9: "
            "t0 t1 px py pz qx qy qz qw."
        )
    if relative_motions.shape[0] < 2:
        raise ValueError(f"{path} needs at least two relative-motion rows to form one EKF update.")
    return relative_motions


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


def build_anchor_times_from_relative_motions(relative_motion_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover anchor times and `7D` per-edge measurements from processed relative motions.

    Each row stores one adjacent-anchor relative pose. Consecutive rows must be
    continuous so the EKF can consume overlapping pairs `(k, k+1)` as one
    stacked triplet update.
    """

    relative_motion_table = np.asarray(relative_motion_table, dtype=np.float64)
    raw_times = relative_motion_table[:, :2]
    time_scale = infer_time_scale_to_seconds(raw_times.reshape(-1))
    edge_start_times_s = raw_times[:, 0] * time_scale
    edge_end_times_s = raw_times[:, 1] * time_scale

    if np.any(edge_end_times_s <= edge_start_times_s):
        raise ValueError("Found a non-positive relative-motion interval in relative_motions.txt.")

    continuity_error = np.max(np.abs(edge_end_times_s[:-1] - edge_start_times_s[1:]))
    if continuity_error > 1e-9:
        raise ValueError(
            "Consecutive rows in relative_motions.txt are not time-continuous, "
            f"maximum discontinuity is {continuity_error:.3e} s."
        )

    anchor_times_s = np.concatenate([edge_start_times_s[:1], edge_end_times_s], axis=0)
    relative_measurements = relative_motion_table[:, 2:5].copy()
    # relative_measurements[:, 3:7] = normalize_quaternions(relative_measurements[:, 3:7])
    # relative_measurements = normalize_quaternions(relative_motion_table[:, 2:9])
    return anchor_times_s, relative_measurements


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
    """Convert two world-frame poses into one `3D` body-frame relative pose `t_ij, q_ij`."""

    translation = rotation_i.T @ (position_j - position_i)
    return translation


def build_triplet_measurement(positions: np.ndarray, quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Build the transformer's stacked `2 x 7` measurement from three GT poses."""

    rotations = Rotation.from_quat(quaternions_xyzw).as_matrix()
    rel_12 = compute_relative_pose(positions[0], rotations[0], positions[1], rotations[1])
    rel_23 = compute_relative_pose(positions[1], rotations[1], positions[2], rotations[2])
    return np.stack([rel_12, rel_23], axis=0)


def perturb_triplet_measurement(clean_measurement_2x3: np.ndarray, rng: np.random.Generator, sigma_translation: float) -> np.ndarray:
    """Add Gaussian translation noise and small-angle rotation noise to a `2 x 7` triplet."""

    noisy = np.asarray(clean_measurement_2x3, dtype=np.float64).copy()
    for edge_idx in range(2):
        noisy[edge_idx, :3] += rng.normal(scale=sigma_translation, size=3)
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
        sigma_na=0.01,       #TUNE!
        sigma_ng=0.0001,      #TUNE! 
        sigma_nba=5e-3,     #TUNE! 
        sigma_nbg=5e-3,     #TUNE! 
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


def resample_positions(query_times_s: np.ndarray, source_times_s: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    """Interpolate a position trajectory onto a target timestamp grid."""

    query_times_s = np.asarray(query_times_s, dtype=np.float64)
    source_times_s = np.asarray(source_times_s, dtype=np.float64)
    source_positions = np.asarray(source_positions, dtype=np.float64)

    if source_times_s.ndim != 1 or source_positions.ndim != 2 or source_positions.shape[1] != 3:
        raise ValueError("Position resampling expects `times: [N]` and `positions: [N, 3]`.")

    return np.column_stack(
        [np.interp(query_times_s, source_times_s, source_positions[:, axis]) for axis in range(3)]
    )


def save_trajectory_comparison_plot(
    path: Path,
    gt_times_s: np.ndarray,
    gt_positions: np.ndarray,
    estimated_positions_at_gt_times: np.ndarray,
    noisy_gt_positions: np.ndarray | None = None,
) -> Path:
    """Save four time-series plots on the processed GT timestamp grid."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    t_rel = gt_times_s - gt_times_s[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for axis_idx, label in enumerate(("x", "y", "z")):
        row = axis_idx // 2
        col = axis_idx % 2
        axis = axes[row, col]
        axis.plot(t_rel, gt_positions[:, axis_idx], label=f"GT {label}", color="tab:blue")
        
        # Plot della GT rumorosa
        if noisy_gt_positions is not None:
            axis.plot(t_rel, noisy_gt_positions[:, axis_idx], label=f"Noisy GT {label}", color="tab:orange", alpha=0.5, linestyle="--")
            
        axis.plot(t_rel, estimated_positions_at_gt_times[:, axis_idx], label=f"EKF {label}", color="tab:green")
        axis.set_title(f"{label.upper()} Position")
        axis.set_xlabel("time [s]")
        axis.set_ylabel(f"{label} [m]")
        axis.grid(True)
        axis.legend()

    position_error = np.linalg.norm(estimated_positions_at_gt_times - gt_positions, axis=1)
    
    # Plot dell'errore della GT rumorosa
    if noisy_gt_positions is not None:
        noisy_error = np.linalg.norm(noisy_gt_positions - gt_positions, axis=1)
        axes[1, 1].plot(t_rel, noisy_error, color="tab:orange", alpha=0.5, linestyle="--", label="Noisy GT Error")
        
    axes[1, 1].plot(t_rel, position_error, color="tab:red", label="EKF Error")
    axes[1, 1].set_title("Total Position Error")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("||p_est - p_gt|| [m]")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def save_rotation_comparison_plot(
    path: Path,
    times_s: np.ndarray,
    gt_quaternions_xyzw: np.ndarray,
    estimated_quaternions_xyzw: np.ndarray,
    noisy_gt_quaternions_xyzw: np.ndarray | None = None,
) -> Path:
    """Save four time-series plots for Roll, Pitch, Yaw and absolute rotation error."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    t_rel = times_s - times_s[0]
    
    # Convert to Euler angles (rad) and unwrap to avoid jumps at +/- 180 deg
    gt_euler_rad = Rotation.from_quat(gt_quaternions_xyzw).as_euler('xyz', degrees=False)
    est_euler_rad = Rotation.from_quat(estimated_quaternions_xyzw).as_euler('xyz', degrees=False)
    
    gt_euler_deg = np.rad2deg(np.unwrap(gt_euler_rad, axis=0))
    est_euler_deg = np.rad2deg(np.unwrap(est_euler_rad, axis=0))

    if noisy_gt_quaternions_xyzw is not None:
        noisy_euler_rad = Rotation.from_quat(noisy_gt_quaternions_xyzw).as_euler('xyz', degrees=False)
        noisy_euler_deg = np.rad2deg(np.unwrap(noisy_euler_rad, axis=0))

    # Compute absolute rotation error point-by-point
    rot_errors_deg = np.array([
        rotation_error_deg(q_gt, q_est) 
        for q_gt, q_est in zip(gt_quaternions_xyzw, estimated_quaternions_xyzw)
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    for axis_idx, label in enumerate(labels):
        row = axis_idx // 2
        col = axis_idx % 2
        axis = axes[row, col]
        axis.plot(t_rel, gt_euler_deg[:, axis_idx], label=f"GT {label}", color="tab:blue")
        
        if noisy_gt_quaternions_xyzw is not None:
            axis.plot(t_rel, noisy_euler_deg[:, axis_idx], label=f"Noisy GT {label}", color="tab:orange", alpha=0.5, linestyle="--")
            
        axis.plot(t_rel, est_euler_deg[:, axis_idx], label=f"EKF {label}", color="tab:green")
        axis.set_title(f"{label} Angle")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("angle [deg]")
        axis.grid(True)
        axis.legend()

    if noisy_gt_quaternions_xyzw is not None:
        noisy_rot_errors_deg = np.array([
            rotation_error_deg(q_gt, q_noise) 
            for q_gt, q_noise in zip(gt_quaternions_xyzw, noisy_gt_quaternions_xyzw)
        ])
        axes[1, 1].plot(t_rel, noisy_rot_errors_deg, color="tab:orange", alpha=0.5, linestyle="--", label="Noisy GT Error")

    axes[1, 1].plot(t_rel, rot_errors_deg, color="tab:red", label="EKF Error")
    axes[1, 1].set_title("Absolute Rotation Error")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("Geodesic Error [deg]")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def save_3d_trajectory_plot(
    path: Path,
    gt_positions: np.ndarray,
    estimated_positions: np.ndarray,
    noisy_gt_positions: np.ndarray | None = None,
) -> Path:
    """Save a 3D plot comparing the estimated trajectory against the ground truth."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot GT
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
            label='Ground Truth', color='tab:blue', linewidth=2)
    
    # Plot Noisy GT (se disponibile) visualizzato come punti semitrasparenti
    if noisy_gt_positions is not None:
        ax.scatter(noisy_gt_positions[:, 0], noisy_gt_positions[:, 1], noisy_gt_positions[:, 2], 
                   label='Noisy GT (Misurazioni)', color='tab:orange', alpha=0.3, s=5)
        
    # Plot Traiettoria Stimata (EKF)
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], 
            label='EKF Estimated', color='tab:green', linewidth=2)

    # Evidenzia il punto di Inizio e Fine
    ax.scatter(*gt_positions[0], color='black', marker='o', s=60, label='Start', zorder=5)
    ax.scatter(*gt_positions[-1], color='red', marker='x', s=60, label='End', zorder=5)

    ax.set_title("3D Trajectory Comparison")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()

    # Rendi gli assi in scala 1:1 in modo che la traiettoria non sia deformata
    max_range = np.array([
        gt_positions[:, 0].max() - gt_positions[:, 0].min(),
        gt_positions[:, 1].max() - gt_positions[:, 1].min(),
        gt_positions[:, 2].max() - gt_positions[:, 2].min()
    ]).max() / 2.0

    mid_x = (gt_positions[:, 0].max() + gt_positions[:, 0].min()) * 0.5
    mid_y = (gt_positions[:, 1].max() + gt_positions[:, 1].min()) * 0.5
    mid_z = (gt_positions[:, 2].max() + gt_positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def fix_quaternion_signs(quaternions_xyzw: np.ndarray) -> np.ndarray:
    q = quaternions_xyzw.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i-1]) < 0:
            q[i] = -q[i]
    return q

import numpy as np

def estimate_roll_pitch_from_gravity(accel_vector: np.ndarray) -> tuple[float, float]:
    """
    Estimates Roll and Pitch [rad].
    """
    ax, ay, az = accel_vector[0], accel_vector[1], accel_vector[2]
    
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    
    return roll, pitch

def test_filter(
    imu_path: Path = ROOT / "data/eds/processed/00_peanuts_dark/imu.csv",
    gt_path: Path = ROOT / "data/eds/processed/00_peanuts_dark/stamped_groundtruth.txt",
    relative_motions_path: Path | None = ROOT / "data/eds/processed/00_peanuts_dark/v1_predicted_relative_motions.txt",
    frame_timestamps_path: Path | None = ROOT / "data/eds/images_timestamps.txt",
    measurement_dt_ms: float = 75.0,
    use_frame_timestamps: bool = False,
    sigma_rel_t: float = 0.1,
    sigma_rel_r_deg: float = 3.0,
    assumed_sigma_rel_t: float | None = None,
    assumed_sigma_rel_r_deg: float | None = None,
    zero_measurement_noise: bool = False,
    seed: int = 7,
    max_frames: int | None = None,
    output_dir: Path | None = ROOT / "inspect_functions" / "outputs" / "test_filter",
) -> dict:
    """Run a noisy relative-pose filter test and return trajectory/error diagnostics.

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
    filter is under-trusting or over-trusting the relative-pose update.

    By default the function uses the processed `relative_motions.txt` file,
    which already contains the transformer's per-edge `7D` outputs between
    adjacent anchor timestamps. In that mode, each EKF update consumes two
    consecutive rows, exactly matching the intended `2 x 7` filter input.
    """

    imu_table = load_imu_table(Path(imu_path))
    gt_table = load_pose_table(Path(gt_path))
    relative_motion_table = (
        load_relative_motion_table(Path(relative_motions_path))
        if relative_motions_path is not None
        else None
    )

    imu_times_s = imu_table[:, 0].astype(np.float64) * infer_time_scale_to_seconds(imu_table[:, 0])
    gt_times_s = gt_table[:, 0].astype(np.float64) * infer_time_scale_to_seconds(gt_table[:, 0])

    overlap_start = max(imu_times_s[0], gt_times_s[0])
    overlap_end = min(imu_times_s[-1], gt_times_s[-1])

    if relative_motion_table is not None:
        measurement_times_s, relative_measurements = build_anchor_times_from_relative_motions(
            relative_motion_table
        )
        valid_mask = (
            (measurement_times_s >= overlap_start - 1e-9)
            & (measurement_times_s <= overlap_end + 1e-9)
        )
        if not np.all(valid_mask):
            if not np.any(valid_mask):
                raise ValueError("The processed anchor times do not overlap with both IMU and ground truth.")
            first_valid = int(np.argmax(valid_mask))
            last_valid = int(len(valid_mask) - np.argmax(valid_mask[::-1]))
            measurement_times_s = measurement_times_s[first_valid:last_valid]
            relative_measurements = relative_measurements[first_valid:last_valid - 1]
        measurement_times_s = np.asarray(measurement_times_s, dtype=np.float64)
        relative_measurements = np.asarray(relative_measurements, dtype=np.float64)
        if max_frames is not None:
            measurement_times_s = measurement_times_s[:max_frames]
            relative_measurements = relative_measurements[: max(0, measurement_times_s.size - 1)]
    else:
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
        relative_measurements = None

    if use_frame_timestamps and relative_motion_table is not None:
        raise ValueError(
            "`use_frame_timestamps` is only supported in the legacy GT-derived mode. "
            "Leave it disabled when using precomputed relative_motions.txt."
        )

    if max_frames is not None and relative_motion_table is None:
        measurement_times_s = measurement_times_s[:max_frames]
    if measurement_times_s.size < 3:
        raise ValueError("Need at least three overlapping frame times to test the triplet update.")
    if relative_measurements is not None and relative_measurements.shape[0] != measurement_times_s.size - 1:
        raise ValueError(
            "The processed relative-motions table does not match the inferred anchor times."
        )
    gt_positions = gt_table[:, 1:4].astype(np.float64)
    gt_quaternions = normalize_quaternions(gt_table[:, 4:8].astype(np.float64))
    # gt_quaternions_xyzw = gt_table[:, [5, 6, 7, 4]].astype(np.float64)
    # gt_quaternions = normalize_quaternions(gt_quaternions_xyzw)
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

    raw_gyro = imu_table[:, 1:4].astype(np.float64)
    raw_accel = imu_table[:, 4:7].astype(np.float64)
    calib_array = np.array([-1.0, -1.0, 1.0])  #180 deg rotation on Z to align gyro axes

    raw_gyro = raw_gyro * calib_array
    raw_accel = raw_accel * calib_array

    initial_rotation = Rotation.from_quat(anchor_quaternions[0]).as_matrix()
    initial_position = anchor_positions[0]
    initial_velocity = (anchor_positions[1] - anchor_positions[0]) / max(
        measurement_times_s[1] - measurement_times_s[0], 1e-9
    )


    # anchor_poses_path = Path(imu_path).parent / "anchor_poses.txt"
    # anchor_poses_table = np.loadtxt(anchor_poses_path, comments="#",skiprows=1, ndmin=2)
    # ap_times_s = anchor_poses_table[:, 0] * 1e-6
    # ap_quaternions = normalize_quaternions(anchor_poses_table[:, 4:8].astype(np.float64))
    # # ap_quaternions = normalize_quaternions(ap_quaternions_xyzw)
    # ap_quaternions = fix_quaternion_signs(ap_quaternions)

    # from filter.utils.math_utils import mat_log
    # bias_estimates = []
    # for i in range(len(ap_times_s) - 1):
    #     dt = ap_times_s[i+1] - ap_times_s[i]
    #     if dt < 1e-6:
    #         continue
    #     R_i = Rotation.from_quat(ap_quaternions[i]).as_matrix()
    #     R_j = Rotation.from_quat(ap_quaternions[i+1]).as_matrix()
    #     omega_true = mat_log(R_i.T @ R_j) / dt
    #     mask = (imu_times_s >= ap_times_s[i]) & (imu_times_s < ap_times_s[i+1])
    #     if mask.sum() == 0:
    #         continue
    #     gyro_mean = np.mean(raw_gyro[mask], axis=0)
    #     bias_estimates.append(gyro_mean - omega_true)

    # initial_bg = np.mean(bias_estimates, axis=0) if bias_estimates else np.zeros(3)
    initial_bg = np.zeros(3)
    initial_ba = np.zeros(3)
    accel_mean = np.mean(raw_accel[:50], axis=0)
    roll_init, pitch_init = estimate_roll_pitch_from_gravity(accel_mean)
    print(f"\n[INIT] IMU Estimate       -> Roll: {np.rad2deg(roll_init):.2f}°, Pitch: {np.rad2deg(pitch_init):.2f}°")
    gt_euler_deg = Rotation.from_matrix(initial_rotation).as_euler('xyz', degrees=True)
    print(f"[INIT] Ground Truth  -> Roll: {gt_euler_deg[0]:.2f}°, Pitch: {gt_euler_deg[1]:.2f}°")
    g_world_estimated = initial_rotation @ (-accel_mean)
    g_world_normalized = g_world_estimated / np.linalg.norm(g_world_estimated) * 9.80665
    ekf.g = g_world_normalized
    accel_world = initial_rotation @ accel_mean
    gravity_world = -ekf.g  # gravity = -g_vector
    accel_bias_world = accel_world - gravity_world
    initial_ba = initial_rotation.T @ accel_bias_world

    ekf.initialize_with_state(
        measurement_times_s[0],
        initial_rotation,
        initial_velocity,
        initial_position,
        initial_bg,
        initial_ba,
    )
    
    ekf.augment_clone()

    joint_covariance = make_default_joint_covariance(assumed_sigma_rel_t)


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
        # if frame_idx <= 3:
        #     P = ekf.state.P
        #     (f"\n[frame {frame_idx}] Diagonal Covariance:")
        #     print(f"  rot  [0:3]:  {np.diag(P)[0:3]}")
        #     print(f"  vel  [3:6]:  {np.diag(P)[3:6]}")
        #     print(f"print  pos  [6:9]:  {np.diag(P)[6:9]}")
        #     print(f"  bg   [9:12]: {np.diag(P)[9:12]}")
        #     print(f"  ba  [12:15]: {np.diag(P)[12:15]}")
        #     if P.shape[0] > 15:
        #         print(f"  clone1_rot [15:18]: {np.diag(P)[15:18]}")
        #         print(f"  clone1_pos [18:21]: {np.diag(P)[18:21]}")
        #     print(f"  P min={P.min():.2e}  P max={P.max():.2e}")
        #     print(f"  P sym: {np.allclose(P, P.T, atol=1e-10)}")
        #     print(f"  P pos def: {np.all(np.linalg.eigvalsh(P) > 0)}")

        if ekf.state.get_clone_count() == 3:
            if relative_measurements is None:
                clean_measurement = build_triplet_measurement(
                    anchor_positions[frame_idx - 2 : frame_idx + 1],
                    anchor_quaternions[frame_idx - 2 : frame_idx + 1],
                )
            else:
                clean_measurement = relative_measurements[frame_idx - 2 : frame_idx]
            noisy_measurement = perturb_triplet_measurement(
                clean_measurement,
                rng,
                measurement_sigma_rel_t,
            )
            update_info = ekf.update(
                {
                    "relative_pose": noisy_measurement,
                    "joint_covariance": joint_covariance,
                }
            )
            # if frame_idx <= 6:
            #     P_post = ekf.state.P
            #     print(f"\n[frame {frame_idx}] POST-UPDATE diagonal:")
            #     print(f"  rot  [0:3]:  {np.diag(P_post)[0:3]}")
            #     print(f"  bg   [9:12]: {np.diag(P_post)[9:12]}")
            #     print(f"  P def pos: {np.all(np.linalg.eigvalsh(P_post) > 0)}")
            residual_norms.append(float(np.linalg.norm(update_info["residual"])))
            delta_norms.append(float(np.linalg.norm(update_info["delta_x"])))
            ekf.marginalize_oldest_clone()
            ekf.marginalize_oldest_clone() #Added a marginalization to protect the independence assumption and avoid drift

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
    plot_mask = (gt_times_s >= dense_times_s[0] - 1e-9) & (gt_times_s <= dense_times_s[-1] + 1e-9)
    plot_gt_times_s = gt_times_s[plot_mask]
    plot_gt_positions = gt_positions[plot_mask]
    plot_estimated_positions = resample_positions(
        plot_gt_times_s,
        dense_times_s,
        dense_estimated_positions,
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
    plot_noisy_gt_positions = plot_gt_positions + rng.normal(
        scale=measurement_sigma_rel_t, size=plot_gt_positions.shape
    )

    noise_axes = rng.normal(size=(dense_gt_quaternions.shape[0], 3))
    noise_axes_norms = np.linalg.norm(noise_axes, axis=1, keepdims=True)
    noise_axes = np.divide(noise_axes, noise_axes_norms, out=np.zeros_like(noise_axes), where=noise_axes_norms!=0)
    
    noise_angles = rng.normal(scale=measurement_sigma_rel_r_rad, size=(dense_gt_quaternions.shape[0], 1))
    noise_quats = np.concatenate([noise_axes * np.sin(noise_angles / 2), np.cos(noise_angles / 2)], axis=1)
    
    dense_noisy_gt_quaternions = (Rotation.from_quat(noise_quats) * Rotation.from_quat(dense_gt_quaternions)).as_quat()

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
            plot_gt_times_s,
            plot_gt_positions,
            plot_estimated_positions,
            noisy_gt_positions=plot_noisy_gt_positions, 
        )
        saved_files["rotation_plot"] = save_rotation_comparison_plot(
            output_dir / "rotation_comparison.png",
            dense_times_s,
            dense_gt_quaternions,
            dense_estimated_quaternions,
            noisy_gt_quaternions_xyzw=dense_noisy_gt_quaternions,
        )
        saved_files["trajectory_3d_plot"] = save_3d_trajectory_plot(
            output_dir / "trajectory_3d.png",
            plot_gt_positions,
            plot_estimated_positions,
            noisy_gt_positions=plot_noisy_gt_positions,
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
        "plot_gt_times_s": plot_gt_times_s,
        "plot_gt_positions": plot_gt_positions,
        "plot_estimated_positions": plot_estimated_positions,
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
        "measurement_dt_ms": float(
            np.median(np.diff(measurement_times_s)) * 1e3 if measurement_times_s.size > 1 else measurement_dt_ms
        ),
        "use_frame_timestamps": bool(use_frame_timestamps),
        "using_precomputed_relative_motions": bool(relative_motion_table is not None),
        "measurement_sigma_rel_t": measurement_sigma_rel_t,
        "measurement_sigma_rel_r_deg": measurement_sigma_rel_r_deg,
        "assumed_sigma_rel_t": assumed_sigma_rel_t,
        "assumed_sigma_rel_r_deg": assumed_sigma_rel_r_deg,
        "zero_measurement_noise": bool(zero_measurement_noise),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "saved_files": {key: str(value) for key, value in saved_files.items()},
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the processed-sequence filter smoke test."""

    parser = argparse.ArgumentParser(description="Run a noisy processed-sequence test of the TLEIO filter.")
    parser.add_argument("--imu", type=Path, default=ROOT / "data/eds/processed/00_peanuts_dark/imu.csv")
    parser.add_argument(
        "--gt",
        type=Path,
        default=ROOT / "data/eds/processed/00_peanuts_dark/stamped_groundtruth.txt",
    )
    parser.add_argument(
        "--relative_motions",
        type=Path,
        default=ROOT / "data/eds/processed/00_peanuts_dark/v1_predicted_relative_motions.txt",
        help="Processed adjacent-anchor relative poses used to build overlapping `2 x 7` EKF updates.",
    )
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
        relative_motions_path=args.relative_motions,
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
    print(f"Using processed rel poses:   {results['using_precomputed_relative_motions']}")
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
        print(f"Rotation plot:               {results['saved_files']['rotation_plot']}")
        print(f"3D Trajectory plot:          {results['saved_files']['trajectory_3d_plot']}")

if __name__ == "__main__":
    main()