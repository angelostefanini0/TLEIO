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
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from filter.imu_buffer import ImuMeasurement
from filter.measurement import make_default_joint_covariance
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
    dataset: str = "testv7"
    sequence: str = "competition_Test_MH001"

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
    plot_ate: bool = False 

    # # IMU preprocessing
    imu_axis_multipliers: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # IMU process noise
    sigma_na: float = 0.0031594227678764424
    sigma_ng: float = 0.00022161895597298104
    sigma_nba: float = 4.5072574649258535e-05
    sigma_nbg: float = 7.374373866121663e-07

    # EKF assumed measurement covariance
    assumed_sigma_rel_t: float = 0.02194332115673975 # for fixed covariance ablation
    meas_cov_scale: float = 0.33099642292388415

    # Initialization offsets applied on top of the first anchor pose/velocity
    initial_position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_velocity_offset_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_euler_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_bg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_ba: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity_world_mps2: tuple[float, float, float] = (0.0, 0.0, 9.80665)
    network_scale: float = 0.5816711433024562
    initial_attitude_sigma_deg: float = 0.3387476210194474
    initial_velocity_sigma_mps: float = 0.1461417023558029
    initial_position_sigma_m: float = 0.004099644013174264
    initial_z_sigma_m: float = 0.007975250416043874
    initial_bg_sigma_rps: float = 0.0007382676285995384
    initial_ba_sigma_mps2: float = 0.025430606254182857

# Global instance of default configuration
CONFIG = RunnerConfig()


def _sequence_path(config: RunnerConfig) -> Path:
    """Resolve the processed sequence directory."""

    sequence_path = config.processed_root / config.sequence
    if not sequence_path.exists():
        raise FileNotFoundError(f"Processed sequence folder does not exist: {sequence_path}")
    return sequence_path


def load_anchor_poses(sequence_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load processed anchor poses from the text file."""

    anchor_path = sequence_path / "anchor_poses.txt"
    # Skip the header and ensure it is a 2D array
    anchor_table = np.atleast_2d(np.loadtxt(anchor_path, dtype=np.float64, skiprows=1))
    if anchor_table.shape[1] != 8:
        raise ValueError(
            f"{anchor_path} has {anchor_table.shape[1]} columns, expected 8: "
            "timestamp px py pz qx qy qz qw."
        )
    # Extract timestamps, positions and quaternions 
    timestamps_us = anchor_table[:, 0].astype(np.int64)
    positions = anchor_table[:, 1:4].astype(np.float64)
    quaternions = anchor_table[:, 4:8].astype(np.float64)
    return timestamps_us, positions, quaternions


def load_relative_motion_table(sequence_path: Path, use_gt: bool) -> np.ndarray:
    """Load processed relative motions and skip any stale non-numeric header lines."""
    # Chooses file based on configuration
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
    # Allows both 5-col (translation only) and 9-col (translation + rotation) formats
    if relative_motions.ndim != 2 or relative_motions.shape[1] < 5:
        raise ValueError(
            f"{rel_path} has shape {relative_motions.shape}, expected N x 5 or N x 9: "
            "t0 t1 px py pz [qx qy qz qw]."
        )
    if relative_motions.shape[0] < 6:
        raise ValueError(f"{rel_path} needs at least six rows to form one clip update.")
    return relative_motions


def load_sequence_imu(sequence_path: Path) -> np.ndarray:
    """Load one processed IMU table with columns `timestamp gx gy gz ax ay az`."""

    imu_path = sequence_path / "imu.csv"
    imu = np.loadtxt(imu_path, delimiter=",", comments="#", ndmin=2)
    if imu.shape[1] != 7:
        raise ValueError(
            f"{imu_path} has {imu.shape[1]} columns, expected 7: timestamp gx gy gz ax ay az."
        )
    # Sort data by timestamp to ensure causality
    return imu[np.argsort(imu[:, 0])]


def infer_time_scale_to_seconds(timestamps: np.ndarray) -> float:
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


def build_anchor_times_from_relative_motions(relative_motion_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover anchor timestamps and the translation-only measurements used by the EKF."""

    raw_times = relative_motion_table[:, :2]
    time_scale = infer_time_scale_to_seconds(raw_times.reshape(-1))
    # Convert timestamps in seconds
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
    if relative_motion_table.shape[1] >= 8:
        relative_sigmas = relative_motion_table[:, 5:8].astype(np.float64)
    else:
        relative_sigmas = None
        
    return anchor_times_s, relative_measurements, relative_sigmas


def validate_anchor_alignment(
    anchor_timestamps_us: np.ndarray,
    relative_anchor_times_s: np.ndarray,
    relative_measurements: np.ndarray,
    relative_sigmas: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def build_exact_imu_segment(
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
    # Mask to take only IMU readings between the start and end of the interval
    interior_mask = (raw_times_s > start_time_s) & (raw_times_s < end_time_s)
    segment_times = list(raw_times_s[interior_mask])
    segment_times.append(float(end_time_s))
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
        measurements.append(
            ImuMeasurement(timestamp=timestamp_s,dt=max(timestamp_s - prev_time_s, 0.0),accel=accel_interp[sample_idx].astype(np.float64),gyro=gyro_interp[sample_idx].astype(np.float64))
        )
        prev_time_s = timestamp_s

    return measurements


def build_anchor_imu_segments(
    imu_table: np.ndarray,
    anchor_timestamps_us: np.ndarray,
    axis_multipliers: tuple[float, float, float],
) -> list[list[ImuMeasurement]]:
    """Precompute one exact propagation segment for each consecutive anchor pair."""

    time_scale = infer_time_scale_to_seconds(imu_table[:, 0])
    raw_times_s = imu_table[:, 0].astype(np.float64) * time_scale
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
            build_exact_imu_segment(raw_times_s,raw_gyro,raw_accel,anchor_times_s[idx],anchor_times_s[idx + 1])
        )
    return segments


def make_filter_args(config: RunnerConfig) -> SimpleNamespace:
    """Create the args namespace consumed by `ImuMSCKF`."""

    return SimpleNamespace(
        sigma_na=float(config.sigma_na),
        sigma_ng=float(config.sigma_ng),
        sigma_nba=float(config.sigma_nba),
        sigma_nbg=float(config.sigma_nbg),
        sigma_rel_t=float(config.assumed_sigma_rel_t),
        meas_cov_scale=float(config.meas_cov_scale),
        network_scale=float(config.network_scale),
        initial_attitude_sigma_rad=float(np.deg2rad(config.initial_attitude_sigma_deg)),
        initial_velocity_sigma_mps=float(config.initial_velocity_sigma_mps),
        initial_position_sigma_m=float(config.initial_position_sigma_m),
        initial_z_sigma_m=float(config.initial_z_sigma_m),
        initial_bg_sigma_rps=float(config.initial_bg_sigma_rps),
        initial_ba_sigma_mps2=float(config.initial_ba_sigma_mps2),
    )


def apply_initial_offsets(
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


def state_to_row(timestamp_s: float, ekf_state) -> np.ndarray:
    """Convert one EKF state into a text-friendly row."""

    quaternion_xyzw = Rotation.from_matrix(ekf_state.R).as_quat()
    return np.concatenate(
        [ np.array([float(timestamp_s)], dtype=np.float64),
            ekf_state.p.astype(np.float64),
            quaternion_xyzw.astype(np.float64), ]
    )


def save_trajectory(path: Path, trajectory_table: np.ndarray) -> Path:
    """Save one trajectory table with timestamp, position, and quaternion."""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "timestamp_s px py pz qx qy qz qw"
    np.savetxt(path, trajectory_table, fmt="%.9f", header=header, comments="")
    return path


def build_ground_truth_trajectory(
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

def compute_transformer_trajectory(
    sequence_path: Path,
    anchor_timestamps_us: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_quaternions: np.ndarray,
) -> np.ndarray | None:
    """
    Computes the trajectory based solely on the Transformer's
    predictions, without any EKF filtering. 
    Useful for visualizing the raw network performance in the 3D plot.
    """
    try:
        # Reload the Transformer's predictions 
        regressed_table = load_relative_motion_table(sequence_path, use_gt=False)
        
        reg_dp = regressed_table[:, 2:5]
                
        # Ensure the lengths match
        limit = min(len(reg_dp), len(anchor_timestamps_us) - 1)
        
        if limit > 0:
            regr_positions = [anchor_positions[0].astype(np.float64)]
            regr_quaternions = [anchor_quaternions[0].astype(np.float64)]
            
            # Integrate the poses 
            for i in range(limit):
                R_curr = Rotation.from_quat(regr_quaternions[-1])
                
                # Translate into the global frame
                p_next = regr_positions[-1] + R_curr.as_matrix() @ reg_dp[i]
                regr_positions.append(p_next)

                regr_quaternions.append(anchor_quaternions[i + 1])
                    
            return build_ground_truth_trajectory(anchor_timestamps_us[:limit + 1], np.array(regr_positions),np.array(regr_quaternions))
    except Exception as e:
        print(f"Warning: Failed to generate the regressed trajectory for the plot: {e}")
        
    return None

def transformer_ate(
    ground_truth_trajectory: np.ndarray,
    regressed_trajectory: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute the Transformer ATE using UZH RPG."""
    import tempfile
    import contextlib
    import io
    from trajectory import Trajectory

    with tempfile.TemporaryDirectory(prefix="rpg_align_transformer_") as temp_dir:
        eval_dir = Path(temp_dir)
        eval_gt_path = eval_dir / "stamped_groundtruth.txt"
        eval_est_path = eval_dir / "stamped_traj_estimate.txt"

        # Copy and format Ground Truth timestamps to seconds
        gt_table = ground_truth_trajectory.copy()
        gt_table[:, 0] *= infer_time_scale_to_seconds(gt_table[:, 0])
        np.savetxt(eval_gt_path, gt_table, fmt="%.9f")

        # Copy and format Transformer estimated timestamps to seconds
        est_table = regressed_trajectory.copy()
        est_table[:, 0] *= infer_time_scale_to_seconds(est_table[:, 0])
        np.savetxt(eval_est_path, est_table, fmt="%.9f")

        # Run UZH RPG Trajectory evaluation silently
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            traj = Trajectory(str(eval_dir), est_type="traj_est")
            if not traj.data_loaded:
                raise RuntimeError("Failed to load files into UZH RPG toolbox for alignment.")
            traj.compute_absolute_error()

        # Extract the ATE RMSE
        ate_rmse = float(traj.abs_errors["abs_e_trans_stats"]["rmse"])

        return ate_rmse
    
def run_filter(config: RunnerConfig) -> dict:
    """Run the relative-motion EKF on one processed sequence."""

    # Directory path for the given dataset sequence
    sequence_path = _sequence_path(config)

    # Load foundational data: anchor ground truths and IMU inputs
    anchor_timestamps_us, anchor_positions, anchor_quaternions = load_anchor_poses(sequence_path)
    # Pass `config.use_gt` to determine which table to load
    relative_motion_table = load_relative_motion_table(sequence_path, config.use_gt)
    imu_table = load_sequence_imu(sequence_path)

    # Extract EKF measurement bounds (times & translations) from the relative motion table
    relative_anchor_times_s, relative_measurements, relative_sigmas = build_anchor_times_from_relative_motions(relative_motion_table)

    # Ensure that anchor_poses timeline perfectly matches the relative_motions timeline
    anchor_timestamps_us, _, relative_measurements, relative_sigmas = validate_anchor_alignment(
        anchor_timestamps_us,
        relative_anchor_times_s,
        relative_measurements,
        relative_sigmas
    )

    if len(anchor_timestamps_us) < 5:
        raise ValueError("Need at least five anchors to run the triplet EKF update.")

    # Slice the IMU data stream to exactly match the durations between anchors
    anchor_imu_segments = build_anchor_imu_segments(imu_table,anchor_timestamps_us,config.imu_axis_multipliers)

    # Determine absolute starting states from the first two available ground truth anchors
    anchor_times_s = anchor_timestamps_us.astype(np.float64) * 1e-6
    p0 = anchor_positions[0].astype(np.float64)
    R0 = Rotation.from_quat(anchor_quaternions[0]).as_matrix()
    dt0 = max(anchor_times_s[1] - anchor_times_s[0], 1e-9)
    v0 = (anchor_positions[1] - anchor_positions[0]) / dt0
    R0, v0, p0 = apply_initial_offsets(R0, v0.astype(np.float64), p0, config)
    bg0 = np.asarray(config.initial_bg, dtype=np.float64)
    ba0 = np.asarray(config.initial_ba, dtype=np.float64)
    # Initialize the MSCKF
    ekf = ImuMSCKF(make_filter_args(config))
    ekf.g = np.asarray(config.gravity_world_mps2, dtype=np.float64)
    ekf.initialize_with_state(anchor_times_s[0], R0, v0, p0, bg0, ba0)
    # Setting for IMU plot
    imu_trajectory_rows = []
    if config.plot_imu:
        imu_ekf = ImuMSCKF(make_filter_args(config))
        imu_ekf.g = np.asarray(config.gravity_world_mps2, dtype=np.float64)
        imu_ekf.initialize_with_state(anchor_times_s[0], R0.copy(), v0.copy(), p0.copy(), bg0.copy(), ba0.copy())
        imu_trajectory_rows.append(state_to_row(anchor_times_s[0], imu_ekf.state))
    # Pre-configure joint covariance matrices for measurements
    joint_covariance = make_default_joint_covariance(float(config.assumed_sigma_rel_t))
    # Diagnostics setup
    trajectory_rows = [state_to_row(anchor_times_s[0], ekf.state)]
    residual_norms: list[float] = []
    delta_norms: list[float] = []
    rejected_updates = 0
    # The first state clone (anchor 0) is recorded before propagating
    ekf.augment_clone()
    # Propagate EKF state forward using IMU segment 0 and augment clone (Initialization)
    for anchor_idx in range(1, 4):
        ekf.propagate(anchor_imu_segments[anchor_idx - 1])
        ekf.augment_clone()
        trajectory_rows.append(state_to_row(anchor_times_s[anchor_idx], ekf.state))

        if config.plot_imu:
            imu_ekf.propagate(anchor_imu_segments[anchor_idx - 1])
            imu_trajectory_rows.append(state_to_row(anchor_times_s[anchor_idx], imu_ekf.state))

    # MAIN FILTER LOOP
    # Iterating starting from anchor 4 ensures we have a window available
    for anchor_idx in range(4, len(anchor_times_s)):
        # Prediction Step (IMU Integration)
        ekf.propagate(anchor_imu_segments[anchor_idx - 1])
        ekf.augment_clone()

        if config.plot_imu:
            imu_ekf.propagate(anchor_imu_segments[anchor_idx - 1])
            imu_trajectory_rows.append(state_to_row(anchor_times_s[anchor_idx], imu_ekf.state))

        # Measurement Extraction
        measurement = relative_measurements[anchor_idx - 4 : anchor_idx].copy()
        current_joint_covariance = joint_covariance.copy()
        # If relative sigmas from Transformer are available, inject them 
        # dynamically into the covariance diagonal.
        if relative_sigmas is not None:
            sigmas = relative_sigmas[anchor_idx - 4 : anchor_idx] # shape (2, 3)
            variances = (sigmas.flatten()) ** 2 
            np.fill_diagonal(current_joint_covariance[0:12, 0:12], variances)
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
            residual_norms.append(float(np.linalg.norm(update_info["residual"])))
            delta_norms.append(float(np.linalg.norm(update_info["delta_x"])))
        # Log current state after correction
        trajectory_rows.append(state_to_row(anchor_times_s[anchor_idx], ekf.state))
        # Drop the oldest historical clone from the sliding window
        ekf.marginalize_oldest_clone()

    # Save the EKF's finalized estimate
    trajectory_table = np.asarray(trajectory_rows, dtype=np.float64)
    sequence_out_dir = config.out_dir / config.sequence
    saved_path = save_trajectory(sequence_out_dir / "stamped_traj_estimate.txt",trajectory_table)
    gt_path = sequence_path / "stamped_groundtruth.txt"
    ground_truth_trajectory = np.loadtxt(gt_path, comments="#", ndmin=2)
    ground_truth_trajectory[:, 0] *= infer_time_scale_to_seconds(ground_truth_trajectory[:, 0])

    regressed_trajectory = None
    # Adds a plot for the transformer output before EKF
    if config.plot_transformer:
        regressed_trajectory = compute_transformer_trajectory(sequence_path,anchor_timestamps_us,anchor_positions,anchor_quaternions)

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
        plot_ate=config.plot_ate,
    )


    return {
        "dataset": config.dataset,
        "sequence": config.sequence,
        "num_anchors": int(len(anchor_times_s)),
        "num_updates_attempted": int(len(anchor_times_s) - 4),
        "num_updates_rejected": int(rejected_updates),
        "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else None,
        "mean_delta_norm": float(np.mean(delta_norms)) if delta_norms else None,
        "trajectory": trajectory_table,
        "ground_truth": ground_truth_trajectory,
        "regressed": regressed_trajectory,
        "imu_only": imu_trajectory,
        "saved_file": str(saved_path) if saved_path is not None else None,
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
        "--plot_ate",
        action="store_true",
        help="Plots ATE aligned TLEIO trajectory"
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
        plot_ate=args.plot_ate,
        **dataset_params,
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
        ate_pos = results["diagnostics"].get("ate_positions")
        show_interactive_3d_plot(
            estimated_trajectory=results["trajectory"],
            ground_truth_trajectory=results["ground_truth"],
            regressed_trajectory=results.get("regressed"),
            imu_trajectory=results.get("imu_only"),
            ate_positions=ate_pos,
        )
    


if __name__ == "__main__":
    main()
