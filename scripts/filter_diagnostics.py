"""Compute and plot development-time diagnostics for filter trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_trajectory_table(path: Path) -> np.ndarray:
    """Load a text trajectory table with columns `timestamp px py pz qx qy qz qw`."""

    table = np.loadtxt(path, comments="#", ndmin=2)
    if table.shape[1] != 8:
        raise ValueError(
            f"{path} has {table.shape[1]} columns, expected 8: "
            "timestamp px py pz qx qy qz qw."
        )
    return table.astype(np.float64)


def normalize_quaternions(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Normalize xyzw quaternions and fail on zero-norm entries."""

    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
    norms = np.linalg.norm(quaternions_xyzw, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Found a near-zero quaternion while normalizing diagnostics input.")
    return quaternions_xyzw / norms


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two xyzw quaternions with sign-corrected spherical interpolation."""

    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
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


def interpolate_poses(
    gt_times_s: np.ndarray,
    gt_positions: np.ndarray,
    gt_quaternions: np.ndarray,
    query_times_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ground-truth poses onto a target timestamp grid."""

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


def rotation_error_deg(reference_quaternion_xyzw: np.ndarray, estimate_quaternion_xyzw: np.ndarray) -> float:
    """Compute the geodesic angle between two xyzw quaternions in degrees."""

    q_ref = reference_quaternion_xyzw / np.linalg.norm(reference_quaternion_xyzw)
    q_est = estimate_quaternion_xyzw / np.linalg.norm(estimate_quaternion_xyzw)
    dot = np.clip(abs(np.dot(q_ref, q_est)), -1.0, 1.0)
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def save_trajectory_comparison_plot(
    path: Path,
    gt_times_s: np.ndarray,
    gt_positions: np.ndarray,
    estimated_positions: np.ndarray,
) -> Path:
    """Save the standard x/y/z position comparison and total position error plot."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    t_rel = gt_times_s - gt_times_s[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for axis_idx, label in enumerate(("x", "y", "z")):
        row = axis_idx // 2
        col = axis_idx % 2
        axis = axes[row, col]
        axis.plot(t_rel, gt_positions[:, axis_idx], label=f"GT {label}", color="tab:blue")
        axis.plot(t_rel, estimated_positions[:, axis_idx], label=f"EKF {label}", color="tab:green")
        axis.set_title(f"{label.upper()} Position")
        axis.set_xlabel("time [s]")
        axis.set_ylabel(f"{label} [m]")
        axis.grid(True)
        axis.legend()

    position_error = np.linalg.norm(estimated_positions - gt_positions, axis=1)
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
) -> Path:
    """Save the standard roll/pitch/yaw comparison and geodesic rotation error plot."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    t_rel = times_s - times_s[0]
    gt_euler_rad = Rotation.from_quat(gt_quaternions_xyzw).as_euler("xyz", degrees=False)
    est_euler_rad = Rotation.from_quat(estimated_quaternions_xyzw).as_euler("xyz", degrees=False)
    gt_euler_deg = np.rad2deg(np.unwrap(gt_euler_rad, axis=0))
    est_euler_deg = np.rad2deg(np.unwrap(est_euler_rad, axis=0))

    rot_errors_deg = np.array(
        [
            rotation_error_deg(q_gt, q_est)
            for q_gt, q_est in zip(gt_quaternions_xyzw, estimated_quaternions_xyzw)
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    for axis_idx, label in enumerate(labels):
        row = axis_idx // 2
        col = axis_idx % 2
        axis = axes[row, col]
        axis.plot(t_rel, gt_euler_deg[:, axis_idx], label=f"GT {label}", color="tab:blue")
        axis.plot(t_rel, est_euler_deg[:, axis_idx], label=f"EKF {label}", color="tab:green")
        axis.set_title(f"{label} Angle")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("angle [deg]")
        axis.grid(True)
        axis.legend()

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
) -> Path:
    """Save a 3D plot comparing the estimated trajectory against ground truth."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        gt_positions[:, 0],
        gt_positions[:, 1],
        gt_positions[:, 2],
        label="Ground Truth",
        color="tab:blue",
        linewidth=2,
    )
    ax.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        estimated_positions[:, 2],
        label="EKF Estimated",
        color="tab:green",
        linewidth=2,
    )
    ax.scatter(*gt_positions[0], color="black", marker="o", s=60, label="Start", zorder=5)
    ax.scatter(*gt_positions[-1], color="red", marker="x", s=60, label="End", zorder=5)
    ax.set_title("3D Trajectory Comparison")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()

    max_range = np.array(
        [
            gt_positions[:, 0].max() - gt_positions[:, 0].min(),
            gt_positions[:, 1].max() - gt_positions[:, 1].min(),
            gt_positions[:, 2].max() - gt_positions[:, 2].min(),
        ]
    ).max() / 2.0
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


def compute_filter_diagnostics(
    estimated_trajectory: np.ndarray,
    ground_truth_trajectory: np.ndarray,
    output_dir: Path | None = None,
    file_prefix: str = "filter",
) -> dict:
    """Compute development metrics and optionally save the standard plots."""

    est = np.asarray(estimated_trajectory, dtype=np.float64)
    gt = np.asarray(ground_truth_trajectory, dtype=np.float64)
    if est.ndim != 2 or est.shape[1] != 8:
        raise ValueError("Estimated trajectory must have shape [N, 8].")
    if gt.ndim != 2 or gt.shape[1] != 8:
        raise ValueError("Ground-truth trajectory must have shape [N, 8].")

    est_times_s = est[:, 0]
    est_positions = est[:, 1:4]
    est_quaternions = normalize_quaternions(est[:, 4:8])

    gt_times_s = gt[:, 0]
    gt_positions = gt[:, 1:4]
    gt_quaternions = normalize_quaternions(gt[:, 4:8])

    aligned_gt_positions, aligned_gt_quaternions = interpolate_poses(
        gt_times_s,
        gt_positions,
        gt_quaternions,
        est_times_s,
    )

    position_errors = est_positions - aligned_gt_positions
    x_rmse_m = float(np.sqrt(np.mean(position_errors[:, 0] ** 2)))
    y_rmse_m = float(np.sqrt(np.mean(position_errors[:, 1] ** 2)))
    z_rmse_m = float(np.sqrt(np.mean(position_errors[:, 2] ** 2)))
    position_error_norms = np.linalg.norm(position_errors, axis=1)
    position_rmse_m = float(np.sqrt(np.mean(position_error_norms**2)))
    max_position_error_m = float(np.max(position_error_norms))

    gt_euler_rad = Rotation.from_quat(aligned_gt_quaternions).as_euler("xyz", degrees=False)
    est_euler_rad = Rotation.from_quat(est_quaternions).as_euler("xyz", degrees=False)
    gt_euler_deg = np.rad2deg(np.unwrap(gt_euler_rad, axis=0))
    est_euler_deg = np.rad2deg(np.unwrap(est_euler_rad, axis=0))
    euler_errors_deg = est_euler_deg - gt_euler_deg

    roll_rmse_deg = float(np.sqrt(np.mean(euler_errors_deg[:, 0] ** 2)))
    pitch_rmse_deg = float(np.sqrt(np.mean(euler_errors_deg[:, 1] ** 2)))
    yaw_rmse_deg = float(np.sqrt(np.mean(euler_errors_deg[:, 2] ** 2)))

    rotation_errors_deg = np.array(
        [
            rotation_error_deg(q_gt, q_est)
            for q_gt, q_est in zip(aligned_gt_quaternions, est_quaternions)
        ]
    )
    rotation_rmse_deg = float(np.sqrt(np.mean(rotation_errors_deg**2)))
    max_rotation_error_deg = float(np.max(rotation_errors_deg))

    saved_files: dict[str, str] = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        saved_files["trajectory_plot"] = str(
            save_trajectory_comparison_plot(
                output_dir / f"{file_prefix}_trajectory_comparison.png",
                est_times_s,
                aligned_gt_positions,
                est_positions,
            )
        )
        saved_files["rotation_plot"] = str(
            save_rotation_comparison_plot(
                output_dir / f"{file_prefix}_rotation_comparison.png",
                est_times_s,
                aligned_gt_quaternions,
                est_quaternions,
            )
        )
        saved_files["trajectory_3d_plot"] = str(
            save_3d_trajectory_plot(
                output_dir / f"{file_prefix}_trajectory_3d.png",
                aligned_gt_positions,
                est_positions,
            )
        )

    return {
        "x_rmse_m": x_rmse_m,
        "y_rmse_m": y_rmse_m,
        "z_rmse_m": z_rmse_m,
        "max_position_error_m": max_position_error_m,
        "position_rmse_m": position_rmse_m,
        "roll_rmse_deg": roll_rmse_deg,
        "pitch_rmse_deg": pitch_rmse_deg,
        "yaw_rmse_deg": yaw_rmse_deg,
        "max_rotation_error_deg": max_rotation_error_deg,
        "rotation_rmse_deg": rotation_rmse_deg,
        "saved_files": saved_files,
    }


def print_filter_run_summary(
    sequence: str,
    num_anchors: int,
    num_updates_attempted: int,
    num_updates_rejected: int,
    mean_residual_norm: float | None,
    mean_delta_norm: float | None,
    diagnostics: dict,
    saved_trajectory_path: str | None = None,
) -> None:
    """Print the development-time filter summary in one place."""

    print(f"Sequence:                 {sequence}")
    print(f"Anchors processed:        {num_anchors}")
    print(f"Updates attempted:        {num_updates_attempted}")
    print(f"Updates rejected:         {num_updates_rejected}")
    print(f"Mean residual norm:       {mean_residual_norm}")
    print(f"Mean correction norm:     {mean_delta_norm}")
    print(f"X direction RMSE [m]:     {diagnostics['x_rmse_m']:.6f}")
    print(f"Y direction RMSE [m]:     {diagnostics['y_rmse_m']:.6f}")
    print(f"Z direction RMSE [m]:     {diagnostics['z_rmse_m']:.6f}")
    print(f"MAX position error [m]:   {diagnostics['max_position_error_m']:.6f}")
    print(f"Position RMSE [m]:        {diagnostics['position_rmse_m']:.6f}")
    print(f"Roll RMSE [deg]:          {diagnostics['roll_rmse_deg']:.6f}")
    print(f"Pitch RMSE [deg]:         {diagnostics['pitch_rmse_deg']:.6f}")
    print(f"Yaw RMSE [deg]:           {diagnostics['yaw_rmse_deg']:.6f}")
    print(f"MAX rotation error [deg]: {diagnostics['max_rotation_error_deg']:.6f}")
    print(f"Rotation RMSE [deg]:      {diagnostics['rotation_rmse_deg']:.6f}")
    if saved_trajectory_path is not None:
        print(f"Saved trajectory:         {saved_trajectory_path}")
    for key, value in diagnostics["saved_files"].items():
        print(f"{key}: {value}")


def main() -> None:
    """Run the diagnostics from the command line on two trajectory files."""

    parser = argparse.ArgumentParser(description="Compute diagnostics for a filter trajectory.")
    parser.add_argument("--estimated", type=Path, required=True, help="Estimated trajectory txt file.")
    parser.add_argument("--ground_truth", type=Path, required=True, help="Ground-truth trajectory txt file.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Optional directory for plots.")
    parser.add_argument("--prefix", type=str, default="filter", help="Prefix used for saved plot filenames.")
    args = parser.parse_args()

    estimated = load_trajectory_table(args.estimated)
    ground_truth = load_trajectory_table(args.ground_truth)
    results = compute_filter_diagnostics(
        estimated,
        ground_truth,
        output_dir=args.output_dir,
        file_prefix=args.prefix,
    )
    print_filter_run_summary(
        sequence=args.prefix,
        num_anchors=int(len(estimated)),
        num_updates_attempted=max(int(len(estimated)) - 2, 0),
        num_updates_rejected=0,
        mean_residual_norm=None,
        mean_delta_norm=None,
        diagnostics=results,
        saved_trajectory_path=str(args.estimated),
    )


if __name__ == "__main__":
    main()
