"""Compute and plot development-time diagnostics for filter trajectories."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.titlesize': 18,
    'legend.fontsize': 18
})

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
EVAL_SRC = ROOT / "evaluation" / "rpg_trajectory_evaluation" / "src" / "rpg_trajectory_evaluation"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))


def get_rpg_ate_and_aligned_trajectory(
    ground_truth_table: np.ndarray,
    estimated_table: np.ndarray
) -> tuple[float, np.ndarray]:
    """Uses UZH RPG Trajectory Evaluation Toolbox to evaluate ATE and align the trajectory."""
    from trajectory import Trajectory

    with tempfile.TemporaryDirectory(prefix="rpg_eval_", dir=ROOT) as temp_dir:
        eval_dir = Path(temp_dir)
        eval_gt_path = eval_dir / "stamped_groundtruth.txt"
        eval_est_path = eval_dir / "stamped_traj_estimate.txt"

        np.savetxt(eval_gt_path, ground_truth_table, fmt="%.9f")
        np.savetxt(eval_est_path, estimated_table, fmt="%.9f")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            traj = Trajectory(str(eval_dir), est_type="traj_est")
            if not traj.data_loaded:
                raise RuntimeError("RPG trajectory loader failed.")
            
            traj.compute_absolute_error()
            ate_rmse = float(traj.abs_errors["abs_e_trans_stats"]["rmse"])
            p_es_aligned = np.copy(traj.p_es_aligned)
            
    return ate_rmse, p_es_aligned


def load_trajectory_table(path: Path) -> np.ndarray:
    """Load a text trajectory table with columns `timestamp px py pz rx ry rz`."""
    table = np.loadtxt(path, comments="#", ndmin=2)
    if table.shape[1] != 8:
        raise ValueError(f"{path} has {table.shape[1]} columns, expected 8.")
    return table.astype(np.float64)


def normalize_quaternions(quaternions_xyzw: np.ndarray) -> np.ndarray:
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
    norms = np.linalg.norm(quaternions_xyzw, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Found a near-zero quaternion while normalizing diagnostics input.")
    return quaternions_xyzw / norms


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
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
    q_ref = reference_quaternion_xyzw / np.linalg.norm(reference_quaternion_xyzw)
    q_est = estimate_quaternion_xyzw / np.linalg.norm(estimate_quaternion_xyzw)
    dot = np.clip(abs(np.dot(q_ref, q_est)), -1.0, 1.0)
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def save_trajectory_comparison_plot(
    path: Path,
    gt_times_s: np.ndarray,
    gt_positions: np.ndarray,
    estimated_positions: np.ndarray,
    regressed_positions: np.ndarray | None = None,
    imu_positions: np.ndarray | None = None,
    ate_positions: np.ndarray | None = None,
    aa_regressed_trajectory: np.ndarray | None = None,
) -> Path:

    path.parent.mkdir(parents=True, exist_ok=True)
    t_rel = gt_times_s - gt_times_s[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for axis_idx, label in enumerate(("x", "y", "z")):
        row = axis_idx // 2
        col = axis_idx % 2
        axis = axes[row, col]
        axis.plot(t_rel, gt_positions[:, axis_idx], label=f"GT {label}", color="tab:blue")
        if regressed_positions is not None:
            axis.plot(t_rel, regressed_positions[:, axis_idx], label=f"EventsFormer {label}", color="red")
        if imu_positions is not None:
            axis.plot(t_rel, imu_positions[:, axis_idx], label=f"IMU {label}", color="tab:purple", linestyle=":")
        axis.plot(t_rel, estimated_positions[:, axis_idx], label=f"TLEIO {label}", color="tab:green")
        
        if ate_positions is not None:
            min_len = min(len(t_rel), len(ate_positions))
            axis.plot(t_rel[:min_len], ate_positions[:min_len, axis_idx], label=f"TLEIO (RPG ATE) {label}", color="tab:red", linestyle="-.")
        if aa_regressed_trajectory is not None:
            axis.plot(t_rel, aa_regressed_trajectory[:, axis_idx], label=f"EventsFormer ATE Aligned {label}", color="y", linestyle="--")

        axis.set_title(f"{label.upper()} Position")
        axis.set_xlabel("time [s]")
        axis.set_ylabel(f"{label} [m]")
        axis.grid(True)
        axis.legend()

    position_error = np.linalg.norm(estimated_positions - gt_positions, axis=1)
    if regressed_positions is not None:
        regressed_error = np.linalg.norm(regressed_positions - gt_positions, axis=1)
        axes[1, 1].plot(t_rel, regressed_error, color="tab:orange", linestyle="--", label="EventsFormer Error")

    if imu_positions is not None:
        imu_error = np.linalg.norm(imu_positions - gt_positions, axis=1)
        axes[1, 1].plot(t_rel, imu_error, color="tab:purple", linestyle=":", label="IMU Error")
    
    if ate_positions is not None:
        min_len = min(len(gt_positions), len(ate_positions))
        ate_error = np.linalg.norm(ate_positions[:min_len] - gt_positions[:min_len], axis=1)
        axes[1, 1].plot(t_rel[:min_len], ate_error, color="tab:red", linestyle="-.", label="TLEIO (RPG ATE) Error")
        
    axes[1, 1].plot(t_rel, position_error, color="tab:red", label="TLEIO Error")
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
    imu_quaternions_xyzw: np.ndarray | None = None,
) -> Path:

    path.parent.mkdir(parents=True, exist_ok=True)
    t_rel = times_s - times_s[0]
    gt_euler_rad = Rotation.from_quat(gt_quaternions_xyzw).as_euler("xyz", degrees=False)
    est_euler_rad = Rotation.from_quat(estimated_quaternions_xyzw).as_euler("xyz", degrees=False)
    gt_euler_deg = np.rad2deg(np.unwrap(gt_euler_rad, axis=0))
    est_euler_deg = np.rad2deg(np.unwrap(est_euler_rad, axis=0))

    rot_errors_deg = np.array([rotation_error_deg(q_gt, q_est) for q_gt, q_est in zip(gt_quaternions_xyzw, estimated_quaternions_xyzw)])

    if imu_quaternions_xyzw is not None:
        imu_euler_rad = Rotation.from_quat(imu_quaternions_xyzw).as_euler("xyz", degrees=False)
        imu_euler_deg = np.rad2deg(np.unwrap(imu_euler_rad, axis=0))
        imu_rot_errors_deg = np.array([rotation_error_deg(q_gt, q_imu) for q_gt, q_imu in zip(gt_quaternions_xyzw, imu_quaternions_xyzw)])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    for axis_idx, label in enumerate(labels):
        row = axis_idx // 2
        col = axis_idx % 2
        axis = axes[row, col]
        axis.plot(t_rel, gt_euler_deg[:, axis_idx], label=f"GT {label}", color="tab:blue")
        axis.plot(t_rel, est_euler_deg[:, axis_idx], label=f"TLEIO {label}", color="tab:green")
        if imu_quaternions_xyzw is not None:
            axes[axis_idx//2, axis_idx%2].plot(t_rel, imu_euler_deg[:, axis_idx], color="tab:purple", linestyle=":", label=f"IMU {label}")
        axis.set_title(f"{label} Angle")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("angle [deg]")
        axis.grid(True)
        axis.legend()

    axes[1, 1].plot(t_rel, rot_errors_deg, color="tab:red", label="TLEIO Error")
    axes[1, 1].set_title("Absolute Rotation Error")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("Geodesic Error [deg]")
    axes[1, 1].grid(True)
    if imu_quaternions_xyzw is not None:
        axes[1, 1].plot(t_rel, imu_rot_errors_deg, color="tab:purple", linestyle=":", label="IMU Error")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_3d_trajectory_plot(
    path: Path,
    gt_positions: np.ndarray,
    estimated_positions: np.ndarray,
    regressed_positions: np.ndarray | None = None,
    imu_positions: np.ndarray | None = None,
    ate_positions: np.ndarray | None = None,
    aa_regressed_trajectory: np.ndarray | None = None,
) -> Path:

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label="Ground Truth", color="tab:blue", linewidth=2)
    
    if regressed_positions is not None:
        ax.plot(regressed_positions[:, 0], regressed_positions[:, 1], regressed_positions[:, 2], label="EventsFormer", color="red", linewidth=2)

    if imu_positions is not None:
        ax.plot(imu_positions[:, 0], imu_positions[:, 1], imu_positions[:, 2], label="IMU Only", color="tab:purple", linestyle=":", linewidth=2)
    if aa_regressed_trajectory is not None:
        ax.plot(aa_regressed_trajectory[:, 0], aa_regressed_trajectory[:, 1], aa_regressed_trajectory[:, 2], label="EventsFormer ATE Aligned", color="y", linestyle="--", linewidth=2)

    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label="TLEIO", color="tab:green", linewidth=2)
    
    if ate_positions is not None:
        min_len = min(len(gt_positions), len(ate_positions))
        ax.plot(ate_positions[:min_len, 0], ate_positions[:min_len, 1], ate_positions[:min_len, 2], label="TLEIO (RPG ATE Aligned)", color="tab:red", linestyle="-.", linewidth=2)
        
    ax.scatter(*gt_positions[0], color="black", marker="o", s=60, label="Start", zorder=5)
    ax.scatter(*gt_positions[-1], color="red", marker="x", s=60, label="End", zorder=5)
    # ax.set_title("3D Trajectory Comparison")
    # ax.set_xlabel("X [m]")
    # ax.set_ylabel("Y [m]")
    # ax.set_zlabel("Z [m]")
    # ax.legend()
    # Aumentato fontsize a 26 per compensare la dimensione della figura (10x10)
    ax.set_title("3D Trajectory Comparison", fontsize=26, pad=20)
    
    # Mantieni il resto delle label proporzionate
    ax.set_xlabel("X [m]", fontsize=14, labelpad=10)
    ax.set_ylabel("Y [m]", fontsize=14, labelpad=10)
    ax.set_zlabel("Z [m]", fontsize=14, labelpad=10)
    
    # Regola la dimensione dei numeri sugli assi
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc="upper right", fontsize=18)

    max_range = np.array([
        gt_positions[:, 0].max() - gt_positions[:, 0].min(),
        gt_positions[:, 1].max() - gt_positions[:, 1].min(),
        gt_positions[:, 2].max() - gt_positions[:, 2].min(),
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


def save_projections_plot(
    path: Path,
    gt_positions: np.ndarray,
    estimated_positions: np.ndarray,
    regressed_positions: np.ndarray | None = None,
    imu_positions: np.ndarray | None = None,
    ate_positions: np.ndarray | None = None,
    aa_regressed_trajectory: np.ndarray | None = None,
) -> Path:

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    planes = [
        (0, 1, "X", "Y", "XY Projection (Top View)"),
        (0, 2, "X", "Z", "XZ Projection (Front View)"),
        (1, 2, "Y", "Z", "YZ Projection (Side View)"),
    ]

    for ax, (idx1, idx2, label1, label2, title) in zip(axes, planes):
        ax.plot(gt_positions[:, idx1], gt_positions[:, idx2], label="Ground Truth", color="tab:blue")
        
        if regressed_positions is not None:
            ax.plot(regressed_positions[:, idx1], regressed_positions[:, idx2], label="EventsFormer", color="red")

        if imu_positions is not None:
            ax.plot(imu_positions[:, idx1], imu_positions[:, idx2], label="IMU Only", color="tab:purple", linestyle=":", alpha=0.7)
        if aa_regressed_trajectory is not None:
            ax.plot(aa_regressed_trajectory[:, idx1], aa_regressed_trajectory[:, idx2], label="EventsFormer ATE Aligned", color="y", linestyle="--")

        ax.plot(estimated_positions[:, idx1], estimated_positions[:, idx2], label="TLEIO", color="tab:green")
        
        if ate_positions is not None:
            min_len = min(len(gt_positions), len(ate_positions))
            ax.plot(ate_positions[:min_len, idx1], ate_positions[:min_len, idx2], label="TLEIO (RPG ATE)", color="tab:red", linestyle="-.")
            
        ax.scatter(gt_positions[0, idx1], gt_positions[0, idx2], color="black", marker="o", s=40, zorder=5)
        ax.scatter(gt_positions[-1, idx1], gt_positions[-1, idx2], color="red", marker="x", s=40, zorder=5)

        ax.set_title(title)
        ax.set_xlabel(f"{label1} [m]")
        ax.set_ylabel(f"{label2} [m]")
        ax.grid(True)
        ax.legend()
        ax.axis('equal')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def show_interactive_3d_plot(
    estimated_trajectory: np.ndarray,
    ground_truth_trajectory: np.ndarray,
    regressed_trajectory: np.ndarray | None = None,
    imu_trajectory: np.ndarray | None = None,
    ate_positions: np.ndarray | None = None,
    aa_regressed_trajectory=None,
) -> None:

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    gt_pos = ground_truth_trajectory[:, 1:4]
    est_pos = estimated_trajectory[:, 1:4]

    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label="Ground Truth", color="tab:blue", linewidth=2)
    
    if regressed_trajectory is not None:
        reg_pos = regressed_trajectory[:, 1:4]
        ax.plot(reg_pos[:, 0], reg_pos[:, 1], reg_pos[:, 2], label="EventsFormer", color="red", linewidth=2)

    if imu_trajectory is not None:
        imu_pos = imu_trajectory[:, 1:4]
        ax.plot(imu_pos[:, 0], imu_pos[:, 1], imu_pos[:, 2], label="IMU Only", color="tab:purple", linestyle=":", linewidth=2)

    if aa_regressed_trajectory is not None:
        aa_pos = aa_regressed_trajectory[:, 1:4]
        ax.plot(aa_pos[:, 0], aa_pos[:, 1], aa_pos[:, 2], label="EventsFormer ATE Aligned", color="y", linestyle="--", linewidth=2)
        
    ax.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], label="TLEIO Estimated", color="tab:green", linewidth=2)
    if ate_positions is not None:
        min_len = min(len(gt_pos), len(ate_positions))
        ax.plot(ate_positions[:min_len, 0], ate_positions[:min_len, 1], ate_positions[:min_len, 2], label="TLEIO (RPG ATE Aligned)", color="tab:red", linestyle="-.", linewidth=2)  
    ax.scatter(*gt_pos[0], color="black", marker="o", s=60, label="Start", zorder=5)
    ax.scatter(*gt_pos[-1], color="red", marker="x", s=60, label="End", zorder=5)
    
    ax.set_title("3D Trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()

    max_range = np.array([
        gt_pos[:, 0].max() - gt_pos[:, 0].min(),
        gt_pos[:, 1].max() - gt_pos[:, 1].min(),
        gt_pos[:, 2].max() - gt_pos[:, 2].min(),
    ]).max() / 2.0
    mid_x = (gt_pos[:, 0].max() + gt_pos[:, 0].min()) * 0.5
    mid_y = (gt_pos[:, 1].max() + gt_pos[:, 1].min()) * 0.5
    mid_z = (gt_pos[:, 2].max() + gt_pos[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def compute_filter_diagnostics(
    estimated_trajectory: np.ndarray,
    ground_truth_trajectory: np.ndarray,
    regressed_trajectory: np.ndarray | None = None,
    imu_trajectory: np.ndarray | None = None,
    output_dir: Path | None = None,
    file_prefix: str = "filter",
    plot_projections: bool = False,
    aa_regressed_trajectory: np.ndarray | None = None,
    plot_ate: bool = False,
) -> dict:

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

    if regressed_trajectory is not None:
        regr = np.asarray(regressed_trajectory, dtype=np.float64)
        aligned_regr_positions, aligned_regr_quaternions = interpolate_poses(
            regr[:, 0],
            regr[:, 1:4],
            normalize_quaternions(regr[:, 4:8]),
            est_times_s,
        )
    else:
        aligned_regr_positions = None
        aligned_regr_quaternions = None

    if aa_regressed_trajectory is not None:
        aa_regr = np.asarray(aa_regressed_trajectory, dtype=np.float64)
        aligned_aa_regr_positions, _ = interpolate_poses(
            aa_regr[:, 0], aa_regr[:, 1:4], normalize_quaternions(aa_regr[:, 4:8]), est_times_s
        )
    else:
        aligned_aa_regr_positions = None

    if imu_trajectory is not None:
        imu_np = np.asarray(imu_trajectory, dtype=np.float64)
        aligned_imu_positions, aligned_imu_quaternions = interpolate_poses(
            imu_np[:, 0],
            imu_np[:, 1:4],
            normalize_quaternions(imu_np[:, 4:8]),
            est_times_s,
        )
    else:
        aligned_imu_positions = None
        aligned_imu_quaternions = None

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

    rotation_errors_deg = np.array([rotation_error_deg(q_gt, q_est) for q_gt, q_est in zip(aligned_gt_quaternions, est_quaternions)])
    rotation_rmse_deg = float(np.sqrt(np.mean(rotation_errors_deg**2)))
    max_rotation_error_deg = float(np.max(rotation_errors_deg))

    ate_est_positions = None
    ate_rmse_m = None
    if plot_ate:
        try:
            ate_rmse_m, ate_est_positions = get_rpg_ate_and_aligned_trajectory(
                ground_truth_table=gt,
                estimated_table=est
            )
        except Exception as e:
            print(f"Attentioe: Failed to calculate ATE through UZH RPG toolbox ({e})")
            ate_est_positions = None

    saved_files: dict[str, str] = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        
        saved_files["trajectory_plot"] = str(
            save_trajectory_comparison_plot(
                output_dir / f"{file_prefix}_trajectory_comparison.png",
                est_times_s,
                aligned_gt_positions,
                est_positions,
                regressed_positions=aligned_regr_positions,
                imu_positions=aligned_imu_positions,
                ate_positions=ate_est_positions,
                aa_regressed_trajectory=aligned_aa_regr_positions,
            )
        )
        saved_files["rotation_plot"] = str(
            save_rotation_comparison_plot(
                output_dir / f"{file_prefix}_rotation_comparison.png",
                est_times_s,
                aligned_gt_quaternions,
                est_quaternions,
                imu_quaternions_xyzw=aligned_imu_quaternions,
            )
        )
        saved_files["trajectory_3d_plot"] = str(
            save_3d_trajectory_plot(
                output_dir / f"{file_prefix}_trajectory_3d.png",
                aligned_gt_positions,
                est_positions,
                regressed_positions=aligned_regr_positions,
                imu_positions=aligned_imu_positions,
                ate_positions=ate_est_positions,
                aa_regressed_trajectory=aligned_aa_regr_positions,
            )
        )
        
        if plot_projections:
            saved_files["projections_plot"] = str(
                save_projections_plot(
                    output_dir / f"{file_prefix}_projections.png",
                    aligned_gt_positions,
                    est_positions,
                    regressed_positions=aligned_regr_positions,
                    imu_positions=aligned_imu_positions,
                    ate_positions=ate_est_positions,
                    aa_regressed_trajectory=aligned_aa_regr_positions,
                )
            )

    return {
        "position_rmse_m": position_rmse_m,
        "x_rmse_m": x_rmse_m,
        "y_rmse_m": y_rmse_m,
        "z_rmse_m": z_rmse_m,
        "max_position_error_m": max_position_error_m,
        "rotation_rmse_deg": rotation_rmse_deg,
        "roll_rmse_deg": roll_rmse_deg,
        "pitch_rmse_deg": pitch_rmse_deg,
        "yaw_rmse_deg": yaw_rmse_deg,
        "max_rotation_error_deg": max_rotation_error_deg,
        "ate_rmse_m": ate_rmse_m,
        "ate_positions": ate_est_positions,
        "saved_files": saved_files,
    }


def print_filter_run_summary(
    dataset: str,
    sequence: str,
    num_anchors: int,
    num_updates_attempted: int,
    num_updates_rejected: int,
    mean_residual_norm: float | None,
    mean_delta_norm: float | None,
    diagnostics: dict,
    saved_trajectory_path: str | None = None,
) -> None:

    w_label = 26
    w_total = 67

    print(f"{'Dataset:':<{w_label}} {dataset}")
    print(f"{'Sequence:':<{w_label}} {sequence}")
    print(f"{'Anchors processed:':<{w_label}} {num_anchors}")
    print(f"{'Updates attempted:':<{w_label}} {num_updates_attempted}")
    print(f"{'Updates rejected:':<{w_label}} {num_updates_rejected}")
    
    print(f"{' Position and Rotation RMSE ':-^{w_total}}")
    print(f"{'Position RMSE [m]:':<{w_label}} {diagnostics['position_rmse_m']:.6f}")
    if diagnostics.get("ate_rmse_m") is not None:
        print(f"{'ATE RMSE (RPG) [m]:':<{w_label}} {diagnostics['ate_rmse_m']:.6f}")
    if diagnostics.get("ate_transformer_m") is not None:
        print(f"{'ATE Transformer [m]:':<{w_label}} {diagnostics['ate_transformer_m']:.6f}")
    print(f"{'Rotation RMSE [deg]:':<{w_label}} {diagnostics['rotation_rmse_deg']:.6f}")
    
    print(f"{' MAX Errors ':-^{w_total}}")
    print(f"{'MAX position error [m]:':<{w_label}} {diagnostics['max_position_error_m']:.6f}")
    print(f"{'MAX rotation error [deg]:':<{w_label}} {diagnostics['max_rotation_error_deg']:.6f}")
    
    print(f"{' Position details ':-^{w_total}}")
    print(f"{'X direction RMSE [m]:':<{w_label}} {diagnostics['x_rmse_m']:.6f}")
    print(f"{'Y direction RMSE [m]:':<{w_label}} {diagnostics['y_rmse_m']:.6f}")
    print(f"{'Z direction RMSE [m]:':<{w_label}} {diagnostics['z_rmse_m']:.6f}")
    
    print(f"{' Rotation details ':-^{w_total}}")
    print(f"{'Roll RMSE [deg]:':<{w_label}} {diagnostics['roll_rmse_deg']:.6f}")
    print(f"{'Pitch RMSE [deg]:':<{w_label}} {diagnostics['pitch_rmse_deg']:.6f}")
    print(f"{'Yaw RMSE [deg]:':<{w_label}} {diagnostics['yaw_rmse_deg']:.6f}")
    
    print("-" * w_total)
    
    if mean_residual_norm is not None:
        print(f"{'Mean residual norm:':<{w_label}} {mean_residual_norm:.6f}")
        print(f"{'Mean correction norm:':<{w_label}} {mean_delta_norm:.6f}")
    else:
        print(f"{'Mean residual norm:':<{w_label}} {mean_residual_norm}")
        print(f"{'Mean correction norm:':<{w_label}} {mean_delta_norm}")
        
    print("-" * w_total)
    
    if saved_trajectory_path is not None:
        print(f"{'Saved trajectory:':<{w_label}} {saved_trajectory_path}")
        
    for key, value in diagnostics.get("saved_files", {}).items():
        print(f"{key + ':':<{w_label}} {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute diagnostics for a filter trajectory.")
    parser.add_argument("--estimated", type=Path, required=True, help="Estimated trajectory txt file.")
    parser.add_argument("--ground_truth", type=Path, required=True, help="Ground-truth trajectory txt file.")
    parser.add_argument("--regressed", type=Path, default=None, help="Optional regressed trajectory txt file.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Optional directory for plots.")
    parser.add_argument("--prefix", type=str, default="filter", help="Prefix used for saved plot filenames.")
    parser.add_argument("--plot_projections", action="store_true", help="Save 2D projection plots.")
    parser.add_argument("--plot_ate", action="store_true", help="Plot RPG Toolbox aligned trajectory to show ATE minimized distance.")
    args = parser.parse_args()

    estimated = load_trajectory_table(args.estimated)
    ground_truth = load_trajectory_table(args.ground_truth)
    
    regressed = None
    if args.regressed is not None:
        regressed = load_trajectory_table(args.regressed)

    results = compute_filter_diagnostics(
        estimated_trajectory=estimated,
        ground_truth_trajectory=ground_truth,
        regressed_trajectory=regressed,
        output_dir=args.output_dir,
        file_prefix=args.prefix,
        plot_ate=args.plot_ate,
    )
    print_filter_run_summary(
        dataset="N/A",
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
