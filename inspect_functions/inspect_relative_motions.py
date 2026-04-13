import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
"""
python inspect_functions/inspect_relative_motions.py \
  --gt data/eds/processed/00_peanuts_dark/stamped_groundtruth.txt \
  --rel path/to/predicted_relative_motions.txt \
  --gt_rel data/eds/processed/00_peanuts_dark/relative_motions.txt \
  --gt_rel_mode rotation
""""""
python inspect_functions/inspect_relative_motions.py --gt data/eds/processed/00_peanuts_dark/stamped_groundtruth.txt --rel path/to/predicted_relative_motions.txt  --gt_rel data/eds/processed/00_peanuts_dark/relative_motions.txt --gt_rel_mode rotation
"""

def load_table(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        first = f.readline().strip()

    skiprows = 1 if first and (first[0].isalpha() or first.startswith("#")) else 0
    data = np.loadtxt(path, skiprows=skiprows, dtype=np.float64)

    if data.ndim == 1:
        data = data[None, :]

    return data


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("Found zero-norm quaternion.")
    return q / n


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = rotvec / theta
    x, y, z = axis
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)

    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def pose_to_T(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rotmat(quat)
    T[:3, 3] = pos
    return T


def T_to_pose(T: np.ndarray):
    pos = T[:3, 3].copy()
    quat = rotmat_to_quat(T[:3, :3])
    return pos, quat


def parse_rel_row_to_T(row: np.ndarray) -> np.ndarray:
    """
    Supported row formats:
    - Axis-vector: t0_us t1_us px py pz rx ry rz
    """

    if row.shape[0] == 8:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotvec_to_rotmat(row[5:8])
        T[:3, 3] = row[2:5]
        return T

    if row.shape[0] == 8:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotvec_to_rotmat(row[5:8])
        T[:3, 3] = row[2:5]
        return T

    raise ValueError(
        f"Unsupported relative motion row with {row.shape[0]} columns. Expected 8"
    )


def describe_rel_format(num_cols: int) -> str:
    if num_cols == 8:
        return "axis-vector + translation"
    return f"unknown ({num_cols} cols)"


def fuse_rel_transforms(pred_row: np.ndarray, gt_row: np.ndarray | None, mode: str) -> np.ndarray:
    T_rel = parse_rel_row_to_T(pred_row)
    if gt_row is None or mode == "none":
        return T_rel

    T_gt = parse_rel_row_to_T(gt_row)

    if mode == "rotation":
        T_rel[:3, :3] = T_gt[:3, :3]
    elif mode == "translation":
        T_rel[:3, 3] = T_gt[:3, 3]
    elif mode == "both":
        T_rel = T_gt
    else:
        raise ValueError(f"Unsupported gt_rel_mode '{mode}'.")

    return T_rel


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
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


def interpolate_gt_pose(
    gt_ts: np.ndarray,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    query_ts: np.ndarray,
):
    right = np.searchsorted(gt_ts, query_ts, side="left")
    right = np.clip(right, 1, len(gt_ts) - 1)
    left = right - 1

    t0 = gt_ts[left].astype(np.float64)
    t1 = gt_ts[right].astype(np.float64)
    alpha = (query_ts.astype(np.float64) - t0) / (t1 - t0)
    alpha = np.clip(alpha, 0.0, 1.0)

    p0 = gt_pos[left]
    p1 = gt_pos[right]
    pos = (1.0 - alpha[:, None]) * p0 + alpha[:, None] * p1

    q0 = gt_quat[left]
    q1 = gt_quat[right]
    quat = np.stack([slerp(a, b, w) for a, b, w in zip(q0, q1, alpha)], axis=0)

    return pos, quat


def rotation_error_deg(q_ref: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    q_ref = normalize_quat(q_ref)
    q_est = normalize_quat(q_est)
    dots = np.abs(np.sum(q_ref * q_est, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def set_axes_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius < 1e-12:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True,
                        help="stamped_groundtruth.txt with columns: timestamp_us px py pz qx qy qz qw")
    parser.add_argument("--rel", type=Path, required=True,
                        help="relative_motions.txt with columns either: "
                             "[t0_us t1_us px py pz rx ry rz]")
    parser.add_argument("--save_dir", type=Path, default=None,
                        help="Optional directory to save figures instead of showing them")
    
    parser.add_argument("--gt_rel", type=Path, default=None,
                        help="Optional GT relative motions used to inspect partial network results.")
    parser.add_argument("--gt_rel_mode", type=str, default="none",
                        choices=["none", "rotation", "translation", "both"],
                        help="How to combine --rel with --gt_rel before integration. "
                             "Default 'rotation' keeps predicted translation and uses GT rotation.")
    
    args = parser.parse_args()

    gt = load_table(args.gt)
    rel = load_table(args.rel)
    gt_rel = load_table(args.gt_rel) if args.gt_rel is not None else None

    if gt.shape[1] != 8:
        raise ValueError(f"{args.gt} has {gt.shape[1]} columns, expected 8.")
    if rel.shape[1] != 8:
        raise ValueError(f"{args.rel} has {rel.shape[1]} columns, expected 8")
    if gt_rel is not None and gt_rel.shape[1] != 8:
        raise ValueError(f"{args.gt_rel} has {gt_rel.shape[1]} columns, expected 8")

    gt_ts = gt[:, 0].astype(np.int64)
    gt_pos = gt[:, 1:4]
    gt_quat = normalize_quat(gt[:, 4:8])

    if len(rel) == 0:
        raise ValueError("Relative motions file is empty.")

    rel_t0 = rel[:, 0].astype(np.int64)
    rel_t1 = rel[:, 1].astype(np.int64)

    if not np.all(rel_t1 > rel_t0):
        raise ValueError("Each relative motion must satisfy t1_us > t0_us.")
    if len(rel) > 1 and not np.array_equal(rel_t0[1:], rel_t1[:-1]):
        raise ValueError("Relative motions do not form a continuous timestamp chain.")
    if gt_rel is not None:
        gt_rel_t0 = gt_rel[:, 0].astype(np.int64)
        gt_rel_t1 = gt_rel[:, 1].astype(np.int64)
        if len(gt_rel) != len(rel):
            raise ValueError("--gt_rel must have the same number of rows as --rel.")
        if not np.array_equal(rel_t0, gt_rel_t0) or not np.array_equal(rel_t1, gt_rel_t1):
            raise ValueError("--gt_rel timestamps do not match --rel timestamps.")

    # Anchor timestamps implied by relative motions
    anchor_ts = np.concatenate([rel_t0[:1], rel_t1])

    # Initial pose from source GT at first anchor
    init_pos, init_quat = interpolate_gt_pose(
        gt_ts, gt_pos, gt_quat, np.array([anchor_ts[0]], dtype=np.int64)
    )
    T_chain = pose_to_T(init_pos[0], init_quat[0])

    # Reconstruct trajectory only from relative motions
    recon_pos = [init_pos[0]]
    recon_quat = [init_quat[0]]

    for i in range(len(rel)):
        gt_rel_row = None if gt_rel is None else gt_rel[i]
        T_rel = fuse_rel_transforms(rel[i], gt_rel_row, args.gt_rel_mode)
        T_chain = T_chain @ T_rel
        p, q = T_to_pose(T_chain)
        recon_pos.append(p)
        recon_quat.append(q)

    recon_pos = np.stack(recon_pos, axis=0)
    recon_quat = normalize_quat(np.stack(recon_quat, axis=0))

    # GT reference at the same anchor timestamps
    ref_pos, ref_quat = interpolate_gt_pose(gt_ts, gt_pos, gt_quat, anchor_ts)
    ref_quat = normalize_quat(ref_quat)

    if gt_rel is not None:
        rel_eval = rel.copy()
        if args.gt_rel_mode == "rotation":
            rel_eval[:, 5:8] = gt_rel[:, 5:8]
        elif args.gt_rel_mode == "translation":
            rel_eval[:, 2:5] = gt_rel[:, 2:5]
        elif args.gt_rel_mode == "both":
            rel_eval[:, 2:8] = gt_rel[:, 2:8]

        pos_err = np.linalg.norm(rel_eval[:, 2:5] - gt_rel[:, 2:5], axis=1)
        rel_quat = normalize_quat(
            np.stack(
                [rotmat_to_quat(rotvec_to_rotmat(rv)) for rv in rel_eval[:, 5:8]],
                axis=0,
            )
        )
        gt_rel_quat = normalize_quat(
            np.stack(
                [rotmat_to_quat(rotvec_to_rotmat(rv)) for rv in gt_rel[:, 5:8]],
                axis=0,
            )
        )
        rot_err = rotation_error_deg(gt_rel_quat, rel_quat)
    else:
        pos_err = np.linalg.norm(recon_pos - ref_pos, axis=1)
        rot_err = rotation_error_deg(ref_quat, recon_quat)

    rel_format = describe_rel_format(rel.shape[1])
    gt_rel_format = "none" if gt_rel is None else describe_rel_format(gt_rel.shape[1])
    error_ref_label = "GT relative motions" if gt_rel is not None else "source GT"

    print(f"Relative motions vs {error_ref_label}")
    print(f"GT poses:                 {len(gt_ts)}")
    print(f"Relative motions:         {len(rel)}")
    print(f"Relative format:          {rel_format}")
    print(f"GT rel format:            {gt_rel_format}")
    print(f"GT rel fusion mode:       {args.gt_rel_mode}")
    print(f"Reconstructed anchors:    {len(anchor_ts)}")
    print(f"Position RMSE [m]:        {np.sqrt(np.mean(pos_err ** 2)):.6e}")
    print(f"Rotation RMSE [deg]:      {np.sqrt(np.mean(rot_err ** 2)):.6e}")

    t_gt = (gt_ts - gt_ts[0]) * 1e-6
    t_anchor = (anchor_ts - gt_ts[0]) * 1e-6
    t_err = (rel_t1 - gt_ts[0]) * 1e-6 if gt_rel is not None else t_anchor

    fig1, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["x", "y", "z"]
    for i in range(3):
        axes[i].plot(t_gt, gt_pos[:, i], label="raw stamped GT")
        axes[i].plot(t_anchor, ref_pos[:, i], "--", label="GT at anchor times")
        axes[i].plot(t_anchor, recon_pos[:, i], label="trajectory from relative motions")
        axes[i].set_ylabel(f"p{labels[i]} [m]")
        axes[i].grid(True)
    axes[0].legend()
    axes[-1].set_xlabel("time [s]")
    fig1.suptitle("Trajectory from Relative Motions vs Source GT")

    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], label="raw stamped GT")
    ax.plot(ref_pos[:, 0], ref_pos[:, 1], "--", label="GT at anchor times")
    ax.plot(recon_pos[:, 0], recon_pos[:, 1], label="trajectory from relative motions")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.legend()
    ax.set_title("XY trajectory comparison")

    fig3 = plt.figure(figsize=(10, 8))
    ax3d = fig3.add_subplot(111, projection="3d")
    ax3d.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label="raw stamped GT")
    ax3d.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], "--", label="GT at anchor times")
    ax3d.plot(
        recon_pos[:, 0],
        recon_pos[:, 1],
        recon_pos[:, 2],
        label="trajectory from relative motions",
    )
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("3D trajectory comparison")
    all_points = np.vstack([gt_pos, ref_pos, recon_pos])
    set_axes_equal_3d(ax3d, all_points)
    ax3d.legend()

    fig4, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t_err, pos_err)
    axes[0].set_ylabel("pos err [m]")
    axes[0].grid(True)
    axes[1].plot(t_err, rot_err)
    axes[1].set_ylabel("rot err [deg]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True)
    fig4.suptitle(f"Error vs {error_ref_label}")

    plt.tight_layout()

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(args.save_dir / "relative_vs_gt_xyz.png", dpi=150, bbox_inches="tight")
        fig2.savefig(args.save_dir / "relative_vs_gt_xy.png", dpi=150, bbox_inches="tight")
        fig3.savefig(args.save_dir / "relative_vs_gt_xyz_3d.png", dpi=150, bbox_inches="tight")
        fig4.savefig(args.save_dir / "relative_vs_gt_error.png", dpi=150, bbox_inches="tight")
        print(f"Saved figures to {args.save_dir}")
    else:
        plt.show()


if __name__ == "__main__":
    main()