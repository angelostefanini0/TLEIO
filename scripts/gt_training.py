import argparse
from pathlib import Path

import numpy as np


def ceil_to_step(x: int, step: int) -> int:
    return ((x + step - 1) // step) * step


def floor_to_step(x: int, step: int) -> int:
    return (x // step) * step


def load_gt(data: np.ndarray):
    """
    Expected format per row:
    timestamp_us px py pz qx qy qz qw
    """
    if data.shape[1] != 8:
        raise ValueError(
            f"Expected 8 columns [t px py pz qx qy qz qw], got {data.shape[1]}"
        )

    ts = data[:, 0].astype(np.int64)
    pos = data[:, 1:4].astype(np.float64)
    quat = data[:, 4:8].astype(np.float64)  # qx qy qz qw

    if not np.all(ts[1:] >= ts[:-1]):
        raise ValueError("GT timestamps must be sorted in ascending order.")

    quat = normalize_quaternions(quat)
    return ts, pos, quat


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Found zero-norm quaternion.")
    return q / norms


def get_anchor_grid(
    gt_timestamps_us: np.ndarray,
    delta_t_us: int = 50_000,
    anchor_step_us: int = 50_000,
) -> np.ndarray:
    """
    Generate a fixed anchor grid aligned to multiples of anchor_step_us.

    Assumes timestamps are already in the same 0-based frame as the events.

    A valid anchor t must satisfy:
    - t >= delta_t_us   so the causal voxel [t - delta_t_us, t) exists
    - t >= first GT timestamp
    - t <= last GT timestamp
    """
    valid_start_us = max(delta_t_us, int(gt_timestamps_us[0]))
    valid_end_us = int(gt_timestamps_us[-1])

    if valid_end_us < valid_start_us:
        return np.empty((0,), dtype=np.int64)

    first_anchor_us = ceil_to_step(valid_start_us, anchor_step_us)
    last_anchor_us = floor_to_step(valid_end_us, anchor_step_us)

    if last_anchor_us < first_anchor_us:
        return np.empty((0,), dtype=np.int64)

    return np.arange(
        first_anchor_us,
        last_anchor_us + 1,
        anchor_step_us,
        dtype=np.int64,
    )


def slerp_single(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Quaternion Spherical Linear intERPolation (SLERP) for q = [qx, qy, qz, qw].
    """
    dot = np.dot(q0, q1)

    # Use shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    # If very close, use normalized lerp
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        return q 

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = alpha * theta_0
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = s0 * q0 + s1 * q1
    return q 


def interpolate_gt_to_anchors(
    gt_timestamps_us: np.ndarray,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    anchors_us: np.ndarray,
):
    """
    Interpolate GT poses at anchor timestamps.

    Translation: linear interpolation
    Rotation: quaternion SLERP
    """
    if len(anchors_us) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 4), dtype=np.float64),
        )

    right_idx = np.searchsorted(gt_timestamps_us, anchors_us, side="left")
    right_idx = np.clip(right_idx, 1, len(gt_timestamps_us) - 1)
    left_idx = right_idx - 1

    t0 = gt_timestamps_us[left_idx].astype(np.float64)
    t1 = gt_timestamps_us[right_idx].astype(np.float64)

    denom = t1 - t0
    if np.any(denom <= 0):
        raise ValueError("Non-increasing GT timestamps encountered.")

    alpha = (anchors_us.astype(np.float64) - t0) / denom
    alpha = np.clip(alpha, 0.0, 1.0)

    p0 = gt_pos[left_idx]
    p1 = gt_pos[right_idx]
    interp_pos = (1.0 - alpha[:, None]) * p0 + alpha[:, None] * p1

    q0 = gt_quat[left_idx]
    q1 = gt_quat[right_idx]
    interp_quat = np.stack(
        [slerp_single(a, b, w) for a, b, w in zip(q0, q1, alpha)],
        axis=0,
    )

    return interp_pos, interp_quat


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    q: [qx, qy, qz, qw]
    returns 3x3 rotation matrix
    """
    qx, qy, qz, qw = q
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    R = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)
    return R

import numpy as np


def rotmat_log_to_rotvec(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a 3D rotation vector using log(R).

    Returns:
        rotvec: shape (3,), where norm(rotvec) is the rotation angle in radians
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected R shape (3,3), got {R.shape}")

    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Skew-symmetric part vee(R - R^T)
    w = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)

    # Small-angle case
    if theta < 1e-5:
        return 0.5 * w

    # Near-pi case
    if np.pi - theta < 1e-5:
        A = (R + np.eye(3)) / 2.0
        axis = np.empty(3, dtype=np.float64)
        axis[0] = np.sqrt(max(A[0, 0], 0.0))
        axis[1] = np.sqrt(max(A[1, 1], 0.0))
        axis[2] = np.sqrt(max(A[2, 2], 0.0))

        # Fix signs using off-diagonal terms
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]

        norm_axis = np.linalg.norm(axis)
        if norm_axis < eps:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= norm_axis
        return theta * axis

    return (theta / (2.0 * np.sin(theta))) * w

def pose_to_T(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotmat(quat_xyzw)
    T[:3, 3] = pos
    return T


def inv_T(T: np.ndarray): 
    R_inv = T[:3, :3].T
    p_inv = -R_inv@T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = p_inv
    return T_inv

def compute_relative_motions(anchor_ts: np.ndarray, anchor_pos: np.ndarray, anchor_quat: np.ndarray):
    """
    For consecutive anchor poses:
    T_rel_i = inv(T_i) @ T_{i+1}

    Returns rows:
    t0_us t1_us px py pz rx ry rz
    where r indicates the rotation vector component
    """
    rows = []

    for i in range(len(anchor_ts) - 1):
        T_i = pose_to_T(anchor_pos[i], anchor_quat[i])
        T_i1 = pose_to_T(anchor_pos[i + 1], anchor_quat[i + 1])

        T_rel = inv_T(T_i) @ T_i1
        axis_vec_rel = rotmat_log_to_rotvec(T_rel[:3,:3])
        trans_rel = T_rel[:3, 3]

        row = np.concatenate([
            np.array([anchor_ts[i], anchor_ts[i + 1]], dtype=np.int64),
            trans_rel,
            axis_vec_rel
        ])
        rows.append(row)

    if len(rows) == 0:
        return np.empty((0, 8), dtype=np.float64)

    return np.stack(rows, axis=0)


def save_anchor_poses(out_path: Path, anchor_ts: np.ndarray, anchor_pos: np.ndarray, anchor_quat: np.ndarray):
    out = np.column_stack([anchor_ts, anchor_pos, anchor_quat])
    header = "timestamp_us px py pz qx qy qz qw"
    np.savetxt(out_path, out, fmt=["%d"] + ["%.10f"] * 7, header=header, comments="")


def save_relative_motions(out_path: Path, rel: np.ndarray):
    header = "t0_us t1_us px py pz rx ry rz"
    if rel.size == 0:
        np.savetxt(out_path, rel, header=header, comments="")
        return
    np.savetxt(
        out_path,
        rel,
        fmt=["%d", "%d"] + ["%.10f"] * 6,
        header=header,
        comments="",
    )
    