"""NumPy helpers for quaternion and rotation algebra.

Quaternions use xyzw order throughout this module.
"""

from __future__ import annotations

import numpy as np


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions stored in xyzw order along the last axis."""
    q = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Found zero-norm quaternion.")
    return q / norms


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Alias for code paths that use the singular helper name."""
    return normalize_quaternions(q)


def slerp_single(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherically interpolate between two quaternions in xyzw order."""
    q0 = normalize_quaternions(q0)
    q1 = normalize_quaternions(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        return normalize_quaternions(q)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = alpha * theta_0

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    q = s0 * q0 + s1 * q1
    return normalize_quaternions(q)


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Alias for single quaternion SLERP."""
    return slerp_single(q0, q1, alpha)


def interpolate_quaternions(
    gt_timestamps_us: np.ndarray,
    gt_quat_xyzw: np.ndarray,
    query_timestamps_us: np.ndarray,
) -> np.ndarray:
    """Interpolate ground-truth orientations at requested timestamps."""
    if len(query_timestamps_us) == 0:
        return np.empty((0, 4), dtype=np.float64)

    right_idx = np.searchsorted(gt_timestamps_us, query_timestamps_us, side="left")
    right_idx = np.clip(right_idx, 1, len(gt_timestamps_us) - 1)
    left_idx = right_idx - 1

    t0 = gt_timestamps_us[left_idx].astype(np.float64)
    t1 = gt_timestamps_us[right_idx].astype(np.float64)
    denom = t1 - t0
    if np.any(denom <= 0):
        raise ValueError("GT timestamps must be strictly increasing for interpolation.")

    alpha = (query_timestamps_us.astype(np.float64) - t0) / denom
    alpha = np.clip(alpha, 0.0, 1.0)

    q0 = gt_quat_xyzw[left_idx]
    q1 = gt_quat_xyzw[right_idx]
    return np.stack(
        [slerp_single(a, b, w) for a, b, w in zip(q0, q1, alpha)],
        axis=0,
    )


def interpolate_gt_pose(
    gt_timestamps_us: np.ndarray,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    query_timestamps_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate positions and SLERP orientations at query times."""
    if len(query_timestamps_us) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 4), dtype=np.float64),
        )

    right_idx = np.searchsorted(gt_timestamps_us, query_timestamps_us, side="left")
    right_idx = np.clip(right_idx, 1, len(gt_timestamps_us) - 1)
    left_idx = right_idx - 1

    t0 = gt_timestamps_us[left_idx].astype(np.float64)
    t1 = gt_timestamps_us[right_idx].astype(np.float64)
    denom = t1 - t0
    if np.any(denom <= 0):
        raise ValueError("GT timestamps must be strictly increasing for interpolation.")

    alpha = (query_timestamps_us.astype(np.float64) - t0) / denom
    alpha = np.clip(alpha, 0.0, 1.0)

    p0 = gt_pos[left_idx]
    p1 = gt_pos[right_idx]
    pos = (1.0 - alpha[:, None]) * p0 + alpha[:, None] * p1
    quat = interpolate_quaternions(gt_timestamps_us, gt_quat, query_timestamps_us)
    return pos, quat


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert one quaternion in xyzw order to a 3 x 3 rotation matrix."""
    qx, qy, qz, qw = normalize_quaternions(q)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Alias for xyzw quaternion-to-rotation conversion."""
    return quat_xyzw_to_rotmat(q)


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector to a 3 x 3 rotation matrix."""
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
    """Convert a 3 x 3 rotation matrix to a quaternion in xyzw order."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected R shape (3,3), got {R.shape}")

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

    return normalize_quaternions(np.array([qx, qy, qz, qw], dtype=np.float64))


def rotmat_log_to_rotvec(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert a 3 x 3 rotation matrix to a rotation vector using log(R)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected R shape (3,3), got {R.shape}")

    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    w = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)

    if theta < 1e-5:
        return 0.5 * w

    if np.pi - theta < 1e-5:
        A = (R + np.eye(3)) / 2.0
        axis = np.empty(3, dtype=np.float64)
        axis[0] = np.sqrt(max(A[0, 0], 0.0))
        axis[1] = np.sqrt(max(A[1, 1], 0.0))
        axis[2] = np.sqrt(max(A[2, 2], 0.0))

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
    """Build a 4 x 4 transform matrix from position and xyzw quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotmat(quat_xyzw)
    T[:3, 3] = pos
    return T


def T_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and xyzw quaternion from a 4 x 4 transform matrix."""
    pos = T[:3, 3].copy()
    quat = rotmat_to_quat(T[:3, :3])
    return pos, quat


def inv_T(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4 x 4 transform matrix."""
    R_inv = T[:3, :3].T
    p_inv = -R_inv @ T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = p_inv
    return T_inv


def rotation_error_deg(q_ref: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Return absolute angular distance between quaternion arrays in degrees."""
    q_ref = normalize_quaternions(q_ref)
    q_est = normalize_quaternions(q_est)
    dots = np.abs(np.sum(q_ref * q_est, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))
