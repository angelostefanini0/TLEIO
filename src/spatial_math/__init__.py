"""Shared spatial math helpers for rotations, quaternions, and poses."""

from .rotations import (
    T_to_pose,
    interpolate_gt_pose,
    interpolate_quaternions,
    inv_T,
    normalize_quat,
    normalize_quaternions,
    pose_to_T,
    quat_to_rotmat,
    quat_xyzw_to_rotmat,
    rotmat_log_to_rotvec,
    rotmat_to_quat,
    rotation_error_deg,
    rotvec_to_rotmat,
    slerp,
    slerp_single,
)

__all__ = [
    "T_to_pose",
    "interpolate_gt_pose",
    "interpolate_quaternions",
    "inv_T",
    "normalize_quat",
    "normalize_quaternions",
    "pose_to_T",
    "quat_to_rotmat",
    "quat_xyzw_to_rotmat",
    "rotmat_log_to_rotvec",
    "rotmat_to_quat",
    "rotation_error_deg",
    "rotvec_to_rotmat",
    "slerp",
    "slerp_single",
]
