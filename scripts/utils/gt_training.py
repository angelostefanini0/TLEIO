import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.spatial_math import (
    interpolate_gt_pose,
    inv_T,
    normalize_quaternions,
    pose_to_T,
    rotmat_log_to_rotvec,
)


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


def interpolate_gt_to_anchors(
    gt_timestamps_us: np.ndarray,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    anchors_us: np.ndarray,
):
    return interpolate_gt_pose(gt_timestamps_us, gt_pos, gt_quat, anchors_us)

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
