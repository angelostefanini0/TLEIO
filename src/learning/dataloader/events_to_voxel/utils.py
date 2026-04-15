from pathlib import Path

import numpy as np
import yaml


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Found zero-norm quaternion in GT orientation data.")
    return q / norms


def slerp_single(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = alpha * theta_0

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


def interpolate_quaternions(
    gt_timestamps_us: np.ndarray,
    gt_quat_xyzw: np.ndarray,
    query_timestamps_us: np.ndarray,
) -> np.ndarray:
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


def build_camera_matrix(intrinsics: list[float]) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def scale_camera_matrix(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[1, 2] *= scale_y
    return K_scaled


def resolve_calibration_path(root_path: Path, seq_path: Path) -> Path:
    local_path = seq_path / "K.yaml"
    if local_path.exists():
        return local_path

    raw_candidate = root_path.parent / "raw" / seq_path.name / "K.yaml"
    if raw_candidate.exists():
        return raw_candidate

    raise FileNotFoundError(
        f"Could not find K.yaml for {seq_path.name}. "
        f"Tried {local_path} and {raw_candidate}."
    )


def load_event_camera_matrix(
    root_path: Path,
    seq_path: Path,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    calibration_path = resolve_calibration_path(root_path, seq_path)
    with open(calibration_path, "r") as fh:
        calibration = yaml.safe_load(fh)

    if "cam1" not in calibration or "intrinsics" not in calibration["cam1"]:
        raise KeyError(f"{calibration_path}: missing cam1 intrinsics.")

    K = build_camera_matrix(calibration["cam1"]["intrinsics"])
    return scale_camera_matrix(K, scale_x, scale_y)


def build_derotation_context(
    seq_info: dict,
    ts_start_us: int,
    ts_end_us: int,
    num_bins: int,
) -> dict:
    window_duration_us = float(ts_end_us - ts_start_us)
    bin_duration_us = window_duration_us / num_bins
    # Use bin centers to minimize the worst-case timestamp mismatch
    # between the accumulated slice and the pose used to de-rotate it.
    bin_center_us = ts_start_us + (np.arange(num_bins, dtype=np.float64) + 0.5) * bin_duration_us
    query_timestamps_us = np.concatenate(
        [bin_center_us, np.array([ts_end_us], dtype=np.float64)],
        axis=0,
    )
    query_quat = interpolate_quaternions(
        seq_info["gt_timestamps_us"],
        seq_info["gt_quat_xyzw"],
        query_timestamps_us,
    )
    return {
        "window_duration_us": np.float32(window_duration_us),
        "camera_matrix": seq_info["camera_matrix"].astype(np.float32),
        "bin_quat_xyzw": query_quat[:-1].astype(np.float32),
        "ref_quat_xyzw": query_quat[-1].astype(np.float32),
    }
