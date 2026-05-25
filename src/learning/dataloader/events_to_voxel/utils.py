"""Pose, calibration, and de-rotation metadata helpers for event voxelization."""

from pathlib import Path

import numpy as np
import yaml

from src.spatial_math import interpolate_quaternions, normalize_quaternions


def build_camera_matrix(intrinsics: list[float]) -> np.ndarray:
    """Build a pinhole camera matrix from ``[fx, fy, cx, cy]`` intrinsics."""
    fx, fy, cx, cy = intrinsics
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def scale_camera_matrix(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Scale a camera matrix to match resized event coordinates.

    The focal lengths and principal point are scaled independently in x and y.
    """
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[1, 2] *= scale_y
    return K_scaled


def resolve_calibration_path(root_path: Path, seq_path: Path) -> Path:
    """Find the event-camera calibration file for a processed sequence.

    The function first checks ``seq_path / "K.yaml"``. If it is not present,
    it falls back to the matching raw sequence folder at
    ``root_path.parent / "raw" / seq_path.name / "K.yaml"``.
    """
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
    """Load and scale the event-camera intrinsic matrix for a sequence.

    Args:
        root_path: Root directory containing processed sequence folders.
        seq_path: Path to the current processed sequence.
        scale_x: Horizontal coordinate scale applied to events.
        scale_y: Vertical coordinate scale applied to events.

    Returns:
        A ``3 x 3`` camera matrix matching the downsampled event resolution.
    """
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
    """Build pose and calibration metadata required for event de-rotation.

    Args:
        seq_info: Sequence metadata containing ``gt_timestamps_us``,
            ``gt_quat_xyzw``, and ``camera_matrix``.
        ts_start_us: Start of the event window in absolute microseconds.
        ts_end_us: End of the event window in absolute microseconds.
        num_bins: Number of temporal de-rotation slices. This can differ from
            the number of final voxel bins.

    Returns:
        A dictionary containing the fixed window duration, scaled camera
        matrix, one interpolated quaternion per de-rotation slice, and the
        reference quaternion at ``ts_end_us``.
    """
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
