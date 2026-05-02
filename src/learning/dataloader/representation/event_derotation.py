"""Event-space de-rotation helpers shared by visualization and voxelization."""

from __future__ import annotations

import numpy as np
import torch

from .trilinear_interpolation import trilinear_voxel_interpolation


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert one quaternion in ``xyzw`` order to a ``3 x 3`` rotation matrix."""
    qx, qy, qz, qw = q
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def homography_from_bin_to_ref(
    camera_matrix: np.ndarray,
    bin_quat_xyzw: np.ndarray,
    ref_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """
    Build the image-plane homography from one pose slice to the reference pose.

    Args:
        camera_matrix: Event-camera intrinsic matrix.
        bin_quat_xyzw: Orientation associated with the source temporal slice.
        ref_quat_xyzw: Reference orientation: the one at the end of the event
        window.

    Returns:
        A `3 x 3` homography that maps source-slice pixel coordinates into
        the reference orientation.
    """
    K = np.asarray(camera_matrix, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    R_ref = quat_xyzw_to_rotmat(np.asarray(ref_quat_xyzw, dtype=np.float64))
    R_bin = quat_xyzw_to_rotmat(np.asarray(bin_quat_xyzw, dtype=np.float64))
    R_ref_from_bin = R_ref.T @ R_bin
    H_ref_from_bin = K @ R_ref_from_bin @ K_inv
    H_ref_from_bin /= H_ref_from_bin[2, 2]
    return H_ref_from_bin

def resolve_derotation_slices(
    duration_ms: float,
    derotation_slices: int | None,
    derotation_slice_ms: float,
) -> int:
    """
    Resolve an explicit or duration-derived number of de-rotation slices.
    """
    if derotation_slices is not None:
        if derotation_slices < 1:
            raise ValueError("derotation_slices must be >= 1.")
        return derotation_slices

    if derotation_slice_ms <= 0:
        raise ValueError("derotation_slice_ms must be positive.")
    return max(1, int(round(duration_ms / derotation_slice_ms)))


def warp_points_with_homography(
    x: np.ndarray,
    y: np.ndarray,
    homography: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Warp event coordinates with a homography.

    Args:
        x: Event x coordinates.
        y: Event y coordinates.
        homography: ``3 x 3`` projective transform.

    Returns:
        Warped x coordinates, warped y coordinates, and a boolean validity mask
        for finite projective results.
    """
    points_h = np.stack([x, y, np.ones_like(x)], axis=0)
    warped_h = np.asarray(homography, dtype=np.float64) @ points_h
    denom = warped_h[2]
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-12)

    warped_x = np.full_like(x, np.nan, dtype=np.float64)
    warped_y = np.full_like(y, np.nan, dtype=np.float64)
    warped_x[valid] = warped_h[0, valid] / denom[valid]
    warped_y[valid] = warped_h[1, valid] / denom[valid]
    valid &= np.isfinite(warped_x) & np.isfinite(warped_y)
    return warped_x, warped_y, valid


def derotate_events_in_slices(
    x: np.ndarray,
    y: np.ndarray,
    t_us: np.ndarray,
    ts_start_us: int,
    ts_end_us: int,
    context: dict,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """De-rotate event coordinates before voxelization.

    The window is split into short temporal slices. Each event receives the
    homography of its slice center, because intra-slice rotation is treated as
    negligible.

    Args:
        x: Event x coordinates in the voxelization resolution.
        y: Event y coordinates in the voxelization resolution.
        t_us: Absolute event timestamps in microseconds.
        ts_start_us: Absolute start timestamp of the fixed event window.
        ts_end_us: Absolute end timestamp of the fixed event window.
        context: De-rotation metadata from `build_derotation_context`.
        width: Output image width used to reject warped events outside bounds.
        height: Output image height used to reject warped events outside bounds.

    Returns:
        Warped x coordinates, warped y coordinates, an in-bounds validity mask,
        and the per-slice homographies used for warping.
    """
    num_slices = len(context["bin_quat_xyzw"])
    if num_slices < 1:
        raise ValueError("Derotation context must contain at least one slice quaternion.")

    window_duration_us = float(ts_end_us - ts_start_us)
    if window_duration_us <= 0:
        raise ValueError("Window duration must be positive.")

    rel_t_us = t_us.astype(np.float64) - float(ts_start_us)
    slice_idx = np.floor(rel_t_us * num_slices / window_duration_us).astype(np.int64)
    slice_idx = np.clip(slice_idx, 0, num_slices - 1)

    derot_x = np.full_like(x, np.nan, dtype=np.float64)
    derot_y = np.full_like(y, np.nan, dtype=np.float64)
    valid = np.zeros(len(x), dtype=bool)
    homographies = []

    for idx in range(num_slices):
        homography = homography_from_bin_to_ref(
            camera_matrix=context["camera_matrix"],
            bin_quat_xyzw=context["bin_quat_xyzw"][idx],
            ref_quat_xyzw=context["ref_quat_xyzw"],
        )
        homographies.append(homography)

        mask = slice_idx == idx
        if not np.any(mask):
            continue

        warped_x, warped_y, warped_valid = warp_points_with_homography(
            x=x[mask],
            y=y[mask],
            homography=homography,
        )

        mask_indices = np.flatnonzero(mask)
        derot_x[mask_indices] = warped_x
        derot_y[mask_indices] = warped_y
        valid[mask_indices] = warped_valid

    valid &= (derot_x >= 0.0) & (derot_x < width) & (derot_y >= 0.0) & (derot_y < height)
    return derot_x, derot_y, valid, homographies


def raw_events_to_fixed_window_voxel(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t_us: np.ndarray,
    ts_start_us: int,
    ts_end_us: int,
    num_bins: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Voxelize events over the fixed physical window ``[ts_start_us, ts_end_us)``.

    This helper is used by the visualization path to build raw and de-rotated
    voxel grids with the same fixed-window time convention as the training
    de-rotation path.
    """
    window_duration_us = float(ts_end_us - ts_start_us)
    if window_duration_us <= 0:
        raise ValueError("Window duration must be positive.")

    x_t = torch.from_numpy(x.astype(np.float32, copy=False))
    y_t = torch.from_numpy(y.astype(np.float32, copy=False))
    p_t = torch.from_numpy(p.astype(np.float32, copy=False))
    time_t = torch.from_numpy((t_us.astype(np.float64) - float(ts_start_us)).astype(np.float32))
    t_norm = (num_bins - 1) * time_t / window_duration_us
    return trilinear_voxel_interpolation(
        x=x_t,
        y=y_t,
        pol=p_t,
        t_norm=t_norm,
        channels=num_bins,
        height=height,
        width=width,
    )
