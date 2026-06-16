"""Compute raw, SE3-aligned, and Sim3-aligned ATE metrics for trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_trajectory(path: Path) -> np.ndarray:
    """Load a trajectory table while skipping non-numeric header rows."""

    rows: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue

    table = np.asarray(rows, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 8:
        raise ValueError(f"{path} has shape {table.shape}, expected N x 8 trajectory table.")
    table = table.copy()
    table[:, 0] = normalize_time_to_seconds(table[:, 0])
    return table


def normalize_time_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Infer seconds vs microseconds/nanoseconds and return seconds."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    positive_diffs = np.diff(timestamps)
    positive_diffs = positive_diffs[positive_diffs > 0.0]
    median_dt = float(np.median(positive_diffs)) if len(positive_diffs) else 0.0
    if median_dt > 1e7:
        return timestamps * 1e-9
    if median_dt > 1e1:
        return timestamps * 1e-6
    return timestamps.copy()


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two xyzw quaternions."""

    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)

    theta0 = np.arccos(dot)
    theta = alpha * theta0
    s0 = np.sin(theta0 - theta) / np.sin(theta0)
    s1 = np.sin(theta) / np.sin(theta0)
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


def interpolate_ground_truth(ground_truth: np.ndarray, query_times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ground-truth positions and quaternions onto query times."""

    gt_times_s = ground_truth[:, 0]
    if query_times_s[0] < gt_times_s[0] - 1e-9 or query_times_s[-1] > gt_times_s[-1] + 1e-9:
        raise ValueError("Estimated timestamps fall outside the ground-truth time range.")

    right = np.searchsorted(gt_times_s, query_times_s, side="left")
    right = np.clip(right, 1, len(gt_times_s) - 1)
    left = right - 1
    t0 = gt_times_s[left]
    t1 = gt_times_s[right]
    alpha = np.clip((query_times_s - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)

    gt_positions = ground_truth[:, 1:4]
    p0 = gt_positions[left]
    p1 = gt_positions[right]
    positions = (1.0 - alpha[:, None]) * p0 + alpha[:, None] * p1

    gt_quaternions = ground_truth[:, 4:8]
    q0 = gt_quaternions[left]
    q1 = gt_quaternions[right]
    quaternions = np.stack([_slerp(a, b, w) for a, b, w in zip(q0, q1, alpha)], axis=0)
    return positions, quaternions


def umeyama_alignment(source: np.ndarray, target: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping source points to target points."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected matching N x 3 point arrays, got {source.shape} and {target.shape}.")

    count = source.shape[0]
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / count

    U, singular_values, Vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        sign[-1, -1] = -1.0
    rotation = U @ sign @ Vt

    if with_scale:
        source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
        if source_variance <= 0.0:
            raise ValueError("Cannot compute Sim3 scale for a degenerate source trajectory.")
        scale = float(np.sum(singular_values * np.diag(sign)) / source_variance)
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_alignment(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a similarity transform to N x 3 points."""

    return scale * (points @ rotation.T) + translation


def ate_rmse(estimated_positions: np.ndarray, reference_positions: np.ndarray) -> float:
    """Compute absolute trajectory error RMSE."""

    errors = estimated_positions - reference_positions
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def rotation_rmse_deg(reference_quaternions: np.ndarray, estimated_quaternions: np.ndarray, align_rotation: np.ndarray | None = None) -> float:
    """Compute quaternion geodesic rotation RMSE in degrees."""

    reference_quaternions = reference_quaternions / np.linalg.norm(reference_quaternions, axis=1, keepdims=True)
    estimated_quaternions = estimated_quaternions / np.linalg.norm(estimated_quaternions, axis=1, keepdims=True)
    if align_rotation is not None:
        estimated_quaternions = (
            Rotation.from_matrix(align_rotation) * Rotation.from_quat(estimated_quaternions)
        ).as_quat()
        estimated_quaternions = estimated_quaternions / np.linalg.norm(estimated_quaternions, axis=1, keepdims=True)

    dots = np.abs(np.sum(reference_quaternions * estimated_quaternions, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    errors_deg = np.rad2deg(2.0 * np.arccos(dots))
    return float(np.sqrt(np.mean(errors_deg * errors_deg)))


def compute_metrics(estimated: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Compute ATE and rotation metrics for one estimated trajectory."""

    gt_positions, gt_quaternions = interpolate_ground_truth(ground_truth, estimated[:, 0])
    estimated_positions = estimated[:, 1:4]
    estimated_quaternions = estimated[:, 4:8]

    se3_scale, se3_rotation, se3_translation = umeyama_alignment(
        estimated_positions, gt_positions, with_scale=False
    )
    sim3_scale, sim3_rotation, sim3_translation = umeyama_alignment(
        estimated_positions, gt_positions, with_scale=True
    )
    se3_positions = apply_alignment(estimated_positions, se3_scale, se3_rotation, se3_translation)
    sim3_positions = apply_alignment(estimated_positions, sim3_scale, sim3_rotation, sim3_translation)

    return {
        "raw_ate_rmse_m": ate_rmse(estimated_positions, gt_positions),
        "se3_aligned_ate_rmse_m": ate_rmse(se3_positions, gt_positions),
        "sim3_aligned_scaled_ate_rmse_m": ate_rmse(sim3_positions, gt_positions),
        "sim3_scale": sim3_scale,
        "raw_rotation_rmse_deg": rotation_rmse_deg(gt_quaternions, estimated_quaternions),
        "se3_aligned_rotation_rmse_deg": rotation_rmse_deg(
            gt_quaternions, estimated_quaternions, align_rotation=se3_rotation
        ),
    }


def compute_metrics_for_paths(estimated_paths: list[Path], ground_truth_path: Path) -> list[dict[str, float | str]]:
    """Compute metrics for several estimated trajectory files."""

    ground_truth = load_trajectory(ground_truth_path)
    rows: list[dict[str, float | str]] = []
    for estimated_path in estimated_paths:
        estimated = load_trajectory(estimated_path)
        metrics = compute_metrics(estimated, ground_truth)
        rows.append({"run": estimated_path.parent.name, "trajectory": str(estimated_path), **metrics})
    return rows


def write_metrics_csv(path: Path, rows: list[dict[str, float | str]]) -> Path:
    """Write a metrics table to CSV."""

    if not rows:
        raise ValueError("Cannot write an empty metrics table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_metrics_markdown(path: Path, rows: list[dict[str, float | str]]) -> Path:
    """Write a compact Markdown metrics summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ATE Metrics",
        "",
        "| run | raw ATE m | SE3 ATE m | Sim3 ATE m | Sim3 scale | raw rotation deg |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {raw:.6f} | {se3:.6f} | {sim3:.6f} | {scale:.9f} | {rot:.6f} |".format(
                run=row["run"],
                raw=row["raw_ate_rmse_m"],
                se3=row["se3_aligned_ate_rmse_m"],
                sim3=row["sim3_aligned_scaled_ate_rmse_m"],
                scale=row["sim3_scale"],
                rot=row["raw_rotation_rmse_deg"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ATE metrics for one or more trajectories.")
    parser.add_argument("--ground_truth", type=Path, required=True, help="Ground-truth trajectory table.")
    parser.add_argument(
        "--estimated",
        type=Path,
        nargs="+",
        required=True,
        help="Estimated trajectory table(s).",
    )
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional output CSV path.")
    parser.add_argument("--output_md", type=Path, default=None, help="Optional output Markdown path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compute_metrics_for_paths(args.estimated, args.ground_truth)
    for row in rows:
        print(
            "{run}: raw={raw:.6f} se3={se3:.6f} sim3={sim3:.6f} scale={scale:.9f} rot={rot:.6f}".format(
                run=row["run"],
                raw=row["raw_ate_rmse_m"],
                se3=row["se3_aligned_ate_rmse_m"],
                sim3=row["sim3_aligned_scaled_ate_rmse_m"],
                scale=row["sim3_scale"],
                rot=row["raw_rotation_rmse_deg"],
            )
        )
    if args.output_csv is not None:
        write_metrics_csv(args.output_csv, rows)
    if args.output_md is not None:
        write_metrics_markdown(args.output_md, rows)


if __name__ == "__main__":
    main()
