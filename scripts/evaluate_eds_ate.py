#!/usr/bin/env python3
"""Evaluate EDS relative-motion predictions directly against raw EDS ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ATE for every prediction folder without running the EKF."
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=ROOT / "data" / "eds" / "predicted_relative_motions",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data" / "eds" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ate_results_eds" / "summary.json",
    )
    parser.add_argument(
        "--alignment",
        choices=("se3", "sim3", "none"),
        default="se3",
        help="Trajectory alignment before computing ATE (default: se3, no scale correction).",
    )
    parser.add_argument(
        "--timestamps-key",
        default="t",
        help="Timestamp dataset in raw events.h5 (default: t).",
    )
    return parser.parse_args()


def load_numeric_table(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append([float(value) for value in line.split()])
            except ValueError:
                continue
    table = np.asarray(rows, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 5:
        raise ValueError(f"{path} must contain at least t0 t1 dx dy dz.")
    return table


def infer_seconds_scale(timestamps: np.ndarray) -> float:
    positive_diffs = np.diff(np.unique(np.asarray(timestamps, dtype=np.float64)))
    positive_diffs = positive_diffs[positive_diffs > 0]
    median_dt = float(np.median(positive_diffs)) if len(positive_diffs) else 0.0
    if median_dt > 1e7:
        return 1e-9
    if median_dt > 10:
        return 1e-6
    return 1.0


def read_event_t0_seconds(events_path: Path, timestamps_key: str) -> float:
    with h5py.File(events_path, "r") as handle:
        key = timestamps_key
        if key not in handle and f"events/{key}" in handle:
            key = f"events/{key}"
        if key not in handle:
            raise KeyError(f"Timestamp dataset '{timestamps_key}' not found in {events_path}.")
        timestamps = handle[key]
        if len(timestamps) == 0:
            raise ValueError(f"No event timestamps in {events_path}.")
        return float(timestamps[0]) * infer_seconds_scale(np.asarray(timestamps[: min(100, len(timestamps))]))


def load_gt_at_anchors(
    sequence_dir: Path,
    anchor_times_s: np.ndarray,
    timestamps_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    gt = np.loadtxt(sequence_dir / "stamped_groundtruth.txt", ndmin=2)
    if gt.shape[1] < 8:
        raise ValueError("Ground truth must contain timestamp, position, and quaternion.")

    gt_times_s = gt[:, 0] * infer_seconds_scale(gt[:, 0])
    if gt_times_s[0] > 1e6 and anchor_times_s[-1] < 1e6:
        event_t0_s = read_event_t0_seconds(sequence_dir / "events.h5", timestamps_key)
        gt_times_s = gt_times_s - event_t0_s

    if anchor_times_s[0] < gt_times_s[0] or anchor_times_s[-1] > gt_times_s[-1]:
        raise ValueError(
            f"Prediction times [{anchor_times_s[0]:.6f}, {anchor_times_s[-1]:.6f}] "
            f"fall outside GT [{gt_times_s[0]:.6f}, {gt_times_s[-1]:.6f}]."
        )

    positions = np.column_stack(
        [np.interp(anchor_times_s, gt_times_s, gt[:, axis]) for axis in range(1, 4)]
    )
    rotations = Rotation.from_quat(gt[:, 4:8])
    quaternions = Slerp(gt_times_s, rotations)(anchor_times_s).as_quat()
    return positions, quaternions


def reconstruct_positions(
    gt_positions: np.ndarray,
    gt_quaternions: np.ndarray,
    relative_translations: np.ndarray,
) -> np.ndarray:
    estimated = np.empty_like(gt_positions)
    estimated[0] = gt_positions[0]
    rotations = Rotation.from_quat(gt_quaternions).as_matrix()
    for index, delta_body in enumerate(relative_translations):
        estimated[index + 1] = estimated[index] + rotations[index] @ delta_body
    return estimated


def align_positions(
    reference: np.ndarray,
    estimate: np.ndarray,
    alignment: str,
) -> tuple[np.ndarray, float]:
    if alignment == "none":
        return estimate.copy(), 1.0

    reference_mean = reference.mean(axis=0)
    estimate_mean = estimate.mean(axis=0)
    ref_centered = reference - reference_mean
    est_centered = estimate - estimate_mean
    covariance = ref_centered.T @ est_centered / len(reference)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt

    scale = 1.0
    if alignment == "sim3":
        variance = np.mean(np.sum(est_centered**2, axis=1))
        if variance <= 0:
            raise ValueError("Cannot estimate Sim(3) scale from a constant trajectory.")
        scale = float(np.sum(singular_values * sign) / variance)

    translation = reference_mean - scale * (rotation @ estimate_mean)
    aligned = (scale * (rotation @ estimate.T)).T + translation
    return aligned, scale


def evaluate_file(
    prediction_path: Path,
    raw_root: Path,
    alignment: str,
    timestamps_key: str,
) -> dict:
    prediction = load_numeric_table(prediction_path)
    time_scale = infer_seconds_scale(prediction[:, :2].reshape(-1))
    starts = prediction[:, 0] * time_scale
    ends = prediction[:, 1] * time_scale
    if np.any(ends <= starts):
        raise ValueError("Prediction contains non-positive time intervals.")
    if len(prediction) > 1 and np.max(np.abs(ends[:-1] - starts[1:])) > 1e-6:
        raise ValueError("Prediction intervals are not continuous.")

    anchor_times_s = np.concatenate([starts[:1], ends])
    sequence_dir = raw_root / prediction_path.stem
    gt_positions, gt_quaternions = load_gt_at_anchors(
        sequence_dir, anchor_times_s, timestamps_key
    )
    estimated = reconstruct_positions(
        gt_positions, gt_quaternions, prediction[:, 2:5]
    )
    aligned, scale = align_positions(gt_positions, estimated, alignment)
    errors = np.linalg.norm(aligned - gt_positions, axis=1)
    return {
        "sequence": prediction_path.stem,
        "num_poses": int(len(errors)),
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "ate_mean_m": float(np.mean(errors)),
        "ate_median_m": float(np.median(errors)),
        "ate_max_m": float(np.max(errors)),
        "alignment_scale": scale,
    }


def main() -> None:
    args = parse_args()
    model_dirs = sorted(path for path in args.pred_root.glob("EDS_*") if path.is_dir())
    if not model_dirs:
        raise FileNotFoundError(f"No EDS prediction directories found under {args.pred_root}")

    summary = {"alignment": args.alignment, "models": [], "failures": []}
    for model_dir in model_dirs:
        rows = []
        print(f"\n{model_dir.name}")
        for prediction_path in sorted(model_dir.glob("*.txt")):
            try:
                row = evaluate_file(
                    prediction_path,
                    args.raw_root,
                    args.alignment,
                    args.timestamps_key,
                )
                rows.append(row)
                print(f"  {row['sequence']:<28} ATE {row['ate_rmse_m']:.6f} m")
            except Exception as exc:
                failure = {
                    "model": model_dir.name,
                    "sequence": prediction_path.stem,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                summary["failures"].append(failure)
                print(f"  {prediction_path.stem:<28} FAILED: {failure['error']}")

        if rows:
            average = float(np.mean([row["ate_rmse_m"] for row in rows]))
            summary["models"].append(
                {
                    "model": model_dir.name,
                    "num_sequences": len(rows),
                    "average_ate_rmse_m": average,
                    "sequences": rows,
                }
            )
            print(f"  {'AVERAGE':<28} ATE {average:.6f} m")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.output}")

    if not summary["models"]:
        raise RuntimeError("No prediction file completed successfully.")


if __name__ == "__main__":
    main()
