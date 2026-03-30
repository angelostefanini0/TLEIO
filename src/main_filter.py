"""Run end-to-end TLEIO inference with the transformer and the EKF.

This file is the filter-branch inference entrypoint. It loads a trained
transformer checkpoint, builds the processed event dataset, steps through each
sequence in timestamp order, propagates IMU data between anchor times, fuses the
transformer's `2 x 7` relative-pose output with the EKF, and returns the final
trajectory estimate for every processed sequence.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from filter.imu_buffer import ImuMeasurement
from filter.scekf import ImuMSCKF
from learning.dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset
from learning.network.build_model import build_model

log = logging.getLogger(__name__)


def get_parser():
    """Build the command-line interface for the TLEIO filter runner."""

    parser = argparse.ArgumentParser(description="TLEIO EKF filter runner")

    # Data
    parser.add_argument("--data_dir",      type=str, required=True)
    parser.add_argument("--out_dir",        type=str, default="output")
    parser.add_argument("--dataset",        type=str, default="tleio")

    # Filter
    parser.add_argument("--sigma_na",       type=float, default=0.01)
    parser.add_argument("--sigma_ng",       type=float, default=0.001)
    parser.add_argument("--sigma_nba",      type=float, default=1e-4)
    parser.add_argument("--sigma_nbg",      type=float, default=1e-5)

    # Network (Phase 2)
    parser.add_argument("--model_path",     type=str,   default=None,
                        help="Path to trained TLEIO network checkpoint")
    parser.add_argument("--window_time",    type=float, default=1.0,
                        help="Duration (s) of each event+IMU triplet window")
    parser.add_argument("--sequence",       type=str, default=None,
                        help="Optional processed sequence folder name to run alone.")
    parser.add_argument("--sigma_rel_t",    type=float, default=0.10,
                        help="Fallback translation measurement sigma when the model does not regress covariance.")
    parser.add_argument("--sigma_rel_r",    type=float, default=0.10,
                        help="Fallback rotation measurement sigma [rad] when the model does not regress covariance.")
    parser.add_argument("--meas_cov_scale", type=float, default=1.0,
                        help="Global scale applied to the transformer's joint measurement covariance.")

    # Misc
    parser.add_argument("--cpu",            action="store_true")
    parser.add_argument("--verbose",        action="store_true")
    return parser


def _resolve_checkpoint_paths(model_path):
    """Resolve a checkpoint file and its companion config directory."""

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if model_path.is_dir():
        candidates = [
            model_path / "checkpoint_best.pth",
            model_path / "checkpoint_last.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                checkpoint_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Could not find a checkpoint in directory {model_path}. Expected checkpoint_best.pth or checkpoint_last.pth."
            )
        config_dir = model_path
    else:
        checkpoint_path = model_path
        config_dir = model_path.parent

    return checkpoint_path, config_dir


def _load_transformer_config(checkpoint_path, config_dir):
    """Load the transformer config and target normalization statistics."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args_pkl = config_dir / "args.pkl"

    if args_pkl.exists():
        with open(args_pkl, "rb") as f:
            model_args = pickle.load(f)
        model_params = model_args["model_params"]
    else:
        if "args" not in checkpoint or "model_params" not in checkpoint:
            raise FileNotFoundError(
                f"Could not recover transformer configuration from {checkpoint_path}. "
                "Expected args.pkl next to the checkpoint or `args`/`model_params` inside the checkpoint."
            )
        model_args = checkpoint["args"]
        model_params = checkpoint["model_params"]
        model_args["model_params"] = model_params

    target_mean = checkpoint.get("target_mean", None)
    target_std = checkpoint.get("target_std", None)
    if target_mean is not None:
        target_mean = np.asarray(target_mean, dtype=np.float32)
    if target_std is not None:
        target_std = np.asarray(target_std, dtype=np.float32)

    return model_args, model_params, target_mean, target_std


def _load_transformer_model(model_path, device):
    """Load the trained transformer and its normalization metadata for inference."""

    checkpoint_path, config_dir = _resolve_checkpoint_paths(model_path)
    model_args, model_params, target_mean, target_std = _load_transformer_config(
        checkpoint_path, config_dir
    )

    build_args = dict(model_args)
    build_args["checkpoint_path"] = str(config_dir)
    build_args["checkpoint"] = checkpoint_path.name

    model, _ = build_model(build_args, model_params)
    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "config_dir": config_dir,
        "args": model_args,
        "model_params": model_params,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def _build_processed_dataset(dataset_root, transformer_bundle):
    """Build the processed event dataset using the transformer's training config."""

    model_args = transformer_bundle["args"]
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Processed dataset root does not exist: {dataset_root}")

    dataset = MultiEventVoxelClipDataset(
        root_path=dataset_root,
        delta_t_ms=int(model_args["delta_t_ms"]),
        num_bins=int(model_args["num_bins"]),
        clip_len=int(model_args["clip_len"]),
    )
    if len(dataset.seq_infos) == 0:
        raise ValueError(
            f"No valid processed sequences were found under {dataset_root}. "
            "Expected each sequence folder to contain events.h5, anchor_poses.txt, and relative_motions.txt."
        )
    return dataset


def _load_anchor_poses(sequence_path):
    """Load anchor timestamps, positions, and quaternions for one processed sequence."""

    anchor_path = Path(sequence_path) / "anchor_poses.txt"
    anchor_table = np.atleast_2d(np.loadtxt(anchor_path, dtype=np.float64, skiprows=1))
    if anchor_table.shape[1] != 8:
        raise ValueError(
            f"{anchor_path} has {anchor_table.shape[1]} columns, expected 8: timestamp px py pz qx qy qz qw."
        )

    timestamps_us = anchor_table[:, 0].astype(np.int64)
    positions = anchor_table[:, 1:4]
    quaternions = anchor_table[:, 4:8]
    return timestamps_us, positions, quaternions


def _load_sequence_imu(sequence_path):
    """Load one processed sequence IMU table with a robust delimiter/header parser."""

    imu_path = Path(sequence_path) / "imu.csv"
    if not imu_path.exists():
        raise FileNotFoundError(f"Missing IMU file for sequence: {imu_path}")

    imu = np.loadtxt(imu_path, delimiter=",", comments="#", ndmin=2)
    if imu.shape[1] != 7:
        raise ValueError(
            f"{imu_path} has {imu.shape[1]} columns, expected 7: timestamp gx gy gz ax ay az."
        )
    imu = imu[np.argsort(imu[:, 0])]
    return imu


def _imu_time_scale_to_seconds(timestamps):
    """Infer whether IMU timestamps are stored in seconds, microseconds, or nanoseconds.

    The processed dataset used by the filter branch stores IMU timestamps in
    microseconds, while raw inputs may still appear in nanoseconds. Looking at
    the median positive time increment is more reliable than the absolute time
    magnitude because processed timestamps are often re-zeroed per sequence.
    """

    timestamps = np.asarray(timestamps, dtype=np.float64)
    positive_diffs = np.diff(timestamps)
    positive_diffs = positive_diffs[positive_diffs > 0]
    median_dt = float(np.median(positive_diffs)) if len(positive_diffs) > 0 else 0.0

    if median_dt > 1e5:
        return 1e-9
    if median_dt > 1e1:
        return 1e-6
    return 1.0


def _build_exact_imu_segment(raw_times_s, raw_gyro, raw_accel, start_time_s, end_time_s):
    """Resample the raw IMU stream so propagation lands exactly on `end_time_s`.

    The processed sequence anchors almost never coincide exactly with raw IMU
    timestamps, so the runner needs one interpolated terminal sample per anchor
    interval to avoid cloning slightly stale states.
    """

    if end_time_s <= start_time_s:
        return []

    if start_time_s < raw_times_s[0] or end_time_s > raw_times_s[-1]:
        raise ValueError(
            "Requested IMU propagation interval falls outside the available IMU time range."
        )

    interior_mask = (raw_times_s > start_time_s) & (raw_times_s < end_time_s)
    segment_times = list(raw_times_s[interior_mask])
    segment_times.append(float(end_time_s))

    gyro_interp = np.column_stack(
        [np.interp(segment_times, raw_times_s, raw_gyro[:, axis]) for axis in range(3)]
    )
    accel_interp = np.column_stack(
        [np.interp(segment_times, raw_times_s, raw_accel[:, axis]) for axis in range(3)]
    )

    measurements = []
    prev_time_s = float(start_time_s)
    for sample_idx, timestamp_s in enumerate(segment_times):
        timestamp_s = float(timestamp_s)
        measurements.append(
            ImuMeasurement(
                timestamp=timestamp_s,
                dt=max(timestamp_s - prev_time_s, 0.0),
                accel=accel_interp[sample_idx].astype(np.float64),
                gyro=gyro_interp[sample_idx].astype(np.float64),
            )
        )
        prev_time_s = timestamp_s

    return measurements


def _build_anchor_imu_segments(imu_table, anchor_timestamps_us):
    """Precompute one exact IMU propagation segment for each consecutive anchor pair."""

    time_scale = _imu_time_scale_to_seconds(imu_table[:, 0])
    raw_times_s = imu_table[:, 0].astype(np.float64) * time_scale
    raw_gyro = imu_table[:, 1:4].astype(np.float64)
    raw_accel = imu_table[:, 4:7].astype(np.float64)
    anchor_times_s = anchor_timestamps_us.astype(np.float64) * 1e-6

    if len(anchor_times_s) == 0:
        return []
    if anchor_times_s[0] < raw_times_s[0] or anchor_times_s[-1] > raw_times_s[-1]:
        raise ValueError(
            "Anchor timestamps fall outside the IMU stream. The filter runner cannot propagate the full sequence."
        )

    segments = []
    for idx in range(len(anchor_times_s) - 1):
        segments.append(
            _build_exact_imu_segment(
                raw_times_s,
                raw_gyro,
                raw_accel,
                anchor_times_s[idx],
                anchor_times_s[idx + 1],
            )
        )
    return segments


def _make_initial_filter_state(anchor_timestamps_us, anchor_positions, anchor_quaternions):
    """Initialize the EKF from the first anchor pose and a finite-difference velocity."""

    p0 = anchor_positions[0].astype(np.float64)
    R0 = Rotation.from_quat(anchor_quaternions[0]).as_matrix()

    if len(anchor_timestamps_us) >= 2:
        dt = max((anchor_timestamps_us[1] - anchor_timestamps_us[0]) * 1e-6, 1e-9)
        v0 = (anchor_positions[1] - anchor_positions[0]) / dt
    else:
        v0 = np.zeros(3, dtype=np.float64)

    bg0 = np.zeros(3, dtype=np.float64)
    ba0 = np.zeros(3, dtype=np.float64)
    t0 = anchor_timestamps_us[0] * 1e-6
    return t0, R0, v0, p0, bg0, ba0


def _denormalize_transformer_output(prediction_2x7, transformer_bundle):
    """Undo target normalization using the stats saved in the transformer checkpoint."""

    target_mean = transformer_bundle["target_mean"]
    target_std = transformer_bundle["target_std"]

    if target_mean is None or target_std is None:
        return prediction_2x7

    return prediction_2x7 * target_std.reshape(1, 7) + target_mean.reshape(1, 7)


def _predict_network_output(model, representation, transformer_bundle):
    """Run the transformer on one clip and package its output for the EKF."""

    device = transformer_bundle["device"]
    with torch.no_grad():
        batch = representation.unsqueeze(0).to(device).float()
        prediction = model(batch)

    joint_covariance = None
    if isinstance(prediction, dict):
        mean = prediction["mean"] if "mean" in prediction else prediction["relative_pose"]
        if "joint_covariance" in prediction:
            joint_covariance = prediction["joint_covariance"]
    elif isinstance(prediction, (tuple, list)):
        mean = prediction[0]
        if len(prediction) > 1:
            joint_covariance = prediction[1]
    else:
        mean = prediction

    mean = mean.detach().cpu().numpy().reshape(2, 7).astype(np.float32)
    mean = _denormalize_transformer_output(mean, transformer_bundle)

    output = {"relative_pose": mean}
    if joint_covariance is not None:
        joint_covariance = joint_covariance.detach().cpu().numpy()
        output["joint_covariance"] = joint_covariance.reshape(12, 12)
    return output


def _state_to_result_entry(timestamp_s, ekf_state):
    """Convert the EKF nominal state into a trajectory record for logging and return."""

    quaternion_xyzw = Rotation.from_matrix(ekf_state.R).as_quat()
    return {
        "t": float(timestamp_s),
        "p": ekf_state.p.copy(),
        "R": ekf_state.R.copy(),
        "q": quaternion_xyzw.copy(),
        "v": ekf_state.v.copy(),
        "bg": ekf_state.bg.copy(),
        "ba": ekf_state.ba.copy(),
    }


def _save_sequence_trajectory(sequence_name, results, out_dir):
    """Save one sequence trajectory in a simple timestamp-position text format."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"traj_estimate_{sequence_name}.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        for result in results:
            p = result["p"]
            f.write(f"{result['t']:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return out_file


def _run_sequence_filter(sequence_name, seq_idx, dataset, transformer_bundle, args):
    """Run transformer+EKF inference on one processed sequence from start to finish."""

    seq_info = dataset.seq_infos[seq_idx]
    seq_path = Path(seq_info["seq_path"])
    anchor_timestamps_us, anchor_positions, anchor_quaternions = _load_anchor_poses(seq_path)
    imu_table = _load_sequence_imu(seq_path)
    anchor_imu_segments = _build_anchor_imu_segments(imu_table, anchor_timestamps_us)

    t0, R0, v0, p0, bg0, ba0 = _make_initial_filter_state(
        anchor_timestamps_us, anchor_positions, anchor_quaternions
    )

    ekf = ImuMSCKF(args)
    ekf.initialize_with_state(t0, R0, v0, p0, bg0, ba0)

    global_start_idx = 0 if seq_idx == 0 else dataset.cum_lengths[seq_idx - 1]
    results = [_state_to_result_entry(t0, ekf.state)]
    ekf.augment_clone()

    if len(anchor_timestamps_us) >= 2:
        ekf.propagate(anchor_imu_segments[0])
        results.append(_state_to_result_entry(anchor_timestamps_us[1] * 1e-6, ekf.state))
        ekf.augment_clone()

    for local_idx in range(seq_info["num_samples"]):
        sample = dataset[global_start_idx + local_idx]
        anchors_us = sample["anchors_us"].cpu().numpy().astype(np.int64)
        clip = sample["representation"]

        expected_anchors = anchor_timestamps_us[local_idx : local_idx + 3]
        if not np.array_equal(anchors_us, expected_anchors):
            raise ValueError(
                "Dataset clip anchors do not match the processed anchor sequence used by the filter runner."
            )

        # The runner starts with clones at t0 and t1, so each clip only needs the
        # final interval propagation to reach its third anchor exactly.
        interval_idx = local_idx + 1
        ekf.propagate(anchor_imu_segments[interval_idx])
        ekf.augment_clone()

        network_output = _predict_network_output(
            transformer_bundle["model"], clip, transformer_bundle
        )
        ekf.update(network_output)
        results.append(_state_to_result_entry(anchors_us[2] * 1e-6, ekf.state))
        ekf.marginalize_oldest_clone()

    return results


def run_filter(args):
    """Run the full transformer+EKF inference pipeline on the processed dataset."""

    if args.model_path is None:
        raise ValueError("A trained transformer checkpoint is required. Pass it via --model_path.")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    transformer_bundle = _load_transformer_model(args.model_path, device)
    dataset = _build_processed_dataset(args.data_dir, transformer_bundle)

    all_results = {}
    try:
        for seq_idx, seq_info in enumerate(dataset.seq_infos):
            sequence_name = Path(seq_info["seq_path"]).name
            if args.sequence is not None and sequence_name != args.sequence:
                continue

            log.info("Running transformer+EKF inference on sequence %s", sequence_name)
            sequence_results = _run_sequence_filter(
                sequence_name, seq_idx, dataset, transformer_bundle, args
            )
            all_results[sequence_name] = sequence_results
            out_file = _save_sequence_trajectory(sequence_name, sequence_results, args.out_dir)
            log.info("Saved trajectory for %s to %s", sequence_name, out_file)
    finally:
        dataset.close()

    if args.sequence is not None and args.sequence not in all_results:
        raise ValueError(
            f"Requested sequence `{args.sequence}` was not found under processed dataset root {args.data_dir}."
        )

    return all_results


if __name__ == "__main__":
    parser = get_parser()
    args   = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run_filter(args)
