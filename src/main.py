from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import yaml
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from filter.imu_buffer import ImuMeasurement
from filter.measurement_triplet import make_default_joint_covariance, predict_relative_pose
from filter.scekf import ImuMSCKF
from filter_diagnostics import compute_filter_diagnostics
from scripts.testing.test import get_outputs_per_motion, load_inference_args, load_target_stats
from scripts.utils.gt_training import (
    compute_relative_motions,
    get_anchor_grid,
    interpolate_gt_to_anchors,
    load_gt,
)
from src.learning.dataloader.events_to_voxel.precomputed_voxel_clip import (
    normalize_nonzero_voxel_,
)
from src.learning.dataloader.events_to_voxel.raw_to_clip import (
    MultiEventVoxelClipDataset,
)
from src.learning.dataloader.events_to_voxel.utils import (
    build_camera_matrix,
    build_derotation_context,
    scale_camera_matrix,
)
from src.learning.dataloader.representation.event_denoising import (
    background_activity_filter_raw,
)
from src.learning.dataloader.representation.voxel_grid import VoxelGrid
from src.learning.network.build_model import build_model, normalize_checkpoint_state_dict


@dataclass
class OnlineConfig:
    raw_sequence_dir: Path
    checkpoint_file: Path
    output_dir: Path
    delta_t_ms: int = 50
    anchor_t_ms: int = 50
    event_time_divisor: int = 1000
    imu_rate_hz: float = 200.0
    gravity_world_mps2: tuple[float, float, float] = (0.0, 0.0, 9.80665)
    use_network_covariance: bool = False
    scale_mode: str = "none"
    derotation_source: str = "filter"
    scale_init: float = 1.0
    scale_alpha: float = 0.01
    scale_min: float = 0.3
    scale_max: float = 2.0
    max_anchors: int | None = None
    plot_projections: bool = True
    show_online_visualization: bool = False
    save_online_visualization: bool = False
    serve_online_visualization: bool = False
    viz_port: int = 8765
    viz_stride: int = 25
    viz_max_events: int = 20000

    sigma_na: float = 0.011065875226523246
    sigma_ng: float = 0.01251528557615725
    sigma_nba: float = 6.536078678232154e-05
    sigma_nbg: float = 2.1514640261497524e-05
    assumed_sigma_rel_t: float = 0.02194332115673975
    meas_cov_scale: float = 1.2649054158337365
    initial_attitude_sigma_deg: float = 0.11534784349262132
    initial_velocity_sigma_mps: float = 1.8658950002457901
    initial_position_sigma_m: float = 0.04181564546764053
    initial_z_sigma_m: float = 0.006867502596918262
    initial_bg_sigma_rps: float = 0.00033573143221825514
    initial_ba_sigma_mps2: float = 0.1779266257977154


class RawTartanEventSlicer:
    def __init__(self, events_file: Path, timestamps_key: str, time_divisor: int):
        self.events_file = events_file
        self.h5f = h5py.File(events_file, "r")
        self.x_ds = self._dataset("x")
        self.y_ds = self._dataset("y")
        self.p_ds = self._dataset("p")
        self.t_ds = self._dataset_by_path(timestamps_key)
        raw_t = np.asarray(self.t_ds, dtype=np.int64)
        t_us = raw_t // int(time_divisor)
        np.maximum.accumulate(t_us, out=t_us)
        self.t0_us = int(t_us[0])
        self.t_us = t_us - self.t0_us
        self.t_final_us = int(self.t_us[-1])

    def _dataset_by_path(self, key: str):
        if key in self.h5f:
            return self.h5f[key]
        if not key.startswith("events/") and f"events/{key}" in self.h5f:
            return self.h5f[f"events/{key}"]
        raise KeyError(f"{self.events_file}: missing dataset '{key}'")

    def _dataset(self, key: str):
        return self._dataset_by_path(f"events/{key}")

    def get_events(self, t_start_us: int, t_end_us: int) -> dict[str, np.ndarray] | None:
        if t_start_us < 0 or t_end_us <= t_start_us or t_start_us > self.t_final_us:
            return None
        start = int(np.searchsorted(self.t_us, t_start_us, side="left"))
        end = int(np.searchsorted(self.t_us, t_end_us, side="left"))
        if end <= start:
            return {
                "x": np.empty(0, dtype=np.float32),
                "y": np.empty(0, dtype=np.float32),
                "p": np.empty(0, dtype=np.uint8),
                "t": np.empty(0, dtype=np.int64),
            }

        p = np.asarray(self.p_ds[start:end])
        if p.size and np.min(p) < 0:
            p = ((p.astype(np.int8) + 1) // 2).astype(np.uint8)

        return {
            "x": np.asarray(self.x_ds[start:end]),
            "y": np.asarray(self.y_ds[start:end]),
            "p": p,
            "t": self.t_us[start:end].astype(np.int64, copy=False),
        }

    def close(self) -> None:
        self.h5f.close()


def load_raw_tartan_gt(raw_sequence_dir: Path, t0_us: int, t_end_us: int):
    pose_path = raw_sequence_dir / "pose_lcam_front.txt"
    time_path = raw_sequence_dir / "imu" / "cam_time.txt"
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing pose file: {pose_path}")
    if not time_path.exists():
        raise FileNotFoundError(f"Missing camera time file: {time_path}")

    pose = np.loadtxt(pose_path, dtype=np.float64, ndmin=2)
    cam_time_s = np.loadtxt(time_path, dtype=np.float64, ndmin=1)
    if len(cam_time_s) != len(pose):
        raise ValueError(f"{time_path} and {pose_path} have different row counts.")

    stamped = np.column_stack([cam_time_s, pose])
    gt_t_us = stamped[:, 0] * 1e6 - int(t0_us)
    keep = (gt_t_us >= 0.0) & (gt_t_us <= int(t_end_us))
    gt = stamped[keep].copy()
    gt[:, 0] = np.rint(gt_t_us[keep]).astype(np.int64)
    if len(gt) < 4:
        raise ValueError("Not enough GT poses after cropping to event time range.")

    ts, pos, quat = load_gt(gt)
    return gt, ts, pos, quat


def make_anchors(gt_ts_us, gt_pos, gt_quat, delta_t_ms: int, anchor_t_ms: int):
    anchors_us = get_anchor_grid(
        gt_timestamps_us=gt_ts_us,
        delta_t_us=int(round(delta_t_ms * 1000.0)),
        anchor_step_us=int(round(anchor_t_ms * 1000.0)),
    )
    anchor_pos, anchor_quat = interpolate_gt_to_anchors(gt_ts_us, gt_pos, gt_quat, anchors_us)
    gt_rel = compute_relative_motions(anchors_us, anchor_pos, anchor_quat)
    return anchors_us, anchor_pos, anchor_quat, gt_rel


def generate_synthetic_imu(gt_ts_us, gt_pos, gt_quat, rate_hz: float, gravity_world):
    gt_times_s = gt_ts_us.astype(np.float64) * 1e-6
    dt_s = 1.0 / float(rate_hz)
    count = int(np.floor((gt_times_s[-1] - gt_times_s[0]) / dt_s)) + 1
    query_times_s = gt_times_s[0] + np.arange(count, dtype=np.float64) * dt_s
    query_times_us = np.rint(query_times_s * 1e6).astype(np.int64)

    accel_world = np.empty((len(query_times_s), 3), dtype=np.float64)
    for axis in range(3):
        spline = CubicSpline(gt_times_s, gt_pos[:, axis], bc_type="not-a-knot")
        accel_world[:, axis] = spline(query_times_s, 2)

    rotations_gt = Rotation.from_quat(gt_quat)
    try:
        rot_spline = RotationSpline(gt_times_s, rotations_gt)
        rotations = rot_spline(query_times_s)
        gyro_body = rot_spline(query_times_s, order=1)
    except Exception:
        slerp = Slerp(gt_times_s, rotations_gt)
        rotations = slerp(query_times_s)
        mats = rotations.as_matrix()
        gyro_body = np.empty((len(query_times_s), 3), dtype=np.float64)
        for idx in range(len(query_times_s)):
            left = max(0, idx - 1)
            right = min(len(query_times_s) - 1, idx + 1)
            dt = max(query_times_s[right] - query_times_s[left], 1e-9)
            gyro_body[idx] = Rotation.from_matrix(mats[left].T @ mats[right]).as_rotvec() / dt

    specific_force_world = accel_world - np.asarray(gravity_world, dtype=np.float64)
    accel_body = rotations.inv().apply(specific_force_world)
    return np.column_stack([query_times_us, gyro_body, accel_body])


def build_exact_imu_segment(raw_times_s, raw_gyro, raw_accel, start_time_s, end_time_s):
    if end_time_s <= start_time_s:
        return []
    interior = (raw_times_s > start_time_s) & (raw_times_s < end_time_s)
    segment_times = list(raw_times_s[interior])
    segment_times.append(float(end_time_s))
    gyro = np.column_stack([np.interp(segment_times, raw_times_s, raw_gyro[:, i]) for i in range(3)])
    accel = np.column_stack([np.interp(segment_times, raw_times_s, raw_accel[:, i]) for i in range(3)])

    measurements = []
    prev = float(start_time_s)
    for idx, timestamp_s in enumerate(segment_times):
        measurements.append(
            ImuMeasurement(
                timestamp=float(timestamp_s),
                dt=max(float(timestamp_s) - prev, 0.0),
                accel=accel[idx].astype(np.float64),
                gyro=gyro[idx].astype(np.float64),
            )
        )
        prev = float(timestamp_s)
    return measurements


def build_anchor_imu_segments(imu_table, anchors_us):
    raw_times_s = imu_table[:, 0].astype(np.float64) * 1e-6
    raw_gyro = imu_table[:, 1:4].astype(np.float64)
    raw_accel = imu_table[:, 4:7].astype(np.float64)
    anchor_times_s = anchors_us.astype(np.float64) * 1e-6
    if anchor_times_s[0] < raw_times_s[0] or anchor_times_s[-1] > raw_times_s[-1]:
        raise ValueError("Anchor timestamps fall outside generated IMU stream.")
    return [
        build_exact_imu_segment(raw_times_s, raw_gyro, raw_accel, anchor_times_s[i], anchor_times_s[i + 1])
        for i in range(len(anchor_times_s) - 1)
    ]


class OnlineVoxelizer:
    def __init__(self, raw_sequence_dir: Path, infer_args: dict, gt_ts_us, gt_quat):
        self.delta_t_us = int(round(float(infer_args["delta_t_ms"]) * 1000.0))
        self.num_bins = int(infer_args["num_bins"])
        self.downsampling_factor = float(infer_args["downsampling_factor"])
        self.patch_size = int(infer_args["model_params"]["patch_size"])
        self.denoising = bool(infer_args.get("denoising", False))
        self.denoise_dt_us = int(infer_args.get("denoise_dt_us", 1000))
        self.denoise_radius = int(infer_args.get("denoise_radius", 1))
        self.denoise_min_supporters = int(infer_args.get("denoise_min_supporters", 1))
        self.denoise_same_polarity_only = bool(infer_args.get("denoise_same_polarity_only", False))
        self.derotate = bool(infer_args.get("derotate", False))
        self.derotation_slices = int(infer_args.get("derotation_slices", 100))
        self.normalize_voxel_nonzero = bool(infer_args.get("normalize_voxel_nonzero", False))

        k_path = raw_sequence_dir / "K.yaml"
        if k_path.exists():
            with k_path.open("r") as fh:
                calibration = yaml.safe_load(fh)
            cam = calibration.get("cam1") or calibration.get("cam0")
            if cam is None:
                raise KeyError(f"{k_path}: missing cam1/cam0 calibration block.")
        else:
            print(f"Missing {k_path}; using default TartanAir pinhole intrinsics.")
            cam = {"intrinsics": [320.0, 320.0, 320.0, 240.0], "resolution": [640, 480]}
        width, height = cam.get("resolution", [640, 480])
        self.original_width = int(width)
        self.original_height = int(height)
        self.new_height, self.new_width = MultiEventVoxelClipDataset.get_downsampled_size(
            self.original_height,
            self.original_width,
            self.downsampling_factor,
            self.patch_size,
        )
        self.scale_x = self.new_width / self.original_width
        self.scale_y = self.new_height / self.original_height
        camera_matrix = scale_camera_matrix(
            build_camera_matrix(cam["intrinsics"]),
            self.scale_x,
            self.scale_y,
        )

        self.seq_info = {
            "gt_timestamps_us": gt_ts_us.astype(np.int64),
            "gt_quat_xyzw": gt_quat.astype(np.float64),
            "camera_matrix": camera_matrix,
        }
        self.voxel_grid = VoxelGrid(
            self.num_bins,
            self.new_height,
            self.new_width,
            derotate=self.derotate,
            derotation_slices=self.derotation_slices,
        )

    def empty_voxel(self):
        return torch.zeros((self.num_bins, self.new_height, self.new_width), dtype=torch.float32)

    def build(
        self,
        events: dict[str, np.ndarray] | None,
        anchor_us: int,
        derotation_context: dict | None = None,
    ):
        if events is None or len(events["t"]) == 0:
            voxel = self.empty_voxel()
        else:
            voxel = self.events_to_voxel_grid(events, anchor_us, derotation_context)
        if self.normalize_voxel_nonzero:
            normalize_nonzero_voxel_(voxel)
        return voxel.float()

    def make_filter_derotation_context(
        self,
        ts_start_us: int,
        ts_end_us: int,
        start_quat_xyzw: np.ndarray,
        end_quat_xyzw: np.ndarray,
    ) -> dict:
        window_duration_us = float(ts_end_us - ts_start_us)
        query = (np.arange(self.derotation_slices, dtype=np.float64) + 0.5) / self.derotation_slices
        key_rots = Rotation.from_quat(np.stack([start_quat_xyzw, end_quat_xyzw], axis=0))
        bin_quat = Slerp([0.0, 1.0], key_rots)(query).as_quat()
        return {
            "window_duration_us": np.float32(window_duration_us),
            "camera_matrix": self.seq_info["camera_matrix"].astype(np.float32),
            "bin_quat_xyzw": bin_quat.astype(np.float32),
            "ref_quat_xyzw": np.asarray(end_quat_xyzw, dtype=np.float32),
        }

    def events_to_voxel_grid(
        self,
        events: dict[str, np.ndarray],
        anchor_us: int,
        derotation_context: dict | None = None,
    ):
        x, y, p, t = events["x"], events["y"], events["p"], events["t"]
        if self.denoising:
            x, y, p, t, _ = background_activity_filter_raw(
                x=x,
                y=y,
                p=p,
                t_us=t,
                height=self.original_height,
                width=self.original_width,
                dt_us=self.denoise_dt_us,
                radius=self.denoise_radius,
                min_supporters=self.denoise_min_supporters,
                same_polarity_only=self.denoise_same_polarity_only,
            )
            if len(t) == 0:
                return self.empty_voxel()

        ts_end_us = int(anchor_us)
        ts_start_us = ts_end_us - self.delta_t_us
        downsampled_x = x * self.scale_x
        downsampled_y = y * self.scale_y

        if self.derotate:
            if derotation_context is None:
                derotation_context = build_derotation_context(
                    seq_info=self.seq_info,
                    ts_start_us=ts_start_us,
                    ts_end_us=ts_end_us,
                    num_bins=self.derotation_slices,
                )
            event_data = {
                "x": downsampled_x.astype(np.float32),
                "y": downsampled_y.astype(np.float32),
                "p": p.astype(np.float32),
                "t": t.astype(np.float64),
                "ts_start_us": ts_start_us,
                "ts_end_us": ts_end_us,
            }
            event_data.update(derotation_context)
        else:
            t = t.astype(np.float32)
            t = t - t[0]
            t = t / t[-1] if t[-1] > 0 else np.zeros_like(t, dtype=np.float32)
            event_data = {
                "x": downsampled_x.astype(np.float32),
                "y": downsampled_y.astype(np.float32),
                "p": p.astype(np.float32),
                "t": t,
            }
        return self.voxel_grid.convert_events(event_data)


class NetworkRunner:
    def __init__(self, checkpoint_file: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.infer_args = load_inference_args(checkpoint_file)
        self.infer_args["device"] = str(self.device)
        self.outputs_per_motion = get_outputs_per_motion(self.infer_args)
        self.clip_len = int(self.infer_args["clip_len"])
        self.inference_count = 0
        self.inference_time_s = 0.0
        self.model, _ = build_model(self.infer_args, self.infer_args["model_params"])

        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.target_mean, self.target_std = load_target_stats(checkpoint, self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(normalize_checkpoint_state_dict(state_dict))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, voxel_window):
        clip = torch.stack(list(voxel_window), dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        with torch.no_grad():
            x = clip.to(self.device).float()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            y_hat = self.model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.inference_time_s += time.perf_counter() - t0
            self.inference_count += 1
            y_hat = y_hat.view(1, self.clip_len - 1, self.outputs_per_motion)
            tr = y_hat[..., :3]
            if self.target_mean is not None and self.target_std is not None:
                tr = tr * self.target_std + self.target_mean
            tr_np = tr[0].cpu().numpy().astype(np.float64)
            sigmas = None
            if self.outputs_per_motion == 6:
                sigmas = np.exp(y_hat[..., 3:][0].cpu().numpy()).astype(np.float64)
        return tr_np, sigmas

    def inference_stats(self):
        mean_s = self.inference_time_s / max(self.inference_count, 1)
        return {
            "count": int(self.inference_count),
            "total_s": float(self.inference_time_s),
            "mean_ms": float(mean_s * 1000.0),
            "hz": float(1.0 / mean_s) if self.inference_count > 0 and mean_s > 0.0 else None,
        }


class OnlineScaleAdapter:
    def __init__(self, mode: str, init: float, alpha: float, scale_min: float, scale_max: float):
        self.mode = mode
        self.scale = float(init)
        self.alpha = float(alpha)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    def update(self, prediction: np.ndarray, reference: np.ndarray | None):
        if self.mode == "none" or reference is None:
            return None
        denom = float(np.sum(prediction * prediction))
        if denom <= 1e-12:
            return None
        candidate = float(np.sum(reference * prediction) / denom)
        candidate = float(np.clip(candidate, self.scale_min, self.scale_max))
        self.scale = (1.0 - self.alpha) * self.scale + self.alpha * candidate
        return candidate

    def apply(self, prediction: np.ndarray, sigmas: np.ndarray | None):
        scaled_prediction = self.scale * prediction
        scaled_sigmas = None if sigmas is None else abs(self.scale) * sigmas
        return scaled_prediction, scaled_sigmas


def make_filter_args(config: OnlineConfig):
    return SimpleNamespace(
        sigma_na=config.sigma_na,
        sigma_ng=config.sigma_ng,
        sigma_nba=config.sigma_nba,
        sigma_nbg=config.sigma_nbg,
        sigma_rel_t=config.assumed_sigma_rel_t,
        meas_cov_scale=config.meas_cov_scale,
        initial_attitude_sigma_rad=float(np.deg2rad(config.initial_attitude_sigma_deg)),
        initial_velocity_sigma_mps=config.initial_velocity_sigma_mps,
        initial_position_sigma_m=config.initial_position_sigma_m,
        initial_z_sigma_m=config.initial_z_sigma_m,
        initial_bg_sigma_rps=config.initial_bg_sigma_rps,
        initial_ba_sigma_mps2=config.initial_ba_sigma_mps2,
    )


def state_to_row(timestamp_s: float, state) -> np.ndarray:
    return np.concatenate(
        [
            np.array([float(timestamp_s)], dtype=np.float64),
            state.p.astype(np.float64),
            Rotation.from_matrix(state.R).as_quat().astype(np.float64),
        ]
    )


def build_ground_truth_trajectory(anchors_us, anchor_pos, anchor_quat):
    return np.column_stack([anchors_us.astype(np.float64) * 1e-6, anchor_pos, anchor_quat])


def clone_relative_predictions(ekf: ImuMSCKF):
    refs = []
    for idx in range(4):
        t_hat, _, _ = predict_relative_pose(
            ekf.state.clone_Rs[idx],
            ekf.state.clone_ps[idx],
            ekf.state.clone_Rs[idx + 1],
            ekf.state.clone_ps[idx + 1],
        )
        refs.append(t_hat)
    return np.stack(refs, axis=0)


def joint_covariance_from_sigmas(sigmas: np.ndarray | None):
    if sigmas is None:
        return None
    covariance = np.zeros((12, 12), dtype=np.float64)
    np.fill_diagonal(covariance, sigmas.reshape(-1) ** 2)
    return covariance


def average_prediction_store(store: dict[tuple[int, int], list[np.ndarray]]):
    rows = []
    for key in sorted(store):
        values = np.stack(store[key], axis=0)
        rows.append(np.concatenate([np.asarray(key, dtype=np.float64), values.mean(axis=0)]))
    if not rows:
        return np.empty((0, 5), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def save_predictions(path: Path, rows: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows.shape[1] == 8:
        header = "t0_us t1_us px py pz sigma_x sigma_y sigma_z"
        fmt = ["%d", "%d"] + ["%.10f"] * 6
    else:
        header = "t0_us t1_us px py pz"
        fmt = ["%d", "%d"] + ["%.10f"] * 3
    np.savetxt(path, rows, fmt=fmt, header=header, comments="")


def integrate_network_trajectory(rows: np.ndarray, anchors_us, anchor_pos, anchor_quat):
    if len(rows) != len(anchors_us) - 1:
        return None
    positions = [anchor_pos[0].astype(np.float64)]
    quats = [anchor_quat[0].astype(np.float64)]
    for idx, row in enumerate(rows):
        R = Rotation.from_quat(anchor_quat[idx]).as_matrix()
        positions.append(positions[-1] + R @ row[2:5])
        quats.append(anchor_quat[idx + 1])
    return build_ground_truth_trajectory(anchors_us, np.asarray(positions), np.asarray(quats))


def save_table(path: Path, table: np.ndarray, header: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, table, fmt="%.10f", header=header, comments="")


def save_online_visualization_frame(
    output_dir: Path,
    sequence_name: str,
    anchor_idx: int,
    events: dict[str, np.ndarray] | None,
    voxelizer: OnlineVoxelizer,
    trajectory_rows: list[np.ndarray],
    gt_anchor_pos: np.ndarray,
    max_events: int,
) -> None:
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "online_visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_events, ax_traj = axes

    if events is not None and len(events["t"]) > 0:
        count = len(events["t"])
        if count > max_events:
            sample = np.linspace(0, count - 1, max_events, dtype=np.int64)
        else:
            sample = slice(None)
        x = np.asarray(events["x"][sample], dtype=np.float64) * voxelizer.scale_x
        y = np.asarray(events["y"][sample], dtype=np.float64) * voxelizer.scale_y
        p = np.asarray(events["p"][sample])
        colors = np.where(p > 0, "#0f8b8d", "#d1495b")
        ax_events.scatter(x, y, c=colors, s=0.12, alpha=0.45, linewidths=0)
        ax_events.set_title(f"events window | n={count}")
    else:
        ax_events.set_title("events window | empty")
    ax_events.set_xlim(0, voxelizer.new_width)
    ax_events.set_ylim(voxelizer.new_height, 0)
    ax_events.set_aspect("equal", adjustable="box")
    ax_events.set_xlabel("x [px]")
    ax_events.set_ylabel("y [px]")

    est = np.asarray(trajectory_rows, dtype=np.float64)
    ax_traj.plot(est[:, 1], est[:, 2], color="#1f77b4", linewidth=2.0, label="estimated")
    upto = min(anchor_idx + 1, len(gt_anchor_pos))
    if upto > 1:
        ax_traj.plot(
            gt_anchor_pos[:upto, 0],
            gt_anchor_pos[:upto, 1],
            color="#222222",
            linestyle="--",
            linewidth=1.1,
            alpha=0.55,
            label="GT reference",
        )
    ax_traj.scatter(est[-1, 1], est[-1, 2], color="#1f77b4", s=24)
    ax_traj.set_title(f"{sequence_name} | anchor {anchor_idx}")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.legend(loc="best")

    fig.tight_layout()
    fig.savefig(viz_dir / f"{sequence_name}_online_{anchor_idx:06d}.png", dpi=140)
    plt.close(fig)


def render_online_visualization_frame(
    path: Path,
    sequence_name: str,
    anchor_idx: int,
    events: dict[str, np.ndarray] | None,
    voxelizer: OnlineVoxelizer,
    trajectory_rows: list[np.ndarray],
    gt_anchor_pos: np.ndarray,
    max_events: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_events, ax_traj = axes

    if events is not None and len(events["t"]) > 0:
        count = len(events["t"])
        sample = np.linspace(0, count - 1, max_events, dtype=np.int64) if count > max_events else slice(None)
        x = np.asarray(events["x"][sample], dtype=np.float64) * voxelizer.scale_x
        y = np.asarray(events["y"][sample], dtype=np.float64) * voxelizer.scale_y
        p = np.asarray(events["p"][sample])
        colors = np.where(p > 0, "#0f8b8d", "#d1495b")
        ax_events.scatter(x, y, c=colors, s=0.12, alpha=0.45, linewidths=0)
        ax_events.set_title(f"events window | n={count}")
    else:
        ax_events.set_title("events window | empty")
    ax_events.set_xlim(0, voxelizer.new_width)
    ax_events.set_ylim(voxelizer.new_height, 0)
    ax_events.set_aspect("equal", adjustable="box")
    ax_events.set_xlabel("x [px]")
    ax_events.set_ylabel("y [px]")

    est = np.asarray(trajectory_rows, dtype=np.float64)
    ax_traj.plot(est[:, 1], est[:, 2], color="#1f77b4", linewidth=2.0, label="estimated")
    upto = min(anchor_idx + 1, len(gt_anchor_pos))
    if upto > 1:
        ax_traj.plot(
            gt_anchor_pos[:upto, 0],
            gt_anchor_pos[:upto, 1],
            color="#222222",
            linestyle="--",
            linewidth=1.1,
            alpha=0.55,
            label="GT reference",
        )
    ax_traj.scatter(est[-1, 1], est[-1, 2], color="#1f77b4", s=24)
    ax_traj.set_title(f"{sequence_name} | anchor {anchor_idx}")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def start_online_visualization_server(output_dir: Path, port: int):
    live_dir = output_dir / "online_live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "index.html").write_text(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TLEIO online visualization</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    header { padding: 10px 14px; background: #1b1b1b; }
    img { display: block; max-width: 100vw; max-height: calc(100vh - 42px); margin: auto; }
  </style>
</head>
<body>
  <header>TLEIO online visualization</header>
  <img id="frame" src="latest.png">
  <script>
    setInterval(() => {
      document.getElementById("frame").src = "latest.png?t=" + Date.now();
    }, 500);
  </script>
</body>
</html>
""".strip()
    )

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(live_dir))
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", int(port)), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"Online visualization server: http://127.0.0.1:{port}")
    print(f"Serving live files from: {live_dir}")
    return httpd, live_dir


class OnlineTrajectoryVisualizer:
    def __init__(
        self,
        sequence_name: str,
        voxelizer: OnlineVoxelizer,
        gt_anchor_pos: np.ndarray,
        max_events: int,
    ):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.sequence_name = sequence_name
        self.voxelizer = voxelizer
        self.gt_anchor_pos = gt_anchor_pos
        self.max_events = max_events
        self.plt.ion()
        self.fig, self.axes = self.plt.subplots(1, 2, figsize=(11, 5))
        self.fig.canvas.manager.set_window_title(f"TLEIO online | {sequence_name}")
        self.fig.show()

    def update(self, anchor_idx: int, events: dict[str, np.ndarray] | None, trajectory_rows: list[np.ndarray]):
        ax_events, ax_traj = self.axes
        ax_events.clear()
        ax_traj.clear()

        if events is not None and len(events["t"]) > 0:
            count = len(events["t"])
            if count > self.max_events:
                sample = np.linspace(0, count - 1, self.max_events, dtype=np.int64)
            else:
                sample = slice(None)
            x = np.asarray(events["x"][sample], dtype=np.float64) * self.voxelizer.scale_x
            y = np.asarray(events["y"][sample], dtype=np.float64) * self.voxelizer.scale_y
            p = np.asarray(events["p"][sample])
            colors = np.where(p > 0, "#0f8b8d", "#d1495b")
            ax_events.scatter(x, y, c=colors, s=0.12, alpha=0.45, linewidths=0)
            ax_events.set_title(f"events window | n={count}")
        else:
            ax_events.set_title("events window | empty")
        ax_events.set_xlim(0, self.voxelizer.new_width)
        ax_events.set_ylim(self.voxelizer.new_height, 0)
        ax_events.set_aspect("equal", adjustable="box")
        ax_events.set_xlabel("x [px]")
        ax_events.set_ylabel("y [px]")

        est = np.asarray(trajectory_rows, dtype=np.float64)
        ax_traj.plot(est[:, 1], est[:, 2], color="#1f77b4", linewidth=2.0, label="estimated")
        upto = min(anchor_idx + 1, len(self.gt_anchor_pos))
        if upto > 1:
            ax_traj.plot(
                self.gt_anchor_pos[:upto, 0],
                self.gt_anchor_pos[:upto, 1],
                color="#222222",
                linestyle="--",
                linewidth=1.1,
                alpha=0.55,
                label="GT reference",
            )
        ax_traj.scatter(est[-1, 1], est[-1, 2], color="#1f77b4", s=24)
        ax_traj.set_title(f"{self.sequence_name} | anchor {anchor_idx}")
        ax_traj.set_xlabel("x [m]")
        ax_traj.set_ylabel("y [m]")
        ax_traj.axis("equal")
        ax_traj.grid(True, alpha=0.25)
        ax_traj.legend(loc="best")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def run(config: OnlineConfig):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    serializable = asdict(config)
    for key, value in serializable.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
    (config.output_dir / "config.json").write_text(json.dumps(serializable, indent=2))

    network = NetworkRunner(config.checkpoint_file)
    events_file = config.raw_sequence_dir / "events.h5"
    slicer = RawTartanEventSlicer(events_file, "events/t", config.event_time_divisor)
    gt_table, gt_ts, gt_pos, gt_quat = load_raw_tartan_gt(
        config.raw_sequence_dir,
        slicer.t0_us,
        slicer.t_final_us,
    )
    anchors_us, anchor_pos, anchor_quat, gt_rel = make_anchors(
        gt_ts,
        gt_pos,
        gt_quat,
        config.delta_t_ms,
        config.anchor_t_ms,
    )
    if config.max_anchors is not None:
        anchors_us = anchors_us[: config.max_anchors]
        anchor_pos = anchor_pos[: config.max_anchors]
        anchor_quat = anchor_quat[: config.max_anchors]
        gt_rel = gt_rel[: max(0, config.max_anchors - 1)]
    if len(anchors_us) < network.clip_len:
        raise ValueError("Not enough anchors for one online network/filter update.")

    imu_table = generate_synthetic_imu(
        gt_ts,
        gt_pos,
        gt_quat,
        config.imu_rate_hz,
        np.asarray(config.gravity_world_mps2),
    )
    imu_segments = build_anchor_imu_segments(imu_table, anchors_us)
    voxelizer = OnlineVoxelizer(config.raw_sequence_dir, network.infer_args, gt_ts, gt_quat)
    if voxelizer.derotate:
        print(f"Derotation source: {config.derotation_source}")

    anchor_times_s = anchors_us.astype(np.float64) * 1e-6
    R0 = Rotation.from_quat(anchor_quat[0]).as_matrix()
    p0 = anchor_pos[0].astype(np.float64)
    v0 = (anchor_pos[1] - anchor_pos[0]) / max(anchor_times_s[1] - anchor_times_s[0], 1e-9)
    ekf = ImuMSCKF(make_filter_args(config))
    ekf.g = np.asarray(config.gravity_world_mps2, dtype=np.float64)
    ekf.initialize_with_state(anchor_times_s[0], R0, v0.astype(np.float64), p0, np.zeros(3), np.zeros(3))

    imu_ekf = ImuMSCKF(make_filter_args(config))
    imu_ekf.g = np.asarray(config.gravity_world_mps2, dtype=np.float64)
    imu_ekf.initialize_with_state(anchor_times_s[0], R0.copy(), v0.astype(np.float64), p0.copy(), np.zeros(3), np.zeros(3))

    scale_adapter = OnlineScaleAdapter(
        config.scale_mode,
        config.scale_init,
        config.scale_alpha,
        config.scale_min,
        config.scale_max,
    )
    default_joint_cov = make_default_joint_covariance(config.assumed_sigma_rel_t)
    voxel_window = deque(maxlen=network.clip_len)
    anchor_window = deque(maxlen=network.clip_len)
    scaled_store = defaultdict(list)
    raw_store = defaultdict(list)
    scale_history = []
    residual_norms = []
    delta_norms = []
    rejected_updates = 0

    first_events = slicer.get_events(int(anchors_us[0] - voxelizer.delta_t_us), int(anchors_us[0]))
    last_filter_quat = Rotation.from_matrix(ekf.state.R).as_quat()
    first_derotation_context = None
    if config.derotation_source == "filter":
        first_derotation_context = voxelizer.make_filter_derotation_context(
            int(anchors_us[0] - voxelizer.delta_t_us),
            int(anchors_us[0]),
            last_filter_quat,
            last_filter_quat,
        )
    voxel_window.append(voxelizer.build(first_events, int(anchors_us[0]), first_derotation_context))
    anchor_window.append(int(anchors_us[0]))
    ekf.augment_clone()

    trajectory_rows = [state_to_row(anchor_times_s[0], ekf.state)]
    imu_rows = [state_to_row(anchor_times_s[0], imu_ekf.state)]
    live_visualizer = None
    if config.show_online_visualization:
        live_visualizer = OnlineTrajectoryVisualizer(
            sequence_name=config.raw_sequence_dir.name,
            voxelizer=voxelizer,
            gt_anchor_pos=anchor_pos,
            max_events=config.viz_max_events,
        )
    live_server = None
    live_dir = None
    if config.serve_online_visualization:
        live_server, live_dir = start_online_visualization_server(config.output_dir, config.viz_port)

    try:
        for anchor_idx in range(1, len(anchors_us)):
            prev_filter_quat = last_filter_quat
            ekf.propagate(imu_segments[anchor_idx - 1])
            ekf.augment_clone()
            imu_ekf.propagate(imu_segments[anchor_idx - 1])
            current_filter_quat = Rotation.from_matrix(ekf.state.R).as_quat()

            events = slicer.get_events(
                int(anchors_us[anchor_idx] - voxelizer.delta_t_us),
                int(anchors_us[anchor_idx]),
            )
            derotation_context = None
            if config.derotation_source == "filter":
                derotation_context = voxelizer.make_filter_derotation_context(
                    int(anchors_us[anchor_idx] - voxelizer.delta_t_us),
                    int(anchors_us[anchor_idx]),
                    prev_filter_quat,
                    current_filter_quat,
                )
            voxel_window.append(voxelizer.build(events, int(anchors_us[anchor_idx]), derotation_context))
            anchor_window.append(int(anchors_us[anchor_idx]))

            if len(voxel_window) == network.clip_len:
                raw_pred, raw_sigmas = network.predict(voxel_window)
                edge_times = list(zip(list(anchor_window)[:-1], list(anchor_window)[1:]))

                for edge, pred_idx in zip(edge_times, range(network.clip_len - 1)):
                    if raw_sigmas is None:
                        raw_store[edge].append(raw_pred[pred_idx])
                    else:
                        raw_store[edge].append(np.concatenate([raw_pred[pred_idx], raw_sigmas[pred_idx]]))

                reference = None
                if config.scale_mode == "gt_debug":
                    reference = gt_rel[anchor_idx - 4 : anchor_idx, 2:5]
                elif config.scale_mode == "filter":
                    reference = clone_relative_predictions(ekf)
                candidate = scale_adapter.update(raw_pred, reference)
                pred, sigmas = scale_adapter.apply(raw_pred, raw_sigmas)
                scale_history.append(
                    [
                        float(anchor_times_s[anchor_idx]),
                        float(scale_adapter.scale),
                        np.nan if candidate is None else float(candidate),
                    ]
                )

                for edge, pred_idx in zip(edge_times, range(network.clip_len - 1)):
                    if sigmas is None:
                        scaled_store[edge].append(pred[pred_idx])
                    else:
                        scaled_store[edge].append(np.concatenate([pred[pred_idx], sigmas[pred_idx]]))

                update_payload = {"relative_pose": pred}
                if config.use_network_covariance and sigmas is not None:
                    update_payload["joint_covariance"] = joint_covariance_from_sigmas(sigmas)
                else:
                    update_payload["joint_covariance"] = default_joint_cov
                update_info = ekf.update(update_payload)
                if update_info.get("rejected", False):
                    rejected_updates += 1
                else:
                    residual_norms.append(float(np.linalg.norm(update_info["residual"])))
                    delta_norms.append(float(np.linalg.norm(update_info["delta_x"])))
                ekf.marginalize_oldest_clone()

            last_filter_quat = Rotation.from_matrix(ekf.state.R).as_quat()
            trajectory_rows.append(state_to_row(anchor_times_s[anchor_idx], ekf.state))
            imu_rows.append(state_to_row(anchor_times_s[anchor_idx], imu_ekf.state))
            if (
                live_visualizer is not None
                and config.viz_stride > 0
                and anchor_idx % config.viz_stride == 0
            ):
                live_visualizer.update(anchor_idx, events, trajectory_rows)
            if (
                live_dir is not None
                and config.viz_stride > 0
                and anchor_idx % config.viz_stride == 0
            ):
                render_online_visualization_frame(
                    path=live_dir / "latest.png",
                    sequence_name=config.raw_sequence_dir.name,
                    anchor_idx=anchor_idx,
                    events=events,
                    voxelizer=voxelizer,
                    trajectory_rows=trajectory_rows,
                    gt_anchor_pos=anchor_pos,
                    max_events=config.viz_max_events,
                )
            if (
                config.save_online_visualization
                and config.viz_stride > 0
                and anchor_idx % config.viz_stride == 0
            ):
                save_online_visualization_frame(
                    output_dir=config.output_dir,
                    sequence_name=config.raw_sequence_dir.name,
                    anchor_idx=anchor_idx,
                    events=events,
                    voxelizer=voxelizer,
                    trajectory_rows=trajectory_rows,
                    gt_anchor_pos=anchor_pos,
                    max_events=config.viz_max_events,
                )
    finally:
        slicer.close()
        if live_server is not None:
            live_server.shutdown()

    trajectory = np.asarray(trajectory_rows, dtype=np.float64)
    imu_trajectory = np.asarray(imu_rows, dtype=np.float64)
    gt_trajectory = build_ground_truth_trajectory(anchors_us, anchor_pos, anchor_quat)
    scaled_rows = average_prediction_store(scaled_store)
    raw_rows = average_prediction_store(raw_store)
    regressed_trajectory = integrate_network_trajectory(scaled_rows, anchors_us, anchor_pos, anchor_quat)

    save_table(config.output_dir / "stamped_traj_estimate.txt", trajectory, "timestamp_s px py pz qx qy qz qw")
    save_table(config.output_dir / "imu_only_trajectory.txt", imu_trajectory, "timestamp_s px py pz qx qy qz qw")
    save_table(config.output_dir / "gt_anchor_trajectory.txt", gt_trajectory, "timestamp_s px py pz qx qy qz qw")
    save_predictions(config.output_dir / "predicted_relative_motions.txt", scaled_rows)
    save_predictions(config.output_dir / "predicted_relative_motions_unscaled.txt", raw_rows)
    if scale_history:
        save_table(config.output_dir / "scale_history.txt", np.asarray(scale_history), "timestamp_s scale candidate_scale")
    np.savetxt(
        config.output_dir / "gt_relative_motions.txt",
        gt_rel,
        fmt=["%d", "%d"] + ["%.10f"] * 6,
        header="t0_us t1_us px py pz rx ry rz",
        comments="",
    )
    np.savetxt(
        config.output_dir / "stamped_groundtruth_online.txt",
        gt_table,
        fmt=["%d"] + ["%.10f"] * 7,
        header="timestamp_us px py pz qx qy qz qw",
        comments="",
    )

    diagnostics = compute_filter_diagnostics(
        trajectory,
        gt_trajectory,
        regressed_trajectory=regressed_trajectory,
        imu_trajectory=imu_trajectory,
        output_dir=config.output_dir,
        file_prefix=config.raw_sequence_dir.name,
        plot_projections=config.plot_projections,
    )
    inference_stats = network.inference_stats()
    summary = {
        "num_anchors": int(len(anchors_us)),
        "num_updates_attempted": int(max(0, len(anchors_us) - 4)),
        "num_updates_rejected": int(rejected_updates),
        "mean_residual_norm": None if not residual_norms else float(np.mean(residual_norms)),
        "mean_delta_norm": None if not delta_norms else float(np.mean(delta_norms)),
        "final_scale": float(scale_adapter.scale),
        "derotation_source": config.derotation_source,
        "show_online_visualization": bool(config.show_online_visualization),
        "online_visualization": bool(config.save_online_visualization),
        "serve_online_visualization": bool(config.serve_online_visualization),
        "network_inference": inference_stats,
        "diagnostics": diagnostics,
    }
    (config.output_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2))
    if inference_stats["hz"] is not None:
        print(
            "Network inference | "
            f"count={inference_stats['count']} | "
            f"mean={inference_stats['mean_ms']:.3f} ms | "
            f"hz={inference_stats['hz']:.2f}"
        )
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run raw Tartan online processing, network inference, and EKF fusion.")
    parser.add_argument("--raw_sequence_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--delta_t_ms", type=int, default=50)
    parser.add_argument("--anchor_t_ms", type=int, default=50)
    parser.add_argument("--event_time_divisor", type=int, default=1000)
    parser.add_argument("--imu_rate_hz", type=float, default=200.0)
    parser.add_argument("--use_network_covariance", action="store_true")
    parser.add_argument("--scale_mode", choices=["none", "gt_debug", "filter"], default="none")
    parser.add_argument("--derotation_source", choices=["filter", "gt"], default="filter")
    parser.add_argument("--scale_init", type=float, default=1.0)
    parser.add_argument("--scale_alpha", type=float, default=0.01)
    parser.add_argument("--scale_min", type=float, default=0.3)
    parser.add_argument("--scale_max", type=float, default=2.0)
    parser.add_argument("--max_anchors", type=int, default=None)
    parser.add_argument("--no_plot_projections", action="store_true")
    parser.add_argument("--show_online_visualization", action="store_true")
    parser.add_argument("--save_online_visualization", action="store_true")
    parser.add_argument("--serve_online_visualization", action="store_true")
    parser.add_argument("--viz_port", type=int, default=8765)
    parser.add_argument("--viz_stride", type=int, default=25)
    parser.add_argument("--viz_max_events", type=int, default=20000)
    args = parser.parse_args()
    return OnlineConfig(
        raw_sequence_dir=args.raw_sequence_dir,
        checkpoint_file=args.checkpoint_file,
        output_dir=args.output_dir,
        delta_t_ms=args.delta_t_ms,
        anchor_t_ms=args.anchor_t_ms,
        event_time_divisor=args.event_time_divisor,
        imu_rate_hz=args.imu_rate_hz,
        use_network_covariance=args.use_network_covariance,
        scale_mode=args.scale_mode,
        derotation_source=args.derotation_source,
        scale_init=args.scale_init,
        scale_alpha=args.scale_alpha,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        max_anchors=args.max_anchors,
        plot_projections=not args.no_plot_projections,
        show_online_visualization=args.show_online_visualization,
        save_online_visualization=args.save_online_visualization,
        serve_online_visualization=args.serve_online_visualization,
        viz_port=args.viz_port,
        viz_stride=args.viz_stride,
        viz_max_events=args.viz_max_events,
    )


if __name__ == "__main__":
    run(parse_args())
