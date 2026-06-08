from pathlib import Path
import bisect
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

#

def normalize_nonzero_voxel_(voxel: torch.Tensor) -> torch.Tensor:
    mask = voxel != 0
    if not bool(mask.any()):
        return voxel

    values = voxel[mask]
    mean = values.mean()
    std = values.std(unbiased=False)
    if torch.isfinite(std) and std.item() > 0:
        voxel[mask] = (values - mean) / std
    else:
        voxel[mask] = values - mean
    return voxel


def build_voxel_rectification_maps(
    calibration_path: Path,
    output_height: int,
    output_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    with calibration_path.open("r") as fh:
        calibration = yaml.safe_load(fh)

    if "cam1" not in calibration:
        raise KeyError(f"{calibration_path}: missing cam1 calibration.")
    cam = calibration["cam1"]
    intrinsics = cam.get("intrinsics")
    distortion = cam.get("distortion_coeffs")
    resolution = cam.get("resolution")
    if intrinsics is None or len(intrinsics) != 4:
        raise ValueError(f"{calibration_path}: expected four cam1 intrinsics.")
    if distortion is None or len(distortion) < 4:
        raise ValueError(f"{calibration_path}: missing distortion_coeffs.")
    if resolution is None or len(resolution) != 2:
        raise ValueError(f"{calibration_path}: expected cam1 resolution [width, height].")

    source_width, source_height = (int(resolution[0]), int(resolution[1]))
    scale_x = output_width / source_width
    scale_y = output_height / source_height
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    camera_matrix = np.array(
        [
            [fx * scale_x, 0.0, cx * scale_x],
            [0.0, fy * scale_y, cy * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(distortion, dtype=np.float64)
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        (output_width, output_height),
        cv2.CV_32FC1,
    )


def rectify_voxel_clip(
    clip: torch.Tensor,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> torch.Tensor:
    clip_np = clip.numpy()
    rectified = np.empty_like(clip_np)
    for time_idx in range(clip_np.shape[0]):
        for bin_idx in range(clip_np.shape[1]):
            rectified[time_idx, bin_idx] = cv2.remap(
                clip_np[time_idx, bin_idx],
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
    return torch.from_numpy(rectified)


class PrecomputedVoxelClipDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        clip_len: int = 3,
        num_bins: int | None = None,
        voxel_filename: str = "derotated_voxels.npy",
        mmap_mode: str | None = "r",
        normalize_voxel_nonzero: bool = False,
        rectify_precomputed: bool = False,
        rectification_calibration: Path | None = None,
    ):
        assert clip_len >= 1
        assert root_path.is_dir()

        self.root_path = root_path
        self.clip_len = clip_len
        self.num_bins = num_bins
        self.voxel_filename = voxel_filename
        self.mmap_mode = mmap_mode
        self.normalize_voxel_nonzero = normalize_voxel_nonzero
        self.rectify_precomputed = rectify_precomputed
        self.rectification_calibration = (
            Path(rectification_calibration)
            if rectification_calibration is not None
            else None
        )
        self.seq_infos = []
        self.cum_lengths = []
        self._voxels = []
        self.preprocessing_args = {}
        self.train_std = None
        self.train_mean = None
        self.eps = 1e-7
        self._rectification_maps = {}

        if self.rectify_precomputed and self.rectification_calibration is None:
            raise ValueError(
                "rectify_precomputed requires rectification_calibration."
            )

        total = 0
        sequence_dirs = sorted([p for p in self.root_path.iterdir() if p.is_dir()])
        for seq_path in sequence_dirs:
            voxels_file = seq_path / self.voxel_filename
            rel_transf_fn = seq_path / "relative_motions.txt"
            if not (voxels_file.exists() and rel_transf_fn.exists()):
                continue

            rel_transf = np.atleast_2d(
                np.loadtxt(rel_transf_fn, dtype=np.float64, skiprows=1)
            )
            if rel_transf.shape[1] != 8:
                raise ValueError(
                    f"{seq_path}: expected 8 columns in relative_motions.txt, "
                    f"got {rel_transf.shape[1]}"
                )
            if np.any(rel_transf[1:, 0] != rel_transf[:-1, 1]):
                raise ValueError(f"{seq_path}: relative motion timestamps are not contiguous.")

            voxel_shape = np.load(voxels_file, mmap_mode="r").shape
            if len(voxel_shape) != 4:
                raise ValueError(
                    f"{voxels_file}: expected shape [N, C, H, W], got {voxel_shape}"
                )
            if self.num_bins is None:
                self.num_bins = int(voxel_shape[1])
            elif voxel_shape[1] != self.num_bins:
                raise ValueError(
                    f"{voxels_file}: expected {self.num_bins} bins, got {voxel_shape[1]}"
                )

            metadata_file = seq_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as fh:
                    metadata = json.load(fh)
                if "num_bins" in metadata and metadata["num_bins"] != self.num_bins:
                    raise ValueError(f"{seq_path}: metadata num_bins does not match voxel shape.")
                for key in (
                    "downsampling_factor",
                    "denoising",
                    "denoise_dt_us",
                    "denoise_radius",
                    "denoise_min_supporters",
                    "denoise_same_polarity_only",
                    "derotate",
                    "derotation_slices",
                ):
                    if key in metadata:
                        if key in self.preprocessing_args and self.preprocessing_args[key] != metadata[key]:
                            raise ValueError(f"{seq_path}: inconsistent precomputed voxel metadata for {key}.")
                        self.preprocessing_args[key] = metadata[key]

            anchors_us = np.concatenate(
                [rel_transf[:1, 0], rel_transf[:, 1]],
                axis=0,
            ).astype(np.int64)
            if voxel_shape[0] != len(anchors_us):
                raise ValueError(
                    f"{voxels_file}: expected {len(anchors_us)} voxels, got {voxel_shape[0]}"
                )

            num_samples = max(0, len(anchors_us) - self.clip_len + 1)
            self.seq_infos.append(
                {
                    "seq_path": seq_path,
                    "voxels_file": voxels_file,
                    "anchors_us": anchors_us,
                    "rel_transf": rel_transf,
                    "num_samples": num_samples,
                }
            )
            self._voxels.append(None)
            total += num_samples
            self.cum_lengths.append(total)

        self.preprocessing_args.setdefault("num_bins", self.num_bins)
        for key, value in self.preprocessing_args.items():
            setattr(self, key, value)

    def __len__(self):
        return self.cum_lengths[-1] if self.cum_lengths else 0

    def locate_index(self, idx, cum_lengths):
        seq_idx = bisect.bisect_right(cum_lengths, idx)
        prev_cum = 0 if seq_idx == 0 else cum_lengths[seq_idx - 1]
        local_idx = idx - prev_cum
        return seq_idx, local_idx

    def _ensure_voxels(self, seq_idx):
        if self._voxels[seq_idx] is None:
            self._voxels[seq_idx] = np.load(
                self.seq_infos[seq_idx]["voxels_file"],
                mmap_mode=self.mmap_mode,
            )

    def _resolve_rectification_calibration(self, seq_idx: int) -> Path:
        calibration = self.rectification_calibration
        if calibration.is_file():
            return calibration

        seq_name = self.seq_infos[seq_idx]["seq_path"].name
        candidate = calibration / seq_name / "K.yaml"
        if candidate.exists():
            return candidate

        fallback = calibration / "K.yaml"
        if fallback.exists() and len(self.seq_infos) == 1:
            return fallback

        raise FileNotFoundError(
            f"Could not find rectification calibration for {seq_name}. "
            f"Tried {candidate}."
        )

    def _rectify_clip(self, clip: torch.Tensor, seq_idx: int) -> torch.Tensor:
        if seq_idx not in self._rectification_maps:
            calibration_path = self._resolve_rectification_calibration(seq_idx)
            self._rectification_maps[seq_idx] = build_voxel_rectification_maps(
                calibration_path=calibration_path,
                output_height=int(clip.shape[-2]),
                output_width=int(clip.shape[-1]),
            )
        return rectify_voxel_clip(clip, *self._rectification_maps[seq_idx])

    def get_relative_motion(self, rel_transf, t0_idx, t0, t1):
        assert (t0 == rel_transf[t0_idx, 0]) and (t1 == rel_transf[t0_idx, 1])
        return rel_transf[t0_idx, 2:5]

    def __getitem__(self, index):
        seq_idx, local_idx = self.locate_index(index, self.cum_lengths)
        self._ensure_voxels(seq_idx)

        seq_info = self.seq_infos[seq_idx]
        anchors = seq_info["anchors_us"][local_idx : local_idx + self.clip_len]
        clip_np = self._voxels[seq_idx][local_idx : local_idx + self.clip_len]
        clip = torch.from_numpy(clip_np.astype(np.float32, copy=True))
        if self.rectify_precomputed:
            clip = self._rectify_clip(clip, seq_idx)
        if self.normalize_voxel_nonzero:
            for t in range(clip.shape[0]):
                normalize_nonzero_voxel_(clip[t])
        clip = clip.permute(1, 0, 2, 3)

        clip_targets = []
        rel_transf = seq_info["rel_transf"]
        for j in range(self.clip_len - 1):
            rel_target = self.get_relative_motion(
                rel_transf,
                local_idx + j,
                int(anchors[j]),
                int(anchors[j + 1]),
            )
            if self.train_mean is not None and self.train_std is not None:
                rel_target = (rel_target - self.train_mean) / self.train_std
            clip_targets.append(torch.as_tensor(rel_target, dtype=torch.float32))

        return {
            "representation": clip,
            "anchors_us": torch.as_tensor(anchors, dtype=torch.int64),
            "target": torch.stack(clip_targets, dim=0),
        }

    def compute_stats(self, indices):
        all_train_targets = []
        for index in indices:
            seq_idx, local_idx = self.locate_index(index, self.cum_lengths)
            rel_transf = self.seq_infos[seq_idx]["rel_transf"]
            targets = rel_transf[local_idx : local_idx + self.clip_len - 1, 2:5]
            all_train_targets.append(targets)

        targets = np.concatenate(all_train_targets, axis=0)
        self.train_std = np.std(targets, axis=0)
        self.train_std = np.maximum(self.train_std, self.eps)
        self.train_mean = np.mean(targets, axis=0)
