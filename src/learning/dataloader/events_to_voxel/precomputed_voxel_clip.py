from pathlib import Path
import bisect
import json

import numpy as np
import torch
from torch.utils.data import Dataset

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


class PrecomputedVoxelClipDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        clip_len: int = 3,
        num_bins: int | None = None,
        voxel_filename: str = "derotated_voxels.npy",
        mmap_mode: str | None = "r",
        normalize_voxel_nonzero: bool = False,
    ):
        assert clip_len >= 1
        assert root_path.is_dir()

        self.root_path = root_path
        self.clip_len = clip_len
        self.num_bins = num_bins
        self.voxel_filename = voxel_filename
        self.mmap_mode = mmap_mode
        self.normalize_voxel_nonzero = normalize_voxel_nonzero
        self.seq_infos = []
        self.cum_lengths = []
        self._voxels = []
        self.preprocessing_args = {}
        self.height = None
        self.width = None
        self.train_std = None
        self.train_mean = None
        self.eps = 1e-7

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
            if self.height is None:
                self.height = int(voxel_shape[2])
                self.width = int(voxel_shape[3])
            elif voxel_shape[2] != self.height or voxel_shape[3] != self.width:
                raise ValueError(
                    f"{voxels_file}: inconsistent voxel spatial shape "
                    f"{voxel_shape[2:4]}, expected {(self.height, self.width)}"
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
        if self.height is not None and self.width is not None:
            self.preprocessing_args["input_height"] = self.height
            self.preprocessing_args["input_width"] = self.width
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
