from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import bisect

from ..representation.event_denoising import background_activity_filter_raw
from ..representation.voxel_grid import VoxelGrid
from .precomputed_voxel_clip import normalize_nonzero_voxel_
from .reader import EDSReader
from .utils import (
    build_derotation_context,
    load_event_camera_matrix,
    normalize_quaternions,
)

"""
Online event-to-voxel dataset used by the training pipeline. Methods of this class 
are also used by the offline processing script to build precomputed voxels for faster
training.
The dataset reads partially processed EDS/Tartan-style sequence folders,
constructs one voxel grid per anchor timestamp, and groups consecutive anchors
into clips for TSformer-VO training.
"""

class MultiEventVoxelClipDataset(Dataset):
    """Build clips of event voxel grids from partially processed sequences.

    Each valid sequence directory under `root_path` must contain
    `anchor_poses.txt`, `relative_motions.txt`, and `events.h5`.
    When de-rotation is enabled, the sequence must also contain
    `stamped_groundtruth.txt` and a `K.yaml` calibration file.

    A dataset item corresponds to `clip_len` consecutive anchors from one
    sequence. For every anchor, the dataset reads the event window ending at
    that anchor, optionally denoises/downsamples/de-rotates the events, and
    voxelizes them.
    """

    @staticmethod
    def get_downsampled_size(
        original_height: int,
        original_width: int,
        downsampling_factor: float,
        patch_size: int,
    ) -> tuple[int, int]:
        """Compute and validate the spatial size after event downsampling.

        The resulting height and width must stay positive and be divisible by
        `patch_size` so the downstream patch-based model can consume the
        voxel grids without padding.
        """
        if downsampling_factor <= 0:
            raise ValueError("downsampling_factor must be > 0.")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0.")

        new_height = int(round(original_height * downsampling_factor))
        new_width = int(round(original_width * downsampling_factor))

        if new_height <= 0 or new_width <= 0:
            raise ValueError(
                "Downsampled spatial size must stay positive, "
                f"got ({new_height}, {new_width})."
            )

        if new_height % patch_size != 0 or new_width % patch_size != 0:
            raise ValueError(
                "Downsampled size must be divisible by patch size. "
                f"factor={downsampling_factor} gives "
                f"({new_height}, {new_width}) with patch_size={patch_size}."
            )

        return new_height, new_width


    def __init__(self, 
                 root_path: Path, 
                 delta_t_ms: int=50, 
                 num_bins: int=5, 
                 clip_len: int = 3, 
                 downsampling_factor: float = 1.0,
                 patch_size: int = 16,
                 denoising: bool = False,
                 denoise_dt_us: int = 1000,
                 denoise_radius: int = 1,
                 denoise_min_supporters: int = 1,
                 denoise_same_polarity_only: bool = False,
                 derotate: bool = False,
                 derotation_slices: int = 100,
                 normalize_voxel_nonzero: bool = False):
        """Initialize the Event-to-Voxel clip dataset.

        Args:
            root_path: Directory containing one subdirectory per processed
                sequence.
            delta_t_ms: Duration of each voxel event window in milliseconds.
                For an anchor at `t`, events are read from
                `[t - delta_t_ms, t)`.
            num_bins: Number of temporal bins in the final voxel grid.
            clip_len: Number of consecutive anchor voxels returned per item.
            downsampling_factor: Spatial scaling applied to event coordinates.
            patch_size: Size of the patch being used to tokenize voxels. 
                Required here to check for divisibility for the downsampled image size.
            denoising: to apply background activity filtering before
                voxelization.
            denoise_dt_us: Temporal support window for denoising.
            denoise_radius: Spatial radius for denoising support checks.
            denoise_min_supporters: Minimum neighbors required to keep an event.
            denoise_same_polarity_only: Whether denoising only counts events
                with the same polarity.
            derotate: Whether to de-rotate event coordinates before
                voxelization.
            derotation_slices: Number of pose slices used for event-space
                de-rotation. This is independent from `num_bins`.
        """

        assert num_bins >= 1
        assert clip_len >= 1
        assert delta_t_ms <= 100, 'if duration is higher than 100 ms'
        assert root_path.is_dir()


        self.seq_infos = []
        self.cum_lengths = []
        self._readers = []
        total = 0

        # Set constants
        self.clip_len = clip_len
        self.patch_size = patch_size
        self.root_path = root_path
        self.original_height = 480
        self.original_width = 640
        
        #Downsampling
        self.downsampling_factor = downsampling_factor
        self.new_height, self.new_width = self.get_downsampled_size(
            self.original_height,
            self.original_width,
            self.downsampling_factor,
            self.patch_size,
        )
        self.scale_y = self.new_height / self.original_height
        self.scale_x = self.new_width / self.original_width
        self.num_bins = num_bins

        #Denoising
        self.denoising = denoising
        self.denoise_dt_us = denoise_dt_us
        self.denoise_radius = denoise_radius
        self.denoise_min_supporters = denoise_min_supporters
        self.denoise_same_polarity_only = denoise_same_polarity_only

        #Derotation
        self.derotate = derotate
        self.derotation_slices = derotation_slices
        self.normalize_voxel_nonzero = normalize_voxel_nonzero

        #the duration of a voxel
        self.delta_t_us = delta_t_ms * 1000

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins,
                                    self.new_height,
                                    self.new_width,
                                    derotate=self.derotate,
                                    derotation_slices=self.derotation_slices)

        #Set the normalization stats to None: 
        self.train_std = None
        self.train_mean = None
        self.eps = 1e-7

        #TARGET DATA LOADING: LIGHTWEIGHT, CAN BE LOADED IN RAM EASILY

        total = 0
        sequence_dirs = sorted([p for p in self.root_path.iterdir() if p.is_dir()])

        for seq_path in sequence_dirs:
            gt_poses_fn = seq_path / "anchor_poses.txt"
            gt_rel_transf_fn = seq_path / "relative_motions.txt"
            events_file = seq_path / "events.h5"
            events_meta_file = seq_path / "events_meta.h5"
            gt_full_fn = seq_path / "stamped_groundtruth.txt"

            # skip folders that do not contain a valid processed sequence
            if not (gt_poses_fn.exists() and gt_rel_transf_fn.exists() and events_file.exists()):
                continue

            # stamped gt needed only when de-rotation happens 
            if self.derotate and not gt_full_fn.exists():
                raise FileNotFoundError(
                    f"{seq_path}: derotation requires stamped_groundtruth.txt."
                )

            anchor_poses = np.atleast_2d(np.loadtxt(gt_poses_fn, dtype=np.float64, skiprows=1))
            rel_transf = np.atleast_2d(np.loadtxt(gt_rel_transf_fn, dtype=np.float64, skiprows=1))
            seq_info = {
                "seq_path": seq_path,
                "events_file": events_file,
                "events_meta_file": events_meta_file if events_meta_file.exists() else None,
                "anchors_us": anchor_poses[:, 0],
                "rel_transf": rel_transf,
            }

            # De-rotation TRUE path
            # De-rotation needs the following: 
            # - the ground truth to find the rotation matrices from each event slice to 
            #   the reference frame (the anchor of each voxel)
            # - the camera matrix to warp the events according to the camera intrinsics

            if self.derotate:
                gt_full = np.atleast_2d(np.loadtxt(gt_full_fn, dtype=np.float64))
                if gt_full.shape[1] < 8:
                    raise ValueError(
                        f"{gt_full_fn}: expected at least 8 columns, got {gt_full.shape[1]}"
                    )
                #Load timestamps, quaternions and camera matrix utilities for reprojection
                seq_info["gt_timestamps_us"] = gt_full[:, 0].astype(np.int64)
                seq_info["gt_quat_xyzw"] = normalize_quaternions(
                    gt_full[:, 4:8].astype(np.float64)
                )
                
                seq_info["camera_matrix"] = load_event_camera_matrix(
                    root_path=self.root_path,
                    seq_path=seq_path,
                    scale_x=self.scale_x,
                    scale_y=self.scale_y,
                )

            #Dimensionality checks
            if anchor_poses.shape[1] != 8:
                raise ValueError(
                    f"{seq_path}: expected 8 columns in anchor_poses.txt, got {anchor_poses.shape[1]}"
                )

            if rel_transf.shape[1] != 8:
                raise ValueError(
                    f"{seq_path}: expected 8 columns in relative_motions.txt, got {rel_transf.shape[1]}"
                )

            anchors_us = seq_info["anchors_us"]
            num_samples = max(0, len(anchors_us) - self.clip_len + 1)
            seq_info["num_samples"] = num_samples
            self.seq_infos.append(seq_info)

            total += num_samples
            self.cum_lengths.append(total)

        self._readers = [None] * len(self.seq_infos)        
    
    def _ensure_reader(self, seq_idx):
        """Open the HDF5 event reader for a sequence on first use."""
        if self._readers[seq_idx] is None:
            events_file = self.seq_infos[seq_idx]["events_file"]
            events_meta_file = self.seq_infos[seq_idx]["events_meta_file"]
            self._readers[seq_idx] = EDSReader(events_file, metadata_file=events_meta_file)
    
    def close(self):
        """Close all open sequence readers held by this dataset instance."""
        if not hasattr(self, "_readers"):
            return
        for i, reader in enumerate(self._readers):
            if reader is not None:
                reader.close()
                self._readers[i] = None

    def __del__(self):
        """Cleanup for HDF5 readers."""
        self.close()

    def __len__(self):
        """Return the number of valid clips across all loaded sequences:
           the number of samples of the dataset"""
        
        return self.cum_lengths[-1] if self.cum_lengths else 0
    
    def locate_index(self, idx, cum_lengths):
        """Map a global dataset index to ``(sequence_index, local_index)``."""
        seq_idx = bisect.bisect_right(cum_lengths, idx)
        prev_cum = 0 if seq_idx == 0 else cum_lengths[seq_idx - 1]
        local_idx = idx - prev_cum
        return seq_idx, local_idx
    
    def _empty_voxel(self):
        """Return an all-zero voxel grid for empty event windows."""
        return torch.zeros(
            (self.num_bins, self.new_height, self.new_width),
            dtype=torch.float32
        )

    def events_to_voxel_grid(
            self, 
            x,
            y,
            p, 
            t, 
            ts_start_us=None, 
            ts_end_us=None, 
            seq_info=None
            ):
        
        """Convert one raw event window into a voxel grid.

        Args:
            x: Event x coordinates in the original sensor resolution.
            y: Event y coordinates in the original sensor resolution.
            p: Event polarities, encoded as 0/1.
            t: Absolute event timestamps in microseconds.
            ts_start_us: Start timestamp of the event window. Required
                when `self.derotate` is true.
            ts_end_us: End timestamp of the event window. Required when
                `self.derotate` is true.
            seq_info: Sequence metadata dictionary. The de-rotation path
                expects ground-truth pose timestamps/quaternions and the event
                camera matrix in this dictionary.

        Returns:
            A `torch.float32` tensor with shape
            `[num_bins, new_height, new_width]`.
        """
        if len(t) == 0:
            return self._empty_voxel()

        #Apply background consistency denoising
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
                return self._empty_voxel()

        #Apply downsampling to reduce per-frame token count
        downsampled_x = x * self.scale_x
        downsampled_y = y * self.scale_y

        #Build and add the de-rotation context for later. Skip time normalization 
        if self.derotate:
            if ts_start_us is None or ts_end_us is None or seq_info is None:
                raise ValueError("Derotation requires window bounds and sequence metadata.")

            event_data = {
                "x": downsampled_x.astype(np.float32),
                "y": downsampled_y.astype(np.float32),
                "p": p.astype(np.float32),
                "t": t.astype(np.float64),
                "ts_start_us": int(ts_start_us),
                "ts_end_us": int(ts_end_us),
            }
            
            event_data.update(build_derotation_context(
                            seq_info=seq_info,
                            ts_start_us=ts_start_us,
                            ts_end_us=ts_end_us,
                            num_bins=self.derotation_slices,
                            )
            )
        else:
            t = t.astype(np.float32)
            t = t - t[0]

            if t[-1] > 0:
                t = t / t[-1]
            else:
                t = np.zeros_like(t, dtype=np.float32)

            event_data = {
                "x": downsampled_x.astype(np.float32),
                "y": downsampled_y.astype(np.float32),
                "p": p.astype(np.float32),
                "t": t,
            }

        #Convert the events into voxels
        voxel = self.voxel_grid.convert_events(event_data)
        return voxel
    
    def get_relative_motion(self, rel_transf, t0_idx, t0, t1):
        """Return the translation target between two consecutive anchors.

        `relative_motions.txt` is expected to contain the source and target
        anchor timestamps in the first two columns, followed by the supervision
        values. This method asserts that the requested anchor pair matches the
        stored row before returning columns 2:5, which are the displacement values.
        """
        
        #Check also the timestamps matches
        assert (t0 == rel_transf[t0_idx, 0]) and (t1 == rel_transf[t0_idx, 1])
        #Get the actual supervision data
        return rel_transf[t0_idx, 2:5]


    def __getitem__(self, index):
        """Return one clip and its pairwise motion targets.

        The output dictionary contains:
            `representation`: voxel clip with shape `[C, T, H, W]`.
            `anchors_us`: anchor timestamps for the clip.
            `target`: relative translation targets with shape
            `[T - 1, target_dim]`.
        """
        #Find sequence index in the whole dataset + local index within that sequence
        seq_idx, local_idx = self.locate_index(index, self.cum_lengths)

        #Open the reader for that sequence (get ready to open the h5 file with events)
        self._ensure_reader(seq_idx)

        #Get the anchor timestamps and build a voxel for each anchor
        seq_anchors = self.seq_infos[seq_idx]["anchors_us"]
        anchors = seq_anchors[local_idx : local_idx + self.clip_len]
        clip_voxels = []
        clip_targets = []

        for j, anchor in enumerate(anchors):
            ts_end_us = int(anchor)
            ts_start_us = ts_end_us - self.delta_t_us

            #Slicing events in the requested window
            reader = self._readers[seq_idx]
            event_data = reader.get_events(ts_start_us, ts_end_us)

            if event_data is None:
                voxel = self._empty_voxel()
            else:
                voxel = self.events_to_voxel_grid(
                    event_data["x"],
                    event_data["y"],
                    event_data["p"],
                    event_data["t"],
                    ts_start_us=ts_start_us,
                    ts_end_us=ts_end_us,
                    seq_info=self.seq_infos[seq_idx],
                )

            if self.normalize_voxel_nonzero:
                normalize_nonzero_voxel_(voxel)
            clip_voxels.append(voxel.float())

            # target for pairwise motion: one target per transition
            if j < len(anchors) - 1:
                t0_idx = local_idx + j
                t0 = int(anchors[j])
                t1 = int(anchors[j + 1])
                rel_transf = self.seq_infos[seq_idx]["rel_transf"]
                rel_target = self.get_relative_motion(rel_transf, t0_idx, t0, t1)   # shape [target_dim (3)]
                
                #Normalize the target based on the training split mean and std
                if self.train_mean is not None and self.train_std is not None: 
                    rel_target = (rel_target - self.train_mean) / self.train_std
                
                clip_targets.append(torch.as_tensor(rel_target, dtype=torch.float32))

        # stack voxels: list of [num_bins, H, W] -> [num_bins, clip_len, H, W]
        clip = torch.stack(clip_voxels, dim=1)

        # stack targets: list of [target_dim] -> [clip_len - 1, target_dim]
        target = torch.stack(clip_targets, dim=0)

        output = {
            "representation": clip,                              # [C, T, H, W]
            "anchors_us": torch.as_tensor(anchors, dtype=torch.int64),
            "target": target,                                    # [T-1, target_dim]
        }

        return output
    
    def compute_stats(self, indices): 
        """Compute target normalization statistics over a subset of indices.

        The method scans the pairwise motion targets for the provided dataset
        indices and stores `self.train_mean` and `self.train_std`. These
        values are later used by `__getitem__` to normalize returned targets.
        """
        all_train_targets = []
        for index in indices: 
            seq_idx, local_idx = self.locate_index(index, self.cum_lengths)

            #Get the anchor timestamps 
            seq_anchors = self.seq_infos[seq_idx]["anchors_us"]
            anchors = seq_anchors[local_idx : local_idx + self.clip_len]
            

            for j in range(len(anchors)):
                # target for pairwise motion: one target per transition
                if j < len(anchors) - 1:
                    t0_idx = local_idx + j
                    t0 = int(anchors[j])
                    t1 = int(anchors[j + 1])
                    rel_transf = self.seq_infos[seq_idx]["rel_transf"]
                    rel_target = self.get_relative_motion(rel_transf, t0_idx, t0, t1)  
                    all_train_targets.append(rel_target)
        
        # stack targets: list of [target_dim] -> [train_samples, target_dim]
        targets = np.stack(all_train_targets, axis=0)
        self.train_std = np.std(targets, axis=0)
        self.train_std = np.maximum(self.train_std, self.eps)
        self.train_mean = np.mean(targets, axis=0)
