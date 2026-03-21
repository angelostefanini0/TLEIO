from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from representation.voxel_grid import VoxelGrid
from representation.event_slicer import EventSlicer
from reader import EDSReader
from utils.io import *

class EventVoxelClipDataset(Dataset):
    # We use the voxel grid representation + clipping for TSformer-VO
    #
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. 01_peanuts_light)
    # ├── events.h5
    # | └── events/y
    # | └── events/t
    # | └── events/p
    # | ├── events/x
    # | └── ms_to_idx
    # ├── imu.csv
    # ├── stamped_groundtruth.csv


    def __init__(self, 
                 seq_path: Path, 
                 mode: str='train', 
                 delta_t_ms: int=50, 
                 num_bins: int=15, 
                 clip_len: int = 3):
        
        assert num_bins >= 1
        assert clip_len >= 1
        assert delta_t_ms <= 100, 'if duration is higher than 100 ms'
        assert seq_path.is_dir()

        # Set constants
        self.seq_path = seq_path
        self.mode = mode
        self.height = 480
        self.width = 640
        self.num_bins = num_bins
        self.clip_len = clip_len
        #the duration of a voxel
        self.delta_t_us = delta_t_ms * 1000

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)
        
        # Set reader lazily
        self.events_file = self.seq_path / "events.h5"
        self.reader = None
    
        # TODO: load supervision data
        gt_poses_fn = self.seq_path / "anchor_poses.txt"
        anchor_poses = np.loadtxt(gt_poses_fn, dtype=np.float64)
        gt_rel_transf_fn = self.seq_path / "relative_motions.txt"
        rel_transf = np.loadtxt(gt_rel_transf_fn, dtype=np.float64)

        if anchor_poses.shape[1] != 8:
            raise ValueError(
                f"Expected 8 columns for anchor poses [t px py pz qx qy qz qw], got {anchor_poses.shape[1]}"
            )
        self.anchors_us = anchor_poses[:, 0]
        
        if anchor_poses.shape[1] != 8:
            raise ValueError(
                f"Expected 14 columns [t0_us t1_us r11 r12 r13 px r21 r22 r23 py r31 r32 r33 pz], got {anchor_poses.shape[1]}"
            )
        self.rel_transf = rel_transf

    
    def _ensure_reader(self):
        if self.reader is None:
            self.reader = EDSReader(self.events_file)
    
    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def __del__(self):
        self.close()

    def __len__(self):
        # The number of samples of the dataset: 
        #Our supervision is made up of transforms in between voxels
        return self.anchors_us.shape - self.clip_len + 1

    def _empty_voxel(self):
        return torch.zeros(
            (self.num_bins, self.height, self.width),
            dtype=torch.float32
        )

    def events_to_voxel_grid(self, x, y, p, t):
        if len(t) == 0:
            return self._empty_voxel()

        t = t.astype(np.float32)
        t = t - t[0]

        if t[-1] > 0:
            t = t / t[-1]
        else:
            t = np.zeros_like(t, dtype=np.float32)
        
        event_data = {
            'x': x.astype(np.float32),
            'y': y.astype(np.float32),
            'p': p.astype(np.float32),
            't': t
        }

        return self.voxel_grid.convert_events(event_data)
    
    def get_relative_motion(self, t0_idx, t0, t1):
        
        #Check also the timestamps match
        assert (t0 == self.rel_transf[t0_idx, 0]) and (t1 == self.rel_transf[t0_idx, 1])
        #Get the actual supervision data
        return self.rel_transf[t0_idx, 2: ]


    def __getitem__(self, index):
        self._ensure_reader()

        #Get the anchor timestamps and build a voxel for each anchor
        anchors = self.anchors_us[index : index + self.clip_len]
        clip_voxels = []
        clip_targets = []

        for j, anchor in enumerate(anchors):
            ts_end_us = int(anchor)
            ts_start_us = ts_end_us - self.delta_t_us

            event_data = self.reader.get_events(ts_start_us, ts_end_us)

            if event_data is None:
                voxel = self._empty_voxel()
            else:
                voxel = self.events_to_voxel_grid(
                    event_data["x"],
                    event_data["y"],
                    event_data["p"],
                    event_data["t"],
                )

            clip_voxels.append(voxel.float())

            # target for pairwise motion: one target per transition
            if j < len(anchors) - 1:
                t0_idx = index + j
                t0 = int(anchors[j])
                t1 = int(anchors[j + 1])
                rel_target = self.get_relative_motion(t0_idx, t0, t1)   # shape [target_dim - 12]
                clip_targets.append(torch.as_tensor(rel_target, dtype=torch.float32))

        # stack voxels: list of [num_bins, H, W] -> [num_bins, clip_len, H, W]
        clip = torch.stack(clip_voxels, dim=1)

        # stack targets: list of [target_dim] -> [clip_len - 1, target_dim]
        target = torch.stack(clip_targets, dim=0)

        output = {
            "representation": clip,                              # [num_bins, clip_len, H, W]
            "anchors_us": torch.as_tensor(anchors, dtype=torch.int64),
            "target": target,                                    # [clip_len - 1, target_dim]
        }

        return output