from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import bisect

from ..representation.voxel_grid import VoxelGrid
from .reader import EDSReader

from ..utils.io import *

class MultiEventVoxelClipDataset(Dataset):
    # We use the voxel grid representation + clipping for TSformer-VO
    #
    # This class assumes the following structure in a sequence directory:
    #
    # processed
    # ├──seq_name (e.g. 01_peanuts_light)
    # ├     ├── events.h5
    # ├     | └── events/y
    # ├     | └── events/t
    # ├     | └── events/p
    # ├     | ├── events/x
    # ├     | └── ms_to_idx
    # ├     ├── imu.csv
    # ├     ├── stamped_groundtruth.csv
    # ├──seq_name (e.g. 02_rocket_earth_light)
    # ...


    def __init__(self, 
                 root_path: Path, 
                 delta_t_ms: int=50, 
                 num_bins: int=5, 
                 clip_len: int = 3):
        
        assert num_bins >= 1
        assert clip_len >= 1
        assert delta_t_ms <= 100, 'if duration is higher than 100 ms'
        assert root_path.is_dir()


        self.seq_infos = []
        self.cum_lengths = []
        self._readers = []
        total = 0

        # Set constants
        self.root_path = root_path
        self.height = 480
        self.width = 640
        self.num_bins = num_bins
        self.clip_len = clip_len
        #the duration of a voxel
        self.delta_t_us = delta_t_ms * 1000

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        #Load lightweight metadata
    
        total = 0
        sequence_dirs = sorted([p for p in self.root_path.iterdir() if p.is_dir()])
        for seq_path in sequence_dirs:
            gt_poses_fn = seq_path / "anchor_poses.txt"
            gt_rel_transf_fn = seq_path / "relative_motions.txt"
            events_file = seq_path / "events.h5"

            # skip folders that do not contain a valid processed sequence
            if not (gt_poses_fn.exists() and gt_rel_transf_fn.exists() and events_file.exists()):
                continue

            anchor_poses = np.atleast_2d(np.loadtxt(gt_poses_fn, dtype=np.float64, skiprows=1))
            rel_transf = np.atleast_2d(np.loadtxt(gt_rel_transf_fn, dtype=np.float64, skiprows=1))

            if anchor_poses.shape[1] != 8:
                raise ValueError(
                    f"{seq_path}: expected 8 columns in anchor_poses.txt, got {anchor_poses.shape[1]}"
                )

            if rel_transf.shape[1] != 9:
                raise ValueError(
                    f"{seq_path}: expected 9 columns in relative_motions.txt, got {rel_transf.shape[1]}"
                )

            anchors_us = anchor_poses[:, 0]
            num_samples = max(0, len(anchors_us) - self.clip_len + 1)
            self.seq_infos.append({
                "seq_path": seq_path,
                "events_file": events_file,
                "anchors_us": anchors_us,
                "rel_transf": rel_transf,
                "num_samples": num_samples,
            })

            total += num_samples
            self.cum_lengths.append(total)

        self._readers = [None] * len(self.seq_infos)

    
    def _ensure_reader(self, seq_idx):
        if self._readers[seq_idx] is None:
            events_file = self.seq_infos[seq_idx]["events_file"]
            self._readers[seq_idx] = EDSReader(events_file)
    
    def close(self):
        if not hasattr(self, "_readers"):
            return
        for i, reader in enumerate(self._readers):
            if reader is not None:
                reader.close()
                self._readers[i] = None

    def __del__(self):
        self.close()

    def __len__(self):
        # The number of samples of the dataset: 
        #Our supervision is made up of transforms in between voxels
        return self.cum_lengths[-1] if self.cum_lengths else 0
    
    def locate_index(self, idx, cum_lengths):
        seq_idx = bisect.bisect_right(cum_lengths, idx)
        prev_cum = 0 if seq_idx == 0 else cum_lengths[seq_idx - 1]
        local_idx = idx - prev_cum
        return seq_idx, local_idx
    
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
    
    def get_relative_motion(self, rel_transf, t0_idx, t0, t1):
        
        #Check also the timestamps matches
        assert (t0 == rel_transf[t0_idx, 0]) and (t1 == rel_transf[t0_idx, 1])
        #Get the actual supervision data
        return rel_transf[t0_idx, 2: ]


    def __getitem__(self, index):
        seq_idx, local_idx = self.locate_index(index, self.cum_lengths)
        self._ensure_reader(seq_idx)

        #Get the anchor timestamps and build a voxel for each anchor
        seq_anchors = self.seq_infos[seq_idx]["anchors_us"]
        anchors = seq_anchors[local_idx : local_idx + self.clip_len]
        clip_voxels = []
        clip_targets = []

        for j, anchor in enumerate(anchors):
            ts_end_us = int(anchor)
            ts_start_us = ts_end_us - self.delta_t_us

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
                )

            clip_voxels.append(voxel.float())

            # target for pairwise motion: one target per transition
            if j < len(anchors) - 1:
                t0_idx = local_idx + j
                t0 = int(anchors[j])
                t1 = int(anchors[j + 1])
                rel_transf = self.seq_infos[seq_idx]["rel_transf"]
                rel_target = self.get_relative_motion(rel_transf, t0_idx, t0, t1)   # shape [target_dim (7)]
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
