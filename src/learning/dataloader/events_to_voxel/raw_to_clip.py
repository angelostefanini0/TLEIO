from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import bisect

from ..representation.voxel_grid import VoxelGrid
from .reader import EDSReader


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

    @staticmethod
    def get_downsampled_size(
        original_height: int,
        original_width: int,
        downsampling_factor: float,
        patch_size: int,
    ) -> tuple[int, int]:
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
                 patch_size: int = 16):
        
        assert num_bins >= 1
        assert clip_len >= 1
        assert delta_t_ms <= 100, 'if duration is higher than 100 ms'
        assert root_path.is_dir()


        self.seq_infos = []
        self.cum_lengths = []
        self._readers = []
        total = 0

        # Set constants
        self.patch_size = patch_size
        self.root_path = root_path
        self.original_height = 480
        self.original_width = 640
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
        self.clip_len = clip_len
        #the duration of a voxel
        self.delta_t_us = delta_t_ms * 1000

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.new_height, self.new_width, normalize=True)

        #Set the normalization stats to None: 
        self.train_std = None
        self.train_mean = None
        self.eps = 1e-7

        #Load lightweight data
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
            

            #Dimensionality checks
            if anchor_poses.shape[1] != 8:
                raise ValueError(
                    f"{seq_path}: expected 8 columns in anchor_poses.txt, got {anchor_poses.shape[1]}"
                )

            if rel_transf.shape[1] != 8:
                raise ValueError(
                    f"{seq_path}: expected 8 columns in relative_motions.txt, got {rel_transf.shape[1]}"
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
            (self.num_bins, self.new_height, self.new_width),
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
        
        downsampled_x = x * self.scale_x
        downsampled_y = y * self.scale_y
        event_data = {
            'x': downsampled_x.astype(np.float32),
            'y': downsampled_y.astype(np.float32),
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
                rel_target = self.get_relative_motion(rel_transf, t0_idx, t0, t1)   # shape [target_dim (6)]
                
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
