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
                 anchor_step_us : int = 50000, 
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
        #The time step in between voxels: could be different from the delta if we want overlap 
        self.anchor_step_us = anchor_step_us 
        #the duration of a voxel
        self.delta_t_us = delta_t_ms * 1000

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)
        
        # Set reader lazily
        self.events_file = self.seq_path / "events.h5"
        self.reader = None
    
        # TODO: load supervision data

    
    def _ensure_reader(self):
        if self.reader is None:
            self.reader = EDSReader(self.events_file)

    def __len__(self):
        # TODO: The number of samples of the dataset: 
        # Need to think how we're gonna act on this
        # Our supervision are transforms in between voxels
        return len(self.timestamps)

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
    
    def get_gt_for_timestamp(self, ts_end):
        # TODO: replace this with actual supervision logic.
        
        return self.gt[ts_end]


    def __getitem__(self, index):
        self._ensure_reader()

        #TODO: Get the anchor timestamps and build a voxel for each anchor
        anchors = self.anchors_us[index : index + self.clip_len]
        ts_end_us = int(self.timestamps[index])
        ts_start_us = ts_end_us - self.delta_t_us

        #Get the ground truth for supervision 
        target = self.get_gt_for_timestamp(ts_end_us)
       
        event_data = self.reader.get_events(ts_start_us, ts_end_us)

        if event_data is None:
            representation = self._empty_voxel()
        else:
            representation = self.events_to_voxel_grid(
                event_data["x"],
                event_data["y"],
                event_data["p"],
                event_data["t"],
            )
       
        output = {
            "representation": representation.float(),
            "timestamp_us": ts_end_us,
            "target": target,
        }

        return output