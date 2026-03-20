from pathlib import Path
import weakref

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from representation.voxel_grid import VoxelGrid
from representation.event_slicer import EventSlicer
from reader import EDSReader
from utils.io import *



class Sequence(Dataset):
    # NOTE: This is just an EXAMPLE class for convenience. Adapt it to your case.
    # In this example, we use the voxel grid representation.
    #
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_11_a)
    # ├── events.h5
    # | └── events/y
    # | └── events/t
    # | └── events/p
    # | ├── events/x
    # | └── ms_to_idx
    # ├── imu.csv
    # ├── stamped_groundtruth.csv


    def __init__(self, seq_path: Path, mode: str='train', delta_t_ms: int=50, num_bins: int=15):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert seq_path.is_dir()

        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        # Set EDS reader
        ev_data_file = seq_path / 'events.h5'
        self.reader = EDSReader(ev_data_file)

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        # load groundtruth 
        gt_path = seq_path / 'timestamps.txt'
        self.gt = load_gt(gt_path)
        self.timestamps = load_timestamps_from_gt(self.gt)

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        return len(self.disp_gt_pathstrings)

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def __getitem__(self, index):
        ts_end = self.timestamps[index]
        # ts_start should be fine (within the window as we removed the first disparity map)
        ts_start = ts_end - self.delta_t_us

        disp_gt_path = Path(self.disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)
        output = {
            'disparity_gt': self.get_disparity_map(disp_gt_path),
            'file_index': file_index,
        }
        for location in self.locations:
            event_data = self.event_slicers[location].get_events(ts_start, ts_end)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y, location)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
            if 'representation' not in output:
                output['representation'] = dict()
            output['representation'][location] = event_representation

        return output