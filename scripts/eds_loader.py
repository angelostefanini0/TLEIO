import glob
import logging
import os
from typing import Tuple

import cv2
import numpy as np
import yaml

from evlib.codec import fileformat


logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, data):
        self.intrinsics = np.eye(3)
        self.intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]] = data["intrinsics"]

        # distortion
        self.distortion_coeffs = np.array(data["distortion_coeffs"])
        self.distortion_model = data["distortion_model"]
        self.resolution = data["resolution"]

        if "T_cn_cnm1" not in data:
            self.R = np.eye(3)
        else:
            self.R = np.array(data['T_cn_cnm1'])[:3,:3]

        self.K = self.intrinsics

class CameraSystem:
    def __init__(self, data):
        # load calibration
        T = np.array(data['cam1']['T_cn_cnm1'])

        self.cam0 = Camera(data['cam0']) #RGB camera
        self.cam1 = Camera(data['cam1']) #event camera

        self.newR = self.cam1.R
        self.newK = self.cam0.K

        self.newres = tuple(self.cam0.resolution)

    def getMapping(self):

        img_mapx, img_mapy = cv2.initUndistortRectifyMap(self.cam0.K,
                                                        self.cam0.distortion_coeffs,
                                                        None,
                                                        self.newK  @ self.newR @ self.cam0.R.T,
                                                        self.newres,
                                                        cv2.CV_32FC1)

        ev_mapx, ev_mapy = cv2.initUndistortRectifyMap(self.cam1.K,
                                                    self.cam1.distortion_coeffs,
                                                    None,
                                                    self.newK  @ self.newR @ self.cam1.R.T,
                                                    self.newres,
                                                    cv2.CV_32FC1)

        W, H = self.cam1.resolution
        coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).T.reshape((-1, 1, 2)).astype("float32")
        points = cv2.undistortPoints(coords, self.cam1.K, self.cam1.distortion_coeffs, None, self.newR @ self.cam1.R.T, self.newK)
        inv_maps = points.reshape((H, W, 2))

        return {"img_mapx": img_mapx,
                "img_mapy": img_mapy,
                "ev_mapx": ev_mapx,
                "ev_mapy": ev_mapy,
                "inv_mapx": inv_maps[...,0],
                "inv_mapy": inv_maps[...,1]}

class EdsDataLoader():
    """Dataloader class for EDS dataset.
    See also: https://rpg.ifi.uzh.ch/eds.html#dataset
    """
    NAME = "EDS"

    def __init__(self, config: dict = {}):
        self._HEIGHT = config["height"]
        self._WIDTH = config["width"]
        root_dir: str = config["root"]
        self.root_dir: str = os.path.expanduser(root_dir)
        self.dataset_dir: str = os.path.join(self.root_dir)
        logger.info(f"Loading directory in {self.dataset_dir}")

    def __del__(self):
        try:
            self.opened_hdf5.close()
        except:
            pass

    def __len__(self):
        if self._len_cache is None:
            self.set_len_cache()
        return self._len_cache

    def index_to_time(self, index: int) -> float:
        """Event index to time"""
        if self._time_cache is None:
            self.set_time_cache()
        return self._time_cache[index]

    def time_to_index(self, time: float, *args, **kwargs) -> int:
        """Time to event index"""
        if self._time_cache is None:
            self.set_time_cache()
        ind = np.searchsorted(self._time_cache, time)
        return ind - 1

    def image_index_to_time(self, index: int) -> float:
        """Image index to time."""
        return self._image_cache["timestamp"][index]

    def time_to_image_index(self, time: float) -> int:
        """Time to image index. It returns the latest image index (before the timestamp).
        So if you want the index right after the timestamp, please +1 to the returned value.
        """
        ind = np.searchsorted(self._image_cache["timestamp"], time)
        return ind - 1

    def set_sequence(self, sequence_name: str) -> None:
        logger.info(f"Use sequence {sequence_name}")
        self.sequence_name = sequence_name
        self.dataset_files = self.get_sequence(sequence_name)
        logger.info("Optimize data loading speed. Loading events...")

        # Load calibration
        with open(os.path.join(self.dataset_dir, self.sequence_name, "K.yaml"), "r") as fh:
            cam_data = yaml.load(fh, Loader=yaml.SafeLoader)
        self.camsys = CameraSystem(cam_data)
        self.maps =self.camsys.getMapping()

        # Load only timestamps for searching events
        self.preloaded_event = fileformat.load_hdf5(self.dataset_files["event"],[("t", "t", np.float64), ])
        self.preloaded_event["t"] /= 1e6   # into [sec]
        self.opened_hdf5 = fileformat.open_hdf5(self.dataset_files["event"])
        self._time_cache = self.preloaded_event["t"]
        # Set length
        self._len_cache = len(self._time_cache)
        self.set_image_index()
        # Only get images where events exist.
        min_timestamp = max(self._image_cache["timestamp"][0], self.preloaded_event["t"][0])
        max_timestamp = min(self._image_cache["timestamp"][-1], self.preloaded_event["t"][-1])
        valid_ind = np.where((min_timestamp <= self._image_cache["timestamp"]) & (max_timestamp >= self._image_cache["timestamp"]))[0]
        valid_ind_start = valid_ind[0]
        valid_ind_end = valid_ind[-1]
        self._image_cache["timestamp"] = self._image_cache["timestamp"][valid_ind_start:valid_ind_end]
        self._image_cache["file_path"] = self._image_cache["file_path"][valid_ind_start:valid_ind_end]
        self._len_image = len(self._image_cache["timestamp"])

    def get_sequence(self, sequence_name: str) -> dict:
        """Get data inside a sequence.
        Inputs:
            sequence_name (str) ... name of the sequence. ex) `slider_depth`.
        Returns
           sequence_file (dict) ... dictionary of the filenames for the sequence.
        """
        # File paths, format check, and make cache.
        data_path = os.path.join(self.dataset_dir, sequence_name)
        ev_file = os.path.join(data_path, "events.h5")

        frame_dir = os.path.join(data_path, "images")
        image_list = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        image_list = [os.path.join(frame_dir, i) for i in image_list]
        ts_file = os.path.join(data_path, "images_timestamps.txt")

        return {"event": ev_file, "image_list": image_list, "image_timestamp": ts_file}

    def set_image_index(self):
        # Set image map 
        ts_iter = fileformat.IteratorTextTimestamps(self.dataset_files["image_timestamp"])
        ts_list = np.zeros((200000000, ), dtype=np.float64)
        cnt = 0
        for i in ts_iter:
            ts_list[cnt:cnt + i["num"]] = i["t"]
            cnt += i["num"]

        self._image_cache = {"timestamp": ts_list[:cnt] / 1e6, "file_path": self.dataset_files["image_list"]}
        assert len(self._image_cache["timestamp"]) == len(self._image_cache["file_path"])

    def load_event(self, start_index: int, end_index: int, *args, **kwargs) -> np.ndarray:
        """Load events from Hdf5 file.
        (timestamp x y polarity), but the x means in width direction, and y means in height direction.

        Returns:
            events (np.ndarray) ... Events. [x, y, t, p] where x is height.
            t is absolute value in second. p is [-1, +1].
        """
        n_events = end_index - start_index
        events = np.zeros((n_events, 4), dtype=np.float64)
        if len(self) <= start_index:
            logger.error(f"Specified {start_index} to {end_index} index for {len(self)}.")
            raise IndexError
        events[:, 2] = np.array(self.preloaded_event["t"][start_index:end_index])
        events[:, 0] = np.array(self.opened_hdf5["y"][start_index:end_index], dtype=np.int16)
        events[:, 1] = np.array(self.opened_hdf5["x"][start_index:end_index], dtype=np.int16)
        events[:, 3] = np.array(self.opened_hdf5["p"][start_index:end_index], dtype=bool)
        events[:, 3] = 2 * events[:, 3] - 1
        return events

    def load_image(self, index: int) -> Tuple[np.ndarray, float]:
        """Load image file and its timestamp
        Args:
            index (int): index of the image.
        Returns:
            Tuple[np.ndarray, float]: (image, timestamp)
        """
        assert index < self._len_image
        image_file = self._image_cache["file_path"][index]
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ts = self._image_cache["timestamp"][index]
        return image, ts
