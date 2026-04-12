"""Store both EKF IMU packets and interpolated IMU windows for the network.

This file serves two purposes on the filter side:
1. keep the original interpolated-array utilities used for learned inputs;
2. provide a simple time-ordered queue of EKF IMU samples that the runner can
   propagate up to a requested timestamp.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class ImuMeasurement:
    """Represent one EKF IMU sample with timestamps and calibrated readings."""

    timestamp: float
    dt: float
    accel: np.ndarray
    gyro: np.ndarray


class ImuBuffer:
    """Store time-ordered IMU measurements for both network and EKF consumers."""

    def __init__(self):
        """Initialize the queue-backed EKF buffer and the array-backed net buffer."""

        self.net_t_us = np.array([], dtype=int)
        self.net_acc = np.array([])
        self.net_gyr = np.array([])
        self._imu_measurements = []

    def add(self, measurement):
        """Append one EKF IMU sample to the propagation queue in timestamp order."""

        timestamp = getattr(measurement, "timestamp", None)
        if timestamp is None:
            raise AttributeError("IMU measurements added to the EKF buffer need a `timestamp` attribute.")

        if self._imu_measurements and timestamp <= self._imu_measurements[-1].timestamp:
            raise ValueError("IMU measurements must be added to the EKF buffer in strictly increasing time order.")

        self._imu_measurements.append(measurement)

    def get_up_to(self, timestamp):
        """Pop and return all queued EKF IMU samples with time `<= timestamp`."""

        split_idx = 0
        while split_idx < len(self._imu_measurements):
            if self._imu_measurements[split_idx].timestamp > timestamp:
                break
            split_idx += 1

        measurements = self._imu_measurements[:split_idx]
        self._imu_measurements = self._imu_measurements[split_idx:]
        return measurements

    def add_data_interpolated(
        self, last_t_us, t_us, last_gyr, gyr, last_acc, acc, requested_interpolated_tus
    ):
        """Interpolate IMU arrays onto requested timestamps for the learned model."""

        assert isinstance(last_t_us, int)
        assert isinstance(t_us, int)
        if last_t_us < 0:
            acc_interp = acc.T
            gyr_interp = gyr.T
        else:
            try:
                acc_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_acc.T, acc.T]),
                    axis=0,
                )(requested_interpolated_tus)
                gyr_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_gyr.T, gyr.T]),
                    axis=0,
                )(requested_interpolated_tus)
            except ValueError as exc:
                print(
                    f"Trying to do interpolation at {requested_interpolated_tus} between {last_t_us} and {t_us}"
                )
                raise exc
        self._add_data(requested_interpolated_tus, acc_interp, gyr_interp)

    def _add_data(self, t_us: int, acc, gyr):
        """Append one interpolated timestamp and its IMU arrays to the net buffer."""

        assert isinstance(t_us, int)
        if len(self.net_t_us) > 0:
            assert (
                t_us > self.net_t_us[-1]
            ), f"trying to insert a data at time {t_us} which is before {self.net_t_us[-1]}"

        self.net_t_us = np.append(self.net_t_us, t_us)
        self.net_acc = np.append(self.net_acc, acc).reshape(-1, 3)
        self.net_gyr = np.append(self.net_gyr, gyr).reshape(-1, 3)

    def get_last_k_data(self, size):
        """Return the latest `k` interpolated IMU samples for the learned model."""

        net_acc = self.net_acc[-size:, :]
        net_gyr = self.net_gyr[-size:, :]
        net_t_us = self.net_t_us[-size:]
        return net_acc, net_gyr, net_t_us

    def get_data_from_to(self, t_begin_us: int, t_us_end: int):
        """Return interpolated IMU arrays between two exact buffered timestamps."""

        assert isinstance(t_begin_us, int)
        assert isinstance(t_us_end, int)
        begin_idx = np.searchsorted(self.net_t_us, t_begin_us)
        end_idx = np.where(self.net_t_us == t_us_end)[0][0]
        net_acc = self.net_acc[begin_idx : end_idx + 1, :]
        net_gyr = self.net_gyr[begin_idx : end_idx + 1, :]
        net_t_us = self.net_t_us[begin_idx : end_idx + 1]
        return net_acc, net_gyr, net_t_us

    def throw_data_before(self, t_begin_us: int):
        """Discard interpolated IMU data older than the requested timestamp."""

        assert isinstance(t_begin_us, int)
        begin_idx = np.where(self.net_t_us == t_begin_us)[0][0]
        self.net_acc = self.net_acc[begin_idx:, :]
        self.net_gyr = self.net_gyr[begin_idx:, :]
        self.net_t_us = self.net_t_us[begin_idx:]

    def total_net_data(self):
        """Return the number of interpolated IMU samples stored for the network."""

        return self.net_t_us.shape[0]

    def debugstring(self, query_us):
        """Print a compact summary of the interpolated timestamps in the buffer."""

        print(f"min:{self.net_t_us[0]}")
        print(f"max:{self.net_t_us[-1]}")
        print(f"que:{query_us}")
        print(f"all:{self.net_t_us}")
