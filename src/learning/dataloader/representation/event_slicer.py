"""HDF5-backed event slicing utilities."""

import math
from typing import Dict, Tuple

import h5py
from numba import jit
import numpy as np

#Taken kindly from DSEC dataloader 
class EventSlicer:
    """Read event windows from HDF5 event arrays using a millisecond index.

    ``events/x``, ``events/y``, and ``events/p`` are read from ``h5f``. Event
    timestamps and ``ms_to_idx`` may come from ``metadata_h5f`` when processed
    timestamps are stored in a sidecar file.
    """

    def __init__(self, h5f: h5py.File, metadata_h5f: h5py.File | None = None):
        """Initialize the event slicer.

        Args:
            h5f: HDF5 file containing raw event arrays.
            metadata_h5f: Optional HDF5 file containing processed timestamps,
                ``ms_to_idx``, ``t_offset``, and polarity metadata.
        """
        self.h5f = h5f
        self.metadata_h5f = metadata_h5f if metadata_h5f is not None else h5f

        self.events = dict()

        # x/y/p live in the raw events file, while t/ms_to_idx may come from a sidecar file.
        for dset_str in ['p', 'x', 'y']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]
        self.events['t'] = self.metadata_h5f['events/t']

        #The mapping from milliseconds to event index can be found in scripts/processing.py:
        self.ms_to_idx = np.asarray(self.metadata_h5f['ms_to_idx'], dtype='int64')
        self.normalize_polarity_to_binary = bool(
            self.metadata_h5f.attrs.get("normalize_polarity_to_binary", 0)
        )

        if "t_offset" in list(self.metadata_h5f.keys()):
            self.t_offset = int(self.metadata_h5f['t_offset'][()])
        else:
            self.t_offset = 0
        
        # ms_to_idx mapping expects times starting from 0 in ms
        # EDS dataset has a common time "frame", with large values. 
        # We process IMU and groundtruth data so that it's relative to the first event 
        # timestamp and in ms Event file is also processed so that the first timestamp is 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        """Return the first absolute timestamp represented by this slicer."""
        return self.t_offset

    def get_final_time_us(self):
        """Return the final absolute timestamp represented by this slicer."""
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events within the specified absolute time window.

        Args:
            t_start_us: Inclusive window start timestamp in microseconds.
            t_end_us: Exclusive window end timestamp in microseconds.

        Returns:
            A dictionary containing ``p``, ``x``, ``y``, and absolute ``t``
            arrays, or ``None`` if the requested window cannot be retrieved
            from the millisecond lookup table.
        """
        assert t_start_us < t_end_us

        # The times in EDS are all at the same "order of magnitude", so there is no offset
        # However we keep it for completeness
        # The inputs for the function are the starting and ending times in us  
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset 
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            if dset_str == 'p' and self.normalize_polarity_to_binary:
                events[dset_str] = (
                    (events[dset_str].astype(np.int8) + 1) // 2
                ).astype(np.uint8)
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative millisecond window for event lookup.

        ``ms_to_idx`` only indexes events at millisecond resolution, so the
        requested microsecond window is expanded outward before fine filtering.

        Args:
            ts_start_us: Start timestamp in microseconds.
            ts_end_us: End timestamp in microseconds.

        Returns:
            Conservative start and end timestamps in milliseconds.
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Find fine-grained offsets inside a conservative event slice.

        Args:
            time_array: Candidate event timestamps in microseconds.
            time_start_us: Inclusive start timestamp in microseconds.
            time_end_us: Exclusive end timestamp in microseconds.

        Returns:
            Start and end offsets such that, in non-edge cases,
            ``time_start_us <= time_array[idx_start:idx_end] < time_end_us``.
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        """Map a millisecond timestamp to the corresponding event index."""
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]
