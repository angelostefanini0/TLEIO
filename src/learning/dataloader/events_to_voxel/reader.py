from pathlib import Path
from typing import Optional, Dict

import hdf5plugin  # noqa: F401  # Registers external HDF5 filters (e.g. Blosc).
import h5py
import numpy as np

from ..representation.event_slicer import EventSlicer


class EDSReader:
    """
    Pure reader for EDS event data.

    Responsibilities:
    - Open events.h5
    - Create EventSlicer
    - Provide temporal slicing interface
    """

    def __init__(self, events_file: str | Path, metadata_file: str | Path | None = None):
        self.events_file = Path(events_file)
        self.metadata_file = Path(metadata_file) if metadata_file is not None else None
        self.h5f = None
        self.metadata_h5f = None
        self.slicer = None

    def _ensure_open(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.events_file, "r")
            if self.metadata_file is None or self.metadata_file == self.events_file:
                self.metadata_h5f = self.h5f
            else:
                self.metadata_h5f = h5py.File(self.metadata_file, "r")
            self.slicer = EventSlicer(self.h5f, self.metadata_h5f)


    def get_events(
        self,
        t_start_us: int,
        t_end_us: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Get events in [t_start_us, t_end_us)

        Returns:
            dict with keys: x, y, t, p
            or None if window is invalid
        """
        self._ensure_open()
        return self.slicer.get_events(t_start_us, t_end_us)


    def get_start_time_us(self) -> int:
        return self.slicer.get_start_time_us()

    def get_final_time_us(self) -> int:
        return self.slicer.get_final_time_us()


    def close(self):
        same_handle = self.metadata_h5f is self.h5f
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None
        if self.metadata_h5f is not None and not same_handle:
            self.metadata_h5f.close()
            self.metadata_h5f = None
        else:
            self.metadata_h5f = None
        self.slicer = None

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()
