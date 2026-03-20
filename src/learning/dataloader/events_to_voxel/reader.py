from pathlib import Path
from typing import Optional, Dict

import h5py
import numpy as np

from src.learning.dataloader.representation.event_slicer import EventSlicer


class EDSReader:
    """
    Pure reader for EDS event data.

    Responsibilities:
    - Open events.h5
    - Create EventSlicer
    - Provide temporal slicing interface
    """

    def __init__(self, events_file: str | Path):
        self.events_file = Path(events_file)
    
        self.h5f = h5py.File(self.events_file, "r")

        self.slicer = EventSlicer(self.h5f)


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

        return self.slicer.get_events(t_start_us, t_end_us)


    def get_start_time_us(self) -> int:
        return self.slicer.get_start_time_us()

    def get_final_time_us(self) -> int:
        return self.slicer.get_final_time_us()


    def close(self):
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()