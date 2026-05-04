"""Voxel-grid event representation used by the online dataloader."""

import cv2
import numpy as np
import torch
from typing import Dict

# Avoid OpenCV spawning its own thread pool inside each DataLoader worker.
cv2.setNumThreads(1)
if hasattr(cv2, "ocl"):
    cv2.ocl.setUseOpenCL(False)

from .event_derotation import derotate_events_in_slices
from .trilinear_interpolation import trilinear_voxel_interpolation



class EventRepresentation:
    """Base interface for event representations."""

    def convert(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
        metadata: Dict | None = None,
    ) -> torch.Tensor:
        """Convert typed event tensors into an event representation."""
        raise NotImplementedError

    def convert_events(self, events: Dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
        """
        Convert a dictionary of event arrays into the representation tensor.

        The default implementation converts `x`, `y`, `p`, and `t` to
        `torch.float32` tensors and forwards all remaining keys as metadata.
        """
        x = events["x"]
        y = events["y"]
        pol = events["p"]
        time = events["t"]
        metadata = {k: v for k, v in events.items() if k not in {"x", "y", "p", "t"}}

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x.astype(np.float32))
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y.astype(np.float32))
        if not isinstance(pol, torch.Tensor):
            pol = torch.from_numpy(pol.astype(np.float32))
        if not isinstance(time, torch.Tensor):
            time = torch.from_numpy(time.astype(np.float32))

        return self.convert(x, y, pol, time, metadata=metadata)


class VoxelGrid(EventRepresentation):
    """
    Convert raw events into dense temporal voxel grids.

    In the standard path, event times are normalized between the first and last
    event in the requested window. In the de-rotation path, event timestamps are
    kept absolute until coordinates are warped into the reference pose; then
    time is normalized with respect to the fixed event window
    `[ts_start_us, ts_end_us)`.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        derotate: bool,
        derotation_slices: int | None = None,
    ):
        self.nb_channels = channels
        self.height = height
        self.width = width
        self.derotate = derotate
        self.derotation_slices = derotation_slices

    @staticmethod
    def _as_numpy_array(value: np.ndarray | torch.Tensor, dtype: np.dtype) -> np.ndarray:
        """Return ``value`` as a CPU numpy array with the requested dtype."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    def convert_events(self, events: Dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
        """
        Convert event dictionaries while preserving numpy arrays for de-rotation.

        The non-de-rotation path delegates to the base implementation where events 
        are converted to tensors straight away. 
        The de-rotation path keeps event arrays in numpy because the coordinate warp 
        uses numpy operations before the shared torch interpolation.
        """

        if not self.derotate:
            return super().convert_events(events)

        metadata = {k: v for k, v in events.items() if k not in {"x", "y", "p", "t"}}
        return self.convert(
            self._as_numpy_array(events["x"], np.float32),
            self._as_numpy_array(events["y"], np.float32),
            self._as_numpy_array(events["p"], np.float32),
            self._as_numpy_array(events["t"], np.float64),
            metadata=metadata,
        )

    def convert(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
        metadata: Dict | None = None,
    ):
        """
        Convert one event window into a `[C, H, W]` voxel grid.

        Args:
            x: Event x coordinates.
            y: Event y coordinates.
            pol: Event polarities encoded as 0/1.
            time: Event timestamps. For the non-de-rotation path these may be
                relative values. For the de-rotation path they must be absolute
                microsecond timestamps.
            metadata: Optional metadata required by de-rotation: `ts_start_us`,
                `ts_end_us`, `camera_matrix`, `bin_quat_xyzw`, and
                `ref_quat_xyzw`.

        Returns:
            A `torch.float32` voxel grid with shape
            `[channels, height, width]`.
        """
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.nb_channels, self.height, self.width
        with torch.no_grad():
            if self.derotate:
                # When de-rotation happens, the time has a slightly different meaning
                # Without de-rotation, the exact physical timestamp of each bin is not important.
                # With de-rotation, each temporal bin is later warped using the pose at that bin’s center. 
                # So the bin assignment must correspond to the real fixed window [ts_start_us, ts_end_us], 
                # not to [first_event_time, last_event_time].
                
                if metadata is None:
                    raise ValueError("Derotation requires metadata.")
                ts_start_us = int(metadata["ts_start_us"])
                ts_end_us = int(metadata["ts_end_us"])
                window_duration_us = float(metadata.get("window_duration_us", ts_end_us - ts_start_us))
                if window_duration_us <= 0:
                    raise ValueError("window_duration_us must be positive.")

                x_np = self._as_numpy_array(x, np.float32)
                y_np = self._as_numpy_array(y, np.float32)
                pol_np = self._as_numpy_array(pol, np.float32)
                time_np = self._as_numpy_array(time, np.float64)

                t_norm_np = (C - 1) * (time_np - float(ts_start_us)) / window_duration_us
                derot_x, derot_y, derot_valid, _ = derotate_events_in_slices(
                    x=x_np,
                    y=y_np,
                    t_us=time_np,
                    ts_start_us=ts_start_us,
                    ts_end_us=ts_end_us,
                    context=metadata,
                    height=H,
                    width=W,
                )
                x = torch.from_numpy(derot_x[derot_valid].astype(np.float32, copy=False))
                y = torch.from_numpy(derot_y[derot_valid].astype(np.float32, copy=False))
                pol = torch.from_numpy(pol_np[derot_valid].astype(np.float32, copy=False))
                t_norm = torch.from_numpy(t_norm_np[derot_valid].astype(np.float32, copy=False))

            else:
                t_min, t_max = time[0], time[-1]
                if t_max > t_min:
                    t_norm = (C - 1) * (time - t_min) / (t_max - t_min)
                else:
                    t_norm = torch.zeros_like(time)

            voxel_grid = trilinear_voxel_interpolation(
                x=x,
                y=y,
                pol=pol,
                t_norm=t_norm,
                channels=C,
                height=H,
                width=W,
            )
        return voxel_grid
