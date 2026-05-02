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


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


class EventRepresentation:
    def convert(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
        metadata: Dict | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def convert_events(self, events: Dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
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
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    def convert_events(self, events: Dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
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

                #TODO: The de-rotation now happens before the trilinear voxel interpolation
                # This is the moment it has to be done 
                
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
