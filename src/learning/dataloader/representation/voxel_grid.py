import cv2
import numpy as np
import torch
from typing import Dict

# Avoid OpenCV spawning its own thread pool inside each DataLoader worker.
cv2.setNumThreads(1)
if hasattr(cv2, "ocl"):
    cv2.ocl.setUseOpenCL(False)


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
    def __init__(self, channels: int, height: int, width: int, derotate: bool):
        self.nb_channels = channels
        self.height = height
        self.width = width
        self.derotate = derotate

    def derotate_voxel_grid(
        self,
        voxel_grid: torch.Tensor,
        camera_matrix: np.ndarray,
        bin_quat_xyzw: np.ndarray,
        ref_quat_xyzw: np.ndarray,
    ) -> torch.Tensor:
        C, H, W = voxel_grid.shape
        if len(bin_quat_xyzw) != C:
            raise ValueError(f"Expected {C} bin quaternions, got {len(bin_quat_xyzw)}.")

        K = np.asarray(camera_matrix, dtype=np.float64)
        K_inv = np.linalg.inv(K)
        R_ref = quat_xyzw_to_rotmat(np.asarray(ref_quat_xyzw, dtype=np.float64))

        voxel_np = voxel_grid.detach().cpu().numpy().astype(np.float32, copy=False)
        warped = np.empty((C, H, W), dtype=np.float32)

        for idx in range(C):
            if not np.any(voxel_np[idx]):
                warped[idx] = voxel_np[idx]
                continue

            R_bin = quat_xyzw_to_rotmat(np.asarray(bin_quat_xyzw[idx], dtype=np.float64))
            R_ref_from_bin = R_ref.T @ R_bin
            # Pinhole rotation warp for the accumulated temporal slice.
            H_ref_from_bin = K @ R_ref_from_bin @ K_inv
            H_ref_from_bin /= H_ref_from_bin[2, 2]

            warped[idx] = cv2.warpPerspective(
                voxel_np[idx],
                H_ref_from_bin,
                dsize=(W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        return torch.from_numpy(warped).to(voxel_grid.device)

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
        device = x.device

        with torch.no_grad():
            voxel_grid = torch.zeros((C, H, W), dtype=torch.float32, device=device)

            if self.derotate:
                # When de-rotation happens, the time has a slightly different meaning
                # Without de-rotation, the exact physical timestamp of each bin is not important.
                # With de-rotation, each temporal bin is later warped using the pose at that bin’s center. 
                # So the bin assignment must correspond to the real fixed window [ts_start_us, ts_end_us], 
                # not to [first_event_time, last_event_time].
                
                if metadata is None:
                    raise ValueError("Derotation requires metadata.")
                window_duration_us = float(metadata["window_duration_us"])
                if window_duration_us <= 0:
                    raise ValueError("window_duration_us must be positive.")
                t_norm = (C - 1) * time / window_duration_us
            else:
                t_min, t_max = time[0], time[-1]
                if t_max > t_min:
                    t_norm = (C - 1) * (time - t_min) / (t_max - t_min)
                else:
                    t_norm = torch.zeros_like(time)

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            value = 2 * pol - 1

            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dt in [0, 1]:
                        xlim, ylim, tlim = x0 + dx, y0 + dy, t0 + dt

                        mask = (xlim < W) & (xlim >= 0) & \
                               (ylim < H) & (ylim >= 0) & \
                               (tlim < C) & (tlim >= 0)

                        interp_weights = value * (1 - (xlim.float() - x).abs()) * \
                                                 (1 - (ylim.float() - y).abs()) * \
                                                 (1 - (tlim.float() - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.derotate:
                voxel_grid = self.derotate_voxel_grid(
                    voxel_grid=voxel_grid,
                    camera_matrix=np.asarray(metadata["camera_matrix"], dtype=np.float64),
                    bin_quat_xyzw=np.asarray(metadata["bin_quat_xyzw"], dtype=np.float64),
                    ref_quat_xyzw=np.asarray(metadata["ref_quat_xyzw"], dtype=np.float64),
                )
                
        return voxel_grid
