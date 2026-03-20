import torch
import numpy as np
from typing import Dict

class EventRepresentation:
    def convert(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def convert_events(self, events: Dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
        x = events["x"]
        y = events["y"]
        pol = events["p"]
        time = events["t"]

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x.astype(np.float32))
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y.astype(np.float32))
        if not isinstance(pol, torch.Tensor):
            pol = torch.from_numpy(pol.astype(np.float32))
        if not isinstance(time, torch.Tensor):
            time = torch.from_numpy(time.astype(np.float32))

        return self.convert(x, y, pol, time)


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.nb_channels = channels
        self.height = height
        self.width = width
        self.normalize = normalize

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.nb_channels, self.height, self.width
        device = x.device
        
        with torch.no_grad():
            voxel_grid = torch.zeros((C, H, W), dtype=torch.float, device=device)

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

            if self.normalize:
                mask_nz = voxel_grid != 0
                if mask_nz.any():
                    mean = voxel_grid[mask_nz].mean()
                    std = voxel_grid[mask_nz].std()
                    if std > 1e-5:
                        voxel_grid[mask_nz] = (voxel_grid[mask_nz] - mean) / std
                    else:
                        voxel_grid[mask_nz] = voxel_grid[mask_nz] - mean

        return voxel_grid