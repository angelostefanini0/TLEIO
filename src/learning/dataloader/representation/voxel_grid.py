"""
Event Data Representation Module

This module provides tools to convert asynchronous event-stream data (from sensors 
like DVS/DAVIS) into structured, grid-like representations for neural networks.

The primary implementation is the 'Voxel Grid', which aggregates events over 
time into a fixed number of temporal bins using trilinear interpolation 
across spatial and temporal dimensions.

Kindly inspired by DSEC implementation.
"""


from __future__ import annotations

from typing import Dict

import numpy as np
import torch


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
            x = torch.from_numpy(x)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)
        if not isinstance(pol, torch.Tensor):
            p = torch.from_numpy(pol)
        if not isinstance(time, torch.Tensor):
            t = torch.from_numpy(time)

        return self.convert(p, t, x, y)


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool = True):
        self.channels = channels
        self.height = height
        self.width = width
        self.normalize = normalize

    def convert(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        device = pol.device
        voxel_grid = torch.zeros(
            (self.channels, self.height, self.width),
            dtype=torch.float32,
            device=device,
        )

        if x.numel() == 0:
            return voxel_grid

        x = x.float()
        y = y.float()
        pol = pol.float()
        time = time.float()

        
        t_norm = (self.channels - 1) * (time - time[0]) / (time[-1] - time[0]) # Normalize time to [0, channels-1]

        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        t0 = torch.floor(t_norm).long()

        # polarity: {0,1} -> {-1,+1}
        value = 2 * pol - 1

        with torch.no_grad():
            for xlim in [x0, x0 + 1]:
                for ylim in [y0, y0 + 1]:
                    for tlim in [t0, t0 + 1]:
                        mask = (
                            (xlim >= 0) & (xlim < self.width) &
                            (ylim >= 0) & (ylim < self.height) &
                            (tlim >= 0) & (tlim < self.channels)
                        )

                        interp_weights = (
                            value
                            * (1 - (xlim.float() - x).abs())
                            * (1 - (ylim.float() - y).abs())
                            * (1 - (tlim.float() - t_norm).abs())
                        )

                        index = (
                            self.height * self.width * tlim
                            + self.width * ylim
                            + xlim
                        )

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].numel() > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

    def __call__(self, events: Dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
        return self.convert_events(events)