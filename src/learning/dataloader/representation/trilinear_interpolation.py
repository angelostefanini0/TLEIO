from __future__ import annotations

import torch


def trilinear_voxel_interpolation(
    x: torch.Tensor,
    y: torch.Tensor,
    pol: torch.Tensor,
    t_norm: torch.Tensor,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    assert x.shape == y.shape == pol.shape == t_norm.shape
    assert x.ndim == 1

    device = x.device
    with torch.no_grad():
        voxel_grid = torch.zeros((channels, height, width), dtype=torch.float32, device=device)

        x0 = x.int()
        y0 = y.int()
        t0 = t_norm.int()
        value = 2 * pol - 1

        for dx in [0, 1]:
            for dy in [0, 1]:
                for dt in [0, 1]:
                    xlim = x0 + dx
                    ylim = y0 + dy
                    tlim = t0 + dt

                    mask = (
                        (xlim < width)
                        & (xlim >= 0)
                        & (ylim < height)
                        & (ylim >= 0)
                        & (tlim < channels)
                        & (tlim >= 0)
                    )

                    interp_weights = (
                        value
                        * (1 - (xlim.float() - x).abs())
                        * (1 - (ylim.float() - y).abs())
                        * (1 - (tlim.float() - t_norm).abs())
                    )

                    index = height * width * tlim.long() + width * ylim.long() + xlim.long()
                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

    return voxel_grid
