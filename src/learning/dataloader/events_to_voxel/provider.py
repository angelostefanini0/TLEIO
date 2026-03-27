from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.learning.dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset

from ..representation.voxel_grid import VoxelGrid
from .reader import EDSReader


def build_loader(
    root_path: Path,
    batch_size=4,
    num_workers=0,
    delta_t_ms=50,
    num_bins=15,
    clip_len=3,
):
    
    dataset = MultiEventVoxelClipDataset(
        root_path=root_path,
        delta_t_ms=delta_t_ms,
        num_bins=num_bins,
        clip_len=clip_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader