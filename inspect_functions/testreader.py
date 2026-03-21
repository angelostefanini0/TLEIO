import torch
import numpy as np

from src.learning.dataloader.events_to_voxel.reader import EDSReader
from src.learning.dataloader.representation.voxel_grid import VoxelGrid


def main():
    events_path = "data/eds/processed/peanuts/events.h5"

    height = 480
    width = 640
    channels = 5

    with EDSReader(events_path) as reader:
        t0 = reader.get_start_time_us()
        t1 = t0 + 10_000  # 10 ms

        print("Start time:", t0)
        print("Final time:", reader.get_final_time_us())
        print("Requested window:", (t0, t1))

        events = reader.get_events(t0, t1)

        if events is None:
            print("Finestra non valida")
            return

        print("Chiavi:", events.keys())
        print("Numero eventi:", len(events["t"]))

        if len(events["t"]) == 0:
            print("Nessun evento nella finestra")
            return

        print("Primo timestamp:", events["t"][0])
        print("Ultimo timestamp:", events["t"][-1])
        print("Primi x:", events["x"][:10])
        print("Primi y:", events["y"][:10])
        print("Prime p:", events["p"][:10])

        # diagnostica utile
        xy = np.stack([events["x"], events["y"]], axis=1)
        unique_xy = np.unique(xy, axis=0)
        print("Unique (x,y) locations:", unique_xy.shape[0])

        # conversione a torch
        x = torch.from_numpy(events["x"]).float()
        y = torch.from_numpy(events["y"]).float()
        p = torch.from_numpy(events["p"]).float()
        t = torch.from_numpy(events["t"]).float()

        voxelizer = VoxelGrid(
            channels=channels,
            height=height,
            width=width,
            normalize=False,
        )

        voxel = voxelizer.convert(x, y, p, t)

        print("\n--- Voxel grid info ---")
        print("Voxel shape:", voxel.shape)
        print("Voxel dtype:", voxel.dtype)
        print("Voxel device:", voxel.device)
        print("Voxel sum:", voxel.sum().item())
        print("Voxel abs sum:", voxel.abs().sum().item())
        print("Voxel min:", voxel.min().item())
        print("Voxel max:", voxel.max().item())

        nonzero = torch.count_nonzero(voxel).item()
        print("Nonzero voxels:", nonzero)

        print("\nNonzero voxels per channel:")
        for c in range(voxel.shape[0]):
            nz = torch.count_nonzero(voxel[c]).item()
            print(f"  channel {c}: {nz}")

        nz_idx = torch.nonzero(voxel)
        if nz_idx.numel() > 0:
            print("\nPrimi 10 voxel non zero [c, y, x]:")
            print(nz_idx[:10])

            print("\nValori corrispondenti:")
            for idx in nz_idx[:10]:
                c, yy, xx = idx.tolist()
                print(f"  voxel[{c}, {yy}, {xx}] = {voxel[c, yy, xx].item()}")

        print("\nPipeline reader + slicer + voxelization OK")


if __name__ == "__main__":
    main()