import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from learning.dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes", "y"}:
        return True
    if v.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute one derotated voxel per anchor."
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--delta_t_ms", type=int, default=50)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--downsampling_factor", type=float, default=1.0)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--denoising", type=str2bool, default=False)
    parser.add_argument("--denoise_dt_us", type=int, default=1000)
    parser.add_argument("--denoise_radius", type=int, default=1)
    parser.add_argument("--denoise_min_supporters", type=int, default=1)
    parser.add_argument("--denoise_same_polarity_only", type=str2bool, default=False)
    parser.add_argument("--derotate", type=str2bool, default=True)
    parser.add_argument("--derotation_slices", type=int, default=100)
    parser.add_argument("--voxel_filename", type=str, default="derotated_voxels.npy")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_sequence_voxels(dataset, seq_idx, output_dir, args):
    seq_info = dataset.seq_infos[seq_idx]
    seq_path = seq_info["seq_path"]
    anchors_us = seq_info["anchors_us"].astype(np.int64)
    out_seq_dir = output_dir / seq_path.name
    out_voxels = out_seq_dir / args.voxel_filename

    if out_voxels.exists() and not args.overwrite:
        raise FileExistsError(f"{out_voxels} already exists. Pass --overwrite to replace it.")

    out_seq_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seq_path / "relative_motions.txt", out_seq_dir / "relative_motions.txt")

    dtype = np.dtype(args.dtype)
    voxels = np.lib.format.open_memmap(
        out_voxels,
        mode="w+",
        dtype=dtype,
        shape=(len(anchors_us), dataset.num_bins, dataset.new_height, dataset.new_width),
    )

    dataset._ensure_reader(seq_idx)
    reader = dataset._readers[seq_idx]
    for i, anchor in enumerate(anchors_us):
        ts_end_us = int(anchor)
        ts_start_us = ts_end_us - dataset.delta_t_us
        events = reader.get_events(ts_start_us, ts_end_us)
        if events is None:
            voxel = dataset._empty_voxel()
        else:
            voxel = dataset.events_to_voxel_grid(
                events["x"],
                events["y"],
                events["p"],
                events["t"],
                ts_start_us=ts_start_us,
                ts_end_us=ts_end_us,
                seq_info=seq_info,
            )
        voxels[i] = voxel.cpu().numpy().astype(dtype, copy=False)

    voxels.flush()
    metadata = {
        "source_sequence": str(seq_path),
        "voxel_file": args.voxel_filename,
        "voxel_shape": list(voxels.shape),
        "voxel_format": "[N, C, H, W]",
        "window": "[anchor - delta_t, anchor)",
        "delta_t_ms": args.delta_t_ms,
        "num_bins": args.num_bins,
        "height": dataset.new_height,
        "width": dataset.new_width,
        "downsampling_factor": args.downsampling_factor,
        "denoising": args.denoising,
        "denoise_dt_us": args.denoise_dt_us,
        "denoise_radius": args.denoise_radius,
        "denoise_min_supporters": args.denoise_min_supporters,
        "denoise_same_polarity_only": args.denoise_same_polarity_only,
        "derotate": args.derotate,
        "derotation_slices": args.derotation_slices,
        "dtype": args.dtype,
    }
    with open(out_seq_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"{seq_path.name}: wrote {len(anchors_us)} voxels -> {out_voxels}")


def main():
    args = parse_args()
    source_root = Path(args.root_dir)
    output_dir = Path(args.output_dir)

    dataset = MultiEventVoxelClipDataset(
        root_path=source_root,
        delta_t_ms=args.delta_t_ms,
        num_bins=args.num_bins,
        clip_len=2,
        downsampling_factor=args.downsampling_factor,
        patch_size=args.patch_size,
        denoising=args.denoising,
        denoise_dt_us=args.denoise_dt_us,
        denoise_radius=args.denoise_radius,
        denoise_min_supporters=args.denoise_min_supporters,
        denoise_same_polarity_only=args.denoise_same_polarity_only,
        derotate=args.derotate,
        derotation_slices=args.derotation_slices,
    )

    try:
        for seq_idx in range(len(dataset.seq_infos)):
            write_sequence_voxels(dataset, seq_idx, output_dir, args)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
