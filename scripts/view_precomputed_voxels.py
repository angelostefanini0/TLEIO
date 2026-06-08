from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Play precomputed Tartan event voxel clips.")
    parser.add_argument("--sequence", default="competition_Test_ME000")
    parser.add_argument("--root", type=Path, default=Path("data/tartanair/precomputed_test"))
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--interval", type=int, default=40)
    parser.add_argument("--combine", choices=("sum", "maxabs"), default="sum")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--gain", type=float, default=1.0)
    args = parser.parse_args()

    path = args.root / args.sequence / "derotated_voxels.npy"
    voxels = np.load(path, mmap_mode="r")
    print(f"Loaded {path}")
    print(f"shape: {voxels.shape}")

    step = max(1, int(args.step))

    def render_rgb(frame_number: int) -> tuple[int, np.ndarray]:
        frame_idx = (frame_number * step) % voxels.shape[0]
        clip = voxels[frame_idx]
        if args.combine == "maxabs":
            strongest_bin = np.argmax(np.abs(clip), axis=0)
            frame = np.take_along_axis(clip, strongest_bin[None, ...], axis=0)[0]
        else:
            frame = clip.sum(axis=0)
        scale = float(np.percentile(np.abs(frame), args.percentile))
        if scale < 1e-6:
            scale = 1.0
        red = args.gain * np.clip(frame, 0.0, None) / scale
        blue = args.gain * np.clip(-frame, 0.0, None) / scale
        rgb = np.ones((*frame.shape, 3), dtype=np.float32)
        red = np.clip(red, 0.0, 1.0)
        blue = np.clip(blue, 0.0, 1.0)
        rgb[..., 1] -= red
        rgb[..., 2] -= red
        rgb[..., 0] -= blue
        rgb[..., 1] -= blue
        rgb = np.clip(rgb, 0.0, 1.0)
        return frame_idx, rgb

    if args.save_dir is not None:
        count = args.num_frames if args.num_frames > 0 else min(50, voxels.shape[0])
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for frame_number in range(count):
            frame_idx, rgb = render_rgb(frame_number)
            out_path = args.save_dir / f"{args.sequence}_voxel_{frame_idx:05d}.png"
            plt.imsave(out_path, rgb)
        print(f"Saved {count} frames to {args.save_dir}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(np.ones((*voxels.shape[-2:], 3), dtype=np.float32), vmin=0.0, vmax=1.0)
    title = ax.set_title(args.sequence)
    ax.axis("off")

    def update(frame_number: int):
        frame_idx, rgb = render_rgb(frame_number)
        image.set_data(rgb)
        title.set_text(f"{args.sequence} | voxel {frame_idx}/{voxels.shape[0]}")
        return image, title

    anim = FuncAnimation(fig, update, interval=args.interval, cache_frame_data=False)
    fig._voxel_animation = anim
    plt.show()


if __name__ == "__main__":
    main()
