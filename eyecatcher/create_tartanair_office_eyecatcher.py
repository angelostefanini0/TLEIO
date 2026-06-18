#!/usr/bin/env python3
"""Create a TartanAir office point-cloud eye-catcher.

The script fuses TartanAir v1 RGB-D frames using the provided camera poses and
writes a colored point cloud, a trajectory line set, and a simple preview image.
It defaults to the repo layout `data/tartanair/office/<difficulty>/<trajectory>`.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download.tartanair_utils import (
    TARTANAIR_FILE_LIST,
    HuggingFaceTartanAirDownloader,
    load_tartanair_file_sizes,
    select_tartanair_archives,
)

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

TARTANAIR_FX = 320.0
TARTANAIR_FY = 320.0
TARTANAIR_CX = 320.0
TARTANAIR_CY = 240.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TartanAir office point cloud and trajectory render.")
    parser.add_argument("--data-root", type=Path, default=Path("data/tartanair"))
    parser.add_argument("--sequence-root", type=Path, default=None)
    parser.add_argument("--difficulty", type=str, default="Easy", choices=["Easy", "Hard"])
    parser.add_argument("--trajectory", type=str, default="P000")
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download the required TartanAir office archives if the sequence is missing.",
    )
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded TartanAir zip archives.")
    parser.add_argument("--output-dir", type=Path, default=Path("eyecatcher/output"))
    parser.add_argument("--frame-stride", type=int, default=10, help="Use every N-th RGB-D frame.")
    parser.add_argument("--pixel-stride", type=int, default=4, help="Back-project every N-th pixel.")
    parser.add_argument("--max-frames", type=int, default=180, help="Maximum number of RGB-D frames to fuse.")
    parser.add_argument("--max-depth", type=float, default=25.0, help="Drop points farther than this many meters.")
    parser.add_argument("--voxel-size", type=float, default=0.08, help="Voxel size for point-cloud downsampling.")
    parser.add_argument(
        "--crop-to-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop reconstructed points to a bounding box around the camera trajectory.",
    )
    parser.add_argument("--trajectory-margin-xy", type=float, default=5.0)
    parser.add_argument("--trajectory-margin-z", type=float, default=2.5)
    parser.add_argument(
        "--trajectory-lift",
        type=float,
        default=0.12,
        help="Small visual z-offset for the rendered trajectory so it remains visible inside the point cloud.",
    )
    parser.add_argument("--trajectory-linewidth", type=float, default=3.0)
    parser.add_argument("--trajectory-halo-linewidth", type=float, default=6.0)
    parser.add_argument("--trajectory-alpha", type=float, default=0.95)
    parser.add_argument("--trajectory-marker-step", type=int, default=25)
    parser.add_argument("--trajectory-marker-size", type=float, default=10.0)
    parser.add_argument(
        "--trajectory-sample-spacing",
        type=float,
        default=0.0,
        help="If positive, render additional samples along the same trajectory centerline at this metric spacing.",
    )
    parser.add_argument(
        "--trajectory-sample-size",
        type=float,
        default=0.0,
        help="Marker size for the additional centerline samples. Use with --trajectory-sample-spacing.",
    )
    parser.add_argument("--point-size", type=float, default=0.14)
    parser.add_argument("--point-alpha", type=float, default=0.58)
    parser.add_argument(
        "--trim-percentile",
        type=float,
        default=0.2,
        help="Trim this percentile from each point-cloud axis after trajectory cropping.",
    )
    parser.add_argument("--max-render-points", type=int, default=180000)
    parser.add_argument("--view-elev", type=float, default=32.0, help="Matplotlib 3D view elevation in degrees.")
    parser.add_argument("--view-azim", type=float, default=-55.0, help="Matplotlib 3D view azimuth in degrees.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def resolve_sequence_root(args: argparse.Namespace) -> Path:
    if args.sequence_root is not None:
        sequence_root = args.sequence_root.resolve()
        if not is_tartanair_rgbd_sequence(sequence_root) and args.download_missing:
            download_required_tartanair_office_data(args, sequence_root)
        return sequence_root

    candidate = args.data_root / "office" / args.difficulty / args.trajectory
    if is_tartanair_rgbd_sequence(candidate):
        return candidate.resolve()

    if args.download_missing:
        download_required_tartanair_office_data(args, candidate)
        if is_tartanair_rgbd_sequence(candidate):
            return candidate.resolve()

    office_root = args.data_root / "office"
    matches = sorted(office_root.glob("*/*")) if office_root.exists() else []
    matches = [path for path in matches if is_tartanair_rgbd_sequence(path)]
    if matches:
        print(f"Default sequence not found, using: {matches[0]}")
        return matches[0].resolve()

    raise FileNotFoundError(
        "Could not find a TartanAir office RGB-D sequence. Expected a folder like\n"
        f"  {candidate}\n"
        "containing image_left/, depth_left/, and pose_left.txt.\n"
        "Download the TartanAir v1 office image_left/depth_left archives first."
    )


def download_required_tartanair_office_data(args: argparse.Namespace, sequence_root: Path) -> None:
    difficulty = args.difficulty.lower()
    archives = ["image_left.zip", "depth_left.zip", "flow_mask.zip"]
    file_sizes = load_tartanair_file_sizes(TARTANAIR_FILE_LIST)
    source_files = []

    for archive_name in archives:
        source_files.extend(
            select_tartanair_archives(
                file_sizes=file_sizes,
                env="office",
                difficulties=[difficulty],
                archive_name=archive_name,
            )
        )

    total_size = sum(file_sizes[source_file] for source_file in source_files)
    print("Downloading TartanAir office data required for the eye-catcher")
    print(f"  difficulty: {args.difficulty}")
    print(f"  trajectory: {args.trajectory}")
    print(f"  archives:   {', '.join(archives)}")
    print(f"  total zip size: {total_size:.3f} GB")

    zip_paths = HuggingFaceTartanAirDownloader().download(source_files, args.data_root)
    for zip_path in zip_paths:
        extract_selected_trajectory(zip_path, sequence_root, args.trajectory)
        if not args.keep_archives:
            zip_path.unlink(missing_ok=True)

    normalize_pose_file_location(sequence_root)


def extract_selected_trajectory(zip_path: Path, sequence_root: Path, trajectory: str) -> None:
    print(f"Extracting {trajectory} from {zip_path}")
    sequence_root.mkdir(parents=True, exist_ok=True)
    trajectory_lower = trajectory.lower()

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            info
            for info in zf.infolist()
            if not info.is_dir() and any(part.lower() == trajectory_lower for part in Path(info.filename).parts)
        ]

        if not members:
            print(f"  no {trajectory} members found; extracting full archive instead")
            zf.extractall(path=zip_path.parent)
            return

        for info in members:
            parts = Path(info.filename).parts
            traj_idx = next(idx for idx, part in enumerate(parts) if part.lower() == trajectory_lower)
            relative_parts = parts[traj_idx + 1 :]
            if not relative_parts:
                continue
            target = sequence_root.joinpath(*relative_parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())


def normalize_pose_file_location(sequence_root: Path) -> None:
    if (sequence_root / "pose_left.txt").exists():
        return
    matches = sorted(sequence_root.rglob("pose_left.txt"))
    if matches:
        matches[0].replace(sequence_root / "pose_left.txt")


def is_tartanair_rgbd_sequence(path: Path) -> bool:
    return (path / "image_left").is_dir() and (path / "depth_left").is_dir() and (path / "pose_left.txt").exists()


def load_pose_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    poses = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if poses.shape[1] != 7:
        raise ValueError(f"{path} must contain tx ty tz qx qy qz qw, got shape {poses.shape}.")
    return poses[:, :3], poses[:, 3:7]


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64, copy=True)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def collect_files(sequence_root: Path) -> tuple[list[Path], list[Path], np.ndarray, np.ndarray]:
    image_files = sorted((sequence_root / "image_left").glob("*.png"))
    depth_files = sorted((sequence_root / "depth_left").glob("*.npy"))
    positions, quats = load_pose_file(sequence_root / "pose_left.txt")

    count = min(len(image_files), len(depth_files), len(positions))
    if count == 0:
        raise ValueError(f"No RGB-D frames found in {sequence_root}.")
    if len(image_files) != len(depth_files) or len(image_files) != len(positions):
        print(
            "Warning: image/depth/pose counts differ; using common prefix length "
            f"{count} ({len(image_files)} images, {len(depth_files)} depths, {len(positions)} poses)."
        )
    return image_files[:count], depth_files[:count], positions[:count], quats[:count]


def backproject_frame(depth: np.ndarray, rgb: np.ndarray, position: np.ndarray, quat: np.ndarray, pixel_stride: int, max_depth: float):
    height, width = depth.shape
    ys, xs = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    z_forward = depth[ys, xs].astype(np.float64)
    colors = rgb[ys, xs].reshape(-1, 3)

    valid = np.isfinite(z_forward) & (z_forward > 0.05) & (z_forward < max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    x_right = (xs[valid].astype(np.float64) - TARTANAIR_CX) * z_forward[valid] / TARTANAIR_FX
    y_down = (ys[valid].astype(np.float64) - TARTANAIR_CY) * z_forward[valid] / TARTANAIR_FY

    # TartanAir poses use NED-style camera axes: x forward, y right, z down.
    points_cam = np.column_stack((z_forward[valid], x_right, y_down))
    points_world = (quat_xyzw_to_matrix(quat) @ points_cam.T).T + position
    return points_world.astype(np.float32), colors[valid.reshape(-1)].astype(np.uint8)


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx], colors[unique_idx]


def crop_points_to_trajectory(
    points: np.ndarray,
    colors: np.ndarray,
    trajectory: np.ndarray,
    margin_xy: float,
    margin_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = trajectory.min(axis=0) - np.array([margin_xy, margin_xy, margin_z])
    upper = trajectory.max(axis=0) + np.array([margin_xy, margin_xy, margin_z])
    keep = np.all((points >= lower) & (points <= upper), axis=1)
    return points[keep], colors[keep]


def trim_axis_outliers(points: np.ndarray, colors: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    if percentile <= 0 or len(points) == 0:
        return points, colors
    lower = np.percentile(points, percentile, axis=0)
    upper = np.percentile(points, 100.0 - percentile, axis=0)
    keep = np.all((points >= lower) & (points <= upper), axis=1)
    return points[keep], colors[keep]


def write_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        vertex.tofile(fh)


def write_trajectory_ply(path: Path, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    edge_count = max(0, len(positions) - 1)
    with path.open("w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(positions)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write(f"element edge {edge_count}\n")
        fh.write("property int vertex1\nproperty int vertex2\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for point in positions:
            fh.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        for idx in range(edge_count):
            fh.write(f"{idx} {idx + 1} 255 64 32\n")


def sample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    if spacing <= 0 or len(points) < 2:
        return np.empty((0, 3), dtype=np.float32)

    samples = []
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            continue
        count = max(1, int(np.ceil(length / spacing)))
        alpha = np.linspace(0.0, 1.0, count, endpoint=False, dtype=np.float64)
        samples.append(start[None, :] + alpha[:, None] * delta[None, :])

    if not samples:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(samples).astype(np.float32)


def render_preview(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    trajectory: np.ndarray,
    max_points: int,
    seed: int,
    trajectory_lift: float,
    trajectory_linewidth: float,
    trajectory_halo_linewidth: float,
    trajectory_alpha: float,
    trajectory_marker_step: int,
    trajectory_marker_size: float,
    trajectory_sample_spacing: float,
    trajectory_sample_size: float,
    point_size: float,
    point_alpha: float,
    view_elev: float,
    view_azim: float,
) -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points_plot = points[idx]
        colors_plot = colors[idx] / 255.0
    else:
        points_plot = points
        colors_plot = colors / 255.0

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2], c=colors_plot, s=point_size, alpha=point_alpha)
    trajectory_plot = trajectory + np.array([0.0, 0.0, trajectory_lift])
    if trajectory_halo_linewidth > 0:
        ax.plot(
            trajectory_plot[:, 0],
            trajectory_plot[:, 1],
            trajectory_plot[:, 2],
            color="white",
            linewidth=trajectory_halo_linewidth,
            alpha=0.88,
        )
    ax.plot(
        trajectory_plot[:, 0],
        trajectory_plot[:, 1],
        trajectory_plot[:, 2],
        color="#ff2600",
        linewidth=trajectory_linewidth,
        alpha=trajectory_alpha,
    )
    if trajectory_marker_step > 0 and trajectory_marker_size > 0:
        marker_points = trajectory_plot[::trajectory_marker_step]
        ax.scatter(
            marker_points[:, 0],
            marker_points[:, 1],
            marker_points[:, 2],
            color="#ff2600",
            s=trajectory_marker_size,
            alpha=min(1.0, trajectory_alpha + 0.05),
        )
    if trajectory_sample_spacing > 0 and trajectory_sample_size > 0:
        sampled_trajectory = sample_polyline(trajectory, trajectory_sample_spacing)
        if len(sampled_trajectory) > 0:
            sampled_trajectory = sampled_trajectory + np.array([0.0, 0.0, trajectory_lift])
            ax.scatter(
                sampled_trajectory[:, 0],
                sampled_trajectory[:, 1],
                sampled_trajectory[:, 2],
                color="#ff2600",
                s=trajectory_sample_size,
                alpha=min(1.0, trajectory_alpha + 0.05),
            )
    ax.scatter(trajectory_plot[0, 0], trajectory_plot[0, 1], trajectory_plot[0, 2], color="#1f77b4", s=46)
    ax.scatter(trajectory_plot[-1, 0], trajectory_plot[-1, 1], trajectory_plot[-1, 2], color="#111111", s=46)
    set_axes_equal(ax, np.vstack((points_plot, trajectory)))
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = np.percentile(points, 1, axis=0)
    maxs = np.percentile(points, 99, axis=0)
    centers = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def main() -> None:
    args = parse_args()
    sequence_root = resolve_sequence_root(args)
    image_files, depth_files, positions, quats = collect_files(sequence_root)

    frame_indices = np.arange(0, len(image_files), args.frame_stride)
    frame_indices = frame_indices[: args.max_frames]
    print(f"Sequence: {sequence_root}")
    print(f"Fusing {len(frame_indices)} frames out of {len(image_files)}")

    all_points, all_colors = [], []
    for counter, frame_idx in enumerate(frame_indices, start=1):
        rgb = np.asarray(Image.open(image_files[frame_idx]).convert("RGB"))
        depth = np.load(depth_files[frame_idx])
        points, colors = backproject_frame(
            depth=depth,
            rgb=rgb,
            position=positions[frame_idx],
            quat=quats[frame_idx],
            pixel_stride=args.pixel_stride,
            max_depth=args.max_depth,
        )
        all_points.append(points)
        all_colors.append(colors)
        if counter % 25 == 0 or counter == len(frame_indices):
            print(f"  processed {counter}/{len(frame_indices)} frames")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    print(f"Raw points: {len(points):,}")
    if args.crop_to_trajectory:
        points, colors = crop_points_to_trajectory(
            points,
            colors,
            positions,
            margin_xy=args.trajectory_margin_xy,
            margin_z=args.trajectory_margin_z,
        )
        print(f"After trajectory crop: {len(points):,}")
    points, colors = trim_axis_outliers(points, colors, args.trim_percentile)
    print(f"After outlier trim: {len(points):,}")
    points, colors = voxel_downsample(points, colors, args.voxel_size)
    print(f"Downsampled points: {len(points):,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_ply = args.output_dir / "office_scene.ply"
    trajectory_ply = args.output_dir / "office_trajectory.ply"
    preview_png = args.output_dir / "office_eyecatcher.png"

    write_pointcloud_ply(scene_ply, points, colors)
    write_trajectory_ply(trajectory_ply, positions)
    render_preview(
        preview_png,
        points,
        colors,
        positions,
        args.max_render_points,
        args.seed,
        args.trajectory_lift,
        args.trajectory_linewidth,
        args.trajectory_halo_linewidth,
        args.trajectory_alpha,
        args.trajectory_marker_step,
        args.trajectory_marker_size,
        args.trajectory_sample_spacing,
        args.trajectory_sample_size,
        args.point_size,
        args.point_alpha,
        args.view_elev,
        args.view_azim,
    )

    print(f"Wrote scene point cloud: {scene_ply}")
    print(f"Wrote trajectory:        {trajectory_ply}")
    print(f"Wrote preview render:    {preview_png}")


if __name__ == "__main__":
    main()
