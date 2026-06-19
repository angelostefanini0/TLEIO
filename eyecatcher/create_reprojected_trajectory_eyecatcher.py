#!/usr/bin/env python3
"""Overlay a TartanAir camera trajectory on a well-chosen RGB frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eyecatcher.create_tartanair_office_eyecatcher import (
    TARTANAIR_CX,
    TARTANAIR_CY,
    TARTANAIR_FX,
    TARTANAIR_FY,
    collect_files,
    quat_xyzw_to_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=Path("data/tartanair/office/Easy/P000"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eyecatcher/output/office_reprojected_trajectory.png"),
    )
    parser.add_argument("--frame", type=int, default=None, help="Frame index; auto-select when omitted.")
    parser.add_argument(
        "--min-progress",
        type=float,
        default=0.30,
        help="Ignore frames before this fraction of the sequence during auto-selection.",
    )
    parser.add_argument("--line-width", type=float, default=4.0)
    parser.add_argument("--halo-width", type=float, default=7.0)
    parser.add_argument("--supersampling", type=int, default=3)
    parser.add_argument(
        "--occlusion-tolerance",
        type=float,
        default=0.35,
        help="Depth tolerance in metres for trajectory visibility testing.",
    )
    return parser.parse_args()


def project_history(
    positions: np.ndarray,
    quats: np.ndarray,
    frame_index: int,
    width: int,
    height: int,
    border: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_world = positions[:frame_index]
    world_to_camera = quat_xyzw_to_matrix(quats[frame_index]).T
    history_camera = (world_to_camera @ (history_world - positions[frame_index]).T).T

    forward = history_camera[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = TARTANAIR_FX * history_camera[:, 1] / forward + TARTANAIR_CX
        v = TARTANAIR_FY * history_camera[:, 2] / forward + TARTANAIR_CY
    pixels = np.column_stack((u, v))
    visible = (
        (forward > 0.15)
        & np.isfinite(pixels).all(axis=1)
        & (u >= border)
        & (u < width - border)
        & (v >= border)
        & (v < height - border)
    )
    return pixels, visible, forward


def test_depth_visibility(
    pixels: np.ndarray,
    in_frame: np.ndarray,
    forward: np.ndarray,
    depth: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    visible = np.zeros_like(in_frame)
    indices = np.flatnonzero(in_frame)
    if not len(indices):
        return visible

    u = np.rint(pixels[indices, 0]).astype(np.int64).clip(0, depth.shape[1] - 1)
    v = np.rint(pixels[indices, 1]).astype(np.int64).clip(0, depth.shape[0] - 1)
    surface_depth = depth[v, u]
    # The small relative term accommodates depth discretization at longer range.
    visible[indices] = forward[indices] <= surface_depth + tolerance + 0.03 * forward[indices]
    return visible


def select_frame(
    positions: np.ndarray,
    quats: np.ndarray,
    depth_files: list[Path],
    width: int,
    height: int,
    min_progress: float,
    occlusion_tolerance: float,
) -> tuple[int, float, float]:
    first_frame = max(2, int(round(min_progress * (len(positions) - 1))))
    best: tuple[float, int, float, float] | None = None

    for frame_index in range(first_frame, len(positions)):
        pixels, in_frame, forward = project_history(positions, quats, frame_index, width, height)
        depth = np.load(depth_files[frame_index], mmap_mode="r")
        visible = test_depth_visibility(
            pixels, in_frame, forward, depth, occlusion_tolerance
        )
        visible_count = int(visible.sum())
        if visible_count < 2:
            continue
        visibility = visible_count / len(visible)
        extent = np.linalg.norm(np.ptp(pixels[visible], axis=0)) / np.hypot(width, height)
        score = visibility * (0.75 + 0.25 * min(extent, 1.0))
        candidate = (score, frame_index, visibility, extent)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        raise ValueError("No frame contains enough projected trajectory points.")
    _, frame_index, visibility, extent = best
    return frame_index, visibility, extent


def blend_color(start: tuple[int, int, int], end: tuple[int, int, int], alpha: float) -> tuple[int, int, int, int]:
    rgb = tuple(round((1.0 - alpha) * a + alpha * b) for a, b in zip(start, end))
    return (*rgb, 245)


def render_overlay(
    image: Image.Image,
    pixels: np.ndarray,
    visible: np.ndarray,
    line_width: float,
    halo_width: float,
    supersampling: int,
) -> Image.Image:
    scale = max(1, supersampling)
    canvas = image.convert("RGBA").resize(
        (image.width * scale, image.height * scale), Image.Resampling.LANCZOS
    )
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    scaled_pixels = pixels * scale
    line_px = max(1, round(line_width * scale))
    halo_px = max(line_px, round(halo_width * scale))

    segments = [
        (index, tuple(scaled_pixels[index]), tuple(scaled_pixels[index + 1]))
        for index in range(len(pixels) - 1)
        if visible[index] and visible[index + 1]
    ]
    for _, start, end in segments:
        draw.line((start, end), fill=(255, 255, 255, 205), width=halo_px)
    denominator = max(1, len(pixels) - 1)
    for index, start, end in segments:
        color = blend_color((0, 174, 239), (255, 49, 0), index / denominator)
        draw.line((start, end), fill=color, width=line_px)

    visible_indices = np.flatnonzero(visible)
    if len(visible_indices):
        for index, radius, color in (
            (visible_indices[0], 5.5, (0, 145, 220, 255)),
            (visible_indices[-1], 6.5, (255, 49, 0, 255)),
        ):
            x, y = scaled_pixels[index]
            r = radius * scale
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 235))
            r *= 0.62
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    result = Image.alpha_composite(canvas, overlay)
    return result.resize(image.size, Image.Resampling.LANCZOS).convert("RGB")


def main() -> None:
    args = parse_args()
    image_files, depth_files, positions, quats = collect_files(args.sequence_root)
    with Image.open(image_files[0]) as first_image:
        width, height = first_image.size

    if args.frame is None:
        frame_index, visibility, extent = select_frame(
            positions,
            quats,
            depth_files,
            width,
            height,
            args.min_progress,
            args.occlusion_tolerance,
        )
    else:
        frame_index = args.frame
        if not 1 <= frame_index < len(image_files):
            raise ValueError(f"--frame must be between 1 and {len(image_files) - 1}.")
        pixels, in_frame, forward = project_history(positions, quats, frame_index, width, height)
        depth = np.load(depth_files[frame_index], mmap_mode="r")
        visible = test_depth_visibility(
            pixels, in_frame, forward, depth, args.occlusion_tolerance
        )
        visibility = float(visible.mean())
        extent = float(np.linalg.norm(np.ptp(pixels[visible], axis=0)) / np.hypot(width, height))

    pixels, in_frame, forward = project_history(positions, quats, frame_index, width, height)
    depth = np.load(depth_files[frame_index], mmap_mode="r")
    visible = test_depth_visibility(
        pixels, in_frame, forward, depth, args.occlusion_tolerance
    )
    with Image.open(image_files[frame_index]) as source_image:
        output_image = render_overlay(
            source_image,
            pixels,
            visible,
            args.line_width,
            args.halo_width,
            args.supersampling,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(args.output, quality=95)
    print(f"Selected frame: {frame_index} ({image_files[frame_index].name})")
    print(f"Visible history: {visible.sum()}/{len(visible)} ({100.0 * visibility:.1f}%)")
    print(f"Image-diagonal extent: {100.0 * extent:.1f}%")
    print(f"Wrote overlay: {args.output}")


if __name__ == "__main__":
    main()
