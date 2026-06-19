#!/usr/bin/env python3
"""Play RGB frames with the trajectory history reprojected as points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eyecatcher.create_reprojected_trajectory_eyecatcher import (
    project_history,
    test_depth_visibility,
)
from eyecatcher.create_tartanair_office_eyecatcher import collect_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=Path("data/tartanair/office/Easy/P000"),
    )
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--occlusion-tolerance", type=float, default=0.35)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("eyecatcher/output/candidates"),
    )
    return parser.parse_args()


def trajectory_color(index: int, count: int) -> tuple[int, int, int]:
    alpha = index / max(1, count - 1)
    start_bgr = np.array([239, 174, 0], dtype=np.float64)
    end_bgr = np.array([0, 49, 255], dtype=np.float64)
    color = np.rint((1.0 - alpha) * start_bgr + alpha * end_bgr).astype(int)
    return tuple(map(int, color))


def render_frame(
    frame_bgr: np.ndarray,
    depth: np.ndarray,
    positions: np.ndarray,
    quats: np.ndarray,
    frame_index: int,
    point_radius: int,
    occlusion_tolerance: float,
) -> tuple[np.ndarray, int]:
    height, width = frame_bgr.shape[:2]
    pixels, in_frame, forward = project_history(
        positions, quats, frame_index, width, height
    )
    visible = test_depth_visibility(
        pixels, in_frame, forward, depth, occlusion_tolerance
    )

    output = frame_bgr.copy()
    for index in np.flatnonzero(visible):
        center = tuple(np.rint(pixels[index]).astype(int))
        cv2.circle(output, center, point_radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(
            output,
            center,
            point_radius,
            trajectory_color(index, len(pixels)),
            -1,
            cv2.LINE_AA,
        )

    label = f"frame {frame_index:04d}"
    controls = "SPACE pause | arrows step | S save | Q quit"
    cv2.rectangle(output, (10, 10), (245, 64), (0, 0, 0), -1)
    cv2.putText(output, label, (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, controls, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, controls, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return output, int(visible.sum())


def main() -> None:
    args = parse_args()
    image_files, depth_files, positions, quats = collect_files(args.sequence_root)
    last_frame = len(image_files) - 1
    frame_index = int(np.clip(args.start_frame, 1, last_frame))
    delay_ms = max(1, round(1000.0 / args.fps))
    paused = False
    window_name = "Reprojected trajectory"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame_bgr = cv2.imread(str(image_files[frame_index]), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise FileNotFoundError(f"Could not read {image_files[frame_index]}")
            depth = np.load(depth_files[frame_index], mmap_mode="r")
            display, visible_count = render_frame(
                frame_bgr,
                depth,
                positions,
                quats,
                frame_index,
                args.point_radius,
                args.occlusion_tolerance,
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKeyEx(0 if paused else delay_ms)
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key in (ord("s"), ord("S")):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                output_path = args.save_dir / f"frame_{frame_index:06d}.png"
                cv2.imwrite(str(output_path), display)
                print(f"Saved {output_path} ({visible_count} visible trajectory points)")
                paused = True
                continue

            # OpenCV returns platform-dependent codes for arrow keys.
            if key in (81, 2424832, ord("a"), ord("A")):
                frame_index = max(1, frame_index - 1)
                paused = True
            elif key in (83, 2555904, ord("d"), ord("D")):
                frame_index = min(last_frame, frame_index + 1)
                paused = True
            elif not paused:
                frame_index = min(last_frame, frame_index + 1)
                if frame_index == last_frame:
                    paused = True
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
