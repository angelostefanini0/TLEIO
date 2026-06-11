"""Play raw TartanEvent windows beside a synchronized estimated trajectory."""

import argparse
from pathlib import Path
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect_relative_motions import load_table, translation_rel_to_T
from src.learning.dataloader.representation.event_slicer import EventSlicer
from src.spatial_math import (
    T_to_pose,
    interpolate_gt_pose,
    interpolate_quaternions,
    normalize_quat,
    pose_to_T,
    quat_to_rotmat,
)

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show raw TartanEvent data and the synchronized estimated XY trajectory."
    )
    parser.add_argument("--events", type=Path, required=True, help="Raw TartanEvent events.h5.")
    parser.add_argument(
        "--events-meta",
        type=Path,
        required=True,
        help="Processed events_meta.h5 containing relative timestamps and ms_to_idx.",
    )
    parser.add_argument(
        "--rel-model",
        type=Path,
        default=None,
        help="Predicted relative motions [t0_us t1_us px py pz ...].",
    )
    parser.add_argument(
        "--trajectory-estimate",
        type=Path,
        default=None,
        help="Final TLEIO trajectory [timestamp_s px py pz qx qy qz qw].",
    )
    parser.add_argument(
        "--rel-rotations",
        type=Path,
        default=None,
        help="Optional relative motions whose rotation vectors are used during reconstruction.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Optional stamped ground truth, shown for reference and used to initialize the trajectory.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--final-hold-s",
        type=float,
        default=2.0,
        help="Seconds to hold the final full-trajectory view.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=250000,
        help="Maximum events rendered per frame; dense windows are uniformly subsampled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional MP4/GIF output. Without it, the visualization is displayed live.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the window even when --output is provided.",
    )
    return parser.parse_args()


def reconstruct_trajectory(
    rel: np.ndarray,
    rel_rotations: np.ndarray | None,
    initial_position: np.ndarray,
    initial_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if rel.shape[1] not in {5, 8}:
        raise ValueError("--rel-model must have 5 or 8 columns.")
    if rel_rotations is not None:
        if rel_rotations.shape[1] != 8:
            raise ValueError("--rel-rotations must have 8 columns.")
        if len(rel_rotations) != len(rel) or not np.array_equal(
            rel_rotations[:, :2].astype(np.int64), rel[:, :2].astype(np.int64)
        ):
            raise ValueError("--rel-model and --rel-rotations timestamps must match.")

    transform = pose_to_T(initial_position, initial_quaternion)
    positions = [initial_position.copy()]
    quaternions = [initial_quaternion.copy()]
    for index, row in enumerate(rel):
        rotation_row = None if rel_rotations is None else rel_rotations[index]
        transform = transform @ translation_rel_to_T(row, rotation_row)
        position, quaternion = T_to_pose(transform)
        positions.append(position)
        quaternions.append(quaternion)
    return np.stack(positions), normalize_quat(np.stack(quaternions))


def event_frame(
    events: dict[str, np.ndarray],
    height: int,
    width: int,
    max_events: int,
) -> np.ndarray:
    frame = np.full((height, width, 3), 248, dtype=np.uint8)
    count = len(events["t"])
    if count == 0:
        return frame

    if max_events > 0 and count > max_events:
        indices = np.linspace(0, count - 1, max_events, dtype=np.int64)
    else:
        indices = slice(None)

    x = np.asarray(events["x"][indices], dtype=np.int64)
    y = np.asarray(events["y"][indices], dtype=np.int64)
    p = np.asarray(events["p"][indices])
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p = x[valid], y[valid], p[valid]

    frame[y[p <= 0], x[p <= 0]] = (35, 105, 210)
    frame[y[p > 0], x[p > 0]] = (215, 50, 50)
    return frame


def figure_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def set_3d_camera(
    axis,
    positions: np.ndarray,
    trajectory_index: int,
    full_center: np.ndarray,
    full_radius: float,
) -> float:
    """Follow the current pose closely, then ease out to the complete trajectory."""
    progress = trajectory_index / max(len(positions) - 1, 1)
    reveal = smoothstep((progress - 0.18) / 0.72)
    visible = positions[: trajectory_index + 1]

    recent_count = max(12, int(round(0.12 * len(positions))))
    recent = visible[-recent_count:]
    local_min = np.min(recent, axis=0)
    local_max = np.max(recent, axis=0)
    local_center = 0.5 * (local_min + local_max)
    local_radius = max(float(np.max(local_max - local_min)) * 0.9, full_radius * 0.07, 0.6)

    center = (1.0 - reveal) * local_center + reveal * full_center
    radius = (1.0 - reveal) * local_radius + reveal * full_radius
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)

    # A restrained camera orbit makes the 3D structure easier to read.
    azimuth = -62.0 + 18.0 * smoothstep(progress)
    elevation = 24.0 + 8.0 * np.sin(np.pi * progress)
    axis.view_init(elev=elevation, azim=azimuth)
    return progress, radius


def main() -> None:
    args = parse_args()
    if args.window_ms <= 0 or args.fps <= 0:
        raise ValueError("--window-ms and --fps must be positive.")

    if (args.rel_model is None) == (args.trajectory_estimate is None):
        raise ValueError("Provide exactly one of --rel-model or --trajectory-estimate.")

    rel = load_table(args.rel_model) if args.rel_model else None
    rel_rotations = load_table(args.rel_rotations) if args.rel_rotations else None
    trajectory_estimate = (
        load_table(args.trajectory_estimate) if args.trajectory_estimate else None
    )
    if trajectory_estimate is not None:
        if trajectory_estimate.shape[1] != 8:
            raise ValueError("--trajectory-estimate must have 8 columns.")
        anchor_timestamps = np.rint(trajectory_estimate[:, 0] * 1e6).astype(np.int64)
        estimated_positions = trajectory_estimate[:, 1:4]
        estimated_quaternions = normalize_quat(trajectory_estimate[:, 4:8])
    else:
        anchor_timestamps = np.concatenate(
            [rel[:1, 0].astype(np.int64), rel[:, 1].astype(np.int64)]
        )

    gt_positions = None
    if args.gt:
        gt = load_table(args.gt)
        if gt.shape[1] != 8:
            raise ValueError("--gt must contain timestamp, position, and quaternion (8 columns).")
        gt_timestamps = gt[:, 0].astype(np.int64)
        gt_positions = gt[:, 1:4]
        gt_quaternions = normalize_quat(gt[:, 4:8])
        initial_position, initial_quaternion = interpolate_gt_pose(
            gt_timestamps,
            gt_positions,
            gt_quaternions,
            anchor_timestamps[:1],
        )
        initial_position = initial_position[0]
        initial_quaternion = initial_quaternion[0]
        gt_anchor_positions, gt_anchor_quaternions = interpolate_gt_pose(
            gt_timestamps,
            gt_positions,
            gt_quaternions,
            anchor_timestamps,
        )
    else:
        initial_position = np.zeros(3, dtype=np.float64)
        initial_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        gt_anchor_positions = None
        gt_anchor_quaternions = None

    if trajectory_estimate is None:
        estimated_positions, estimated_quaternions = reconstruct_trajectory(
            rel, rel_rotations, initial_position, initial_quaternion
        )

    start_us = max(int(round(args.start_s * 1e6)), int(anchor_timestamps[0]))
    final_us = int(anchor_timestamps[-1])
    if args.duration_s is not None:
        final_us = min(final_us, start_us + int(round(args.duration_s * 1e6)))
    if start_us >= final_us:
        raise ValueError("The selected interval does not overlap the predicted trajectory.")

    frame_step_us = max(1, int(round(1e6 / args.fps)))
    window_us = int(round(args.window_ms * 1000.0))
    frame_timestamps = np.arange(start_us, final_us + 1, frame_step_us, dtype=np.int64)
    smooth_positions = np.column_stack(
        [
            np.interp(frame_timestamps, anchor_timestamps, estimated_positions[:, axis])
            for axis in range(3)
        ]
    )
    smooth_quaternions = interpolate_quaternions(
        anchor_timestamps, estimated_quaternions, frame_timestamps
    )
    if gt_anchor_positions is not None:
        smooth_gt_positions = np.column_stack(
            [
                np.interp(frame_timestamps, anchor_timestamps, gt_anchor_positions[:, axis])
                for axis in range(3)
            ]
        )
    else:
        smooth_gt_positions = None

    plt.style.use("default")
    fig = plt.figure(figsize=(14, 6.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
    event_axis = fig.add_subplot(grid[0, 0])
    trajectory_axis = fig.add_subplot(grid[0, 1], projection="3d")
    event_artist = event_axis.imshow(np.full((args.height, args.width, 3), 248, np.uint8))
    event_axis.axis("off")
    event_axis.set_title("Raw event stream", fontsize=14, fontweight="semibold")

    estimated_line, = trajectory_axis.plot(
        [], [], color="#0057b8", linewidth=2.1, label="TLEIO estimate", zorder=4
    )
    estimated_point, = trajectory_axis.plot(
        [], [], "o", color="#0057b8", markeredgecolor="white", markeredgewidth=1.5,
        markersize=8, zorder=5
    )
    gt_line = None
    if gt_anchor_positions is not None:
        gt_line, = trajectory_axis.plot(
            [],
            [],
            [],
            color="#df596a",
            linewidth=1.2,
            linestyle="--",
            alpha=0.55,
            label="ground truth",
            zorder=1,
        )
    all_positions = estimated_positions
    if gt_anchor_positions is not None:
        all_positions = np.vstack([all_positions, gt_anchor_positions])
    xyz_min = np.min(all_positions, axis=0)
    xyz_max = np.max(all_positions, axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    radius = max(float(np.max(xyz_max - xyz_min)) * 0.58, 1.0)
    trajectory_axis.set_xlim(center[0] - radius, center[0] + radius)
    trajectory_axis.set_ylim(center[1] - radius, center[1] + radius)
    trajectory_axis.set_zlim(center[2] - radius, center[2] + radius)
    trajectory_axis.set_box_aspect((1, 1, 1))
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.set_zlabel("z [m]")
    trajectory_axis.set_title("Estimated 3D trajectory", fontsize=14, fontweight="semibold")
    trajectory_axis.view_init(elev=28, azim=-58)
    trajectory_axis.grid(True, alpha=0.16)
    trajectory_axis.xaxis.pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
    trajectory_axis.yaxis.pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
    trajectory_axis.zaxis.pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
    trajectory_axis.xaxis.pane.set_edgecolor((0.82, 0.85, 0.88, 1.0))
    trajectory_axis.yaxis.pane.set_edgecolor((0.82, 0.85, 0.88, 1.0))
    trajectory_axis.zaxis.pane.set_edgecolor((0.82, 0.85, 0.88, 1.0))
    for axis in (trajectory_axis.xaxis, trajectory_axis.yaxis, trajectory_axis.zaxis):
        axis._axinfo["grid"]["color"] = (0.45, 0.50, 0.55, 0.18)
        axis._axinfo["grid"]["linewidth"] = 0.65
    trajectory_axis.legend(loc="upper right", framealpha=0.9)
    status = trajectory_axis.text2D(
        0.025, 0.965, "", transform=trajectory_axis.transAxes, va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": "#b8c2cc", "alpha": 0.9},
    )
    fig.suptitle(
        "TLEIO: Raw Events, 3D Trajectory and Orientation",
        fontsize=16, fontweight="semibold"
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    orientation_artists = [
        trajectory_axis.plot([], [], [], color=color, linewidth=1.5, zorder=6)[0]
        for color in ("#d62728", "#2ca02c", "#1f77b4")
    ]

    writer = None
    writer_kind = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".gif":
            import imageio.v2 as imageio

            writer = imageio.get_writer(args.output, mode="I", fps=args.fps, loop=0)
            writer_kind = "imageio"
        else:
            import cv2

            frame_width = int(round(fig.get_figwidth() * fig.dpi))
            frame_height = int(round(fig.get_figheight() * fig.dpi))
            writer = cv2.VideoWriter(
                str(args.output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (frame_width, frame_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {args.output}")
            writer_kind = "opencv"

    show_live = args.show or args.output is None
    if show_live:
        plt.ion()
        fig.show()

    rendered = 0
    try:
        with h5py.File(args.events, "r") as event_file, h5py.File(
            args.events_meta, "r"
        ) as metadata_file:
            slicer = EventSlicer(event_file, metadata_file)
            current_event_count = 0
            playback_start = time.perf_counter()
            for frame_index, timestamp_us in enumerate(frame_timestamps):
                events = slicer.get_events(
                    max(0, int(timestamp_us) - window_us), int(timestamp_us)
                )
                if events is not None:
                    current_event_count = len(events["t"])
                    frame = event_frame(events, args.height, args.width, args.max_events)
                    event_artist.set_data(frame)
                    event_axis.set_title(
                        f"Raw events | {args.window_ms:g} ms | "
                        f"{current_event_count:,} events",
                        fontsize=14,
                        fontweight="semibold",
                    )

                visible = smooth_positions[: frame_index + 1]
                estimated_line.set_data_3d(visible[:, 0], visible[:, 1], visible[:, 2])
                estimated_point.set_data_3d(
                    [visible[-1, 0]], [visible[-1, 1]], [visible[-1, 2]]
                )
                if gt_line is not None:
                    gt_visible = smooth_gt_positions[: frame_index + 1]
                    gt_line.set_data_3d(
                        gt_visible[:, 0], gt_visible[:, 1], gt_visible[:, 2]
                    )

                progress, camera_radius = set_3d_camera(
                    trajectory_axis,
                    smooth_positions,
                    frame_index,
                    center,
                    radius,
                )
                orientation_scale = max(camera_radius * 0.11, 0.12)
                rotation = quat_to_rotmat(smooth_quaternions[frame_index])
                origin = visible[-1]
                for axis_index, artist in enumerate(orientation_artists):
                    direction = rotation[:, axis_index]
                    endpoint = origin + orientation_scale * direction
                    artist.set_data_3d(
                        [origin[0], endpoint[0]],
                        [origin[1], endpoint[1]],
                        [origin[2], endpoint[2]],
                    )
                elapsed_s = (timestamp_us - start_us) * 1e-6
                total_s = (final_us - start_us) * 1e-6
                status.set_text(
                    f"time: {elapsed_s:.2f} / {total_s:.2f} s\n"
                    f"position: ({visible[-1, 0]:.2f}, {visible[-1, 1]:.2f}, "
                    f"{visible[-1, 2]:.2f}) m\n"
                    f"frame: X red | Y green | Z blue\n"
                    f"trajectory: {100.0 * progress:.0f}%"
                )

                if writer is not None:
                    rendered_frame = figure_rgb(fig)
                    if writer_kind == "opencv":
                        import cv2

                        writer.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
                    else:
                        writer.append_data(rendered_frame)
                if show_live:
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    plt.pause(0.001)
                    target_elapsed = (frame_index + 1) / args.fps
                    remaining = target_elapsed - (time.perf_counter() - playback_start)
                    if remaining > 0:
                        time.sleep(remaining)
                rendered += 1
                print(
                    f"Frame {frame_index + 1}/{len(frame_timestamps)} | "
                    f"t={timestamp_us * 1e-6:.2f}s | events={current_event_count:,}",
                    end="\r",
                    flush=True,
                )

            if rendered > 0 and args.final_hold_s > 0:
                if gt_line is not None:
                    gt_line.set_data_3d(
                        gt_anchor_positions[:, 0],
                        gt_anchor_positions[:, 1],
                        gt_anchor_positions[:, 2],
                    )
                trajectory_axis.set_xlim(center[0] - radius, center[0] + radius)
                trajectory_axis.set_ylim(center[1] - radius, center[1] + radius)
                trajectory_axis.set_zlim(center[2] - radius, center[2] + radius)
                trajectory_axis.view_init(elev=28, azim=-44)
                status.set_text(
                    f"complete trajectory\n"
                    f"duration: {(final_us - start_us) * 1e-6:.2f} s\n"
                    "frame: X red | Y green | Z blue"
                )
                hold_frames = max(1, int(round(args.final_hold_s * args.fps)))
                for _ in range(hold_frames):
                    if writer is not None:
                        rendered_frame = figure_rgb(fig)
                        if writer_kind == "opencv":
                            import cv2

                            writer.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
                        else:
                            writer.append_data(rendered_frame)
                    if show_live:
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        plt.pause(max(0.001, 1.0 / args.fps))
    except OSError as exc:
        if hdf5plugin is None:
            raise RuntimeError(
                "The TartanEvent HDF5 compression plugin is missing. "
                "Install it with: pip install hdf5plugin"
            ) from exc
        raise
    finally:
        if writer is not None:
            if writer_kind == "opencv":
                writer.release()
            else:
                writer.close()
        if not show_live:
            plt.close(fig)
        print()

    if rendered == 0:
        raise RuntimeError("No synchronized frames were rendered.")
    if args.output:
        print(f"Saved visualization: {args.output}")
    if show_live and args.output is None:
        print("Close the figure or press Ctrl+C to exit.")
        try:
            while plt.fignum_exists(fig.number):
                plt.pause(0.1)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
