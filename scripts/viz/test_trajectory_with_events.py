import argparse
import sys
import time
from typing import Any
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.viz.eds_loader import EdsDataLoader
from scripts.inspect_relative_motions import load_table, translation_rel_to_T
from scripts.viz.matplotlib_utils import create_live_trajectory_viewer, update_live_trajectory_viewer
from src.learning.dataloader.representation.event_denoising import background_activity_filter_events
from scripts.utils.config import default_config_path, parse_args_with_config
from src.spatial_math import (
    T_to_pose,
    interpolate_gt_pose,
    normalize_quat,
    pose_to_T,
    quat_to_rotmat,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes", "y"}:
        return True
    if v.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play EDS events overlaid on RGB frames.")
    parser.add_argument("--root", type=str, default=None, help="Dataset root parent directory.")
    parser.add_argument("--sequence", type=str, default=None, help="Sequence folder name.")
    parser.add_argument("--rel-model", type=str, default=None, help="Path to the relative motions predicted by the model")
    parser.add_argument("--rel-gt", type=str, default=None, help="Path to the ground truth relative motions")
    parser.add_argument("--gt", type=str, default=None, help="Path to the ground truth")
    parser.add_argument("--height", type=int, default=480, help="Image height.")
    parser.add_argument("--width", type=int, default=640, help="Image width.")
    parser.add_argument("--start-img", type=int, default=0, help="Starting RGB frame index.")
    parser.add_argument("--num-frames", type=int, default=20000, help="Number of frames to play.")
    parser.add_argument("--fps", type=float, default=12.5, help="Playback FPS. Use 0 for uncapped.")
    parser.add_argument(
        "--event-alpha",
        type=float,
        default=0.4,
        help="Overlay strength for the RGB frame background in [0, 1].",
    )
    parser.add_argument(
        "--denoising",
        type=str2bool,
        default=False,
        help="Apply background-activity filtering before visualization.",
    )
    parser.add_argument(
        "--denoise-dt-us",
        type=int,
        default=1000,
        help="Temporal support window in microseconds for background-activity filtering.",
    )
    parser.add_argument(
        "--denoise-radius",
        type=int,
        default=1,
        help="Spatial neighborhood radius for background-activity filtering.",
    )
    parser.add_argument(
        "--denoise-min-supporters",
        type=int,
        default=1,
        help="Minimum number of recent neighboring events required to keep an event.",
    )
    parser.add_argument(
        "--denoise-same-polarity-only",
        type=str2bool,
        default=False,
        help="Require supporting events to have the same polarity.",
    )
    return parse_args_with_config(
        parser,
        default_config_path("test_trajectory_with_events"),
        required=("root", "sequence"),
    )


def visualize_event(
    events: np.ndarray,
    mapx: np.ndarray,
    mapy: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    if len(events) > 0:
        events = events.copy()
        events[:, 0] = np.clip(events[:, 0], 0, height - 1)
        events[:, 1] = np.clip(events[:, 1], 0, width - 1)
        colors = np.where(
            events[:, 3:4] == 1,
            np.array([255, 0, 0], dtype=np.uint8),
            np.array([0, 0, 255], dtype=np.uint8),
        )
        image[events[:, 0].astype(np.int32), events[:, 1].astype(np.int32), :] = colors

    return cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)


def visualize_image(image: np.ndarray, mapx: np.ndarray, mapy: np.ndarray) -> np.ndarray:
    return cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)


def overlay_events_on_image(
    image: Any,
    events: np.ndarray,
    maps: dict,
    event_alpha: float,
) -> np.ndarray:
    height, width, _ = image.shape
    img_ev = visualize_event(events, maps["ev_mapx"], maps["ev_mapy"], height, width)
    img_frame = visualize_image(image, maps["img_mapx"], maps["img_mapy"])

    white_mask = np.all(img_ev >= 240, axis=2)
    out = img_frame.astype(np.float32)

    if np.any(~white_mask):
        blended = img_frame.astype(np.float32) * (1.0 - event_alpha) + img_ev.astype(np.float32) * event_alpha
        out[~white_mask] = blended[~white_mask]

    return np.clip(out, 0, 255).astype(np.uint8)


def play_events_only(args: argparse.Namespace, loader: EdsDataLoader) -> None:
    window_name = f"EDS Playback - {args.sequence}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_delay_ms = 1 if args.fps <= 0 else max(1, int(round(1000.0 / args.fps)))
    end_img = min(args.start_img + args.num_frames, loader._len_image - 1)

    try:
        for imgi in range(args.start_img, end_img):
            img, t0 = loader.load_image(imgi)
            _, t1 = loader.load_image(imgi + 1)
            i0 = loader.time_to_index(t0)
            i1 = loader.time_to_index(t1)
            ev = loader.load_event(i0, i1)
            num_raw = len(ev)

            if args.denoising:
                ev, _ = background_activity_filter_events(
                    events=ev,
                    height=args.height,
                    width=args.width,
                    dt_us=args.denoise_dt_us,
                    radius=args.denoise_radius,
                    min_supporters=args.denoise_min_supporters,
                    same_polarity_only=args.denoise_same_polarity_only,
                )

            frame = overlay_events_on_image(img, ev, loader.maps, args.event_alpha)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            events_text = f"{len(ev)}/{num_raw}" if args.denoising else f"{num_raw}"
            denoise_text = "denoise ON" if args.denoising else "denoise OFF"
            cv2.putText(
                frame_bgr,
                f"frame {imgi} | t {t0:.6f} s | events {events_text} | {denoise_text}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame_bgr)
            print(
                f"Frame {imgi} | timestamp: {t0:.6f} s | "
                f"next: {t1:.6f} s | dt: {t1 - t0:.6f} s | events: {events_text}",
                end="\r",
                flush=True,
            )

            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if args.fps <= 0:
                time.sleep(0)
    finally:
        cv2.destroyAllWindows()
        print()


def build_reconstructed_trajectory(
    rel: np.ndarray,
    gt_rel: np.ndarray | None,
    init_pos: np.ndarray,
    init_quat: np.ndarray,
):
    T_chain = pose_to_T(init_pos, init_quat)
    recon_pos = [init_pos]
    recon_quat = [init_quat]

    for i in range(len(rel)):
        gt_rel_row = None if gt_rel is None else gt_rel[i]
        T_rel = translation_rel_to_T(rel[i], gt_rel_row)
        T_chain = T_chain @ T_rel
        p, q = T_to_pose(T_chain)
        recon_pos.append(p)
        recon_quat.append(q)

    recon_pos = np.stack(recon_pos, axis=0)
    recon_quat = normalize_quat(np.stack(recon_quat, axis=0))
    return recon_pos, recon_quat


def find_latest_anchor_index(anchor_ts: np.ndarray, query_ts: float) -> int:
    idx = np.searchsorted(anchor_ts, query_ts, side="right") - 1
    return int(np.clip(idx, 0, len(anchor_ts) - 1))


def quat_to_xy_camera_axes(quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_x_xy = np.empty((len(quat_xyzw), 2), dtype=np.float64)
    camera_y_xy = np.empty((len(quat_xyzw), 2), dtype=np.float64)

    for i, quat in enumerate(quat_xyzw):
        R_world_from_camera = quat_to_rotmat(quat)
        x_axis_xy = R_world_from_camera[:2, 0]
        y_axis_xy = R_world_from_camera[:2, 1]

        x_norm = np.linalg.norm(x_axis_xy)
        y_norm = np.linalg.norm(y_axis_xy)

        if x_norm < 1e-12:
            camera_x_xy[i] = np.array([1.0, 0.0], dtype=np.float64)
        else:
            camera_x_xy[i] = x_axis_xy / x_norm

        if y_norm < 1e-12:
            camera_y_xy[i] = np.array([0.0, -1.0], dtype=np.float64)
        else:
            camera_y_xy[i] = -y_axis_xy / y_norm

    return camera_x_xy, camera_y_xy


def main() -> None:
    args = parse_args()
    loader = EdsDataLoader(
        config={
            "root": args.root,
            "sequence": args.sequence,
            "height": args.height,
            "width": args.width,
        }
    )
    loader.set_sequence(args.sequence)

    trajectory_args = [args.rel_model, args.rel_gt, args.gt]
    has_trajectory = all(value is not None for value in trajectory_args)
    if any(value is not None for value in trajectory_args) and not has_trajectory:
        raise ValueError("Provide --rel-model, --rel-gt, and --gt together for trajectory playback.")

    if not has_trajectory:
        play_events_only(args, loader)
        return

    # SET UP VISUALIZATION OF FRAMES + EVENTS
    min_event_ts = loader.min_event_ts

    viewer = create_live_trajectory_viewer(
        image_shape=(args.height, args.width, 3),
        title=f"Trajectory - {args.sequence}",
    )
    frame_delay_s = 0.001 if args.fps <= 0 else 1.0 / args.fps
    end_img = min(args.start_img + args.num_frames, loader._len_image - 1)

    # SET UP GROUND TRUTH AND PREDICTED TRAJECTORY PLOTTING
    gt = load_table(args.gt)
    rel = load_table(args.rel_model)
    gt_rel = load_table(args.rel_gt) 

    if gt.shape[1] != 8:
        raise ValueError(f"{args.gt} has {gt.shape[1]} columns, expected 8.")
    if rel.shape[1] != 5:
        raise ValueError(f"{args.rel_model} has {rel.shape[1]} columns, expected 5")
    if gt_rel is not None and gt_rel.shape[1] != 8:
        raise ValueError(f"{args.gt_rel} has {gt_rel.shape[1]} columns, expected 8")

    gt_ts = gt[:, 0].astype(np.int64)
    gt_pos = gt[:, 1:4]
    gt_quat = normalize_quat(gt[:, 4:8])

    if len(rel) == 0:
        raise ValueError("Relative motions file is empty.")

    rel_t0 = rel[:, 0].astype(np.int64)
    rel_t1 = rel[:, 1].astype(np.int64)

    if not np.all(rel_t1 > rel_t0):
        raise ValueError("Each relative motion must satisfy t1_us > t0_us.")
    if len(rel) > 1 and not np.array_equal(rel_t0[1:], rel_t1[:-1]):
        raise ValueError("Relative motions do not form a continuous timestamp chain.")
    if gt_rel is not None:
        gt_rel_t0 = gt_rel[:, 0].astype(np.int64)
        gt_rel_t1 = gt_rel[:, 1].astype(np.int64)
        if len(gt_rel) != len(rel):
            raise ValueError("--gt_rel must have the same number of rows as --rel.")
        if not np.array_equal(rel_t0, gt_rel_t0) or not np.array_equal(rel_t1, gt_rel_t1):
            raise ValueError("--gt_rel timestamps do not match --rel timestamps.")

    # Anchor timestamps implied by relative motions
    anchor_ts = np.concatenate([rel_t0[:1], rel_t1])

    if args.start_img >= end_img:
        raise ValueError("No frames available to visualize with the requested --start-img/--num-frames.")

    _, start_t1 = loader.load_image(args.start_img + 1)
    start_anchor_query_us = int(round((start_t1 - min_event_ts) * 1e6))
    start_anchor_idx = find_latest_anchor_index(anchor_ts, start_anchor_query_us)
    start_anchor_ts = anchor_ts[start_anchor_idx]

    # Re-anchor the reconstruction at the first plotted frame instead of the
    # first GT pose in the full sequence.
    init_pos, init_quat = interpolate_gt_pose(
        gt_ts, gt_pos, gt_quat, np.array([start_anchor_ts], dtype=np.int64)
    )
    rel_partial = rel[start_anchor_idx:]
    gt_rel_partial = None if gt_rel is None else gt_rel[start_anchor_idx:]

    recon_pos_pred_partial, recon_quat_pred_partial = build_reconstructed_trajectory(
        rel_partial, gt_rel_partial, init_pos[0], init_quat[0]
    )
    recon_pos_gt_partial, recon_quat_gt_partial = interpolate_gt_pose(
        gt_ts, gt_pos, gt_quat, anchor_ts[start_anchor_idx:]
    )
    pred_heading_xy_partial, pred_lateral_xy_partial = quat_to_xy_camera_axes(recon_quat_pred_partial)
    gt_heading_xy_partial, gt_lateral_xy_partial = quat_to_xy_camera_axes(recon_quat_gt_partial)

    blank_frame = np.full((args.height, args.width, 3), 255, dtype=np.uint8)
    update_live_trajectory_viewer(
        viewer,
        blank_frame,
        recon_pos_pred_partial[:1],
        recon_pos_gt_partial[:1],
        pred_dir_xy=pred_heading_xy_partial[0],
        pred_perp_xy=pred_lateral_xy_partial[0],
        gt_dir_xy=gt_heading_xy_partial[0],
        gt_perp_xy=gt_lateral_xy_partial[0],
        frame_idx=args.start_img,
        timestamp_s=float(start_anchor_ts) * 1e-6,
        pause_s=0.001,
    )

    try:
        for imgi in range(args.start_img, end_img):
            # IMAGE VISUALIZATION 
            img, t0 = loader.load_image(imgi)
            _, t1 = loader.load_image(imgi + 1)
            relative_ft1 = int(round((t1 - min_event_ts) * 1e6))
            i0 = loader.time_to_index(t0)
            i1 = loader.time_to_index(t1)
            ev = loader.load_event(i0, i1)
            num_raw = len(ev)

            if args.denoising:
                ev, _ = background_activity_filter_events(
                    events=ev,
                    height=args.height,
                    width=args.width,
                    dt_us=args.denoise_dt_us,
                    radius=args.denoise_radius,
                    min_supporters=args.denoise_min_supporters,
                    same_polarity_only=args.denoise_same_polarity_only,
                )

            frame = overlay_events_on_image(img, ev, loader.maps, args.event_alpha)
            events_text = f"{len(ev)}/{num_raw}" if args.denoising else f"{num_raw}"
            print(f"Frame {imgi} | events: {events_text} | dt: {t1 - t0:.6f} s", end="\r", flush=True)

            # TRAJECTORY PLOTTING
            anchor_idx = find_latest_anchor_index(anchor_ts, relative_ft1)
            partial_anchor_idx = anchor_idx - start_anchor_idx

            update_live_trajectory_viewer(
                viewer,
                frame,
                recon_pos_pred_partial[: partial_anchor_idx + 1],
                recon_pos_gt_partial[: partial_anchor_idx + 1],
                pred_dir_xy=pred_heading_xy_partial[partial_anchor_idx],
                pred_perp_xy=pred_lateral_xy_partial[partial_anchor_idx],
                gt_dir_xy=gt_heading_xy_partial[partial_anchor_idx],
                gt_perp_xy=gt_lateral_xy_partial[partial_anchor_idx],
                frame_idx=imgi,
                timestamp_s=float(anchor_ts[anchor_idx]) * 1e-6,
                pause_s=frame_delay_s,
            )

            if viewer.quit_requested or viewer.closed:
                break

    finally:
        plt.close("all")
        print()


if __name__ == "__main__":
    main()
