import argparse
import sys
from typing import Any
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eds_loader import EdsDataLoader
from inspect_functions.inspect_relative_motions import *
from matplotlib_utils import create_live_trajectory_viewer, update_live_trajectory_viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play EDS events overlaid on RGB frames.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root parent directory.")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence folder name.")
    parser.add_argument("--rel-model", type=str, required=True, help="Path to the relative motions predicted by the model")
    parser.add_argument("--rel-gt", type=str, required=True, help="Path to the ground truth relative motions")
    parser.add_argument("--gt", type=str, required=True, help="Path to the ground truth")
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
    return parser.parse_args()


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


def build_reconstructed_trajectory(
    rel: np.ndarray,
    gt_rel: np.ndarray | None,
    init_pos: np.ndarray,
    init_quat: np.ndarray,
    gt_rel_mode: str,
):
    T_chain = pose_to_T(init_pos, init_quat)
    recon_pos = [init_pos]
    recon_quat = [init_quat]

    for i in range(len(rel)):
        gt_rel_row = None if gt_rel is None else gt_rel[i]
        T_rel = fuse_rel_transforms(rel[i], gt_rel_row, gt_rel_mode)
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


def main() -> None:
    args = parse_args()
    # SET UP VISUALIZATION OF FRAMES + EVENTS
    loader = EdsDataLoader(
        config={
            "root": args.root,
            "sequence": args.sequence,
            "height": args.height,
            "width": args.width,
        }
    )
    loader.set_sequence(args.sequence)
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
    if rel.shape[1] != 8:
        raise ValueError(f"{args.rel_model} has {rel.shape[1]} columns, expected 8")
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

    # Initial pose from source GT at first anchor
    init_pos, init_quat = interpolate_gt_pose(
        gt_ts, gt_pos, gt_quat, np.array([anchor_ts[0]], dtype=np.int64)
    )

    recon_pos_pred, recon_quat_pred = build_reconstructed_trajectory(
        rel, gt_rel, init_pos[0], init_quat[0], "rotation"
    )
    recon_pos_gt, recon_quat_gt = interpolate_gt_pose(gt_ts, gt_pos, gt_quat, anchor_ts)
    recon_quat_gt = normalize_quat(recon_quat_gt)

    blank_frame = np.full((args.height, args.width, 3), 255, dtype=np.uint8)
    update_live_trajectory_viewer(
        viewer,
        blank_frame,
        recon_pos_pred[:1],
        recon_pos_gt[:1],
        frame_idx=args.start_img,
        timestamp_s=float(anchor_ts[0]) * 1e-6,
        pause_s=0.001,
    )

    try:
        for imgi in range(args.start_img, end_img):
            # IMAGE VISUALIZATION 
            img, t0 = loader.load_image(imgi)
            _, t1 = loader.load_image(imgi + 1)
            relative_ft1 = (t1 - min_event_ts) * 1e6
            i0 = loader.time_to_index(t0)
            i1 = loader.time_to_index(t1)
            ev = loader.load_event(i0, i1)

            frame = overlay_events_on_image(img, ev, loader.maps, args.event_alpha)
            print(f"Frame {imgi} | events: {len(ev)} | dt: {t1 - t0:.6f} s", end="\r", flush=True)

            # TRAJECTORY PLOTTING
            anchor_idx = find_latest_anchor_index(anchor_ts, relative_ft1)

            update_live_trajectory_viewer(
                viewer,
                frame,
                recon_pos_pred[: anchor_idx + 1],
                recon_pos_gt[: anchor_idx + 1],
                frame_idx=imgi,
                timestamp_s=t1,
                pause_s=frame_delay_s,
            )

            if viewer.quit_requested or viewer.closed:
                break

    finally:
        plt.close("all")
        print()


if __name__ == "__main__":
    main()
