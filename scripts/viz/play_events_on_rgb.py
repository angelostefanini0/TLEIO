import argparse
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.viz.eds_loader import EdsDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play EDS events overlaid on RGB frames.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root parent directory.")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence folder name.")
    parser.add_argument("--height", type=int, default=480, help="Image height.")
    parser.add_argument("--width", type=int, default=640, help="Image width.")
    parser.add_argument("--start-img", type=int, default=1, help="Starting RGB frame index.")
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

    window_name = f"EDS Playback - {args.sequence}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_delay_ms = 1 if args.fps <= 0 else max(1, int(round(1000.0 / args.fps)))
    end_img = min(args.start_img + args.num_frames, loader._len_image - 1)

    try:
        for imgi in range(args.start_img, end_img):
            img, t1 = loader.load_image(imgi)
            _, t2 = loader.load_image(imgi + 1)
            i1 = loader.time_to_index(t1)
            i2 = loader.time_to_index(t2)
            ev = loader.load_event(i1, i2)

            frame = overlay_events_on_image(img, ev, loader.maps, args.event_alpha)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, frame_bgr)
            print(f"Frame {imgi} | events: {len(ev)} | dt: {t2 - t1:.6f} s", end="\r", flush=True)

            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if args.fps <= 0:
                time.sleep(0)
    finally:
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
