from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.config import default_config_path, parse_args_with_config
from scripts.utils.gt_training import (
    compute_relative_motions,
    get_anchor_grid,
    interpolate_gt_to_anchors,
    load_gt,
    save_anchor_poses,
    save_relative_motions,
)
from src.spatial_math import quat_to_rotmat, rotmat_to_quat


DEFAULT_EVENT_CHUNK_LINES = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DAVIS 240C txt sequences into the processed TLEIO layout."
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        help="Root containing DAVIS sequence folders, e.g. data/DAVIS_240C.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Output root for processed sequences.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="",
        help="Optional single sequence name to process.",
    )
    parser.add_argument("--delta_t_ms", type=float, default=50.0)
    parser.add_argument("--anchor_t_ms", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument(
        "--pose-frame-remap",
        choices=("none", "davis_to_tleio"),
        default="none",
        help="Apply a fixed camera-frame rotation to DAVIS GT quaternions before supervision.",
    )
    parser.add_argument(
        "--event-chunk-lines",
        type=int,
        default=DEFAULT_EVENT_CHUNK_LINES,
        help="Number of event txt rows converted per chunk.",
    )
    parser.add_argument("--overwrite", action="store_true")

    return parse_args_with_config(
        parser,
        default_config_path("processing_davis240c"),
        required=("root", "save_path"),
    )


def discover_sequences(root: Path, sequence: str = "") -> list[Path]:
    candidates = sorted(
        seq for seq in root.iterdir()
        if seq.is_dir() and (seq / "events.txt").exists() and (seq / "groundtruth.txt").exists()
    )
    if sequence:
        candidates = [seq for seq in candidates if seq.name == sequence]
        if not candidates:
            raise FileNotFoundError(f"Sequence '{sequence}' not found under {root}.")
    if not candidates:
        raise FileNotFoundError(f"No DAVIS sequences with events.txt and groundtruth.txt found under {root}.")
    return candidates


def append_dataset(dataset: h5py.Dataset, values: np.ndarray, offset: int) -> int:
    next_offset = offset + len(values)
    dataset.resize((next_offset,))
    dataset[offset:next_offset] = values
    return next_offset


def iter_event_chunks(events_txt: Path, chunk_lines: int):
    lines: list[str] = []
    with events_txt.open("r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(line)
            if len(lines) >= chunk_lines:
                yield parse_event_lines(lines)
                lines.clear()
        if lines:
            yield parse_event_lines(lines)


def parse_event_lines(lines: list[str]) -> np.ndarray:
    values = np.fromstring("".join(lines), sep=" ", dtype=np.float64)
    if values.size % 4 != 0:
        raise ValueError("events.txt rows must have four columns: t x y polarity.")
    return values.reshape((-1, 4))


def build_ms_to_idx(t_us: np.ndarray) -> np.ndarray:
    if len(t_us) == 0:
        return np.zeros((1,), dtype=np.int64)
    if np.any(t_us[1:] < t_us[:-1]):
        raise ValueError("Event timestamps must be sorted in non-decreasing order.")
    max_ms = int(np.ceil(float(t_us[-1]) / 1000.0))
    ms_grid_us = np.arange(max_ms + 1, dtype=np.int64) * 1000
    return np.searchsorted(t_us, ms_grid_us, side="left").astype(np.int64)


def write_events_h5(events_txt: Path, out_h5: Path, chunk_lines: int, overwrite: bool) -> tuple[int, int, int]:
    if chunk_lines <= 0:
        raise ValueError("--event-chunk-lines must be > 0.")
    if out_h5.exists() and not overwrite:
        raise FileExistsError(f"{out_h5} already exists. Pass --overwrite to replace it.")

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "w-"
    offset = 0
    t0_us: int | None = None
    last_t = -1

    with h5py.File(out_h5, mode) as h5f:
        events = h5f.create_group("events")
        chunks = (min(chunk_lines, DEFAULT_EVENT_CHUNK_LINES),)
        d_t = events.create_dataset("t", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=chunks)
        d_x = events.create_dataset("x", shape=(0,), maxshape=(None,), dtype=np.uint16, chunks=chunks)
        d_y = events.create_dataset("y", shape=(0,), maxshape=(None,), dtype=np.uint16, chunks=chunks)
        d_p = events.create_dataset("p", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=chunks)

        for chunk in iter_event_chunks(events_txt, chunk_lines):
            t_abs_us = np.rint(chunk[:, 0] * 1e6).astype(np.int64)
            if t0_us is None:
                t0_us = int(t_abs_us[0])
            t_rel_us = t_abs_us - t0_us
            if len(t_rel_us) and int(t_rel_us[0]) < last_t:
                raise ValueError(f"{events_txt}: event timestamps are not sorted.")
            last_t = int(t_rel_us[-1])

            x = np.rint(chunk[:, 1]).astype(np.uint16)
            y = np.rint(chunk[:, 2]).astype(np.uint16)
            p = np.rint(chunk[:, 3]).astype(np.uint8)

            offset = append_dataset(d_t, t_rel_us, offset)
            append_dataset(d_x, x, offset - len(x))
            append_dataset(d_y, y, offset - len(y))
            append_dataset(d_p, p, offset - len(p))

        if t0_us is None:
            raise ValueError(f"{events_txt} is empty.")

        t_us = d_t[...]
        h5f.create_dataset("ms_to_idx", data=build_ms_to_idx(t_us), dtype=np.int64)
        h5f.attrs["normalize_polarity_to_binary"] = 0

    return t0_us, last_t, offset


def convert_seconds_table(path: Path, t0_us: int, last_event_us: int) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    timestamps_us = np.rint(data[:, 0] * 1e6).astype(np.int64) - t0_us
    keep = (timestamps_us >= 0) & (timestamps_us <= last_event_us)
    converted = data[keep].copy()
    converted[:, 0] = timestamps_us[keep]
    return converted


def davis_to_tleio_frame_rotation() -> np.ndarray:
    """Rotation from original DAVIS camera frame to the model/TLEIO frame.

    This matches the empirical output-axis relation found on DAVIS:
    model [x,y,z] ~= DAVIS [z,x,y], so DAVIS-frame GT is transformed before
    relative-motion generation instead of remapping predictions after inference.
    """
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def remap_pose_frame(gt: np.ndarray, remap: str) -> np.ndarray:
    if remap == "none":
        return gt
    if remap != "davis_to_tleio":
        raise ValueError(f"Unsupported pose-frame remap: {remap}")

    out = gt.copy()
    R_davis_model = davis_to_tleio_frame_rotation()
    remapped_quat = []
    for q in out[:, 4:8]:
        R_world_davis = quat_to_rotmat(q)
        R_world_model = R_world_davis @ R_davis_model
        remapped_quat.append(rotmat_to_quat(R_world_model))
    out[:, 4:8] = np.stack(remapped_quat, axis=0)
    return out


def write_groundtruth_and_supervision(
    raw_gt: Path,
    out_dir: Path,
    t0_us: int,
    last_event_us: int,
    delta_t_ms: float,
    anchor_t_ms: float,
    pose_frame_remap: str,
) -> None:
    gt = convert_seconds_table(raw_gt, t0_us, last_event_us)
    if gt.shape[1] != 8:
        raise ValueError(f"{raw_gt}: expected columns t px py pz qx qy qz qw.")
    gt = remap_pose_frame(gt, pose_frame_remap)

    stamped_gt = out_dir / "stamped_groundtruth.txt"
    np.savetxt(stamped_gt, gt, fmt=["%d"] + ["%.10f"] * 7)

    delta_t_us = int(round(delta_t_ms * 1000.0))
    anchor_step_us = int(round(anchor_t_ms * 1000.0))
    ts_gt, pos_gt, quat_gt = load_gt(gt)
    anchors_us = get_anchor_grid(ts_gt, delta_t_us=delta_t_us, anchor_step_us=anchor_step_us)
    anchor_pos, anchor_quat = interpolate_gt_to_anchors(ts_gt, pos_gt, quat_gt, anchors_us)
    rel = compute_relative_motions(anchors_us, anchor_pos, anchor_quat)

    save_anchor_poses(out_dir / "anchor_poses.txt", anchors_us, anchor_pos, anchor_quat)
    save_relative_motions(out_dir / "relative_motions.txt", rel)

    print(f"GT poses:           {len(ts_gt)}")
    print(f"Anchors generated:  {len(anchors_us)}")
    print(f"Relative motions:   {len(rel)}")
    if pose_frame_remap != "none":
        print(f"Pose frame remap:   {pose_frame_remap}")


def write_imu_if_present(raw_imu: Path, out_dir: Path, t0_us: int, last_event_us: int) -> None:
    if not raw_imu.exists():
        return
    imu = convert_seconds_table(raw_imu, t0_us, last_event_us)
    if imu.shape[1] != 7:
        raise ValueError(f"{raw_imu}: expected columns t ax ay az gx gy gz.")
    reordered = np.column_stack([imu[:, 0], imu[:, 4:7], imu[:, 1:4]])
    np.savetxt(
        out_dir / "imu.csv",
        reordered,
        delimiter=",",
        fmt=["%d"] + ["%.10f"] * 6,
        header="timestamp_us,gx,gy,gz,ax,ay,az",
        comments="# ",
    )


def write_calibration(raw_calib: Path, out_dir: Path, width: int, height: int) -> None:
    values = np.loadtxt(raw_calib, dtype=np.float64, ndmin=1)
    if values.size < 4:
        raise ValueError(f"{raw_calib}: expected at least fx fy cx cy.")
    calib = {
        "cam1": {
            "intrinsics": [float(v) for v in values[:4]],
            "resolution": [int(width), int(height)],
        }
    }
    if values.size > 4:
        calib["cam1"]["distortion_coeffs"] = [float(v) for v in values[4:]]
    with (out_dir / "K.yaml").open("w") as fh:
        yaml.safe_dump(calib, fh, sort_keys=False)


def process_sequence(seq_dir: Path, out_root: Path, args: argparse.Namespace) -> None:
    out_dir = out_root / seq_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {seq_dir.name} -> {out_dir}")
    t0_us, last_event_us, num_events = write_events_h5(
        seq_dir / "events.txt",
        out_dir / "events.h5",
        chunk_lines=args.event_chunk_lines,
        overwrite=args.overwrite,
    )
    write_groundtruth_and_supervision(
        seq_dir / "groundtruth.txt",
        out_dir,
        t0_us=t0_us,
        last_event_us=last_event_us,
        delta_t_ms=args.delta_t_ms,
        anchor_t_ms=args.anchor_t_ms,
        pose_frame_remap=args.pose_frame_remap,
    )
    write_imu_if_present(seq_dir / "imu.txt", out_dir, t0_us, last_event_us)
    if (seq_dir / "calib.txt").exists():
        write_calibration(seq_dir / "calib.txt", out_dir, args.width, args.height)

    print(f"Events:             {num_events}")
    print(f"First raw event us: {t0_us}")
    print(f"Last event us:      {last_event_us}")


def main() -> None:
    args = parse_args()
    for seq_dir in discover_sequences(args.root, args.sequence):
        process_sequence(seq_dir, args.save_path, args)


if __name__ == "__main__":
    main()
