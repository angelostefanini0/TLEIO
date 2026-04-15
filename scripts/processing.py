from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gt_training import *

"""This script processes event-based datasets stored in HDF5 files and augments it with a temporal lookup table called `ms_to_idx`.

The input folder is expected to contain raw event data (timestamps, pixel coordinates, and polarity), either in the root or under an `events/` group. 
The script reads the event timestamps, verifies that they are sorted, and computes a mapping from each millisecond to the index of the first 
event occurring at or after that time.

The resulting `ms_to_idx` array enables fast temporal slicing of events without repeatedly searching through the full timestamp array.

A new HDF5 file is created as output for each input sequence. This file contains:
- An `events/` group with the datasets `p`, `t`, `x`, and `y`
- The computed `ms_to_idx` dataset stored at the root level
Additionally, supplementary files such as `imu.csv` and `stamped_groundtruth.txt` are copied to the output directory if they exist.
Command to run:
python scripts/processing.py data/eds/raw \
--save-path data/eds/processed_train \
--save_path_validation data/eds/processed_validation \
--validation-seq 3 \
--save_path_testing data/eds/processed_testing \
--test-seq 0,6 \
--overwrite \
--timestamps-key t \
--process_gt imu.csv stamped_groundtruth.txt \
--delta_t_ms 50 \
--anchor_hz 20


python scripts/processing.py data/eds/raw --save-path data/eds/processed_train --save_path_validation data/eds/processed_validation --validation-seq 3 --save_path_testing data/eds/processed_testing --test-seq 0,6 --overwrite --timestamps-key t --process_gt imu.csv stamped_groundtruth.txt --delta_t_ms 50 --anchor_hz 20

"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ms_to_idx lookup table from event timestamps in an HDF5 file."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the folder with the raw dataset",
    )
    parser.add_argument(
        "--timestamps-key",
        type=str,
        default="events/t",
        help="Internal HDF5 path to the timestamps dataset. Default: events/t",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Path where the output files will be saved.",
    )
    parser.add_argument(
        "--save_path_testing",
        type=Path,
        default=None,
        help="Path where the testing sequence output files will be saved.",
    )
    parser.add_argument(
        "--save_path_validation",
        type=Path,
        default=None,
        help="Path where the validation sequence output files will be saved.",
    )
    parser.add_argument(
        "--test-seq",
        type=str,
        default="",
        help=(
            "Comma-separated list of sequences to save under --save_path_testing. "
            "You can use indices like '0,6' or sequence names like "
            "'00_peanuts_dark,06_ziggy_and_fuzz'."
        ),
    )
    parser.add_argument(
        "--validation-seq",
        type=str,
        default="",
        help=(
            "Comma-separated list of sequences to save under --save_path_validation. "
            "You can use indices like '1,3' or sequence names like "
            "'01_peanuts_light,03_rocket_earth_dark'."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ms_to_idx",
        help="Name of the dataset in the output file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )
    parser.add_argument(
        "--remove-raw",
        action="store_true",
        help="Remove raw input files after processing.",
    )
    parser.add_argument(
        "--process_gt",
        nargs="*",
        default=[],
        help=(
            "Optional list of supplementary files to process from the source file directory "
            "to the output file directory. Example: --process_gt imu.csv stamped_groundtruth.txt"
        ),
    )
    parser.add_argument(
        "--delta_t_ms", 
        type=float, default=50.0, 
        help="Voxel duration in ms"
    )
    
    parser.add_argument(
        "--anchor_hz", 
        type=float, 
        default=20.0, 
        help="Anchor frequency in Hz"
    )

    return parser.parse_args()


def parse_sequence_selection(value: str, available_names: list[str]) -> set[str]:
    if not value.strip():
        return set()

    available_set = set(available_names)
    prefix_to_name = {}
    for name in available_names:
        prefix = name.split("_", 1)[0]
        if prefix.isdigit():
            prefix_to_name[str(int(prefix))] = name
            prefix_to_name[prefix] = name

    selected = set()
    items = [item.strip() for item in value.split(",") if item.strip()]
    for item in items:
        if item in available_set:
            selected.add(item)
            continue

        if item in prefix_to_name:
            selected.add(prefix_to_name[item])
            continue

        raise ValueError(
            f"Unknown sequence '{item}' in --test-seq. "
            f"Use indices like '0,6' or sequence names from: {', '.join(available_names)}"
        )

    return selected


def ensure_file_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    return path


def load_timestamps(h5_path: Path, timestamps_key: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if timestamps_key not in f:
            raise KeyError(f"Dataset '{timestamps_key}' not found.")
        t = f[timestamps_key][...]
    return np.asarray(t, dtype=np.int64)


def validate_sorted_non_decreasing(t: np.ndarray) -> None:
    if np.any(t[1:] < t[:-1]):
        raise ValueError("Timestamps must be sorted in non-decreasing order.")


def build_ms_to_idx(t_us: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Builds the mapping from milliseconds to indices using the following rule: 
    # This is the mapping from milliseconds to event index:
    It is defined such that
    (1) t[ms_to_idx[ms]] >= ms*1000, for ms > 0
    (2) t[ms_to_idx[ms] - 1] < ms*1000, for ms > 0
    (3) ms_to_idx[0] == 0
    , where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
    
    As an example, given 't' and 'ms':
    t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
    ms:   0       1       2       3       4       5       6       7       8       9
    
    we get
    
    ms_to_idx:
          0       2       2       3       3       3       5       5       8       9
    """
    validate_sorted_non_decreasing(t_us)
    
    t0 = t_us[0]
    print(f"Initial time:{t0}")
    t_relative = t_us - t0
    
    max_ms = int(np.ceil(t_relative[-1] / 1000.0))
    ms_grid_us = np.arange(max_ms + 1, dtype=np.int64) * 1000
    
    ms_to_idx = np.searchsorted(t_relative, ms_grid_us, side="left").astype(np.int64)
    return ms_to_idx, int(t0)

def write_to_new_file(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    data: np.ndarray,
    overwrite: bool, 
    t_us: np.ndarray
) -> None:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = "w" if overwrite else "w-"
    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, mode) as f_out:
        events_out = f_out.create_group("events")
        for key in ["p", "t", "x", "y"]:
            if key == "t":
                src_data = (t_us - t_us[0]).astype(t_us.dtype)
                src_dtype = src_data.dtype
            else:
                if f"events/{key}" in f_in:
                    src = f_in[f"events/{key}"]
                elif key in f_in:
                    src = f_in[key]
                else:
                    raise KeyError(f"Dataset '{key}' not found in root or in events/")

                src_data = src[...]
                src_dtype = src.dtype

            events_out.create_dataset(key, data=src_data, dtype=src_dtype)

        f_out.create_dataset(dataset_name, data=data, dtype=data.dtype)


def process_gt(
    source_h5: Path,
    dest_h5: Path,
    files_to_copy: list[str],
    t0: int, 
    delta_t_ms: float, 
    anchor_hz: float
) -> None:
    
    source_dir = source_h5.parent
    dest_dir = dest_h5.parent

    for filename in files_to_copy:
        s_file = source_dir / filename
        d_file = dest_dir / filename

        if not s_file.exists():
            raise FileNotFoundError(f"Supplementary file not found: {s_file}")

        print(f"Processing: {filename} -> {dest_dir}")
        offset_timestamps(s_file, dest_dir, d_file, t0, delta_t_ms, anchor_hz)


def copy_calibration_if_present(source_h5: Path, dest_h5: Path) -> None:
    source_dir = source_h5.parent
    dest_dir = dest_h5.parent
    calibration_path = source_dir / "K.yaml"
    if not calibration_path.exists():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calibration_path, dest_dir / calibration_path.name)
    print(f"Copied calibration: {calibration_path.name} -> {dest_dir}")


def offset_timestamps(s_file: Path,
                      d_folder: Path, 
                      d_file: Path, 
                      t0:int, 
                      delta_t_ms: float, 
                      anchor_hz: float
                      ) -> np.ndarray: 

    if s_file.suffix.lower() == ".csv":
        #the imu file is in [ns]
        load_delimiter = ","
        save_delimiter = ","
        scale = 1e-3
        gt = False
    elif s_file.suffix.lower() == ".txt":
        #the groundtruth file is in [s]
        load_delimiter = None   # any whitespace
        save_delimiter = " "   
        scale = 1e6
        gt = True
    else:
        raise ValueError(f"Unsupported file type: {s_file.suffix}. Use .csv or .txt")

    data = np.loadtxt(s_file, delimiter=load_delimiter, ndmin=2)

    if data.shape[1] < 1:
        raise ValueError("Input file must contain at least one column.")

    #Offset the timestamps
    converted_first_col = data[:, 0] * scale - t0
    keep_mask = converted_first_col >= 0

    filtered = data[keep_mask].copy()
    filtered[:, 0] = converted_first_col[keep_mask].astype(np.int64)
    fmt = ["%d"] + ["%.10f"] * (filtered.shape[1] - 1)
    
    np.savetxt(d_file, filtered, delimiter=save_delimiter, fmt=fmt)
    
    #Actually process the gt 
    if gt: 
        generate_supervision(delta_t_ms, anchor_hz, filtered, d_folder)

def generate_supervision(
        delta_t_ms: float = 50.0,
        anchor_hz: float = 20.0, 
        gt_data: np.ndarray = None, 
        out_dir : Path = None,
        ): 
    delta_t_us = int(round(delta_t_ms * 1000.0))
    anchor_step_us = int(round(1e6 / anchor_hz))
    
    ts_gt, pos_gt, quat_gt = load_gt(gt_data)

    anchors_us = get_anchor_grid(
        gt_timestamps_us=ts_gt,
        delta_t_us=delta_t_us,
        anchor_step_us=anchor_step_us,
    )

    anchor_pos, anchor_quat = interpolate_gt_to_anchors(
        gt_timestamps_us=ts_gt,
        gt_pos=pos_gt,
        gt_quat=quat_gt,
        anchors_us=anchors_us,
    )

    rel = compute_relative_motions(
        anchor_ts=anchors_us,
        anchor_pos=anchor_pos,
        anchor_quat=anchor_quat,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_anchor_poses(out_dir / "anchor_poses.txt", anchors_us, anchor_pos, anchor_quat)
    save_relative_motions(out_dir / "relative_motions.txt", rel)

    print(f"GT poses:           {len(ts_gt)}")
    print(f"Anchors generated:  {len(anchors_us)}")
    print(f"Relative motions:   {len(rel)}")
    if len(anchors_us) > 0:
        print(f"First anchor [us]:  {anchors_us[0]}")
        print(f"Last anchor  [us]:  {anchors_us[-1]}")
        print(f"Step         [us]:  {anchor_step_us}")

def main() -> None:
    """
    Parse command-line arguments for building an ms_to_idx lookup table
    from event timestamps stored in an HDF5 file.

    Examples:
        1. Read timestamps from the default key "events/t" and save output:
            python processing.py /path/to/events.h5 \
                --save-path /path/to/ms_to_idx.h5

        2. Same as above, but overwrite the output file if it already exists:
            python processing.py /path/to/events.h5 \
                --save-path /path/to/ms_to_idx.h5 \
                --overwrite

        3. Save specific sequences to a dedicated testing directory:
            python processing.py /path/to/raw \
                --save-path /path/to/processed \
                --save_path_testing /path/to/processed_testing \
                --test-seq 0,6

        4. Read timestamps from a different internal HDF5 key:
            python processing.py /path/to/events.h5 \
                --timestamps-key t \
                --save-path /path/to/ms_to_idx.h5

        5. Save the lookup table under a custom dataset name in the output file:
            python processing.py /path/to/events.h5 \
                --save-path /path/to/ms_to_idx.h5 \
                --dataset-name custom_ms_to_idx

    """
    args = parse_args()
    input_path = ensure_file_exists(args.file)
    sequence_dirs = sorted([p for p in input_path.iterdir() if p.is_dir()])
    sequence_names = sorted([p.name for p in input_path.iterdir() if p.is_dir()])
    testing_sequences = parse_sequence_selection(args.test_seq, sequence_names)
    validation_sequences = parse_sequence_selection(args.validation_seq, sequence_names)

    if testing_sequences and args.save_path_testing is None:
        raise ValueError("Provide --save_path_testing when using --test-seq.")
    if validation_sequences and args.save_path_validation is None:
        raise ValueError("Provide --save_path_validation when using --validation-seq.")
    overlap = testing_sequences & validation_sequences
    if overlap:
        raise ValueError(
            f"Sequences cannot be both testing and validation: {', '.join(sorted(overlap))}"
        )

    for seq, seq_name in zip(sequence_dirs, sequence_names): 
        event_path = seq / "events.h5"
        t_us = load_timestamps(event_path, args.timestamps_key)
        print(f"Loaded timestamps from: {event_path}")
        
        ms_to_idx, t0 = build_ms_to_idx(t_us)
        print(f"Recording duration: {len(ms_to_idx) - 1} ms")

    
        if args.save_path or args.save_path_testing:
            base_save_path = args.save_path
            if seq_name in testing_sequences:
                base_save_path = args.save_path_testing
            elif seq_name in validation_sequences:
                base_save_path = args.save_path_validation

            if base_save_path is None:
                raise ValueError(
                    f"No output path configured for sequence '{seq_name}'. "
                    "Set --save-path for training sequences."
                )

            output_path = base_save_path / seq_name / "events.h5"
            write_to_new_file(
                input_path=event_path,
                output_path=output_path,
                dataset_name=args.dataset_name,
                data=ms_to_idx,
                overwrite=args.overwrite,
                t_us=t_us
            )
            print(f"Wrote dataset '{args.dataset_name}' into: {output_path}")
            
            if args.process_gt:
                process_gt(event_path, output_path, args.process_gt, t0, args.delta_t_ms, args.anchor_hz)
            copy_calibration_if_present(event_path, output_path)
        
        # delete recursively the raw input files if specified
        if args.remove_raw:
            if seq.exists() and seq.is_dir():
                shutil.rmtree(seq)
                print(f"Successfully removed: {seq}", flush=True)


if __name__ == "__main__":
    main()
