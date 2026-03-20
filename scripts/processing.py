from __future__ import annotations
import shutil
import argparse
from pathlib import Path
from typing import Optional


import h5py
import numpy as np

"""This script processes an event-based dataset stored in an HDF5 file and augments it with a temporal lookup table called `ms_to_idx`.

The input file is expected to contain event data (timestamps, pixel coordinates, and polarity), either in the root or under an `events/` group. 
The script reads the event timestamps, verifies that they are sorted, and computes a mapping from each millisecond to the index of the first 
event occurring at or after that time.

The resulting `ms_to_idx` array enables fast temporal slicing of events without repeatedly searching through the full timestamp array.

A new HDF5 file is created as output. This file contains:
- An `events/` group with the datasets `p`, `t`, `x`, and `y`
- The computed `ms_to_idx` dataset stored at the root level
Additionally, supplementary files such as `imu.csv` and `stamped_groundtruth.txt` are copied to the output directory if they exist.
Command to run:
python scripts/processing.py data/eds/raw/peanuts/events.h5 --save-path data/eds/processed/peanuts/events.h5 --overwrite --timestamps-key t"""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ms_to_idx lookup table from event timestamps in an HDF5 file."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the input HDF5 file containing event timestamps.",
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
        help="Path where the output HDF5 file will be saved.",
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
        "--copy-files",
        nargs="*",
        default=[],
        help=(
            "Optional list of supplementary files to copy from the source file directory "
            "to the output file directory. Example: --copy-files imu.csv stamped_groundtruth.txt"
    ),
)
    return parser.parse_args()


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


def build_ms_to_idx(t_us: np.ndarray) -> np.ndarray:
    validate_sorted_non_decreasing(t_us)
    
    t0 = t_us[0]
    t_relative = t_us - t0
    
    max_ms = int(np.ceil(t_relative[-1] / 1000.0))
    ms_grid_us = np.arange(max_ms + 1, dtype=np.int64) * 1000
    
    ms_to_idx = np.searchsorted(t_relative, ms_grid_us, side="left").astype(np.int64)
    return ms_to_idx

def write_to_new_file(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    data: np.ndarray,
    overwrite: bool
) -> None:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = "w" if overwrite else "w-"
    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, mode) as f_out:
        events_out = f_out.create_group("events")
        for key in ["p", "t", "x", "y"]:
            if f"events/{key}" in f_in:
                src = f_in[f"events/{key}"]
            elif key in f_in:
                src = f_in[key]
            else:
                raise KeyError(f"Dataset '{key}' not found in root or in events/")
            events_out.create_dataset(key, data=src[...], dtype=src.dtype)

        f_out.create_dataset(dataset_name, data=data, dtype=data.dtype)


def copy_supplementary_files(
    source_h5: Path,
    dest_h5: Path,
    files_to_copy: list[str],
) -> None:
    
    source_dir = source_h5.parent
    dest_dir = dest_h5.parent

    for filename in files_to_copy:
        s_file = source_dir / filename
        d_file = dest_dir / filename

        if not s_file.exists():
            raise FileNotFoundError(f"Supplementary file not found: {s_file}")

        print(f"Copying: {filename} -> {dest_dir}")
        shutil.copy2(s_file, d_file)
     
         

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

        3. Read timestamps from a different internal HDF5 key:
            python processing.py /path/to/events.h5 \
                --timestamps-key t \
                --save-path /path/to/ms_to_idx.h5

        4. Save the lookup table under a custom dataset name in the output file:
            python processing.py /path/to/events.h5 \
                --save-path /path/to/ms_to_idx.h5 \
                --dataset-name custom_ms_to_idx

    """
    args = parse_args()
    input_path = ensure_file_exists(args.file)

   
    t_us = load_timestamps(input_path, args.timestamps_key)
    print(f"Loaded timestamps from: {input_path}")
    
    ms_to_idx = build_ms_to_idx(t_us)
    print(f"Recording duration: {len(ms_to_idx) - 1} ms")

   
    if args.save_path:
        write_to_new_file(
            input_path=input_path,
            output_path=args.save_path,
            dataset_name=args.dataset_name,
            data=ms_to_idx,
            overwrite=args.overwrite,
        )
        print(f"Wrote dataset '{args.dataset_name}' into: {args.save_path}")
        
        if args.copy_files:
            copy_supplementary_files(input_path, args.save_path, args.copy_files)
        

if __name__ == "__main__":
    main()