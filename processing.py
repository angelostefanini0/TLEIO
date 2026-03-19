import h5py
import numpy as np
##
"""
Build a millisecond-to-event-index lookup table for an HDF5 event file.

Definition:
    For timestamps t in microseconds and ms in milliseconds, ms_to_idx is defined so that:
      (1) t[ms_to_idx[ms]] >= ms * 1000, for ms > 0
      (2) t[ms_to_idx[ms] - 1] <  ms * 1000, for ms > 0
      (3) ms_to_idx[0] == 0

This script:
  - reads an input HDF5 file containing event timestamps
  - computes ms_to_idx using binary search
  - optionally writes the result back into the same file or to a new HDF5 file

Expected default timestamp dataset path:
    /events/t

Usage examples:
    python build_ms_to_idx.py /path/to/events.h5
    python build_ms_to_idx.py /path/to/events.h5 --timestamps-key events/t
    python build_ms_to_idx.py /path/to/events.h5 --write-inplace
    python build_ms_to_idx.py /path/to/events.h5 --output /path/to/ms_to_idx.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


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
        help=(
            "Path where the output HDF5 file will be saved. "
            "If omitted, the script only computes and reports."
        ),
    )
    return parser.parse_args()


def ensure_file_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")
    return path


def load_timestamps(h5_path: Path, timestamps_key: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if timestamps_key not in f:
            raise KeyError(
                f"Timestamps dataset '{timestamps_key}' not found in file: {h5_path}"
            )
        t = f[timestamps_key][...]

    t = np.asarray(t)
    if t.ndim != 1:
        raise ValueError(
            f"Timestamps dataset must be 1D, got shape {t.shape} for key '{timestamps_key}'."
        )
    if t.size == 0:
        raise ValueError("Timestamps dataset is empty.")
    if not np.issubdtype(t.dtype, np.integer):
        raise TypeError(
            f"Timestamps must be integer microseconds, got dtype {t.dtype}."
        )

    return t.astype(np.int64, copy=False)


def validate_sorted_non_decreasing(t: np.ndarray) -> None:
    if np.any(t[1:] < t[:-1]):
        raise ValueError(
            "Timestamps are not sorted in non-decreasing order. "
            "This lookup requires ordered timestamps."
        )
    if t[0] < 0:
        raise ValueError("Negative timestamps are not supported.")


def build_ms_to_idx(t_us: np.ndarray) -> np.ndarray:
    """
    Build ms_to_idx such that for each millisecond ms:
      ms_to_idx[ms] = first index i with t_us[i] >= ms * 1000

    Returns:
        np.ndarray of shape (max_ms + 1,), dtype=int64
    """
    validate_sorted_non_decreasing(t_us)

    max_ms = int(t_us[-1] // 1000)
    min_ms = int(t_us[0] // 1000)
    ms_grid_us = np.arange((max_ms-min_ms) + 1, dtype=np.int64) * 1000

    # searchsorted(..., side="left") gives the first index i where t_us[i] >= value
    ms_to_idx = np.searchsorted(t_us, ms_grid_us, side="left").astype(np.int64)

    # Enforce property (3) explicitly
    ms_to_idx[0] = min_ms
    return ms_to_idx

def write_to_new_file(
    output_path: Path,
    dataset_name: str,
    data: np.ndarray
) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    mode = "w-"
    with h5py.File(output_path, mode) as f:
        f.create_dataset(dataset_name, data=data, dtype=data.dtype)


def main() -> None:
    args = parse_args()
    input_path = ensure_file_exists(args.file)

    t_us = load_timestamps(input_path, args.timestamps_key)
    ms_to_idx = build_ms_to_idx(t_us)
   

    print(f"Loaded timestamps from: {input_path}")
    print(f"Timestamps key:         {args.timestamps_key}")
    print(f"Number of events:       {len(t_us)}")
    print(f"First timestamp [us]:   {int(t_us[0])}")
    print(f"Last timestamp [us]:    {int(t_us[-1])}")
    print(f"Lookup length:          {len(ms_to_idx)}")
    print(f"Last ms covered:        {len(ms_to_idx) - 1}")

    preview_len = min(10, len(ms_to_idx))
    print(f"Preview ms_to_idx[:{preview_len}]: {ms_to_idx[:preview_len].tolist()}")

    write_to_new_file(
        output_path=args.output,
        dataset_name=args.dataset_name,
        data=ms_to_idx,
        overwrite=args.overwrite,
    )
    print(f"Wrote dataset '{args.dataset_name}' into: {args.output}")

if __name__ == "__main__":
    main()