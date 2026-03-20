#!/usr/bin/env python3
"""
Compare an ms_to_idx dataset you generated against the reference ms_to_idx
stored in a DSEC-style HDF5 file.

It checks:
- same shape
- same dtype (optional warning only)
- exact equality for every entry
- first mismatching indices, if any

Example:
    python scripts/check_ms_to_idx.py \
        --reference ./data/dsec/events.h5 \
        --candidate ./data/processed/ms_to_idx.h5

If the dataset names differ:

    python scripts/check_ms_to_idx.py \
        --reference ./data/dsec/events.h5 \
        --reference-key ms_to_idx \
        --candidate ./data/processed/ms_to_idx.h5 \
        --candidate-key ms_to_idx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hdf5plugin
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a generated ms_to_idx dataset against a reference one."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to the reference HDF5 file containing the ground-truth ms_to_idx.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to the candidate HDF5 file containing the generated ms_to_idx.",
    )
    parser.add_argument(
        "--reference-key",
        type=str,
        default="ms_to_idx",
        help="Dataset key inside the reference HDF5 file. Default: ms_to_idx",
    )
    parser.add_argument(
        "--candidate-key",
        type=str,
        default="ms_to_idx",
        help="Dataset key inside the candidate HDF5 file. Default: ms_to_idx",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=20,
        help="Maximum number of mismatches to print. Default: 20",
    )
    return parser.parse_args()


def ensure_file_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")
    return path


def load_dataset(h5_path: Path, key: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(f"Dataset '{key}' not found in file: {h5_path}")
        data = f[key][...]
    return np.asarray(data)

def print_reference_timestamp_start() -> None:
    reference_timestamps_file = "./data/interlaken_00_c_events_left/events.h5"
    reference_timestamps_key = "events/t"

    with h5py.File(reference_timestamps_file, "r") as f:
        if reference_timestamps_key not in f:
            raise KeyError(
                f"Dataset '{reference_timestamps_key}' not found in {reference_timestamps_file}"
            )
        t_ref = np.asarray(f[reference_timestamps_key][...], dtype=np.int64)

    print("Reference timestamp start check:")
    print(f"  first timestamp [us]: {int(t_ref[0])}")
    print(f"  first timestamp [ms]: {t_ref[0] / 1000.0:.6f}")
    print(f"  first 10 timestamps [us]: {t_ref[:10].tolist()}")
    print()

    if abs(int(t_ref[0])) < 1000:
        print("Timestamps start roughly from 0.")
    else:
        print("Timestamps do not start roughly from 0.")
    print()


def main() -> None:
    args = parse_args()

    ref_path = ensure_file_exists(args.reference)
    cand_path = ensure_file_exists(args.candidate)

    ref = load_dataset(ref_path, args.reference_key)
    cand = load_dataset(cand_path, args.candidate_key)
    
    print("Loaded datasets:")
    print(f"  reference: {ref_path} [{args.reference_key}]")
    print(f"  candidate: {cand_path} [{args.candidate_key}]")
    print()
    print(f"Reference shape: {ref.shape}, dtype: {ref.dtype}")
    print(f"Candidate shape: {cand.shape}, dtype: {cand.dtype}")
    print()
    #In DSEC events start from 0
    print_reference_timestamp_start()

    if ref.shape != cand.shape:
        print("FAIL: shapes differ")
        print(f"  reference shape: {ref.shape}")
        print(f"  candidate shape: {cand.shape}")
        return

    if ref.dtype != cand.dtype:
        print("WARNING: dtypes differ")
        print(f"  reference dtype: {ref.dtype}")
        print(f"  candidate dtype: {cand.dtype}")
        print()

    equal_mask = (ref == cand)
    all_equal = bool(np.all(equal_mask))

    if all_equal:
        print(f"PASS: all {ref.size} entries match exactly")
        return

    mismatch_idx = np.flatnonzero(~equal_mask)
    num_mismatches = mismatch_idx.size

    print("FAIL: datasets differ")
    print(f"  total entries:   {ref.size}")
    print(f"  mismatches:      {num_mismatches}")
    print(f"  match ratio:     {(ref.size - num_mismatches) / ref.size:.6f}")
    print()

    to_show = mismatch_idx[:args.max_mismatches]
    print(f"First {len(to_show)} mismatches:")
    for idx in to_show:
        print(
            f"  idx={int(idx)} | "
            f"reference={int(ref[idx])} | candidate={int(cand[idx])}"
        )


if __name__ == "__main__":
    main()