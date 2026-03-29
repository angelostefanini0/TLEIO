#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def load_table(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        first = f.readline().strip()

    skiprows = 1 if first and (first[0].isalpha() or first.startswith("#")) else 0
    data = np.loadtxt(path, skiprows=skiprows, dtype=np.float64)

    if data.size == 0:
        return np.empty((0, 9), dtype=np.float64)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] != 9:
        raise ValueError(
            f"{path} has {data.shape[1]} columns, expected 9: "
            "t0_us t1_us px py pz qx qy qz qw"
        )

    return data

def find_relative_motion_files(processed_root: Path) -> list[Path]:
    return sorted(processed_root.rglob("relative_motions.txt"))

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dataset-wide mean/std over relative motion targets."
    )
    parser.add_argument(
        "processed_root",
        type=Path,
        help="Root directory containing processed sequence folders.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="relative_motion_stats.txt",
        help="Name of the stats file to write at the processed root.",
    )
    args = parser.parse_args()

    processed_root = args.processed_root.resolve()
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed root does not exist: {processed_root}")

    rel_files = find_relative_motion_files(processed_root)
    if not rel_files:
        raise FileNotFoundError(
            f"No relative_motions.txt files found under {processed_root}"
        )

    all_targets = []
    for rel_file in rel_files:
        rel = load_table(rel_file)
        if len(rel) == 0:
            continue

        all_targets.append(rel[:, 2:9].copy())

    if not all_targets:
        raise ValueError("All relative_motions.txt files are empty.")

    all_targets = np.concatenate(all_targets, axis=0)
    target_mean = all_targets.mean(axis=0, dtype=np.float64)
    target_std = all_targets.std(axis=0, dtype=np.float64)
    output = np.stack([target_mean, target_std], axis=0)

    output_path = processed_root / args.filename
    np.savetxt(
        output_path,
        output,
        fmt="%.10f",
        header="px py pz qx qy qz qw",
        comments="",
    )

    print(f"Found {len(rel_files)} relative_motions.txt files")
    print(f"Aggregated {len(all_targets)} relative motion targets")
    print(f"Wrote stats to {output_path}")


if __name__ == "__main__":
    main()
