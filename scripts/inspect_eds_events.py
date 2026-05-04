#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an EDS events.h5 file and summarize how events are stored."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to an EDS sequence directory or directly to events.h5",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of sample events to print from the beginning of the file.",
    )
    parser.add_argument(
        "--check-count",
        type=int,
        default=1_000_000,
        help="How many timestamps to use for the monotonicity check.",
    )
    return parser.parse_args()


def resolve_events_path(path: Path) -> Path:
    if path.is_file():
        return path
    return path / "events.h5"


def find_dataset(h5f: h5py.File, key: str) -> h5py.Dataset:
    if key in h5f:
        return h5f[key]
    nested_key = f"events/{key}"
    if nested_key in h5f:
        return h5f[nested_key]
    raise KeyError(f"Could not find dataset '{key}' or '{nested_key}' in file.")


def print_dataset_inventory(h5f: h5py.File) -> None:
    print("Datasets found:")

    def visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            print(f"  - {name}: shape={obj.shape}, dtype={obj.dtype}")

    h5f.visititems(visitor)


def main() -> int:
    args = parse_args()
    events_path = resolve_events_path(args.path)

    if not events_path.exists():
        raise FileNotFoundError(f"events file not found: {events_path}")

    with h5py.File(events_path, "r") as h5f:
        print(f"File: {events_path.resolve()}")
        print_dataset_inventory(h5f)
        print()

        t = find_dataset(h5f, "t")
        x = find_dataset(h5f, "x")
        y = find_dataset(h5f, "y")
        p = find_dataset(h5f, "p")

        n_events = int(t.shape[0])
        if not (x.shape[0] == y.shape[0] == p.shape[0] == n_events):
            raise ValueError("Event datasets do not all have the same length.")

        print("Event storage summary:")
        print("  - logical event column order: [t_us, x, y, p]")
        print(f"  - total events: {n_events}")
        print(f"  - timestamps dtype: {t.dtype}")
        print(f"  - x/y dtype: {x.dtype}/{y.dtype}")
        print(f"  - polarity dtype: {p.dtype}")

        t0 = int(t[0])
        t1 = int(t[-1])
        duration_us = t1 - t0
        duration_s = duration_us / 1e6
        rate_mev_s = (n_events / duration_s / 1e6) if duration_s > 0 else float("nan")

        print(f"  - first timestamp [us]: {t0}")
        print(f"  - last timestamp  [us]: {t1}")
        print(f"  - duration [s]: {duration_s:.6f}")
        print(f"  - average event rate [Mev/s]: {rate_mev_s:.3f}")

        sample_count = min(max(args.sample_size, 1), n_events)
        check_count = min(max(args.check_count, 2), n_events)

        x_sample = np.asarray(x[:sample_count])
        y_sample = np.asarray(y[:sample_count])
        p_sample = np.asarray(p[:sample_count])
        t_sample = np.asarray(t[:sample_count], dtype=np.int64)
        t_check = np.asarray(t[:check_count], dtype=np.int64)

        x_min = int(np.min(x_sample))
        x_max = int(np.max(x_sample))
        y_min = int(np.min(y_sample))
        y_max = int(np.max(y_sample))
        unique_p = np.unique(np.asarray(p[: min(100_000, n_events)])).tolist()
        monotonic = bool(np.all(np.diff(t_check) >= 0))

        print(f"  - sample x range: [{x_min}, {x_max}]")
        print(f"  - sample y range: [{y_min}, {y_max}]")
        print(f"  - sample polarity values: {unique_p}")
        print(
            f"  - timestamps non-decreasing on first {check_count} events: {monotonic}"
        )

        print()
        print("First events:")
        print("  idx | t_us | x | y | p")
        for idx in range(sample_count):
            print(
                f"  {idx:>3} | {int(t_sample[idx])} | "
                f"{int(x_sample[idx])} | {int(y_sample[idx])} | {int(p_sample[idx])}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
