#!/usr/bin/env python3
import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


SEQUENCES = [
    "00_peanuts_dark",
    "01_peanuts_light",
    "02_rocket_earth_light",
    "03_rocket_earth_dark",
    "04_floor_loop",
    "05_rpg_building",
    "06_ziggy_and_fuzz",
    "07_ziggy_and_fuzz_hdr",
    "08_peanuts_running",
    "09_ziggy_flying_pieces",
    "10_office",
    "11_all_characters",
    "12_floor_eight_loop",
    "13_airplane",
    "14_ziggy_in_the_arena",
    "15_apartment_day",
]

BASE_URL = "https://download.ifi.uzh.ch/rpg/eds/dataset"


def parse_testing_arg(value: str, allowed_names: list[str]) -> set[str]:
    """
    Accepts a comma-separated list of either:
      - indices: 0,3,5
      - zero-padded indices: 00,03,05
      - sequence names: 00_peanuts_dark,03_rocket_earth_dark
    """
    if not value.strip():
        return set()

    out = set()
    items = [item.strip() for item in value.split(",") if item.strip()]
    for item in items:
        if item in allowed_names:
            out.add(item)
            continue

        # integer index like 3 or 03
        try:
            idx = int(item)
        except ValueError:
            raise ValueError(
                f"Invalid testing entry '{item}'. Use indices like '0,3,5' "
                f"or names like '00_peanuts_dark,03_rocket_earth_dark'."
            )

        if idx < 0 or idx >= len(SEQUENCES):
            raise ValueError(
                f"Testing index {idx} out of range. Valid range is 0..{len(SEQUENCES)-1}."
            )
        out.add(SEQUENCES[idx])

    return out


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            downloaded = block_num * block_size
            print(f"\rDownloading {dest.name}: {downloaded / (1024**2):.1f} MB", end="", flush=True)
            return

        downloaded = min(block_num * block_size, total_size)
        pct = 100.0 * downloaded / total_size
        print(
            f"\rDownloading {dest.name}: {pct:6.2f}% "
            f"({downloaded / (1024**3):.2f}/{total_size / (1024**3):.2f} GB)",
            end="",
            flush=True,
        )

    try:
        urllib.request.urlretrieve(url, dest, reporthook=reporthook)
        print()
    except Exception:
        if dest.exists():
            dest.unlink()
        raise


def extract_tgz(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)


def maybe_strip_single_top_level_dir(seq_dir: Path) -> None:
    """
    If extraction created seq_dir/<single_subdir>/..., move contents up one level.
    """
    entries = [p for p in seq_dir.iterdir()]
    if len(entries) != 1 or not entries[0].is_dir():
        return

    inner = entries[0]
    tmp_items = list(inner.iterdir())
    for item in tmp_items:
        item.rename(seq_dir / item.name)
    inner.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EDS dataset sequences.")
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory where sequences will be stored.",
    )
    parser.add_argument(
        "num_sequences",
        type=int,
        help=f"How many sequences to download, from 1 to {len(SEQUENCES)}. "
             f"The script downloads the first N sequences in canonical order.",
    )
    parser.add_argument(
        "testing",
        type=str,
        help=(
            "Comma-separated list of test sequences among the selected ones. "
            "You can use indices like '0,3,5' or names like "
            "'00_peanuts_dark,03_rocket_earth_dark'. "
            "Use '' for none."
        ),
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded .tgz files after extraction.",
    )
    args = parser.parse_args()

    if args.num_sequences < 1 or args.num_sequences > len(SEQUENCES):
        raise ValueError(f"num_sequences must be between 1 and {len(SEQUENCES)}.")

    selected = SEQUENCES[: args.num_sequences]
    testing = parse_testing_arg(args.testing, SEQUENCES)

    unknown = testing.difference(selected)
    if unknown:
        raise ValueError(
            "Some testing sequences are not in the selected first N sequences: "
            + ", ".join(sorted(unknown))
        )

    root = args.root.resolve()
    raw_root = root / "raw"
    testing_root = root / "testing"
    archives_root = root / "_archives"

    raw_root.mkdir(parents=True, exist_ok=True)
    testing_root.mkdir(parents=True, exist_ok=True)
    archives_root.mkdir(parents=True, exist_ok=True)

    print("Selected sequences:")
    for seq in selected:
        split = "testing" if seq in testing else "train/root"
        print(f"  - {seq} -> {split}")

    for seq in selected:
        url = f"{BASE_URL}/{seq}/{seq}.tgz"
        archive_path = archives_root / f"{seq}.tgz"
        dest_parent = testing_root if seq in testing else raw_root
        seq_dir = dest_parent / seq

        if seq_dir.exists() and any(seq_dir.iterdir()):
            print(f"Skipping {seq}: already extracted at {seq_dir}")
            continue

        print(f"\n==> {seq}")
        print(f"URL: {url}")

        download_file(url, archive_path)

        print(f"Extracting to {seq_dir} ...")
        extract_tgz(archive_path, seq_dir)
        maybe_strip_single_top_level_dir(seq_dir)

        if not args.keep_archives:
            archive_path.unlink(missing_ok=True)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)