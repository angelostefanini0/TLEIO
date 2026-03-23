#!/usr/bin/env python3
"""
Example Usage
python scripts/download_unified_tartan.py \
  --root ../data/tartanair \
  --env Office \
  --difficulty easy hard
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import tartanair as ta

TARTANEVENT_ROOT_URL = (
    "https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/TartanEvent"
)


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def download_tartanevent(root: Path, env_zip: str, unzip: bool, delete_zip: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)

    zip_name = env_zip if env_zip.endswith(".zip") else f"{env_zip}.zip"
    target_zip = root / zip_name
    url = f"{TARTANEVENT_ROOT_URL}/{zip_name}"

    print(f"Downloading TartanEvent: {url}")
    run(["curl", "-L", "--fail", "-C", "-", "-o", str(target_zip), url])

    print(f"Verifying archive: {target_zip}")
    with zipfile.ZipFile(target_zip, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupted zip archive, first bad file: {bad}")

    if unzip:
        print(f"Extracting {target_zip} into {root}")
        run(["unzip", "-o", str(target_zip), "-d", str(root)])

        if delete_zip:
            print(f"Deleting {target_zip}")
            target_zip.unlink(missing_ok=True)


def download_tartanair_imu(root: Path, env: str, difficulties: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)

    print(f"Initializing TartanAir at {root}")
    ta.init(str(root))

    print(f"Downloading TartanAir IMU for env={env}, difficulty={difficulties}")
    ta.download(
        env=env,
        difficulty=difficulties,
        modality=["imu"],
        camera_name=["lcam_front"],
        unzip=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Common root folder")
    parser.add_argument("--env", type=str, required=True, help='Environment name, e.g. "office"')
    parser.add_argument(
        "--difficulty",
        nargs="+",
        default=["easy", "hard"],
        choices=["easy", "hard"],
        help="TartanAir difficulties to download",
    )
    parser.add_argument(
        "--skip-event",
        action="store_true",
        help="Do not download the TartanEvent zip",
    )
    parser.add_argument(
        "--skip-air",
        action="store_true",
        help="Do not download TartanAir via the Python API",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded TartanEvent zip after extraction",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_event:
        download_tartanevent(
            root=root,
            env_zip=args.env,
            unzip=True,
            delete_zip=not args.keep_zip,
        )

    if not args.skip_air:
        download_tartanair_imu(
            root=root,
            env=args.env,
            difficulties=args.difficulty,
        )

    print("\nDone.")
    print(f"Common dataset root: {root}")
    print("TartanEvent and TartanAir should now share the same folder tree under this root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())