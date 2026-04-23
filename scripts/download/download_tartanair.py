#!/usr/bin/env python3
"""
Example:
python scripts/download_tartanair.py \
  --root data/tartanair \
  --env-event office \
  --env-air Office \
  --difficulty easy hard
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path
import urllib.request
import time

TARTANEVENT_ROOT_URL = (
    "https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/TartanEvent"
)

def download_with_retry(url, target):
    for i in range(3): # Try 3 times
        try:
            urllib.request.urlretrieve(url, target)
            return
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}. Retrying...")
            time.sleep(5)

def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def download_tartanevent(root: Path, env_zip: str, unzip: bool, delete_zip: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)

    env_folder = root / env_zip + "_events"
    env_folder.mkdir(parents=True, exist_ok=True)

    zip_name = env_zip if env_zip.endswith(".zip") else f"{env_zip}.zip"
    target_zip = root / zip_name
    url = f"{TARTANEVENT_ROOT_URL}/{zip_name}"

    print(f"Downloading TartanEvent: {url}")
    # run(["curl", "-L", "--fail", "-C", "-", "-o", str(target_zip), url])
    download_with_retry(url, str(target_zip))

    print(f"Verifying archive: {target_zip}")
    with zipfile.ZipFile(target_zip, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupted zip archive, first bad file: {bad}")

    if unzip:
        print(f"Extracting {target_zip} into {env_folder}")
        run(["unzip", "-o", str(target_zip), "-d", str(env_folder)])

        if delete_zip:
            print(f"Deleting {target_zip}")
            target_zip.unlink(missing_ok=True)


def download_tartanair_imu(root: Path, env: str, difficulties: list[str]) -> None:
    import tartanair as ta

    root.mkdir(parents=True, exist_ok=True)

    print(f"Initializing TartanAir at {root}")
    ta.init(str(root))

    print(f"Downloading TartanAir IMU for env={env}, difficulty={difficulties}")
    ta.download(
        env=[env],
        difficulty=difficulties,
        modality=['imu', 'pose'],
        camera_name=["lcam_front"],
        unzip=True,
    )


def move_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if item.is_dir() and target.is_dir():
                move_contents(item, target)
                try:
                    item.rmdir()
                except OSError:
                    pass
            else:
                print(f"Skipping existing file: {target}")
        else:
            shutil.move(str(item), str(target))


def remove_empty_tree(path: Path, stop_at: Path) -> None:
    while path.exists() and path != stop_at:
        try:
            path.rmdir()
        except OSError:
            break
        path = path.parent


def normalize_tartanair_layout(root: Path, env_event: str, env_air: str, difficulties: list[str]) -> None:
    """
    Goal:
      - final env folder name: lowercase event name, e.g. root/office
      - final difficulty folders: Easy / Hard
      - if TartanEvent created root/office_events/Easy, merge it into root/office/Easy
      - if TartanEvent created root/office_events/Hard, merge it into root/office/Hard
      - if TartanAir created root/Office/Data_easy, merge it into root/office/Easy
      - if TartanAir created root/Office/Data_hard, merge it into root/office/Hard
    """
    env_final = root / env_event
    env_air_dir = root / env_air

    env_final.mkdir(parents=True, exist_ok=True)

    diff_map = {
        "easy": ("Data_easy", "Easy"),
        "hard": ("Data_hard", "Hard"),
    }

    for diff in difficulties:
        air_diff_name, final_diff_name = diff_map[diff]
        dst = env_final / final_diff_name
        sources = [
            root / env_event + "_events" / final_diff_name,      # TartanEvent zip layout
            env_air_dir / air_diff_name, # TartanAir layout
        ]

        for src in sources:
            if not src.exists() or src.resolve() == dst.resolve():
                continue
            print(f"Merging {src} -> {dst}")
            move_contents(src, dst)
            remove_empty_tree(src, root)

    # If TartanAir also created env-level files directly under root/Office, move them too
    if env_air_dir.exists() and env_air_dir != env_final:
        for item in list(env_air_dir.iterdir()):
            target = env_final / item.name
            if target.exists():
                if item.is_dir() and target.is_dir():
                    move_contents(item, target)
                    remove_empty_tree(item, env_air_dir)
                else:
                    print(f"Skipping existing path: {target}")
            else:
                shutil.move(str(item), str(target))

        remove_empty_tree(env_air_dir, root)

    # Clean up any empty top-level difficulty folders left behind by TartanEvent.
    for diff in difficulties:
        final_diff_name = diff_map[diff][1]
        top_level_diff = root / final_diff_name
        if top_level_diff.exists() and top_level_diff != env_final / final_diff_name:
            remove_empty_tree(top_level_diff, root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Common root folder")
    parser.add_argument("--env-event", type=str, required=True, help='TartanEvent env, e.g. "office"')
    parser.add_argument("--env-air", type=str, required=True, help='TartanAir env, e.g. "Office"')
    parser.add_argument(
        "--difficulty",
        nargs="+",
        default=["easy", "hard"],
        choices=["easy", "hard"],
        help="Difficulties to download",
    )
    parser.add_argument("--skip-event", action="store_true")
    parser.add_argument("--skip-air", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_air:
        download_tartanair_imu(
            root=root,
            env=args.env_air,
            difficulties=args.difficulty,
        )

    if not args.skip_event:
        download_tartanevent(
            root=root,
            env_zip=args.env_event,
            unzip=True,
            delete_zip=not args.keep_zip,
        )

    

    normalize_tartanair_layout(
        root=root,
        env_event=args.env_event,
        env_air=args.env_air,
        difficulties=args.difficulty,
    )

    print("\nDone.")
    print(f"Unified dataset root: {root / args.env_event}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
