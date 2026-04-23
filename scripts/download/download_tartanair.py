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
from collections.abc import Iterable

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

    env_folder = root / f"{env_zip}_events"
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


def cleanup_empty_dirs(root_dir: Path, stop_at: Path) -> None:
    if not root_dir.exists():
        return

    subdirs = sorted(
        (path for path in root_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in subdirs:
        remove_empty_tree(path, stop_at)
    remove_empty_tree(root_dir, stop_at)


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def collect_difficulty_sources(
    source_root: Path,
    env_event: str,
    env_air: str,
    final_diff_name: str,
    air_diff_name: str,
) -> list[Path]:
    candidates = [
        source_root / env_event / final_diff_name,
        source_root / f"{env_event}_events" / final_diff_name,
        source_root / f"{env_event}_events" / env_event / final_diff_name,
        source_root / env_air / air_diff_name,
        source_root / env_air / final_diff_name,
    ]

    event_root = source_root / f"{env_event}_events"
    if event_root.exists():
        for child in event_root.iterdir():
            if child.is_dir():
                candidates.append(child / final_diff_name)

    air_root = source_root / env_air
    if air_root.exists():
        for child in air_root.iterdir():
            if child.is_dir():
                candidates.append(child / air_diff_name)
                candidates.append(child / final_diff_name)

    return [path for path in unique_paths(candidates) if path.exists() and path.is_dir()]


def move_env_level_items(source_root: Path, env_event: str, env_air: str, env_final: Path) -> None:
    env_air_dir = source_root / env_air
    if env_air_dir.exists() and env_air_dir.resolve() != env_final.resolve():
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

        remove_empty_tree(env_air_dir, source_root)

    event_root = source_root / f"{env_event}_events"
    if not event_root.exists():
        return

    cleanup_empty_dirs(event_root, source_root)


def normalize_tartanair_layout(
    root: Path,
    env_event: str,
    env_air: str,
    difficulties: list[str],
    merge_roots: list[Path] | None = None,
) -> None:
    """
    Goal:
      - final env folder name: lowercase event name, e.g. root/office
      - final difficulty folders: Easy / Hard
      - if TartanEvent created root/office_events/Easy, merge it into root/office/Easy
      - if TartanEvent created root/office_events/Hard, merge it into root/office/Hard
      - if TartanAir created root/Office/Data_easy, merge it into root/office/Easy
      - if TartanAir created root/Office/Data_hard, merge it into root/office/Hard
      - if a previous partial download already normalized into another root, merge that too
    """
    env_final = root / env_event
    env_final.mkdir(parents=True, exist_ok=True)
    source_roots = unique_paths([root, *((merge_roots or []))])

    diff_map = {
        "easy": ("Data_easy", "Easy"),
        "hard": ("Data_hard", "Hard"),
    }

    for diff in difficulties:
        air_diff_name, final_diff_name = diff_map[diff]
        dst = env_final / final_diff_name
        seen_sources = set()
        for source_root in source_roots:
            sources = collect_difficulty_sources(
                source_root=source_root,
                env_event=env_event,
                env_air=env_air,
                final_diff_name=final_diff_name,
                air_diff_name=air_diff_name,
            )
            for src in sources:
                src_key = str(src.resolve())
                if src_key in seen_sources or src.resolve() == dst.resolve():
                    continue
                seen_sources.add(src_key)
                print(f"Merging {src} -> {dst}")
                move_contents(src, dst)
                remove_empty_tree(src, source_root)

    for source_root in source_roots:
        move_env_level_items(
            source_root=source_root,
            env_event=env_event,
            env_air=env_air,
            env_final=env_final,
        )

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
    parser.add_argument(
        "--merge-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional additional partial dataset root(s) to merge into --root during "
            "normalization. Useful when events and IMU were downloaded in separate runs."
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    merge_roots = [path.resolve() for path in args.merge_root]
    
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
        merge_roots=merge_roots,
    )

    print("\nDone.")
    print(f"Unified dataset root: {root / args.env_event}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
