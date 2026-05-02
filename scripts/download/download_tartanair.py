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
from tqdm import tqdm

TARTANEVENT_ROOT_URL = (
    "https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/TartanEvent"
)

def download_with_retry(url: str, target: str) -> None:
    import os
    import zipfile
    final_path = Path(target)
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    
    if final_path.exists():
        if zipfile.is_zipfile(final_path):
            print(f"\nFile {final_path.name} already fully downloaded and valid.")
            return
        else:
            print(f"\nFound incomplete {final_path.name}. Auto-renaming to .part to resume...")
            final_path.replace(part_path)

    while True:
        try:
            req = urllib.request.Request(url)
            file_size = part_path.stat().st_size if part_path.exists() else 0
            
            if file_size > 0:
                req.add_header("Range", f"bytes={file_size}-")
                print(f"Attempt: Resuming {final_path.name} from {file_size / (1024**3):.2f} GB...")
            else:
                print(f"Attempt: Starting new download for {final_path.name}...")

            with urllib.request.urlopen(req, timeout=30) as response:
                is_partial = response.status == 206
                mode = "ab" if is_partial else "wb"
                
                if not is_partial and file_size > 0:
                    print("Server ignored Range header. Restarting...")
                    file_size = 0
                    mode = "wb"
                
                total_size = int(response.headers.get('content-length', 0))
                if is_partial:
                    total_size += file_size

                with open(part_path, mode) as f:
                    with tqdm(
                        total=total_size, 
                        initial=file_size, 
                        unit='B', 
                        unit_scale=True, 
                        unit_divisor=1024,
                        desc=final_path.name
                    ) as pbar:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno()) 
                            pbar.update(len(chunk))
            
            part_path.replace(final_path)
            print(f"\nDownload completed successfully: {final_path.name}")
            return

        except urllib.error.HTTPError as e:
            if e.code == 416:
                print("\nFile is already fully downloaded.")
                part_path.replace(final_path)
                return
            print(f"\nAttempt failed (HTTP {e.code}). Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"\nAttempt failed: {e}. Retrying in 5s...")
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
        with zipfile.ZipFile(target_zip, "r") as zf:
            zf.extractall(path=env_folder)

        if delete_zip:
            print(f"Deleting {target_zip}")
            target_zip.unlink(missing_ok=True)


def download_tartanair_imu(root: Path, env: str, difficulties: list[str]) -> None:
    import tartanair as ta
    import time
    import zipfile

    root.mkdir(parents=True, exist_ok=True)

    print(f"Initializing TartanAir at {root}")
    ta.init(str(root))

    for diff in difficulties:
        print(f"\nDownloading TartanAir IMU for env={env}, difficulty={diff}")
        # Loop to ensure download goes to the end
        while True:
            try:
                ta.download(
                    env=[env],
                    difficulty=[diff],
                    modality=['imu'],
                    camera_name=["lcam_front"],
                    unzip=True,
                )
                break 
                
            except Exception as e:
                print(f"\nIMU download failed for {diff} (Error: {e}). Retrying in 5s...")
                time.sleep(5)

        diff_dir = root / env / f"Data_{diff}"
        if diff_dir.exists():
            for item in diff_dir.iterdir():
                if item.suffix == '.zip':
                    print(f"Extracting {item.name}...")
                    try:
                        with zipfile.ZipFile(item, 'r') as zf:
                            zf.extractall(diff_dir)
                        item.unlink() 
                    except Exception as e:
                        print(f"Error while extracting {item.name}: {e}")


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
    import os
    import shutil
    import re

    source_roots = [root]
    if merge_roots:
        source_roots.extend(merge_roots)

    final_env_dir = root / env_event.lower()
    final_env_dir.mkdir(parents=True, exist_ok=True)

    traj_pattern = re.compile(r"^P\d{3,4}$", re.IGNORECASE)
    
    allowed_diffs = [d.lower() for d in difficulties]

    env_keywords = [env_event.lower(), env_air.lower(), f"{env_event.lower()}_events"]

    for source_root in source_roots:
        if not source_root.exists():
            continue

        for root_path, dirs, files in os.walk(source_root):
            current_dir = Path(root_path)

            if traj_pattern.match(current_dir.name):
                path_str = str(current_dir).lower()

                if not any(k in path_str for k in env_keywords):
                    continue

                diff = None
                for d in allowed_diffs:
                    if d in path_str:
                        diff = d.capitalize() 
                        break

                if not diff:
                    continue 

                target_traj_dir = final_env_dir / diff / current_dir.name.upper()
                target_traj_dir.mkdir(parents=True, exist_ok=True)

                if current_dir.resolve() == target_traj_dir.resolve():
                    continue

                for item in current_dir.iterdir():
                    target_item = target_traj_dir / item.name

                    if item.is_file():
                        if not target_item.exists():
                            shutil.move(str(item), str(target_item))
                    elif item.is_dir():
                        target_item.mkdir(parents=True, exist_ok=True)
                        for sub_item in item.iterdir():
                            target_sub_item = target_item / sub_item.name
                            if not target_sub_item.exists():
                                shutil.move(str(sub_item), str(target_sub_item))
                        try:
                            item.rmdir()
                        except OSError:
                            pass

                try:
                    current_dir.rmdir()
                except OSError:
                    pass

        
        protected_dirs = [ (final_env_dir / d.capitalize()).resolve() for d in allowed_diffs ]
        
        for root_path, dirs, files in os.walk(source_root, topdown=False):
            current_dir = Path(root_path)

            if current_dir == source_root or current_dir == final_env_dir:
                continue
            if current_dir.resolve() in protected_dirs:
                continue

            path_str = str(current_dir).lower()
            if not any(k in path_str for k in env_keywords):
                continue

            try:
                current_dir.rmdir()
            except OSError:
                pass




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
