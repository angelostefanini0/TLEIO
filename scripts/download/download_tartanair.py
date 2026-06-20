#!/usr/bin/env python3
"""Download and normalize TartanAir/TartanEvent training environments."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download.tartanair_utils import (
    TARTANAIR_FILE_LIST,
    check_timestamps_pose_line_count,
    download_tartanair_huggingface,
    download_tartanevent,
    write_cam_time_from_event_timestamps,
)
from scripts.utils.config import default_config_path, parse_args_with_config


def prepare_training_layout(
    root: Path,
    env: str,
    difficulties: list[str],
    air_archive_name: str,
    keep_air_payload: bool,
) -> None:
    env_dir = root / env.lower()
    air_payload_dir_name = Path(air_archive_name).stem

    for diff in difficulties:
        diff_dir = env_dir / diff.capitalize()
        if not diff_dir.exists():
            continue

        for traj_dir in sorted(p for p in diff_dir.iterdir() if p.is_dir()):
            pose_left = traj_dir / "pose_left.txt"
            pose_lcam_front = traj_dir / "pose_lcam_front.txt"
            timestamps_file = traj_dir / "timestamps.txt"
            pose_source = pose_left if pose_left.exists() else pose_lcam_front

            check_timestamps_pose_line_count(timestamps_file, pose_source)
            if pose_left.exists():
                shutil.copy2(pose_left, pose_lcam_front)

            cam_time_file = traj_dir / "imu" / "cam_time.txt"
            if timestamps_file.exists() and not cam_time_file.exists():
                write_cam_time_from_event_timestamps(timestamps_file, cam_time_file)

            timestamps_file.unlink(missing_ok=True)
            pose_left.unlink(missing_ok=True)
            shutil.rmtree(traj_dir / "flow", ignore_errors=True)

            air_payload_dir = traj_dir / air_payload_dir_name
            if air_payload_dir.exists() and air_payload_dir.is_dir() and not keep_air_payload:
                shutil.rmtree(air_payload_dir)


def normalize_tartanair_layout(
    root: Path,
    env: str,
    difficulties: list[str],
    merge_roots: list[Path] | None = None,
) -> None:
    source_roots = [root]
    if merge_roots:
        source_roots.extend(merge_roots)

    final_env_dir = root / env.lower()
    final_env_dir.mkdir(parents=True, exist_ok=True)

    traj_pattern = re.compile(r"^P\d{3,4}$", re.IGNORECASE)
    allowed_diffs = [d.lower() for d in difficulties]
    env_keywords = [env.lower(), f"{env.lower()}_events"]

    for source_root in source_roots:
        if not source_root.exists():
            continue

        for root_path, _, _ in os.walk(source_root):
            current_dir = Path(root_path)
            if not traj_pattern.match(current_dir.name):
                continue

            path_str = str(current_dir).lower()
            if not any(keyword in path_str for keyword in env_keywords):
                continue

            diff = next((d.capitalize() for d in allowed_diffs if d in path_str), None)
            if diff is None:
                continue

            target_traj_dir = final_env_dir / diff / current_dir.name.upper()
            target_traj_dir.mkdir(parents=True, exist_ok=True)
            if current_dir.resolve() == target_traj_dir.resolve():
                continue

            for item in current_dir.iterdir():
                target_item = target_traj_dir / item.name
                if item.is_file():
                    if target_item.exists() and item.name in {"pose_left.txt", "pose_lcam_front.txt"}:
                        target_item.unlink()
                    if not target_item.exists():
                        shutil.move(str(item), str(target_item))
                elif item.is_dir():
                    target_item.mkdir(parents=True, exist_ok=True)
                    for sub_item in item.iterdir():
                        target_sub_item = target_item / sub_item.name
                        if not target_sub_item.exists():
                            shutil.move(str(sub_item), str(target_sub_item))
                    item.rmdir()

            try:
                current_dir.rmdir()
            except OSError:
                pass

        protected_dirs = [(final_env_dir / d.capitalize()).resolve() for d in allowed_diffs]
        for root_path, _, _ in os.walk(source_root, topdown=False):
            current_dir = Path(root_path)
            if current_dir in {source_root, final_env_dir} or current_dir.resolve() in protected_dirs:
                continue
            if any(keyword in str(current_dir).lower() for keyword in env_keywords):
                try:
                    current_dir.rmdir()
                except OSError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Download TartanAir/TartanEvent training environments.")
    parser.add_argument("--root", type=Path, default=None, help="Common root folder.")
    parser.add_argument("--env", type=str, nargs="+", default=None, help='Environment(s), e.g. "office carwelding".')
    parser.add_argument(
        "--difficulty",
        nargs="+",
        default=["easy", "hard"],
        choices=["easy", "hard"],
        help="Difficulties to download.",
    )
    parser.add_argument("--skip-event", action="store_true", help="Skip TartanEvent event archives.")
    parser.add_argument("--skip-air", action="store_true", help="Skip TartanAir pose archives.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep downloaded archives after extraction.")
    parser.add_argument(
        "--air-file-list",
        type=Path,
        default=TARTANAIR_FILE_LIST,
        help="TartanAir zip list used by the Hugging Face downloader.",
    )
    parser.add_argument(
        "--air-archive-name",
        type=str,
        default="flow_mask.zip",
        help="TartanAir archive to download. The default is the small archive containing pose_left.txt.",
    )
    parser.add_argument(
        "--keep-air-images",
        action="store_true",
        help="Keep the extracted TartanAir payload folder.",
    )
    parser.add_argument(
        "--merge-root",
        type=Path,
        action="append",
        default=[],
        help="Additional partial dataset root(s) to merge into --root during normalization.",
    )
    args = parse_args_with_config(
        parser,
        default_config_path("download_tartanair"),
        required=("root", "env"),
    )

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    merge_roots = [path.resolve() for path in args.merge_root]

    for env in args.env:
        print("\n" + "=" * 72)
        print(f"Preparing Tartan training environment: env={env}")
        print("=" * 72)

        if not args.skip_air:
            download_tartanair_huggingface(
                root=root,
                env=env,
                difficulties=args.difficulty,
                file_list=args.air_file_list,
                archive_name=args.air_archive_name,
                delete_zip=not args.keep_zip,
            )

        if not args.skip_event:
            download_tartanevent(root=root, env_zip=env, delete_zip=not args.keep_zip)

        normalize_tartanair_layout(root, env, args.difficulty, merge_roots)
        prepare_training_layout(
            root=root,
            env=env,
            difficulties=args.difficulty,
            air_archive_name=args.air_archive_name,
            keep_air_payload=args.keep_air_images,
        )

    print("\nDone.")
    print("Unified dataset roots:")
    for env in args.env:
        print(f"  - {root / env.lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
