#!/usr/bin/env python3
"""Download TartanEvent/TartanAir competition data."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download.tartanair_utils import (
    TARTANAIR_COMPETITION_AIR_URL,
    TARTANAIR_COMPETITION_ARCHIVE,
    TARTANEVENT_COMPETITION_URL,
    download_google_drive_file,
    download_with_retry,
    extract_tar_archive,
    extract_zip_archives,
    extract_zip_url,
    write_cam_time_from_event_timestamps,
)
from scripts.utils.config import default_config_path, parse_args_with_config


def sequence_aliases(selection: str) -> set[str]:
    aliases: set[str] = set()
    for item in selection.split(","):
        seq = item.strip()
        if not seq:
            continue
        aliases.add(seq)
        if seq.startswith("competition_Test_"):
            aliases.add(seq.removeprefix("competition_Test_"))
        else:
            aliases.add(f"competition_Test_{seq}")
    return aliases


def path_matches_sequence(path: str, aliases: set[str]) -> bool:
    if not aliases:
        return True
    parts = set(Path(path).parts)
    return any(alias in parts or any(alias in part for part in parts) for alias in aliases)


def normalize_competition_layout(root: Path, aliases: set[str] | None = None) -> None:
    aliases = aliases or set()
    traj_pattern = re.compile(r"^(P\d{3,4}|competition_Test_[A-Z]{2}\d{3}|[A-Z]{2}\d{3})$", re.IGNORECASE)
    for traj_dir in sorted(p for p in root.rglob("*") if p.is_dir() and traj_pattern.match(p.name)):
        if aliases and traj_dir.name not in aliases:
            continue
        pose_left = traj_dir / "pose_left.txt"
        pose_lcam_front = traj_dir / "pose_lcam_front.txt"
        if pose_left.exists() and not pose_lcam_front.exists():
            shutil.copy2(pose_left, pose_lcam_front)

        timestamps_file = traj_dir / "timestamps.txt"
        cam_time_file = traj_dir / "imu" / "cam_time.txt"
        if timestamps_file.exists() and not cam_time_file.exists():
            write_cam_time_from_event_timestamps(timestamps_file, cam_time_file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download TartanEvent/TartanAir competition data.")
    parser.add_argument("--root", type=Path, default=None, help="Tartan data root.")
    parser.add_argument(
        "--seq",
        type=str,
        default="",
        help=(
            "Comma-separated competition sequence selector, for example MH001 "
            "or competition_Test_MH001. Empty downloads/extracts all sequences."
        ),
    )
    parser.add_argument("--skip-event", action="store_true", help="Skip TartanEvent competition events.")
    parser.add_argument("--skip-air", action="store_true", help="Skip TartanAir competition poses/images.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded archives after extraction.")
    args = parse_args_with_config(
        parser,
        default_config_path("download_tartanair_competition"),
        required=("root",),
    )

    root = args.root.resolve()
    competition_root = root / "competition"
    archives_root = root / "_archives"
    competition_root.mkdir(parents=True, exist_ok=True)
    archives_root.mkdir(parents=True, exist_ok=True)
    aliases = sequence_aliases(args.seq)
    member_filter = None if not aliases else lambda name: path_matches_sequence(name, aliases)

    if aliases:
        print(f"Selected sequence aliases: {', '.join(sorted(aliases))}")

    if not args.skip_event:
        event_archive = archives_root / "TartanEvent_competition.zip"
        if aliases and not args.keep_archives:
            extract_zip_url(
                TARTANEVENT_COMPETITION_URL,
                extract_root=competition_root,
                member_filter=member_filter,
            )
        else:
            download_with_retry(TARTANEVENT_COMPETITION_URL, event_archive)
            extract_zip_archives(
                [event_archive],
                delete_zip=not args.keep_archives,
                extract_root=competition_root,
                member_filter=member_filter,
            )

    if not args.skip_air:
        air_archive = archives_root / TARTANAIR_COMPETITION_ARCHIVE
        download_google_drive_file(TARTANAIR_COMPETITION_AIR_URL, air_archive)
        extract_tar_archive(
            air_archive,
            extract_root=competition_root,
            delete_archive=not args.keep_archives,
            member_filter=member_filter,
        )

    normalize_competition_layout(competition_root, aliases)
    print(f"\nDone. Competition data root: {competition_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
