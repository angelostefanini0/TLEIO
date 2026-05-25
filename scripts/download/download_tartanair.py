#!/usr/bin/env python3
"""
Example:
python scripts/download_tartanair.py \
  --root data/tartanair \
  --env office \
  --difficulty easy hard

Multiple environments can be passed in one list:
python scripts/download_tartanair.py \
  --root data/tartanair \
  --env office carwelding \
  --difficulty easy hard

Run this when these TartanEvent folders are already present under --root.
The office_events folder is the staging folder for office, so office is the
environment name to pass here.
python scripts/download/download_tartanair.py \
  --root data/tartanair \
  --env abandonedfactory abandonedfactory_night amusement carwelding endofworld gascola hospital japanesealley neighborhood ocean office oldtown seasidetown seasonsforest seasonsforest_winter soulcity westerndesert \
  --difficulty easy hard \
  --skip-event
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath
import urllib.request
import time
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.config import default_config_path, parse_args_with_config

TARTANEVENT_ROOT_URL = (
    "https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/TartanEvent"
)
TARTANAIR_FILE_LIST = REPO_ROOT / "download_training_zipfiles.txt"
TARTANAIR_FILE_LIST_URL = (
    "https://raw.githubusercontent.com/castacks/tartanair_tools/master/"
    "download_training_zipfiles.txt"
)
TARTANAIR_HF_REPO_ID = "theairlabcmu/tartanair"

def download_with_retry(url: str, target: str) -> None:
    import os
    # Ora usiamo SOLO final_path, niente più part_path
    final_path = Path(target)
    
    # Controllo integrità iniziale
    if final_path.exists():
        if zipfile.is_zipfile(final_path):
            print(f"\nFile {final_path.name} already fully downloaded and valid.")
            return
        else:
            print(f"\nFound incomplete {final_path.name}. Resuming directly on the file...")

    while True:
        try:
            req = urllib.request.Request(url)
            file_size = final_path.stat().st_size if final_path.exists() else 0
            
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

                with open(final_path, mode) as f:
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
            
            actual_size = final_path.stat().st_size
            if total_size > 0 and actual_size < total_size:
                raise Exception(f"Network error. Downloaded {actual_size} / {total_size} bytes.")

            print(f"\nDownload completed successfully: {final_path.name}")
            return

        except urllib.error.HTTPError as e:
            if e.code == 416:
                print("\nFile is already fully downloaded.")
                return
            print(f"\nAttempt failed (HTTP {e.code}). Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"\nAttempt failed: {e}. Retrying in 5s...")
            time.sleep(5)
            

def download_tartanevent(root: Path, env_zip: str, unzip: bool, delete_zip: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)

    env_folder = root / f"{env_zip}_events"
    env_folder.mkdir(parents=True, exist_ok=True)

    zip_name = env_zip if env_zip.endswith(".zip") else f"{env_zip}.zip"
    target_zip = root / zip_name
    url = f"{TARTANEVENT_ROOT_URL}/{zip_name}"

    print(f"Downloading TartanEvent: {url}")
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


class HuggingFaceTartanAirDownloader:
    def __init__(self, repo_id: str = TARTANAIR_HF_REPO_ID, chunk_size: int = 100) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Hugging Face TartanAir download requires huggingface-hub: "
                "pip install huggingface-hub"
            ) from exc

        self.repo_id = repo_id
        self.chunk_size = chunk_size
        self.snapshot_download = snapshot_download

    def download(self, filelist: list[str], root: Path) -> list[Path]:
        downloaded = []
        print(f"Downloading {len(filelist)} TartanAir file(s) from Hugging Face: {self.repo_id}")
        for start in range(0, len(filelist), self.chunk_size):
            chunk = filelist[start : start + self.chunk_size]
            self.snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=str(root),
                allow_patterns=chunk,
            )
            downloaded.extend(root / PurePosixPath(source_file) for source_file in chunk)

        return downloaded


def ensure_tartanair_file_list(file_list: Path) -> None:
    if file_list.exists():
        return
    if file_list.name != "download_training_zipfiles.txt":
        raise FileNotFoundError(f"TartanAir file list not found: {file_list}")

    print(f"Downloading TartanAir v1 file list -> {file_list}")
    file_list.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TARTANAIR_FILE_LIST_URL, file_list)


def load_tartanair_file_sizes(file_list: Path) -> dict[str, float]:
    ensure_tartanair_file_list(file_list)
    if not file_list.exists():
        raise FileNotFoundError(f"TartanAir file list not found: {file_list}")

    file_sizes = {}
    with open(file_list, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2 or not parts[0].endswith(".zip"):
                continue
            file_sizes[parts[0]] = float(parts[1])
    return file_sizes


def select_tartanair_archives(
    file_sizes: dict[str, float],
    env: str,
    difficulties: list[str],
    archive_name: str,
) -> list[str]:
    wanted_difficulties = {diff.lower() for diff in difficulties}
    selected = []

    for source_file in file_sizes:
        parts = PurePosixPath(source_file).parts
        if len(parts) < 3:
            continue
        if parts[0].lower() != env.lower():
            continue
        if parts[-1].lower() != archive_name.lower():
            continue

        diff_part = parts[1].lower()
        diff = diff_part.replace("data_", "")
        if diff in wanted_difficulties:
            selected.append(source_file)

    if not selected:
        expected = []
        for diff in difficulties:
            expected.append(f"{env}/{diff.capitalize()}/{archive_name}")
            expected.append(f"{env}/Data_{diff.lower()}/{archive_name}")
        raise ValueError(
            f"No TartanAir archive found for env={env}, difficulties={difficulties}, "
            f"archive={archive_name}. "
            f"Expected entries like: {expected}"
        )

    return sorted(selected)


def extract_zip_archives(zip_paths: list[Path], delete_zip: bool) -> None:
    for zip_path in zip_paths:
        print(f"Verifying archive: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupted zip archive {zip_path}, first bad file: {bad}")

            print(f"Extracting {zip_path} into {zip_path.parent}")
            zf.extractall(path=zip_path.parent)

        if delete_zip:
            print(f"Deleting {zip_path}")
            zip_path.unlink(missing_ok=True)



def download_tartanair_huggingface(
    root: Path,
    env: str,
    difficulties: list[str],
    file_list: Path,
    archive_name: str,
    delete_zip: bool,
) -> None:
    root.mkdir(parents=True, exist_ok=True)

    file_sizes = load_tartanair_file_sizes(file_list)
    archives = select_tartanair_archives(
        file_sizes,
        env,
        difficulties,
        archive_name,
    )
    total_size = sum(file_sizes[source_file] for source_file in archives)

    print("Hugging Face TartanAir download")
    print(f"  env:          {env}")
    print(f"  difficulties: {', '.join(difficulties)}")
    print(f"  archive:      {archive_name}")
    print(f"  file list:    {file_list}")
    print(f"  repo:         {TARTANAIR_HF_REPO_ID}")
    print(f"  total size:   {total_size:.3f} GB")

    downloader = HuggingFaceTartanAirDownloader()
    zip_paths = downloader.download(archives, root)
    extract_zip_archives(zip_paths, delete_zip=delete_zip)


def count_nonempty_lines(path: Path) -> int:
    with open(path, "r") as fh:
        return sum(1 for line in fh if line.strip())


def check_timestamps_pose_line_count(timestamps_file: Path, pose_file: Path) -> None:
    if not timestamps_file.exists() or not pose_file.exists():
        return

    timestamps_count = count_nonempty_lines(timestamps_file)
    pose_count = count_nonempty_lines(pose_file)
    if timestamps_count != pose_count:
        raise RuntimeError(
            f"TartanEvent/TartanAir frame count mismatch in {pose_file.parent}: "
            f"{timestamps_file.name} has {timestamps_count} lines, "
            f"{pose_file.name} has {pose_count} lines."
        )
    print(f"[OK] {pose_file.parent.name}: {timestamps_file.name} and {pose_file.name} have {timestamps_count} lines.")


def write_cam_time_from_event_timestamps(timestamps_file: Path, cam_time_file: Path) -> None:
    values = []
    with open(timestamps_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                values.append(float(line))

    if not values:
        return

    scale = 1e-9 if max(abs(v) for v in values) > 1e6 else 1.0
    cam_time_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cam_time_file, "w") as fh:
        for value in values:
            fh.write(f"{value * scale:.12f}\n")


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
            check_timestamps_pose_line_count(timestamps_file, pose_left)
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
    import os
    import shutil
    import re

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
                        elif item.name in {"pose_left.txt", "pose_lcam_front.txt"}:
                            target_item.unlink()
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
    parser.add_argument("--root", type=Path, default=None, help="Common root folder")
    parser.add_argument(
        "--env",
        type=str,
        nargs="+",
        default=None,
        help='Environment(s), e.g. "office" or "office carwelding"',
    )
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
        "--air-file-list",
        type=Path,
        default=TARTANAIR_FILE_LIST,
        help="TartanAir zip list used by the direct downloader.",
    )
    parser.add_argument(
        "--air-archive-name",
        type=str,
        default="flow_mask.zip",
        help="Archive name to download from the TartanAir file list. For v1 training, flow_mask.zip is the small archive used to get pose_left.txt.",
    )
    parser.add_argument(
        "--keep-air-images",
        action="store_true",
        help="Keep the extracted TartanAir payload folder after extracting pose_left.txt.",
    )
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
        print(f"Preparing Tartan environment: env={env}")
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
            download_tartanevent(
                root=root,
                env_zip=env,
                unzip=True,
                delete_zip=not args.keep_zip,
            )

        normalize_tartanair_layout(
            root=root,
            env=env,
            difficulties=args.difficulty,
            merge_roots=merge_roots,
        )
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
