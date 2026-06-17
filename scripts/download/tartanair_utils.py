from __future__ import annotations

import re
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]

TARTANEVENT_ROOT_URL = (
    "https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/TartanEvent"
)
TARTANEVENT_COMPETITION_URL = (
    "https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/"
    "TartanEvent_competition.zip"
)
TARTANAIR_FILE_LIST = REPO_ROOT / "download_training_zipfiles.txt"
TARTANAIR_FILE_LIST_URL = (
    "https://raw.githubusercontent.com/castacks/tartanair_tools/master/"
    "download_training_zipfiles.txt"
)
TARTANAIR_HF_REPO_ID = "theairlabcmu/tartanair"
TARTANAIR_COMPETITION_AIR_URL = (
    "https://drive.google.com/file/d/1N9BkpQuibIyIBkLxVPUuoB-eDOMFqY8D/view?usp=sharing"
)
TARTANAIR_COMPETITION_ARCHIVE = "tartanair-test-mono-release.tar.gz"


def download_with_retry(url: str, target: Path) -> None:
    import os

    target.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            req = urllib.request.Request(url)
            file_size = target.stat().st_size if target.exists() else 0

            if file_size > 0:
                req.add_header("Range", f"bytes={file_size}-")
                print(f"Attempt: resuming {target.name} from {file_size / (1024**3):.2f} GB")
            else:
                print(f"Attempt: starting {target.name}")

            with urllib.request.urlopen(req, timeout=30) as response:
                is_partial = response.status == 206
                mode = "ab" if is_partial else "wb"

                if file_size > 0 and not is_partial:
                    file_size = 0
                    mode = "wb"

                total_size = int(response.headers.get("content-length", 0))
                if is_partial:
                    total_size += file_size

                with target.open(mode) as fh:
                    with tqdm(
                        total=total_size,
                        initial=file_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=target.name,
                    ) as pbar:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            fh.write(chunk)
                            fh.flush()
                            os.fsync(fh.fileno())
                            pbar.update(len(chunk))

            if total_size > 0 and target.stat().st_size < total_size:
                raise RuntimeError(
                    f"Network error. Downloaded {target.stat().st_size} / {total_size} bytes."
                )

            print(f"Download completed: {target}")
            return

        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                print(f"Download already complete: {target}")
                return
            print(f"Attempt failed (HTTP {exc.code}). Retrying in 5s...")
            time.sleep(5)
        except Exception as exc:
            print(f"Attempt failed: {exc}. Retrying in 5s...")
            time.sleep(5)


def download_google_drive_file(url: str, target: Path) -> None:
    file_id = _google_drive_file_id(url)
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    response = opener.open(direct_url)
    token = _google_drive_confirm_token(response)
    if token:
        response = opener.open(f"{direct_url}&confirm={token}")

    target.parent.mkdir(parents=True, exist_ok=True)
    total_size = int(response.headers.get("content-length", 0))
    with target.open("wb") as fh:
        with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=target.name) as pbar:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                pbar.update(len(chunk))


def _google_drive_file_id(url: str) -> str:
    match = re.search(r"/d/([^/]+)", url)
    if match:
        return match.group(1)
    query_id = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("id")
    if query_id:
        return query_id[0]
    raise ValueError(f"Could not parse Google Drive file id from {url}")


def _google_drive_confirm_token(response) -> str | None:
    for key, value in response.headers.items():
        if key.lower() != "set-cookie":
            continue
        match = re.search(r"download_warning[^=]*=([^;]+)", value)
        if match:
            return match.group(1)
    return None


class HuggingFaceTartanAirDownloader:
    def __init__(self, repo_id: str = TARTANAIR_HF_REPO_ID, chunk_size: int = 100) -> None:
        from huggingface_hub import snapshot_download

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


def download_tartanevent(root: Path, env_zip: str, delete_zip: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    env_folder = root / f"{env_zip}_events"
    env_folder.mkdir(parents=True, exist_ok=True)

    zip_name = env_zip if env_zip.endswith(".zip") else f"{env_zip}.zip"
    target_zip = root / zip_name
    download_with_retry(f"{TARTANEVENT_ROOT_URL}/{zip_name}", target_zip)
    extract_zip_archives([target_zip], delete_zip=delete_zip, extract_root=env_folder)


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
    file_sizes = {}
    with file_list.open("r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].endswith(".zip"):
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
        diff = parts[1].lower().replace("data_", "")
        if parts[0].lower() == env.lower() and parts[-1].lower() == archive_name.lower():
            if diff in wanted_difficulties:
                selected.append(source_file)

    if not selected:
        raise ValueError(
            f"No TartanAir archive found for env={env}, difficulties={difficulties}, "
            f"archive={archive_name}."
        )
    return sorted(selected)


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
    archives = select_tartanair_archives(file_sizes, env, difficulties, archive_name)
    total_size = sum(file_sizes[source_file] for source_file in archives)

    print("Hugging Face TartanAir download")
    print(f"  env:          {env}")
    print(f"  difficulties: {', '.join(difficulties)}")
    print(f"  archive:      {archive_name}")
    print(f"  repo:         {TARTANAIR_HF_REPO_ID}")
    print(f"  total size:   {total_size:.3f} GB")

    zip_paths = HuggingFaceTartanAirDownloader().download(archives, root)
    extract_zip_archives(zip_paths, delete_zip=delete_zip)


def extract_zip_archives(zip_paths: list[Path], delete_zip: bool, extract_root: Path | None = None) -> None:
    for zip_path in zip_paths:
        root = extract_root or zip_path.parent
        print(f"Verifying archive: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupted zip archive {zip_path}, first bad file: {bad}")
            print(f"Extracting {zip_path} into {root}")
            zf.extractall(path=root)
        if delete_zip:
            zip_path.unlink(missing_ok=True)


def extract_tar_archive(archive_path: Path, extract_root: Path, delete_archive: bool) -> None:
    print(f"Extracting {archive_path} into {extract_root}")
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=extract_root)
    if delete_archive:
        archive_path.unlink(missing_ok=True)


def count_nonempty_lines(path: Path) -> int:
    with path.open("r") as fh:
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
    values = [float(line.strip()) for line in timestamps_file.read_text().splitlines() if line.strip()]
    if not values:
        return

    scale = 1e-9 if max(abs(v) for v in values) > 1e6 else 1.0
    cam_time_file.parent.mkdir(parents=True, exist_ok=True)
    with cam_time_file.open("w") as fh:
        for value in values:
            fh.write(f"{value * scale:.12f}\n")
